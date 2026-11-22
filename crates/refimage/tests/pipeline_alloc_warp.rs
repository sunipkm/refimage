//! A serial-tiled [`Runner`] with a cubic, prefiltered warp in the chain still does
//! zero heap allocation per frame: the coefficient plane is compile-time scratch and
//! a serial strategy keeps the warp's rows off the rayon pool. Own binary, one test,
//! so the counting allocator sees only this work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use refimage::pipeline::{BoundaryMode, ImageSpec, InterpOrder, Interpolation, Pipeline, Strategy};
use refimage::{
    BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, ImageRef, PixelData, PixelType,
};

static ARMED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[test]
fn serial_tiled_warp_is_allocation_free() {
    let (w, h) = (96usize, 160usize);
    let pat = BayerPattern::Rggb;
    let (s, c) = 25f64.to_radians().sin_cos();
    let inverse = [[c, -s, 40.0], [s, c, -10.0], [0.0, 0.0, 1.0]];

    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .warp(
            inverse,
            120,
            100,
            Interpolation::new(InterpOrder::Cubic, BoundaryMode::Nearest, 0.0),
        )
        .map_coordinates(
            (0..60 * 50)
                .map(|i| (i / 60) as f64 * 1.7)
                .collect::<Vec<_>>(),
            (0..60 * 50)
                .map(|i| (i % 60) as f64 * 1.9)
                .collect::<Vec<_>>(),
            60,
            50,
            Interpolation::new(InterpOrder::Cubic, BoundaryMode::Reflect, 0.0),
        )
        .to_luma()
        .convert(PixelType::U8)
        .compile(
            ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16),
            Strategy::tiled(12),
        )
        .expect("compile");
    assert!(runner.is_tiled(), "expected a tiled runner");

    let mut frame: Vec<u16> = (0..w * h).map(|i| ((i * 7 + 3) % 251) as u16).collect();
    for _ in 0..3 {
        let img =
            DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pat)).unwrap());
        runner.run(&img).unwrap();
    }

    COUNT.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    for _ in 0..16 {
        let img =
            DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pat)).unwrap());
        let out = runner.run(&img).unwrap();
        std::hint::black_box(out.as_raw_u8()[0]);
    }
    ARMED.store(false, Ordering::Relaxed);

    let allocs = COUNT.load(Ordering::Relaxed);
    assert_eq!(
        allocs, 0,
        "serial-tiled warp run allocated {allocs} times over 16 frames"
    );
}
