//! Proves a compiled serial-tiled [`Runner`] touches the heap zero times per
//! frame. Lives in its own integration binary so the counting global allocator
//! sees only this test's work (one test → no concurrent test threads).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use refimage::pipeline::{ImageSpec, Pipeline, Strategy};
use refimage::{BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, ImageRef, PixelType};

// Second test would race the global counter; keep this binary single-test.

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
fn serial_tiled_run_is_allocation_free() {
    let (w, h) = (96usize, 160usize);
    let pat = BayerPattern::Rggb;

    // Cubic is the hungriest kernel; luma + convert exercise the rest of a
    // chain; the trailing flip runs as a sequential geo tail over `out_buf`.
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Cubic)
        .to_luma()
        .convert(PixelType::U8)
        .flip_vertical()
        .compile(
            ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16),
            Strategy::tiled(12),
        )
        .expect("compile");
    assert!(runner.is_tiled(), "expected a tiled runner");

    let mut frame: Vec<u16> = (0..w * h).map(|i| ((i * 7 + 3) % 251) as u16).collect();

    // Warm up: first runs may touch lazy statics, etc.
    for _ in 0..3 {
        let img =
            DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pat)).unwrap());
        runner.run(&img).unwrap();
    }

    COUNT.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    for _ in 0..32 {
        let img =
            DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pat)).unwrap());
        let out = runner.run(&img).unwrap();
        std::hint::black_box(out.as_raw_u8()[0]);
    }
    ARMED.store(false, Ordering::Relaxed);

    let allocs = COUNT.load(Ordering::Relaxed);
    assert_eq!(
        allocs, 0,
        "compiled serial-tiled run allocated {allocs} times over 32 frames"
    );
}
