//! Once compiled with a serial strategy, the C entry points that run frames allocate
//! nothing per frame: `ref_runner_run` into an existing image, `ref_runner_run_into`
//! a caller buffer, `ref_runner_run_image` (metadata without cards), and padded
//! (`row_stride`) input after the first frame. Own binary, one test, so the counting
//! allocator sees only this work.

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use refimage_c::*;

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

fn ok(s: RefStatus) {
    assert_eq!(s, RefStatus::REF_OK, "{:?}", unsafe {
        std::ffi::CStr::from_ptr(ref_last_error())
    });
}

/// Allocations made by `f`.
fn count(f: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    COUNT.load(Ordering::SeqCst)
}

#[test]
fn running_frames_does_not_allocate() {
    const W: usize = 96;
    const H: usize = 64;
    const STRIDE: usize = W * 2 + 32;
    let raw: Vec<u16> = (0..W * H).map(|i| (i * 37 % 4096) as u16).collect();
    let mut padded = vec![0u8; STRIDE * H];
    for y in 0..H {
        let row: &[u8] = bytemuck::cast_slice(&raw[y * W..(y + 1) * W]);
        padded[y * STRIDE..y * STRIDE + W * 2].copy_from_slice(row);
    }
    let frame = RefFrame {
        data: raw.as_ptr().cast(),
        width: W,
        height: H,
        row_stride: 0,
        pixel_type: REF_PIXEL_U12,
        color: REF_COLOR_BAYER_GRBG,
        channels: 0,
        color_name: ptr::null(),
    };
    let padded_frame = RefFrame {
        data: padded.as_ptr().cast(),
        row_stride: STRIDE,
        ..frame
    };
    let shift = [1.0, 0.0, 0.5, 0.0, 1.0, -0.25, 0.0, 0.0, 1.0];
    let cubic = RefInterpolation {
        order: 3,
        ..ref_interpolation_default()
    };

    unsafe {
        let mut p = ptr::null_mut();
        ok(ref_pipeline_new(&mut p));
        ok(ref_pipeline_debayer(p, REF_DEMOSAIC_CUBIC));
        ok(ref_pipeline_warp(p, shift.as_ptr(), W - 8, H - 8, &cubic));
        ok(ref_pipeline_to_luma(p));
        ok(ref_pipeline_convert(p, REF_PIXEL_U8));
        let strategy = RefStrategy {
            parallel: false,
            tile_rows: 16,
            ..ref_strategy_default()
        };
        let mut r = ptr::null_mut();
        ok(ref_pipeline_compile(p, &frame, &strategy, &mut r));

        let mut out = ptr::null_mut();
        ok(ref_runner_run(r, &frame, &mut out)); // creates `out`
        ok(ref_runner_run(r, &padded_frame, &mut out)); // sizes the staging buffer
        let mut src = ptr::null_mut();
        ok(ref_image_new(&frame, &mut src));
        ok(ref_image_set_exposure(src, 0.5));
        ok(ref_image_set_frame_id(src, 3));
        let mut info = RefImageInfo::default();
        ok(ref_runner_output_info(r, &mut info));
        let mut user = vec![0u8; info.nbytes];

        let n = count(|| {
            for _ in 0..8 {
                ok(ref_runner_run(r, &frame, &mut out));
                ok(ref_runner_run(r, &padded_frame, &mut out));
                ok(ref_runner_run_into(
                    r,
                    &frame,
                    user.as_mut_ptr().cast(),
                    user.len(),
                    0,
                ));
                ok(ref_runner_run_image(r, src, &mut out));
            }
        });
        assert_eq!(n, 0, "{n} allocations while running frames");

        // The counter does see the library's allocations: a fresh output allocates.
        let mut fresh = ptr::null_mut();
        assert!(count(|| ok(ref_runner_run(r, &frame, &mut fresh))) > 0);
        ref_image_free(fresh);

        let mut same = false;
        let mut expect = ptr::null_mut();
        ok(ref_pipeline_apply(p, src, &mut expect));
        ok(ref_image_equal(out, expect, &mut same));
        assert!(same, "the runner disagrees with apply");

        ref_image_free(expect);
        ref_image_free(src);
        ref_image_free(out);
        ref_runner_free(r);
        ref_pipeline_free(p);
    }
}
