//! Geometric relocations, executed on `f32`-backed byte buffers.
//!
//! Each function reinterprets its buffers as `bpp`-byte pixels and moves them;
//! none allocate.

use bytemuck::{cast_slice, cast_slice_mut};

/// Copy the `(ow, oh)` sub-rectangle at `(x, y)` out of a `sw`-wide `bpp`-B/px
/// image (`src`) into `dst`, packed at width `ow`.
#[allow(clippy::too_many_arguments)]
pub(super) fn geo_crop(
    src: &[f32],
    dst: &mut [f32],
    sw: usize,
    bpp: usize,
    x: usize,
    y: usize,
    ow: usize,
    oh: usize,
) {
    let sb = cast_slice::<f32, u8>(src);
    let db = cast_slice_mut::<f32, u8>(dst);
    let s_row = sw * bpp;
    let d_row = ow * bpp;
    for r in 0..oh {
        let s = (y + r) * s_row + x * bpp;
        db[r * d_row..r * d_row + d_row].copy_from_slice(&sb[s..s + d_row]);
    }
}

/// Copy the `(ow, oh)` window at `(x, y)` out of an `sw * sh` `bpp`-B/px image
/// (`src`) into `dst` (packed at width `ow`). Any part of the window past the
/// source edge is left as-is in `dst` — the caller zeroes `dst` first, so it
/// reads back as zero padding.
#[allow(clippy::too_many_arguments)]
pub(super) fn geo_roi(
    src: &[f32],
    dst: &mut [f32],
    sw: usize,
    sh: usize,
    bpp: usize,
    x: usize,
    y: usize,
    ow: usize,
    oh: usize,
) {
    let sb = cast_slice::<f32, u8>(src);
    let db = cast_slice_mut::<f32, u8>(dst);
    db[..ow * oh * bpp].fill(0);
    if x >= sw || y >= sh {
        return;
    }
    let s_row = sw * bpp;
    let d_row = ow * bpp;
    let copy_w = ow.min(sw - x) * bpp;
    let copy_h = oh.min(sh - y);
    for r in 0..copy_h {
        let s = (y + r) * s_row + x * bpp;
        let d = r * d_row;
        db[d..d + copy_w].copy_from_slice(&sb[s..s + copy_w]);
    }
}

/// Mirror a `w * h` `bpp`-B/px image from `src` into `dst`.
pub(super) fn geo_flip(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    bpp: usize,
    fh: bool,
    fv: bool,
) {
    let sb = cast_slice::<f32, u8>(src);
    let db = cast_slice_mut::<f32, u8>(dst);
    let row = w * bpp;
    for r in 0..h {
        let sr = if fv { h - 1 - r } else { r };
        if fh {
            for c in 0..w {
                let sc = w - 1 - c;
                let s = sr * row + sc * bpp;
                let d = r * row + c * bpp;
                db[d..d + bpp].copy_from_slice(&sb[s..s + bpp]);
            }
        } else {
            let s = sr * row;
            db[r * row..r * row + row].copy_from_slice(&sb[s..s + row]);
        }
    }
}

/// Rotate a `w * h` `bpp`-B/px image a quarter turn (clockwise, or `ccw`) from
/// `src` into `dst` (whose logical shape is `h * w`). Blocked so neither side
/// strides the whole image per pixel.
pub(super) fn geo_rot90(src: &[f32], dst: &mut [f32], w: usize, h: usize, bpp: usize, ccw: bool) {
    const B: usize = 32;
    let sb = cast_slice::<f32, u8>(src);
    let db = cast_slice_mut::<f32, u8>(dst);
    let (ow, oh) = (h, w); // output shape
    let mut by = 0;
    while by < oh {
        let by1 = (by + B).min(oh);
        let mut bx = 0;
        while bx < ow {
            let bx1 = (bx + B).min(ow);
            for oy in by..by1 {
                for ox in bx..bx1 {
                    let (ix, iy) = if ccw {
                        (w - 1 - oy, ox)
                    } else {
                        (oy, h - 1 - ox)
                    };
                    let s = (iy * w + ix) * bpp;
                    let d = (oy * ow + ox) * bpp;
                    db[d..d + bpp].copy_from_slice(&sb[s..s + bpp]);
                }
            }
            bx += B;
        }
        by += B;
    }
}
