//! The runtime: [`run_chain`] executes a step list over a buffer pair;
//! [`fill_tiled`] drives the tiled, haloed, optionally-parallel sweep.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{DynamicImageRef, ImageRef, PixelType};

use super::geom::{geo_crop, geo_flip, geo_roi, geo_rot90};
use super::kernels::{
    Demosaic, convert_inplace, debayer_into, luma_inplace, scale_inplace, scale_pixels_inplace,
};
use super::plan::{Step, StepKind};
use super::resample::geo_resize;
use super::spec::pixel_size;
use super::{ImageSpec, PipelineError};

/// Execute every step over an image whose input (shape `w * h`) already sits in
/// `buf_a`. Geometric steps update the running dimensions. Returns `true` if the
/// result ended up in `buf_a`, `false` if in `buf_b`.
pub(super) fn run_chain(
    steps: &[Step],
    coeffs: &[Box<[f64]>],
    buf_a: &mut [f32],
    buf_b: &mut [f32],
    w: usize,
    h: usize,
    mut demosaic: Demosaic<'_>,
) -> Result<bool, PipelineError> {
    let mut cur_a = true;
    let (mut cw, mut ch) = (w, h);
    for step in steps {
        let n = cw * ch * step.in_channels as usize;
        let bpp = step.in_channels as usize * pixel_size(step.in_pt)?;
        match step.kind {
            StepKind::Debayer(method) => {
                let pat = step.bayer.expect("debayer step carries a pattern");
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                debayer_into(src, dst, step.in_pt, cw, ch, pat, method, &mut demosaic)?;
                cur_a = !cur_a;
            }
            StepKind::Convert => {
                let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                convert_inplace(buf, step.in_pt, step.out_pt, n)?;
            }
            StepKind::Scale { gain, offset } => {
                let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                scale_inplace(buf, step.in_pt, n, gain, offset);
            }
            StepKind::ScalePixels(factor) => {
                let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                scale_pixels_inplace(buf, step.in_pt, n, factor)?;
            }
            StepKind::Luma => {
                if !step.luma_identity {
                    let cf = &coeffs[step.coeff_idx];
                    let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                    luma_inplace(buf, step.in_pt, step.in_channels as usize, n, cf);
                }
            }
            StepKind::Crop { x, y, w: ow, h: oh } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_crop(src, dst, cw, bpp, x, y, ow, oh);
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
            StepKind::Roi { x, y, w: ow, h: oh } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_roi(src, dst, cw, ch, bpp, x, y, ow, oh);
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
            StepKind::Flip {
                horizontal,
                vertical,
            } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_flip(src, dst, cw, ch, bpp, horizontal, vertical);
                cur_a = !cur_a;
            }
            StepKind::Rot90 { ccw } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_rot90(src, dst, cw, ch, bpp, ccw);
                std::mem::swap(&mut cw, &mut ch);
                cur_a = !cur_a;
            }
            StepKind::Resize {
                w: ow,
                h: oh,
                filter,
            } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_resize(
                    src,
                    dst,
                    cw,
                    ch,
                    step.in_channels as usize,
                    step.in_pt,
                    ow,
                    oh,
                    filter,
                )?;
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
            StepKind::Nop => {
                // Do nothing.
            }
        }
    }
    Ok(cur_a)
}

/// Expand an output range `[lo, hi)` to a source range that (a) starts on an even
/// boundary when `even` (keeps the Bayer phase), (b) carries `halo` real
/// rows/cols of context on each side of the kept region, and (c) is at least
/// `2*halo + 3` wide so the serial kernel has room. Never shrinks past
/// `[lo, hi)`; clamps to `[0, limit)`.
fn halo_range(lo: usize, hi: usize, limit: usize, halo: usize, even: bool) -> (usize, usize) {
    if halo == 0 && !even {
        return (lo, hi);
    }
    let (mut a, mut b) = if even {
        (
            lo.saturating_sub(halo) & !1,
            ((hi + halo + 1) & !1).min(limit),
        )
    } else {
        (lo.saturating_sub(halo), (hi + halo).min(limit))
    };
    let step = if even { 2 } else { 1 };
    let min_span = 2 * halo + 3;
    while b - a < min_span {
        if b < limit {
            b = (b + step).min(limit);
        } else if a >= step {
            a -= step;
        } else {
            break;
        }
    }
    (a, b)
}

/// Tile geometry shared by every tile in one `fill_tiled` call.
#[derive(Clone, Copy)]
pub(super) struct TileGeom {
    pub(super) tile_rows: usize,
    pub(super) tile_cols: usize, // 0 == full width
    pub(super) halo: usize,
    pub(super) even: bool,
}

/// Fill `out_buf` by running the chain independently on each tile. Bands (row
/// strips) are the unit of parallelism; each band is swept left-to-right in
/// `tile_cols`-wide column tiles.
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_tiled(
    steps: &[Step],
    coeffs: &[Box<[f64]>],
    in_spec: &ImageSpec,
    out_spec: &ImageSpec,
    frame: &[u8],
    // Byte stride of one frame row, and the byte offset of the tiled body's
    // origin into `frame` (folded leading crops). For an un-cropped input these
    // are `in_spec.width * in_bpp` and `0`.
    frame_stride: usize,
    frame_off: usize,
    out_buf: &mut [f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    demosaic_scratch: &mut [f32],
    geom: TileGeom,
    parallel: bool,
) -> Result<(), PipelineError> {
    let TileGeom {
        tile_rows,
        tile_cols,
        halo,
        even,
    } = geom;
    let w = in_spec.width;
    let h = in_spec.height;
    let cols = if tile_cols == 0 || tile_cols >= w {
        w
    } else {
        tile_cols
    };
    let in_bpp = in_spec.bpp()?;
    let out_bpp = out_spec.bpp()?;
    let out_row = w * out_bpp;
    let out_bytes = h * out_row;
    let n_cols = w.div_ceil(cols);
    let cap_a = scratch_a.len();
    let cap_b = scratch_b.len();
    let demo_len = demosaic_scratch.len();

    // One band = one contiguous `[y0, y1)` slab of the output.
    let do_band = |band_idx: usize,
                   band: &mut [u8],
                   sa: &mut [f32],
                   sb: &mut [f32],
                   demo: &mut [f32]|
     -> Result<(), PipelineError> {
        let y0 = band_idx * tile_rows;
        if y0 >= h {
            return Ok(());
        }
        let y1 = (y0 + tile_rows).min(h);
        let (ry0, ry1) = halo_range(y0, y1, h, halo, even);
        let sh = ry1 - ry0;

        for col in 0..n_cols {
            let x0 = col * cols;
            let x1 = (x0 + cols).min(w);
            let (rx0, rx1) = halo_range(x0, x1, w, halo, even);
            let sw = rx1 - rx0;

            // Assemble the padded input sub-rect into `sa`, packed at width `sw`.
            let sa_u8 = cast_slice_mut::<f32, u8>(&mut *sa);
            let s_in_row = sw * in_bpp;
            for r in 0..sh {
                let src = frame_off + (ry0 + r) * frame_stride + rx0 * in_bpp;
                sa_u8[r * s_in_row..r * s_in_row + s_in_row]
                    .copy_from_slice(&frame[src..src + s_in_row]);
            }

            // Run the chain on the tile with a serial demosaic kernel.
            let dm = if demo.is_empty() {
                Demosaic::Alloc
            } else {
                Demosaic::Pooled(&mut *demo)
            };
            let cur_a = run_chain(steps, coeffs, &mut *sa, &mut *sb, sw, sh, dm)?;
            let res: &[f32] = if cur_a { &*sa } else { &*sb };
            let res_u8 = cast_slice::<f32, u8>(res);
            let s_out_row = sw * out_bpp;

            // Scatter the valid center rect into the band.
            let y_skip = y0 - ry0;
            if x0 == 0 && x1 == w && sw == w {
                let take = (y1 - y0) * out_row;
                let s0 = y_skip * s_out_row;
                band[..take].copy_from_slice(&res_u8[s0..s0 + take]);
            } else {
                let x_skip_b = (x0 - rx0) * out_bpp;
                let span_b = (x1 - x0) * out_bpp;
                for r in 0..(y1 - y0) {
                    let d = r * out_row + x0 * out_bpp;
                    let s = (y_skip + r) * s_out_row + x_skip_b;
                    band[d..d + span_b].copy_from_slice(&res_u8[s..s + span_b]);
                }
            }
        }
        Ok(())
    };

    #[cfg(feature = "rayon")]
    if parallel {
        use rayon::prelude::*;
        let out_u8 = &mut cast_slice_mut::<f32, u8>(out_buf)[..out_bytes];
        return out_u8
            .par_chunks_mut(tile_rows * out_row)
            .enumerate()
            .try_for_each_init(
                || {
                    (
                        vec![0.0f32; cap_a],
                        vec![0.0f32; cap_b],
                        vec![0.0f32; demo_len],
                    )
                },
                |(sa, sb, demo), (idx, band)| do_band(idx, band, sa, sb, demo),
            );
    }
    #[cfg(not(feature = "rayon"))]
    let _ = (parallel, cap_a, cap_b, demo_len);

    let out_u8 = &mut cast_slice_mut::<f32, u8>(out_buf)[..out_bytes];
    for (idx, band) in out_u8.chunks_mut(tile_rows * out_row).enumerate() {
        do_band(
            idx,
            band,
            &mut *scratch_a,
            &mut *scratch_b,
            &mut *demosaic_scratch,
        )?;
    }
    Ok(())
}

fn pick<'x>(a: &'x mut [f32], b: &'x mut [f32], cur_a: bool) -> (&'x mut [f32], &'x mut [f32]) {
    if cur_a { (a, b) } else { (b, a) }
}

pub(super) fn current<'x>(a: &'x mut [f32], b: &'x mut [f32], cur_a: bool) -> &'x mut [f32] {
    if cur_a { a } else { b }
}

pub(super) fn view<'a>(
    buf: &'a mut [f32],
    spec: &ImageSpec,
) -> Result<DynamicImageRef<'a>, PipelineError> {
    let n = spec.elems();
    let (w, h) = (spec.width, spec.height);
    let cs = spec.cspace.clone();
    Ok(match spec.pixel_type.storage() {
        PixelType::U8 => {
            let d = &mut cast_slice_mut::<f32, u8>(buf)[..n];
            DynamicImageRef::from(ImageRef::<u8>::create(d, w, h, cs)?)
        }
        PixelType::U16 => {
            let d = &mut cast_slice_mut::<f32, u16>(buf)[..n];
            // Re-apply a `U10` / `U12` / `U14` tag; a plain `U16` clears it.
            let img =
                ImageRef::<u16>::create(d, w, h, cs)?.with_bit_depth(spec.pixel_type.bit_depth());
            DynamicImageRef::from(img)
        }
        PixelType::F32 => {
            let d = &mut buf[..n];
            DynamicImageRef::from(ImageRef::<f32>::create(d, w, h, cs)?)
        }
        other => return Err(PipelineError::UnsupportedPixelType(other)),
    })
}
