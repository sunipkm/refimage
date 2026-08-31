//! Separable resampling for [`Op::ResizeToFit`](super::Op::ResizeToFit).
//!
//! A two-pass filter — horizontal, then vertical — with edge samples taken by
//! clamp extension. On downscale the kernel support is widened by the downscale
//! ratio, so the output is an area-weighted average of the source rather than a
//! set of point samples; this is what suppresses aliasing. Weights are computed
//! once per output row/column and applied to every channel. Accumulation is in
//! `f64`; the final write saturates into the stored type's `[min, max]` range.
//!
//! Work is split into **bands** of output rows. Each band horizontal-resamples
//! only the source rows it needs into a small local strip, then vertical-
//! resamples that strip into its slice of the output — one fused pass, cache-
//! friendly, with no full-size intermediate plane. Bands are independent and
//! fan out over a rayon pool when the feature is on. Every output element is
//! still produced by one task summing its taps in a fixed order, so the result
//! does not depend on the band size or thread count.

use bytemuck::{cast_slice, cast_slice_mut};
use serde::{Deserialize, Serialize};

use crate::{PixelStor, PixelType};

use super::{pixel_size, PipelineError};

/// Resampling filter for [`Op::ResizeToFit`](super::Op::ResizeToFit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResizeFilter {
    /// Triangle (linear) filter, 2-tap at unit scale. Fast; mild softening.
    Bilinear,
    /// Catmull-Rom cubic, 4-tap at unit scale. Sharper than [`Bilinear`](Self::Bilinear),
    /// with slight overshoot at high-contrast edges.
    Bicubic,
    /// Lanczos windowed sinc with `a = 3`, 6-tap at unit scale. Highest
    /// fidelity; the most ringing.
    Lanczos3,
}

impl ResizeFilter {
    /// Kernel radius in source pixels, at unit (1:1) scale.
    fn support(self) -> f64 {
        match self {
            ResizeFilter::Bilinear => 1.0,
            ResizeFilter::Bicubic => 2.0,
            ResizeFilter::Lanczos3 => 3.0,
        }
    }

    /// Kernel weight at `x` (in source pixels, unit scale).
    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            ResizeFilter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            ResizeFilter::Bicubic => {
                // Catmull-Rom (a = -0.5).
                if x < 1.0 {
                    ((1.5 * x - 2.5) * x) * x + 1.0
                } else if x < 2.0 {
                    (((-0.5 * x + 2.5) * x) - 4.0) * x + 2.0
                } else {
                    0.0
                }
            }
            ResizeFilter::Lanczos3 => {
                if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Normalized sinc, `sin(pi x) / (pi x)`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Largest `(width, height)` that fits inside `max_w` x `max_h` at the aspect
/// ratio of `w` x `h`. Enlarges when the source is smaller than the box; each
/// side is at least 1 and never exceeds its bound.
pub(super) fn resize_dims(w: usize, h: usize, max_w: usize, max_h: usize) -> (usize, usize) {
    let scale = f64::min(max_w as f64 / w as f64, max_h as f64 / h as f64);
    let nw = ((w as f64 * scale).round() as usize).clamp(1, max_w);
    let nh = ((h as f64 * scale).round() as usize).clamp(1, max_h);
    (nw, nh)
}

/// One source sample and its weight for a given output coordinate.
struct Tap {
    idx: usize,
    weight: f64,
}

/// Resampling taps for every output coordinate along one axis. Source indices
/// are clamped into `[0, src_len)` (edge extension) and each list sums to 1.
fn axis_taps(src_len: usize, dst_len: usize, filter: ResizeFilter) -> Vec<Vec<Tap>> {
    let ratio = src_len as f64 / dst_len as f64;
    // Widen the kernel when downscaling so it averages rather than point-samples.
    let scale = ratio.max(1.0);
    let support = filter.support() * scale;
    let last = src_len as isize - 1;

    (0..dst_len)
        .map(|x| {
            let center = (x as f64 + 0.5) * ratio;
            let left = (center - support).floor() as isize;
            let right = (center + support).ceil() as isize;

            let mut taps: Vec<Tap> = Vec::new();
            let mut sum = 0.0;
            for s in left..=right {
                let w = filter.eval((s as f64 + 0.5 - center) / scale);
                if w == 0.0 {
                    continue;
                }
                sum += w;
                let idx = s.clamp(0, last) as usize;
                // `s` ascends and the clamp is monotonic, so equal indices are
                // always adjacent — fold them into the previous tap.
                match taps.last_mut() {
                    Some(t) if t.idx == idx => t.weight += w,
                    _ => taps.push(Tap { idx, weight: w }),
                }
            }

            if sum.abs() < 1e-12 {
                let idx = (center.floor() as isize).clamp(0, last) as usize;
                vec![Tap { idx, weight: 1.0 }]
            } else {
                for t in &mut taps {
                    t.weight /= sum;
                }
                taps
            }
        })
        .collect()
}

/// Resample the `sw` x `sh`, `channels`-interleaved image of element type `pt`
/// in `src` into the `ow` x `oh` image in `dst`. Both buffers are `f32`-backed
/// byte storage; `src`/`dst` are only read/written within their logical extent.
#[allow(clippy::too_many_arguments)]
pub(super) fn geo_resize(
    src: &[f32],
    dst: &mut [f32],
    sw: usize,
    sh: usize,
    channels: usize,
    pt: PixelType,
    ow: usize,
    oh: usize,
    filter: ResizeFilter,
) -> Result<(), PipelineError> {
    pixel_size(pt)?;
    match pt {
        PixelType::U8 => resize_typed::<u8>(
            cast_slice(src),
            cast_slice_mut(dst),
            sw,
            sh,
            channels,
            ow,
            oh,
            filter,
        ),
        PixelType::U16 => resize_typed::<u16>(
            cast_slice(src),
            cast_slice_mut(dst),
            sw,
            sh,
            channels,
            ow,
            oh,
            filter,
        ),
        PixelType::F32 => resize_typed::<f32>(src, dst, sw, sh, channels, ow, oh, filter),
        other => return Err(PipelineError::UnsupportedPixelType(other)),
    }
    Ok(())
}

/// Target working-set size for one band's horizontal-pass strip.
const BAND_TARGET_BYTES: usize = 256 * 1024;

#[allow(clippy::too_many_arguments)]
fn resize_typed<T: PixelStor>(
    src: &[T],
    dst: &mut [T],
    sw: usize,
    sh: usize,
    ch: usize,
    ow: usize,
    oh: usize,
    filter: ResizeFilter,
) {
    let xt = axis_taps(sw, ow, filter);
    let yt = axis_taps(sh, oh, filter);
    let round = T::PIXEL_TYPE != PixelType::F32;
    let out_row = ow * ch;
    let band_rows = (BAND_TARGET_BYTES / (out_row * 4).max(1)).clamp(1, oh);

    // One band of output rows: horizontal-resample just the source rows those
    // outputs read into a local strip, then vertical-resample into `out`.
    let do_band = |b: usize, out: &mut [T]| {
        let oy0 = b * band_rows;
        let oy1 = (oy0 + band_rows).min(oh);
        if oy0 >= oy1 {
            return;
        }

        // Source rows this band touches (taps are clamped and ascending).
        let mut sy_lo = sh;
        let mut sy_hi = 0usize;
        for taps in &yt[oy0..oy1] {
            sy_lo = sy_lo.min(taps[0].idx);
            sy_hi = sy_hi.max(taps[taps.len() - 1].idx + 1);
        }

        // Horizontal pass -> strip; strip row `r` is source row `sy_lo + r`.
        let mut strip = vec![0f32; (sy_hi - sy_lo) * out_row];
        for (r, sr) in (sy_lo..sy_hi).enumerate() {
            let s_base = sr * sw * ch;
            let m_base = r * out_row;
            for (ox, taps) in xt.iter().enumerate() {
                for c in 0..ch {
                    let mut acc = 0f64;
                    for tap in taps {
                        acc += tap.weight * src[s_base + tap.idx * ch + c].to_f64();
                    }
                    strip[m_base + ox * ch + c] = acc as f32;
                }
            }
        }

        // Vertical pass. Integer outputs round to nearest; `f32` keeps the raw
        // weighted sum.
        for (i, taps) in yt[oy0..oy1].iter().enumerate() {
            let d_base = i * out_row;
            for ox in 0..ow {
                for c in 0..ch {
                    let mut acc = 0f64;
                    for tap in taps {
                        acc += tap.weight * strip[(tap.idx - sy_lo) * out_row + ox * ch + c] as f64;
                    }
                    out[d_base + ox * ch + c] = T::from_f64(if round { acc.round() } else { acc });
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        dst[..oh * out_row]
            .par_chunks_mut(band_rows * out_row)
            .enumerate()
            .for_each(|(b, out)| do_band(b, out));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (b, out) in dst[..oh * out_row]
            .chunks_mut(band_rows * out_row)
            .enumerate()
        {
            do_band(b, out);
        }
    }
}
