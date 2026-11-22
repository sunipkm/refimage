//! The per-pixel-op kernels: demosaic, pixel-type conversion, affine scale, and
//! luminance. Each rewrites an `f32`-backed byte buffer in place (or, for
//! demosaic, from one buffer into another).

use bytemuck::cast_slice_mut;

use crate::demosaic::{
    ColorFilterArray, RasterMut, run_demosaic_imagedata, run_demosaic_imagedata_serial,
};
use crate::{
    BayerPattern, ColorSpace, DemosaicMethod, Enlargeable, ImageRef, PixelStor, PixelType, U10,
    U12, U14,
};

use super::spec::pixel_size;
use super::{PipelineError, ScaleFactor};

/// How the debayer step gets its working memory.
pub(super) enum Demosaic<'s> {
    /// Whole-frame path: the kernel manages its own scratch (may parallelise).
    Alloc,
    /// Tile path: serial kernel, working rows taken from this pooled buffer.
    Pooled(&'s mut [f32]),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn debayer_into(
    src: &mut [f32],
    dst: &mut [f32],
    in_pt: PixelType,
    w: usize,
    h: usize,
    pat: BayerPattern,
    method: DemosaicMethod,
    demosaic: &mut Demosaic<'_>,
) -> Result<(), PipelineError> {
    match in_pt {
        PixelType::U8 => debayer_typed::<u8>(
            cast_slice_mut(src),
            cast_slice_mut(dst),
            w,
            h,
            pat,
            method,
            demosaic,
        ),
        // `U10`/`U12`/`U14` clamp interpolation overshoot against the sensor's
        // true range instead of the full `u16` one.
        PixelType::U10 => debayer_typed::<U10>(
            cast_slice_mut(src),
            cast_slice_mut(dst),
            w,
            h,
            pat,
            method,
            demosaic,
        ),
        PixelType::U12 => debayer_typed::<U12>(
            cast_slice_mut(src),
            cast_slice_mut(dst),
            w,
            h,
            pat,
            method,
            demosaic,
        ),
        PixelType::U14 => debayer_typed::<U14>(
            cast_slice_mut(src),
            cast_slice_mut(dst),
            w,
            h,
            pat,
            method,
            demosaic,
        ),
        PixelType::U16 => debayer_typed::<u16>(
            cast_slice_mut(src),
            cast_slice_mut(dst),
            w,
            h,
            pat,
            method,
            demosaic,
        ),
        PixelType::F32 => debayer_typed::<f32>(src, dst, w, h, pat, method, demosaic),
    }
}

#[allow(clippy::too_many_arguments)]
fn debayer_typed<T: PixelStor + Enlargeable + bytemuck::AnyBitPattern>(
    src: &mut [T],
    dst: &mut [T],
    w: usize,
    h: usize,
    pat: BayerPattern,
    method: DemosaicMethod,
    demosaic: &mut Demosaic<'_>,
) -> Result<(), PipelineError> {
    let elems = w * h;
    let cfa: ColorFilterArray = ColorSpace::Bayer(pat).try_into()?;
    let img = ImageRef::<T> {
        data: &mut src[..elems],
        len: elems,
        width: w as u16,
        height: h as u16,
        cspace: ColorSpace::Bayer(pat),
        bit_depth: None,
    };
    let mut raster = RasterMut::new(w, h, &mut dst[..w * h * 3]);
    match demosaic {
        Demosaic::Alloc => run_demosaic_imagedata(&img, cfa, method, &mut raster)?,
        Demosaic::Pooled(pool) => {
            let scratch: &mut [T] = cast_slice_mut(pool);
            run_demosaic_imagedata_serial(&img, cfa, method, &mut raster, scratch)?
        }
    }
    Ok(())
}

/// Rewrite `n` elements of `buf` from `in_pt` to `out_pt` in place, with the same
/// per-pixel rescaling as [`PixelStor::cast_u8`] and friends. Shrinking and
/// equal-size conversions run forward, growing conversions run backward, so the
/// write for element `i` never lands on bytes of an element not yet read.
pub(super) fn convert_inplace(
    buf: &mut [f32],
    in_pt: PixelType,
    out_pt: PixelType,
    n: usize,
) -> Result<(), PipelineError> {
    let is = pixel_size(in_pt)?;
    let os = pixel_size(out_pt)?;
    if in_pt == out_pt {
        return Ok(());
    }
    let b = cast_slice_mut::<f32, u8>(buf);
    let mut apply = |i: usize| {
        let o = convert_one(&b[i * is..i * is + is], in_pt, out_pt);
        b[i * os..i * os + os].copy_from_slice(&o[..os]);
    };
    if os <= is {
        for i in 0..n {
            apply(i);
        }
    } else {
        for i in (0..n).rev() {
            apply(i);
        }
    }
    Ok(())
}

/// Rescale one already-decoded source sample into the packed bytes of
/// `out_pt` (`pixel_size(out_pt)` of them are meaningful). `T::cast_u8` /
/// `cast_u16` / `cast_f32` saturate against `T::DEFAULT_MIN/MAX_VALUE`, so a
/// tagged `U10`/`U12`/`U14` source rescales against its true range rather than
/// the full `u16` one. `out_pt` is always a storage type — never `U10`/`U12`/
/// `U14` — because [`Op::Convert`](super::Op::Convert) rejects any other
/// target ([`PipelineError::ConvertTargetNotStorage`]).
fn cast_to_bytes<T: PixelStor>(v: T, out_pt: PixelType) -> [u8; 4] {
    let pad2 = |b: [u8; 2]| [b[0], b[1], 0, 0];
    let pad1 = |b: u8| [b, 0, 0, 0];
    match out_pt {
        PixelType::U8 => pad1(v.cast_u8()),
        PixelType::U16 => pad2(v.cast_u16().to_ne_bytes()),
        PixelType::F32 => v.cast_f32().to_ne_bytes(),
        _ => unreachable!("Op::Convert target is always a storage type"),
    }
}

/// Convert one packed element; returns up to 4 bytes, of which the caller
/// writes `pixel_size(out_pt)`. `in_pt`/`out_pt` are assumed distinct.
fn convert_one(src: &[u8], in_pt: PixelType, out_pt: PixelType) -> [u8; 4] {
    let u16_at = |src: &[u8]| u16::from_ne_bytes([src[0], src[1]]);
    match in_pt {
        PixelType::U8 => cast_to_bytes(src[0], out_pt),
        PixelType::U10 => cast_to_bytes(U10(u16_at(src)), out_pt),
        PixelType::U12 => cast_to_bytes(U12(u16_at(src)), out_pt),
        PixelType::U14 => cast_to_bytes(U14(u16_at(src)), out_pt),
        PixelType::U16 => cast_to_bytes(u16_at(src), out_pt),
        PixelType::F32 => {
            cast_to_bytes(f32::from_ne_bytes([src[0], src[1], src[2], src[3]]), out_pt)
        }
    }
}

/// `y = x * gain + offset` in raw storage units, saturating back into `T`
/// (`T::DEFAULT_MIN/MAX_VALUE` — the true sensor range for `U10`/`U12`/`U14`).
fn scale_typed<T: PixelStor + bytemuck::AnyBitPattern>(
    buf: &mut [f32],
    n: usize,
    gain: f64,
    offset: f64,
) {
    for x in &mut cast_slice_mut::<f32, T>(buf)[..n] {
        *x = T::from_f64((*x).to_f64() * gain + offset);
    }
}

/// Apply `y = x * gain + offset` to `n` raw elements of `buf` in place,
/// saturating back into `pt` (`[0.0, 1.0]` for `f32`).
pub(super) fn scale_inplace(buf: &mut [f32], pt: PixelType, n: usize, gain: f64, offset: f64) {
    match pt {
        PixelType::U8 => scale_typed::<u8>(buf, n, gain, offset),
        PixelType::U10 => scale_typed::<U10>(buf, n, gain, offset),
        PixelType::U12 => scale_typed::<U12>(buf, n, gain, offset),
        PixelType::U14 => scale_typed::<U14>(buf, n, gain, offset),
        PixelType::U16 => scale_typed::<u16>(buf, n, gain, offset),
        PixelType::F32 => scale_typed::<f32>(buf, n, gain, offset),
    }
}

/// `round(x * num / den)` for `n` raw elements of `buf`, in widened integer
/// arithmetic (no float round-trip), clamped into `T::DEFAULT_MIN/MAX_VALUE` —
/// the sensor's true range for `U10`/`U12`/`U14`, not the full `u16` one.
fn scale_pixels_rational_typed<T: PixelStor + bytemuck::AnyBitPattern>(
    buf: &mut [f32],
    n: usize,
    num: i128,
    den: i128,
) {
    let lo = T::DEFAULT_MIN_VALUE.to_i64().unwrap() as i128;
    let hi = T::DEFAULT_MAX_VALUE.to_i64().unwrap() as i128;
    for x in &mut cast_slice_mut::<f32, T>(buf)[..n] {
        let raw = x.to_i64().unwrap() as i128;
        let v = rational_round(raw, num, den).clamp(lo, hi);
        *x = <T as num_traits::NumCast>::from(v as i64).unwrap();
    }
}

/// Multiply `n` raw elements of `buf` in place by `factor`, saturating back into
/// `pt` (`[0.0, 1.0]` for `f32`). [`ScaleFactor::Rational`] runs in widened
/// integer arithmetic on integer types (no float round-trip); [`ScaleFactor::Float`]
/// reuses the affine-scale rounding with a zero offset.
pub(super) fn scale_pixels_inplace(
    buf: &mut [f32],
    pt: PixelType,
    n: usize,
    factor: ScaleFactor,
) -> Result<(), PipelineError> {
    let ScaleFactor::Rational { num, den } = factor else {
        let ScaleFactor::Float(f) = factor else {
            unreachable!()
        };
        scale_inplace(buf, pt, n, f, 0.0);
        return Ok(());
    };
    if den == 0 {
        return Err(PipelineError::BadScaleFactor);
    }
    let (num, den) = (num as i128, den as i128);
    match pt {
        PixelType::U8 => scale_pixels_rational_typed::<u8>(buf, n, num, den),
        PixelType::U10 => scale_pixels_rational_typed::<U10>(buf, n, num, den),
        PixelType::U12 => scale_pixels_rational_typed::<U12>(buf, n, num, den),
        PixelType::U14 => scale_pixels_rational_typed::<U14>(buf, n, num, den),
        PixelType::U16 => scale_pixels_rational_typed::<u16>(buf, n, num, den),
        PixelType::F32 => scale_typed::<f32>(buf, n, num as f64 / den as f64, 0.0),
    }
    Ok(())
}

/// `round(x * num / den)` (round half away from zero), sign-correct for a
/// negative `num` or `den`.
fn rational_round(x: i128, num: i128, den: i128) -> i128 {
    let (mut p, mut q) = (x * num, den);
    if q < 0 {
        p = -p;
        q = -q;
    }
    if p >= 0 {
        (p + q / 2) / q
    } else {
        -((-p + q / 2) / q)
    }
}

pub(super) fn luma_inplace(
    buf: &mut [f32],
    in_pt: PixelType,
    channels: usize,
    total: usize,
    weights: &[f64],
) {
    match in_pt {
        PixelType::U8 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, u8>(buf), weights)
        }
        PixelType::U10 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, U10>(buf), weights)
        }
        PixelType::U12 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, U12>(buf), weights)
        }
        PixelType::U14 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, U14>(buf), weights)
        }
        PixelType::U16 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, u16>(buf), weights)
        }
        PixelType::F32 => crate::coreimpls::run_luma(channels, total, buf, weights),
    }
}
