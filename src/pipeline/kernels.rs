//! The per-pixel-op kernels: demosaic, pixel-type conversion, affine scale, and
//! luminance. Each rewrites an `f32`-backed byte buffer in place (or, for
//! demosaic, from one buffer into another).

use bytemuck::cast_slice_mut;

use crate::demosaic::{
    run_demosaic_imagedata, run_demosaic_imagedata_serial, ColorFilterArray, RasterMut,
};
use crate::{
    BayerPattern, ColorSpace, DemosaicMethod, Enlargeable, ImageRef, PixelStor, PixelType,
};

use super::spec::pixel_size;
use super::PipelineError;

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
        other => Err(PipelineError::UnsupportedPixelType(other)),
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

/// Convert one packed element; returns up to 4 bytes, of which the caller writes
/// `pixel_size(out_pt)`. `in_pt`/`out_pt` are assumed distinct and supported.
fn convert_one(src: &[u8], in_pt: PixelType, out_pt: PixelType) -> [u8; 4] {
    let pad2 = |b: [u8; 2]| [b[0], b[1], 0, 0];
    let pad1 = |b: u8| [b, 0, 0, 0];
    match (in_pt, out_pt) {
        (PixelType::U8, PixelType::U16) => pad2(src[0].cast_u16().to_ne_bytes()),
        (PixelType::U8, PixelType::F32) => src[0].cast_f32().to_ne_bytes(),
        (PixelType::U16, PixelType::U8) => pad1(u16::from_ne_bytes([src[0], src[1]]).cast_u8()),
        (PixelType::U16, PixelType::F32) => u16::from_ne_bytes([src[0], src[1]])
            .cast_f32()
            .to_ne_bytes(),
        (PixelType::F32, PixelType::U8) => {
            pad1(f32::from_ne_bytes([src[0], src[1], src[2], src[3]]).cast_u8())
        }
        (PixelType::F32, PixelType::U16) => pad2(
            f32::from_ne_bytes([src[0], src[1], src[2], src[3]])
                .cast_u16()
                .to_ne_bytes(),
        ),
        _ => unreachable!("identity handled by caller; other types rejected by pixel_size"),
    }
}

/// Apply `y = x * gain + offset` to `n` raw elements of `buf` in place,
/// saturating back into `pt` (`[0.0, 1.0]` for `f32`).
pub(super) fn scale_inplace(
    buf: &mut [f32],
    pt: PixelType,
    n: usize,
    gain: f64,
    offset: f64,
) -> Result<(), PipelineError> {
    match pt {
        PixelType::U8 => {
            for x in &mut cast_slice_mut::<f32, u8>(buf)[..n] {
                *x = u8::from_f64((*x).to_f64() * gain + offset);
            }
        }
        PixelType::U16 => {
            for x in &mut cast_slice_mut::<f32, u16>(buf)[..n] {
                *x = u16::from_f64((*x).to_f64() * gain + offset);
            }
        }
        PixelType::F32 => {
            for x in &mut buf[..n] {
                *x = f32::from_f64((*x).to_f64() * gain + offset);
            }
        }
        other => return Err(PipelineError::UnsupportedPixelType(other)),
    }
    Ok(())
}

pub(super) fn luma_inplace(
    buf: &mut [f32],
    in_pt: PixelType,
    channels: usize,
    total: usize,
    weights: &[f64],
) -> Result<(), PipelineError> {
    match in_pt {
        PixelType::U8 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, u8>(buf), weights)
        }
        PixelType::U16 => {
            crate::coreimpls::run_luma(channels, total, cast_slice_mut::<f32, u16>(buf), weights)
        }
        PixelType::F32 => crate::coreimpls::run_luma(channels, total, buf, weights),
        other => return Err(PipelineError::UnsupportedPixelType(other)),
    }
    Ok(())
}
