mod bayer;
mod border_mirror;
mod border_none;
mod border_replicate;
mod errcode;
#[macro_use]
mod none;
mod raster;
#[macro_use]
mod rotate;
mod cubic;
mod linear;
mod nearestneighbour;

pub use bayer::{BayerRead, ColorFilterArray};
pub use errcode::BayerError;
pub use errcode::BayerResult;

use crate::coretraits::Enlargeable;
use crate::ImageProps;
use crate::ImageRef;
use crate::PixelStor;

/// Mutable raster structure.
pub(crate) struct RasterMut<'a, T: PixelStor> {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    stride: usize,
    buf: &'a mut [T],
}

/// The demosaicing algorithm to use to fill in the missing color channels.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DemosaicMethod {
    /// No interpolation.
    None,
    /// Nearest neighbour interpolation.
    Nearest,
    /// Linear interpolation.
    Linear,
    /// Cubic interpolation.
    Cubic,
}

pub(crate) fn run_demosaic_imagedata<T>(
    r: &ImageRef<T>,
    cfa: ColorFilterArray,
    alg: DemosaicMethod,
    dst: &mut RasterMut<'_, T>,
) -> BayerResult<()>
where
    T: PixelStor + Enlargeable,
{
    match alg {
        DemosaicMethod::None => crate::demosaic::none::run_imagedata(r, cfa, dst),
        DemosaicMethod::Nearest => crate::demosaic::nearestneighbour::run_imagedata(r, cfa, dst),
        DemosaicMethod::Linear => crate::demosaic::linear::run_imagedata(r, cfa, dst),
        DemosaicMethod::Cubic => crate::demosaic::cubic::run_imagedata(r, cfa, dst),
    }
}

/// Scratch elements a serial demosaic run needs for a `width`-pixel image,
/// across every [`DemosaicMethod`]. Size a pool once with this.
pub(crate) fn demosaic_serial_scratch_len(width: usize) -> usize {
    use crate::demosaic::{cubic, linear, nearestneighbour, none};
    none::serial_scratch_len(width)
        .max(nearestneighbour::serial_scratch_len(width))
        .max(linear::serial_scratch_len(width))
        .max(cubic::serial_scratch_len(width))
}

/// Demosaic without spawning kernel-internal parallelism, writing all working
/// rows into `scratch` (at least [`demosaic_serial_scratch_len`] elements).
///
/// This is the entry point the tiled pipeline uses: strips are already the unit
/// of parallelism, so the kernel itself must stay serial, and the scratch is
/// pooled instead of allocated per strip.
pub(crate) fn run_demosaic_imagedata_serial<T>(
    r: &ImageRef<T>,
    cfa: ColorFilterArray,
    alg: DemosaicMethod,
    dst: &mut RasterMut<'_, T>,
    scratch: &mut [T],
) -> BayerResult<()>
where
    T: PixelStor + Enlargeable,
{
    if r.width() < 2 || r.height() < 2 {
        return Err(BayerError::WrongResolution);
    }
    let s = r.as_slice();
    match alg {
        DemosaicMethod::None => crate::demosaic::none::debayer_serial(s, cfa, dst, scratch),
        DemosaicMethod::Nearest => {
            crate::demosaic::nearestneighbour::debayer_serial(s, cfa, dst, scratch)
        }
        DemosaicMethod::Linear => crate::demosaic::linear::debayer_serial(s, cfa, dst, scratch),
        DemosaicMethod::Cubic => crate::demosaic::cubic::debayer_serial(s, cfa, dst, scratch),
    }
}
