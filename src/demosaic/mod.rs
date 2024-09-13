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

use crate::traits::Enlargeable;
use crate::ImageData;
use crate::ImageOwned;
use crate::PixelStor;
#[allow(unused_imports)]
use crate::{DynamicImageData, DynamicImageOwned, GenericImage, GenericImageOwned};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    r: &ImageData<T>,
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

pub(crate) fn run_demosaic_imageowned<T>(
    r: &ImageOwned<T>,
    cfa: ColorFilterArray,
    alg: DemosaicMethod,
    dst: &mut RasterMut<'_, T>,
) -> BayerResult<()>
where
    T: PixelStor + Enlargeable,
{
    match alg {
        DemosaicMethod::None => crate::demosaic::none::run_imageowned(r, cfa, dst),
        DemosaicMethod::Nearest => crate::demosaic::nearestneighbour::run_imageowned(r, cfa, dst),
        DemosaicMethod::Linear => crate::demosaic::linear::run_imageowned(r, cfa, dst),
        DemosaicMethod::Cubic => crate::demosaic::cubic::run_imageowned(r, cfa, dst),
    }
}

/// Trait to apply a Demosaic algorithm to an image.
///
/// This trait is implemented for [`ImageData`], [`DynamicImageData`], [`GenericImage`] and
/// their owned counterparts, [`ImageOwned`], [`DynamicImageOwned`] and [`GenericImageOwned`].
pub trait Debayer<'b: 'a, 'a>
where
    Self: Sized,
{
    /// Debayer the image.
    ///
    /// This function returns an error if the image is not a Bayer pattern image.
    ///
    /// # Arguments
    /// - `alg`: The demosaicing algorithm to use.
    ///
    /// Possible algorithms are:
    /// - [`DemosaicMethod::None`]: No interpolation.
    /// - [`DemosaicMethod::Nearest`]: Nearest neighbour interpolation.
    /// - [`DemosaicMethod::Linear`]: Linear interpolation.
    /// - [`DemosaicMethod::Cubic`]: Cubic interpolation.
    ///
    /// # Errors
    /// - If the image is not a Bayer pattern image.
    /// - If the image is not a single channel image.
    fn debayer(&'b self, alg: DemosaicMethod) -> Result<Self, BayerError>;
}
