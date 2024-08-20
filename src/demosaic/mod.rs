mod bayer;
mod errcode;
mod border_mirror;
mod border_none;
mod border_replicate;
#[macro_use]
mod none;
mod raster;
#[macro_use]
mod rotate;
mod nearestneighbour;
mod linear;
mod cubic;

pub use errcode::BayerError;
pub use errcode::BayerResult;
pub use bayer::{BayerRead, ColorFilterArray};

use crate::traits::Enlargeable;
use crate::ImageData;
use crate::PixelStor;

/// Mutable raster structure.
pub(crate)struct RasterMut<'a, T: PixelStor> {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    stride: usize,
    buf: &'a mut [T],
}

/// The demosaicing algorithm to use to fill in the missing data.
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub enum Demosaic {
    /// No interpolation.
    None,
    /// Nearest neighbour interpolation.
    NearestNeighbour,
    /// Linear interpolation.
    Linear,
    /// Cubic interpolation.
    Cubic,
}

pub(crate) fn run_demosaic<T>(r: &ImageData<T>, cfa: ColorFilterArray, alg: Demosaic,
    dst: &mut RasterMut<'_, T>)
    -> BayerResult<()> 
    where T: PixelStor + Enlargeable {
match alg {
    Demosaic::None => crate::demosaic::none::run(r, cfa, dst),
    Demosaic::NearestNeighbour => crate::demosaic::nearestneighbour::run(r, cfa, dst),
    Demosaic::Linear => crate::demosaic::linear::run(r, cfa, dst),
    Demosaic::Cubic => crate::demosaic::cubic::run(r, cfa, dst),
}
}