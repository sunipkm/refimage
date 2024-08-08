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
pub use bayer::{BayerRead, CFA};

use crate::Primitive;

/// Mutable raster structure.
pub struct RasterMut<'a, T: Primitive> {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    stride: usize,
    buf: &'a mut [T],
}
