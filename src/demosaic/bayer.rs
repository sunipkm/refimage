//! Bayer image definitions.

use crate::demosaic::BayerResult;
/// The 2x2 colour filter array (CFA) pattern.
///
/// The sequence of R, G, B describe the colours of the top-left,
/// top-right, bottom-left, and bottom-right pixels in the 2x2 block,
/// in that order.
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub enum ColorFilterArray {
    Bggr,
    Gbrg,
    Grbg,
    Rggb,
}

/// Trait for reading Bayer lines.
pub trait BayerRead<T> {
    fn read_row(&self, r: &[T], dst: &mut [T]) -> BayerResult<()>;
}

impl ColorFilterArray {
    /// The 2x2 pixel block obtained when moving right 1 column.
    pub fn next_x(self) -> Self {
        match self {
            ColorFilterArray::Bggr => ColorFilterArray::Gbrg,
            ColorFilterArray::Gbrg => ColorFilterArray::Bggr,
            ColorFilterArray::Grbg => ColorFilterArray::Rggb,
            ColorFilterArray::Rggb => ColorFilterArray::Grbg,
        }
    }

    /// The 2x2 pixel block obtained when moving down 1 row.
    pub fn next_y(self) -> Self {
        match self {
            ColorFilterArray::Bggr => ColorFilterArray::Grbg,
            ColorFilterArray::Gbrg => ColorFilterArray::Rggb,
            ColorFilterArray::Grbg => ColorFilterArray::Bggr,
            ColorFilterArray::Rggb => ColorFilterArray::Gbrg,
        }
    }
}
