//! Bayer image definitions.

use crate::demosaic::BayerResult;
/// The 2x2 colour filter array (CFA) pattern.
///
/// The sequence of R, G, B describe the colours of the top-left,
/// top-right, bottom-left, and bottom-right pixels in the 2x2 block,
/// in that order.
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub enum CFA {
    BGGR,
    GBRG,
    GRBG,
    RGGB,
}

/// Trait for reading Bayer lines.
pub trait BayerRead<T> {
    fn read_row(&self, r: &[T], dst: &mut [T]) -> BayerResult<()>;
}

impl CFA {
    /// The 2x2 pixel block obtained when moving right 1 column.
    pub fn next_x(self) -> Self {
        match self {
            CFA::BGGR => CFA::GBRG,
            CFA::GBRG => CFA::BGGR,
            CFA::GRBG => CFA::RGGB,
            CFA::RGGB => CFA::GRBG,
        }
    }

    /// The 2x2 pixel block obtained when moving down 1 row.
    pub fn next_y(self) -> Self {
        match self {
            CFA::BGGR => CFA::GRBG,
            CFA::GBRG => CFA::RGGB,
            CFA::GRBG => CFA::BGGR,
            CFA::RGGB => CFA::GBRG,
        }
    }
}
