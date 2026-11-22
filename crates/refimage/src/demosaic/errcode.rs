//! Bayer error codes.

use thiserror::Error;

use crate::ColorSpace;

pub type BayerResult<T> = Result<T, BayerError>;

#[derive(Debug, Error)]
/// Error codes for the Bayer demosaicing.
pub enum BayerError {
    #[error("{0:?} is not a Bayer color space")]
    /// The image's color space is not a Bayer mosaic.
    InvalidColorSpace(ColorSpace),
    #[error("Wrong color resolution")]
    /// The image is not the right size.
    WrongResolution,
    #[error("Wrong color depth")]
    /// The image is not the right depth.
    WrongDepth,
}
