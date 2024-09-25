//! Bayer error codes.

use thiserror::Error;

pub type BayerResult<T> = Result<T, BayerError>;

#[derive(Debug, Error)]
/// Error codes for the Bayer demosaicing.
pub enum BayerError {
    #[error("Invalid Color Filter Array: {0}")]
    /// Generic failure.  Please try to make something more meaningful.
    InvalidColorSpace(&'static str),
    #[error("Wrong color resolution")]
    /// The image is not the right size.
    WrongResolution,
    #[error("Wrong color depth")]
    /// The image is not the right depth.
    WrongDepth,
}
