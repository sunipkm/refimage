//! Concrete error types.

use thiserror::Error;

/// Errors from constructing an image or reinterpreting its backing bytes.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ImageError {
    /// A dimension exceeds the 65535-pixel limit, or `width * height * channels`
    /// overflows `usize`.
    #[error("image dimensions are too large")]
    TooLarge,
    /// The backing store is empty.
    #[error("image data is empty")]
    EmptyData,
    /// The width is zero.
    #[error("image width is zero")]
    ZeroWidth,
    /// The height is zero.
    #[error("image height is zero")]
    ZeroHeight,
    /// The backing store is shorter than `width * height * channels`.
    #[error("not enough data for image: need {expected} elements, got {got}")]
    InsufficientData {
        /// Elements the image requires.
        expected: usize,
        /// Elements actually supplied.
        got: usize,
    },
    /// A byte value did not name a valid [`PixelType`](crate::PixelType).
    #[error("invalid pixel type discriminant: {0}")]
    InvalidPixelType(i8),
    /// Reinterpreting `&[u8]` as the pixel type failed (alignment, size, slop).
    #[error("byte cast failed: {0}")]
    Cast(&'static str),
    /// A type-erased image was downcast to the wrong concrete pixel type.
    #[error("dynamic image is not of the requested pixel type")]
    PixelTypeMismatch,
}

/// `Result` alias for [`ImageError`].
pub type ImageResult<T> = Result<T, ImageError>;
