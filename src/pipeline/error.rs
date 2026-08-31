//! The [`PipelineError`](PipelineError) type.

use thiserror::Error;

#[allow(unused_imports)]
use crate::{
    pipeline::{Op, Pipeline, Runner},
    BayerError, ImageError, PixelType,
};

use super::ImageSpec;

#[derive(Debug, Error)]
#[non_exhaustive]
/// Errors from compiling or running a [`Pipeline`].
pub enum PipelineError {
    /// A pixel type outside the supported `{u8, u16, f32}` set was requested.
    #[error("unsupported pixel type: {0:?}")]
    UnsupportedPixelType(PixelType),
    /// An image dimension is zero or exceeds 65535.
    #[error("image dimension is zero or exceeds 65535")]
    BadDimensions,
    /// [`Op::Crop`] asked for a rectangle reaching outside the current image.
    #[error("crop rect {rect:?} does not fit inside image {image:?}")]
    CropOutOfBounds {
        /// The requested `(x, y, width, height)`.
        rect: (usize, usize, usize, usize),
        /// The `(width, height)` available at that point in the chain.
        image: (usize, usize),
    },
    /// [`Op::Roi`] was given an origin outside the current image.
    #[error("ROI origin {origin:?} is outside image {image:?}")]
    RoiOutOfBounds {
        /// The requested `(x, y)` origin.
        origin: (usize, usize),
        /// The `(width, height)` available at that point in the chain.
        image: (usize, usize),
    },
    /// [`Op::Rotate90`] / [`Op::Rotate270`] was applied to a Bayer image.
    #[error("90° rotation is not valid on a Bayer image; debayer first")]
    RotateOnBayer,
    /// [`Op::ResizeToFit`] was applied to a Bayer image.
    #[error("resize is not valid on a Bayer image; debayer first")]
    ResizeOnBayer,
    /// [`Op::Debayer`] was applied to a non-Bayer color space.
    #[error("debayer requires a Bayer color space")]
    NotBayer,
    /// [`Op::Debayer`] was applied to a multi-channel image.
    #[error("debayer requires a single-channel image, got {0} channels")]
    DebayerChannels(u8),
    /// A luma op was applied to a Bayer image (debayer first).
    #[error("luma conversion is not valid on a Bayer image; debayer first")]
    LumaOnBayer,
    /// Luma weight count does not match the channel count.
    #[error("luma coefficient count ({coeffs}) does not match channel count ({channels})")]
    LumaCoeffMismatch {
        /// Channels in the image at that point in the chain.
        channels: u8,
        /// Weights supplied.
        coeffs: usize,
    },
    /// The frame passed to [`Runner::run`] does not match the compiled spec
    /// (and the `grow` feature is off).
    #[error("input frame {got:?} does not match the compiled spec {expected:?}")]
    InputMismatch {
        /// Spec the runner was compiled for.
        expected: Box<ImageSpec>,
        /// Spec of the frame actually passed.
        got: Box<ImageSpec>,
    },
    /// A demosaic kernel failed.
    #[error("demosaic error: {0}")]
    Bayer(#[from] BayerError),
    /// Reconstructing the output view failed.
    #[error("image error: {0}")]
    Image(#[from] ImageError),
}
