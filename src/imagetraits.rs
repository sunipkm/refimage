use crate::{BayerPattern, ColorSpace, PixelType};

#[allow(unused_imports)]
use crate::{
    DynamicImageOwned, DynamicImageRef, GenericImage, GenericImageOwned, GenericImageRef,
    ImageOwned, ImageRef,
};

/// A trait for shifting Bayer patterns.
pub trait BayerShift {
    /// Shift the Bayer pattern by `x` and `y` pixels.
    fn shift(&self, x: usize, y: usize) -> Self;
    /// Flip the Bayer pattern horizontally.
    fn flip_horizontal(&self) -> Self;
    /// Flip the Bayer pattern vertically.
    fn flip_vertical(&self) -> Self;
}

impl BayerShift for BayerPattern {
    fn shift(&self, x: usize, y: usize) -> Self {
        match self {
            BayerPattern::Rggb => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Rggb,
                (1, 0) => BayerPattern::Gbrg,
                (0, 1) => BayerPattern::Grbg,
                (1, 1) => BayerPattern::Bggr,
                _ => unreachable!(),
            },
            BayerPattern::Gbrg => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Gbrg,
                (1, 0) => BayerPattern::Rggb,
                (0, 1) => BayerPattern::Bggr,
                (1, 1) => BayerPattern::Grbg,
                _ => unreachable!(),
            },
            BayerPattern::Grbg => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Grbg,
                (1, 0) => BayerPattern::Bggr,
                (0, 1) => BayerPattern::Rggb,
                (1, 1) => BayerPattern::Gbrg,
                _ => unreachable!(),
            },
            BayerPattern::Bggr => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Bggr,
                (1, 0) => BayerPattern::Grbg,
                (0, 1) => BayerPattern::Gbrg,
                (1, 1) => BayerPattern::Rggb,
                _ => unreachable!(),
            },
        }
    }
    fn flip_horizontal(&self) -> Self {
        match self {
            BayerPattern::Rggb => BayerPattern::Grbg,
            BayerPattern::Gbrg => BayerPattern::Bggr,
            BayerPattern::Grbg => BayerPattern::Rggb,
            BayerPattern::Bggr => BayerPattern::Gbrg,
        }
    }
    fn flip_vertical(&self) -> Self {
        match self {
            BayerPattern::Rggb => BayerPattern::Gbrg,
            BayerPattern::Gbrg => BayerPattern::Rggb,
            BayerPattern::Grbg => BayerPattern::Bggr,
            BayerPattern::Bggr => BayerPattern::Grbg,
        }
    }
}

/// A trait for converting an image to a luminance image.
///
/// This trait is implemented for [`ImageRef`], [`DynamicImageRef`], [`GenericImageRef`] and
/// their owned counterparts, [`ImageOwned`], [`DynamicImageOwned`], [`GenericImageOwned`]
/// and [`GenericImage`].
pub trait ToLuma {
    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    fn to_luma(&mut self) -> Result<(), &'static str>;

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    fn to_luma_custom(&mut self, coeffs: &[f64]) -> Result<(), &'static str>;
}

/// A trait for accessing the properties of an image.
pub trait ImageProps {
    /// The output type of [`ImageProps::cast_u8`].
    type OutputU8;
    /// Get the width of the image.
    fn width(&self) -> usize;

    /// Get the height of the image.
    fn height(&self) -> usize;

    /// Get the number of channels in the image.
    fn channels(&self) -> u8;

    /// Get the color space of the image.
    fn color_space(&self) -> ColorSpace;

    /// Get the pixel type of the image.
    fn pixel_type(&self) -> PixelType;

    /// Get the length of the image data.
    fn len(&self) -> usize;

    /// Check if the data is empty.
    fn is_empty(&self) -> bool;

    /// Convert the image to a `u8` image.
    ///
    /// Conversion is done by scaling the pixel values to the range `[0, 255]`.
    ///
    /// # Note: This operation is parallelized if the `rayon` feature is enabled.
    fn cast_u8(&self) -> Self::OutputU8;
}
