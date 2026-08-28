use crate::{BayerPattern, ColorSpace, PixelType};

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

/// A trait for accessing the properties of an image.
pub trait ImageProps {
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
}
