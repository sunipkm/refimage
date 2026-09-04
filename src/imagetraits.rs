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

/// Type-erased access to an image's raw sample buffer.
///
/// Implemented by [`DynamicImageRef`](crate::DynamicImageRef),
/// [`DynamicImageOwned`](crate::DynamicImageOwned),
/// [`GenericImageRef`](crate::GenericImageRef),
/// [`GenericImageOwned`](crate::GenericImageOwned) and
/// [`GenericImage`](crate::GenericImage), so the same accessors work whatever
/// concrete type a value has.
///
/// The typed accessors ([`as_slice_u16`](Self::as_slice_u16) etc.) return `None`
/// unless the image's element type matches exactly — a `u16` buffer tagged
/// 10-/12-/14-bit still reads back through [`as_slice_u16`](Self::as_slice_u16).
/// A returned slice holds [`len`](ImageProps::len) samples, which can be shorter
/// than the backing allocation.
pub trait PixelData {
    /// The whole sample buffer reinterpreted as bytes (native endianness).
    fn as_raw_u8(&self) -> &[u8];

    /// [`as_raw_u8`](Self::as_raw_u8), but `None` on a reinterpret-cast failure
    /// instead of panicking.
    fn as_raw_u8_checked(&self) -> Option<&[u8]>;

    /// The samples as `&[u8]`, or `None` unless the element type is `u8`.
    fn as_slice_u8(&self) -> Option<&[u8]>;

    /// The samples as `&[u16]`, or `None` unless the element type is `u16`.
    fn as_slice_u16(&self) -> Option<&[u16]>;

    /// The samples as `&[f32]`, or `None` unless the element type is `f32`.
    fn as_slice_f32(&self) -> Option<&[f32]>;

    /// The samples as `&mut [u8]`, or `None` unless the element type is `u8`.
    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]>;

    /// The samples as `&mut [u16]`, or `None` unless the element type is `u16`.
    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]>;

    /// The samples as `&mut [f32]`, or `None` unless the element type is `f32`.
    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]>;
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
