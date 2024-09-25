use crate::{demosaic::ColorFilterArray, BayerPattern, ColorSpace, PixelType};

impl TryFrom<i8> for PixelType {
    type Error = &'static str;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::U8),
            16 => Ok(Self::U16),
            -32 => Ok(Self::F32),
            _ => Err("Invalid value for PixelType"),
        }
    }
}

impl TryInto<ColorFilterArray> for ColorSpace {
    type Error = &'static str;

    fn try_into(self) -> Result<ColorFilterArray, Self::Error> {
        match self {
            ColorSpace::Bayer(pat) => match pat {
                BayerPattern::Bggr => Ok(ColorFilterArray::Bggr),
                BayerPattern::Gbrg => Ok(ColorFilterArray::Gbrg),
                BayerPattern::Grbg => Ok(ColorFilterArray::Grbg),
                BayerPattern::Rggb => Ok(ColorFilterArray::Rggb),
            },
            ColorSpace::Gray | ColorSpace::GrayAlpha => {
                Err("Gray color space not supported for Bayer images.")
            }
            ColorSpace::Rgb | ColorSpace::Rgba => {
                Err("RGB color space not supported for Bayer images.")
            }
            ColorSpace::Custom(_) => Err("Custom color space not supported for Bayer images."),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<ColorSpace> for BayerPattern {
    fn into(self) -> ColorSpace {
        ColorSpace::Bayer(self)
    }
}
