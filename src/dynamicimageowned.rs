use crate::{BayerError, ColorSpace, DemosaicMethod, DynamicImageOwned, ImageOwned, PixelType};
use crate::{Debayer, DynamicImageData};

macro_rules! dynamic_map(
    ($dynimage: expr, $image: pat => $action: expr) => ({
        use DynamicImageOwned::*;
        match $dynimage {
            U8($image) => U8($action),
            U16($image) => U16($action),
            F32($image) => F32($action),
        }
    });

    ($dynimage: expr, $image:pat_param, $action: expr) => (
        match $dynimage {
            DynamicImageOwned::U8($image) => $action,
            DynamicImageOwned::U16($image) => $action,
            DynamicImageOwned::F32($image) => $action,
        }
    );
);

impl DynamicImageOwned {
    /// Get the width of the image.
    pub fn width(&self) -> usize {
        dynamic_map!(self, ref image, { image.width() })
    }

    /// Get the height of the image.
    pub fn height(&self) -> usize {
        dynamic_map!(self, ref image, { image.height() })
    }

    /// Get the number of channels in the image.
    pub fn channels(&self) -> u8 {
        dynamic_map!(self, ref image, { image.channels() })
    }

    /// Get the color space of the image.
    pub fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, ref image, { image.color_space() })
    }
}

impl<'a: 'b, 'b> Debayer<'a, 'b> for DynamicImageOwned {
    fn debayer(&self, alg: DemosaicMethod) -> Result<Self, BayerError> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.debayer(alg)?)),
            U16(image) => Ok(U16(image.debayer(alg)?)),
            F32(image) => Ok(F32(image.debayer(alg)?)),
        }
    }
}

impl DynamicImageOwned {
    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma(&self) -> Result<DynamicImageOwned, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.into_luma()?)),
            U16(image) => Ok(U16(image.into_luma()?)),
            F32(image) => Ok(F32(image.into_luma()?)),
        }
    }

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion. The number of weights must match
    ///   the number of channels in the image.
    ///
    /// # Errors
    /// - If the number of weights does not match the number of channels in the image.
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_custom(&self, coeffs: &[f64]) -> Result<DynamicImageOwned, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.into_luma_custom(coeffs)?)),
            U16(image) => Ok(U16(image.into_luma_custom(coeffs)?)),
            F32(image) => Ok(F32(image.into_luma_custom(coeffs)?)),
        }
    }

    /// Convert the image to a [`DynamicImageOwned`] with [`u8`] pixel type.
    ///
    /// Note: This operation is parallelized if the `rayon` feature is enabled.
    pub fn into_u8(self) -> DynamicImageOwned {
        match self {
            DynamicImageOwned::U8(data) => DynamicImageOwned::U8(data),
            DynamicImageOwned::U16(data) => DynamicImageOwned::U8(data.into_u8()),
            DynamicImageOwned::F32(data) => DynamicImageOwned::U8(data.into_u8()),
        }
    }
}

impl From<&DynamicImageOwned> for PixelType {
    fn from(data: &DynamicImageOwned) -> Self {
        match data {
            DynamicImageOwned::U8(_) => PixelType::U8,
            DynamicImageOwned::U16(_) => PixelType::U16,
            DynamicImageOwned::F32(_) => PixelType::F32,
        }
    }
}

macro_rules! tryfrom_dynimgdata_imgdata {
    ($type:ty, $variant:path) => {
        impl<'a> TryFrom<DynamicImageOwned> for ImageOwned<$type> {
            type Error = &'static str;

            fn try_from(data: DynamicImageOwned) -> Result<Self, Self::Error> {
                match data {
                    $variant(data) => Ok(data),
                    _ => Err("Data is not of type u8"),
                }
            }
        }
    };
}

tryfrom_dynimgdata_imgdata!(u8, DynamicImageOwned::U8);
tryfrom_dynimgdata_imgdata!(u16, DynamicImageOwned::U16);
tryfrom_dynimgdata_imgdata!(f32, DynamicImageOwned::F32);

macro_rules! from_imgdata_dynimg {
    ($type:ty, $variant:path) => {
        impl<'a> From<ImageOwned<$type>> for DynamicImageOwned {
            fn from(data: ImageOwned<$type>) -> Self {
                $variant(data)
            }
        }
    };
}

from_imgdata_dynimg!(u8, DynamicImageOwned::U8);
from_imgdata_dynimg!(u16, DynamicImageOwned::U16);
from_imgdata_dynimg!(f32, DynamicImageOwned::F32);

impl DynamicImageOwned {
    /// Get the data as a slice of `u8`, regardless of the underlying type.
    pub fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, ref image, { image.as_u8_slice() })
    }

    /// Get the data as a slice of `u8`, regardless of the underlying type.
    pub fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, ref image, { image.as_u8_slice_checked() })
    }

    /// Get the data as a slice of `u8`.
    pub fn as_slice_u8(&self) -> Option<&[u8]> {
        match self {
            DynamicImageOwned::U8(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `u8`.
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        match self {
            DynamicImageOwned::U8(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of `u16`.
    pub fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageOwned::U16(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `u16`.
    pub fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        match self {
            DynamicImageOwned::U16(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of `f32`.
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageOwned::F32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `f32`.
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        match self {
            DynamicImageOwned::F32(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }
}

impl<'a> From<DynamicImageData<'a>> for DynamicImageOwned {
    fn from(data: DynamicImageData<'a>) -> Self {
        match data {
            DynamicImageData::U8(data) => DynamicImageOwned::U8(data.into()),
            DynamicImageData::U16(data) => DynamicImageOwned::U16(data.into()),
            DynamicImageData::F32(data) => DynamicImageOwned::F32(data.into()),
        }
    }
}
