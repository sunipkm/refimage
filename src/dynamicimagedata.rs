use crate::{BayerError, ColorSpace, DemosaicMethod, DynamicImageData, ImageData, PixelType};
use crate::{Debayer, DynamicImageOwned};

macro_rules! dynamic_map(
    ($dynimage: expr, $image: pat => $action: expr) => ({
        use DynamicImageData::*;
        match $dynimage {
            U8($image) => U8($action),
            U16($image) => U16($action),
            F32($image) => F32($action),
        }
    });

    ($dynimage: expr, $image:pat_param, $action: expr) => (
        match $dynimage {
            DynamicImageData::U8($image) => $action,
            DynamicImageData::U16($image) => $action,
            DynamicImageData::F32($image) => $action,
        }
    );
);

impl<'a> DynamicImageData<'a> {
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

impl<'a: 'b, 'b> Debayer<'a, 'b> for DynamicImageData<'b> {
    fn debayer(&'b self, alg: DemosaicMethod) -> Result<Self, BayerError> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(U8(image.debayer(alg)?)),
            U16(image) => Ok(U16(image.debayer(alg)?)),
            F32(image) => Ok(F32(image.debayer(alg)?)),
        }
    }
}

impl<'a: 'b, 'b> DynamicImageData<'a> {
    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma(&'a self) -> Result<DynamicImageData<'a>, &'static str> {
        use DynamicImageData::*;
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
    pub fn into_luma_custom(
        &'a self,
        coeffs: &[f64],
    ) -> Result<DynamicImageData<'a>, &'static str> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(U8(image.into_luma_custom(coeffs)?)),
            U16(image) => Ok(U16(image.into_luma_custom(coeffs)?)),
            F32(image) => Ok(F32(image.into_luma_custom(coeffs)?)),
        }
    }
}

impl<'a> From<&DynamicImageData<'a>> for PixelType {
    fn from(data: &DynamicImageData<'a>) -> Self {
        match data {
            DynamicImageData::U8(_) => PixelType::U8,
            DynamicImageData::U16(_) => PixelType::U16,
            DynamicImageData::F32(_) => PixelType::F32,
        }
    }
}

macro_rules! tryfrom_dynimgdata_imgdata {
    ($type:ty, $variant:path) => {
        impl<'a> TryFrom<DynamicImageData<'a>> for ImageData<'a, $type> {
            type Error = &'static str;

            fn try_from(data: DynamicImageData<'a>) -> Result<Self, Self::Error> {
                match data {
                    $variant(data) => Ok(data),
                    _ => Err("Data is not of type u8"),
                }
            }
        }
    };
}

tryfrom_dynimgdata_imgdata!(u8, DynamicImageData::U8);
tryfrom_dynimgdata_imgdata!(u16, DynamicImageData::U16);
tryfrom_dynimgdata_imgdata!(f32, DynamicImageData::F32);

macro_rules! from_imgdata_dynimg {
    ($type:ty, $variant:path) => {
        impl<'a> From<ImageData<'a, $type>> for DynamicImageData<'a> {
            fn from(data: ImageData<'a, $type>) -> Self {
                $variant(data)
            }
        }
    };
}

from_imgdata_dynimg!(u8, DynamicImageData::U8);
from_imgdata_dynimg!(u16, DynamicImageData::U16);
from_imgdata_dynimg!(f32, DynamicImageData::F32);

impl<'a> DynamicImageData<'a> {
    /// Get the data as a slice of [`u8`], regardless of the underlying type.
    pub fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, ref image, { image.as_u8_slice() })
    }

    /// Get the data as a slice of [`u8`], regardless of the underlying type.
    pub fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, ref image, { image.as_u8_slice_checked() })
    }

    /// Get the data as a slice of [`u8`].
    pub fn as_slice_u8(&self) -> Option<&[u8]> {
        match self {
            DynamicImageData::U8(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of [`u8`].
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        match self {
            DynamicImageData::U8(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of [`u16`].
    pub fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageData::U16(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of [`u16`].
    pub fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        match self {
            DynamicImageData::U16(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of [`f32`].
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageData::F32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of [`f32`].
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        match self {
            DynamicImageData::F32(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Convert the image to a [`DynamicImageOwned`] with [`u8`] pixel type.
    /// 
    /// Note: This operation is parallelized if the `rayon` feature is enabled.
    pub fn into_u8(self) -> DynamicImageOwned {
        use DynamicImageData::*;
        match self {
            U8(data) => DynamicImageOwned::U8(data.into()),
            U16(data) => DynamicImageOwned::U8(data.into_u8()),
            F32(data) => DynamicImageOwned::U8(data.into_u8()),
        }
    }
}
