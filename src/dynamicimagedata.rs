use crate::{
    AlphaChannel, BayerError, ColorSpace, DemosaicMethod, DynamicImageData, ImageData, PixelType,
    ToLuma,
};
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
    type Output = DynamicImageOwned;
    fn debayer(&'b self, alg: DemosaicMethod) -> Result<Self::Output, BayerError> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(DynamicImageOwned::U8(image.debayer(alg)?)),
            U16(image) => Ok(DynamicImageOwned::U16(image.debayer(alg)?)),
            F32(image) => Ok(DynamicImageOwned::F32(image.debayer(alg)?)),
        }
    }
}

impl<'a: 'b, 'b, T> ToLuma<'a, 'b, T> for DynamicImageData<'a> {
    type Output = DynamicImageOwned;

    fn to_luma(&'b self) -> Result<Self::Output, &'static str> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(DynamicImageOwned::U8(image.to_luma()?)),
            U16(image) => Ok(DynamicImageOwned::U16(image.to_luma()?)),
            F32(image) => Ok(DynamicImageOwned::F32(image.to_luma()?)),
        }
    }

    fn to_luma_alpha(&'b self) -> Result<Self::Output, &'static str> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(DynamicImageOwned::U8(image.to_luma_alpha()?)),
            U16(image) => Ok(DynamicImageOwned::U16(image.to_luma_alpha()?)),
            F32(image) => Ok(DynamicImageOwned::F32(image.to_luma_alpha()?)),
        }
    }

    fn to_luma_custom(&'b self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(DynamicImageOwned::U8(image.to_luma_custom(coeffs)?)),
            U16(image) => Ok(DynamicImageOwned::U16(image.to_luma_custom(coeffs)?)),
            F32(image) => Ok(DynamicImageOwned::F32(image.to_luma_custom(coeffs)?)),
        }
    }

    fn to_luma_alpha_custom(&'b self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(DynamicImageOwned::U8(image.to_luma_alpha_custom(coeffs)?)),
            U16(image) => Ok(DynamicImageOwned::U16(image.to_luma_alpha_custom(coeffs)?)),
            F32(image) => Ok(DynamicImageOwned::F32(image.to_luma_alpha_custom(coeffs)?)),
        }
    }
}

macro_rules! impl_alphachannel {
    ($type:ty, $intype:ty, $variant_a:path, $variant_b:expr) => {
        impl<'a: 'b, 'b> AlphaChannel<'a, 'b, $type, $intype> for DynamicImageData<'a> {
            type ImageOutput = DynamicImageOwned;
            type AlphaOutput = Vec<$type>;

            fn add_alpha(&'a self, alpha: $intype) -> Result<Self::ImageOutput, &'static str> {
                use DynamicImageData::*;
                match self {
                    $variant_a(image) => {
                        let image = image.add_alpha(alpha)?;
                        Ok($variant_b(image))
                    }
                    _ => Err("Data is not of type u8"),
                }
            }

            fn remove_alpha(
                &'a self,
            ) -> Result<(Self::ImageOutput, Self::AlphaOutput), &'static str> {
                use DynamicImageData::*;
                match self {
                    $variant_a(image) => {
                        let (image, alpha) = image.remove_alpha()?;
                        Ok(($variant_b(image), alpha))
                    }
                    _ => Err("Data is not of type u8"),
                }
            }
        }
    };
}

impl_alphachannel!(u8, &[u8], U8, DynamicImageOwned::U8);
impl_alphachannel!(u16, &[u16], U16, DynamicImageOwned::U16);
impl_alphachannel!(f32, &[f32], F32, DynamicImageOwned::F32);

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
    pub fn into_u8(&self) -> DynamicImageOwned {
        use DynamicImageData::*;
        match self {
            U8(data) => DynamicImageOwned::U8(data.into()),
            U16(data) => DynamicImageOwned::U8(data.into_u8()),
            F32(data) => DynamicImageOwned::U8(data.into_u8()),
        }
    }
}
