use crate::{
    AlphaChannel, BayerError, ColorSpace, DemosaicMethod, DynamicImageOwned, ImageOwned,
    ImageProps, PixelType, ToLuma,
};
use crate::{Debayer, DynamicImageRef};

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

impl ImageProps for DynamicImageOwned {
    type OutputU8 = DynamicImageOwned;

    fn width(&self) -> usize {
        dynamic_map!(self, ref image, { image.width() })
    }

    fn height(&self) -> usize {
        dynamic_map!(self, ref image, { image.height() })
    }

    fn channels(&self) -> u8 {
        dynamic_map!(self, ref image, { image.channels() })
    }

    fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, ref image, { image.color_space() })
    }

    fn pixel_type(&self) -> PixelType {
        dynamic_map!(self, ref image, { image.pixel_type() })
    }

    fn into_u8(&self) -> Self::OutputU8 {
        match self {
            DynamicImageOwned::U8(_) => self.clone(),
            DynamicImageOwned::U16(data) => DynamicImageOwned::U8(data.into_u8()),
            DynamicImageOwned::F32(data) => DynamicImageOwned::U8(data.into_u8()),
        }
    }

    fn len(&self) -> usize {
        dynamic_map!(self, ref image, { image.len() })
    }

    fn is_empty(&self) -> bool {
        dynamic_map!(self, ref image, { image.is_empty() })
    }
}

impl<'a: 'b, 'b> Debayer<'a, 'b> for DynamicImageOwned {
    type Output = DynamicImageOwned;
    fn debayer(&self, alg: DemosaicMethod) -> Result<Self::Output, BayerError> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.debayer(alg)?)),
            U16(image) => Ok(U16(image.debayer(alg)?)),
            F32(image) => Ok(F32(image.debayer(alg)?)),
        }
    }
}

impl<'a: 'b, 'b, T> ToLuma<'a, 'b, T> for DynamicImageOwned {
    type Output = DynamicImageOwned;

    fn to_luma(&self) -> Result<Self::Output, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.to_luma()?)),
            U16(image) => Ok(U16(image.to_luma()?)),
            F32(image) => Ok(F32(image.to_luma()?)),
        }
    }

    fn to_luma_alpha(&self) -> Result<Self::Output, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.to_luma_alpha()?)),
            U16(image) => Ok(U16(image.to_luma_alpha()?)),
            F32(image) => Ok(F32(image.to_luma_alpha()?)),
        }
    }

    fn to_luma_custom(&self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.to_luma_custom(coeffs)?)),
            U16(image) => Ok(U16(image.to_luma_custom(coeffs)?)),
            F32(image) => Ok(F32(image.to_luma_custom(coeffs)?)),
        }
    }

    fn to_luma_alpha_custom(&self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        use DynamicImageOwned::*;
        match self {
            U8(image) => Ok(U8(image.to_luma_alpha_custom(coeffs)?)),
            U16(image) => Ok(U16(image.to_luma_alpha_custom(coeffs)?)),
            F32(image) => Ok(F32(image.to_luma_alpha_custom(coeffs)?)),
        }
    }
}

macro_rules! impl_alphachannel {
    ($type:ty, $intype:ty, $variant:path) => {
        impl<'a: 'b, 'b> AlphaChannel<'a, 'b, $type, $intype> for DynamicImageOwned {
            type ImageOutput = DynamicImageOwned;
            type AlphaOutput = Vec<$type>;

            fn add_alpha(&self, alpha: $intype) -> Result<Self::ImageOutput, &'static str> {
                use DynamicImageOwned::*;
                match self {
                    $variant(image) => {
                        let image = image.add_alpha(alpha)?;
                        Ok($variant(image))
                    }
                    _ => Err("Data is not of type u8"),
                }
            }

            fn remove_alpha(&self) -> Result<(Self::ImageOutput, Self::AlphaOutput), &'static str> {
                use DynamicImageOwned::*;
                match self {
                    $variant(image) => {
                        let (image, alpha) = image.remove_alpha()?;
                        Ok(($variant(image), alpha))
                    }
                    _ => Err("Data is not of type u8"),
                }
            }
        }
    };
}

impl_alphachannel!(u8, &[u8], U8);
impl_alphachannel!(u16, &[u16], U16);
impl_alphachannel!(f32, &[f32], F32);

impl DynamicImageOwned {
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

impl<'a> From<&DynamicImageRef<'_>> for DynamicImageOwned {
    fn from(data: &DynamicImageRef<'_>) -> Self {
        match data {
            DynamicImageRef::U8(data) => DynamicImageOwned::U8(data.into()),
            DynamicImageRef::U16(data) => DynamicImageOwned::U16(data.into()),
            DynamicImageRef::F32(data) => DynamicImageOwned::F32(data.into()),
        }
    }
}
