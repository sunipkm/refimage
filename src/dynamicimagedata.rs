use crate::{BayerError, ColorSpace, Demosaic, DynamicImageData, ImageData, PixelType};

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

    /// Debayer the image.
    pub fn debayer(&self, alg: Demosaic) -> Result<DynamicImageData, BayerError> {
        use DynamicImageData::*;
        match self {
            U8(image) => Ok(U8(image.debayer(alg)?)),
            U16(image) => Ok(U16(image.debayer(alg)?)),
            F32(image) => Ok(F32(image.debayer(alg)?)),
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

impl<'a> From<ImageData<'a, u8>> for DynamicImageData<'a> {
    fn from(data: ImageData<'a, u8>) -> Self {
        DynamicImageData::U8(data)
    }
}

impl<'a> TryFrom<DynamicImageData<'a>> for ImageData<'a, u8> {
    type Error = &'static str;

    fn try_from(data: DynamicImageData<'a>) -> Result<Self, Self::Error> {
        match data {
            DynamicImageData::U8(data) => Ok(data),
            _ => Err("Data is not of type u8"),
        }
    }
}

impl<'a> TryFrom<DynamicImageData<'a>> for ImageData<'a, u16> {
    type Error = &'static str;

    fn try_from(data: DynamicImageData<'a>) -> Result<Self, Self::Error> {
        match data {
            DynamicImageData::U16(data) => Ok(data),
            _ => Err("Data is not of type u16"),
        }
    }
}

impl<'a> TryFrom<DynamicImageData<'a>> for ImageData<'a, f32> {
    type Error = &'static str;

    fn try_from(data: DynamicImageData<'a>) -> Result<Self, Self::Error> {
        match data {
            DynamicImageData::F32(data) => Ok(data),
            _ => Err("Data is not of type f32"),
        }
    }
}

impl<'a> From<ImageData<'a, u16>> for DynamicImageData<'a> {
    fn from(data: ImageData<'a, u16>) -> Self {
        DynamicImageData::U16(data)
    }
}

impl<'a> From<ImageData<'a, f32>> for DynamicImageData<'a> {
    fn from(data: ImageData<'a, f32>) -> Self {
        DynamicImageData::F32(data)
    }
}

impl<'a> DynamicImageData<'a> {
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
            DynamicImageData::U8(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `u8`.
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        match self {
            DynamicImageData::U8(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of `u16`.
    pub fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageData::U16(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `u16`.
    pub fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        match self {
            DynamicImageData::U16(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    /// Get the data as a slice of `f32`.
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageData::F32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    /// Get the data as a mutable slice of `f32`.
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        match self {
            DynamicImageData::F32(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }
}
