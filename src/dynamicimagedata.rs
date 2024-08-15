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

#[cfg(feature = "image")]
pub mod image_interop {
    use image::{DynamicImage, ImageBuffer};

    use crate::{ColorSpace, DataStor, DynamicImageData, ImageData};

    impl TryFrom<DynamicImage> for DynamicImageData<'_> {
        type Error = &'static str;

        fn try_from(data: DynamicImage) -> Result<Self, Self::Error> {
            let wid = data.width() as u16;
            let hei = data.height() as u16;
            match data {
                DynamicImage::ImageLuma8(data) => Ok(DynamicImageData::U8(
                    ImageData::new(
                        DataStor::from_owned(data.into_raw()),
                        wid,
                        hei,
                        ColorSpace::Gray,
                    )
                    .map_err(|_| "Could not create DynamicImageData from ImageLuma8")?,
                )),
                DynamicImage::ImageRgb8(data) => Ok(DynamicImageData::U8(
                    ImageData::new(
                        DataStor::from_owned(data.into_raw()),
                        wid,
                        hei,
                        ColorSpace::Rgb,
                    )
                    .map_err(|_| "Could not create DynamicImageData from ImageRgb8")?,
                )),
                DynamicImage::ImageLuma16(data) => Ok(DynamicImageData::U16(
                    ImageData::new(
                        DataStor::from_owned(data.into_raw()),
                        wid,
                        hei,
                        ColorSpace::Gray,
                    )
                    .map_err(|_| "Could not create DynamicImageData from ImageLuma16")?,
                )),
                DynamicImage::ImageRgb16(data) => Ok(DynamicImageData::U16(
                    ImageData::new(
                        DataStor::from_owned(data.into_raw()),
                        wid,
                        hei,
                        ColorSpace::Rgb,
                    )
                    .map_err(|_| "Could not create DynamicImageData from ImageRgb16")?,
                )),
                DynamicImage::ImageRgb32F(data) => Ok(DynamicImageData::F32(
                    ImageData::new(
                        DataStor::from_owned(data.into_raw()),
                        wid,
                        hei,
                        ColorSpace::Rgb,
                    )
                    .map_err(|_| "Could not create DynamicImageData from ImageRgb32F")?,
                )),
                _ => Err("Alpha channel not supported"),
            }
        }
    }

    impl<'a> TryFrom<DynamicImageData<'a>> for DynamicImage {
        type Error = &'static str;

        fn try_from(value: DynamicImageData<'a>) -> Result<Self, Self::Error> {
            use DynamicImageData::*;
            let width = value.width() as u32;
            let height = value.height() as u32;
            let cspace = value.color_space();
            let channels = value.channels();
            if channels > 4 {
                return Err("Too many channels");
            }
            match cspace {
                ColorSpace::Gray => match value {
                    U8(data) => Ok(DynamicImage::ImageLuma8(
                        ImageBuffer::from_vec(width, height, data.into_vec())
                            .ok_or("Could not create Gray8 image")?,
                    )),
                    U16(data) => Ok(DynamicImage::ImageLuma16(
                        ImageBuffer::from_vec(width, height, data.into_vec())
                            .ok_or("Could not create Gray16 image")?,
                    )),
                    F32(_) => Err("Gray32F not supported"),
                },
                ColorSpace::Rgb => match value {
                    U8(data) => Ok(DynamicImage::ImageRgb8(
                        ImageBuffer::from_vec(width, height, data.into_vec())
                            .ok_or("Could not create Rgb8 image")?,
                    )),
                    U16(data) => Ok(DynamicImage::ImageRgb16(
                        ImageBuffer::from_vec(width, height, data.into_vec())
                            .ok_or("Could not create Rgb16 image")?,
                    )),
                    F32(data) => Ok(DynamicImage::ImageRgb32F(
                        ImageBuffer::from_vec(width, height, data.into_vec())
                            .ok_or("Could not create Rgb32F image")?,
                    )),
                },
                _ => Err("Unsupported color space"),
            }
        }
    }
}

#[cfg(feature = "image")]
mod test {
    use image::DynamicImage;

    use crate::{ColorSpace, ImageData};

    use super::DynamicImageData;

    #[test]
    fn test_dynamicimagedata() {
        let mut data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let ds = crate::DataStor::from_mut_ref(data.as_mut_slice());
        let a = ImageData::new(ds, 3, 2, ColorSpace::Gray).expect("Failed to create ImageData");
        let b = DynamicImageData::from(a.clone());
        let c: DynamicImage = DynamicImage::try_from(b).unwrap();
        let c_ = c.resize(128, 128, image::imageops::FilterType::Nearest);
        let _d: DynamicImageData = c_
            .try_into()
            .expect("Failed to convert DynamicImage to DynamicImageData");
        assert_eq!(_d.width(), 128);
    }
}
