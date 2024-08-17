//! Image interop
pub use image::{DynamicImage, ImageBuffer};

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

mod test {

    #[test]
    fn test_dynamicimagedata() {
        use super::DynamicImageData;
        use crate::{ColorSpace, ImageData};
        use image::DynamicImage;
        let mut data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let ds = crate::DataStor::from_mut_ref(data.as_mut_slice());
        let a = ImageData::new(ds, 3, 2, ColorSpace::Gray).expect("Failed to create ImageData");
        let b = DynamicImageData::from(a.clone());
        let c = DynamicImage::try_from(b).unwrap();
        let c_ = c.resize(128, 128, image::imageops::FilterType::Nearest);
        let _d: DynamicImageData = c_
            .try_into()
            .expect("Failed to convert DynamicImage to DynamicImageData");
        assert_eq!(_d.width(), 128);
    }
}
