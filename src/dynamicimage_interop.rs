//! Image interop
use image::ImageBuffer;

use crate::{
    ColorSpace, DynamicImage, DynamicImageRef, GenericImage, GenericImageOwned, GenericImageRef,
    ImageProps,
};

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl<'a> TryFrom<DynamicImageRef<'a>> for DynamicImage {
    type Error = &'static str;

    fn try_from(value: DynamicImageRef<'a>) -> Result<Self, Self::Error> {
        use DynamicImageRef::*;
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

use crate::{DynamicImageOwned, ImageOwned};

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl TryFrom<DynamicImage> for DynamicImageOwned {
    type Error = &'static str;

    fn try_from(data: DynamicImage) -> Result<Self, Self::Error> {
        let wid = data.width() as u16;
        let hei = data.height() as u16;
        match data {
            DynamicImage::ImageLuma8(data) => Ok(DynamicImageOwned::U8(
                ImageOwned::new(data.into_raw(), wid.into(), hei.into(), ColorSpace::Gray)
                    .map_err(|_| "Could not create DynamicImageOwned from ImageLuma8")?,
            )),
            DynamicImage::ImageRgb8(data) => Ok(DynamicImageOwned::U8(
                ImageOwned::new(data.into_raw(), wid.into(), hei.into(), ColorSpace::Rgb)
                    .map_err(|_| "Could not create DynamicImageOwned from ImageRgb8")?,
            )),
            DynamicImage::ImageLuma16(data) => Ok(DynamicImageOwned::U16(
                ImageOwned::new(data.into_raw(), wid.into(), hei.into(), ColorSpace::Gray)
                    .map_err(|_| "Could not create DynamicImageOwned from ImageLuma16")?,
            )),
            DynamicImage::ImageRgb16(data) => Ok(DynamicImageOwned::U16(
                ImageOwned::new(data.into_raw(), wid.into(), hei.into(), ColorSpace::Rgb)
                    .map_err(|_| "Could not create DynamicImageOwned from ImageRgb16")?,
            )),
            DynamicImage::ImageRgb32F(data) => Ok(DynamicImageOwned::F32(
                ImageOwned::new(data.into_raw(), wid.into(), hei.into(), ColorSpace::Rgb)
                    .map_err(|_| "Could not create DynamicImageOwned from ImageRgb32F")?,
            )),
            _ => Err("Unknown image type"),
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl TryFrom<DynamicImageOwned> for DynamicImage {
    type Error = &'static str;

    fn try_from(value: DynamicImageOwned) -> Result<Self, Self::Error> {
        use DynamicImageOwned::*;
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

impl TryFrom<GenericImageOwned> for DynamicImage {
    type Error = &'static str;

    fn try_from(value: GenericImageOwned) -> Result<Self, Self::Error> {
        value.image.try_into()
    }
}

impl TryFrom<GenericImageRef<'_>> for DynamicImage {
    type Error = &'static str;

    fn try_from(value: GenericImageRef<'_>) -> Result<Self, Self::Error> {
        value.image.try_into()
    }
}

impl TryFrom<GenericImage<'_>> for DynamicImage {
    type Error = &'static str;

    fn try_from(value: GenericImage<'_>) -> Result<Self, Self::Error> {
        match value {
            GenericImage::Own(data) => data.try_into(),
            GenericImage::Ref(data) => data.try_into(),
        }
    }
}

mod test {

    #[test]
    fn test_dynamicimagedata() {
        use super::DynamicImageRef;
        use crate::{ColorSpace, ImageRef};
        use image::DynamicImage;
        let mut data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let a =
            ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).expect("Failed to create ImageRef");
        let b = DynamicImageRef::from(a);
        let c = DynamicImage::try_from(b).unwrap();
        assert_eq!(c.width(), 3);
    }

    #[test]
    fn test_dynamicimageowned() {
        use super::DynamicImageOwned;
        use crate::ImageProps;
        use crate::{ColorSpace, ImageOwned};
        use image::DynamicImage;
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let a = ImageOwned::new(data, 3, 2, ColorSpace::Gray).expect("Failed to create ImageRef");
        let b = DynamicImageOwned::from(a.clone());
        let c = DynamicImage::try_from(b).unwrap();
        let c_ = c.resize(128, 128, image::imageops::FilterType::Nearest);
        let _d: DynamicImageOwned = c_
            .try_into()
            .expect("Failed to convert DynamicImage to DynamicImageOwned");
        assert_eq!(_d.width(), 128);
    }
}
