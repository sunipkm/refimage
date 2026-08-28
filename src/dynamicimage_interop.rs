//! Image interop
use image::ImageBuffer;
use thiserror::Error;

use crate::{
    ColorSpace, DynamicImage, DynamicImageRef, GenericImage, GenericImageOwned, GenericImageRef,
    ImageError, ImageProps,
};

/// Errors from converting between [`image::DynamicImage`] and this crate's types.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum InteropError {
    /// The image has more than 4 channels.
    #[error("too many channels: {0}")]
    TooManyChannels(u8),
    /// `Gray32F` has no representation in this crate.
    #[error("Gray32F images are not supported")]
    Gray32FUnsupported,
    /// The color space cannot be mapped to an [`image`] buffer type.
    #[error("unsupported color space: {0:?}")]
    UnsupportedColorSpace(ColorSpace),
    /// The [`image::DynamicImage`] variant has no representation in this crate.
    #[error("unsupported image::DynamicImage variant")]
    UnknownImageType,
    /// The pixel buffer could not be wrapped in an [`image`] buffer (bad length).
    #[error("could not build image buffer")]
    BadBuffer,
    /// Rebuilding this crate's image from the decoded buffer failed.
    #[error(transparent)]
    Image(#[from] ImageError),
}

/// `Result` alias for [`InteropError`].
pub type InteropResult<T> = Result<T, InteropError>;

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl<'a> TryFrom<DynamicImageRef<'a>> for DynamicImage {
    type Error = InteropError;

    fn try_from(value: DynamicImageRef<'a>) -> Result<Self, Self::Error> {
        use DynamicImageRef::*;
        let width = value.width() as u32;
        let height = value.height() as u32;
        let cspace = value.color_space();
        let channels = value.channels();
        if channels > 4 {
            return Err(InteropError::TooManyChannels(channels));
        }
        match cspace {
            ColorSpace::Gray => match value {
                U8(data) => Ok(DynamicImage::ImageLuma8(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                U16(data) => Ok(DynamicImage::ImageLuma16(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                F32(_) => Err(InteropError::Gray32FUnsupported),
            },
            ColorSpace::Rgb => match value {
                U8(data) => Ok(DynamicImage::ImageRgb8(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                U16(data) => Ok(DynamicImage::ImageRgb16(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                F32(data) => Ok(DynamicImage::ImageRgb32F(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
            },
            _ => Err(InteropError::UnsupportedColorSpace(cspace)),
        }
    }
}

use crate::{DynamicImageOwned, ImageOwned};

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl TryFrom<DynamicImage> for DynamicImageOwned {
    type Error = InteropError;

    fn try_from(data: DynamicImage) -> Result<Self, Self::Error> {
        let wid = data.width() as u16;
        let hei = data.height() as u16;
        match data {
            DynamicImage::ImageLuma8(data) => Ok(DynamicImageOwned::U8(ImageOwned::new(
                data.into_raw(),
                wid.into(),
                hei.into(),
                ColorSpace::Gray,
            )?)),
            DynamicImage::ImageRgb8(data) => Ok(DynamicImageOwned::U8(ImageOwned::new(
                data.into_raw(),
                wid.into(),
                hei.into(),
                ColorSpace::Rgb,
            )?)),
            DynamicImage::ImageLuma16(data) => Ok(DynamicImageOwned::U16(ImageOwned::new(
                data.into_raw(),
                wid.into(),
                hei.into(),
                ColorSpace::Gray,
            )?)),
            DynamicImage::ImageRgb16(data) => Ok(DynamicImageOwned::U16(ImageOwned::new(
                data.into_raw(),
                wid.into(),
                hei.into(),
                ColorSpace::Rgb,
            )?)),
            DynamicImage::ImageRgb32F(data) => Ok(DynamicImageOwned::F32(ImageOwned::new(
                data.into_raw(),
                wid.into(),
                hei.into(),
                ColorSpace::Rgb,
            )?)),
            _ => Err(InteropError::UnknownImageType),
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl TryFrom<DynamicImageOwned> for DynamicImage {
    type Error = InteropError;

    fn try_from(value: DynamicImageOwned) -> Result<Self, Self::Error> {
        use DynamicImageOwned::*;
        let width = value.width() as u32;
        let height = value.height() as u32;
        let cspace = value.color_space();
        let channels = value.channels();
        if channels > 4 {
            return Err(InteropError::TooManyChannels(channels));
        }
        match cspace {
            ColorSpace::Gray => match value {
                U8(data) => Ok(DynamicImage::ImageLuma8(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                U16(data) => Ok(DynamicImage::ImageLuma16(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                F32(_) => Err(InteropError::Gray32FUnsupported),
            },
            ColorSpace::Rgb => match value {
                U8(data) => Ok(DynamicImage::ImageRgb8(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                U16(data) => Ok(DynamicImage::ImageRgb16(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
                F32(data) => Ok(DynamicImage::ImageRgb32F(
                    ImageBuffer::from_vec(width, height, data.into_vec())
                        .ok_or(InteropError::BadBuffer)?,
                )),
            },
            _ => Err(InteropError::UnsupportedColorSpace(cspace)),
        }
    }
}

impl TryFrom<GenericImageOwned> for DynamicImage {
    type Error = InteropError;

    fn try_from(value: GenericImageOwned) -> Result<Self, Self::Error> {
        value.image.try_into()
    }
}

impl TryFrom<GenericImageRef<'_>> for DynamicImage {
    type Error = InteropError;

    fn try_from(value: GenericImageRef<'_>) -> Result<Self, Self::Error> {
        value.image.try_into()
    }
}

impl TryFrom<GenericImage<'_>> for DynamicImage {
    type Error = InteropError;

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
