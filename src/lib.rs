#![deny(missing_docs)]
// #![deny(exported_private_dependencies)]
//! Crate to contain image data with a reference to a backing storage.

mod traits;
#[macro_use]
mod demosaic;
mod datastor;
mod imagedata;
#[macro_use]
mod dynamicimagedata;
#[cfg(feature = "image")]
pub mod dynamicimage_interop;
#[cfg(feature = "serialize")]
mod dynamicimagedata_serde;

pub(crate) use datastor::DataStor;
use demosaic::ColorFilterArray;
pub use demosaic::{BayerError, Demosaic};
pub use traits::{Lerp, Primitive};

#[cfg(feature = "image")]
pub use image::DynamicImage; // Used for image interop

#[cfg(feature = "serde")]
pub use serde::{Deserializer, Serializer};

/// Concrete type to hold image data.
#[derive(Debug, PartialEq, Clone)]
pub struct ImageData<'a, T: Primitive> {
    data: DataStor<'a, T>,
    width: u16,
    height: u16,
    channels: u8,
    cspace: ColorSpace,
}

/// Holds image data with a generic primitive type.
#[derive(Debug, PartialEq, Clone)]
pub enum DynamicImageData<'a> {
    /// Image data with a `u8` primitive type.
    U8(ImageData<'a, u8>),
    /// Image data with a `u16` primitive type.
    U16(ImageData<'a, u16>),
    /// Image data with a `f32` primitive type.
    F32(ImageData<'a, f32>),
}

/// Enum to describe the color space of the image.
#[repr(u8)]
#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy, PartialOrd)]
pub enum ColorSpace {
    /// Grayscale image.
    Gray = 0xa0,
    /// Bayer mosaic BGGR.
    Bggr = 0xa1,
    /// Bayer mosaic GBRG.
    Gbrg = 0xa2,
    /// Bayer mosaic GRBG.
    Grbg = 0xa3,
    /// Bayer mosaic RGGB.
    Rggb = 0xa4,
    /// RGB image.
    Rgb = 0xb0,
}

impl TryFrom<u8> for ColorSpace {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0xa0 => Ok(Self::Gray),
            0xa1 => Ok(Self::Bggr),
            0xa2 => Ok(Self::Gbrg),
            0xa3 => Ok(Self::Grbg),
            0xa4 => Ok(Self::Rggb),
            0xb0 => Ok(Self::Rgb),
            _ => Err("Invalid value for ColorSpace"),
        }
    }
}

/// Enum to describe the pixel type of the image.
/// The underlying `i8` representation conforms to the FITS standard.
#[repr(i8)]
#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PixelType {
    /// 8-bit unsigned integer.
    U8 = 8,
    /// 16-bit unsigned integer.
    U16 = 16,
    /// 32-bit floating point.
    F32 = -32,
}

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
    type Error = ();

    fn try_into(self) -> Result<ColorFilterArray, Self::Error> {
        match self {
            ColorSpace::Bggr => Ok(ColorFilterArray::Bggr),
            ColorSpace::Gbrg => Ok(ColorFilterArray::Gbrg),
            ColorSpace::Grbg => Ok(ColorFilterArray::Grbg),
            ColorSpace::Rggb => Ok(ColorFilterArray::Rggb),
            _ => Err(()),
        }
    }
}

mod test {
    #[test]
    fn test_datastor() {
        let mut data = vec![1, 2, 3, 4, 5];
        let ds = crate::DataStor::from_mut_ref(data.as_mut_slice());
        let _a = ds.to_owned();
    }

    #[test]
    fn test_debayer() {
        // color_backtrace::install();
        let src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
        ];
        let expected = [
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
            169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
        ];
        let img = crate::ImageData::new(
            crate::DataStor::from_owned(src.into()),
            4,
            4,
            crate::ColorSpace::Rggb,
        )
        .expect("Failed to create ImageData");
        let a = img.debayer(crate::Demosaic::None);
        assert!(a.is_ok());
        let a = a.unwrap(); // at this point, a is an ImageData struct
        assert!(a.channels() == 3);
        assert!(a.width() == 4);
        assert!(a.height() == 4);
        assert!(a.color_space() == crate::ColorSpace::Rgb);
        assert_eq!(a.as_slice(), &expected);
    }

    #[cfg(feature = "image")]
    #[test]
    fn test_dynamicimagedata() {
        use crate::{ColorSpace, DynamicImageData, ImageData};
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
