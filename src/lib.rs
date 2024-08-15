#![deny(missing_docs)]
//! Crate to contain image data with a reference to a backing storage.

mod traits;
#[macro_use]
mod demosaic;
mod datastor;
mod imagedata;
#[macro_use]
mod dynamicimagedata;

use demosaic::ColorFilterArray;
pub use demosaic::{BayerError, Demosaic};
pub use traits::{Lerp, Primitive};
pub(crate) use datastor::DataStor;

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
    Gray,
    /// Bayer mosaic BGGR.
    Bggr,
    /// Bayer mosaic GBRG.
    Gbrg,
    /// Bayer mosaic GRBG.
    Grbg,
    /// Bayer mosaic RGGB.
    Rggb,
    /// RGB image.
    Rgb,
}

/// Enum to describe the pixel type of the image.
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PixelType {
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit floating point.
    F32,
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
        ).expect("Failed to create ImageData");
        let a = img.debayer(crate::Demosaic::None);
        assert!(a.is_ok());
        let a = a.unwrap(); // at this point, a is an ImageData struct
        assert!(a.channels() == 3);
        assert!(a.width() == 4);
        assert!(a.height() == 4);
        assert!(a.color_space() == crate::ColorSpace::Rgb);
        assert_eq!(a.as_slice(), &expected);
    }
}
