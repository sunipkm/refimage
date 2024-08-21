#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// #![deny(exported_private_dependencies)]

//! Crate to handle image data backed either by a contiguous slice or a vector.
//! 
//! The image data is stored in a row-major order and can be of different pixel 
//! types - `u8`, `u16`, and `f32`. The image data supports arbitrary color spaces
//! and number of channels, but the number of channels must be consistent across the
//! image. The image size is limited to 65535 x 65535 pixels. In case the image is a
//! mosaic image, the crate supports debayering of the image data.
//! 
//! The crate additionally supports serialization and deserialization of the image
//! data using the `serde` framework. The crate, by default, compiles with the [`flate2`]
//! crate to compress the data before serialization. The compression can be disabled
//! by setting the `serde_flate` feature to `false`.
//! 
//! The crate provides a concrete type [`ImageData`] to store image data and a type-erased
//! version [`DynamicImageData`] to store image data with different pixel types. 
//! Additionally, the crate provides a [`GenericImage`] type to store a [`DynamicImageData`]
//! with additional metadata, such as the image creation timestamp, and many more. The 
//! metadata keys must be 80 characters or less. Uniqueness of the keys is not enforced,
//! but is strongly recommended; the keys are case-insensitive.
//! 
//! The crate, with the optional `image` feature, provides can convert between 
//! [`DynamicImageData`] and [`DynamicImage`] from the [`image`] crate.
//! With the optional `fitsio` feature, the crate can write a [`GenericImage`], with
//! all associated metadata, to a [FITS](https://fits.gsfc.nasa.gov/fits_primer.html) file.
//! 
//! # Usage
//! ```
//! use refimage::{ImageData, ColorSpace, DynamicImageData, GenericImage};
//! use std::time::SystemTime;
//! use std::path::Path;
//! 
//! let data = vec![1u8, 2, 3, 4, 5, 6]; // 3x2 grayscale image
//! let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap(); // Create ImageData
//! let img = DynamicImageData::from(img); // Convert to DynamicImageData
//! let mut img = GenericImage::new(SystemTime::now(), img); // Create GenericImage with creation time info
//! img.insert_key("CAMERANAME", "Canon EOS 5D Mark IV".to_string()).unwrap(); // Insert metadata
//! let serialized = bincode::serialize(&img).unwrap(); // Serialize the image
//! let deserialized: GenericImage = bincode::deserialize(&serialized).unwrap(); // Deserialize the image
//! ```
//! 
//! 

mod traits;
#[macro_use]
mod demosaic;
mod datastor;
mod imagedata;
#[macro_use]
mod dynamicimagedata;
#[cfg(feature = "image")]
mod dynamicimage_interop;
mod dynamicimagedata_serde;
#[cfg(feature = "fitsio")]
mod fitsio_interop;
#[cfg(feature = "fitsio")]
#[cfg_attr(docsrs, doc(cfg(feature = "fitsio")))]
pub use fitsio_interop::FitsCompression;

mod metadata;
pub use metadata::{GenericImage, GenericLineItem, CAMERANAME_KEY, PROGRAMNAME_KEY};

pub(crate) use datastor::DataStor;
use demosaic::ColorFilterArray;
pub use demosaic::{BayerError, Demosaic};
pub use traits::PixelStor;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use image::DynamicImage; // Used for image interop

pub use serde::{Deserializer, Serializer};

pub use imagedata::ImageData;

/// Image data with a dynamic pixel type.
/// 
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
/// More variants that adhere to these principles may get added in the future, in
/// particular to cover other combinations typically used. The data is stored in a single
/// contiguous buffer, which is either backed by a slice or a vector, and aims to enable
/// reuse of allocated memory without re-allocation.
/// 
/// # Note
/// Alpha channels are not trivially supported. They can be added by using a custom
/// color space.
///
/// # Usage
/// 
/// ```
/// use refimage::{ImageData, ColorSpace, DynamicImageData};
/// 
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// let img = DynamicImageData::from(img);
/// 
/// ```
///
/// This type acts as a type-erased version of `ImageData` and can be used to store
/// image data with different pixel types. The pixel type is determined at runtime.
#[derive(Debug, PartialEq, Clone)]
#[non_exhaustive]
pub enum DynamicImageData<'a> {
    /// Image data with a `u8` primitive type.
    U8(ImageData<'a, u8>),
    /// Image data with a `u16` primitive type.
    U16(ImageData<'a, u16>),
    /// Image data with a `f32` primitive type.
    F32(ImageData<'a, f32>),
}

/// Description of the color space of the image.
/// 
/// The colorspace information is used to enable debayering of the image data, and
/// for interpretation of single or multi-channel images.
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
    /// Custom color space.
    Custom = 0xff,
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
            0xff => Ok(Self::Custom),
            _ => Err("Invalid value for ColorSpace"),
        }
    }
}

/// Enum to describe the primitive pixel type of the image.
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

// Can't use the macro-call itself within the `doc` attribute. So force it to eval it as part of
// the macro invocation.
//
// The inspiration for the macro and implementation is from
// <https://github.com/GuillaumeGomez/doc-comment>
//
// MIT License
//
// Copyright (c) 2018 Guillaume Gomez
macro_rules! insert_as_doc {
    { $content:expr } => {
        #[allow(unused_doc_comments)]
        #[doc = $content] extern { }
    }
}

// Provides the README.md as doc, to ensure the example works!
insert_as_doc!(include_str!("../README.MD"));