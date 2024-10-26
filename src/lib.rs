#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// #![deny(exported_private_dependencies)]

//! Crate to handle image data backed either by a contiguous slice or a vector.
//!
//! The image data is stored in a row-major order and can be of different pixel
//! types - `u8`, `u16`, and `f32`. The image data supports arbitrary color spaces
//! and number of channels, but the number of channels must be consistent with the
//! length of the backing storage.
//! The image size is limited to 65535 x 65535 pixels. In case the image is a
//! Bayer mosaic image, the crate supports debayering of the image data.
//!
//! The crate additionally supports serialization and deserialization of the image
//! data using the `serde` framework.
//!
//! The crate provides a concrete type [`ImageRef`] to store image data and a type-erased
//! version [`DynamicImageRef`] to store image data with different pixel types.
//! Additionally, the crate provides a [`GenericImageRef`] type to store a [`DynamicImageRef`]
//! with additional metadata, such as the image creation timestamp, and many more. The
//! metadata keys must be 80 characters or less. Uniqueness of the keys is not enforced,
//! but is strongly recommended; the keys are case-insensitive.
//!
//! The crate, with the optional `image` feature, provides can convert between
//! [`DynamicImageRef`] and [`DynamicImage`] from the [`image`] crate.
//! With the optional `fitsio` feature, the crate can write a [`GenericImageRef`], with
//! all associated metadata, to a [FITS](https://fits.gsfc.nasa.gov/fits_primer.html) file.
//!
//! # Usage
//! ```
//! use refimage::{ImageRef, ColorSpace, DynamicImageRef, GenericImageRef, GenericImageOwned};
//! use std::time::SystemTime;
//! use std::path::Path;
//!
//! let mut data = vec![1u8, 2, 3, 4, 5, 6, 0, 0]; // 3x2 grayscale image, with extra padding that will be ignored
//! let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap(); // Create ImageRef
//! let img = DynamicImageRef::from(img); // Convert to DynamicImageRef
//! let mut img = GenericImageRef::new(SystemTime::now(), img); // Create GenericImageRef with creation time info
//! img.insert_key("CAMERANAME", "Canon EOS 5D Mark IV".to_string()).unwrap(); // Insert metadata
//! let serialized = bincode::serialize(&img).unwrap(); // Serialize the image
//! let deserialized: GenericImageOwned = bincode::deserialize(&serialized).unwrap(); // Deserialize the image
//! ```
//! # Optional Features
//! Features are available to extend the functionalities of the core `refimage` data types:
//! - `rayon`: Parallelizes [`GenericImageRef::to_luma`] (and similar), [`GenericImageRef::to_luma_custom`], [`GenericImageRef::into_u8`] and [`GenericImageRef::debayer`] functions (<b>enabled</b> by default).
//! - `fitsio`: Exposes [`FitsWrite`] trait to write [`GenericImageRef`] and [`GenericImageOwned`] (<b>disabled</b> by default).
//! - `image`: Enables [`TryFrom`] conversions between [`DynamicImage`] and [`DynamicImageRef`], [`DynamicImageOwned`] (<b>disabled</b> by default).
//!

mod coreimpls;
mod coretraits;
mod imagetraits;
#[macro_use]
mod demosaic;
mod imageowned;
mod imageref;
#[macro_use]
mod dynamicimageref;
#[macro_use]
mod dynamicimageowned;
#[cfg(feature = "image")]
mod dynamicimage_interop;
mod dynamicimage_serde;
#[cfg(feature = "fitsio")]
mod fitsio_interop;
mod genericimage;
mod genericimageowned;
mod genericimageref;
#[cfg(feature = "fitsio")]
#[cfg_attr(docsrs, doc(cfg(feature = "fitsio")))]
pub use fitsio_interop::{FitsCompression, FitsError, FitsWrite};

pub use genericimageowned::GenericImageOwned;
pub use genericimageref::GenericImageRef;

mod metadata;
pub use metadata::{
    GenericLineItem, GenericValue, CAMERANAME_KEY, EXPOSURE_KEY, PROGRAMNAME_KEY, TIMESTAMP_KEY,
};

pub use coretraits::{Enlargeable, PixelStor};
pub use demosaic::{BayerError, Debayer, DemosaicMethod};
pub use genericimage::GenericImage;
pub use imagetraits::{BayerShift, ImageProps, ToLuma};
use serde::{Deserialize, Serialize};

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use image::DynamicImage; // Used for image interop

pub use serde::{Deserializer, Serializer};

pub use imageowned::ImageOwned;
pub use imageref::ImageRef;

mod optimumexposure;
pub use optimumexposure::{CalcOptExp, OptimumExposure, OptimumExposureBuilder};

/// Image data with a dynamic pixel type, backed by a mutable slice of data.
///
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
/// More variants that adhere to these principles may get added in the future, in
/// particular to cover other combinations typically used. The data is stored in a single
/// contiguous buffer, which is backed by a mutable slice, and aims to enable
/// reuse of allocated memory without re-allocation.
///
/// # Note
/// - Does not support alpha channel natively.
/// - Internally [`DynamicImageRef`] and [`DynamicImageOwned`] serialize to the same
///   representation, and [`DynamicImageRef`] can be deserialized into [`DynamicImageOwned`] only.
///
/// # Usage
///
/// ```
/// use refimage::{ImageRef, ColorSpace, DynamicImageRef};
///
/// let mut data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
/// let img = DynamicImageRef::from(img);
///
/// ```
///
/// This type acts as a type-erased version of `ImageRef` and can be used to store
/// image data with different pixel types. The pixel type is determined at runtime.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum DynamicImageRef<'a> {
    /// Image data with a `u8` primitive type.
    U8(ImageRef<'a, u8>),
    /// Image data with a `u16` primitive type.
    U16(ImageRef<'a, u16>),
    /// Image data with a `f32` primitive type.
    F32(ImageRef<'a, f32>),
}

/// Image data with a dynamic pixel type, backed by owned data.
///
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
/// More variants that adhere to these principles may get added in the future, in
/// particular to cover other combinations typically used. The data is stored in a single
/// contiguous buffer, which is backed by a vector.
///
/// # Note
/// - Does not support alpha channel natively.
/// - [`DynamicImageRef`] implements [`Serialize`] and [`Deserialize`] traits, and can be
///   deserialized from a [`DynamicImageRef`].
///
/// # Usage
///
/// ```
/// use refimage::{ImageOwned, ColorSpace, DynamicImageOwned};
///
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageOwned::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// let img = DynamicImageOwned::from(img);
///
/// ```
///
/// This type acts as a type-erased version of `ImageRef` and can be used to store
/// image data with different pixel types. The pixel type is determined at runtime.
#[derive(Debug, PartialEq, Clone)]
#[non_exhaustive]
pub enum DynamicImageOwned {
    /// [`ImageOwned`] with a `u8` primitive type.
    U8(ImageOwned<u8>),
    /// [`ImageOwned`] with a `u16` primitive type.
    U16(ImageOwned<u16>),
    /// [`ImageOwned`] with a `f32` primitive type.
    F32(ImageOwned<f32>),
}

/// Description of the color space of the image.
///
/// The colorspace information is used to enable debayering of the image data, and
/// for interpretation of single or multi-channel images.
#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Grayscale image.
    Gray = 0b000,
    /// Bayer mosaic image
    Bayer(BayerPattern) = 0b001,
    /// RGB image.
    Rgb = 0b100,
    /// Custom color space.
    Custom(u8, String) = 0b111,
}

/// Enum to describe the Bayer pattern of the image.
///
/// The Bayer pattern is used to interpret the raw image data from a Bayer mosaic image.
#[non_exhaustive]
#[derive(Debug, PartialEq, Copy, Clone, PartialOrd, Eq, Ord, Serialize, Deserialize)]
pub enum BayerPattern {
    /// BGGR Bayer pattern.
    Bggr,
    /// GBRG Bayer pattern.
    Gbrg,
    /// GRBG Bayer pattern.
    Grbg,
    /// RGGB Bayer pattern.
    Rggb,
}

/// Enum to describe the primitive pixel type of the image.
/// The underlying `i8` representation conforms to the FITS standard.
#[repr(i8)]
#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PixelType {
    /// 8-bit unsigned integer.
    U8 = 8,
    /// 16-bit unsigned integer.
    U16 = 16,
    /// 32-bit unsigned integer.
    U32 = 32,
    /// 64-bit unsigned integer.
    U64 = 64,
    /// 8-bit signed integer.
    I8 = -8,
    /// 16-bit signed integer.
    I16 = -16,
    /// 32-bit signed integer.
    I32 = -128,
    /// 64-bit signed integer.
    I64 = -78,
    /// 32-bit floating point.
    F32 = -32,
    /// 64-bit floating point.
    F64 = -64,
}

mod test {
    #[test]
    fn test_debayer() {
        use crate::demosaic::Debayer;
        use crate::ImageProps;
        // color_backtrace::install();
        let mut src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
        ];
        let expected = [
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
            169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
        ];
        let img = crate::ImageRef::create(
            &mut src,
            4,
            4,
            crate::ColorSpace::Bayer(crate::BayerPattern::Rggb),
        )
        .expect("Failed to create ImageRef");
        let a = img.debayer(crate::DemosaicMethod::None);
        assert!(a.is_ok());
        let a = a.unwrap(); // at this point, a is an ImageRef struct
        assert!(a.channels() == 3);
        assert!(a.width() == 4);
        assert!(a.height() == 4);
        assert!(a.color_space() == crate::ColorSpace::Rgb);
        assert_eq!(a.as_slice(), &expected);
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
