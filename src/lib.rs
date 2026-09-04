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
//! use refimage::chrono::DateTime;
//! use std::time::Duration;
//!
//! let mut data = vec![1u8, 2, 3, 4, 5, 6, 0, 0]; // 3x2 grayscale image, with extra padding that will be ignored
//! let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap(); // Create ImageRef
//! let img = DynamicImageRef::from(img); // Convert to DynamicImageRef
//! let now = DateTime::from_timestamp(1_700_000_000, 0).unwrap(); // in an app: chrono::Utc::now()
//! let mut img = GenericImageRef::new(now, Duration::from_millis(20), img); // timestamp + exposure are mandatory
//! img.insert_key("CAMERANAME", "Canon EOS 5D Mark IV".to_string()).unwrap(); // Insert metadata
//! let serialized = bincode::serialize(&img).unwrap(); // Serialize the image
//! let deserialized: GenericImageOwned = bincode::deserialize(&serialized).unwrap(); // Deserialize the image
//! ```
//! # Processing pipelines
//! All pixel conversions — debayer, luminance, pixel-type conversion, affine pixel
//! scaling, crop, ROI, flips, 90° rotations, aspect-preserving resize — are
//! [`Op`](pipeline::Op)s on a declarative, reusable
//! [`Pipeline`](pipeline::Pipeline). [`apply`](pipeline::Pipeline::apply)
//! runs it once and returns an owned image; given a [`GenericImageRef`] it returns a
//! [`GenericImageOwned`] with the metadata carried across unchanged.
//! [`compile`](pipeline::Pipeline::compile)-ing against a concrete
//! [`ImageSpec`](pipeline::ImageSpec) pre-allocates every buffer, yielding a
//! [`Runner`](pipeline::Runner) that processes successive frames with zero
//! per-frame allocation (serial-tiled strategy).
//!
//! # FITS
//! [`GenericImageRef`] / [`GenericImageOwned`] can be written to the [Flexible Image Transport
//! System](https://fits.gsfc.nasa.gov/fits_standard.html) via the [`FitsWrite`] trait. 
//! It supports uncompressed output and the
//! tile-compression convention with `GZIP_1` and `RICE_1`.
//!
//! # Optional Features
//! Features are available to extend the functionalities of the core `refimage` data types:
//! - `rayon`: Parallelizes the luminance / demosaic / cast kernels inside the [`pipeline`], and enables its parallel [`Strategy`](pipeline::Strategy) variants (<b>enabled</b> by default).
//! - `grow`: Lets a compiled [`Runner`](pipeline::Runner) reallocate its buffers when handed a frame whose shape differs from the one it was compiled for (<b>enabled</b> by default).
//! - `image`: Enables [`TryFrom`] conversions between [`DynamicImage`] and [`DynamicImageRef`], [`DynamicImageOwned`] (<b>disabled</b> by default).
//!

mod coreimpls;
mod coretraits;
mod demosaic;
#[cfg(feature = "image")]
mod dynamicimage_interop;
mod dynamicimage_serde;
mod dynamicimageowned;
mod dynamicimageref;
mod error;
mod fits;
mod genericimage;
mod genericimageowned;
mod genericimageref;
mod imageowned;
mod imageref;
mod imagetraits;
mod imageview;
mod metadata;
mod optimumexposure;
pub mod pipeline;

/// Re-export of the [`chrono`](https://docs.rs/chrono) crate. Image timestamps are
/// [`chrono::DateTime<Utc>`](chrono::DateTime); the caller supplies them (typically
/// `refimage::chrono::Utc::now()` — which needs `chrono`'s `clock` feature, so add
/// `chrono` as a direct dependency of your binary if you rely on `now()`).
pub use chrono;
pub use coretraits::{Enlargeable, PixelStor};
pub use demosaic::{BayerError, DemosaicMethod};
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use dynamicimage_interop::{InteropError, InteropResult};
pub use dynamicimage_serde::{SerdeError, SerdeResult};
pub use error::{ImageError, ImageResult};
pub use fits::{
    create_fits, create_fits_to, AutoTile, DitherSeed, FitsCompression, FitsCompressionKind,
    FitsError, FitsResult, FitsWrite, FitsWriter, FixedTile, Gzip, Hcompress, Quantize, Rice,
};
pub use genericimage::GenericImage;
pub use genericimageowned::GenericImageOwned;
pub use genericimageref::GenericImageRef;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use image::DynamicImage; // Used for image interop
pub use imageowned::ImageOwned;
pub use imageref::ImageRef;
pub use imagetraits::{BayerShift, ImageProps, PixelData};
pub use imageview::{DynamicImageView, ImageView};
pub use metadata::{
    GenericLineItem, GenericValue, InsertValue, MetaCollection, Metadata, MetadataError,
    MetadataResult, CAMERANAME_KEY, EXPOSURE_KEY, FRAMEID_KEY, PROGRAMNAME_KEY, TIMESTAMP_KEY,
};
pub use optimumexposure::{
    CalcOptExp, ExposureError, ExposureResult, OptimumExposure, OptimumExposureBuilder,
    OptimumExposureResult,
};
pub use serde::{Deserialize, Serialize};
use serde::{Deserializer, Serializer};

/// Image data with a dynamic pixel type, backed by a mutable slice of data.
///
/// This represents a _matrix_ of _pixels_ whose element type is one of `u8`,
/// `u16`, or `f32` (a `u16` buffer may additionally be tagged 10-/12-/14-bit —
/// see [`ImageRef::with_bit_depth`]). The matrix is stored in _row-major_ order
/// in a single contiguous buffer, backed by a mutable slice, and aims to enable
/// reuse of allocated memory without re-allocation.
///
/// Raw sample access is via the [`PixelData`] trait; a shared read-only borrow
/// is a [`DynamicImageView`] ([`view`](DynamicImageRef::view)).
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
/// This represents a _matrix_ of _pixels_ whose element type is one of `u8`,
/// `u16`, or `f32` (a `u16` buffer may additionally be tagged 10-/12-/14-bit —
/// see [`ImageOwned::with_bit_depth`]). The matrix is stored in _row-major_
/// order in a single contiguous buffer, backed by a vector.
///
/// Raw sample access is via the [`PixelData`] trait; a shared read-only borrow
/// is a [`DynamicImageView`] ([`view`](DynamicImageOwned::view)).
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
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ColorSpace {
    /// Grayscale image.
    Gray,
    /// Bayer mosaic image
    Bayer(BayerPattern),
    /// RGB image.
    Rgb,
    /// Custom color space.
    ///
    /// The first byte is the number of channels, and the string describes the colorspace.
    Custom(u8, String),
}

impl ColorSpace {
    /// Number of interleaved channels per pixel implied by this color space:
    /// `1` for [`Gray`](Self::Gray) and [`Bayer`](Self::Bayer), `3` for
    /// [`Rgb`](Self::Rgb), and the stored count for [`Custom`](Self::Custom).
    pub const fn channels(&self) -> u8 {
        match self {
            ColorSpace::Gray | ColorSpace::Bayer(_) => 1,
            ColorSpace::Rgb => 3,
            ColorSpace::Custom(ch, _) => *ch,
        }
    }
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

/// The primitive element type of an image's samples.
///
/// Only the six variants here can be stored in a [`DynamicImageRef`] /
/// [`DynamicImageOwned`] or processed by a [`pipeline`]: the storage widths
/// [`U8`](Self::U8), [`U16`](Self::U16) and [`F32`](Self::F32), plus the three
/// sub-container machine-vision depths [`U10`](Self::U10) / [`U12`](Self::U12) /
/// [`U14`](Self::U14) that live right-aligned in a `u16` (see
/// [`ImageRef::with_bit_depth`]). The `#[repr(i8)]` discriminants follow the FITS
/// `BITPIX` convention (`U8` = 8, `U16` = 16, `F32` = -32); `U10` / `U12` / `U14`
/// are a `refimage` extension that serializes as its 16-bit [`storage`](Self::storage).
#[repr(i8)]
#[non_exhaustive]
#[derive(Debug, PartialEq, Clone, Copy, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PixelType {
    /// 8-bit unsigned integer.
    U8 = 8,
    /// 10-bit unsigned integer, stored right-aligned in a `u16` (values
    /// `0..=1023`). A machine-vision sensor depth; carries the same storage
    /// as [`U16`](Self::U16) but only the low 10 bits are meaningful.
    U10 = 10,
    /// 12-bit unsigned integer, stored right-aligned in a `u16` (values
    /// `0..=4095`).
    U12 = 12,
    /// 14-bit unsigned integer, stored right-aligned in a `u16` (values
    /// `0..=16383`).
    U14 = 14,
    /// 16-bit unsigned integer.
    U16 = 16,
    /// 32-bit floating point.
    F32 = -32,
}

impl PixelType {
    /// Number of meaningful bits per sample — `10` / `12` / `14` for the
    /// sub-container machine-vision depths, otherwise the storage width.
    pub const fn bit_depth(self) -> u8 {
        match self {
            PixelType::U8 => 8,
            PixelType::U10 => 10,
            PixelType::U12 => 12,
            PixelType::U14 => 14,
            PixelType::U16 => 16,
            PixelType::F32 => 32,
        }
    }

    /// The storage `PixelType` — [`U10`](Self::U10) / [`U12`](Self::U12) /
    /// [`U14`](Self::U14) all live in a `u16` and collapse to
    /// [`U16`](Self::U16); every other variant maps to itself. This is what
    /// determines the on-disk / on-wire byte layout (and FITS `BITPIX`).
    pub const fn storage(self) -> PixelType {
        match self {
            PixelType::U10 | PixelType::U12 | PixelType::U14 => PixelType::U16,
            PixelType::U8 | PixelType::U16 | PixelType::F32 => self,
        }
    }
}

mod test {
    #[test]
    fn test_debayer() {
        use crate::pipeline::Pipeline;
        use crate::{ImageProps, PixelData};
        // color_backtrace::install();
        let mut src: [u8; 16] = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
        ];
        let expected: [u8; 48] = [
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
            169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
        ];
        let img = crate::DynamicImageRef::from(
            crate::ImageRef::new(
                &mut src,
                4,
                4,
                crate::ColorSpace::Bayer(crate::BayerPattern::Rggb),
            )
            .expect("Failed to create ImageRef"),
        );
        let a = Pipeline::new()
            .debayer(crate::DemosaicMethod::None)
            .apply(&img)
            .expect("debayer pipeline");
        assert!(a.channels() == 3);
        assert!(a.width() == 4);
        assert!(a.height() == 4);
        assert!(a.color_space() == crate::ColorSpace::Rgb);
        assert_eq!(a.as_raw_u8(), &expected);
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
        #[allow(missing_abi)]
        #[doc = $content] unsafe extern { }
    }
}

// Provides the README.md as doc, to ensure the example works!
insert_as_doc!(include_str!("../README.MD"));
