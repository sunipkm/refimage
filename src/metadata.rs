use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ColorSpace;
#[allow(unused)]
use crate::GenericImage;

extern crate paste;

/// Errors from inserting, replacing, removing, or reading metadata.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MetadataError {
    /// The key is empty.
    #[error("metadata key cannot be empty")]
    EmptyKey,
    /// The key is longer than 80 characters.
    #[error("metadata key cannot be longer than 80 characters")]
    KeyTooLong,
    /// The comment is empty.
    #[error("metadata comment cannot be empty")]
    EmptyComment,
    /// The comment is longer than 4096 characters.
    #[error("metadata comment cannot be longer than 4096 characters")]
    CommentTooLong,
    /// The string value is empty.
    #[error("metadata value cannot be empty")]
    EmptyValue,
    /// The string value is longer than 4096 characters.
    #[error("metadata value cannot be longer than 4096 characters")]
    ValueTooLong,
    /// No entry exists for the requested key.
    #[error("metadata key not found")]
    KeyNotFound,
    /// The key names a reserved entry that cannot be inserted, replaced or removed.
    #[error("metadata key {0:?} is reserved")]
    ReservedKey(&'static str),
    /// A stored [`GenericValue`] was requested as an incompatible type.
    #[error("metadata value has a different type")]
    WrongValueType,
}

/// `Result` alias for [`MetadataError`].
pub type MetadataResult<T> = Result<T, MetadataError>;

/// Key for the timestamp metadata.
/// This key is inserted by default when creating a new [`GenericImageRef`], [`GenericImageOwned`] or [`GenericImage`].
pub const TIMESTAMP_KEY: &str = "TIMESTAMP";
/// Key for the camera name metadata.
pub const CAMERANAME_KEY: &str = "CAMERA";
/// Key for the name of the program that generated this object.
pub const PROGRAMNAME_KEY: &str = "PROGNAME";
/// Key for exposure time metadata of the image.
pub const EXPOSURE_KEY: &str = "EXPOSURE";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// A metadata item.
///
/// This struct holds a metadata item, which is a key-value pair with an optional comment.
///
/// # Usage
/// This struct is not meant to be used directly. Instead, use the [`crate::GenericImageRef`]
/// struct and associated methods to insert new metadata items, or to get existing
/// metadata items.
///
/// # Valid Types
/// The valid types for the metadata value are:
/// - [`u8`] | [`u16`] | [`u32`] | [`u64`]
/// - [`i8`] | [`i16`] | [`i32`] | [`i64`]
/// - [`f32`] | [`f64`]
/// - [`ColorSpace`]
/// - [`std::time::Duration`] | [`std::time::SystemTime`]
/// - [`String`] | [`&str`]
///
/// The metadata values are encapsulated in a type-erased enum [`GenericValue`].
///
/// # Note
/// - The metadata key is case-insensitive and is stored as an uppercase string.
/// - When saving to a FITS file, the metadata comment may be truncated.
/// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] are
///   1. Stored as two consecutive metadata items, split into seconds ([`u64`])
///      and nanoseconds ([`u64`]). The keys are suffixed with `_S` and `_NS`.
///   2. Metadata of type [`Duration`] is stored as a single floating point
///      number ([`f64`]), in seconds, under the original key.
///
pub struct GenericLineItem {
    pub(crate) value: GenericValue,
    pub(crate) comment: Option<String>,
}

/// A collection of metadata items.
pub type MetaCollection = HashMap<String, GenericLineItem>;

/// A type-erased enum to hold a metadata value.
///
/// The set of variants mirrors what a FITS header card can carry. Integer metadata of
/// any width is promoted losslessly to [`GenericValue::Integer`] (`i64`); `u64` is not a
/// supported metadata type because it cannot be promoted without loss. `f32` and `f64`
/// both land in [`GenericValue::Real`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GenericValue {
    /// An integer. All integer metadata is stored here as `i64`.
    Integer(i64),
    /// A floating point number. `f32` metadata is widened to `f64`.
    Real(f64),
    /// Color space of the image ([`ColorSpace`]).
    ColorSpace(crate::ColorSpace),
    /// A [`Duration`].
    Duration(Duration),
    /// A [`SystemTime`].
    SystemTime(SystemTime),
    /// A string.
    String(String),
}

impl GenericLineItem {
    /// Get the comment of the metadata value.
    pub fn get_comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    /// Get the value of the metadata item.
    pub fn get_value(&self) -> &GenericValue {
        &self.value
    }
}

/// `From<int> for GenericValue` via lossless widening to `i64`.
macro_rules! impl_from_int {
    ($t:ty) => {
        impl From<$t> for GenericValue {
            fn from(value: $t) -> Self {
                GenericValue::Integer(i64::from(value))
            }
        }
    };
}

impl_from_int!(u8);
impl_from_int!(u16);
impl_from_int!(u32);
impl_from_int!(i8);
impl_from_int!(i16);
impl_from_int!(i32);
impl_from_int!(i64);

impl From<f32> for GenericValue {
    fn from(value: f32) -> Self {
        GenericValue::Real(f64::from(value))
    }
}

impl From<f64> for GenericValue {
    fn from(value: f64) -> Self {
        GenericValue::Real(value)
    }
}

macro_rules! impl_from_genericvalue {
    ($t:ty, $variant:path) => {
        impl From<$t> for GenericValue {
            fn from(value: $t) -> Self {
                $variant(value)
            }
        }
    };
}

impl_from_genericvalue!(ColorSpace, GenericValue::ColorSpace);
impl_from_genericvalue!(Duration, GenericValue::Duration);
impl_from_genericvalue!(SystemTime, GenericValue::SystemTime);
impl_from_genericvalue!(String, GenericValue::String);

/// `TryInto<int> for GenericValue` from the stored `i64`, range-checked.
macro_rules! impl_tryinto_int {
    ($t:ty) => {
        impl TryInto<$t> for GenericValue {
            type Error = MetadataError;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    GenericValue::Integer(x) => {
                        <$t>::try_from(x).map_err(|_| MetadataError::WrongValueType)
                    }
                    _ => Err(MetadataError::WrongValueType),
                }
            }
        }
    };
}

impl_tryinto_int!(u8);
impl_tryinto_int!(u16);
impl_tryinto_int!(u32);
impl_tryinto_int!(i8);
impl_tryinto_int!(i16);
impl_tryinto_int!(i32);
impl_tryinto_int!(i64);

impl TryInto<f32> for GenericValue {
    type Error = MetadataError;

    fn try_into(self) -> Result<f32, Self::Error> {
        match self {
            GenericValue::Real(x) => Ok(x as f32),
            _ => Err(MetadataError::WrongValueType),
        }
    }
}

impl TryInto<f64> for GenericValue {
    type Error = MetadataError;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            GenericValue::Real(x) => Ok(x),
            _ => Err(MetadataError::WrongValueType),
        }
    }
}

macro_rules! impl_tryinto_genericvalue {
    ($t:ty, $variant:path) => {
        impl TryInto<$t> for GenericValue {
            type Error = MetadataError;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    $variant(x) => Ok(x),
                    _ => Err(MetadataError::WrongValueType),
                }
            }
        }
    };
}

impl_tryinto_genericvalue!(ColorSpace, GenericValue::ColorSpace);
impl_tryinto_genericvalue!(Duration, GenericValue::Duration);
impl_tryinto_genericvalue!(SystemTime, GenericValue::SystemTime);
impl_tryinto_genericvalue!(String, GenericValue::String);

/// Trait to insert a metadata value into a [`MetaCollection`].
pub trait InsertValue {
    /// Insert a metadata value into a [`MetaCollection`] by name.
    fn insert(f: &mut MetaCollection, name: &str, value: Self) -> Result<(), MetadataError>;

    /// Replace a metadata value in a [`MetaCollection`] by name.
    fn replace(
        f: &mut MetaCollection,
        name: &str,
        value: Self,
    ) -> Result<GenericLineItem, MetadataError>;
}

macro_rules! insert_value_impl {
    ($t:ty) => {
        impl InsertValue for $t {
            fn insert(
                f: &mut MetaCollection,
                name: &str,
                value: Self,
            ) -> Result<(), MetadataError> {
                name_check(name)?;
                let line = GenericLineItem {
                    value: value.into(),
                    comment: None,
                };
                f.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace(
                f: &mut MetaCollection,
                name: &str,
                value: Self,
            ) -> Result<GenericLineItem, MetadataError> {
                name_check(name)?;
                let line = GenericLineItem {
                    value: value.into(),
                    comment: None,
                };
                f.insert(name.to_uppercase(), line)
                    .ok_or(MetadataError::KeyNotFound)
            }
        }

        impl InsertValue for ($t, &str) {
            fn insert(
                f: &mut MetaCollection,
                name: &str,
                value: Self,
            ) -> Result<(), MetadataError> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = GenericLineItem {
                    value: value.0.into(),
                    comment: Some(value.1.to_owned()),
                };
                f.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace(
                f: &mut MetaCollection,
                name: &str,
                value: Self,
            ) -> Result<GenericLineItem, MetadataError> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = GenericLineItem {
                    value: value.0.into(),
                    comment: Some(value.1.to_owned()),
                };
                f.insert(name.to_uppercase(), line)
                    .ok_or(MetadataError::KeyNotFound)
            }
        }
    };
}

pub(crate) fn name_check(name: &str) -> Result<(), MetadataError> {
    if name.is_empty() {
        Err(MetadataError::EmptyKey)
    } else if name.len() > 80 {
        Err(MetadataError::KeyTooLong)
    } else {
        Ok(())
    }
}

fn comment_check(comment: &str) -> Result<(), MetadataError> {
    if comment.is_empty() {
        Err(MetadataError::EmptyComment)
    } else if comment.len() > 4096 {
        Err(MetadataError::CommentTooLong)
    } else {
        Ok(())
    }
}

#[allow(dead_code)]
fn str_value_check(value: &str) -> Result<(), MetadataError> {
    if value.is_empty() {
        Err(MetadataError::EmptyValue)
    } else if value.len() > 4096 {
        Err(MetadataError::ValueTooLong)
    } else {
        Ok(())
    }
}

insert_value_impl!(u8);
insert_value_impl!(u16);
insert_value_impl!(u32);
insert_value_impl!(i8);
insert_value_impl!(i16);
insert_value_impl!(i32);
insert_value_impl!(i64);
insert_value_impl!(f32);
insert_value_impl!(f64);
insert_value_impl!(ColorSpace);
insert_value_impl!(String);
insert_value_impl!(Duration);
insert_value_impl!(SystemTime);

impl InsertValue for &str {
    fn insert(f: &mut MetaCollection, name: &str, value: Self) -> Result<(), MetadataError> {
        name_check(name)?;
        str_value_check(value)?;
        let line = GenericLineItem {
            value: value.to_owned().into(),
            comment: None,
        };
        f.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace(
        f: &mut MetaCollection,
        name: &str,
        value: Self,
    ) -> Result<GenericLineItem, MetadataError> {
        name_check(name)?;
        str_value_check(value)?;
        let value = GenericLineItem {
            value: value.to_owned().into(),
            comment: None,
        };
        f.insert(name.to_uppercase(), value)
            .ok_or(MetadataError::KeyNotFound)
    }
}

impl InsertValue for (&str, &str) {
    fn insert(f: &mut MetaCollection, name: &str, value: Self) -> Result<(), MetadataError> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let line = GenericLineItem {
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace(
        f: &mut MetaCollection,
        name: &str,
        value: Self,
    ) -> Result<GenericLineItem, MetadataError> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let value = GenericLineItem {
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.insert(name.to_uppercase(), value)
            .ok_or(MetadataError::KeyNotFound)
    }
}

macro_rules! impl_getter {
    ($t:ty) => {
        ::paste::paste! {
            #[doc = "Get the metadata value of type [`" $t " `]."]
            #[inline(always)]
            pub fn [<get_value_ $t:lower>](&self) -> Option<$t> {
                self.clone().try_into().ok()
            }
        }
    };
}

impl GenericValue {
    impl_getter!(u8);
    impl_getter!(u16);
    impl_getter!(u32);
    impl_getter!(i8);
    impl_getter!(i16);
    impl_getter!(i32);
    impl_getter!(i64);
    impl_getter!(f32);
    impl_getter!(f64);
    impl_getter!(Duration);
    impl_getter!(SystemTime);

    /// Get the `String` metadata value.
    #[inline(always)]
    pub fn get_value_string(&self) -> Option<&str> {
        match self {
            GenericValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

mod test {

    #[test]
    fn test_operate_owned() {
        use crate::pipeline::Pipeline;
        use crate::{
            BayerPattern, DemosaicMethod, DynamicImageOwned, DynamicImageRef, GenericImageOwned,
            ImageOwned, ImageProps, ImageRef,
        };
        use std::time::SystemTime;

        let data = vec![0u8; 256];
        let img = ImageOwned::from_owned(data, 16, 16, BayerPattern::Grbg.into()).unwrap();
        let img = DynamicImageOwned::from(img);
        let mut img = GenericImageOwned::new(SystemTime::now(), img);

        img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
        img.insert_key("TESTING_THIS_LONG_KEY", "This is a long key")
            .unwrap();
        let img2 = img
            .operate(|x| {
                let mut raw = x.as_raw_u8().to_vec();
                let src = DynamicImageRef::from(
                    ImageRef::<u8>::from_u8_mut(&mut raw, x.width(), x.height(), x.color_space())
                        .unwrap(),
                );
                Pipeline::new()
                    .debayer(DemosaicMethod::Linear)
                    .apply(&src)
                    .map_err(|_| "debayer failed")
            })
            .unwrap();
        let img3 = img.operate(|x| Ok::<_, &str>(x.clone())).unwrap();
        assert_eq!(img, img3);
        assert_eq!(img.get_metadata(), img2.get_metadata());
        assert_eq!(img.get_image().width(), img2.get_image().width());
        assert_eq!(img.get_image().height(), img2.get_image().height());
        assert_eq!(img.get_image().channels() * 3, img2.get_image().channels());
    }
}
