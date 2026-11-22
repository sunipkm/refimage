use std::time::Duration;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
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

/// Name of the timestamp field, reserved because [`Metadata`] stores it as a
/// typed field rather than a map entry.
pub const TIMESTAMP_KEY: &str = "TIMESTAMP";
/// Key for the camera name metadata.
pub const CAMERANAME_KEY: &str = "CAMERA";
/// Key for the name of the program that generated this object.
pub const PROGRAMNAME_KEY: &str = "PROGNAME";
/// Name of the exposure field, reserved because [`Metadata`] stores it as a
/// typed field rather than a map entry.
pub const EXPOSURE_KEY: &str = "EXPOSURE";
/// Name of the frame-ID field, reserved because [`Metadata`] stores it as a
/// typed field rather than a map entry.
pub const FRAMEID_KEY: &str = "FRAMEID";

/// Stored `frame_id` value for "unset".
const FRAMEID_UNSET: i64 = i64::MIN;

fn frameid_default() -> i64 {
    FRAMEID_UNSET
}

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
/// - [`std::time::Duration`] | [`chrono::DateTime<Utc>`](chrono::DateTime)
/// - [`String`] | [`&str`]
///
/// The metadata values are encapsulated in a type-erased enum [`GenericValue`].
///
/// # Note
/// - The metadata key is case-insensitive and is stored as an uppercase string.
/// - When saving to a FITS file, the metadata comment may be truncated.
/// - Metadata of type [`std::time::Duration`] or [`chrono::DateTime<Utc>`](chrono::DateTime)
///   is stored as two consecutive metadata items, split into seconds and
///   nanoseconds, with keys suffixed `_S` and `_NS`, plus a convenient base card
///   (`f64` seconds for a `Duration`, an ISO-8601 string for a timestamp).
///
pub struct GenericLineItem {
    pub(crate) value: GenericValue,
    pub(crate) comment: Option<String>,
}

/// A collection of metadata items, keyed by uppercase name and kept in
/// insertion order.
pub type MetaCollection = IndexMap<String, GenericLineItem>;

/// Image metadata: a mandatory typed core with an ordered map of extra items.
///
/// Every [`GenericImageRef`](crate::GenericImageRef) /
/// [`GenericImageOwned`](crate::GenericImageOwned) carries metadata. The
/// `timestamp` (a UTC [`DateTime`]) and `exposure`
/// are stored as typed fields.
///
/// A zero [`exposure`](Metadata::exposure) (`Duration::ZERO`) implies "unknown / not
/// applicable" (e.g. a synthetic or stacked frame).
///
/// `frame_id` is a `u32` acquisition counter, unset by default.
/// Use [`set_frame_id`](Metadata::set_frame_id) to set the value, and [`frame_id`](Metadata::frame_id) to retrieve it as an `Option<u32>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Metadata {
    timestamp: DateTime<Utc>,
    exposure: Duration,
    #[serde(default = "frameid_default")]
    frame_id: i64,
    extra: MetaCollection,
}

impl Metadata {
    /// Create a metadata block with an empty extra-metadata map.
    ///
    /// Pass `Duration::ZERO` for `exposure` when the image has no meaningful
    /// single exposure.
    pub fn new(timestamp: DateTime<Utc>, exposure: Duration) -> Self {
        Self {
            timestamp,
            exposure,
            frame_id: frameid_default(),
            extra: MetaCollection::new(),
        }
    }

    /// The image creation timestamp (UTC).
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    /// The exposure duration (`Duration::ZERO` if unknown / not applicable).
    pub fn exposure(&self) -> Duration {
        self.exposure
    }

    /// The acquisition frame ID, or `None` if unset.
    pub fn frame_id(&self) -> Option<u32> {
        u32::try_from(self.frame_id).ok()
    }

    /// Set the timestamp.
    pub fn set_timestamp(&mut self, timestamp: DateTime<Utc>) {
        self.timestamp = timestamp;
    }

    /// Set the exposure duration.
    pub fn set_exposure(&mut self, exposure: Duration) {
        self.exposure = exposure;
    }

    /// Set the acquisition frame ID.
    pub fn set_frame_id(&mut self, frame_id: u32) {
        self.frame_id = i64::from(frame_id);
    }

    /// Borrow the ordered map of extra metadata items.
    pub fn extra(&self) -> &MetaCollection {
        &self.extra
    }

    /// Number of extra metadata items (excludes the typed timestamp / exposure).
    pub fn len(&self) -> usize {
        self.extra.len()
    }

    /// `true` if there are no extra metadata items.
    pub fn is_empty(&self) -> bool {
        self.extra.is_empty()
    }

    /// Iterate the extra metadata items in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &GenericLineItem)> {
        self.extra.iter()
    }

    /// Look up an extra metadata item by name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&GenericLineItem> {
        self.extra.get(&name.to_uppercase())
    }

    /// Insert an extra metadata item.
    ///
    /// # Errors
    /// [`MetadataError::ReservedKey`] for `TIMESTAMP` / `EXPOSURE` / `FRAMEID` (use
    /// [`set_timestamp`](Self::set_timestamp) / [`set_exposure`](Self::set_exposure) /
    /// [`set_frame_id`](Self::set_frame_id)),
    /// otherwise a key/comment/value validation error.
    pub fn insert<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), MetadataError> {
        reserved_check(name)?;
        T::insert(&mut self.extra, name, value)
    }

    /// Replace an existing extra metadata item.
    ///
    /// # Errors
    /// [`MetadataError::ReservedKey`] for `TIMESTAMP` / `EXPOSURE` / `FRAMEID`,
    /// [`MetadataError::KeyNotFound`] if the key is absent, otherwise a
    /// validation error.
    pub fn replace<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<GenericLineItem, MetadataError> {
        reserved_check(name)?;
        T::replace(&mut self.extra, name, value)
    }

    /// Remove an extra metadata item, returning it.
    ///
    /// # Errors
    /// [`MetadataError::ReservedKey`] for `TIMESTAMP` / `EXPOSURE` / `FRAMEID`,
    /// [`MetadataError::KeyNotFound`] if the key is absent, otherwise a
    /// key-validation error.
    pub fn remove(&mut self, name: &str) -> Result<GenericLineItem, MetadataError> {
        reserved_check(name)?;
        name_check(name)?;
        self.extra
            .shift_remove(&name.to_uppercase())
            .ok_or(MetadataError::KeyNotFound)
    }
}

fn reserved_check(name: &str) -> Result<(), MetadataError> {
    match name.to_uppercase().as_str() {
        TIMESTAMP_KEY => Err(MetadataError::ReservedKey(TIMESTAMP_KEY)),
        EXPOSURE_KEY => Err(MetadataError::ReservedKey(EXPOSURE_KEY)),
        FRAMEID_KEY => Err(MetadataError::ReservedKey(FRAMEID_KEY)),
        _ => Ok(()),
    }
}

/// A type-erased enum to hold a metadata value.
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
    /// A UTC timestamp.
    Timestamp(DateTime<Utc>),
    /// A string.
    String(String),
}

impl GenericLineItem {
    /// The comment attached to this metadata item, if any.
    pub fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    /// The value of this metadata item.
    pub fn value(&self) -> &GenericValue {
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
impl_from_genericvalue!(DateTime<Utc>, GenericValue::Timestamp);
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
impl_tryinto_genericvalue!(DateTime<Utc>, GenericValue::Timestamp);
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
insert_value_impl!(DateTime<Utc>);

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
            #[doc = "This value as [`" $t " `], or `None` if it is a different type."]
            #[inline(always)]
            pub fn [<value_ $t:lower>](&self) -> Option<$t> {
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

    /// This value as a UTC timestamp, or `None` if it is a different type.
    #[inline(always)]
    pub fn value_timestamp(&self) -> Option<DateTime<Utc>> {
        self.clone().try_into().ok()
    }

    /// This value as a string, or `None` if it is a different type.
    #[inline(always)]
    pub fn value_string(&self) -> Option<&str> {
        match self {
            GenericValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

mod test {

    /// `Pipeline::apply` on an owned metadata-bearing image keeps the metadata and
    /// returns a `GenericImageOwned`.
    #[test]
    fn apply_on_generic_owned_keeps_metadata() {
        use crate::pipeline::Pipeline;
        use crate::{
            BayerPattern, DemosaicMethod, DynamicImageOwned, GenericImageOwned, ImageOwned,
            ImageProps,
        };
        use chrono::DateTime;
        use std::time::Duration;

        let data = vec![0u8; 256];
        let img = ImageOwned::from_owned(data, 16, 16, BayerPattern::Grbg.into()).unwrap();
        let img = DynamicImageOwned::from(img);
        let ts = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut img = GenericImageOwned::new(ts, Duration::from_millis(10), img);

        img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
        img.insert_key("TESTING_THIS_LONG_KEY", "This is a long key")
            .unwrap();

        let out: GenericImageOwned = Pipeline::new()
            .debayer(DemosaicMethod::Linear)
            .apply(&img)
            .unwrap();

        assert_eq!(img.metadata(), out.metadata());
        assert_eq!(img.timestamp(), out.timestamp());
        assert_eq!(img.image().width(), out.image().width());
        assert_eq!(img.image().height(), out.image().height());
        assert_eq!(img.image().channels() * 3, out.image().channels());
    }

    #[test]
    fn metadata_typed_core_and_reserved_keys() {
        use super::{Metadata, MetadataError};
        use chrono::DateTime;
        use std::time::Duration;

        let ts = DateTime::from_timestamp(1_000_000, 0).unwrap();
        let mut m = Metadata::new(ts, Duration::from_millis(250));
        assert_eq!(m.timestamp(), ts);
        assert_eq!(m.exposure(), Duration::from_millis(250));
        assert_eq!(m.frame_id(), None);
        m.set_frame_id(42);
        assert_eq!(m.frame_id(), Some(42));
        assert!(m.is_empty());

        assert_eq!(
            m.insert("TIMESTAMP", 1u8),
            Err(MetadataError::ReservedKey("TIMESTAMP"))
        );
        assert_eq!(
            m.insert("exposure", 1u8),
            Err(MetadataError::ReservedKey("EXPOSURE"))
        );
        assert_eq!(
            m.insert("frameid", 1u8),
            Err(MetadataError::ReservedKey("FRAMEID"))
        );

        m.insert("Camera", "cam").unwrap();
        m.insert("GAIN", 3u16).unwrap();
        assert_eq!(m.len(), 2);
        // case-insensitive lookup
        assert!(m.get("camera").is_some());
        // insertion order is preserved
        let keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["CAMERA", "GAIN"]);

        m.set_exposure(Duration::ZERO);
        assert_eq!(m.exposure(), Duration::ZERO);
        let epoch = DateTime::from_timestamp(0, 0).unwrap();
        m.set_timestamp(epoch);
        assert_eq!(m.timestamp(), epoch);

        assert!(m.remove("GAIN").is_ok());
        assert_eq!(m.remove("GAIN"), Err(MetadataError::KeyNotFound));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn metadata_roundtrips_through_bincode() {
        use super::Metadata;
        use chrono::DateTime;
        use std::time::Duration;

        let mut m = Metadata::new(
            DateTime::from_timestamp(42, 0).unwrap(),
            Duration::from_secs(1),
        );
        m.insert("A", 1i32).unwrap();
        m.insert("B", "two").unwrap();
        m.set_frame_id(7);
        let bytes = bincode::serialize(&m).unwrap();
        let back: Metadata = bincode::deserialize(&bytes).unwrap();
        assert_eq!(m, back);
        assert_eq!(back.frame_id(), Some(7));

        // pre-epoch timestamps round-trip too (they did not with `SystemTime`).
        let pre = Metadata::new(
            DateTime::from_timestamp(-2_000_000_000, 0).unwrap(),
            Duration::ZERO,
        );
        let back: Metadata = bincode::deserialize(&bincode::serialize(&pre).unwrap()).unwrap();
        assert_eq!(pre, back);
    }
}
