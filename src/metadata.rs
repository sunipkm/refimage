use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::{BayerError, Debayer, DemosaicMethod, DynamicImageData};

/// Key for the timestamp metadata.
/// This key is inserted by default when creating a new [`GenericImage`].
pub const TIMESTAMP_KEY: &str = "TIMESTAMP";
/// Key for the camera name metadata.
pub const CAMERANAME_KEY: &str = "CAMERA";
/// Key for the name of the program that generated this object.
pub const PROGRAMNAME_KEY: &str = "PROGNAME";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// A metadata item.
///
/// This struct holds a metadata item, which is a key-value pair with an optional comment.
///
/// # Usage
/// This struct is not meant to be used directly. Instead, use the [`crate::GenericImage`]
/// struct and associated methods to insert new metadata items, or to get existing
/// metadata items.
///
/// # Valid Types
/// The valid types for the metadata value are:
/// - [`u8`] | [`u16`] | [`u32`] | [`u64`]
/// - [`i8`] | [`i16`] | [`i32`] | [`i64`]
/// - [`f32`] | [`f64`]
/// - [`std::time::Duration`] | [`std::time::SystemTime`]
/// - [`String`] | [`&str`]
///
/// # Note
/// - The metadata key is case-insensitive and is stored as an uppercase string.
/// - When saving to a FITS file, the metadata comment may be truncated.
/// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] is split
///   and stored as two consecutive metadata items, with the same key, split into
///   seconds ([`u64`]) and microseconds ([`u32`]).
///
pub struct GenericLineItem(pub(crate) PrvGenLineItem);

impl From<PrvGenLineItem> for GenericLineItem {
    fn from(item: PrvGenLineItem) -> Self {
        Self(item)
    }
}

/// A serializable, generic image with metadata.
///
/// This struct holds an image with associated metadata. The metadata is stored as a vector of
/// [`GenericLineItem`] structs. The image data is stored as a [`DynamicImageData`].
///
/// # Usage
/// ```
/// use refimage::{ImageData, DynamicImageData, GenericImage, ColorSpace};
/// use std::time::SystemTime;
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// let img = DynamicImageData::from(img);
/// let mut img = GenericImage::new(std::time::SystemTime::now(), img);
///
/// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericImage<'a> {
    metadata: Vec<GenericLineItem>,
    #[serde(borrow)]
    image: DynamicImageData<'a>,
}

impl<'a> GenericImage<'a> {
    /// Create a new `GenericImage` with metadata.
    ///
    /// # Arguments
    /// - `tstamp`: The timestamp of the image.
    /// - `image`: The image data, of type `DynamicImageData`.
    ///
    /// # Example
    /// ```
    /// use refimage::{ImageData, DynamicImageData, GenericImage, ColorSpace};
    /// use std::time::SystemTime;
    /// let data = vec![1u8, 2, 3, 4, 5, 6];
    /// let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
    /// let img = DynamicImageData::from(img);
    /// let mut img = GenericImage::new(std::time::SystemTime::now(), img);
    ///
    /// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
    /// ```
    pub fn new(tstamp: SystemTime, image: DynamicImageData<'a>) -> Self {
        let metadata = vec![PrvGenLineItem::SystemTime(PrvLineItem {
            name: TIMESTAMP_KEY.to_string(),
            value: tstamp,
            comment: Some("Timestamp of the image".to_owned()),
        })
        .into()];
        Self { metadata, image }
    }

    /// Insert a metadata value into the [`GenericImage`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value. The name must be non-empty and less than 80 characters.
    /// - `value`: The value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or `std::time::SystemTime` or a tuple of a primitive type and a comment ().
    /// # Valid Types
    /// The valid types for the metadata value are:
    /// - [`u8`] | [`u16`] | [`u32`] | [`u64`]
    /// - [`i8`] | [`i16`] | [`i32`] | [`i64`]
    /// - [`f32`] | [`f64`]
    /// - [`std::time::Duration`] | [`std::time::SystemTime`]
    /// - [`String`] | [`&str`]
    ///
    /// # Note
    /// - The metadata key is case-insensitive and is stored as an uppercase string.
    /// - When saving to a FITS file, the metadata comment may be truncated.
    /// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] is split
    ///   and stored as two consecutive metadata items, with the same key, split into
    ///   seconds ([`u64`]) and microseconds ([`u32`]).
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        T::insert_key(self, name, value)
    }

    /// Remove a metadata value from the [`GenericImage`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value to remove.
    ///
    /// # Returns
    /// - `Ok(())` if the key was removed successfully.
    /// - `Err("Key not found")` if the key was not found.
    /// - `Err("Key cannot be empty")` if the key is an empty string.
    /// - `Err("Key cannot be longer than 80 characters")` if the key is longer than 80 characters.
    pub fn remove_key(&mut self, name: &str) -> Result<(), &'static str> {
        name_check(name)?;
        let mut not_found = true;
        self.metadata.retain(|x| {
            let found = x.name() == name;
            not_found &= x.name() != name;
            found
        });
        if not_found {
            Err("Key not found")
        } else {
            Ok(())
        }
    }

    /// Replace a metadata value in the [`GenericImage`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value to replace.
    /// - `value`: The new value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or `std::time::SystemTime` or a tuple of a value type and a comment.
    ///
    /// # Returns
    /// - `Ok(())` if the key was replaced successfully.
    /// - `Err("Key not found")` if the key was not found.
    ///
    pub fn replace_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<(), &'static str> {
        T::replace(self, name, value)
    }

    /// Get the underlying [`DynamicImageData`].
    ///
    /// # Returns
    /// The underlying [`DynamicImageData`] of the [`GenericImage`].
    pub fn get_image(&self) -> &DynamicImageData<'a> {
        &self.image
    }

    /// Get the contained metadata as a slice of [`GenericLineItem`]s.
    ///
    /// # Returns
    /// A slice of [`GenericLineItem`]s containing the metadata.
    pub fn get_metadata(&self) -> &[GenericLineItem] {
        &self.metadata
    }

    /// Get a specific metadata value by name.
    ///
    /// Returns the first metadata value with the given name.
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value.
    pub fn get_key(&self, name: &str) -> Option<&GenericLineItem> {
        name_check(name).ok()?;
        self.metadata.iter().find(|x| x.name() == name)
    }
}

impl<'a: 'b, 'b> Debayer<'a, 'b> for GenericImage<'b> {
    fn debayer(&'b self, method: DemosaicMethod) -> Result<Self, BayerError> {
        let img = self.image.debayer(method)?;
        let meta = self.metadata.clone();
        Ok(GenericImage {
            metadata: meta,
            image: img,
        })
    }
}

impl<'a: 'b, 'b> GenericImage<'a> {
    /// Apply a function to the image data.
    ///
    /// This function allows replacing the underlying image data with a new image data.
    ///
    /// # Arguments
    /// - `f`: The function to apply to the image data.
    ///   The function must take a [`DynamicImageData`] and return a [`DynamicImageData`].
    pub fn operate<F>(self, f: F) -> Result<GenericImage<'b>, &'static str>
    where
        F: FnOnce(DynamicImageData<'b>) -> Result<DynamicImageData<'b>, &'static str>,
    {
        let img = f(self.image)?;
        Ok(GenericImage {
            metadata: self.metadata,
            image: img,
        })
    }
}

macro_rules! impl_functions {
    ($(#[$attr:meta])* => $name: ident, $type: ty) => {
        $(#[$attr])*
        pub fn $name(&self) -> $type {
            self.0.$name()
        }
    };
}

impl GenericLineItem {
    impl_functions! {
        /// Get the name of the metadata value.
        => name, &str
    }
    impl_functions! {
        /// Get the comment of the metadata value.
        => get_comment, Option<&str>
    }
    impl_functions! {
        /// Get the `u8` metadata value.
        => get_value_u8, Option<u8>
    }
    impl_functions! {
        /// Get the `u16` metadata value.
        => get_value_u16, Option<u16>
    }
    impl_functions! {
        /// Get the `u32` metadata value.
        => get_value_u32, Option<u32>
    }
    impl_functions! {
        /// Get the `u64` metadata value.
        => get_value_u64, Option<u64>
    }
    impl_functions! {
        /// Get the `i8` metadata value.
        => get_value_i8, Option<i8>
    }
    impl_functions! {
        /// Get the `i16` metadata value.
        => get_value_i16, Option<i16>
    }
    impl_functions! {
        /// Get the `i32` metadata value.
        => get_value_i32, Option<i32>
    }
    impl_functions! {
        /// Get the `i64` metadata value.
        => get_value_i64, Option<i64>
    }
    impl_functions! {
        /// Get the `f32` metadata value.
        => get_value_f32, Option<f32>
    }
    impl_functions! {
        /// Get the `f64` metadata value.
        => get_value_f64, Option<f64>
    }
    impl_functions! {
        /// Get the `std::time::Duration` metadata value.
        => get_value_duration, Option<Duration>
    }
    impl_functions! {
        /// Get the `std::time::SystemTime` metadata value.
        => get_value_systemtime, Option<SystemTime>
    }
    impl_functions! {
        /// Get the `String` metadata value.
        => get_value_string, Option<&str>
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PrvLineItem<T: InsertValue> {
    pub(crate) name: String,
    pub(crate) value: T,
    pub(crate) comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
/// A type-erasure enum to hold a metadata item.
pub(crate) enum PrvGenLineItem {
    /// A `u8` metadata value.
    U8(PrvLineItem<u8>),
    /// A `u16` metadata value.
    U16(PrvLineItem<u16>),
    /// A `u32` metadata value.
    U32(PrvLineItem<u32>),
    /// A `u64` metadata value.
    U64(PrvLineItem<u64>),
    /// An `i8` metadata value.
    I8(PrvLineItem<i8>),
    /// An `i16` metadata value.
    I16(PrvLineItem<i16>),
    /// An `i32` metadata value.
    I32(PrvLineItem<i32>),
    /// An `i64` metadata value.
    I64(PrvLineItem<i64>),
    /// An `f32` metadata value.
    F32(PrvLineItem<f32>),
    /// An `f64` metadata value.
    F64(PrvLineItem<f64>),
    /// A `std::time::Duration` metadata value.
    Duration(PrvLineItem<Duration>),
    /// A `std::time::SystemTime` metadata value.
    SystemTime(PrvLineItem<SystemTime>),
    /// A `String` metadata value.
    String(PrvLineItem<String>),
}

/// Trait to insert a metadata value into a [`GenericImage`].
pub trait InsertValue {
    /// Insert a metadata value by name.
    fn insert_key(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str>;

    /// Replace a metadata value by name.
    fn replace(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str>;
}

macro_rules! insert_value_impl {
    ($t:ty, $datatype:expr) => {
        impl InsertValue for $t {
            /// Insert a metadata value.
            fn insert_key(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = PrvLineItem {
                    name: name.to_string().to_uppercase(),
                    value,
                    comment: None,
                };
                f.metadata.push(GenericLineItem($datatype(line)));
                Ok(())
            }

            /// Replace a metadata value.
            fn replace(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
                name_check(name)?;
                let mut not_found = true;
                f.metadata.retain(|x| {
                    let found = x.name() != name;
                    not_found &= found;
                    found
                });
                if not_found {
                    Err("Key not found")
                } else {
                    Self::insert_key(f, name, value)
                }
            }
        }

        impl InsertValue for ($t, &str) {
            /// Insert a metadata value with a comment.
            fn insert_key(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = PrvLineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.0,
                    comment: Some(value.1.to_owned()),
                };
                f.metadata.push(GenericLineItem($datatype(line)));
                Ok(())
            }

            /// Replace a metadata value with a comment.
            fn replace(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
                name_check(name)?;
                let mut not_found = true;
                f.metadata.retain(|x| {
                    let found = x.name() != name;
                    not_found &= found;
                    found
                });
                if not_found {
                    Err("Key not found")
                } else {
                    Self::insert_key(f, name, value)
                }
            }
        }
    };
}

fn name_check(name: &str) -> Result<(), &'static str> {
    if name.is_empty() {
        Err("Key cannot be empty")
    } else if name.len() > 80 {
        Err("Key cannot be longer than 80 characters")
    } else {
        Ok(())
    }
}

fn comment_check(comment: &str) -> Result<(), &'static str> {
    if comment.is_empty() {
        Err("Comment cannot be empty")
    } else if comment.len() > 4096 {
        Err("Comment cannot be longer than 4096 characters")
    } else {
        Ok(())
    }
}

#[allow(dead_code)]
fn str_value_check(value: &str) -> Result<(), &'static str> {
    if value.is_empty() {
        Err("Value cannot be empty")
    } else if value.len() > 4096 {
        Err("Value cannot be longer than 4096 characters")
    } else {
        Ok(())
    }
}

insert_value_impl!(u8, PrvGenLineItem::U8);
insert_value_impl!(u16, PrvGenLineItem::U16);
insert_value_impl!(u32, PrvGenLineItem::U32);
insert_value_impl!(u64, PrvGenLineItem::U64);
insert_value_impl!(i8, PrvGenLineItem::I8);
insert_value_impl!(i16, PrvGenLineItem::I16);
insert_value_impl!(i32, PrvGenLineItem::I32);
insert_value_impl!(i64, PrvGenLineItem::I64);
insert_value_impl!(f32, PrvGenLineItem::F32);
insert_value_impl!(f64, PrvGenLineItem::F64);
insert_value_impl!(String, PrvGenLineItem::String);
insert_value_impl!(Duration, PrvGenLineItem::Duration);
insert_value_impl!(SystemTime, PrvGenLineItem::SystemTime);

impl InsertValue for &str {
    fn insert_key(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value)?;
        let line = PrvLineItem {
            name: name.to_string().to_uppercase(),
            value: value.to_owned(),
            comment: None,
        };
        f.metadata
            .push(GenericLineItem(PrvGenLineItem::String(line)));
        Ok(())
    }

    fn replace(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        let mut not_found = true;
        f.metadata.retain(|x| {
            let found = x.name() != name;
            not_found &= found;
            found
        });
        if not_found {
            Err("Key not found")
        } else {
            Self::insert_key(f, name, value)
        }
    }
}

impl InsertValue for (&str, &str) {
    fn insert_key(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let line = PrvLineItem {
            name: name.to_string().to_uppercase(),
            value: value.0.to_owned(),
            comment: Some(value.1.to_owned()),
        };
        f.metadata
            .push(GenericLineItem(PrvGenLineItem::String(line)));
        Ok(())
    }

    fn replace(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        let mut not_found = true;
        f.metadata.retain(|x| {
            let found = x.name() != name;
            not_found &= found;
            found
        });
        if not_found {
            Err("Key not found")
        } else {
            Self::insert_key(f, name, value)
        }
    }
}

impl PrvGenLineItem {
    /// Get the name of the metadata value.
    pub fn name(&self) -> &str {
        match self {
            PrvGenLineItem::U8(x) => &x.name,
            PrvGenLineItem::U16(x) => &x.name,
            PrvGenLineItem::U32(x) => &x.name,
            PrvGenLineItem::U64(x) => &x.name,
            PrvGenLineItem::I8(x) => &x.name,
            PrvGenLineItem::I16(x) => &x.name,
            PrvGenLineItem::I32(x) => &x.name,
            PrvGenLineItem::I64(x) => &x.name,
            PrvGenLineItem::F32(x) => &x.name,
            PrvGenLineItem::F64(x) => &x.name,
            PrvGenLineItem::Duration(x) => &x.name,
            PrvGenLineItem::SystemTime(x) => &x.name,
            PrvGenLineItem::String(x) => &x.name,
        }
    }

    /// Get the comment of the metadata value.
    pub fn get_comment(&self) -> Option<&str> {
        match self {
            PrvGenLineItem::U8(x) => x.comment.as_deref(),
            PrvGenLineItem::U16(x) => x.comment.as_deref(),
            PrvGenLineItem::U32(x) => x.comment.as_deref(),
            PrvGenLineItem::U64(x) => x.comment.as_deref(),
            PrvGenLineItem::I8(x) => x.comment.as_deref(),
            PrvGenLineItem::I16(x) => x.comment.as_deref(),
            PrvGenLineItem::I32(x) => x.comment.as_deref(),
            PrvGenLineItem::I64(x) => x.comment.as_deref(),
            PrvGenLineItem::F32(x) => x.comment.as_deref(),
            PrvGenLineItem::F64(x) => x.comment.as_deref(),
            PrvGenLineItem::Duration(x) => x.comment.as_deref(),
            PrvGenLineItem::SystemTime(x) => x.comment.as_deref(),
            PrvGenLineItem::String(x) => x.comment.as_deref(),
        }
    }

    /// Get the `u8` metadata value.
    pub fn get_value_u8(&self) -> Option<u8> {
        match self {
            PrvGenLineItem::U8(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u16` metadata value.
    pub fn get_value_u16(&self) -> Option<u16> {
        match self {
            PrvGenLineItem::U16(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u32` metadata value.
    pub fn get_value_u32(&self) -> Option<u32> {
        match self {
            PrvGenLineItem::U32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u64` metadata value.
    pub fn get_value_u64(&self) -> Option<u64> {
        match self {
            PrvGenLineItem::U64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i8` metadata value.
    pub fn get_value_i8(&self) -> Option<i8> {
        match self {
            PrvGenLineItem::I8(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i16` metadata value.
    pub fn get_value_i16(&self) -> Option<i16> {
        match self {
            PrvGenLineItem::I16(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i32` metadata value.
    pub fn get_value_i32(&self) -> Option<i32> {
        match self {
            PrvGenLineItem::I32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i64` metadata value.
    pub fn get_value_i64(&self) -> Option<i64> {
        match self {
            PrvGenLineItem::I64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `f32` metadata value.
    pub fn get_value_f32(&self) -> Option<f32> {
        match self {
            PrvGenLineItem::F32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `f64` metadata value.
    pub fn get_value_f64(&self) -> Option<f64> {
        match self {
            PrvGenLineItem::F64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `std::time::Duration` metadata value.
    pub fn get_value_duration(&self) -> Option<Duration> {
        match self {
            PrvGenLineItem::Duration(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `std::time::SystemTime` metadata value.
    pub fn get_value_systemtime(&self) -> Option<SystemTime> {
        match self {
            PrvGenLineItem::SystemTime(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `String` metadata value.
    pub fn get_value_string(&self) -> Option<&str> {
        match self {
            PrvGenLineItem::String(x) => Some(&x.value),
            _ => None,
        }
    }
}

mod test {
    #[test]
    fn test_operate() {
        use crate::{ColorSpace, DynamicImageData, GenericImage, ImageData};
        use std::time::SystemTime;

        let data = vec![1u8, 2, 3, 4, 5, 6];
        let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
        let img = DynamicImageData::from(img);
        let mut img = GenericImage::new(SystemTime::now(), img);

        img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();

        let img2 = img.clone().operate(|x| Ok(x)).unwrap();
        assert_eq!(img, img2);
    }
}
