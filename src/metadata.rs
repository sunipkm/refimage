use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

#[allow(unused)]
use crate::GenericImage;
use crate::{genericimageowned::GenericImageOwned, genericimageref::GenericImageRef, ColorSpace};

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

/// A type-erased enum to hold a metadata value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GenericValue {
    /// An unsigned 8-bit integer.
    U8(u8),
    /// An unsigned 16-bit integer.
    U16(u16),
    /// An unsigned 32-bit integer.
    U32(u32),
    /// An unsigned 64-bit integer.
    U64(u64),
    /// A signed 8-bit integer.
    I8(i8),
    /// A signed 16-bit integer.
    I16(i16),
    /// A signed 32-bit integer.
    I32(i32),
    /// A signed 64-bit integer.
    I64(i64),
    /// A 32-bit floating point number.
    F32(f32),
    /// A 64-bit floating point number.
    F64(f64),
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

macro_rules! impl_from_genericvalue {
    ($t:ty, $variant:path) => {
        impl From<$t> for GenericValue {
            fn from(value: $t) -> Self {
                $variant(value)
            }
        }
    };
}

impl_from_genericvalue!(u8, GenericValue::U8);
impl_from_genericvalue!(u16, GenericValue::U16);
impl_from_genericvalue!(u32, GenericValue::U32);
impl_from_genericvalue!(u64, GenericValue::U64);
impl_from_genericvalue!(i8, GenericValue::I8);
impl_from_genericvalue!(i16, GenericValue::I16);
impl_from_genericvalue!(i32, GenericValue::I32);
impl_from_genericvalue!(i64, GenericValue::I64);
impl_from_genericvalue!(f32, GenericValue::F32);
impl_from_genericvalue!(f64, GenericValue::F64);
impl_from_genericvalue!(ColorSpace, GenericValue::ColorSpace);
impl_from_genericvalue!(Duration, GenericValue::Duration);
impl_from_genericvalue!(SystemTime, GenericValue::SystemTime);
impl_from_genericvalue!(String, GenericValue::String);

macro_rules! impl_tryinto_genericvalue {
    ($t:ty, $variant:path) => {
        impl TryInto<$t> for GenericValue {
            type Error = String;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    $variant(x) => Ok(x),
                    _ => Err(format!("Invalid type {:?}", self)),
                }
            }
        }
    };
}

impl_tryinto_genericvalue!(u8, GenericValue::U8);
impl_tryinto_genericvalue!(u16, GenericValue::U16);
impl_tryinto_genericvalue!(u32, GenericValue::U32);
impl_tryinto_genericvalue!(u64, GenericValue::U64);
impl_tryinto_genericvalue!(i8, GenericValue::I8);
impl_tryinto_genericvalue!(i16, GenericValue::I16);
impl_tryinto_genericvalue!(i32, GenericValue::I32);
impl_tryinto_genericvalue!(i64, GenericValue::I64);
impl_tryinto_genericvalue!(f32, GenericValue::F32);
impl_tryinto_genericvalue!(f64, GenericValue::F64);
impl_tryinto_genericvalue!(ColorSpace, GenericValue::ColorSpace);
impl_tryinto_genericvalue!(Duration, GenericValue::Duration);
impl_tryinto_genericvalue!(SystemTime, GenericValue::SystemTime);
impl_tryinto_genericvalue!(String, GenericValue::String);

/// Trait to insert a metadata value into a [`GenericImageRef`].
pub trait InsertValue {
    /// Insert a metadata value into a [`GenericImageRef`] by name.
    fn insert_key_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str>;

    /// Insert a metadata value into a [`GenericImageOwned`] by name.
    fn insert_key_go(
        f: &mut GenericImageOwned,
        name: &str,
        value: Self,
    ) -> Result<(), &'static str>;

    /// Replace a metadata value in a [`GenericImageRef`] by name.
    fn replace_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str>;

    /// Replace a metadata value in a [`GenericImageOwned`] by name.
    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str>;
}

macro_rules! insert_value_impl {
    ($t:ty, $datatype:expr) => {
        impl InsertValue for $t {
            fn insert_key_gi(
                f: &mut GenericImageRef,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = GenericLineItem {
                    value: value.into(),
                    comment: None,
                };
                f.metadata.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace_gi(
                f: &mut GenericImageRef,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                f.metadata.remove(name).ok_or("Key not found")?;
                Self::insert_key_gi(f, name, value)
            }

            fn insert_key_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = GenericLineItem {
                    value: value.into(),
                    comment: None,
                };
                f.metadata.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                f.metadata.remove(name).ok_or("Key not found")?;
                Self::insert_key_go(f, name, value)
            }
        }

        impl InsertValue for ($t, &str) {
            fn insert_key_gi(
                f: &mut GenericImageRef,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = GenericLineItem {
                    value: value.0.into(),
                    comment: Some(value.1.to_owned()),
                };
                f.metadata.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace_gi(
                f: &mut GenericImageRef,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                f.metadata.remove(name).ok_or("Key not found")?;
                Self::insert_key_gi(f, name, value)
            }

            fn insert_key_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = GenericLineItem {
                    value: value.0.into(),
                    comment: Some(value.1.to_owned()),
                };
                f.metadata.insert(name.to_uppercase(), line);
                Ok(())
            }

            fn replace_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                f.metadata.remove(name).ok_or("Key not found")?;
                Self::insert_key_go(f, name, value)
            }
        }
    };
}

pub(crate) fn name_check(name: &str) -> Result<(), &'static str> {
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
insert_value_impl!(ColorSpace, PrvGenLineItem::ColorSpace);
insert_value_impl!(String, PrvGenLineItem::String);
insert_value_impl!(Duration, PrvGenLineItem::Duration);
insert_value_impl!(SystemTime, PrvGenLineItem::SystemTime);

impl InsertValue for &str {
    fn insert_key_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value)?;
        let line = GenericLineItem {
            value: value.to_owned().into(),
            comment: None,
        };
        f.metadata.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        f.metadata.remove(name).ok_or("Key not found")?;
        Self::insert_key_gi(f, name, value)
    }

    fn insert_key_go(
        f: &mut GenericImageOwned,
        name: &str,
        value: Self,
    ) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value)?;
        let line = GenericLineItem {
            value: value.to_owned().into(),
            comment: None,
        };
        f.metadata.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        f.metadata.remove(name).ok_or("Key not found")?;
        Self::insert_key_go(f, name, value)
    }
}

impl InsertValue for (&str, &str) {
    fn insert_key_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let line = GenericLineItem {
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.metadata.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace_gi(f: &mut GenericImageRef, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        f.metadata.remove(name).ok_or("Key not found")?;
        Self::insert_key_gi(f, name, value)
    }

    fn insert_key_go(
        f: &mut GenericImageOwned,
        name: &str,
        value: Self,
    ) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let line = GenericLineItem {
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.metadata.insert(name.to_uppercase(), line);
        Ok(())
    }

    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        f.metadata.remove(name).ok_or("Key not found")?;
        Self::insert_key_go(f, name, value)
    }
}

impl GenericValue {
    /// Get the `u8` metadata value.
    pub fn get_value_u8(&self) -> Option<u8> {
        // The clone here is a trivial copy, so it's fine.
        self.clone().try_into().ok()
    }

    /// Get the `u16` metadata value.
    pub fn get_value_u16(&self) -> Option<u16> {
        self.clone().try_into().ok()
    }

    /// Get the `u32` metadata value.
    pub fn get_value_u32(&self) -> Option<u32> {
        self.clone().try_into().ok()
    }

    /// Get the `u64` metadata value.
    pub fn get_value_u64(&self) -> Option<u64> {
        self.clone().try_into().ok()
    }

    /// Get the `i8` metadata value.
    pub fn get_value_i8(&self) -> Option<i8> {
        self.clone().try_into().ok()
    }

    /// Get the `i16` metadata value.
    pub fn get_value_i16(&self) -> Option<i16> {
        self.clone().try_into().ok()
    }

    /// Get the `i32` metadata value.
    pub fn get_value_i32(&self) -> Option<i32> {
        self.clone().try_into().ok()
    }

    /// Get the `i64` metadata value.
    pub fn get_value_i64(&self) -> Option<i64> {
        self.clone().try_into().ok()
    }

    /// Get the `f32` metadata value.
    pub fn get_value_f32(&self) -> Option<f32> {
        self.clone().try_into().ok()
    }

    /// Get the `f64` metadata value.
    pub fn get_value_f64(&self) -> Option<f64> {
        self.clone().try_into().ok()
    }

    /// Get the `std::time::Duration` metadata value.
    pub fn get_value_duration(&self) -> Option<Duration> {
        self.clone().try_into().ok()
    }

    /// Get the `std::time::SystemTime` metadata value.
    pub fn get_value_systemtime(&self) -> Option<SystemTime> {
        self.clone().try_into().ok()
    }

    /// Get the `String` metadata value.
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
        use crate::Debayer;
        use crate::{BayerPattern, DynamicImageOwned, GenericImageOwned, ImageOwned, ImageProps};
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
                let x = x.debayer(crate::DemosaicMethod::Linear).unwrap();
                Ok(x)
            })
            .unwrap();
        let img3 = img.operate(|x| Ok(x.clone())).unwrap();
        assert_eq!(img, img3);
        assert_eq!(img.get_metadata(), img2.get_metadata());
        assert_eq!(img.get_image().width(), img2.get_image().width());
        assert_eq!(img.get_image().height(), img2.get_image().height());
        assert_eq!(img.get_image().channels() * 3, img2.get_image().channels());
    }
}
