use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::DynamicImageData;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Holds a metadata line item.
pub struct LineItem<T: InsertValue> {
    name: String,
    value: T,
    comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// A type-erasure enum to hold a metadata value.
pub enum GenericLineItem {
    /// A `u8` metadata value.
    U8(LineItem<u8>),
    /// A `u16` metadata value.
    U16(LineItem<u16>),
    /// A `u32` metadata value.
    U32(LineItem<u32>),
    /// A `u64` metadata value.
    U64(LineItem<u64>),
    /// An `i8` metadata value.
    I8(LineItem<i8>),
    /// An `i16` metadata value.
    I16(LineItem<i16>),
    /// An `i32` metadata value.
    I32(LineItem<i32>),
    /// An `i64` metadata value.
    I64(LineItem<i64>),
    /// An `f32` metadata value.
    F32(LineItem<f32>),
    /// An `f64` metadata value.
    F64(LineItem<f64>),
    /// A `std::time::Duration` metadata value.
    Duration(LineItem<Duration>),
    /// A `std::time::SystemTime` metadata value.
    SystemTime(LineItem<SystemTime>),
    /// A `String` metadata value.
    String(LineItem<String>),
}

/// Holds an image, with associated metadata.
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
    pub fn new(tstamp: SystemTime, image: DynamicImageData<'a>) -> Self {
        let metadata = vec![GenericLineItem::SystemTime(LineItem {
            name: "TSTAMP".to_string(),
            value: tstamp,
            comment: Some("Timestamp of the image".to_owned()),
        })];
        Self { metadata, image }
    }

    /// Insert a metadata value into the `GenericImage`.
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value. The name must be non-empty and less than 80 characters.
    /// - `value`: The value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or `std::time::SystemTime` or a tuple of a primitive type and a comment ().
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        T::insert_key(self, name, value)
    }

    /// Remove a metadata value from the `GenericImage`.
    pub fn remove_key(&mut self, name: &str) -> Result<(), &'static str> {
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

    /// Replace a metadata value in the `GenericImage`.
    pub fn replace_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<(), &'static str> {
        T::replace(self, name, value)
    }

    /// Get the image data.
    pub fn get_image(&self) -> &DynamicImageData<'a> {
        &self.image
    }

    /// Get the metadata
    pub fn get_metadata(&self) -> &[GenericLineItem] {
        &self.metadata
    }
}

/// Trait to insert a metadata value into a `GenericLineItem`.
pub trait InsertValue {
    /// Insert a metadata value by name.
    fn insert_key(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str>;

    /// Remove a metadata value by name.
    fn remove(f: &mut GenericImage, name: &str) -> Result<(), &'static str> {
        let mut not_found = true;
        f.metadata.retain(|x| {
            let found = x.name() != name;
            not_found &= found;
            found
        });
        if not_found {
            Err("Key not found")
        } else {
            Ok(())
        }
    }

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
                let line = LineItem {
                    name: name.to_string().to_uppercase(),
                    value,
                    comment: None,
                };
                f.metadata.push($datatype(line));
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
                let line = LineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.0,
                    comment: Some(value.1.to_owned()),
                };
                f.metadata.push($datatype(line));
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

insert_value_impl!(u8, GenericLineItem::U8);
insert_value_impl!(u16, GenericLineItem::U16);
insert_value_impl!(u32, GenericLineItem::U32);
insert_value_impl!(u64, GenericLineItem::U64);
insert_value_impl!(i8, GenericLineItem::I8);
insert_value_impl!(i16, GenericLineItem::I16);
insert_value_impl!(i32, GenericLineItem::I32);
insert_value_impl!(i64, GenericLineItem::I64);
insert_value_impl!(f32, GenericLineItem::F32);
insert_value_impl!(f64, GenericLineItem::F64);
insert_value_impl!(String, GenericLineItem::String);
insert_value_impl!(Duration, GenericLineItem::Duration);
insert_value_impl!(SystemTime, GenericLineItem::SystemTime);

impl GenericLineItem {
    /// Get the name of the metadata value.
    pub fn name(&self) -> &str {
        match self {
            GenericLineItem::U8(x) => &x.name,
            GenericLineItem::U16(x) => &x.name,
            GenericLineItem::U32(x) => &x.name,
            GenericLineItem::U64(x) => &x.name,
            GenericLineItem::I8(x) => &x.name,
            GenericLineItem::I16(x) => &x.name,
            GenericLineItem::I32(x) => &x.name,
            GenericLineItem::I64(x) => &x.name,
            GenericLineItem::F32(x) => &x.name,
            GenericLineItem::F64(x) => &x.name,
            GenericLineItem::Duration(x) => &x.name,
            GenericLineItem::SystemTime(x) => &x.name,
            GenericLineItem::String(x) => &x.name,
        }
    }

    /// Get the comment of the metadata value.
    pub fn get_comment(&self) -> Option<&str> {
        match self {
            GenericLineItem::U8(x) => x.comment.as_deref(),
            GenericLineItem::U16(x) => x.comment.as_deref(),
            GenericLineItem::U32(x) => x.comment.as_deref(),
            GenericLineItem::U64(x) => x.comment.as_deref(),
            GenericLineItem::I8(x) => x.comment.as_deref(),
            GenericLineItem::I16(x) => x.comment.as_deref(),
            GenericLineItem::I32(x) => x.comment.as_deref(),
            GenericLineItem::I64(x) => x.comment.as_deref(),
            GenericLineItem::F32(x) => x.comment.as_deref(),
            GenericLineItem::F64(x) => x.comment.as_deref(),
            GenericLineItem::Duration(x) => x.comment.as_deref(),
            GenericLineItem::SystemTime(x) => x.comment.as_deref(),
            GenericLineItem::String(x) => x.comment.as_deref(),
        }
    }

    /// Get the `u8` metadata value.
    pub fn get_value_u8(&self) -> Option<u8> {
        match self {
            GenericLineItem::U8(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u16` metadata value.
    pub fn get_value_u16(&self) -> Option<u16> {
        match self {
            GenericLineItem::U16(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u32` metadata value.
    pub fn get_value_u32(&self) -> Option<u32> {
        match self {
            GenericLineItem::U32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `u64` metadata value.
    pub fn get_value_u64(&self) -> Option<u64> {
        match self {
            GenericLineItem::U64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i8` metadata value.
    pub fn get_value_i8(&self) -> Option<i8> {
        match self {
            GenericLineItem::I8(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i16` metadata value.
    pub fn get_value_i16(&self) -> Option<i16> {
        match self {
            GenericLineItem::I16(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i32` metadata value.
    pub fn get_value_i32(&self) -> Option<i32> {
        match self {
            GenericLineItem::I32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `i64` metadata value.
    pub fn get_value_i64(&self) -> Option<i64> {
        match self {
            GenericLineItem::I64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `f32` metadata value.
    pub fn get_value_f32(&self) -> Option<f32> {
        match self {
            GenericLineItem::F32(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `f64` metadata value.
    pub fn get_value_f64(&self) -> Option<f64> {
        match self {
            GenericLineItem::F64(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `std::time::Duration` metadata value.
    pub fn get_value_duration(&self) -> Option<Duration> {
        match self {
            GenericLineItem::Duration(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `std::time::SystemTime` metadata value.
    pub fn get_value_systemtime(&self) -> Option<SystemTime> {
        match self {
            GenericLineItem::SystemTime(x) => Some(x.value),
            _ => None,
        }
    }

    /// Get the `String` metadata value.
    pub fn get_value_string(&self) -> Option<&str> {
        match self {
            GenericLineItem::String(x) => Some(&x.value),
            _ => None,
        }
    }
}