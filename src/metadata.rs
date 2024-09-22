use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::{BayerError, ColorSpace, Debayer, DemosaicMethod, DynamicImageData, DynamicImageOwned};

/// Key for the timestamp metadata.
/// This key is inserted by default when creating a new [`GenericImage`].
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
/// This struct is not meant to be used directly. Instead, use the [`crate::GenericImage`]
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
/// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] is split
///   and stored as two consecutive metadata items, with the same key, split into
///   seconds ([`u64`]) and microseconds ([`u32`]).
///
pub struct GenericLineItem {
    pub(crate) name: String,
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

/// A serializable, generic image with metadata, backed by [`DynamicImageData`].
///
/// This struct holds an image with associated metadata. The metadata is stored as a vector of
/// [`GenericLineItem`] structs. The image data is stored as a [`DynamicImageData`].
///
/// # Note
/// - Alpha channels are not trivially supported. They can be added by using a custom
///   color space.
/// - Internally [`GenericImage`] and [`GenericImageOwned`] serialize to the same
///   representation, and can be deserialized into each other.
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

/// A serializable, generic image with metadata, backed by [`DynamicImageOwned`].
///
/// The image data is backed either by owned data, or a slice.
///
/// This struct holds an image with associated metadata. The metadata is stored as a vector of
/// [`GenericLineItem`] structs. The image data is stored as a [`DynamicImageOwned`].
///
/// /// # Note
/// - Alpha channels are not trivially supported. They can be added by using a custom
///   color space.
/// - Internally [`GenericImage`] and [`GenericImageOwned`] serialize to the same
///   representation, and can be deserialized into each other.
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
pub struct GenericImageOwned {
    metadata: Vec<GenericLineItem>,
    image: DynamicImageOwned,
}

impl<'a> GenericImage<'a> {
    /// Create a new [`GenericImage`] with metadata.
    ///
    /// # Arguments
    /// - `tstamp`: The timestamp of the image.
    /// - `image`: The image data, of type [`DynamicImageData`].
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
        let metadata = vec![GenericLineItem {
            name: TIMESTAMP_KEY.to_string(),
            value: tstamp.into(),
            comment: Some("Timestamp of the image".to_owned()),
        }];
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
    /// - [`ColorSpace`]
    /// - [`std::time::Duration`] | [`std::time::SystemTime`]
    /// - [`String`] | [`&str`]
    ///
    /// # Note
    /// - The metadata key is case-insensitive and is stored as an uppercase string.
    /// - Re-inserting a timestamp key will return an error.
    /// - When saving to a FITS file, the metadata comment may be truncated.
    /// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] is split
    ///   and stored as two consecutive metadata items, with the same key, split into
    ///   seconds ([`u64`]) and microseconds ([`u32`]).
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        if name.to_uppercase() == TIMESTAMP_KEY {
            return Err("Cannot re-insert timestamp key");
        }
        T::insert_key_gi(self, name, value)
    }

    /// Remove a metadata value from the [`GenericImage`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value to remove.
    ///
    /// # Returns
    /// - `Ok(())` if the key was removed successfully.
    /// - `Err("Can not remove timestamp key")` if the key is the timestamp key.
    /// - `Err("Key not found")` if the key was not found.
    /// - `Err("Key cannot be empty")` if the key is an empty string.
    /// - `Err("Key cannot be longer than 80 characters")` if the key is longer than 80 characters.
    pub fn remove_key(&mut self, name: &str) -> Result<(), &'static str> {
        if name.to_uppercase() == TIMESTAMP_KEY {
            return Err("Cannot remove timestamp key");
        }
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
        T::replace_gi(self, name, value)
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

    /// Convert the image to a [`GenericImageOwned`] with [`u8`] pixel type.
    ///
    /// Note: This operation is parallelized if the `rayon` feature is enabled.
    pub fn into_u8(self) -> GenericImageOwned {
        let img = self.image.into_u8();
        GenericImageOwned {
            metadata: self.metadata,
            image: img,
        }
    }
}

impl GenericImageOwned {
    /// Create a new [`GenericImageOwned`] with metadata.
    ///
    /// # Arguments
    /// - `tstamp`: The timestamp of the image.
    /// - `image`: The image data, of type [`DynamicImageOwned`].
    ///
    /// # Example
    /// ```
    /// use refimage::{ImageOwned, DynamicImageOwned, GenericImageOwned, ColorSpace};
    /// use std::time::SystemTime;
    /// let data = vec![1u8, 2, 3, 4, 5, 6];
    /// let img = ImageOwned::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
    /// let img = DynamicImageOwned::from(img);
    /// let mut img = GenericImageOwned::new(std::time::SystemTime::now(), img);
    ///
    /// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
    /// ```
    pub fn new(tstamp: SystemTime, image: DynamicImageOwned) -> Self {
        let metadata = vec![GenericLineItem {
            name: TIMESTAMP_KEY.to_string(),
            value: tstamp.into(),
            comment: Some("Timestamp of the image".to_owned()),
        }];
        Self { metadata, image }
    }

    /// Insert a metadata value into the [`GenericImageOwned`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value. The name must be non-empty and less than 80 characters.
    /// - `value`: The value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or `std::time::SystemTime` or a tuple of a primitive type and a comment ().
    /// # Valid Types
    /// The valid types for the metadata value are:
    /// - [`u8`] | [`u16`] | [`u32`] | [`u64`]
    /// - [`i8`] | [`i16`] | [`i32`] | [`i64`]
    /// - [`f32`] | [`f64`]
    /// - [`ColorSpace`]
    /// - [`std::time::Duration`] | [`std::time::SystemTime`]
    /// - [`String`] | [`&str`]
    ///
    /// # Note
    /// - The metadata key is case-insensitive and is stored as an uppercase string.
    /// - Re-inserting a timestamp key will return an error.
    /// - When saving to a FITS file, the metadata comment may be truncated.
    /// - Metadata of type [`std::time::Duration`] or [`std::time::SystemTime`] is split
    ///   and stored as two consecutive metadata items, with the same key, split into
    ///   seconds ([`u64`]) and microseconds ([`u32`]).
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        if name.to_uppercase() == TIMESTAMP_KEY {
            return Err("Cannot re-insert timestamp key");
        }
        T::insert_key_go(self, name, value)
    }

    /// Remove a metadata value from the [`GenericImageOwned`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value to remove.
    ///
    /// # Returns
    /// - `Ok(())` if the key was removed successfully.
    /// - `Err("Can not remove timestamp key")` if the key is the timestamp key.
    /// - `Err("Key not found")` if the key was not found.
    /// - `Err("Key cannot be empty")` if the key is an empty string.
    /// - `Err("Key cannot be longer than 80 characters")` if the key is longer than 80 characters.
    pub fn remove_key(&mut self, name: &str) -> Result<(), &'static str> {
        if name.to_uppercase() == TIMESTAMP_KEY {
            return Err("Cannot remove timestamp key");
        }
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

    /// Replace a metadata value in the [`GenericImageOwned`].
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
        T::replace_go(self, name, value)
    }

    /// Get the underlying [`DynamicImageOwned`].
    ///
    /// # Returns
    /// The underlying [`DynamicImageOwned`] of the [`GenericImageOwned`].
    pub fn get_image(&self) -> &DynamicImageOwned {
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

    /// Convert the image to a [`GenericImageOwned`] with [`u8`] pixel type.
    ///
    /// Note: This operation is parallelized if the `rayon` feature is enabled.
    pub fn into_u8(self) -> GenericImageOwned {
        let img = self.image.into_u8();
        GenericImageOwned {
            metadata: self.metadata,
            image: img,
        }
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

impl<'a: 'b, 'b> Debayer<'a, 'b> for GenericImageOwned {
    fn debayer(&self, method: DemosaicMethod) -> Result<Self, BayerError> {
        let img = self.image.debayer(method)?;
        let meta = self.metadata.clone();
        Ok(Self {
            metadata: meta,
            image: img,
        })
    }
}

impl<'a: 'b, 'b> GenericImage<'a> {
    /// Apply a function to the image data.
    ///
    /// This function copies the metadata of the current image, and replaces the underlying
    /// image data with the result of the function.
    ///
    /// # Arguments
    /// - `f`: The function to apply to the image data.
    ///   The function must take a reference to an [`DynamicImageData`] and return a [`DynamicImageData`].
    pub fn operate<F>(&'a self, f: F) -> Result<GenericImage<'b>, &'static str>
    where
        F: FnOnce(&'a DynamicImageData<'a>) -> Result<DynamicImageData<'b>, &'static str>,
    {
        let img = f(&(self.image))?;
        Ok(GenericImage {
            metadata: self.metadata.clone(),
            image: img,
        })
    }

    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma(&'a self) -> Result<GenericImage<'b>, &'static str> {
        let img = self.image.into_luma()?;
        Ok(GenericImage {
            metadata: self.metadata.clone(),
            image: img,
        })
    }

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion. The number of weights must match
    ///   the number of channels in the image.
    ///
    /// # Errors
    /// - If the number of weights does not match the number of channels in the image.
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_custom(&'a self, coeffs: &[f64]) -> Result<GenericImage<'b>, &'static str> {
        let img = self.image.into_luma_custom(coeffs)?;
        Ok(GenericImage {
            metadata: self.metadata.clone(),
            image: img,
        })
    }
}

impl GenericImageOwned {
    /// Apply a function to the image data.
    ///
    /// This function copies the metadata of the current image, and replaces the underlying
    /// image data with the result of the function.
    ///
    /// # Arguments
    /// - `f`: The function to apply to the image data.
    ///   The function must take a reference to an [`DynamicImageData`] and return a [`DynamicImageData`].
    pub fn operate<F>(&self, f: F) -> Result<Self, &'static str>
    where
        F: FnOnce(&DynamicImageOwned) -> Result<DynamicImageOwned, &'static str>,
    {
        let img = f(&(self.image))?;
        Ok(GenericImageOwned {
            metadata: self.metadata.clone(),
            image: img,
        })
    }

    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma(&self) -> Result<Self, &'static str> {
        let img = self.image.into_luma()?;
        Ok(Self {
            metadata: self.metadata.clone(),
            image: img,
        })
    }

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion. The number of weights must match
    ///   the number of channels in the image.
    ///
    /// # Errors
    /// - If the number of weights does not match the number of channels in the image.
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_custom(&self, coeffs: &[f64]) -> Result<Self, &'static str> {
        let img = self.image.into_luma_custom(coeffs)?;
        Ok(Self {
            metadata: self.metadata.clone(),
            image: img,
        })
    }
}

impl<'a> From<GenericImage<'a>> for GenericImageOwned {
    fn from(img: GenericImage<'a>) -> Self {
        Self {
            metadata: img.metadata,
            image: img.image.into(),
        }
    }
}

impl GenericLineItem {
    /// Get the name of the metadata value.
    pub fn name(&self) -> &str {
        &self.name
    }

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

/// Trait to insert a metadata value into a [`GenericImage`].
pub trait InsertValue {
    /// Insert a metadata value into a [`GenericImage`] by name.
    fn insert_key_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str>;

    /// Insert a metadata value into a [`GenericImageOwned`] by name.
    fn insert_key_go(
        f: &mut GenericImageOwned,
        name: &str,
        value: Self,
    ) -> Result<(), &'static str>;

    /// Replace a metadata value in a [`GenericImage`] by name.
    fn replace_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str>;

    /// Replace a metadata value in a [`GenericImageOwned`] by name.
    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str>;
}

macro_rules! insert_value_impl {
    ($t:ty, $datatype:expr) => {
        impl InsertValue for $t {
            fn insert_key_gi(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = GenericLineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.into(),
                    comment: None,
                };
                f.metadata.push(line);
                Ok(())
            }

            fn replace_gi(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
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
                    Self::insert_key_gi(f, name, value)
                }
            }

            fn insert_key_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = GenericLineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.into(),
                    comment: None,
                };
                f.metadata.push(line);
                Ok(())
            }

            fn replace_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
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
                    Self::insert_key_go(f, name, value)
                }
            }
        }

        impl InsertValue for ($t, &str) {
            fn insert_key_gi(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                comment_check(value.1)?;
                let line = GenericLineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.0.into(),
                    comment: Some(value.1.to_owned()),
                };
                f.metadata.push(line);
                Ok(())
            }

            fn replace_gi(
                f: &mut GenericImage,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
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
                    Self::insert_key_gi(f, name, value)
                }
            }

            fn insert_key_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
                name_check(name)?;
                let line = GenericLineItem {
                    name: name.to_string().to_uppercase(),
                    value: value.0.into(),
                    comment: None,
                };
                f.metadata.push(line);
                Ok(())
            }

            fn replace_go(
                f: &mut GenericImageOwned,
                name: &str,
                value: Self,
            ) -> Result<(), &'static str> {
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
                    Self::insert_key_go(f, name, value)
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
    fn insert_key_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value)?;
        let line = GenericLineItem {
            name: name.to_string().to_uppercase(),
            value: value.to_owned().into(),
            comment: None,
        };
        f.metadata.push(line);
        Ok(())
    }

    fn replace_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
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
            Self::insert_key_gi(f, name, value)
        }
    }

    fn insert_key_go(
        f: &mut GenericImageOwned,
        name: &str,
        value: Self,
    ) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value)?;
        let line = GenericLineItem {
            name: name.to_string().to_uppercase(),
            value: value.to_owned().into(),
            comment: None,
        };
        f.metadata.push(line);
        Ok(())
    }

    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str> {
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
            Self::insert_key_go(f, name, value)
        }
    }
}

impl InsertValue for (&str, &str) {
    fn insert_key_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
        name_check(name)?;
        str_value_check(value.0)?;
        comment_check(value.1)?;
        let line = GenericLineItem {
            name: name.to_string().to_uppercase(),
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.metadata.push(line);
        Ok(())
    }

    fn replace_gi(f: &mut GenericImage, name: &str, value: Self) -> Result<(), &'static str> {
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
            Self::insert_key_gi(f, name, value)
        }
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
            name: name.to_string().to_uppercase(),
            value: value.0.to_owned().into(),
            comment: Some(value.1.to_owned()),
        };
        f.metadata.push(line);
        Ok(())
    }

    fn replace_go(f: &mut GenericImageOwned, name: &str, value: Self) -> Result<(), &'static str> {
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
            Self::insert_key_go(f, name, value)
        }
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
    fn test_operate_generic() {
        use crate::Debayer;
        use crate::{BayerPattern, DynamicImageData, GenericImage, ImageData};
        use std::time::SystemTime;

        let data = vec![0u8; 256];
        let img = ImageData::from_owned(data, 16, 16, BayerPattern::Grbg.into()).unwrap();
        let img = DynamicImageData::from(img);
        let mut img = GenericImage::new(SystemTime::now(), img);

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

    #[test]
    fn test_operate_owned() {
        use crate::Debayer;
        use crate::{BayerPattern, DynamicImageOwned, GenericImageOwned, ImageOwned};
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
