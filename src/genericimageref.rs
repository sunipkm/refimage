use std::{
    collections::HashMap,
    time::{Duration, SystemTime},
};

use serde::Serialize;

use crate::{
    genericimageowned::GenericImageOwned,
    metadata::{name_check, InsertValue},
    BayerError, Debayer, DemosaicMethod, DynamicImageRef, GenericLineItem, EXPOSURE_KEY,
    TIMESTAMP_KEY,
};

/// A serializable, generic image with metadata, backed by [`DynamicImageRef`].
///
/// This struct holds an image with associated metadata. The metadata is stored as a vector of
/// [`GenericLineItem`] structs. The image data is stored as a [`DynamicImageRef`].
///
/// # Note
/// - Internally [`GenericImageRef`] and [`GenericImageOwned`] serialize to the same
///   representation, and can be deserialized into each other.
///
/// # Usage
/// ```
/// use refimage::{ImageRef, DynamicImageRef, GenericImageRef, ColorSpace};
/// use std::time::SystemTime;
/// let mut data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
/// let img = DynamicImageRef::from(img);
/// let mut img = GenericImageRef::new(std::time::SystemTime::now(), img);
///
/// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
/// ```
#[derive(Debug, PartialEq, Serialize)]
pub struct GenericImageRef<'a> {
    pub(crate) metadata: HashMap<String, GenericLineItem>,
    #[serde(borrow)]
    pub(crate) image: DynamicImageRef<'a>,
}

impl<'a> GenericImageRef<'a> {
    /// Create a new [`GenericImageRef`] with metadata.
    ///
    /// # Arguments
    /// - `tstamp`: The timestamp of the image.
    /// - `image`: The image data, of type [`DynamicImageRef`].
    ///
    /// # Example
    /// ```
    /// use refimage::{ImageRef, DynamicImageRef, GenericImageRef, ColorSpace};
    /// use std::time::SystemTime;
    /// let mut data = vec![1u8, 2, 3, 4, 5, 6];
    /// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
    /// let img = DynamicImageRef::from(img);
    /// let mut img = GenericImageRef::new(std::time::SystemTime::now(), img);
    ///
    /// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
    /// ```
    pub fn new(tstamp: SystemTime, image: DynamicImageRef<'a>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert(
            TIMESTAMP_KEY.to_string(),
            GenericLineItem {
                value: tstamp.into(),
                comment: Some("Timestamp of the image".to_owned()),
            },
        );
        Self { metadata, image }
    }

    /// Get the timestamp of the image.
    pub fn get_timestamp(&self) -> SystemTime {
        self.metadata
            .get(TIMESTAMP_KEY)
            .and_then(|x| x.get_value().clone().try_into().ok())
            .unwrap() // Safe to unwrap, as the timestamp key is always inserted
    }

    /// Get the exposure time of the image.
    pub fn get_exposure(&self) -> Option<Duration> {
        self.metadata
            .get(EXPOSURE_KEY)
            .and_then(|x| x.get_value().clone().try_into().ok())
    }

    /// Insert a metadata value into the [`GenericImageRef`].
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
    ///   seconds ([`u64`]) and nanoseconds ([`u64`]).
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        if name.to_uppercase() == TIMESTAMP_KEY {
            return Err("Cannot re-insert timestamp key");
        }
        T::insert_key_gi(self, name, value)
    }

    /// Remove a metadata value from the [`GenericImageRef`].
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
        self.metadata.remove(name).ok_or("Key not found")?;
        Ok(())
    }

    /// Replace a metadata value in the [`GenericImageRef`].
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

    /// Get the underlying [`DynamicImageRef`].
    ///
    /// # Returns
    /// The underlying [`DynamicImageRef`] of the [`GenericImageRef`].
    pub fn get_image(&self) -> &DynamicImageRef<'a> {
        &self.image
    }

    /// Get the contained metadata as a slice of [`GenericLineItem`]s.
    ///
    /// # Returns
    /// A slice of [`GenericLineItem`]s containing the metadata.
    pub fn get_metadata(&self) -> &HashMap<String, GenericLineItem> {
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
        self.metadata.get(name)
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

impl<'a: 'b, 'b> Debayer<'a, 'b> for GenericImageRef<'b> {
    type Output = GenericImageOwned;
    fn debayer(&'b self, method: DemosaicMethod) -> Result<Self::Output, BayerError> {
        let img = self.image.debayer(method)?;
        let meta = self.metadata.clone();
        Ok(Self::Output {
            metadata: meta,
            image: img,
        })
    }
}
