use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::{
    metadata::InsertValue, CalcOptExp, DynamicImageRef, ExposureResult, GenericLineItem,
    ImageProps, Metadata, MetadataError, OptimumExposure, OptimumExposureResult,
};

#[allow(unused_imports)]
use crate::{ColorSpace, GenericImageOwned};

/// A serializable, generic image with metadata, backed by [`DynamicImageRef`].
///
/// This struct holds an image with its [`Metadata`]. The image data is stored as a
/// [`DynamicImageRef`].
///
/// # Note
/// - Internally [`GenericImageRef`] and [`GenericImageOwned`] serialize to the same
///   representation, and can be deserialized into each other.
///
/// # Usage
/// ```
/// use refimage::{ImageRef, GenericImageRef, ColorSpace};
/// use refimage::chrono::{DateTime, Utc};
/// use std::time::Duration;
/// let mut data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
/// // the caller supplies the UTC timestamp — e.g. `Utc::now()` in application code
/// let now: DateTime<Utc> = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
/// let mut img = GenericImageRef::new(now, Duration::from_millis(20), img);
///
/// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
/// ```
#[derive(Debug, PartialEq, Serialize)]
pub struct GenericImageRef<'a> {
    pub(crate) metadata: Metadata,
    #[serde(borrow)]
    pub(crate) image: DynamicImageRef<'a>,
}

impl<'a> GenericImageRef<'a> {
    /// Create a new [`GenericImageRef`] with metadata.
    ///
    /// # Arguments
    /// - `timestamp`: The image creation time, as a UTC [`DateTime`] (in
    ///   application code this is typically `chrono::Utc::now()`).
    /// - `exposure`: The exposure duration (`Duration::ZERO` if not applicable).
    /// - `image`: The image data (anything convertible into [`DynamicImageRef`],
    ///   e.g. an [`ImageRef`](crate::ImageRef)).
    ///
    /// # Example
    /// ```
    /// use refimage::{ImageRef, GenericImageRef, ColorSpace};
    /// use refimage::chrono::DateTime;
    /// use std::time::Duration;
    /// let mut data = vec![1u8, 2, 3, 4, 5, 6];
    /// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
    /// let now = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    /// let mut img = GenericImageRef::new(now, Duration::from_millis(20), img);
    ///
    /// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
    /// ```
    pub fn new(
        timestamp: DateTime<Utc>,
        exposure: Duration,
        image: impl Into<DynamicImageRef<'a>>,
    ) -> Self {
        Self {
            metadata: Metadata::new(timestamp, exposure),
            image: image.into(),
        }
    }

    /// Create a new [`GenericImageRef`] from an existing [`Metadata`] block.
    pub fn with_metadata(metadata: Metadata, image: impl Into<DynamicImageRef<'a>>) -> Self {
        Self {
            metadata,
            image: image.into(),
        }
    }

    /// Get the UTC timestamp of the image.
    pub fn get_timestamp(&self) -> DateTime<Utc> {
        self.metadata.timestamp()
    }

    /// Get the exposure time of the image (`Duration::ZERO` if not applicable).
    pub fn get_exposure(&self) -> Duration {
        self.metadata.exposure()
    }

    /// Set the exposure time of the image.
    pub fn set_exposure(&mut self, exposure: Duration) {
        self.metadata.set_exposure(exposure);
    }

    /// Get the acquisition frame ID of the image, or `None` if unset.
    pub fn get_frame_id(&self) -> Option<u32> {
        self.metadata.frame_id()
    }

    /// Set the acquisition frame ID of the image.
    pub fn set_frame_id(&mut self, frame_id: u32) {
        self.metadata.set_frame_id(frame_id);
    }

    /// Insert a metadata value into the [`GenericImageRef`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value. The name must be non-empty and less than 80 characters.
    /// - `value`: The value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or a UTC `chrono::DateTime` or a tuple of a primitive type and a comment ().
    /// # Valid Types
    /// The valid types for the metadata value are:
    /// - [`u8`] | [`u16`] | [`u32`] | [`u64`]
    /// - [`i8`] | [`i16`] | [`i32`] | [`i64`]
    /// - [`f32`] | [`f64`]
    /// - [`ColorSpace`]
    /// - [`std::time::Duration`] | [`chrono::DateTime<Utc>`](crate::chrono::DateTime)
    /// - [`String`] | [`&str`]
    ///
    /// # Note
    /// - The metadata key is case-insensitive and is stored as an uppercase string.
    /// - The `TIMESTAMP`, `EXPOSURE` and `FRAMEID` keys are reserved (they are typed fields);
    ///   use [`set_exposure`](Self::set_exposure) / [`Metadata::set_timestamp`].
    /// - When saving to a FITS file, the metadata comment may be truncated.
    /// - Metadata of type [`std::time::Duration`] or a UTC `chrono::DateTime` is
    ///   stored as two consecutive metadata items split into seconds and nanoseconds
    ///   (keys suffixed `_S` and `_NS`), plus a base card.
    pub fn insert_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<(), MetadataError> {
        self.metadata.insert(name, value)
    }

    /// Remove a metadata value from the [`GenericImageRef`].
    ///
    /// # Errors
    /// [`MetadataError::ReservedKey`] for `TIMESTAMP` / `EXPOSURE` / `FRAMEID`,
    /// [`MetadataError::KeyNotFound`] if absent, or a key-validation error.
    pub fn remove_key(&mut self, name: &str) -> Result<GenericLineItem, MetadataError> {
        self.metadata.remove(name)
    }

    /// Replace a metadata value in the [`GenericImageRef`].
    ///
    /// # Errors
    /// [`MetadataError::KeyNotFound`] if the key was not present.
    pub fn replace_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<GenericLineItem, MetadataError> {
        self.metadata.replace(name, value)
    }

    /// Get the underlying [`DynamicImageRef`].
    ///
    /// # Returns
    /// The underlying [`DynamicImageRef`] of the [`GenericImageRef`].
    pub fn get_image(&self) -> &DynamicImageRef<'a> {
        &self.image
    }

    /// Get the underlying [`DynamicImageRef`] mutably.
    pub fn get_image_mut(&mut self) -> &mut DynamicImageRef<'a> {
        &mut self.image
    }

    /// Borrow the image's [`Metadata`].
    pub fn get_metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Mutably borrow the image's [`Metadata`].
    pub fn get_metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// Get a specific extra metadata item by name (case-insensitive).
    pub fn get_key(&self, name: &str) -> Option<&GenericLineItem> {
        self.metadata.get(name)
    }
}

impl CalcOptExp for GenericImageRef<'_> {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        self.image.calc_opt_exp(eval, exposure, bin)
    }
}

impl GenericImageRef<'_> {
    /// Optimum exposure and binning, using this image's own recorded
    /// [`exposure`](Self::get_exposure).
    ///
    /// # Errors
    /// [`ExposureError::ZeroExposure`](crate::ExposureError::ZeroExposure) if the
    /// recorded exposure is `Duration::ZERO`, otherwise see
    /// [`ExposureError`](crate::ExposureError).
    pub fn optimum_exposure(
        &mut self,
        eval: &OptimumExposure,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        let exposure = self.get_exposure();
        self.image.calc_opt_exp(eval, exposure, bin)
    }
}

impl ImageProps for GenericImageRef<'_> {
    fn width(&self) -> usize {
        self.image.width()
    }

    fn height(&self) -> usize {
        self.image.height()
    }

    fn channels(&self) -> u8 {
        self.image.channels()
    }

    fn color_space(&self) -> crate::ColorSpace {
        self.image.color_space()
    }

    fn pixel_type(&self) -> crate::PixelType {
        self.image.pixel_type()
    }

    fn len(&self) -> usize {
        self.image.len()
    }

    fn is_empty(&self) -> bool {
        self.image.is_empty()
    }
}

impl GenericImageRef<'_> {
    /// Get the data as a slice of `u8`, regardless of the underlying type.
    pub fn as_raw_u8(&self) -> &[u8] {
        self.image.as_raw_u8()
    }

    /// Get the data as a slice of `u8`, regardless of the underlying type.
    pub fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        self.image.as_raw_u8_checked()
    }

    /// Get the data as a slice of `u8`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_slice_u8(&self) -> Option<&[u8]> {
        self.image.as_slice_u8()
    }

    /// Get the data as a mutable slice of `u8`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        self.image.as_mut_slice_u8()
    }

    /// Get the data as a slice of `u16`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_slice_u16(&self) -> Option<&[u16]> {
        self.image.as_slice_u16()
    }

    /// Get the data as a mutable slice of `u16`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        self.image.as_mut_slice_u16()
    }

    /// Get the data as a slice of `f32`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        self.image.as_slice_f32()
    }

    /// Get the data as a mutable slice of `f32`.
    ///
    /// # Note
    /// The returned slice is not guaranteed to have the correct length.
    /// Use [`GenericImageRef::len`] to get the correct length.
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        self.image.as_mut_slice_f32()
    }
}

mod test {
    #[test]
    fn test_optimum_exposure() {
        use crate::CalcOptExp;
        let opt_exp = crate::OptimumExposureBuilder::default()
            .pixel_exclusion(1)
            .build()
            .unwrap();
        let mut img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let img = crate::ImageRef::new(img.as_mut_slice(), 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let img = crate::DynamicImageRef::from(img);
        let ts = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut img = crate::GenericImageRef::new(ts, std::time::Duration::from_secs(2), img);
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);
    }
}
