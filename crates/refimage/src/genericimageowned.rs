use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    genericimageref::GenericImageRef, metadata::InsertValue, CalcOptExp, DynamicImageOwned,
    ExposureResult, GenericLineItem, ImageProps, Metadata, OptimumExposure, OptimumExposureResult,
    PixelData,
};

#[allow(unused_imports)]
use crate::{ColorSpace, DynamicImageRef};

/// A serializable, generic image with metadata, backed by [`DynamicImageOwned`].
///
/// This struct holds an image with its [`Metadata`]. The image data is stored as a
/// [`DynamicImageOwned`].
///
/// # Note
/// - Internally [`GenericImageRef`] and [`GenericImageOwned`] serialize to the same
///   representation, and [`GenericImageRef`] can be deserialized to [`GenericImageOwned`].
///
/// # Usage
/// ```
/// use refimage::{ImageOwned, GenericImageOwned, ColorSpace};
/// use refimage::chrono::DateTime;
/// use std::time::Duration;
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageOwned::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// let now = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
/// let mut img = GenericImageOwned::new(now, Duration::from_millis(20), img);
///
/// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericImageOwned {
    pub(crate) metadata: Metadata,
    pub(crate) image: DynamicImageOwned,
}

impl GenericImageOwned {
    /// Create a new [`GenericImageOwned`] with metadata.
    ///
    /// # Arguments
    /// - `timestamp`: The image creation time, as a UTC [`DateTime`] (in
    ///   application code this is typically `chrono::Utc::now()`).
    /// - `exposure`: The exposure duration (`Duration::ZERO` if not applicable).
    /// - `image`: The image data (anything convertible into [`DynamicImageOwned`],
    ///   e.g. an [`ImageOwned`](crate::ImageOwned)).
    ///
    /// # Example
    /// ```
    /// use refimage::{ImageOwned, GenericImageOwned, ColorSpace};
    /// use refimage::chrono::DateTime;
    /// use std::time::Duration;
    /// let data = vec![1u8, 2, 3, 4, 5, 6];
    /// let img = ImageOwned::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
    /// let now = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    /// let mut img = GenericImageOwned::new(now, Duration::from_millis(20), img);
    ///
    /// img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
    /// ```
    pub fn new(
        timestamp: DateTime<Utc>,
        exposure: Duration,
        image: impl Into<DynamicImageOwned>,
    ) -> Self {
        Self {
            metadata: Metadata::new(timestamp, exposure),
            image: image.into(),
        }
    }

    /// Create a new [`GenericImageOwned`] from an existing [`Metadata`] block.
    pub fn with_metadata(metadata: Metadata, image: impl Into<DynamicImageOwned>) -> Self {
        Self {
            metadata,
            image: image.into(),
        }
    }

    /// The UTC timestamp of the image.
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.metadata.timestamp()
    }

    /// The exposure time of the image (`Duration::ZERO` if not applicable).
    pub fn exposure(&self) -> Duration {
        self.metadata.exposure()
    }

    /// Set the exposure time of the image.
    pub fn set_exposure(&mut self, exposure: Duration) {
        self.metadata.set_exposure(exposure);
    }

    /// The acquisition frame ID of the image, or `None` if unset.
    pub fn frame_id(&self) -> Option<u32> {
        self.metadata.frame_id()
    }

    /// Set the acquisition frame ID of the image.
    pub fn set_frame_id(&mut self, frame_id: u32) {
        self.metadata.set_frame_id(frame_id);
    }

    /// Insert a metadata value into the [`GenericImageOwned`].
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
    ) -> Result<(), crate::MetadataError> {
        self.metadata.insert(name, value)
    }

    /// Remove a metadata value from the [`GenericImageOwned`].
    ///
    /// # Errors
    /// [`MetadataError::ReservedKey`](crate::MetadataError::ReservedKey) for
    /// `TIMESTAMP` / `EXPOSURE` / `FRAMEID`,
    /// [`MetadataError::KeyNotFound`](crate::MetadataError::KeyNotFound) if the
    /// key is absent, otherwise a key-validation error.
    pub fn remove_key(&mut self, name: &str) -> Result<GenericLineItem, crate::MetadataError> {
        self.metadata.remove(name)
    }

    /// Replace a metadata value in the [`GenericImageOwned`].
    ///
    /// # Errors
    /// [`MetadataError::KeyNotFound`](crate::MetadataError::KeyNotFound) if the
    /// key was not present.
    pub fn replace_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<GenericLineItem, crate::MetadataError> {
        self.metadata.replace(name, value)
    }

    /// The underlying [`DynamicImageOwned`].
    pub fn image(&self) -> &DynamicImageOwned {
        &self.image
    }

    /// The underlying [`DynamicImageOwned`], mutably.
    pub fn image_mut(&mut self) -> &mut DynamicImageOwned {
        &mut self.image
    }

    /// Borrow the image's [`Metadata`].
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Mutably borrow the image's [`Metadata`].
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// A specific extra metadata item by name (case-insensitive).
    pub fn key(&self, name: &str) -> Option<&GenericLineItem> {
        self.metadata.get(name)
    }
}

impl ImageProps for GenericImageOwned {
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

impl<'a> From<GenericImageRef<'a>> for GenericImageOwned {
    fn from(img: GenericImageRef<'a>) -> Self {
        Self {
            metadata: img.metadata,
            image: (&img.image).into(),
        }
    }
}

impl CalcOptExp for GenericImageOwned {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        self.image.calc_opt_exp(eval, exposure, bin)
    }
}

impl GenericImageOwned {
    /// Optimum exposure and binning, using this image's own recorded
    /// [`exposure`](Self::exposure).
    ///
    /// # Errors
    /// [`ExposureError::ZeroExposure`](crate::ExposureError::ZeroExposure) if the
    /// recorded exposure is `Duration::ZERO`, otherwise see [`ExposureError`](crate::ExposureError).
    pub fn optimum_exposure(
        &mut self,
        eval: &OptimumExposure,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        let exposure = self.exposure();
        self.image.calc_opt_exp(eval, exposure, bin)
    }
}

impl PixelData for GenericImageOwned {
    fn as_raw_u8(&self) -> &[u8] {
        self.image.as_raw_u8()
    }
    fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        self.image.as_raw_u8_checked()
    }
    fn as_slice_u8(&self) -> Option<&[u8]> {
        self.image.as_slice_u8()
    }
    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        self.image.as_mut_slice_u8()
    }
    fn as_slice_u16(&self) -> Option<&[u16]> {
        self.image.as_slice_u16()
    }
    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        self.image.as_mut_slice_u16()
    }
    fn as_slice_f32(&self) -> Option<&[f32]> {
        self.image.as_slice_f32()
    }
    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
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
        let img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let img = crate::ImageOwned::from_owned(img, 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let img = crate::DynamicImageOwned::from(img);
        let ts = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut img = crate::GenericImageOwned::new(ts, std::time::Duration::from_secs(2), img);
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);

        // `optimum_exposure` sources the exposure from metadata.
        let res2 = img.optimum_exposure(&opt_exp, 1).unwrap();
        assert!(res2.exposure <= std::time::Duration::from_secs(10));
    }
}
