use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::imagetraits::ImageProps;
use crate::metadata::InsertValue;
use crate::{genericimageowned::GenericImageOwned, genericimageref::GenericImageRef};
use crate::{
    CalcOptExp, ColorSpace, DynamicImageView, ExposureResult, GenericLineItem, Metadata,
    OptimumExposure, OptimumExposureResult, PixelData, PixelType,
};

#[derive(Debug, PartialEq, Serialize)]
/// A serializable, generic image with metadata, backed by either
/// a [`GenericImageRef`] or a [`GenericImageOwned`].
pub enum GenericImage<'a> {
    /// Holds a [`GenericImageRef`].
    Ref(GenericImageRef<'a>),
    /// Holds a [`GenericImageOwned`].
    Own(GenericImageOwned),
}

impl Clone for GenericImage<'_> {
    fn clone(&self) -> Self {
        match self {
            GenericImage::Ref(data) => {
                let meta = data.metadata.clone();
                GenericImage::Own(GenericImageOwned {
                    metadata: meta,
                    image: (&data.image).into(),
                })
            }
            GenericImage::Own(data) => GenericImage::Own(data.clone()),
        }
    }
}

macro_rules! dynamic_map(
    ($dynimage: expr, $image: pat => $action: expr) => ({
        use GenericImage::*;
        match $dynimage {
            Ref($image) => Ref($action),
            Own($image) => Own($action),
        }
    });

    ($dynimage: expr, $image:pat_param, $action: expr) => (
        match $dynimage {
            GenericImage::Ref($image) => $action,
            GenericImage::Own($image) => $action,
        }
    );
);

impl GenericImage<'_> {
    /// The UTC timestamp of the image.
    pub fn timestamp(&self) -> DateTime<Utc> {
        dynamic_map!(self, image, { image.timestamp() })
    }

    /// The exposure time of the image (`Duration::ZERO` if not applicable).
    pub fn exposure(&self) -> Duration {
        dynamic_map!(self, image, { image.exposure() })
    }

    /// Set the exposure time of the image.
    pub fn set_exposure(&mut self, exposure: Duration) {
        dynamic_map!(self, image, { image.set_exposure(exposure) })
    }

    /// The acquisition frame ID of the image, or `None` if unset.
    pub fn frame_id(&self) -> Option<u32> {
        dynamic_map!(self, image, { image.frame_id() })
    }

    /// Set the acquisition frame ID of the image.
    pub fn set_frame_id(&mut self, frame_id: u32) {
        dynamic_map!(self, image, { image.set_frame_id(frame_id) })
    }

    /// Insert a metadata value into the [`GenericImage`].
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
    /// - Re-inserting a timestamp key will return an error.
    /// - When saving to a FITS file, the metadata comment may be truncated.
    /// - Metadata of type [`std::time::Duration`] or a UTC `chrono::DateTime` is
    ///   stored as two consecutive metadata items split into seconds and nanoseconds
    ///   (keys suffixed `_S` and `_NS`), plus a base card.
    pub fn insert_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<(), crate::MetadataError> {
        dynamic_map!(self, image, { image.insert_key(name, value) })
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
    pub fn remove_key(&mut self, name: &str) -> Result<GenericLineItem, crate::MetadataError> {
        dynamic_map!(self, image, { image.remove_key(name) })
    }

    /// Replace a metadata value in the [`GenericImageOwned`].
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value to replace.
    /// - `value`: The new value to insert. The value is either a primitive type, a `String`, or a `std::time::Duration` or a UTC `chrono::DateTime` or a tuple of a value type and a comment.
    ///
    /// # Returns
    /// - `Ok(())` if the key was replaced successfully.
    /// - `Err("Key not found")` if the key was not found.
    ///
    pub fn replace_key<T: InsertValue>(
        &mut self,
        name: &str,
        value: T,
    ) -> Result<GenericLineItem, crate::MetadataError> {
        dynamic_map!(self, image, { image.replace_key(name, value) })
    }

    /// A read-only, type-erased [`DynamicImageView`] over the samples, whichever
    /// half this [`GenericImage`] holds.
    pub fn image(&self) -> DynamicImageView<'_> {
        match self {
            GenericImage::Ref(g) => g.image().view(),
            GenericImage::Own(g) => g.image().view(),
        }
    }

    /// Borrow the image's [`Metadata`].
    pub fn metadata(&self) -> &Metadata {
        dynamic_map!(self, image, { image.metadata() })
    }

    /// Mutably borrow the image's [`Metadata`].
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        dynamic_map!(self, image, { image.metadata_mut() })
    }

    /// A specific extra metadata item by name (case-insensitive).
    pub fn key(&self, name: &str) -> Option<&GenericLineItem> {
        dynamic_map!(self, image, { image.key(name) })
    }

    /// Convert into an owned [`GenericImageOwned`], copying the samples when this
    /// is the borrowed half.
    pub fn into_owned(self) -> GenericImageOwned {
        self.into()
    }
}

impl ImageProps for GenericImage<'_> {
    fn width(&self) -> usize {
        dynamic_map!(self, image, { image.image.width() })
    }

    fn height(&self) -> usize {
        dynamic_map!(self, image, { image.image.height() })
    }

    fn channels(&self) -> u8 {
        dynamic_map!(self, image, { image.image.channels() })
    }

    fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, image, { image.image.color_space() })
    }

    fn pixel_type(&self) -> PixelType {
        dynamic_map!(self, image, { image.image.pixel_type() })
    }

    fn len(&self) -> usize {
        dynamic_map!(self, image, { image.image.len() })
    }

    fn is_empty(&self) -> bool {
        dynamic_map!(self, image, { image.image.is_empty() })
    }
}

impl From<GenericImageOwned> for GenericImage<'_> {
    fn from(img: GenericImageOwned) -> Self {
        Self::Own(img)
    }
}

impl<'a> From<GenericImageRef<'a>> for GenericImage<'a> {
    fn from(img: GenericImageRef<'a>) -> Self {
        Self::Ref(img)
    }
}

impl From<GenericImage<'_>> for GenericImageOwned {
    fn from(val: GenericImage<'_>) -> Self {
        match val {
            GenericImage::Own(data) => data,
            GenericImage::Ref(data) => data.into(),
        }
    }
}

impl<'a> TryInto<GenericImageRef<'a>> for GenericImage<'a> {
    type Error = &'static str;

    fn try_into(self) -> Result<GenericImageRef<'a>, Self::Error> {
        match self {
            GenericImage::Ref(data) => Ok(data),
            _ => Err("Image is not GenericImageRef."),
        }
    }
}

impl CalcOptExp for GenericImage<'_> {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        match self {
            GenericImage::Ref(img) => img.calc_opt_exp(eval, exposure, bin),
            GenericImage::Own(img) => img.calc_opt_exp(eval, exposure, bin),
        }
    }
}

impl GenericImage<'_> {
    /// Optimum exposure and binning, using this image's own recorded
    /// [`exposure`](Self::exposure).
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
        let exposure = self.exposure();
        self.calc_opt_exp(eval, exposure, bin)
    }
}

impl PixelData for GenericImage<'_> {
    fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, image, { image.as_raw_u8() })
    }
    fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, image, { image.as_raw_u8_checked() })
    }
    fn as_slice_u8(&self) -> Option<&[u8]> {
        dynamic_map!(self, image, { image.as_slice_u8() })
    }
    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        dynamic_map!(self, image, { image.as_mut_slice_u8() })
    }
    fn as_slice_u16(&self) -> Option<&[u16]> {
        dynamic_map!(self, image, { image.as_slice_u16() })
    }
    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        dynamic_map!(self, image, { image.as_mut_slice_u16() })
    }
    fn as_slice_f32(&self) -> Option<&[f32]> {
        dynamic_map!(self, image, { image.as_slice_f32() })
    }
    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        dynamic_map!(self, image, { image.as_mut_slice_f32() })
    }
}

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
    let img = crate::GenericImageRef::new(ts, Duration::from_secs(2), img);
    let mut img = crate::GenericImage::from(img);
    let res = img
        .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
        .unwrap();
    assert_eq!(res.exposure, std::time::Duration::from_secs(10));
    assert_eq!(res.bin, 1);
}
