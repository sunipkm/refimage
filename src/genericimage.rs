use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::{Duration, SystemTime};

use serde::Serialize;

use crate::imagetraits::ImageProps;
use crate::metadata::InsertValue;
use crate::{genericimageowned::GenericImageOwned, genericimageref::GenericImageRef};
use crate::{
    BayerError, CalcOptExp, ColorSpace, Debayer, DemosaicMethod, GenericLineItem, OptimumExposure,
    PixelType, SelectRoi, ToLuma,
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
    /// Get the timestamp of the image.
    pub fn get_timestamp(&self) -> SystemTime {
        dynamic_map!(self, ref image, { image.get_timestamp() })
    }

    /// Get the exposure time of the image.
    pub fn get_exposure(&self) -> Option<Duration> {
        dynamic_map!(self, ref image, { image.get_exposure() })
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
    ///   seconds ([`u64`]) and nanoseconds ([`u64`]).
    pub fn insert_key<T: InsertValue>(&mut self, name: &str, value: T) -> Result<(), &'static str> {
        dynamic_map!(self, ref mut image, { image.insert_key(name, value) })
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
        dynamic_map!(self, ref mut image, { image.remove_key(name) })
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
        dynamic_map!(self, ref mut image, { image.replace_key(name, value) })
    }

    // /// Get the underlying [`DynamicImageOwned`].
    // ///
    // /// # Returns
    // /// The underlying [`DynamicImageOwned`] of the [`GenericImageOwned`].
    // pub fn get_image(&self) -> &DynamicImageOwned {
    //     dynamic_map!(self, ref image => image.get_image())
    // }

    /// Get the contained metadata as a slice of [`GenericLineItem`]s.
    ///
    /// # Returns
    /// A slice of [`GenericLineItem`]s containing the metadata.
    pub fn get_metadata(&self) -> &HashMap<String, GenericLineItem> {
        dynamic_map!(self, ref image, { image.get_metadata() })
    }

    /// Get a specific metadata value by name.
    ///
    /// Returns the first metadata value with the given name.
    ///
    /// # Arguments
    /// - `name`: The name of the metadata value.
    pub fn get_key(&self, name: &str) -> Option<&GenericLineItem> {
        dynamic_map!(self, ref image, { image.get_key(name) })
    }
}

impl ImageProps for GenericImage<'_> {
    type OutputU8 = GenericImageOwned;

    fn width(&self) -> usize {
        dynamic_map!(self, ref image, { image.image.width() })
    }

    fn height(&self) -> usize {
        dynamic_map!(self, ref image, { image.image.height() })
    }

    fn channels(&self) -> u8 {
        dynamic_map!(self, ref image, { image.image.channels() })
    }

    fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, ref image, { image.image.color_space() })
    }

    fn pixel_type(&self) -> PixelType {
        dynamic_map!(self, ref image, { image.image.pixel_type() })
    }

    fn len(&self) -> usize {
        dynamic_map!(self, ref image, { image.image.len() })
    }

    fn is_empty(&self) -> bool {
        dynamic_map!(self, ref image, { image.image.is_empty() })
    }

    fn cast_u8(&self) -> Self::OutputU8 {
        let meta = self.get_metadata().clone();
        match self {
            GenericImage::Ref(image) => GenericImageOwned {
                metadata: meta,
                image: image.image.into_u8(),
            },
            GenericImage::Own(image) => GenericImageOwned {
                metadata: meta,
                image: image.image.clone().into_u8(),
            },
        }
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

impl ToLuma for GenericImage<'_> {
    fn to_luma(&mut self) -> Result<(), &'static str> {
        match self {
            GenericImage::Ref(image) => Ok(image.to_luma()?),
            GenericImage::Own(image) => Ok(image.to_luma()?),
        }
    }

    fn to_luma_custom(&mut self, coeffs: &[f64]) -> Result<(), &'static str> {
        match self {
            GenericImage::Ref(image) => Ok(image.to_luma_custom(coeffs)?),
            GenericImage::Own(image) => Ok(image.to_luma_custom(coeffs)?),
        }
    }
}

macro_rules! impl_toluma {
    ($inp: ty, $mid: ty) => {
        impl ToLuma for $inp {
            fn to_luma(&mut self) -> Result<(), &'static str> {
                self.get_image_mut().to_luma()
            }

            fn to_luma_custom(&mut self, coeffs: &[f64]) -> Result<(), &'static str> {
                self.get_image_mut().to_luma_custom(coeffs)
            }
        }
    };
}

impl_toluma!(GenericImageRef<'_>, DynamicImageRef<'_>);
impl_toluma!(GenericImageOwned, DynamicImageOwned);

impl<'a: 'b, 'b> GenericImage<'a> {
    /// Debayer a [`GenericImage`] using the specified algorithm.
    pub fn debayer(&'a self, method: DemosaicMethod) -> Result<GenericImage<'b>, BayerError> {
        match self {
            GenericImage::Ref(image) => Ok(image.debayer(method)?.into()),
            GenericImage::Own(image) => Ok(image.debayer(method)?.into()),
        }
    }
}

impl SelectRoi for GenericImage<'_> {
    type Output = GenericImage<'static>;

    fn select_roi(
        &self,
        x: usize,
        y: usize,
        w: NonZeroUsize,
        h: NonZeroUsize,
    ) -> Result<Self::Output, &'static str> {
        match self {
            GenericImage::Ref(image) => Ok(image.select_roi(x, y, w, h)?.into()),
            GenericImage::Own(image) => Ok(image.select_roi(x, y, w, h)?.into()),
        }
    }
}

impl CalcOptExp for GenericImage<'_> {
    fn calc_opt_exp(
        self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u8,
    ) -> Result<(Duration, u16), &'static str> {
        match self {
            GenericImage::Ref(img) => img.calc_opt_exp(eval, exposure, bin),
            GenericImage::Own(img) => img.calc_opt_exp(eval, exposure, bin),
        }
    }
}

impl GenericImage<'_> {
    /// Get the data as a slice of [`u8`], regardless of the underlying type.
    pub fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, ref image, { image.image.as_raw_u8() })
    }

    /// Get the data as a slice of [`u8`], regardless of the underlying type.
    pub fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, ref image, { image.image.as_raw_u8_checked() })
    }

    /// Get the data as a slice of [`u8`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_slice_u8(&self) -> Option<&[u8]> {
        dynamic_map!(self, ref image, { image.image.as_slice_u8() })
    }

    /// Get the data as a mutable slice of [`u8`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        dynamic_map!(self, ref mut image, { image.image.as_mut_slice_u8() })
    }

    /// Get the data as a slice of [`u16`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_slice_u16(&self) -> Option<&[u16]> {
        dynamic_map!(self, ref image, { image.image.as_slice_u16() })
    }

    /// Get the data as a mutable slice of [`u16`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        dynamic_map!(self, ref mut image, { image.image.as_mut_slice_u16() })
    }

    /// Get the data as a slice of [`f32`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        dynamic_map!(self, ref image, { image.image.as_slice_f32() })
    }

    /// Get the data as a mutable slice of [`f32`].
    ///
    /// # Note
    /// The returned slice may not be the same length as the image.
    /// Use [`len`](GenericImage::len) to get the length of the image.
    pub fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        dynamic_map!(self, ref mut image, { image.image.as_mut_slice_f32() })
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
    let img = crate::GenericImageRef::new(SystemTime::now(), img);
    let img = crate::GenericImage::from(img);
    let exp = std::time::Duration::from_secs(10); // expected exposure
    let bin = 1; // expected binning
    let res = img.calc_opt_exp(&opt_exp, exp, bin).unwrap();
    assert_eq!(res, (exp, bin as u16));
}
