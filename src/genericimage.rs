use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde::Serialize;

use crate::imagetraits::ImageProps;
use crate::metadata::InsertValue;
use crate::{
    genericimageowned::GenericImageOwned, genericimageref::GenericImageRef, DynamicImageOwned,
    DynamicImageRef,
};
use crate::{AlphaChannel, ColorSpace, GenericLineItem, PixelType, ToLuma};

#[derive(Debug, PartialEq, Serialize)]
/// A serializable, generic image with metadata, backed by either
/// a [`GenericImageRef`] or a [`GenericImageOwned`].
pub enum GenericImage<'a> {
    /// Holds a [`GenericImageRef`].
    Ref(GenericImageRef<'a>),
    /// Holds a [`GenericImageOwned`].
    Own(GenericImageOwned),
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

    fn into_u8(&self) -> Self::OutputU8 {
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

impl TryInto<GenericImageOwned> for GenericImage<'_> {
    type Error = &'static str;

    fn try_into(self) -> Result<GenericImageOwned, Self::Error> {
        match self {
            GenericImage::Own(data) => Ok(data),
            _ => Err("Image is not GenericImageOwned."),
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

impl<'a: 'b, 'b, T> ToLuma<'a, 'b, T> for GenericImage<'_> {
    type Output = GenericImageOwned;

    fn to_luma(&self) -> Result<Self::Output, &'static str> {
        match self {
            GenericImage::Ref(image) => <GenericImageRef<'_> as ToLuma<'_, '_, T>>::to_luma(image),
            GenericImage::Own(image) => <GenericImageOwned as ToLuma<'_, '_, T>>::to_luma(image),
        }
    }

    fn to_luma_alpha(&'a self) -> Result<Self::Output, &'static str> {
        match self {
            GenericImage::Ref(image) => {
                <GenericImageRef<'_> as ToLuma<'_, '_, T>>::to_luma_alpha(image)
            }
            GenericImage::Own(image) => {
                <GenericImageOwned as ToLuma<'_, '_, T>>::to_luma_alpha(image)
            }
        }
    }

    fn to_luma_custom(&'a self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        match self {
            GenericImage::Ref(image) => {
                <GenericImageRef<'_> as ToLuma<'_, '_, T>>::to_luma_custom(image, coeffs)
            }
            GenericImage::Own(image) => {
                <GenericImageOwned as ToLuma<'_, '_, T>>::to_luma_custom(image, coeffs)
            }
        }
    }

    fn to_luma_alpha_custom(&'a self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
        match self {
            GenericImage::Ref(image) => {
                <GenericImageRef<'_> as ToLuma<'_, '_, T>>::to_luma_alpha_custom(image, coeffs)
            }
            GenericImage::Own(image) => {
                <GenericImageOwned as ToLuma<'_, '_, T>>::to_luma_alpha_custom(image, coeffs)
            }
        }
    }
}

macro_rules! impl_toluma {
    ($inp: ty, $mid: ty) => {
        impl<'a: 'b, 'b, T> ToLuma<'a, 'b, T> for $inp {
            type Output = GenericImageOwned;

            fn to_luma(&'a self) -> Result<Self::Output, &'static str> {
                let img = <$mid as ToLuma<'_, '_, T>>::to_luma(self.get_image())?;
                let meta = self.metadata.clone();
                Ok(Self::Output {
                    metadata: meta,
                    image: img,
                })
            }

            fn to_luma_alpha(&'a self) -> Result<Self::Output, &'static str> {
                let img = <$mid as ToLuma<'_, '_, T>>::to_luma_alpha(self.get_image())?;
                let meta = self.metadata.clone();
                Ok(Self::Output {
                    metadata: meta,
                    image: img,
                })
            }

            fn to_luma_custom(&'a self, coeffs: [f64; 3]) -> Result<Self::Output, &'static str> {
                let img = <$mid as ToLuma<'_, '_, T>>::to_luma_custom(self.get_image(), coeffs)?;
                let meta = self.metadata.clone();
                Ok(Self::Output {
                    metadata: meta,
                    image: img,
                })
            }

            fn to_luma_alpha_custom(
                &'a self,
                coeffs: [f64; 3],
            ) -> Result<Self::Output, &'static str> {
                let img =
                    <$mid as ToLuma<'_, '_, T>>::to_luma_alpha_custom(self.get_image(), coeffs)?;
                let meta = self.metadata.clone();
                Ok(Self::Output {
                    metadata: meta,
                    image: img,
                })
            }
        }
    };
}

impl_toluma!(GenericImageRef<'a>, DynamicImageRef<'_>);
impl_toluma!(GenericImageOwned, DynamicImageOwned);

macro_rules! impl_alphachannel {
    ($type: ty, $inp: ty, $mid: ty) => {
        impl<'a: 'b, 'b> AlphaChannel<'a, 'b, $type, &[$type]> for $inp {
            type ImageOutput = GenericImageOwned;
            type AlphaOutput = Vec<$type>;

            fn remove_alpha(
                &'b self,
            ) -> Result<(Self::ImageOutput, Self::AlphaOutput), &'static str> {
                let (img, alpha) = <$mid as AlphaChannel<'_, '_, $type, &[$type]>>::remove_alpha(
                    self.get_image(),
                )?;
                let meta = self.metadata.clone();
                Ok((
                    Self::ImageOutput {
                        metadata: meta,
                        image: img,
                    },
                    alpha,
                ))
            }

            fn add_alpha(&'a self, alpha: &[$type]) -> Result<Self::ImageOutput, &'static str> {
                let img = <$mid as AlphaChannel<'_, '_, $type, &[$type]>>::add_alpha(
                    self.get_image(),
                    alpha,
                )?;
                let meta = self.metadata.clone();
                Ok(Self::ImageOutput {
                    metadata: meta,
                    image: img,
                })
            }
        }
    };
}

impl_alphachannel!(u8, GenericImageRef<'a>, DynamicImageRef<'_>);
impl_alphachannel!(u16, GenericImageRef<'a>, DynamicImageRef<'_>);
impl_alphachannel!(f32, GenericImageRef<'a>, DynamicImageRef<'_>);

impl_alphachannel!(u8, GenericImageOwned, DynamicImageOwned);
impl_alphachannel!(u16, GenericImageOwned, DynamicImageOwned);
impl_alphachannel!(f32, GenericImageOwned, DynamicImageOwned);
