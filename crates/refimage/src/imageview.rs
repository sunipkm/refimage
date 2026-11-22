//! [`ImageView`] / [`DynamicImageView`] — a shared *read-only* borrow of image
//! samples, produced from either a [`DynamicImageRef`] or a
//! [`DynamicImageOwned`]. This is what [`GenericImage::image`](crate::GenericImage::image)
//! returns, so the same accessor works whichever half a [`GenericImage`](crate::GenericImage)
//! holds.

use core::num::NonZeroU8;

use crate::{
    ColorSpace, DynamicImageOwned, DynamicImageRef, ImageOwned, ImageProps, ImageRef, PixelData,
    PixelStor, PixelType,
};

/// A read-only, single-type view over an image's samples: a borrowed slice plus
/// the shape and (optional) sub-container bit-depth tag. The immutable analogue
/// of [`ImageRef`].
#[derive(Debug, PartialEq)]
pub struct ImageView<'a, T: PixelStor> {
    pub(crate) data: &'a [T],
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) cspace: ColorSpace,
    pub(crate) bit_depth: Option<NonZeroU8>,
}

impl<T: PixelStor> ImageView<'_, T> {
    /// The samples, as many as [`len`](ImageProps::len).
    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

impl<T: PixelStor> ImageProps for ImageView<'_, T> {
    #[inline(always)]
    fn width(&self) -> usize {
        self.width as usize
    }
    #[inline(always)]
    fn height(&self) -> usize {
        self.height as usize
    }
    #[inline(always)]
    fn channels(&self) -> u8 {
        self.cspace.channels()
    }
    #[inline(always)]
    fn color_space(&self) -> ColorSpace {
        self.cspace.clone()
    }
    #[inline(always)]
    fn pixel_type(&self) -> PixelType {
        match self.bit_depth.map(|b| b.get()) {
            Some(10) => PixelType::U10,
            Some(12) => PixelType::U12,
            Some(14) => PixelType::U14,
            _ => T::PIXEL_TYPE,
        }
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T: PixelStor> ImageRef<'a, T> {
    /// A read-only [`ImageView`] over this image's samples.
    pub fn view(&self) -> ImageView<'_, T> {
        ImageView {
            data: self.as_slice(),
            width: self.width,
            height: self.height,
            cspace: self.cspace.clone(),
            bit_depth: self.bit_depth,
        }
    }
}

impl<T: PixelStor> ImageOwned<T> {
    /// A read-only [`ImageView`] over this image's samples.
    pub fn view(&self) -> ImageView<'_, T> {
        ImageView {
            data: self.as_slice(),
            width: self.width,
            height: self.height,
            cspace: self.cspace.clone(),
            bit_depth: self.bit_depth,
        }
    }
}

/// A type-erased, read-only view over an image's samples — the immutable
/// analogue of [`DynamicImageRef`], borrowed from either that or a
/// [`DynamicImageOwned`].
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum DynamicImageView<'a> {
    /// `u8` samples.
    U8(ImageView<'a, u8>),
    /// `u16` samples (also carries the `U10` / `U12` / `U14` tag).
    U16(ImageView<'a, u16>),
    /// `f32` samples.
    F32(ImageView<'a, f32>),
}

macro_rules! view_map {
    ($v:expr, $img:pat_param, $action:expr) => {
        match $v {
            DynamicImageView::U8($img) => $action,
            DynamicImageView::U16($img) => $action,
            DynamicImageView::F32($img) => $action,
        }
    };
}

impl DynamicImageRef<'_> {
    /// A type-erased read-only [`DynamicImageView`] over this image's samples.
    pub fn view(&self) -> DynamicImageView<'_> {
        match self {
            DynamicImageRef::U8(i) => DynamicImageView::U8(i.view()),
            DynamicImageRef::U16(i) => DynamicImageView::U16(i.view()),
            DynamicImageRef::F32(i) => DynamicImageView::F32(i.view()),
        }
    }
}

impl DynamicImageOwned {
    /// A type-erased read-only [`DynamicImageView`] over this image's samples.
    pub fn view(&self) -> DynamicImageView<'_> {
        match self {
            DynamicImageOwned::U8(i) => DynamicImageView::U8(i.view()),
            DynamicImageOwned::U16(i) => DynamicImageView::U16(i.view()),
            DynamicImageOwned::F32(i) => DynamicImageView::F32(i.view()),
        }
    }
}

impl ImageProps for DynamicImageView<'_> {
    fn width(&self) -> usize {
        view_map!(self, i, i.width())
    }
    fn height(&self) -> usize {
        view_map!(self, i, i.height())
    }
    fn channels(&self) -> u8 {
        view_map!(self, i, i.channels())
    }
    fn color_space(&self) -> ColorSpace {
        view_map!(self, i, i.color_space())
    }
    fn pixel_type(&self) -> PixelType {
        view_map!(self, i, i.pixel_type())
    }
    fn len(&self) -> usize {
        view_map!(self, i, i.len())
    }
    fn is_empty(&self) -> bool {
        view_map!(self, i, i.is_empty())
    }
}

impl PixelData for DynamicImageView<'_> {
    fn as_raw_u8(&self) -> &[u8] {
        view_map!(self, i, bytemuck::cast_slice(i.data))
    }
    fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        view_map!(self, i, bytemuck::try_cast_slice(i.data).ok())
    }
    fn as_slice_u8(&self) -> Option<&[u8]> {
        match self {
            DynamicImageView::U8(i) => Some(i.data),
            _ => None,
        }
    }
    fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageView::U16(i) => Some(i.data),
            _ => None,
        }
    }
    fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageView::F32(i) => Some(i.data),
            _ => None,
        }
    }
    /// Always `None` — a [`DynamicImageView`] is read-only.
    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        None
    }
    /// Always `None` — a [`DynamicImageView`] is read-only.
    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        None
    }
    /// Always `None` — a [`DynamicImageView`] is read-only.
    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::DateTime;

    use crate::{
        ColorSpace, DynamicImageOwned, DynamicImageRef, GenericImage, GenericImageOwned,
        GenericImageRef, ImageOwned, ImageProps, ImageRef, PixelData, PixelType,
    };

    #[test]
    fn view_round_trips_shape_and_bytes() {
        let mut data: Vec<u16> = (0..12).collect();
        let owned = ImageOwned::from_owned(data.clone(), 4, 3, ColorSpace::Gray).unwrap();
        let dyno = DynamicImageOwned::from(owned);
        let v = dyno.view();
        assert_eq!((v.width(), v.height(), v.channels()), (4, 3, 1));
        assert_eq!(v.pixel_type(), PixelType::U16);
        assert_eq!(v.as_slice_u16(), Some(&data[..]));
        assert!(v.as_slice_u8().is_none());
        assert_eq!(v.as_raw_u8(), dyno.as_raw_u8());

        let iref = ImageRef::new(&mut data, 4, 3, ColorSpace::Gray).unwrap();
        let dynr = DynamicImageRef::from(iref);
        assert_eq!(dynr.view().as_slice_u16(), dyno.view().as_slice_u16());
    }

    #[test]
    fn view_keeps_sub_container_tag() {
        let img = ImageOwned::from_owned(vec![7u16; 6], 3, 2, ColorSpace::Gray)
            .unwrap()
            .with_bit_depth(12u8);
        let dyn_img = DynamicImageOwned::from(img);
        assert_eq!(dyn_img.view().pixel_type(), PixelType::U12);
    }

    #[test]
    fn generic_image_unified_view() {
        let ts = DateTime::from_timestamp(1_700_000_000, 0).unwrap();

        let owned = GenericImageOwned::new(
            ts,
            Duration::from_millis(5),
            ImageOwned::from_owned(vec![1u8, 2, 3, 4, 5, 6], 3, 2, ColorSpace::Gray).unwrap(),
        );
        let g_owned = GenericImage::from(owned);

        let mut buf = vec![1u8, 2, 3, 4, 5, 6];
        let g_ref = GenericImage::from(GenericImageRef::new(
            ts,
            Duration::from_millis(5),
            ImageRef::new(&mut buf, 3, 2, ColorSpace::Gray).unwrap(),
        ));

        assert_eq!(g_owned.image().as_slice_u8(), Some(&[1u8, 2, 3, 4, 5, 6][..]));
        assert_eq!(
            g_owned.image().as_slice_u8(),
            g_ref.image().as_slice_u8(),
        );
        assert_eq!(g_ref.image().width(), 3);
    }
}
