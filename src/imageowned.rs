use std::time::Duration;

use crate::{
    imagetraits::ImageProps, CalcOptExp, ColorSpace, ExposureResult, ImageError, ImageRef,
    OptimumExposure, OptimumExposureResult, PixelStor, PixelType,
};
use bytemuck::AnyBitPattern;

/// A structure that holds image data backed by a vector.
///
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
///
/// [`ImageOwned`] supports arbitrary color spaces and number of channels, but the number
/// of channels must be consistent across the image. The data is stored in a single
/// contiguous buffer.
///
/// Alpha channels are not natively supported.
///
/// # Usage
/// ```
/// use refimage::{ImageOwned, ColorSpace};
///
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageOwned::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct ImageOwned<T: PixelStor> {
    pub(crate) data: Vec<T>,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) channels: u8,
    pub(crate) cspace: ColorSpace,
}

impl<T: PixelStor> ImageOwned<T> {
    pub(crate) fn new(
        data: Vec<T>,
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        if height > u16::MAX as usize || width > u16::MAX as usize {
            return Err(ImageError::TooLarge);
        }
        if data.is_empty() {
            return Err(ImageError::EmptyData);
        }
        if width == 0 {
            return Err(ImageError::ZeroWidth);
        }
        if height == 0 {
            return Err(ImageError::ZeroHeight);
        }
        let channels = match cspace {
            ColorSpace::Gray | ColorSpace::Bayer(_) => 1,
            ColorSpace::Rgb => 3,
            ColorSpace::Custom(ch, _) => ch as usize,
        };
        let len = data.len();
        let tot = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(channels))
            .ok_or(ImageError::TooLarge)?;
        if tot > len {
            return Err(ImageError::InsufficientData {
                expected: tot,
                got: len,
            });
        }
        let mut img = ImageOwned {
            data,
            width: width as u16,
            height: height as u16,
            channels: channels as u8,
            cspace,
        };
        img.data.truncate(tot);
        Ok(img)
    }

    /// Create a new [`ImageOwned`] from a slice of data.
    ///
    /// Images can not be larger than 65535x65535 pixels.
    ///
    /// # Arguments
    /// - `data`: The data slice. It is copied into the image.
    /// - `width`: The width of the image.
    /// - `height`: The height of the image.
    /// - `cspace`: The color space of the image ([`ColorSpace`]).
    ///
    /// # Errors
    /// See [`ImageError`].
    pub fn from_ref(
        data: &[T],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        Self::new(data.into(), width, height, cspace)
    }

    /// Create a new [`ImageOwned`] from owned data.
    ///
    /// Images can not be larger than 65535x65535 pixels.
    ///
    /// # Arguments
    /// - `data`: Owned data ([`Vec`]).
    /// - `width`: The width of the image.
    /// - `height`: The height of the image.
    /// - `cspace`: The color space of the image ([`ColorSpace`]).
    ///
    /// # Errors
    /// See [`ImageError`].
    pub fn from_owned(
        data: Vec<T>,
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        Self::new(data, width, height, cspace)
    }

    /// Get the underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Get the underlying data as a vector.
    ///
    /// Note: This function returns a copy of the data.
    pub fn into_vec(self) -> Vec<T> {
        self.data.clone()
    }

    /// Get a raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get a raw mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get an iterator over the data.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Get a mutable iterator over the data.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Get a u8 slice of the data.
    ///
    /// # Safety
    /// This function uses [`bytemuck::cast_slice`] to cast the data to a slice of u8.
    /// As such, it is unsafe, but it is safe to use since the data is vector of
    /// primitive types.
    pub fn as_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }

    /// Safely get a u8 slice of the data.
    pub fn as_u8_slice_checked(&self) -> Option<&[u8]> {
        bytemuck::try_cast_slice(self.as_slice()).ok()
    }
}

impl<T: PixelStor + AnyBitPattern> ImageOwned<T> {
    /// Get the data as a mutable slice of `u8`, regardless of the pixel type.
    ///
    /// Uses [`bytemuck::cast_slice_mut`]; safe because the backing store is a
    /// vector of primitive types.
    pub fn as_mut_u8_slice(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(self.as_mut_slice())
    }
}

impl<T: PixelStor> ImageProps for ImageOwned<T> {
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
        self.channels
    }

    #[inline(always)]
    fn color_space(&self) -> ColorSpace {
        self.cspace.clone()
    }

    #[inline(always)]
    fn pixel_type(&self) -> PixelType {
        T::PIXEL_TYPE
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

impl<T: PixelStor + AnyBitPattern> ImageOwned<T> {
    /// Create a new [`ImageOwned`] from a mutable slice of `u8` data.
    ///
    /// Images can not be larger than 65535x65535 pixels.
    ///
    /// `data` is cast to the pixel type `T` using [`bytemuck::try_cast_slice_mut`].
    /// `data` must have length (`width` * `height` * `channels` * `sizeof(T)`), and
    /// aligned to the size of `T`.
    ///
    /// # Safety
    /// The endianness of the data is determined by the system, and the data is assumed
    /// to be in native endianness. This function is not safe to use in a cross-platform
    /// environment.
    ///
    /// # Arguments
    /// - `data`: The [`&mut [u8]`] data slice.
    /// - `width`: The width of the image.
    /// - `height`: The height of the image.
    /// - `cspace`: The color space of the image ([`ColorSpace`]).
    ///
    /// # Errors
    /// [`ImageError::Cast`] for a byte-reinterpretation failure, otherwise as
    /// [`ImageOwned::from_ref`].
    pub fn from_u8(
        data: &[u8],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        let data = bytemuck::try_cast_slice(data)
            .map_err(|e| ImageError::Cast(crate::imageref::cast_msg(e)))?;
        Self::from_ref(data, width, height, cspace)
    }
}

impl<'a, T: PixelStor> From<&ImageRef<'a, T>> for ImageOwned<T> {
    fn from(data: &ImageRef<'a, T>) -> Self {
        Self {
            data: data.data[..data.len].to_vec(),
            width: data.width,
            height: data.height,
            channels: data.channels,
            cspace: data.cspace.clone(),
        }
    }
}

impl<T: PixelStor> CalcOptExp for ImageOwned<T> {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        eval.calculate(self.data.as_mut_slice(), exposure, bin)
    }
}

mod test {

    #[test]
    fn test_u8_src() {
        let mut data = vec![181u16, 178, 118, 183, 85, 131];
        let img =
            crate::ImageOwned::from_owned(data.clone(), 3, 2, crate::ColorSpace::Gray).unwrap();
        let data = bytemuck::cast_slice_mut(&mut data);
        let img2 = crate::ImageOwned::<u16>::from_u8(data, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img.as_slice(), img2.as_slice());
    }

    #[test]
    fn test_optimum_exposure() {
        use crate::CalcOptExp;
        let opt_exp = crate::OptimumExposureBuilder::default()
            .pixel_exclusion(1)
            .build()
            .unwrap();
        let img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = crate::ImageOwned::from_owned(img, 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);
    }
}
