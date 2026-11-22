use std::time::Duration;

use crate::{
    imagetraits::ImageProps, CalcOptExp, ColorSpace, ExposureResult, ImageError, OptimumExposure,
    OptimumExposureResult, PixelStor, PixelType,
};
use bytemuck::{AnyBitPattern, PodCastError};

/// A structure that holds image data backed by a slice or a vector.
///
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
///
/// [`ImageRef`] supports arbitrary color spaces and number of channels, but the number
/// of channels must be consistent across the image. The data is stored in a single
/// contiguous buffer.
/// Alpha channels are not natively supported.
///
/// # Usage
/// ```
/// use refimage::{ImageRef, ColorSpace};
///
/// let mut data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageRef::new(&mut data, 3, 2, ColorSpace::Gray).unwrap();
/// ```
#[derive(Debug, PartialEq)]
pub struct ImageRef<'a, T: PixelStor> {
    pub(crate) data: &'a mut [T],
    pub(crate) len: usize,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) cspace: ColorSpace,
    /// Meaningful bits per sample when fewer than the storage width — a
    /// `u16` image carrying 10- or 12-bit machine-vision data. `None` means
    /// "the full storage width" (the common case); see
    /// [`ImageRef::with_bit_depth`].
    pub(crate) bit_depth: Option<core::num::NonZeroU8>,
}

impl<'a, T: PixelStor> ImageRef<'a, T> {
    pub(crate) fn create(
        data: &'a mut [T],
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
        let len = data.len();
        let tot = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(cspace.channels() as usize))
            .ok_or(ImageError::TooLarge)?;
        if tot > len {
            return Err(ImageError::InsufficientData {
                expected: tot,
                got: len,
            });
        }

        Ok(Self {
            data,
            len: tot,
            width: width as u16,
            height: height as u16,
            cspace,
            bit_depth: None,
        })
    }

    /// Tag the image with its meaningful bit depth (`10` / `12` / `14`),
    /// making [`pixel_type`](ImageProps::pixel_type) report
    /// [`PixelType::U10`](crate::PixelType::U10) etc. Only valid on a
    /// `u16`-backed image (`8` / `16` / `0` / `None` clear the tag).
    pub fn with_bit_depth(mut self, bits: impl Into<Option<u8>>) -> Self {
        self.bit_depth = match bits.into() {
            Some(b @ (10 | 12 | 14)) if T::PIXEL_TYPE == PixelType::U16 => {
                core::num::NonZeroU8::new(b)
            }
            _ => None,
        };
        self
    }

    /// Create a new [`ImageRef`] from a mutable slice of data.
    /// The mutable slice of data must be at least `width * height * channels` long.
    ///
    /// Images can not be larger than 65535x65535 pixels.
    ///
    /// # Arguments
    /// - `data`: The data slice.
    /// - `width`: The width of the image.
    /// - `height`: The height of the image.
    /// - `cspace`: The color space of the image ([`ColorSpace`]).
    ///
    /// # Errors
    /// See [`ImageError`].
    pub fn new(
        data: &'a mut [T],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        Self::create(data, width, height, cspace)
    }

    /// Get the underlying data as a slice.
    ///
    /// # Note
    /// The underlying data is not guaranteed to be the same length as the image.
    /// Use [`ImageRef::len`] to get the length of the image data.
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    /// Get the underlying data as a mutable slice.
    ///
    /// # Note
    /// The underlying data is not guaranteed to be the same length as the image.
    /// Use [`ImageRef::len`] to get the length of the image data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    /// Get the underlying data as a vector.
    ///
    /// If the data is owned, this will return the owned data. If the data is a reference,
    /// this will return a copy of the data.
    pub fn into_vec(self) -> Vec<T> {
        self.data[..self.len].to_vec()
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
        self.data[..self.len].iter()
    }

    /// Get a mutable iterator over the data.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data[..self.len].iter_mut()
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

impl<T: PixelStor> ImageProps for ImageRef<'_, T> {
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
        self.len
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T: PixelStor + AnyBitPattern> ImageRef<'a, T> {
    /// Create a new [`ImageRef`] from a mutable slice of `u8` data.
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
    /// [`ImageRef::new`].
    pub fn from_u8_mut(
        data: &'a mut [u8],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, ImageError> {
        let data = bytemuck::try_cast_slice_mut(data).map_err(|e| ImageError::Cast(cast_msg(e)))?;
        Self::new(data, width, height, cspace)
    }
}

pub(crate) fn cast_msg(e: PodCastError) -> &'static str {
    use PodCastError::*;
    match e {
        TargetAlignmentGreaterAndInputNotAligned => {
            "target alignment greater and input not aligned"
        }
        OutputSliceWouldHaveSlop => "output slice would have slop",
        SizeMismatch => "size mismatch",
        AlignmentMismatch => "alignment mismatch",
    }
}

impl<T: PixelStor> CalcOptExp for ImageRef<'_, T> {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        eval.calculate(self.as_mut_slice(), exposure, bin)
    }
}

mod test {
    #[test]
    fn test_u8_src() {
        let mut data = vec![181u16, 178, 118, 183, 85, 131];
        let mut data2 = data.clone();
        let img = crate::ImageRef::new(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        let ptr = bytemuck::cast_slice_mut(&mut data2);
        let img2 = crate::ImageRef::<u16>::from_u8_mut(ptr, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img.as_slice(), img2.as_slice());
        let mut data = vec![181u8, 178, 118, 183, 85, 131];
        let img = crate::ImageRef::new(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        // let ptr = bytemuck::cast_slice_mut(&mut data);
        drop(img);
        let img2 =
            crate::ImageRef::<u8>::from_u8_mut(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img2.as_slice(), &[181, 178, 118, 183, 85, 131]);
        drop(img2);
        let img = crate::ImageRef::new(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img.as_slice(), &[181, 178, 118, 183, 85, 131]);
    }

    #[test]
    fn test_optimum_exposure() {
        use crate::CalcOptExp;
        let opt_exp = crate::OptimumExposureBuilder::default()
            .pixel_exclusion(1)
            .build()
            .unwrap();
        let mut imgsrc = vec![0u8, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 9, 8];
        let mut img = crate::ImageRef::new(imgsrc.as_mut_slice(), 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        // Near-black frame -> exposure clamped to the 10 s default maximum.
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);
        assert!(res.clamped);
    }
}
