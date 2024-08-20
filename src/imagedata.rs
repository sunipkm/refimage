use crate::{
    demosaic::{run_demosaic, RasterMut},
    traits::Enlargeable,
    BayerError, ColorSpace, DataStor, Demosaic, ImageData, PixelStor,
};
use num_traits::CheckedEuclid;

impl<'a, T: PixelStor> ImageData<'a, T> {
    /// Create a new image data struct.
    pub(crate) fn new(
        data: DataStor<'a, T>,
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        if height > u16::MAX as usize || width > u16::MAX as usize {
            return Err("Image too large.");
        }
        if data.is_empty() {
            return Err("Data is empty");
        }
        if width == 0 {
            return Err("Width is zero");
        }
        if height == 0 {
            return Err("Height is zero");
        }
        let len = data.len();
        let tot = width
            .checked_mul(height)
            .ok_or("Image too large.")?;
        let (channels, rem) = len
            .checked_div_rem_euclid(&tot)
            .ok_or("Could not determine number of channels.")?;
        if rem != 0 {
            return Err("Data length does not match image size.");
        }
        if channels > u8::MAX.into() {
            return Err("Too many channels.");
        }
        if channels > 1 && cspace < ColorSpace::Rgb {
            return Err("Too many channels for color space.");
        } else if channels > 3 && cspace == ColorSpace::Rgb {
            return Err("Too many channels for RGB.");
        }

        Ok(ImageData {
            data,
            width: width as u16,
            height: height as u16,
            channels: channels as u8,
            cspace,
        })
    }

    /// Create a new image data struct from owned data.
    pub fn from_mut_ref(
        data: &'a mut [T],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        ImageData::new(DataStor::from_mut_ref(data), width, height, cspace)
    }

    /// Create a new image data struct from owned data.
    pub fn from_owned(
        data: Vec<T>,
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        ImageData::new(DataStor::from_owned(data), width, height, cspace)
    }

    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Get the data as a vector.
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    /// Get the raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get the raw mutable pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get an iterator over the data.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /// Get a mutable iterator over the data.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }

    /// Get a u8 slice of the data.
    pub fn as_u8_slice(&self) -> &[u8] {
        self.data.as_u8_slice()
    }

    /// Safely get a u8 slice of the data.
    pub fn as_u8_slice_checked(&self) -> Option<&[u8]> {
        self.data.as_u8_slice_checked()
    }

    /// Get the length of the data.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the width of the image.
    pub fn width(&self) -> usize {
        self.width.into()
    }

    /// Get the height of the image.
    pub fn height(&self) -> usize {
        self.height.into()
    }

    /// Get the number of channels in the image.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get the color space of the image.
    pub fn color_space(&self) -> ColorSpace {
        self.cspace
    }
}

impl<'a, T: PixelStor + Enlargeable> ImageData<'a, T> {
    /// Debayer the image.
    pub fn debayer(&self, alg: Demosaic) -> Result<ImageData<T>, BayerError> {
        let cfa = self.cspace.try_into().map_err(|_| BayerError::NoGood)?;
        if self.channels > 1 || self.cspace == ColorSpace::Gray || self.cspace == ColorSpace::Rgb {
            return Err(BayerError::WrongDepth);
        }
        let mut dst = vec![T::zero(); self.width() * self.height() * 3];
        let mut dst = RasterMut::new(self.width(), self.height(), &mut dst);
        run_demosaic(self, cfa, alg, &mut dst)?;
        Ok(ImageData {
            data: DataStor::from_owned(dst.as_mut_slice().into()),
            width: self.width,
            height: self.height,
            channels: 3,
            cspace: ColorSpace::Rgb,
        })
    }
}
