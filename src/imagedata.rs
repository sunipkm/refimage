use crate::{
    demosaic::{run_demosaic_imagedata, Debayer, RasterMut},
    traits::{cast_u8, Enlargeable},
    BayerError, ColorSpace, DataStor, DemosaicMethod, ImageOwned, PixelStor,
};
use bytemuck::{AnyBitPattern, PodCastError};
use num_traits::CheckedEuclid;

/// A structure that holds image data backed by a slice or a vector.
///
/// This represents a _matrix_ of _pixels_ which are composed of primitive and common
/// types, i.e. `u8`, `u16`, and `f32`. The matrix is stored in a _row-major_ order.
///
/// [`ImageData`] supports arbitrary color spaces and number of channels, but the number
/// of channels must be consistent across the image. The data is stored in a single
/// contiguous buffer.
///
/// # Usage
/// ```
/// use refimage::{ImageData, ColorSpace};
///
/// let data = vec![1u8, 2, 3, 4, 5, 6];
/// let img = ImageData::from_owned(data, 3, 2, ColorSpace::Gray).unwrap();
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct ImageData<'a, T: PixelStor> {
    pub(crate) data: DataStor<'a, T>,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) channels: u8,
    pub(crate) cspace: ColorSpace,
}

impl<'a, T: PixelStor> ImageData<'a, T> {
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
        let tot = width.checked_mul(height).ok_or("Image too large.")?;
        let (channels, rem) = len
            .checked_div_rem_euclid(&tot)
            .ok_or("Could not determine number of channels.")?;
        if rem != 0 {
            return Err("Data length does not match image size.");
        }
        if channels > u8::MAX.into() {
            return Err("Too many channels.");
        }

        if let Some(exp_channels) = match cspace {
            ColorSpace::Gray | ColorSpace::Bayer(_) => Some(1),
            ColorSpace::GrayAlpha | ColorSpace::BayerAlpha(_) => Some(2),
            ColorSpace::Rgb => Some(3),
            ColorSpace::Rgba => Some(4),
            _ => None,
        } {
            if channels != exp_channels {
                return Err("Invalid number of channels.");
            }
        }

        Ok(Self {
            data,
            width: width as u16,
            height: height as u16,
            channels: channels as u8,
            cspace,
        })
    }

    /// Create a new [`ImageData`] from a mutable slice of data.
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
    /// - If the image is too large.
    /// - If the data is empty.
    /// - If the width is zero.
    /// - If the height is zero.
    /// - If the data length does not match the image size.
    /// - If there are too many channels for grayscale/Bayer pattern images.
    /// - If color space is RGB and number of channels is not 3.
    pub fn from_mut_ref(
        data: &'a mut [T],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        Self::new(DataStor::from_mut_ref(data), width, height, cspace)
    }

    /// Create a new [`ImageData`] struct from owned data.
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
    /// - If the image is too large.
    /// - If the data is empty.
    /// - If the width is zero.
    /// - If the height is zero.
    /// - If the data length does not match the image size.
    /// - If there are too many channels for grayscale/Bayer pattern images.
    /// - If color space is RGB and number of channels is not 3.
    pub fn from_owned(
        data: Vec<T>,
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        Self::new(DataStor::from_owned(data), width, height, cspace)
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
    /// If the data is owned, this will return the owned data. If the data is a reference,
    /// this will return a copy of the data.
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
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
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /// Get a mutable iterator over the data.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }

    /// Get a u8 slice of the data.
    ///
    /// # Safety
    /// This function uses [`bytemuck::cast_slice`] to cast the data to a slice of u8.
    /// As such, it is unsafe, but it is safe to use since the data is vector of
    /// primitive types.
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
        self.cspace.clone() // in most cases, this is a cheap operation
    }
}

impl <'a, T: PixelStor + AnyBitPattern> ImageData<'a, T> {
        /// Create a new [`ImageData`] from a mutable slice of `u8` data.
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
    /// - Byte casting errors: [`PodCastError`].
    /// - If the image is too large.
    /// - If the data is empty.
    /// - If the width is zero.
    /// - If the height is zero.
    /// - If the data length does not match the image size.
    /// - If there are too many channels for grayscale/Bayer pattern images.
    /// - If color space is RGB and number of channels is not 3.
    pub fn from_u8_mut(
        data: &'a mut [u8],
        width: usize,
        height: usize,
        cspace: ColorSpace,
    ) -> Result<Self, &'static str> {
        let data = bytemuck::try_cast_slice_mut(data).map_err(|e| {
            use PodCastError::*;
            match e {
                TargetAlignmentGreaterAndInputNotAligned => {
                    "Target alignment greater and input not aligned"
                }
                OutputSliceWouldHaveSlop => "Output slice would have slop",
                SizeMismatch => "Size mismatch",
                AlignmentMismatch => "Alignment mismatch",
            }
        })?;
        Self::from_mut_ref(data, width, height, cspace)
    }
}

impl<T: PixelStor> ImageData<'_, T> {
    /// Convert the image to a [`ImageOwned`] with [`u8`] pixel type.
    ///
    /// Conversion is done by scaling the pixel values to the range `[0, 255]`.
    ///
    /// Note: This operation is parallelized if the `rayon` feature is enabled.
    pub fn into_u8(&self) -> ImageOwned<u8> {
        let out = cast_u8(self.data.as_slice());
        ImageOwned {
            data: out,
            width: self.width() as _,
            height: self.height() as _,
            cspace: self.cspace.clone(),
            channels: self.channels(),
        }
    }
}

impl<'a: 'b, 'b, T: PixelStor + Enlargeable> ImageData<'a, T> {
    /// Apply a function to the image data.
    ///
    /// # Arguments
    /// - `f`: The function to apply to the image data.
    ///   The function must take a reference to an [`ImageData<T>`] and return a [`ImageData<T>`].
    pub fn operate<F>(&'a self, f: F) -> Result<ImageData<'b, T>, &'static str>
    where
        F: FnOnce(&'a ImageData<'a, T>) -> Result<ImageData<'b, T>, &'static str>,
    {
        let img = f(self)?;
        Ok(img)
    }

    /// Convert the image to a luminance image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma(&self) -> Result<Self, &'static str> {
        self.into_luma_custom([0.299, 0.587, 0.114])
    }

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_custom(&self, wts: [f64; 3]) -> Result<Self, &'static str> {
        // at this point, number of channels must match number of weights
        match self.cspace {
            ColorSpace::Gray => Ok(self.clone()),
            ColorSpace::GrayAlpha => {
                let out = self.data.as_slice().iter().step_by(2).copied().collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Gray,
                )
            }
            ColorSpace::Rgb | ColorSpace::Rgba => {
                let out = crate::traits::run_luma(self.channels.into(), self.data.as_slice(), &wts);
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Gray,
                )
            }
            ColorSpace::Bayer(_) | ColorSpace::BayerAlpha(_) => Err("Image is not debayered."),
            ColorSpace::Custom(_) => Err("Custom color space not supported."),
        }
    }

    /// Convert the image to a luminance alpha image.
    ///
    /// This function uses the formula `Y = 0.299R + 0.587G + 0.114B` to calculate the
    /// corresponding luminance image.
    ///
    /// The alpha channel is copied from the original image, if present.
    /// Otherwise, the alpha channel is set to maximum value.
    ///
    /// # Errors
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_alpha(&self) -> Result<Self, &'static str> {
        self.into_luma_alpha_custom([0.299, 0.587, 0.114])
    }

    /// Convert the image to a luminance image with custom coefficients.
    ///
    /// # Arguments
    /// - `wts`: The weights to use for the conversion. The number of weights must be 3.
    ///
    /// # Errors
    /// - If the number of weights is not 3.
    /// - If the image is not debayered and is not a grayscale image.
    /// - If the image is not an RGB image.
    pub fn into_luma_alpha_custom(&self, wts: [f64; 3]) -> Result<Self, &'static str> {
        match self.cspace {
            ColorSpace::Gray => {
                let out: Vec<_> = self
                    .data
                    .as_slice()
                    .iter()
                    .flat_map(|x| [*x, T::DEFAULT_MAX_VALUE])
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::GrayAlpha,
                )
            }
            ColorSpace::GrayAlpha => Ok(self.clone()),
            ColorSpace::Rgb | ColorSpace::Rgba => {
                let out =
                    crate::traits::run_luma_alpha(self.channels.into(), self.data.as_slice(), &wts);
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::GrayAlpha,
                )
            }
            ColorSpace::Bayer(_) | ColorSpace::BayerAlpha(_) => Err("Image is not debayered."),
            ColorSpace::Custom(_) => Err("Custom color space not supported."),
        }
    }

    /// Add an alpha channel to the image.
    pub fn add_alpha(&self, luma: T) -> Result<Self, &'static str> {
        match &self.cspace {
            ColorSpace::Gray => {
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(self.channels.into())
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.push(luma);
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::GrayAlpha,
                )
            }
            ColorSpace::Rgb => {
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(3)
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.push(luma);
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Rgba,
                )
            }
            ColorSpace::Bayer(b) => {
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(1)
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.push(luma);
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::BayerAlpha(*b),
                )
            }
            ColorSpace::GrayAlpha | ColorSpace::Rgba | ColorSpace::BayerAlpha(_) => {
                Ok(self.clone())
            }
            ColorSpace::Custom(_) => Err("Custom color space not supported."),
        }
    }

    /// Remove the alpha channel from the image.
    pub fn remove_alpha(&self) -> Result<Self, &'static str> {
        match &self.cspace {
            ColorSpace::Gray | ColorSpace::Rgb | ColorSpace::Bayer(_) => Ok(self.clone()),
            ColorSpace::GrayAlpha => {
                let new = self.channels - 1;
                if new > self.channels {
                    return Err("Too many channels.");
                }
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(self.channels.into())
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.pop();
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Gray,
                )
            }
            ColorSpace::Rgba => {
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(4)
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.pop();
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Rgb,
                )
            }
            ColorSpace::BayerAlpha(b) => {
                let out = self
                    .data
                    .as_slice()
                    .chunks_exact(2)
                    .flat_map(|x| {
                        let mut x = x.to_vec();
                        x.pop();
                        x
                    })
                    .collect();
                Self::new(
                    DataStor::from_owned(out),
                    self.width(),
                    self.height(),
                    ColorSpace::Bayer(*b),
                )
            }
            ColorSpace::Custom(_) => Err("Custom color space not supported."),
        }
    }
}

impl<'a: 'b, 'b, T: PixelStor + Enlargeable> Debayer<'a, 'b> for ImageData<'b, T> {
    fn debayer(&self, alg: DemosaicMethod) -> Result<ImageData<T>, BayerError> {
        let cfa = self
            .cspace
            .clone()
            .try_into()
            .map_err(|_| BayerError::NoGood)?;
        if self.channels > 1 || self.cspace == ColorSpace::Gray || self.cspace == ColorSpace::Rgb {
            return Err(BayerError::WrongDepth);
        }
        let mut dst = vec![T::zero(); self.width() * self.height() * 3];
        let mut dst = RasterMut::new(self.width(), self.height(), &mut dst);
        run_demosaic_imagedata(self, cfa, alg, &mut dst)?;
        Ok(Self {
            data: DataStor::from_owned(dst.as_mut_slice().into()),
            width: self.width,
            height: self.height,
            channels: 3,
            cspace: ColorSpace::Rgb,
        })
    }
}

mod test {

    #[test]
    fn test_into_luma() {
        use crate::{ColorSpace, ImageData};
        let data = vec![
            181u8, 178, 118, 183, 85, 131, 82, 143, 196, 108, 64, 33, 174, 43, 18, 236, 19, 179,
            178, 132, 14, 32, 82, 1, 185, 221, 160, 112, 67, 179, 248, 104, 31, 105, 33, 100, 73,
            108, 241, 108, 208, 44, 138, 91, 188, 251, 132, 25, 233, 5, 51, 189, 41, 39, 62, 236,
            71, 150, 85, 11, 46, 95, 108, 228, 36, 187, 144, 203, 34, 218, 116, 207, 111, 168, 181,
            172, 186, 245, 223, 187, 203, 64, 70, 160, 23, 112, 11, 149, 76, 182, 206, 203, 137,
            60, 83, 94, 103, 91, 146, 176, 186, 244, 59, 144, 171, 120, 79, 144, 143, 184, 41, 137,
            4, 141, 70, 167, 51, 212, 39, 219, 102, 206, 124, 10, 92, 159, 193, 115, 132, 156, 58,
            1, 41, 89, 145, 111, 225, 177, 233, 18, 221, 20, 199, 34, 2, 189, 214, 101, 170, 33,
            223, 95, 127, 106, 169, 198, 195, 23, 29, 202, 68, 31, 127, 210, 77, 229, 204, 132, 45,
            70, 241, 160, 14, 25, 125, 10, 25, 171, 1, 13, 212, 188, 143, 139, 13, 138, 17, 128,
            226, 78, 84, 212, 230, 201, 22, 27, 189, 225, 141, 115, 64, 99, 103, 109, 173, 234,
            115, 172, 169, 208, 137, 203, 59, 108, 52, 160, 102, 185, 186, 251, 23, 185, 242, 219,
            195, 242, 75, 202, 153, 198, 102, 103, 151, 228, 211, 57, 178, 26, 254, 38, 47, 189,
            118, 246, 184, 104, 195, 40, 108, 155, 158, 47, 27, 138, 212, 61, 113, 24, 111, 171,
            47, 0, 57, 91, 213, 155, 254, 241, 58, 60, 204, 235, 37, 130, 6, 125, 185, 64, 228,
            242, 117, 52, 215, 126, 115, 50, 147, 203, 220, 192, 175, 137, 40, 191, 17, 191, 122,
            136, 168, 215, 220, 153, 179, 123, 189, 1, 45, 68, 108, 234, 98, 236, 178, 32, 141, 5,
            46, 191, 1, 81, 169, 48, 138, 89, 208, 88, 217, 183, 105, 87, 94, 53, 125, 6, 86, 201,
            11, 65, 227, 101, 221, 47, 97, 15, 192, 191, 231, 199, 119, 47, 24, 44, 33, 207, 100,
            147, 116, 60, 104, 215, 36, 95, 61, 133, 4, 89, 71, 0, 98, 82, 210, 179, 193, 29, 59,
            148, 209, 172, 231, 206, 46, 103, 106, 37, 128, 104, 201, 143, 249, 251, 18, 92, 114,
            92, 211, 129, 153, 168, 90, 133, 78, 254, 169, 125, 36, 26, 190, 126, 212, 77, 219,
            163, 61, 46, 79, 167, 50, 49, 126, 154, 105, 21, 212, 92, 5, 125, 163, 84, 35, 40, 150,
            121, 127, 37, 149, 240, 75, 56, 81, 79, 163, 153, 182, 123, 17, 64, 57, 134, 162, 179,
            148, 228, 179, 71, 15, 116, 249, 39, 15, 39, 2, 171, 103, 64, 19, 192, 101, 235, 119,
            241, 181, 117, 118, 68, 137, 33, 88, 203, 30, 127, 126, 62, 182, 247, 10, 96, 77, 109,
            183, 223, 129, 216, 76, 141, 43, 232, 169, 100, 147, 196, 182, 155, 196, 50, 211, 252,
            220, 231, 60, 252, 64, 230, 193, 29, 217, 164, 137, 113, 149, 93, 20, 86, 10, 220, 54,
            161, 198, 119, 231, 235, 89, 23, 88, 167, 116, 133, 74, 244, 64, 1, 131, 106, 130, 44,
            248, 152, 79, 82, 237, 113, 137, 228, 17, 31, 244, 28, 38, 32, 69, 215, 215, 81, 12,
            215, 172, 73, 199, 219, 74, 103, 244, 217, 171, 60, 50, 252, 147, 100, 26, 28, 72, 162,
            215, 136, 192, 166, 178, 108, 194, 48, 37, 153, 51, 10, 169, 238, 173, 209, 189, 133,
            164, 93, 111, 156, 129, 171, 54, 157, 13, 46, 9, 201, 23, 234, 87, 175, 168, 133, 230,
            114, 90, 214, 240, 69, 90, 27, 199, 158, 150, 100, 94, 204, 35, 103, 216, 120, 122, 43,
            117, 204, 59, 88, 185, 128, 161, 87, 71, 179, 154, 39, 7, 183, 17, 138, 95, 178, 133,
            196, 249, 210, 68, 64, 230, 250, 181, 230, 34, 101, 154, 247, 171, 254, 254, 205, 147,
            54, 250, 48, 174, 237, 81, 201, 170, 28, 166, 185, 52, 57, 128, 110, 64, 64, 64, 204,
            58, 73, 55, 101, 94, 180, 232, 172, 126, 45, 242, 185, 49, 146, 203, 152, 198, 176,
            174, 44, 17, 26, 140, 117, 32, 186, 233, 213, 8, 135, 199, 218, 5, 16, 114, 170, 13,
            91, 171, 247, 88, 158, 95, 220, 127, 126, 12, 3, 124, 198, 134, 151, 21, 98, 200, 157,
            131, 82, 216, 142, 218, 19, 142, 73, 108, 155, 51, 254, 221, 41, 85, 57, 60, 176,
        ];
        let img = ImageData::from_owned(data, 16, 16, ColorSpace::Rgb).unwrap();
        let luma = img.into_luma().unwrap();
        let expected = vec![
            172, 119, 130, 73, 79, 102, 132, 57, 203, 93, 138, 62, 112, 159, 116, 155, 78, 85, 165,
            95, 81, 110, 166, 156, 152, 188, 199, 78, 73, 109, 196, 77, 100, 189, 121, 98, 155, 59,
            124, 111, 165, 75, 140, 80, 81, 185, 105, 126, 135, 133, 136, 153, 75, 103, 170, 203,
            82, 58, 46, 53, 190, 64, 105, 96, 189, 144, 116, 102, 202, 174, 166, 81, 160, 109, 223,
            139, 173, 145, 116, 161, 138, 193, 94, 144, 113, 87, 138, 43, 183, 112, 203, 56, 118,
            146, 151, 124, 198, 86, 131, 163, 175, 147, 65, 154, 88, 50, 67, 105, 138, 126, 73, 75,
            67, 165, 59, 215, 65, 56, 129, 103, 73, 52, 32, 168, 81, 186, 195, 97, 122, 217, 72,
            166, 154, 114, 128, 133, 133, 89, 127, 106, 67, 44, 102, 113, 76, 122, 89, 166, 49,
            155, 198, 43, 99, 32, 70, 143, 197, 112, 70, 92, 94, 90, 107, 167, 110, 179, 179, 167,
            236, 133, 176, 154, 124, 49, 138, 177, 217, 77, 121, 110, 116, 176, 98, 140, 51, 34,
            171, 55, 116, 120, 219, 76, 105, 69, 166, 166, 90, 76, 209, 188, 116, 141, 109, 41,
            154, 166, 145, 212, 65, 146, 151, 171, 75, 105, 148, 88, 69, 80, 148, 228, 84, 207, 87,
            203, 213, 168, 200, 163, 164, 104, 63, 103, 86, 209, 91, 100, 172, 159, 36, 74, 195,
            182, 23, 68, 206, 128, 113, 96, 131, 164, 111, 172, 97, 105, 99, 72,
        ];
        assert_eq!(luma.as_slice(), &expected[..]);
    }

    #[test]
    fn test_u8_src() {
        let mut data = vec![181u16, 178, 118, 183, 85, 131];
        let img =
            crate::ImageData::from_owned(data.clone(), 3, 2, crate::ColorSpace::Gray).unwrap();
        let ptr = bytemuck::cast_slice_mut(&mut data);
        let img2 =
            crate::ImageData::<u16>::from_u8_mut(ptr, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img.as_slice(), img2.as_slice());
        let mut data = vec![181u8, 178, 118, 183, 85, 131];
        let img = crate::ImageData::from_mut_ref(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        // let ptr = bytemuck::cast_slice_mut(&mut data);
        drop(img);
        let img2 =
            crate::ImageData::<u8>::from_u8_mut(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img2.as_slice(), &[181, 178, 118, 183, 85, 131]);
        drop(img2);
        let img = crate::ImageData::from_mut_ref(&mut data, 3, 2, crate::ColorSpace::Gray).unwrap();
        assert_eq!(img.as_slice(), &[181, 178, 118, 183, 85, 131]);
    }
}
