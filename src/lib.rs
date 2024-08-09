#![deny(missing_docs)]
//! Crate to contain image data with a reference to a backing storage.

mod traits;
#[macro_use]
mod demosaic;
pub use demosaic::{BayerError, Demosaic};
use demosaic::{run_demosaic, RasterMut, ColorFilterArray};
use traits::Enlargeable;
pub use traits::{Lerp, Primitive};

/// Enum to hold the data store.
#[derive(Debug, PartialEq)]
pub enum DataStor<'a, T: Primitive> {
    /// A reference to a slice of data.
    Ref(&'a mut [T]),
    /// Owned data.
    Own(Vec<T>),
}

impl<'a, T: Primitive> DataStor<'a, T> {
    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        match self {
            DataStor::Ref(data) => data,
            DataStor::Own(data) => data.as_slice(),
        }
    }

    /// Get the data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            DataStor::Ref(data) => data,
            DataStor::Own(data) => data,
        }
    }
}

impl<'a, T: Primitive> DataStor<'a, T> {
    /// Convert to owned data.
    pub fn to_owned(self) -> Self {
        match self {
            DataStor::Ref(data) => DataStor::Own(data.to_vec()),
            DataStor::Own(data) => DataStor::Own(data),
        }
    }
}

impl<'a, T: Primitive> Clone for DataStor<'a, T> {
    fn clone(&self) -> Self {
        match self {
            DataStor::Ref(data) => DataStor::Own(data.to_vec()),
            DataStor::Own(data) => DataStor::Own(data.clone()),
        }
    }
}

/// Struct to hold image data.
#[derive(Debug, PartialEq, Clone)]
pub struct ImageData<'a, T: Primitive> {
    data: DataStor<'a, T>,
    width: u16,
    height: u16,
    channels: u8,
    cspace: ColorSpace,
}

/// Enum to describe the color space of the image.
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ColorSpace {
    /// Grayscale image.
    Gray,
    /// RGB image.
    Rgb,
    /// Bayer mosaic BGGR.
    Bggr,
    /// Bayer mosaic GBRG.
    Gbrg,
    /// Bayer mosaic GRBG.
    Grbg,
    /// Bayer mosaic RGGB.
    Rggb,
}

impl TryInto<ColorFilterArray> for ColorSpace {
    type Error = ();

    fn try_into(self) -> Result<ColorFilterArray, Self::Error> {
        match self {
            ColorSpace::Bggr => Ok(ColorFilterArray::Bggr),
            ColorSpace::Gbrg => Ok(ColorFilterArray::Gbrg),
            ColorSpace::Grbg => Ok(ColorFilterArray::Grbg),
            ColorSpace::Rggb => Ok(ColorFilterArray::Rggb),
            _ => Err(()),
        }
    }
}

impl<'a, T: Primitive> ImageData<'a, T> {
    /// Create a new image data struct.
    pub fn new(
        data: DataStor<'a, T>,
        width: u16,
        height: u16,
        channels: u8,
        cspace: ColorSpace,
    ) -> Self {
        ImageData {
            data,
            width,
            height,
            channels,
            cspace,
        }
    }

    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get the data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
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

impl<'a, T: Primitive + Enlargeable> ImageData<'a, T> {
    /// Debayer the image.
    pub fn debayer(&self, alg: Demosaic) -> Result<ImageData<T>, BayerError> {
        let cfa = self.cspace.try_into().map_err(|_| BayerError::NoGood)?;
        if self.channels > 1 || self.cspace == ColorSpace::Gray || self.cspace == ColorSpace::Rgb {
            return Err(BayerError::WrongDepth);
        }
        let mut dst = vec![T::zero(); self.width() * self.height() * 3];
        let mut dst = RasterMut::new(self.width(), self.height(), &mut dst);
        run_demosaic(self, cfa, alg, &mut dst)?;
        Ok(ImageData::new(
            DataStor::Own(dst.as_mut_slice().into()),
            self.width,
            self.height,
            3,
            ColorSpace::Rgb,
        ))
    }
}

mod test {
    #[test]
    fn test_datastor() {
        let mut data = vec![1, 2, 3, 4, 5];
        let ds = crate::DataStor::Ref(&mut data);
        let _a = ds.to_owned();
    }

    #[test]
    fn test_debayer() {
        // color_backtrace::install();
        let src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
        ];
        let expected = [
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
            169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
        ];
        let img = crate::ImageData::new(
            crate::DataStor::Own(src.into()),
            4,
            4,
            1,
            crate::ColorSpace::Rggb,
        );
        let a = img.debayer(crate::Demosaic::None);
        assert!(a.is_ok());
        let a = a.unwrap(); // at this point, a is an ImageData struct
        assert!(a.channels() == 3);
        assert!(a.width() == 4);
        assert!(a.height() == 4);
        assert!(a.color_space() == crate::ColorSpace::Rgb);
        assert_eq!(a.as_slice(), &expected);
    }
}
