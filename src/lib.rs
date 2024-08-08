#![deny(missing_docs)]
//! Crate to contain image data with a reference to a backing storage.

mod traits;
#[macro_use]
mod demosaic;
use demosaic::CFA;
pub use traits::{Primitive, Lerp};

// use bayer::{run_demosaic, Demosaic, CFA};
// pub use bayer::{BayerDepth, BayerError, BayerResult};

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

mod test {
    use super::DataStor;

    #[test]
    fn test_datastor() {
        let mut data = vec![1, 2, 3, 4, 5];
        let mut ds = DataStor::Ref(&mut data);
        let a = ds.to_owned();
    }
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

impl TryInto<CFA> for ColorSpace {
    type Error = ();

    fn try_into(self) -> Result<CFA, Self::Error> {
        match self {
            ColorSpace::Bggr => Ok(CFA::BGGR),
            ColorSpace::Gbrg => Ok(CFA::GBRG),
            ColorSpace::Grbg => Ok(CFA::GRBG),
            ColorSpace::Rggb => Ok(CFA::RGGB),
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

    // Debayer the image.
    // pub fn debayer(&self, alg: Demosaic) -> Result<ImageData<T>, BayerError> {
    //     let depth = BayerDepth::Depth16BE;
    //     let cfa = self.cspace.try_into().map_err(|_| BayerError::NoGood)?;
    //     if self.channels > 1 || self.cspace == ColorSpace::Gray || self.cspace == ColorSpace::Rgb {
    //         return Err(BayerError::WrongDepth);
    //     }
    //     let mut dst = Vec::<T>::with_capacity(self.width() * self.height() * 3);
    //     let src = self.as_slice();
    //     run_demosaic(src, depth, cfa, alg, dst.as_mut_slice())?;
    //     todo!()
    // }
}
