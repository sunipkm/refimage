#[cfg(feature = "rayon")]
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

use crate::{
    demosaic::ColorFilterArray, BayerError, BayerPattern, ColorSpace, ImageError, PixelStor,
    PixelType,
};

impl TryFrom<i8> for PixelType {
    type Error = ImageError;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::U8),
            10 => Ok(Self::U10),
            12 => Ok(Self::U12),
            14 => Ok(Self::U14),
            16 => Ok(Self::U16),
            -32 => Ok(Self::F32),
            other => Err(ImageError::InvalidPixelType(other)),
        }
    }
}

impl TryFrom<ColorSpace> for ColorFilterArray {
    type Error = BayerError;

    fn try_from(value: ColorSpace) -> Result<ColorFilterArray, Self::Error> {
        match value {
            ColorSpace::Bayer(pat) => Ok(match pat {
                BayerPattern::Bggr => ColorFilterArray::Bggr,
                BayerPattern::Gbrg => ColorFilterArray::Gbrg,
                BayerPattern::Grbg => ColorFilterArray::Grbg,
                BayerPattern::Rggb => ColorFilterArray::Rggb,
            }),
            other => Err(BayerError::InvalidColorSpace(other)),
        }
    }
}

#[allow(clippy::from_over_into)]
impl Into<ColorSpace> for BayerPattern {
    fn into(self) -> ColorSpace {
        ColorSpace::Bayer(self)
    }
}

/// Run the luminance conversion on a slice of pixel data.
///
/// The caller (the pipeline) guarantees `channels == wts.len()`.
pub(crate) fn run_luma<T: PixelStor>(channels: usize, len: usize, data: &mut [T], wts: &[f64]) {
    debug_assert_eq!(channels, wts.len());
    #[cfg(not(feature = "rayon"))]
    {
        let len = len / channels;
        for i in 0..len {
            let v = T::from_f64(
                data[i * channels..(i + 1) * channels]
                    .iter()
                    .zip(wts.iter())
                    .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
            );
            data[i] = v;
        }
    }
    #[cfg(feature = "rayon")]
    {
        if len > 1024 * 1024 {
            // for large images, use parallel processing
            data[..len]
                .par_chunks_exact_mut(channels)
                .for_each(|chunk| {
                    let v = T::from_f64(
                        chunk
                            .iter()
                            .zip(wts.iter())
                            .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
                    );
                    chunk[0] = v;
                });
            let len = len / channels;
            for i in 0..len {
                data[i] = data[i * channels];
            }
        } else {
            // for small images, use sequential processing
            let len = len / channels;
            for i in 0..len {
                let v = T::from_f64(
                    data[i * channels..(i + 1) * channels]
                        .iter()
                        .zip(wts.iter())
                        .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
                );
                data[i] = v;
            }
        }
    }
}

impl ColorSpace {
    /// Check if the color space is a Bayer pattern.
    pub fn is_bayer(&self) -> bool {
        matches!(self, Self::Bayer(_))
    }
}
