#[cfg(feature = "rayon")]
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

use crate::{demosaic::ColorFilterArray, BayerPattern, ColorSpace, PixelStor, PixelType};

impl TryFrom<i8> for PixelType {
    type Error = &'static str;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::U8),
            16 => Ok(Self::U16),
            -32 => Ok(Self::F32),
            _ => Err("Invalid value for PixelType"),
        }
    }
}

impl TryInto<ColorFilterArray> for ColorSpace {
    type Error = &'static str;

    fn try_into(self) -> Result<ColorFilterArray, Self::Error> {
        match self {
            ColorSpace::Bayer(pat) => match pat {
                BayerPattern::Bggr => Ok(ColorFilterArray::Bggr),
                BayerPattern::Gbrg => Ok(ColorFilterArray::Gbrg),
                BayerPattern::Grbg => Ok(ColorFilterArray::Grbg),
                BayerPattern::Rggb => Ok(ColorFilterArray::Rggb),
            },
            ColorSpace::Gray => Err("Gray color space not supported in Bayer images."),
            ColorSpace::Rgb => Err("RGB color space not supported in Bayer images."),
            ColorSpace::Custom(_, _) => Err("Custom color space not supported in Bayer images."),
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
pub(crate) fn run_luma<T: PixelStor>(
    channels: usize,
    len: usize,
    data: &mut [T],
    wts: &[f64],
) -> Result<(), &'static str> {
    if channels != wts.len() {
        return Err("Number of channels and weights do not match.");
    }
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
    Ok(())
}

impl ColorSpace {
    /// Check if the color space is a Bayer pattern.
    pub fn is_bayer(&self) -> bool {
        matches!(self, Self::Bayer(_))
    }
}
