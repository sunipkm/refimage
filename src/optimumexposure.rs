#![warn(missing_docs)]
use std::{cmp::Ord, time::Duration};

use crate::PixelStor;

#[derive(Debug, Clone, PartialEq)]
/// Builder for the [`OptimumExposure`] calculator.
///
/// The default values are:
/// * `percentile_pix` - 0.995
/// * `pixel_tgt` - 40000. / 65536.
/// * `pixel_uncertainty` - 5000. / 65536.
/// * `pixel_exclusion` - 100
/// * `min_allowed_exp` - 1 ms
/// * `max_allowed_exp` - 10 s
/// * `max_allowed_bin` - 1
pub struct OptimumExposureBuilder {
    percentile_pix: f32,
    pixel_tgt: f32,
    pixel_uncertainty: f32,
    pixel_exclusion: u32,
    min_allowed_exp: Duration,
    max_allowed_exp: Duration,
    max_allowed_bin: u16,
}

impl Default for OptimumExposureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimumExposureBuilder {
    fn new() -> Self {
        Self {
            percentile_pix: 0.995,
            pixel_tgt: 40000. / 65536.,
            pixel_uncertainty: 5000. / 65536.,
            pixel_exclusion: 100,
            min_allowed_exp: Duration::from_millis(1),
            max_allowed_exp: Duration::from_secs(10),
            max_allowed_bin: 1,
        }
    }

    /// Set the percentile of the pixel values to use as the target pixel value.
    ///
    /// The pixels are sorted in ascending order and the pixel at the percentile
    /// is targeted for optimization.
    pub fn percentile_pix(mut self, percentile_pix: f32) -> Self {
        self.percentile_pix = percentile_pix;
        self
    }

    /// Set the target pixel value.
    ///
    /// The target pixel value is the value that the algorithm will try to reach.
    pub fn pixel_tgt(mut self, pixel_tgt: f32) -> Self {
        self.pixel_tgt = pixel_tgt;
        self
    }

    /// Set the uncertainty of the target pixel value.
    ///
    /// The pixel value is considered to be within the target if it is within the
    /// target value plus or minus the uncertainty.
    pub fn pixel_uncertainty(mut self, pixel_uncertainty: f32) -> Self {
        self.pixel_uncertainty = pixel_uncertainty;
        self
    }

    /// Set the number of pixels to exclude from the top of the image.
    ///
    /// The pixels are sorted in ascending order and the top `pixel_exclusion` pixels
    /// are excluded from the optimization.
    pub fn pixel_exclusion(mut self, pixel_exclusion: u32) -> Self {
        self.pixel_exclusion = pixel_exclusion;
        self
    }

    /// Set the minimum allowed exposure time.
    ///
    /// The minimum allowed exposure time is the shortest exposure time that the
    /// algorithm will consider.
    pub fn min_allowed_exp(mut self, min_allowed_exp: Duration) -> Self {
        self.min_allowed_exp = min_allowed_exp;
        self
    }

    /// Set the maximum allowed exposure time.
    ///
    /// The maximum allowed exposure time is the longest exposure time that the
    /// algorithm will consider.
    pub fn max_allowed_exp(mut self, max_allowed_exp: Duration) -> Self {
        self.max_allowed_exp = max_allowed_exp;
        self
    }

    /// Set the maximum allowed binning.
    ///
    /// The maximum allowed binning is the largest binning factor that the algorithm
    /// will consider to minimize the exposure time.
    pub fn max_allowed_bin(mut self, max_allowed_bin: u16) -> Self {
        self.max_allowed_bin = max_allowed_bin;
        self
    }

    /// Build the [`OptimumExposure`] object.
    pub fn build(self) -> Result<OptimumExposure, &'static str> {
        if !(1.6e-5f32..=1f32).contains(&self.pixel_tgt) {
            return Err("Target pixel value must be between 1.6e-5 and 1");
        }

        if !(1.6e-5f32..=1f32).contains(&self.pixel_uncertainty) {
            return Err("Pixel uncertainty must be between 1.6e-5 and 1");
        }

        if self.percentile_pix < 0f32 || self.percentile_pix > 1f32 {
            return Err("Percentile must be between 0 and 1.");
        }

        if self.min_allowed_exp >= self.max_allowed_exp {
            return Err("Minimum allowed exposure must be less than maximum allowed exposure");
        }

        if self.pixel_exclusion > 65536 {
            return Err("Pixel exclusion must be less than 65536");
        }

        if self.max_allowed_bin > 32 {
            return Err("Maximum allowed binning must be less than 32");
        }

        Ok(OptimumExposure {
            percentile_pix: self.percentile_pix,
            pixel_tgt: self.pixel_tgt,
            pixel_uncertainty: self.pixel_uncertainty,
            pixel_exclusion: self.pixel_exclusion,
            min_allowed_exp: self.min_allowed_exp,
            max_allowed_exp: self.max_allowed_exp,
            max_allowed_bin: self.max_allowed_bin,
        })
    }
}

impl From<OptimumExposure> for OptimumExposureBuilder {
    fn from(opt_exp: OptimumExposure) -> Self {
        OptimumExposureBuilder {
            percentile_pix: opt_exp.percentile_pix,
            pixel_tgt: opt_exp.pixel_tgt,
            pixel_uncertainty: opt_exp.pixel_uncertainty,
            pixel_exclusion: opt_exp.pixel_exclusion,
            min_allowed_exp: opt_exp.min_allowed_exp,
            max_allowed_exp: opt_exp.max_allowed_exp,
            max_allowed_bin: opt_exp.max_allowed_bin,
        }
    }
}

/// Configuration used to find the optimum exposure.
///
///
/// # Options
///  * `percentile_pix` - The percentile of the pixel values to use as the target pixel value, in fraction.
///  * `pixel_tgt` - The target pixel value, in fraction.
///  * `pixel_tol` - The uncertainty of the target pixel value, in fraction.
///  * `pixel_exclusion` - The number of pixels to exclude from the top of the image.
///  * `min_exposure` - The minimum allowed exposure time.
///  * `max_exposure` - The maximum allowed exposure time.
///  * `max_bin` - The maximum allowed binning.
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimumExposure {
    percentile_pix: f32,
    pixel_tgt: f32,
    pixel_uncertainty: f32,
    min_allowed_exp: Duration,
    max_allowed_exp: Duration,
    max_allowed_bin: u16,
    pixel_exclusion: u32,
}

impl OptimumExposure {
    /// Find the optimum exposure time and binning to reach a target pixel value.
    /// The algorithm does not use any hysteresis and uses simple scaling.
    ///
    /// # Arguments
    ///  * `mut img` - The image luminance data as a vector of u16 that is consumed.
    ///  * `exposure` - The exposure duration used to obtain this image luminance data.
    ///  * `bin` - The binning used to obtain this image luminance data.
    ///
    /// # Returns
    ///  * `Ok((Duration, u16))` - The optimum exposure time and binning.
    ///
    /// # Errors
    ///  - Errors are returned as static string slices.
    pub fn calculate<T: PixelStor + Ord>(
        &self,
        img: &mut [T],
        len: usize,
        exposure: Duration,
        bin: u8,
    ) -> Result<(Duration, u16), &'static str> {
        let mut target_exposure;

        let mut change_bin = true;

        let pixel_tgt = self.pixel_tgt;
        let pixel_uncertainty = self.pixel_uncertainty;
        let percentile_pix = self.percentile_pix;
        let min_allowed_exp = self.min_allowed_exp;
        let max_allowed_exp = self.max_allowed_exp;
        let max_allowed_bin = self.max_allowed_bin;
        let pixel_exclusion = self.pixel_exclusion;

        if !(1.6e-5f32..=1f32).contains(&pixel_tgt) {
            return Err("Target pixel value must be between 1.6e-5 and 1");
        }

        if !(1.6e-5f32..=1f32).contains(&pixel_uncertainty) {
            return Err("Pixel uncertainty must be between 1.6e-5 and 1");
        }

        if !(0f32..=1f32).contains(&percentile_pix) {
            return Err("Percentile must be between 0 and 1");
        }

        if min_allowed_exp >= max_allowed_exp {
            return Err("Minimum allowed exposure must be less than maximum allowed exposure");
        }

        if pixel_exclusion > img.len() as u32 {
            return Err("Pixel exclusion must be less than the number of pixels");
        }

        let max_allowed_bin = if max_allowed_bin < 2 {
            1
        } else {
            max_allowed_bin
        };

        let pixel_tgt = pixel_tgt * (T::DEFAULT_MAX_VALUE).to_f32();
        let pixel_uncertainty = pixel_uncertainty * (T::DEFAULT_MAX_VALUE).to_f32();

        if max_allowed_bin < 2 {
            change_bin = false;
        }
        let mut bin = bin as u16;
        img[..len].sort();
        let mut coord: usize;
        if percentile_pix > 0.99999 {
            coord = len - 1_usize;
        } else {
            coord = (percentile_pix * (len - 1) as f32).floor() as usize;
        }
        if coord < pixel_exclusion as usize {
            coord = len - 1 - pixel_exclusion as usize;
        }
        let val = img[..len].get(coord);
        let val = match val {
            Some(v) => (*v).to_f32(),
            None => 1e-5_f32,
        };

        if (pixel_tgt - val).abs() < pixel_uncertainty {
            return Ok((exposure, bin));
        }

        let val = {
            if val <= 1e-5 {
                1e-5
            } else {
                val
            }
        };

        target_exposure = Duration::from_secs_f64(
            (pixel_tgt as f64 * exposure.as_micros() as f64 * 1e-6 / val as f64).abs(),
        );

        if change_bin {
            let mut tgt_exp = target_exposure;
            let mut bin_ = bin;
            if tgt_exp < max_allowed_exp {
                while tgt_exp < max_allowed_exp && bin_ > 2 {
                    bin_ /= 2;
                    tgt_exp *= 4;
                }
            } else {
                while tgt_exp > max_allowed_exp && bin_ * 2 <= max_allowed_bin {
                    bin_ *= 2;
                    tgt_exp /= 4;
                }
            }
            target_exposure = tgt_exp;
            bin = bin_;
        }

        if target_exposure > max_allowed_exp {
            target_exposure = max_allowed_exp;
        }

        if target_exposure < min_allowed_exp {
            target_exposure = min_allowed_exp;
        }

        if bin < 1 {
            bin = 1;
        }
        if bin > max_allowed_bin {
            bin = max_allowed_bin;
        }

        Ok((target_exposure, bin))
    }

    /// Retrieve the builder for the [`OptimumExposure`] calculator.
    /// This is useful for changing the configuration of the calculator.
    pub fn get_builder(&self) -> OptimumExposureBuilder {
        OptimumExposureBuilder {
            percentile_pix: self.percentile_pix,
            pixel_tgt: self.pixel_tgt,
            pixel_uncertainty: self.pixel_uncertainty,
            pixel_exclusion: self.pixel_exclusion,
            min_allowed_exp: self.min_allowed_exp,
            max_allowed_exp: self.max_allowed_exp,
            max_allowed_bin: self.max_allowed_bin,
        }
    }
}

/// Trait to calculate the optimum exposure time and binning.
///
/// This trait abstracts the retrieval of underlying image data.
pub trait CalcOptExp {
    /// Calculate the optimum exposure time and binning.
    ///
    /// # Arguments
    /// * `eval` - The [`OptimumExposure`] calculator.
    /// * `exposure` - The exposure duration used to obtain the image data.
    /// * `bin` - The binning used to obtain the image data.
    ///
    /// # Returns
    /// * `Ok((Duration, u16))` - The optimum exposure time and binning.
    /// * `Err(&'static str)` - Error message.
    ///
    /// # Note
    /// The image data is consumed by the function.
    fn calc_opt_exp(
        self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u8,
    ) -> Result<(Duration, u16), &'static str>;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_optimum_exposure() {
        let opt_exp = OptimumExposureBuilder::default()
            .pixel_exclusion(1)
            .build()
            .unwrap();
        let mut img = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let exp = Duration::from_secs(10); // expected exposure
        let bin = 1; // expected binning
        let len = img.len();
        let res = opt_exp.calculate(&mut img, len, exp, bin).unwrap();
        assert_eq!(res, (exp, bin as u16));
        assert_eq!(
            opt_exp.get_builder(),
            OptimumExposureBuilder::default().pixel_exclusion(1)
        );
        let img = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0];
        let img = crate::ImageOwned::from_owned(img, 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let exp = Duration::from_secs(10); // expected exposure
        let bin = 1; // expected binning
        let res = img.calc_opt_exp(&opt_exp, exp, bin).unwrap();
        assert_eq!(res, (exp, bin as u16));
    }
}
