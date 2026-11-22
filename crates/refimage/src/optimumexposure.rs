#![warn(missing_docs)]
use std::{cmp::Ordering, time::Duration};

use thiserror::Error;

use crate::PixelStor;

/// Errors from configuring or running the optimum-exposure calculator.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ExposureError {
    /// `pixel_tgt` is outside `[1.6e-5, 1.0]`.
    #[error("target pixel value must be in [1.6e-5, 1.0]")]
    PixelTargetRange,
    /// `pixel_uncertainty` is outside `[1.6e-5, 1.0]`.
    #[error("pixel uncertainty must be in [1.6e-5, 1.0]")]
    PixelUncertaintyRange,
    /// `percentile_pix` is outside `[0.0, 1.0]`.
    #[error("percentile must be in [0.0, 1.0]")]
    PercentileRange,
    /// `min_allowed_exp >= max_allowed_exp`.
    #[error("minimum allowed exposure must be less than the maximum")]
    ExposureBounds,
    /// `pixel_exclusion` is not smaller than the pixel count (or exceeds the
    /// 65536 cap at build time).
    #[error("pixel exclusion is larger than the number of pixels")]
    PixelExclusionTooLarge,
    /// `max_allowed_bin` exceeds 32.
    #[error("maximum allowed binning must be at most 32")]
    BinTooLarge,
    /// The image has no pixels.
    #[error("cannot compute an exposure for an empty image")]
    EmptyImage,
    /// The reference `exposure` is `Duration::ZERO`, so no scale factor can be
    /// derived from it.
    #[error("reference exposure is zero")]
    ZeroExposure,
}

/// `Result` alias for [`ExposureError`].
pub type ExposureResult<T> = Result<T, ExposureError>;

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
    pub fn build(self) -> Result<OptimumExposure, ExposureError> {
        if !(1.6e-5f32..=1f32).contains(&self.pixel_tgt) {
            return Err(ExposureError::PixelTargetRange);
        }

        if !(1.6e-5f32..=1f32).contains(&self.pixel_uncertainty) {
            return Err(ExposureError::PixelUncertaintyRange);
        }

        if self.percentile_pix < 0f32 || self.percentile_pix > 1f32 {
            return Err(ExposureError::PercentileRange);
        }

        if self.min_allowed_exp >= self.max_allowed_exp {
            return Err(ExposureError::ExposureBounds);
        }

        if self.pixel_exclusion > 65536 {
            return Err(ExposureError::PixelExclusionTooLarge);
        }

        if self.max_allowed_bin > 32 {
            return Err(ExposureError::BinTooLarge);
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

/// Outcome of [`OptimumExposure::calculate`] / [`CalcOptExp::calc_opt_exp`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OptimumExposureResult {
    /// Recommended exposure for the next acquisition.
    pub exposure: Duration,
    /// Recommended binning for the next acquisition (always `>= 1`).
    pub bin: u16,
    /// Measured value at the configured percentile, as a fraction of the pixel
    /// type's full scale (`[0, 1]`).
    pub measured: f32,
    /// `true` if `measured` was already within
    /// [`pixel_uncertainty`](OptimumExposureBuilder::pixel_uncertainty) of the
    /// target; `exposure` and `bin` are then the unchanged inputs.
    pub within_target: bool,
    /// `true` if `exposure` was clamped to the configured min / max bound.
    pub clamped: bool,
}

/// Multiply a [`Duration`] by an `f64`, saturating at [`Duration::MAX`] and
/// flooring at [`Duration::ZERO`] instead of panicking on overflow or a
/// non-finite factor.
fn scale_duration(d: Duration, factor: f64) -> Duration {
    let secs = d.as_secs_f64() * factor;
    if !secs.is_finite() || secs < 0.0 {
        return Duration::ZERO;
    }
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
}

impl OptimumExposure {
    /// Find the optimum exposure and binning to bring the target pixel to
    /// [`pixel_tgt`](OptimumExposureBuilder::pixel_tgt).
    ///
    /// The sample at the configured percentile is measured — ignoring the
    /// [`pixel_exclusion`](OptimumExposureBuilder::pixel_exclusion) hottest
    /// pixels — then the exposure is scaled linearly toward the target. When a
    /// binning range is allowed, exposure is traded against binning to keep it
    /// below [`max_allowed_exp`](OptimumExposureBuilder::max_allowed_exp).
    ///
    /// # Arguments
    ///  * `img` - Image (or luminance) samples. **Reordered in place**: a
    ///    partial ordering around the percentile is performed (`O(n)` average),
    ///    so callers must not rely on the element order afterwards. `NaN`s are
    ///    treated as equal.
    ///  * `exposure` - Exposure used to acquire `img`.
    ///  * `bin` - Binning used to acquire `img`.
    ///
    /// # Errors
    ///  * [`ExposureError::EmptyImage`] if `img` is empty.
    ///  * [`ExposureError::PixelExclusionTooLarge`] if the configured exclusion
    ///    leaves no pixels.
    pub fn calculate<T: PixelStor>(
        &self,
        img: &mut [T],
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        let len = img.len();
        if len == 0 {
            return Err(ExposureError::EmptyImage);
        }
        if exposure.is_zero() {
            return Err(ExposureError::ZeroExposure);
        }
        if self.pixel_exclusion as usize >= len {
            return Err(ExposureError::PixelExclusionTooLarge);
        }

        let full_scale = T::DEFAULT_MAX_VALUE.to_f32();
        let pixel_tgt = self.pixel_tgt * full_scale;
        let pixel_uncertainty = self.pixel_uncertainty * full_scale;
        let max_allowed_bin = self.max_allowed_bin.max(1);

        // Percentile index, capped just below the hottest `pixel_exclusion`
        // pixels so cosmic-ray hits and hot pixels don't drive the exposure.
        let hot_cutoff = len - 1 - self.pixel_exclusion as usize;
        let coord = if self.percentile_pix >= 1.0 {
            hot_cutoff
        } else {
            ((self.percentile_pix * (len - 1) as f32).floor() as usize).min(hot_cutoff)
        };

        // Only the element that lands at `coord` matters — no full sort.
        let (_, nth, _) =
            img.select_nth_unstable_by(coord, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let val = (*nth).to_f32().max(1e-5);
        let measured = (val / full_scale).clamp(0.0, 1.0);

        if (pixel_tgt - val).abs() < pixel_uncertainty {
            return Ok(OptimumExposureResult {
                exposure,
                bin: bin.max(1),
                measured,
                within_target: true,
                clamped: false,
            });
        }

        // Linear scaling: exposure ∝ target / measured.
        let mut target_exposure = scale_duration(exposure, (pixel_tgt as f64 / val as f64).abs());
        let mut bin = bin.max(1);

        if max_allowed_bin >= 2 {
            if target_exposure < self.max_allowed_exp {
                while target_exposure < self.max_allowed_exp && bin > 1 {
                    bin /= 2;
                    target_exposure = scale_duration(target_exposure, 4.0);
                }
            } else {
                while target_exposure > self.max_allowed_exp
                    && bin.saturating_mul(2) <= max_allowed_bin
                {
                    bin *= 2;
                    target_exposure = scale_duration(target_exposure, 0.25);
                }
            }
        }

        let mut clamped = false;
        if target_exposure > self.max_allowed_exp {
            target_exposure = self.max_allowed_exp;
            clamped = true;
        }
        if target_exposure < self.min_allowed_exp {
            target_exposure = self.min_allowed_exp;
            clamped = true;
        }

        Ok(OptimumExposureResult {
            exposure: target_exposure,
            bin: bin.clamp(1, max_allowed_bin),
            measured,
            within_target: false,
            clamped,
        })
    }

    /// Retrieve the builder for the [`OptimumExposure`] calculator.
    /// This is useful for changing the configuration of the calculator.
    pub fn get_builder(&self) -> OptimumExposureBuilder {
        (*self).into()
    }
}

/// Trait to calculate the optimum exposure time and binning.
///
/// This trait abstracts the retrieval of underlying image data.
pub trait CalcOptExp {
    /// Calculate the optimum exposure and binning.
    ///
    /// # Arguments
    /// * `eval` - The [`OptimumExposure`] calculator.
    /// * `exposure` - The exposure duration used to obtain the image data.
    /// * `bin` - The binning used to obtain the image data.
    ///
    /// # Errors
    /// See [`ExposureError`].
    ///
    /// # Note
    /// The underlying pixel buffer is reordered in place (see
    /// [`OptimumExposure::calculate`]).
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult>;
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
        let mut img = vec![0u16, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let exp = Duration::from_secs(10);
        let res = opt_exp.calculate(&mut img, exp, 1).unwrap();
        // A near-black frame wants a much longer exposure -> clamped to the max.
        assert_eq!(res.exposure, opt_exp.max_allowed_exp);
        assert_eq!(res.bin, 1);
        assert!(res.clamped);
        assert!(!res.within_target);

        assert_eq!(
            opt_exp.get_builder(),
            OptimumExposureBuilder::default().pixel_exclusion(1)
        );

        let img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0];
        let mut img = crate::ImageOwned::from_owned(img, 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let res = img.calc_opt_exp(&opt_exp, exp, 1).unwrap();
        assert_eq!(res.exposure, opt_exp.max_allowed_exp);
        assert_eq!(res.bin, 1);
    }

    #[test]
    fn already_on_target_is_unchanged() {
        let eval = OptimumExposureBuilder::default()
            .pixel_exclusion(0)
            .build()
            .unwrap();
        // percentile pixel sits right on `pixel_tgt` (40000 / 65536 of u16 range).
        let mut img = vec![40_000u16; 1000];
        let exp = Duration::from_millis(200);
        let res = eval.calculate(&mut img, exp, 2).unwrap();
        assert!(res.within_target);
        assert_eq!(res.exposure, exp);
        assert_eq!(res.bin, 2);
        assert!(!res.clamped);
    }

    #[test]
    fn empty_and_over_excluded_images_error() {
        let eval = OptimumExposureBuilder::default().build().unwrap();
        assert_eq!(
            eval.calculate(&mut [] as &mut [u16], Duration::from_secs(1), 1),
            Err(ExposureError::EmptyImage)
        );
        let eval = OptimumExposureBuilder::default()
            .pixel_exclusion(8)
            .build()
            .unwrap();
        let mut img = vec![0u16; 8];
        assert_eq!(
            eval.calculate(&mut img, Duration::from_secs(1), 1),
            Err(ExposureError::PixelExclusionTooLarge)
        );
    }

    #[test]
    fn float_images_are_supported() {
        let eval = OptimumExposureBuilder::default()
            .pixel_exclusion(0)
            .build()
            .unwrap();
        // Bright frame (percentile pixel near full scale) -> shorten the exposure.
        let mut img = vec![0.9f32; 500];
        let res = eval.calculate(&mut img, Duration::from_secs(1), 1).unwrap();
        assert!(res.measured > 0.8);
        assert!(res.exposure < Duration::from_secs(1));
    }
}
