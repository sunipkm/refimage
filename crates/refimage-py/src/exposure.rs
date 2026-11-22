//! `refimage.OptimumExposure` / `refimage.ExposureResult`.

use std::time::Duration;

use pyo3::prelude::*;
use refimage::{CalcOptExp, OptimumExposureBuilder, OptimumExposureResult};

use crate::image::{Exposure, Image};
use crate::{NdPixels, err};

/// Finds the exposure (and binning) that brings a chosen percentile of the pixels to a
/// target level, as a fraction of full scale.
///
/// `percentile_pix` picks the pixel (0.995 = the 99.5th percentile) after ignoring the
/// `pixel_exclusion` hottest pixels; the exposure is scaled linearly so that pixel lands
/// at `pixel_tgt` (+/- `pixel_uncertainty`), clamped to `[min_exposure, max_exposure]`
/// seconds, trading against binning up to `max_bin`.
#[pyclass(module = "refimage", name = "OptimumExposure", frozen)]
pub struct OptimumExposure {
    inner: refimage::OptimumExposure,
}

#[pymethods]
impl OptimumExposure {
    #[new]
    #[pyo3(signature = (*, percentile_pix = 0.995, pixel_tgt = 40000.0 / 65536.0, pixel_uncertainty = 5000.0 / 65536.0, pixel_exclusion = 100, min_exposure = None, max_exposure = None, max_bin = 1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        percentile_pix: f32,
        pixel_tgt: f32,
        pixel_uncertainty: f32,
        pixel_exclusion: u32,
        min_exposure: Option<Exposure>,
        max_exposure: Option<Exposure>,
        max_bin: u16,
    ) -> PyResult<Self> {
        let min = min_exposure.map(Exposure::duration).transpose()?;
        let max = max_exposure.map(Exposure::duration).transpose()?;
        let inner = OptimumExposureBuilder::default()
            .percentile_pix(percentile_pix)
            .pixel_tgt(pixel_tgt)
            .pixel_uncertainty(pixel_uncertainty)
            .pixel_exclusion(pixel_exclusion)
            .min_allowed_exp(min.unwrap_or(Duration::from_millis(1)))
            .max_allowed_exp(max.unwrap_or(Duration::from_secs(10)))
            .max_allowed_bin(max_bin)
            .build()
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// Evaluate `image` (a `refimage.Image` or numpy array) taken at `exposure` (seconds
    /// or `timedelta`; defaults to the image's own) and `bin`.
    ///
    /// The percentile search partially reorders the samples, so by default it runs on a
    /// copy. `in_place=True` skips the copy and **scrambles the pixel order** of a
    /// `refimage.Image` -- use it only on a frame you are about to discard.
    #[pyo3(signature = (image, exposure = None, bin = 1, *, in_place = false))]
    fn calculate(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        exposure: Option<Exposure>,
        bin: u16,
        in_place: bool,
    ) -> PyResult<ExposureResult> {
        let eval = &self.inner;
        let exposure = exposure.map(Exposure::duration).transpose()?;
        let res = if let Ok(img) = image.cast::<Image>() {
            let exposure = match exposure {
                Some(e) => e,
                None => img.borrow().store.get().exposure(),
            };
            if in_place {
                let mut img = img.borrow_mut();
                let store = &mut img.store;
                py.detach(|| store.calc_opt_exp_in_place(eval, exposure, bin))
            } else {
                let mut copy = img.borrow().store.to_owned_image();
                py.detach(|| copy.calc_opt_exp(eval, exposure, bin))
            }
        } else {
            let exposure = exposure
                .ok_or_else(|| err("exposure is required when evaluating a numpy array"))?;
            match NdPixels::extract(image)? {
                NdPixels::U8(a) => {
                    let mut v: Vec<u8> = a.as_array().iter().copied().collect();
                    py.detach(|| eval.calculate(&mut v, exposure, bin))
                }
                NdPixels::U16(a) => {
                    let mut v: Vec<u16> = a.as_array().iter().copied().collect();
                    py.detach(|| eval.calculate(&mut v, exposure, bin))
                }
                NdPixels::F32(a) => {
                    let mut v: Vec<f32> = a.as_array().iter().copied().collect();
                    py.detach(|| eval.calculate(&mut v, exposure, bin))
                }
            }
        };
        Ok(res.map_err(err)?.into())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// The outcome of `OptimumExposure.calculate`.
#[pyclass(module = "refimage", name = "ExposureResult", frozen, get_all)]
pub struct ExposureResult {
    /// Recommended exposure for the next frame, in seconds.
    exposure: f64,
    /// Recommended binning (>= 1).
    bin: u16,
    /// Measured percentile value, as a fraction of full scale.
    measured: f32,
    /// The measurement was already within tolerance (exposure and bin unchanged).
    within_target: bool,
    /// The exposure hit `min_exposure` or `max_exposure`.
    clamped: bool,
}

impl From<OptimumExposureResult> for ExposureResult {
    fn from(r: OptimumExposureResult) -> Self {
        Self {
            exposure: r.exposure.as_secs_f64(),
            bin: r.bin,
            measured: r.measured,
            within_target: r.within_target,
            clamped: r.clamped,
        }
    }
}

#[pymethods]
impl ExposureResult {
    fn __repr__(&self) -> String {
        format!(
            "ExposureResult(exposure={}, bin={}, measured={}, within_target={}, clamped={})",
            self.exposure,
            self.bin,
            self.measured,
            if self.within_target { "True" } else { "False" },
            if self.clamped { "True" } else { "False" },
        )
    }
}
