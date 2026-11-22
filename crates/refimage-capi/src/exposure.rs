//! The optimum-exposure calculator.

use std::ffi::c_void;
use std::time::Duration;

use refimage::{
    CalcOptExp, OptimumExposure, OptimumExposureBuilder, OptimumExposureResult, PixelType,
};

use crate::RefStatus::{self, *};
use crate::frame::RefFrame;
use crate::image::{RefImage, duration, owned};
use crate::{Context, Error, RefPixelType, Result, as_mut, as_ref, ffi, put};

/// Settings for the optimum-exposure search. Start from
/// `ref_exposure_options_default`.
///
/// The `percentile_pix` pixel (0.995 = the 99.5th percentile, after ignoring the
/// `pixel_exclusion` brightest) is brought to `pixel_tgt` (+/- `pixel_uncertainty`),
/// as fractions of full scale, by scaling the exposure linearly within
/// `[min_exposure, max_exposure]` seconds and binning up to `max_bin`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefExposureOptions {
    /// Percentile of the pixel to measure, in `(0, 1]`.
    pub percentile_pix: f32,
    /// Target level for that pixel, as a fraction of full scale.
    pub pixel_tgt: f32,
    /// Accepted deviation from the target, as a fraction of full scale.
    pub pixel_uncertainty: f32,
    /// Brightest pixels ignored (hot pixels, cosmic rays).
    pub pixel_exclusion: u32,
    /// Shortest allowed exposure, seconds.
    pub min_exposure: f64,
    /// Longest allowed exposure, seconds.
    pub max_exposure: f64,
    /// Largest allowed binning factor (`1` disables binning).
    pub max_bin: u16,
}

/// The outcome of an optimum-exposure calculation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RefExposureResult {
    /// Recommended exposure for the next frame, seconds.
    pub exposure: f64,
    /// Recommended binning.
    pub bin: u16,
    /// Measured percentile value, as a fraction of full scale.
    pub measured: f32,
    /// The measurement was already within tolerance (exposure and bin unchanged).
    pub within_target: bool,
    /// The exposure hit `min_exposure` or `max_exposure`.
    pub clamped: bool,
}

/// 99.5th percentile, target 40000/65536 +/- 5000/65536, 100 excluded pixels,
/// 1 ms - 10 s, no binning.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_exposure_options_default() -> RefExposureOptions {
    RefExposureOptions {
        percentile_pix: 0.995,
        pixel_tgt: 40000.0 / 65536.0,
        pixel_uncertainty: 5000.0 / 65536.0,
        pixel_exclusion: 100,
        min_exposure: 0.001,
        max_exposure: 10.0,
        max_bin: 1,
    }
}

fn evaluator(o: Option<&RefExposureOptions>) -> Result<OptimumExposure> {
    let o = o.copied().unwrap_or_else(|| ref_exposure_options_default());
    OptimumExposureBuilder::default()
        .percentile_pix(o.percentile_pix)
        .pixel_tgt(o.pixel_tgt)
        .pixel_uncertainty(o.pixel_uncertainty)
        .pixel_exclusion(o.pixel_exclusion)
        .min_allowed_exp(duration(o.min_exposure, "min_exposure")?)
        .max_allowed_exp(duration(o.max_exposure, "max_exposure")?)
        .max_allowed_bin(o.max_bin)
        .build()
        .status(REF_ERR_INVALID_ARG)
}

fn result(r: OptimumExposureResult) -> RefExposureResult {
    RefExposureResult {
        exposure: r.exposure.as_secs_f64(),
        bin: r.bin,
        measured: r.measured,
        within_target: r.within_target,
        clamped: r.clamped,
    }
}

/// The exposure to evaluate at: `seconds` when positive, else the image's own.
fn exposure_or(seconds: f64, own: Duration) -> Result<Duration> {
    if seconds > 0.0 {
        duration(seconds, "exposure")
    } else {
        Ok(own)
    }
}

/// Evaluate `img`, taken at `exposure` seconds (`<= 0` uses the image's own exposure)
/// and binning `bin`. Works on a copy; `img` is unchanged. `options` may be `NULL`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_exposure_calculate(
    options: *const RefExposureOptions,
    img: *const RefImage,
    exposure: f64,
    bin: u16,
    out: *mut RefExposureResult,
) -> RefStatus {
    ffi! {
        let eval = evaluator(options.as_ref())?;
        let img = as_ref(img, "img")?;
        let exp = exposure_or(exposure, img.inner.exposure())?;
        let mut copy = img.inner.image().clone();
        let r = copy.calc_opt_exp(&eval, exp, bin).status(REF_ERR_EXPOSURE)?;
        put(out, result(r), "out")
    }
}

/// `ref_exposure_calculate` without the copy: the percentile search **reorders
/// `img`'s pixels**. Use it only on a frame about to be discarded. `options` may
/// be `NULL` for `ref_exposure_options_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_exposure_calculate_in_place(
    options: *const RefExposureOptions,
    img: *mut RefImage,
    exposure: f64,
    bin: u16,
    out: *mut RefExposureResult,
) -> RefStatus {
    ffi! {
        let eval = evaluator(options.as_ref())?;
        let img = as_mut(img, "img")?;
        let exp = exposure_or(exposure, img.inner.exposure())?;
        let r = img.inner.calc_opt_exp(&eval, exp, bin).status(REF_ERR_EXPOSURE)?;
        put(out, result(r), "out")
    }
}

/// Evaluate a caller frame taken at `exposure` seconds (> 0). The frame is copied. `options` may
/// be `NULL` for `ref_exposure_options_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_exposure_calculate_frame(
    options: *const RefExposureOptions,
    frame: *const RefFrame,
    exposure: f64,
    bin: u16,
    out: *mut RefExposureResult,
) -> RefStatus {
    ffi! {
        let eval = evaluator(options.as_ref())?;
        let frame = as_ref(frame, "frame")?;
        let spec = frame.spec()?;
        let mut staging = Vec::new();
        let mut img = owned(&spec, Some(frame.packed(&spec, &mut staging)?))?;
        let r = img
            .calc_opt_exp(&eval, duration(exposure, "exposure")?, bin)
            .status(REF_ERR_EXPOSURE)?;
        put(out, result(r), "out")
    }
}

/// Evaluate `count` samples of `pixel_type` at `data` in place, with no copy: the
/// samples are **reordered**. `data` must be aligned for the sample type. `options` may
/// be `NULL` for `ref_exposure_options_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_exposure_calculate_buffer(
    options: *const RefExposureOptions,
    data: *mut c_void,
    count: usize,
    pixel_type: RefPixelType,
    exposure: f64,
    bin: u16,
    out: *mut RefExposureResult,
) -> RefStatus {
    ffi! {
        let eval = evaluator(options.as_ref())?;
        let pt = crate::pixel_type(pixel_type)?;
        let exp = duration(exposure, "exposure")?;
        if data.is_null() {
            return Err(Error::new(REF_ERR_NULL, "data is NULL"));
        }
        let align = crate::item_size(pt);
        if !(data as usize).is_multiple_of(align) {
            return Err(Error::invalid(format!("data must be {align}-byte aligned")));
        }
        let r = match pt.storage() {
            PixelType::U8 => eval.calculate(std::slice::from_raw_parts_mut(data.cast::<u8>(), count), exp, bin),
            PixelType::U16 => eval.calculate(std::slice::from_raw_parts_mut(data.cast::<u16>(), count), exp, bin),
            _ => eval.calculate(std::slice::from_raw_parts_mut(data.cast::<f32>(), count), exp, bin),
        }
        .status(REF_ERR_EXPOSURE)?;
        put(out, result(r), "out")
    }
}
