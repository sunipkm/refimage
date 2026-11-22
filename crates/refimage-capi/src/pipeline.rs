//! `RefPipeline` (the recipe) and `RefRunner` (a compiled, buffer-reusing pipeline).

use std::ffi::{c_char, c_void};

use refimage::pipeline::{self as rp, Op, ScaleFactor};
use refimage::{GenericImageOwned, ImageProps, PixelData};

use crate::RefStatus::{self, *};
use crate::frame::{CFrame, RefFrame, RefImageInfo};
use crate::image::{RefImage, fresh_metadata, store};
use crate::{
    Context, Error, REF_BOUNDARY_CONSTANT, REF_STRATEGY_TILED, RefBoundary, RefBuffer, RefDemosaic,
    RefPixelType, RefResizeFilter, RefStrategyKind, Result, as_mut, as_ref, cstr, ffi, free_handle,
    put, put_buffer, put_handle, slice,
};

/// A reusable processing recipe: an ordered list of operations. Build one with
/// `ref_pipeline_new` and the `ref_pipeline_*` op functions (each appends one op),
/// or from JSON with `ref_pipeline_from_json`. Ops are validated when the pipeline
/// is compiled or applied, not when they are added.
pub struct RefPipeline {
    inner: rp::Pipeline,
}

impl RefPipeline {
    fn push(&mut self, op: Op) {
        self.inner = std::mem::take(&mut self.inner).push(op);
    }
}

/// Spline interpolation for warps. Start from `ref_interpolation_default`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefInterpolation {
    /// Spline order: `0` nearest, `1` bilinear, `3` bicubic B-spline.
    pub order: u8,
    /// A `REF_BOUNDARY_*` constant.
    pub mode: RefBoundary,
    /// Fill value for `REF_BOUNDARY_CONSTANT`, in raw stored units.
    pub cval: f64,
    /// Apply scipy's B-spline prefilter for order 3 (`true` matches scipy's default).
    pub prefilter: bool,
}

/// Bilinear, constant boundary, `cval = 0`, prefilter on.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_interpolation_default() -> RefInterpolation {
    RefInterpolation {
        order: 1,
        mode: REF_BOUNDARY_CONSTANT,
        cval: 0.0,
        prefilter: true,
    }
}

/// How a runner sweeps the frame. Start from `ref_strategy_default`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefStrategy {
    /// `REF_STRATEGY_TILED` or `REF_STRATEGY_SEQUENTIAL`.
    pub kind: RefStrategyKind,
    /// Output rows per band; `0` picks automatically. Tiled only.
    pub tile_rows: usize,
    /// Output columns per tile; `0` keeps bands full width. Tiled only.
    pub tile_cols: usize,
    /// Spread bands over a thread pool. With `false`, runs allocate nothing once
    /// compiled. Tiled only.
    pub parallel: bool,
}

/// Tiled, automatic band height, full-width bands, parallel.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_strategy_default() -> RefStrategy {
    RefStrategy {
        kind: REF_STRATEGY_TILED,
        tile_rows: 0,
        tile_cols: 0,
        parallel: true,
    }
}

// --- Lifecycle ------------------------------------------------------------------

/// Create an empty (identity) pipeline.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_new(out: *mut *mut RefPipeline) -> RefStatus {
    ffi! { put_handle(out, RefPipeline { inner: rp::Pipeline::new() }, "out") }
}

/// Copy a pipeline.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_clone(
    p: *const RefPipeline,
    out: *mut *mut RefPipeline,
) -> RefStatus {
    ffi! {
        let p = as_ref(p, "pipeline")?;
        put_handle(out, RefPipeline { inner: p.inner.clone() }, "out")
    }
}

/// Free a pipeline (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_free(p: *mut RefPipeline) {
    unsafe { free_handle(p) }
}

/// Parse a pipeline from JSON (the serde form Rust and Python use; see
/// `ref_pipeline_to_json`). Every op is available this way.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_from_json(
    json: *const c_char,
    out: *mut *mut RefPipeline,
) -> RefStatus {
    ffi! {
        let inner: rp::Pipeline = serde_json::from_str(cstr(json, "json")?).status(REF_ERR_DECODE)?;
        put_handle(out, RefPipeline { inner }, "out")
    }
}

/// The pipeline as JSON, NUL-terminated in `out` (release with `ref_buffer_free`).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_to_json(
    p: *const RefPipeline,
    out: *mut RefBuffer,
) -> RefStatus {
    ffi! {
        let json = serde_json::to_vec(&as_ref(p, "pipeline")?.inner).status(REF_ERR_PIPELINE)?;
        put_buffer(out, json)
    }
}

/// Number of ops.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_len(p: *const RefPipeline, len: *mut usize) -> RefStatus {
    ffi! { put(len, as_ref(p, "pipeline")?.inner.ops().len(), "len") }
}

/// Drop no-ops, fold flip/rotation runs and merge nested crops, in place. The output
/// is unchanged.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_optimize(p: *mut RefPipeline) -> RefStatus {
    ffi! {
        let p = as_mut(p, "pipeline")?;
        p.inner = std::mem::take(&mut p.inner).optimize();
        Ok(())
    }
}

// --- Ops ------------------------------------------------------------------------

unsafe fn push(p: *mut RefPipeline, op: impl FnOnce() -> Result<Op>) -> Result<()> {
    let p = unsafe { as_mut(p, "pipeline") }?;
    p.push(op()?);
    Ok(())
}

/// Demosaic a Bayer image to RGB (a `REF_DEMOSAIC_*` constant).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_debayer(
    p: *mut RefPipeline,
    method: RefDemosaic,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::Debayer(crate::demosaic(method)?))) }
}

/// Convert to luma (Rec.601 weights); the image becomes gray.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_to_luma(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::ToLuma)) }
}

/// Convert to luma with one weight per input channel (`count` must match the
/// channel count at compile time).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_to_luma_weights(
    p: *mut RefPipeline,
    weights: *const f64,
    count: usize,
) -> RefStatus {
    ffi! {
        let w = slice(weights, count, "weights")?.to_vec();
        push(p, || Ok(Op::ToLumaCustom(w)))
    }
}

/// `y = x * gain + offset` on the raw stored value, saturated into the pixel type.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_scale(
    p: *mut RefPipeline,
    gain: f64,
    offset: f64,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::Scale { gain, offset })) }
}

/// `y = round(x * num / den)`, exact on integer images.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_scale_rational(
    p: *mut RefPipeline,
    num: i64,
    den: i64,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::ScalePixels(ScaleFactor::Rational { num, den }))) }
}

/// `y = x * factor`, saturated into the pixel type.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_scale_by(p: *mut RefPipeline, factor: f64) -> RefStatus {
    ffi! { push(p, || Ok(Op::ScalePixels(ScaleFactor::Float(factor)))) }
}

/// Rescale into another pixel type (a `REF_PIXEL_*` constant).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_convert(
    p: *mut RefPipeline,
    pixel_type: RefPixelType,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::Convert(crate::pixel_type(pixel_type)?))) }
}

/// Keep the `width` x `height` rectangle at (`x`, `y`); it must lie inside the image.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_crop(
    p: *mut RefPipeline,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::Crop { x, y, width, height })) }
}

/// Like `ref_pipeline_crop`, but overhang past the edge is zero-filled.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_roi(
    p: *mut RefPipeline,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> RefStatus {
    ffi! { push(p, || Ok(Op::Roi { x, y, width, height })) }
}

/// Mirror left to right (Bayer patterns are re-phased).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_flip_horizontal(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::FlipHorizontal)) }
}

/// Mirror top to bottom (Bayer patterns are re-phased).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_flip_vertical(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::FlipVertical)) }
}

/// Rotate 90 degrees clockwise.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_rotate_90(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::Rotate90)) }
}

/// Rotate 180 degrees.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_rotate_180(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::Rotate180)) }
}

/// Rotate 90 degrees counter-clockwise.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_rotate_270(p: *mut RefPipeline) -> RefStatus {
    ffi! { push(p, || Ok(Op::Rotate270)) }
}

/// Resample to the largest size inside `max_width` x `max_height` at the same aspect
/// ratio (a `REF_RESIZE_*` filter).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_resize_to_fit(
    p: *mut RefPipeline,
    max_width: usize,
    max_height: usize,
    filter: RefResizeFilter,
) -> RefStatus {
    ffi! {
        push(p, || {
            Ok(Op::ResizeToFit { max_width, max_height, filter: crate::resize_filter(filter)? })
        })
    }
}

/// Warp through `inverse`, a row-major 3x3 homography taking `(column, row, 1)` of an
/// output pixel to the input (scikit-image's `warp` convention), into a
/// `width` x `height` output. `interp` may be `NULL` for `ref_interpolation_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_warp(
    p: *mut RefPipeline,
    inverse: *const f64,
    width: usize,
    height: usize,
    interp: *const RefInterpolation,
) -> RefStatus {
    ffi! {
        let m = slice(inverse, 9, "inverse")?;
        let inverse = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let interp = crate::interpolation(interp.as_ref())?;
        push(p, || Ok(Op::Warp { inverse, width, height, interp }))
    }
}

/// Sample the input at explicit coordinates (scipy's `map_coordinates`): output pixel
/// `(y, x)` reads input row `rows[y * width + x]`, column `cols[y * width + x]`. Both
/// arrays hold `width * height` values and are copied. `interp` may be `NULL` for
/// `ref_interpolation_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_map_coordinates(
    p: *mut RefPipeline,
    rows: *const f64,
    cols: *const f64,
    width: usize,
    height: usize,
    interp: *const RefInterpolation,
) -> RefStatus {
    ffi! {
        let n = width
            .checked_mul(height)
            .ok_or_else(|| Error::invalid("width * height overflows"))?;
        let rows = slice(rows, n, "rows")?.to_vec();
        let cols = slice(cols, n, "cols")?.to_vec();
        let interp = crate::interpolation(interp.as_ref())?;
        push(p, || Ok(Op::MapCoordinates { rows, cols, width, height, interp }))
    }
}

// --- Running --------------------------------------------------------------------

/// Run the pipeline once over `img` into a new image carrying `img`'s metadata.
/// Allocates; for a stream of frames compile a runner instead.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_apply(
    p: *const RefPipeline,
    img: *const RefImage,
    out: *mut *mut RefImage,
) -> RefStatus {
    ffi! {
        let p = as_ref(p, "pipeline")?;
        let img = as_ref(img, "img")?;
        let res = p.inner.apply(&img.inner).status(REF_ERR_PIPELINE)?;
        put_handle(out, RefImage::new(res), "out")
    }
}

/// A compiled pipeline holding its scratch buffers, reused for every frame. Opaque;
/// see `ref_runner_*`.
pub struct RefRunner {
    inner: rp::Runner,
    /// Packed copy of padded (`row_stride`) input frames; reused.
    staging: Vec<u8>,
}

/// Validate the pipeline against frames shaped like `like` (its `data` is ignored)
/// and allocate every buffer. `strategy` may be `NULL` for `ref_strategy_default`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_pipeline_compile(
    p: *const RefPipeline,
    like: *const RefFrame,
    strategy: *const RefStrategy,
    out: *mut *mut RefRunner,
) -> RefStatus {
    ffi! {
        let p = as_ref(p, "pipeline")?;
        let spec = as_ref(like, "like")?.spec()?;
        let strategy = crate::strategy(strategy.as_ref())?;
        let inner = p.inner.compile(spec, strategy).status(REF_ERR_PIPELINE)?;
        put_handle(out, RefRunner { inner, staging: Vec::new() }, "out")
    }
}

/// Free a runner (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_free(r: *mut RefRunner) {
    unsafe { free_handle(r) }
}

/// Rebuild the plan and buffers for frames shaped like `like`.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_recompile(
    r: *mut RefRunner,
    like: *const RefFrame,
) -> RefStatus {
    ffi! {
        let spec = as_ref(like, "like")?.spec()?;
        as_mut(r, "runner")?.inner.recompile(spec).status(REF_ERR_PIPELINE)
    }
}

/// Write the result for a frame into `*out`: when `*out` is `NULL` a new image is
/// created (timestamp now, exposure 0) and stored there; otherwise the existing image
/// is overwritten, reusing its pixel buffer when it already has the output shape (no
/// allocation) and keeping its metadata.
///
/// A frame of a different shape than the runner was compiled for recompiles it.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_run(
    r: *mut RefRunner,
    frame: *const RefFrame,
    out: *mut *mut RefImage,
) -> RefStatus {
    ffi! {
        let r = as_mut(r, "runner")?;
        let frame = as_ref(frame, "frame")?;
        let out = as_mut(out, "out")?;
        let spec = frame.spec()?;
        let bytes = frame.packed(&spec, &mut r.staging)?;
        let res = r.inner.run(&CFrame { bytes, spec: &spec }).status(REF_ERR_PIPELINE)?;
        match out.as_mut() {
            Some(img) => store(&mut img.inner, &res),
            None => {
                let img = GenericImageOwned::with_metadata(fresh_metadata(), &res);
                *out = Box::into_raw(Box::new(RefImage::new(img)));
            }
        }
        Ok(())
    }
}

/// Like `ref_runner_run` for an image input: the result also receives `img`'s
/// metadata. `*out` may be `img` itself (processed in place).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_run_image(
    r: *mut RefRunner,
    img: *const RefImage,
    out: *mut *mut RefImage,
) -> RefStatus {
    ffi! {
        let r = as_mut(r, "runner")?;
        let out = as_mut(out, "out")?;
        if img.is_null() {
            return Err(Error::new(REF_ERR_NULL, "img is NULL"));
        }
        if std::ptr::eq(*out, img) {
            // In place: the input is read before the output is written.
            let img = &mut **out;
            let res = r.inner.run(img.inner.image()).status(REF_ERR_PIPELINE)?;
            store(&mut img.inner, &res);
            return Ok(());
        }
        let src = &*img;
        let res = r.inner.run(src.inner.image()).status(REF_ERR_PIPELINE)?;
        match out.as_mut() {
            Some(dst) => {
                store(&mut dst.inner, &res);
                dst.inner.metadata_mut().clone_from(src.inner.metadata());
            }
            None => {
                let g = GenericImageOwned::with_metadata(src.inner.metadata().clone(), &res);
                *out = Box::into_raw(Box::new(RefImage::new(g)));
            }
        }
        Ok(())
    }
}

/// Run a frame and copy the result into caller memory: `dst` holds `dst_len` bytes,
/// rows `dst_row_stride` bytes apart (`0` = packed). Needs at least
/// `(height - 1) * stride + row_bytes` bytes of the output (see
/// `ref_runner_output_info`); never allocates on the caller's side.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_run_into(
    r: *mut RefRunner,
    frame: *const RefFrame,
    dst: *mut c_void,
    dst_len: usize,
    dst_row_stride: usize,
) -> RefStatus {
    ffi! {
        let r = as_mut(r, "runner")?;
        let frame = as_ref(frame, "frame")?;
        if dst.is_null() {
            return Err(Error::new(REF_ERR_NULL, "dst is NULL"));
        }
        let spec = frame.spec()?;
        let bytes = frame.packed(&spec, &mut r.staging)?;
        let res = r.inner.run(&CFrame { bytes, spec: &spec }).status(REF_ERR_PIPELINE)?;
        let info = RefImageInfo::of(res.width(), res.height(), &res.color_space(), res.pixel_type());
        let row = info.row_bytes;
        let stride = if dst_row_stride == 0 { row } else { dst_row_stride };
        if stride < row {
            return Err(Error::invalid(format!("dst_row_stride {stride} is shorter than a row ({row} bytes)")));
        }
        let need = if info.height == 0 { 0 } else { (info.height - 1) * stride + row };
        if dst_len < need {
            return Err(Error::new(
                REF_ERR_BUFFER_TOO_SMALL,
                format!("dst holds {dst_len} bytes; the output needs {need}"),
            ));
        }
        let src = res.as_raw_u8();
        let dst = dst.cast::<u8>();
        for (y, line) in src.chunks_exact(row).enumerate() {
            std::ptr::copy_nonoverlapping(line.as_ptr(), dst.add(y * stride), row);
        }
        Ok(())
    }
}

/// The input shape the runner is compiled for.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_input_info(
    r: *const RefRunner,
    out: *mut RefImageInfo,
) -> RefStatus {
    ffi! { put(out, RefImageInfo::of_spec(as_ref(r, "runner")?.inner.input_spec()), "out") }
}

/// The output shape `ref_runner_run` produces.
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_output_info(
    r: *const RefRunner,
    out: *mut RefImageInfo,
) -> RefStatus {
    ffi! { put(out, RefImageInfo::of_spec(as_ref(r, "runner")?.inner.output_spec()), "out") }
}

/// Whether runs cut frames into tiles (`false` when tiling cannot help).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_is_tiled(r: *const RefRunner, out: *mut bool) -> RefStatus {
    ffi! { put(out, as_ref(r, "runner")?.inner.is_tiled(), "out") }
}

/// Independently tiled passes (0 when not tiled; more when a geometric op splits the
/// chain).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_tiled_pass_count(
    r: *const RefRunner,
    out: *mut usize,
) -> RefStatus {
    ffi! { put(out, as_ref(r, "runner")?.inner.tiled_pass_count(), "out") }
}

/// Scratch memory held by the runner, in bytes (excluding the padded-frame staging
/// buffer).
#[must_use]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_runner_scratch_bytes(
    r: *const RefRunner,
    out: *mut usize,
) -> RefStatus {
    ffi! { put(out, as_ref(r, "runner")?.inner.scratch_bytes(), "out") }
}
