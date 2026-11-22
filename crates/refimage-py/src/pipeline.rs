//! `refimage.Pipeline`, `refimage.ImageSpec` and `refimage.Runner`.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use refimage::pipeline::{self as rp, Op};
use refimage::{ColorSpace, DynamicImageRef, GenericImageOwned, ImageProps, Metadata, PixelType};

use crate::image::{Image, now};
use crate::{
    NdPixels, cspace_to_py, dtype_name, err, infer_cspace, parse_cspace, parse_demosaic,
    parse_filter, parse_interp, parse_pixel_type, parse_strategy, pixel_type_name,
};

/// A 3x3 matrix from any nested sequence or array.
pub(crate) fn matrix3(obj: &Bound<'_, PyAny>) -> PyResult<[[f64; 3]; 3]> {
    let rows: Vec<Vec<f64>> = obj
        .extract()
        .map_err(|_| err("inverse_map must be a 3x3 array of numbers"))?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        let shape: Vec<usize> = rows.iter().map(Vec::len).collect();
        return Err(err(format!(
            "inverse_map must be 3x3, not {} rows of lengths {shape:?}",
            rows.len()
        )));
    }
    let mut m = [[0.0; 3]; 3];
    for (dst, src) in m.iter_mut().zip(&rows) {
        dst.copy_from_slice(src);
    }
    Ok(m)
}

/// `(rows, cols, width, height)` from a `(2, H, W)` coordinate array.
pub(crate) fn coordinate_planes(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<f64>, usize, usize)> {
    let np = obj.py().import("numpy")?;
    let arr = np.call_method1("asarray", (obj, "float64"))?;
    let arr: PyReadonlyArrayDyn<'_, f64> = arr.extract()?;
    let (h, w) = match *arr.shape() {
        [2, h, w] => (h, w),
        ref s => {
            return Err(err(format!(
                "coordinates must have shape (2, rows, cols): the input row and column of \
                 each output pixel, not {s:?}"
            )));
        }
    };
    let view = arr.as_array();
    let rows = view.index_axis(numpy::ndarray::Axis(0), 0).iter().copied().collect();
    let cols = view.index_axis(numpy::ndarray::Axis(0), 1).iter().copied().collect();
    Ok((rows, cols, w, h))
}

/// A reusable, serializable processing recipe. Every builder method returns a new
/// pipeline, so a base recipe can be extended without being changed:
///
/// ```python
/// p = refimage.Pipeline().debayer("linear").to_luma().convert("u8")
/// runner = p.compile(first_frame)          # allocate once
/// out = runner.run(frame, out=out)         # then reuse `out` for every frame
/// ```
#[pyclass(module = "refimage", name = "Pipeline", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Default)]
pub struct Pipeline {
    pub(crate) inner: rp::Pipeline,
}

impl Pipeline {
    fn with(&self, op: Op) -> Self {
        Self {
            inner: self.inner.clone().push(op),
        }
    }
}

/// What to compile a [`Pipeline`] against.
fn spec_of(like: &Bound<'_, PyAny>, color_space: Option<&Bound<'_, PyAny>>) -> PyResult<rp::ImageSpec> {
    if let Ok(s) = like.cast::<ImageSpec>() {
        return Ok(s.get().inner.clone());
    }
    if let Ok(img) = like.cast::<Image>() {
        return Ok(rp::ImageSpec::from_dynamic(&img.borrow().inner));
    }
    let px = NdPixels::extract(like)?;
    let (h, w, c) = px.hwc()?;
    let cs = infer_cspace(c, color_space.map(parse_cspace).transpose()?)?;
    Ok(rp::ImageSpec::new(w, h, cs, px.storage()))
}

#[pymethods]
impl Pipeline {
    /// An empty (identity) pipeline.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Demosaic a Bayer image to RGB: `"none"`, `"nearest"`, `"linear"` or `"cubic"`.
    #[pyo3(signature = (method = "linear"))]
    fn debayer(&self, method: &str) -> PyResult<Self> {
        Ok(self.with(Op::Debayer(parse_demosaic(method)?)))
    }

    /// Luma (Rec.601 by default, or per-channel `weights`); the image becomes gray.
    #[pyo3(signature = (weights = None))]
    fn to_luma(&self, weights: Option<Vec<f64>>) -> Self {
        self.with(weights.map_or(Op::ToLuma, Op::ToLumaCustom))
    }

    /// `y = x * gain + offset` on the raw stored value, saturated into the pixel type.
    #[pyo3(signature = (gain, offset = 0.0))]
    fn scale(&self, gain: f64, offset: f64) -> Self {
        self.with(Op::Scale { gain, offset })
    }

    /// `y = round(x * num / den)`, exact in integer arithmetic on integer images.
    fn scale_rational(&self, num: i64, den: i64) -> Self {
        self.with(Op::ScalePixels(rp::ScaleFactor::Rational { num, den }))
    }

    /// `y = x * factor`, saturated into the pixel type.
    fn scale_by(&self, factor: f64) -> Self {
        self.with(Op::ScalePixels(rp::ScaleFactor::Float(factor)))
    }

    /// Rescale into another storage type: `"u8"`, `"u16"` or `"f32"` (or a numpy dtype).
    fn convert(&self, pixel_type: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.with(Op::Convert(parse_pixel_type(pixel_type)?)))
    }

    /// The `width` x `height` rectangle at `(x, y)`; must lie inside the image.
    fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        self.with(Op::Crop {
            x,
            y,
            width,
            height,
        })
    }

    /// Like `crop`, but overhang past the edge is zero-filled.
    fn roi(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        self.with(Op::Roi {
            x,
            y,
            width,
            height,
        })
    }

    /// Mirror left to right (Bayer patterns are re-phased).
    fn flip_horizontal(&self) -> Self {
        self.with(Op::FlipHorizontal)
    }

    /// Mirror top to bottom (Bayer patterns are re-phased).
    fn flip_vertical(&self) -> Self {
        self.with(Op::FlipVertical)
    }

    /// Rotate 90 degrees clockwise.
    fn rotate_90(&self) -> Self {
        self.with(Op::Rotate90)
    }

    /// Rotate 180 degrees (Bayer patterns are re-phased).
    fn rotate_180(&self) -> Self {
        self.with(Op::Rotate180)
    }

    /// Rotate 90 degrees counter-clockwise.
    fn rotate_270(&self) -> Self {
        self.with(Op::Rotate270)
    }

    /// Resample to the largest size inside `max_width` x `max_height` at the same aspect
    /// ratio: `"bilinear"`, `"bicubic"` or `"lanczos3"`.
    #[pyo3(signature = (max_width, max_height, filter = "bilinear"))]
    fn resize_to_fit(&self, max_width: usize, max_height: usize, filter: &str) -> PyResult<Self> {
        Ok(self.with(Op::ResizeToFit {
            max_width,
            max_height,
            filter: parse_filter(filter)?,
        }))
    }

    /// Warp through `inverse_map`, a 3x3 homography taking `(column, row, 1)` of an
    /// output pixel to the input (scikit-image's convention), into an output of
    /// `output_shape` `(rows, cols)`. Spline `order` 0, 1 or 3 (cubic prefilters like
    /// scipy unless `prefilter=False`); `mode` is `"constant"`, `"nearest"`/`"edge"`,
    /// `"mirror"`, `"reflect"`/`"symmetric"` or `"wrap"`; `cval` fills `"constant"`, in raw
    /// stored units.
    #[pyo3(signature = (inverse_map, output_shape, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
    fn warp(
        &self,
        inverse_map: &Bound<'_, PyAny>,
        output_shape: (usize, usize),
        order: u8,
        mode: &str,
        cval: f64,
        prefilter: bool,
    ) -> PyResult<Self> {
        let (rows, cols) = output_shape;
        Ok(self.with(Op::Warp {
            inverse: matrix3(inverse_map)?,
            width: cols,
            height: rows,
            interp: parse_interp(order, mode, cval, prefilter)?,
        }))
    }

    /// Sample at `coordinates`, shape `(2, rows, cols)`: the input row and column of every
    /// output pixel (scipy's `map_coordinates`). Same interpolation arguments as `warp`.
    #[pyo3(signature = (coordinates, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
    fn map_coordinates(
        &self,
        coordinates: &Bound<'_, PyAny>,
        order: u8,
        mode: &str,
        cval: f64,
        prefilter: bool,
    ) -> PyResult<Self> {
        let (rows, cols, width, height) = coordinate_planes(coordinates)?;
        Ok(self.with(Op::MapCoordinates {
            rows,
            cols,
            width,
            height,
            interp: parse_interp(order, mode, cval, prefilter)?,
        }))
    }

    /// The same pipeline with no-ops dropped, flip/rotation runs folded and nested crops
    /// merged. Output-preserving.
    fn optimize(&self) -> Self {
        Self {
            inner: self.inner.clone().optimize(),
        }
    }

    /// The ops as JSON-style Python objects (dicts and strings).
    #[getter]
    fn ops<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let json = serde_json::to_string(self.inner.ops()).map_err(err)?;
        Ok(py.import("json")?.call_method1("loads", (json,))?.cast_into::<PyList>()?)
    }

    /// The recipe as JSON (the same serde representation Rust uses).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(err)
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Ok(Self {
            inner: serde_json::from_str(json).map_err(err)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        Ok((slf.get_type().getattr("from_json")?, (slf.get().to_json()?,)))
    }

    fn __len__(&self) -> usize {
        self.inner.ops().len()
    }

    fn __repr__(&self) -> String {
        let names: Vec<String> = self
            .inner
            .ops()
            .iter()
            .map(|op| match op {
                Op::MapCoordinates { width, height, .. } => format!("MapCoordinates({width}x{height})"),
                other => format!("{other:?}"),
            })
            .collect();
        format!("Pipeline([{}])", names.join(", "))
    }

    /// Run once and return a new image. `image` is a `refimage.Image` (its metadata is
    /// kept) or a numpy array (`color_space` as for `Image`).
    #[pyo3(signature = (image, *, color_space = None))]
    fn apply(
        &self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        color_space: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Image> {
        let out = if let Ok(img) = image.cast::<Image>() {
            let img = img.borrow();
            let src = &img.inner;
            py.detach(|| self.inner.apply(src))
        } else {
            let cs = color_space.map(parse_cspace).transpose()?;
            let src = Image::from_numpy(image, cs, Metadata::new(now(), Default::default()), None)?.inner;
            py.detach(|| self.inner.apply(&src))
        };
        Ok(Image::from_inner(out.map_err(err)?))
    }

    /// Validate the chain against `like` (an `Image`, a numpy array or an `ImageSpec`)
    /// and allocate every buffer, returning a `Runner` that reuses them frame after
    /// frame. `strategy="tiled"` (the default) cuts frames into bands of `tile_rows`
    /// rows (0 = automatic), optionally `tile_cols` wide, fanned over threads when
    /// `parallel`; `parallel=False` makes `run` allocation-free once compiled.
    /// `"sequential"` sweeps whole frames.
    #[pyo3(signature = (like, *, color_space = None, strategy = "tiled", tile_rows = 0, tile_cols = 0, parallel = true))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self,
        py: Python<'_>,
        like: &Bound<'_, PyAny>,
        color_space: Option<&Bound<'_, PyAny>>,
        strategy: &str,
        tile_rows: usize,
        tile_cols: usize,
        parallel: bool,
    ) -> PyResult<Runner> {
        let spec = spec_of(like, color_space)?;
        let strategy = parse_strategy(strategy, tile_rows, tile_cols, parallel)?;
        let inner = py.detach(|| self.inner.compile(spec, strategy)).map_err(err)?;
        Ok(Runner { inner })
    }
}

/// The shape and pixel type a pipeline is compiled for.
#[pyclass(module = "refimage", name = "ImageSpec", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub struct ImageSpec {
    pub(crate) inner: rp::ImageSpec,
}

#[pymethods]
impl ImageSpec {
    #[new]
    #[pyo3(signature = (width, height, color_space = None, pixel_type = None))]
    fn new(
        width: usize,
        height: usize,
        color_space: Option<&Bound<'_, PyAny>>,
        pixel_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let cs = color_space.map(parse_cspace).transpose()?.unwrap_or(ColorSpace::Gray);
        let pt = pixel_type.map(parse_pixel_type).transpose()?.unwrap_or(PixelType::U16);
        Ok(Self {
            inner: rp::ImageSpec::new(width, height, cs, pt),
        })
    }

    /// Width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }

    /// Height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }

    /// Interleaved channels per pixel.
    #[getter]
    fn channels(&self) -> u8 {
        self.inner.cspace.channels()
    }

    /// The color space, as `Image.color_space` reports it.
    #[getter]
    fn color_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cspace_to_py(py, &self.inner.cspace)
    }

    /// `"u8"`, `"u10"`, `"u12"`, `"u14"`, `"u16"` or `"f32"`.
    #[getter]
    fn pixel_type(&self) -> &'static str {
        pixel_type_name(self.inner.pixel_type)
    }

    /// The numpy shape of a matching frame.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        let c = self.inner.cspace.channels() as usize;
        if c == 1 {
            vec![self.inner.height, self.inner.width]
        } else {
            vec![self.inner.height, self.inner.width, c]
        }
    }

    /// The numpy dtype of a matching frame.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .call_method1("dtype", (dtype_name(self.inner.pixel_type),))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ImageSpec(width={}, height={}, color_space={}, pixel_type={:?})",
            self.inner.width,
            self.inner.height,
            cspace_to_py(py, &self.inner.cspace)?.repr()?,
            pixel_type_name(self.inner.pixel_type),
        ))
    }
}

/// A numpy array presented to the runner as a frame, without copying it.
struct NdFrame<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
    cspace: ColorSpace,
    pixel_type: PixelType,
}

impl ImageProps for NdFrame<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn channels(&self) -> u8 {
        self.cspace.channels()
    }
    fn color_space(&self) -> ColorSpace {
        self.cspace.clone()
    }
    fn pixel_type(&self) -> PixelType {
        self.pixel_type
    }
    fn len(&self) -> usize {
        self.width * self.height * self.cspace.channels() as usize
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl rp::Frame for NdFrame<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.bytes
    }
}

/// A compiled pipeline holding its scratch buffers. `run` reuses them for every frame;
/// pass `out=` to reuse the output image too.
#[pyclass(module = "refimage", name = "Runner")]
pub struct Runner {
    inner: rp::Runner,
}

/// Copy the runner's result into `out` (or a fresh image) with `meta`.
fn finish(
    result: &DynamicImageRef<'_>,
    out: Option<&mut Image>,
    meta: Option<&Metadata>,
) -> PyResult<Option<GenericImageOwned>> {
    match out {
        Some(o) => {
            o.store(result)?;
            if let Some(m) = meta {
                o.inner.metadata_mut().clone_from(m);
            }
            Ok(None)
        }
        None => {
            let meta = meta.cloned().unwrap_or_else(|| Metadata::new(now(), Default::default()));
            Ok(Some(GenericImageOwned::with_metadata(meta, result)))
        }
    }
}

#[pymethods]
impl Runner {
    /// Process one frame: a `refimage.Image` or a numpy array (read in place, no copy,
    /// when C-contiguous; its shape and dtype must match `input_spec` -- a different
    /// shape recompiles automatically). Returns the result as a new `Image` carrying
    /// the input's metadata, or, given `out`, writes into `out` and returns it: when
    /// `out` already has the output shape and type no pixel memory is allocated, and
    /// numpy views of `out` see the new frame. `out` may be the input image itself.
    #[pyo3(signature = (frame, out = None))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
        out: Option<Bound<'py, Image>>,
    ) -> PyResult<Bound<'py, Image>> {
        let runner = &mut self.inner;
        if let Ok(src) = frame.cast::<Image>() {
            if let Some(o) = out.as_ref().filter(|o| o.is(src)) {
                // In place: one mutable borrow serves as both input and output.
                let mut img = o.borrow_mut();
                let img = &mut *img;
                py.detach(|| -> PyResult<()> {
                    let r = runner.run(img.inner.image()).map_err(err)?;
                    img.store(&r)
                })?;
                return Ok(o.clone());
            }
            let src = src.borrow();
            let src = &src.inner;
            let mut dest = out.as_ref().map(|o| o.borrow_mut());
            let dest_ref = dest.as_deref_mut();
            let fresh = py.detach(|| -> PyResult<_> {
                let r = runner.run(src.image()).map_err(err)?;
                finish(&r, dest_ref, Some(src.metadata()))
            })?;
            drop(dest);
            return match fresh {
                Some(g) => Bound::new(py, Image::from_inner(g)),
                None => Ok(out.expect("out was given")),
            };
        }

        let px = NdPixels::extract(frame)?;
        let (h, w, c) = px.hwc()?;
        let want = runner.input_spec();
        // Keep the compiled color space / bit-depth tag when the array agrees with it.
        let cspace = if want.cspace.channels() as usize == c {
            want.cspace.clone()
        } else {
            infer_cspace(c, None)?
        };
        let pixel_type = if want.pixel_type.storage() == px.storage() {
            want.pixel_type
        } else {
            px.storage()
        };
        let bytes = px.bytes();
        let nd = NdFrame {
            bytes: &bytes,
            width: w,
            height: h,
            cspace,
            pixel_type,
        };
        let mut dest = out.as_ref().map(|o| o.borrow_mut());
        let dest_ref = dest.as_deref_mut();
        let fresh = py.detach(|| -> PyResult<_> {
            let r = runner.run(&nd).map_err(err)?;
            finish(&r, dest_ref, None)
        })?;
        drop(dest);
        match fresh {
            Some(g) => Bound::new(py, Image::from_inner(g)),
            None => Ok(out.expect("out was given")),
        }
    }

    /// `run` as a call: `runner(frame, out=None)`.
    #[pyo3(signature = (frame, out = None))]
    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
        out: Option<Bound<'py, Image>>,
    ) -> PyResult<Bound<'py, Image>> {
        self.run(py, frame, out)
    }

    /// The input shape this runner is compiled for.
    #[getter]
    fn input_spec(&self) -> ImageSpec {
        ImageSpec {
            inner: self.inner.input_spec().clone(),
        }
    }

    /// The shape `run` produces.
    #[getter]
    fn output_spec(&self) -> ImageSpec {
        ImageSpec {
            inner: self.inner.output_spec().clone(),
        }
    }

    /// Whether `run` cuts frames into tiles (`False` when tiling cannot help).
    #[getter]
    fn is_tiled(&self) -> bool {
        self.inner.is_tiled()
    }

    /// Independently tiled passes (0 when not tiled; more when a geometric op splits
    /// the chain).
    #[getter]
    fn tiled_pass_count(&self) -> usize {
        self.inner.tiled_pass_count()
    }

    /// Scratch memory held, in bytes.
    #[getter]
    fn scratch_bytes(&self) -> usize {
        self.inner.scratch_bytes()
    }

    /// Rebuild the plan and buffers for a new input (`Image`, array or `ImageSpec`).
    #[pyo3(signature = (like, *, color_space = None))]
    fn recompile(
        &mut self,
        py: Python<'_>,
        like: &Bound<'_, PyAny>,
        color_space: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let spec = spec_of(like, color_space)?;
        let runner = &mut self.inner;
        py.detach(|| runner.recompile(spec)).map_err(err)
    }

    fn __repr__(&self) -> String {
        let (i, o) = (self.inner.input_spec(), self.inner.output_spec());
        format!(
            "Runner({}x{} {:?} {} -> {}x{} {:?} {}, tiled={}, scratch={} B)",
            i.width,
            i.height,
            i.cspace,
            pixel_type_name(i.pixel_type),
            o.width,
            o.height,
            o.cspace,
            pixel_type_name(o.pixel_type),
            self.inner.is_tiled(),
            self.inner.scratch_bytes()
        )
    }
}
