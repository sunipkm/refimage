//! `refimage.Image`: pixels + metadata (a [`Store`]: owned, or adopted from another
//! extension module's numpy array) with a buffer-protocol export so numpy can view the
//! pixels without copying.

use std::ffi::{CStr, c_int};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use pyo3::exceptions::{PyBufferError, PyKeyError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyString};
use refimage::chrono::{DateTime, Utc};
use refimage::pipeline::Pipeline;
use refimage::{
    ColorSpace, DynamicImageOwned, DynamicImageRef, FitsCompression, FitsWrite, GenericImageOwned,
    GenericValue, Gzip, Hcompress, ImageOwned, ImageProps, InsertValue, Metadata, PixelType,
    Quantize, Rice,
};

use crate::store::Store;

use crate::{
    NdPixels, cspace_to_py, dtype_name, err, infer_cspace, parse_cspace, parse_interp,
    pixel_type_name,
};

/// An image: pixels (`uint8`, `uint16` -- optionally tagged 10/12/14-bit -- or `float32`,
/// row-major, channels interleaved) plus metadata (timestamp, exposure, frame id and any
/// number of named FITS-style cards).
///
/// `numpy.asarray(img)` (or `img.numpy()`) is a zero-copy, writable view of the pixels,
/// shaped `(height, width)` for one channel or `(height, width, channels)`.
#[pyclass(module = "refimage", name = "Image")]
pub struct Image {
    pub(crate) store: Store,
    /// Live buffer-protocol exports. While non-zero the pixel buffer must not be
    /// reallocated, because numpy arrays point into it.
    exports: AtomicUsize,
}

/// An exposure as float seconds or a `datetime.timedelta`.
#[derive(FromPyObject)]
pub(crate) enum Exposure {
    Delta(Duration),
    Seconds(f64),
}

impl Exposure {
    pub(crate) fn duration(self) -> PyResult<Duration> {
        match self {
            Self::Delta(d) => Ok(d),
            Self::Seconds(s) => Duration::try_from_secs_f64(s)
                .map_err(|_| err(format!("exposure must be finite and non-negative, not {s}"))),
        }
    }
}

impl Image {
    pub(crate) fn from_inner(inner: GenericImageOwned) -> Self {
        Self::from_store(Store::owned(inner))
    }

    pub(crate) fn from_store(store: Store) -> Self {
        Self {
            store,
            exports: AtomicUsize::new(0),
        }
    }

    /// Copy `src` into this image's pixels, reusing the buffer when the shape and type
    /// already match (no allocation); otherwise replace it, which is refused while numpy
    /// views of the old buffer exist.
    pub(crate) fn store(&mut self, src: &DynamicImageRef<'_>) -> PyResult<()> {
        // Same shape and storage: copied in place (into an adopted buffer too).
        if self.store.write(src).is_ok() {
            return Ok(());
        }
        if self.exports.load(Ordering::Acquire) > 0 {
            return Err(PyBufferError::new_err(
                "the output image changes shape or type, but numpy arrays still view its \
                 pixels; drop them (or pass a fresh `out`) first",
            ));
        }
        self.store.replace(src);
        Ok(())
    }

    pub(crate) fn from_numpy(
        data: &Bound<'_, PyAny>,
        cspace: Option<ColorSpace>,
        metadata: Metadata,
        bit_depth: Option<u8>,
    ) -> PyResult<Self> {
        let px = NdPixels::extract(data)?;
        let (h, w, c) = px.hwc()?;
        let cs = infer_cspace(c, cspace)?;
        let owned: DynamicImageOwned = match &px {
            NdPixels::U8(a) => {
                ImageOwned::from_owned(a.as_array().iter().copied().collect(), w, h, cs)
                    .map_err(err)?
                    .into()
            }
            NdPixels::U16(a) => {
                ImageOwned::from_owned(a.as_array().iter().copied().collect(), w, h, cs)
                    .map_err(err)?
                    .with_bit_depth(bit_depth)
                    .into()
            }
            NdPixels::F32(a) => {
                ImageOwned::from_owned(a.as_array().iter().copied().collect(), w, h, cs)
                    .map_err(err)?
                    .into()
            }
        };
        if bit_depth.is_some() && px.storage() != PixelType::U16 {
            return Err(err("bit_depth only applies to uint16 images"));
        }
        Ok(Self::from_inner(GenericImageOwned::with_metadata(
            metadata, owned,
        )))
    }

    fn item_size(&self) -> usize {
        match self.store.get().pixel_type().storage() {
            PixelType::U8 => 1,
            PixelType::U16 => 2,
            _ => 4,
        }
    }
}

pub(crate) fn now() -> DateTime<Utc> {
    Utc::now()
}

fn value_to_py<'py>(py: Python<'py>, v: &GenericValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        GenericValue::Integer(i) => i.into_pyobject(py)?.into_any(),
        GenericValue::Real(f) => f.into_pyobject(py)?.into_any(),
        GenericValue::ColorSpace(cs) => cspace_to_py(py, cs)?,
        GenericValue::Duration(d) => d.into_pyobject(py)?.into_any(),
        GenericValue::Timestamp(t) => t.into_pyobject(py)?.into_any(),
        GenericValue::String(s) => PyString::new(py, s).into_any(),
        other => PyString::new(py, &format!("{other:?}")).into_any(),
    })
}

fn put<T: InsertValue>(meta: &mut Metadata, key: &str, value: T) -> PyResult<()> {
    if meta.get(key).is_some() {
        meta.replace(key, value).map_err(err)?;
    } else {
        meta.insert(key, value).map_err(err)?;
    }
    Ok(())
}

/// Insert or replace `key`, with an optional comment.
fn set_key(
    meta: &mut Metadata,
    key: &str,
    value: &Bound<'_, PyAny>,
    comment: Option<&str>,
) -> PyResult<()> {
    macro_rules! put {
        ($v:expr) => {
            match comment {
                Some(c) => put(meta, key, ($v, c)),
                None => put(meta, key, $v),
            }
        };
    }
    if value.is_instance_of::<PyInt>() {
        put!(value.extract::<i64>()?)
    } else if value.is_instance_of::<PyFloat>() {
        put!(value.extract::<f64>()?)
    } else if let Ok(s) = value.cast::<PyString>() {
        put!(s.to_cow()?.into_owned())
    } else if let Ok(t) = value.extract::<DateTime<Utc>>() {
        put!(t)
    } else if let Ok(d) = value.extract::<Duration>() {
        put!(d)
    } else {
        Err(PyTypeError::new_err(format!(
            "metadata values must be int, float, str, datetime or timedelta, not {}",
            value.get_type().name()?
        )))
    }
}

fn compression(
    kind: Option<&str>,
    level: Option<u8>,
    tile_rows: Option<usize>,
    scale: i32,
    smooth: bool,
    quantize: Option<f64>,
) -> PyResult<FitsCompression> {
    let q = quantize.map_or_else(Quantize::new, |l| Quantize::new().level(l));
    Ok(match kind {
        None | Some("none") => FitsCompression::NONE,
        Some("gzip") => {
            let g = level.map_or_else(Gzip::new, |l| Gzip::new().level(l));
            match tile_rows {
                Some(r) => g.tile_rows(r).into(),
                None => g.into(),
            }
        }
        Some("rice") => {
            let r = Rice::new().quantize(q);
            match tile_rows {
                Some(t) => r.tile_rows(t).into(),
                None => r.into(),
            }
        }
        Some("hcompress") => {
            let h = Hcompress::new().scale(scale).smooth(smooth).quantize(q);
            match tile_rows {
                Some(t) => h.tile_rows(t).into(),
                None => h.into(),
            }
        }
        Some(other) => {
            return Err(err(format!(
                "unknown FITS compression {other:?} (use None, \"gzip\", \"rice\" or \"hcompress\")"
            )));
        }
    })
}

#[pymethods]
impl Image {
    /// Copy a numpy array (`uint8`, `uint16` or `float32`; `(H, W)` or `(H, W, C)`) into a
    /// new image. `color_space` defaults to `"gray"` for one channel and `"rgb"` for three;
    /// Bayer data is `"rggb"`, `"bggr"`, `"gbrg"` or `"grbg"`, anything else a
    /// `(channels, name)` tuple. `timestamp` (an aware `datetime`) defaults to now;
    /// `exposure` is seconds or a `timedelta`; `bit_depth` tags `uint16` data as 10, 12 or
    /// 14 bit.
    #[new]
    #[pyo3(signature = (data, color_space = None, *, timestamp = None, exposure = None, bit_depth = None, frame_id = None))]
    fn new(
        data: &Bound<'_, PyAny>,
        color_space: Option<&Bound<'_, PyAny>>,
        timestamp: Option<DateTime<Utc>>,
        exposure: Option<Exposure>,
        bit_depth: Option<u8>,
        frame_id: Option<u32>,
    ) -> PyResult<Self> {
        let cs = color_space.map(parse_cspace).transpose()?;
        let exposure = exposure
            .map(Exposure::duration)
            .transpose()?
            .unwrap_or_default();
        let mut meta = Metadata::new(timestamp.unwrap_or_else(now), exposure);
        if let Some(id) = frame_id {
            meta.set_frame_id(id);
        }
        Self::from_numpy(data, cs, meta, bit_depth)
    }

    /// Decode an image serialized with `to_bytes()` -- or by Rust with
    /// `bincode::serialize` of a `GenericImageRef` / `GenericImageOwned`.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let inner: GenericImageOwned = py
            .detach(|| bincode::deserialize(data))
            .map_err(|e| err(format!("not a serialized refimage image: {e}")))?;
        Ok(Self::from_inner(inner))
    }

    /// Serialize pixels and metadata (bincode, the same bytes Rust's
    /// `bincode::serialize(&generic_image)` produces).
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let store = &self.store;
        let bytes = py.detach(|| store.serialize()).map_err(err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let ctor = slf.get_type().getattr("from_bytes")?;
        Ok((ctor, (slf.borrow().to_bytes(slf.py())?,)))
    }

    /// A deep copy (pixels and metadata).
    fn copy(&self) -> Self {
        Self::from_inner(self.store.to_owned_image())
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Image>()
            .is_ok_and(|o| o.borrow().store.same(&self.store))
    }

    fn __repr__(&self) -> String {
        let cs = match self.store.get().color_space() {
            ColorSpace::Custom(n, name) => format!("{name}[{n}]"),
            ColorSpace::Bayer(p) => format!("{p:?}").to_lowercase(),
            other => format!("{other:?}").to_lowercase(),
        };
        format!(
            "Image({}x{} {} {}, exposure={}s, timestamp={})",
            self.store.get().width(),
            self.store.get().height(),
            cs,
            pixel_type_name(self.store.get().pixel_type()),
            self.store.get().exposure().as_secs_f64(),
            self.store.get().timestamp().to_rfc3339(),
        )
    }

    /// A zero-copy, writable numpy view of the pixels (same as `numpy.asarray(img)`).
    fn numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("numpy")?.call_method1("asarray", (slf,))
    }

    /// Width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.store.get().width()
    }

    /// Height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.store.get().height()
    }

    /// Interleaved channels per pixel (1 for gray and Bayer, 3 for RGB).
    #[getter]
    fn channels(&self) -> u8 {
        self.store.get().channels()
    }

    /// The numpy shape: `(height, width)` or `(height, width, channels)`.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        let (h, w, c) = (
            self.store.get().height(),
            self.store.get().width(),
            self.store.get().channels(),
        );
        if c == 1 {
            vec![h, w]
        } else {
            vec![h, w, c as usize]
        }
    }

    /// The numpy dtype of the pixels.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .call_method1("dtype", (dtype_name(self.store.get().pixel_type()),))
    }

    /// `"u8"`, `"u10"`, `"u12"`, `"u14"`, `"u16"` or `"f32"`.
    #[getter]
    fn pixel_type(&self) -> &'static str {
        pixel_type_name(self.store.get().pixel_type())
    }

    /// Meaningful bits per sample (10/12/14 for tagged `uint16` data).
    #[getter]
    fn bit_depth(&self) -> u8 {
        self.store.get().pixel_type().bit_depth()
    }

    /// `"gray"`, `"rgb"`, a Bayer pattern (`"rggb"`, `"bggr"`, `"gbrg"`, `"grbg"`) or a `(channels, name)` tuple.
    #[getter]
    fn color_space<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cspace_to_py(py, &self.store.get().color_space())
    }

    /// Size of the pixel data in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.store.get().len() * self.item_size()
    }

    /// Acquisition time (UTC `datetime`).
    #[getter]
    fn timestamp(&self) -> DateTime<Utc> {
        self.store.get().timestamp()
    }

    #[setter]
    fn set_timestamp(&mut self, ts: DateTime<Utc>) {
        self.store.metadata_mut().set_timestamp(ts);
    }

    /// Exposure in seconds (`0.0` when unknown). Set with seconds or a `timedelta`.
    #[getter]
    fn exposure(&self) -> f64 {
        self.store.get().exposure().as_secs_f64()
    }

    #[setter]
    fn set_exposure(&mut self, exposure: Exposure) -> PyResult<()> {
        self.store.metadata_mut().set_exposure(exposure.duration()?);
        Ok(())
    }

    /// Acquisition frame counter, or `None`.
    #[getter]
    fn frame_id(&self) -> Option<u32> {
        self.store.get().frame_id()
    }

    #[setter]
    fn set_frame_id(&mut self, id: u32) {
        self.store.metadata_mut().set_frame_id(id);
    }

    /// The extra metadata cards as an ordered `{KEY: value}` dict (a copy).
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, item) in self.store.get().metadata().iter() {
            d.set_item(k, value_to_py(py, item.value())?)?;
        }
        Ok(d)
    }

    /// The value of metadata card `key` (case-insensitive), or `default`.
    #[pyo3(signature = (key, default = None))]
    fn get_key<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.store.get().key(key) {
            Some(item) => Ok(Some(value_to_py(py, item.value())?)),
            None => Ok(default),
        }
    }

    /// The comment of metadata card `key`, or `None`.
    fn get_comment(&self, key: &str) -> Option<String> {
        self.store.get().key(key)?.comment().map(str::to_owned)
    }

    /// Insert or replace metadata card `key` (stored upper-case; at most 80 characters)
    /// with an int, float, str, aware datetime or timedelta, and an optional comment.
    /// `TIMESTAMP`, `EXPOSURE` and `FRAMEID` are reserved: use the properties.
    #[pyo3(signature = (key, value, comment = None))]
    fn set_key(
        &mut self,
        key: &str,
        value: &Bound<'_, PyAny>,
        comment: Option<&str>,
    ) -> PyResult<()> {
        set_key(self.store.metadata_mut(), key, value, comment)
    }

    /// Remove metadata card `key` and return its value.
    fn remove_key<'py>(&mut self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        if self.store.get().key(key).is_none() {
            return Err(PyKeyError::new_err(key.to_owned()));
        }
        let item = self.store.metadata_mut().remove(key).map_err(err)?;
        value_to_py(py, item.value())
    }

    /// Run `pipeline` once and return a new image with this image's metadata.
    fn apply(&self, py: Python<'_>, pipeline: &crate::pipeline::Pipeline) -> PyResult<Self> {
        let store = &self.store;
        let out = py.detach(|| store.apply(&pipeline.inner)).map_err(err)?;
        Ok(Self::from_inner(out))
    }

    /// Warp through `inverse_map`, a 3x3 homography taking `(column, row, 1)` of an output
    /// pixel to the input (scikit-image's convention), into `output_shape` `(rows, cols)`
    /// (this image's by default). Spline `order` 0, 1 or 3; `mode` is a scipy boundary
    /// mode (`"constant"`, `"nearest"`/`"edge"`, `"mirror"`, `"reflect"`/`"symmetric"`,
    /// `"wrap"`); `cval` fills `"constant"` in raw stored units. Same as
    /// `Pipeline().warp(...)`; returns a new image with this image's metadata.
    #[pyo3(signature = (inverse_map, output_shape = None, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
    #[allow(clippy::too_many_arguments)]
    fn warp(
        &self,
        py: Python<'_>,
        inverse_map: &Bound<'_, PyAny>,
        output_shape: Option<(usize, usize)>,
        order: u8,
        mode: &str,
        cval: f64,
        prefilter: bool,
    ) -> PyResult<Self> {
        let inverse = crate::pipeline::matrix3(inverse_map)?;
        let (rows, cols) =
            output_shape.unwrap_or((self.store.get().height(), self.store.get().width()));
        let interp = parse_interp(order, mode, cval, prefilter)?;
        let p = Pipeline::new().warp(inverse, cols, rows, interp);
        let store = &self.store;
        let out = py.detach(|| store.apply(&p)).map_err(err)?;
        Ok(Self::from_inner(out))
    }

    /// Sample at `coordinates`, shape `(2, rows, cols)` holding the input row and column of
    /// every output pixel (scipy's `map_coordinates` convention). Same arguments as
    /// `warp`; returns a new `(rows, cols)` image with this image's metadata.
    #[pyo3(signature = (coordinates, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
    fn map_coordinates(
        &self,
        py: Python<'_>,
        coordinates: &Bound<'_, PyAny>,
        order: u8,
        mode: &str,
        cval: f64,
        prefilter: bool,
    ) -> PyResult<Self> {
        let (rows, cols, w, h) = crate::pipeline::coordinate_planes(coordinates)?;
        let interp = parse_interp(order, mode, cval, prefilter)?;
        let p = Pipeline::new().map_coordinates(rows, cols, w, h, interp);
        let store = &self.store;
        let out = py.detach(|| store.apply(&p)).map_err(err)?;
        Ok(Self::from_inner(out))
    }

    /// Write the image and its metadata to a FITS file at `path`. `compression` is
    /// `None`, `"gzip"` (`level` 0-9), `"rice"` or `"hcompress"` (`scale`, `smooth`);
    /// `tile_rows` sets rows per compressed tile and `quantize` the float quantization
    /// level. Returns the path written.
    #[pyo3(signature = (path, compression = None, *, overwrite = false, level = None, tile_rows = None, scale = 0, smooth = false, quantize = None))]
    #[allow(clippy::too_many_arguments)]
    fn write_fits(
        &self,
        py: Python<'_>,
        path: std::path::PathBuf,
        compression: Option<&str>,
        overwrite: bool,
        level: Option<u8>,
        tile_rows: Option<usize>,
        scale: i32,
        smooth: bool,
        quantize: Option<f64>,
    ) -> PyResult<std::path::PathBuf> {
        let c = self::compression(compression, level, tile_rows, scale, smooth, quantize)?;
        let store = &self.store;
        py.detach(|| {
            store.with_generic(
                |g| g.write_fits(&path, c.clone(), overwrite),
                |g| g.write_fits(&path, c.clone(), overwrite),
            )
        })
        .map_err(|e| match e {
            refimage::FitsError::Io(io) => PyErr::from(io),
            other => err(other),
        })
    }

    /// The FITS file `write_fits` would write, as bytes.
    #[pyo3(signature = (compression = None, *, level = None, tile_rows = None, scale = 0, smooth = false, quantize = None))]
    #[allow(clippy::too_many_arguments)]
    fn to_fits_bytes<'py>(
        &self,
        py: Python<'py>,
        compression: Option<&str>,
        level: Option<u8>,
        tile_rows: Option<usize>,
        scale: i32,
        smooth: bool,
        quantize: Option<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let c = self::compression(compression, level, tile_rows, scale, smooth, quantize)?;
        let store = &self.store;
        let bytes = py
            .detach(|| store.with_generic(|g| g.fits_bytes(c.clone()), |g| g.fits_bytes(c.clone())))
            .map_err(err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Export the pixels as a C-contiguous, writable buffer.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null Py_buffer"));
        }
        if flags & ffi::PyBUF_F_CONTIGUOUS == ffi::PyBUF_F_CONTIGUOUS
            && flags & ffi::PyBUF_C_CONTIGUOUS != ffi::PyBUF_C_CONTIGUOUS
        {
            let me = slf.borrow();
            if me.store.get().width() > 1 && me.store.get().height() > 1 {
                return Err(PyBufferError::new_err(
                    "image pixels are C-contiguous, not Fortran",
                ));
            }
        }
        let mut me = slf.borrow_mut();
        let (h, w, c) = (
            me.store.get().height(),
            me.store.get().width(),
            me.store.get().channels() as usize,
        );
        let item = me.item_size();
        let format: &'static CStr = match me.store.get().pixel_type().storage() {
            PixelType::U8 => c"B",
            PixelType::U16 => c"H",
            _ => c"f",
        };
        let len = h * w * c * item;
        let raw = me.store.bytes_mut();
        if raw.len() < len {
            return Err(PyBufferError::new_err(
                "pixel buffer shorter than its shape",
            ));
        }
        let buf = raw.as_mut_ptr();
        // shape[0..3], strides[3..6]; freed in __releasebuffer__.
        let dims: Box<[ffi::Py_ssize_t; 6]> = Box::new([
            h as ffi::Py_ssize_t,
            w as ffi::Py_ssize_t,
            c as ffi::Py_ssize_t,
            (w * c * item) as ffi::Py_ssize_t,
            (c * item) as ffi::Py_ssize_t,
            item as ffi::Py_ssize_t,
        ]);
        let dims = Box::into_raw(dims).cast::<ffi::Py_ssize_t>();
        me.exports.fetch_add(1, Ordering::AcqRel);
        drop(me);
        // SAFETY: `view` is non-null and owned by the caller for the export's lifetime;
        // `buf` stays valid while `exports > 0` because nothing reallocates the pixels then.
        unsafe {
            (*view).buf = buf.cast();
            (*view).obj = slf.into_any().into_ptr();
            (*view).len = len as ffi::Py_ssize_t;
            (*view).readonly = 0;
            (*view).itemsize = item as ffi::Py_ssize_t;
            (*view).format = if flags & ffi::PyBUF_FORMAT == ffi::PyBUF_FORMAT {
                format.as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = if c == 1 { 2 } else { 3 };
            (*view).shape = if flags & ffi::PyBUF_ND == ffi::PyBUF_ND {
                dims
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
                dims.add(3)
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = dims.cast();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // SAFETY: `internal` is the box `__getbuffer__` leaked for this export.
        unsafe {
            drop(Box::from_raw(
                (*view).internal.cast::<[ffi::Py_ssize_t; 6]>(),
            ));
        }
        self.exports.fetch_sub(1, Ordering::AcqRel);
    }
}
