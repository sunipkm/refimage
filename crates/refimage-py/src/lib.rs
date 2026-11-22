//! `refimage._native`: Python bindings for `refimage`.
//!
//! - [`image::Image`] (`refimage.Image`) wraps a `GenericImageOwned`: pixels plus metadata.
//!   It exports its pixel buffer through the buffer protocol, so `numpy.asarray(img)` is a
//!   zero-copy, writable view.
//! - [`pipeline::Pipeline`] / [`pipeline::Runner`] wrap the `refimage::pipeline` recipe and
//!   its compiled, buffer-reusing runner, including the warp / map-coordinates ops.
//! - [`exposure::OptimumExposure`] wraps the optimum-exposure calculator.
//! - [`interop`] serves the `refimage-pyo3` crate, through which other extension modules
//!   create and read these objects without copying pixels.
//!
//! Every pixel-processing call releases the GIL.

mod exposure;
mod image;
mod interop;
mod pipeline;
mod store;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use refimage::pipeline::{BoundaryMode, InterpOrder, Interpolation, ResizeFilter, Strategy};
use refimage::{BayerPattern, ColorSpace, DemosaicMethod, PixelType};

pyo3::create_exception!(
    refimage,
    RefImageError,
    PyValueError,
    "Raised for invalid images, pipelines, metadata and FITS output."
);

/// Wrap any `refimage` error as a [`RefImageError`].
pub(crate) fn err(e: impl std::fmt::Display) -> PyErr {
    RefImageError::new_err(e.to_string())
}

// --- Enum names ----------------------------------------------------------------

/// A color space from `"gray"`, `"rgb"`, a Bayer pattern (`"rggb"`, `"bayer_rggb"`, ...),
/// or a `(channels, name)` tuple for a custom one.
pub(crate) fn parse_cspace(obj: &Bound<'_, PyAny>) -> PyResult<ColorSpace> {
    if let Ok(t) = obj.cast::<PyTuple>() {
        let (n, name): (u8, String) = t.extract()?;
        if n == 0 {
            return Err(err("a custom color space needs at least one channel"));
        }
        return Ok(ColorSpace::Custom(n, name));
    }
    let s = obj
        .cast::<PyString>()
        .map_err(|_| err("color_space must be a string or a (channels, name) tuple"))?
        .to_cow()?
        .to_ascii_lowercase();
    let pat = s
        .strip_prefix("bayer_")
        .or_else(|| s.strip_prefix("bayer:"))
        .unwrap_or(&s);
    Ok(match pat {
        "gray" | "grey" | "mono" | "l" => ColorSpace::Gray,
        "rgb" => ColorSpace::Rgb,
        "rggb" => ColorSpace::Bayer(BayerPattern::Rggb),
        "bggr" => ColorSpace::Bayer(BayerPattern::Bggr),
        "gbrg" => ColorSpace::Bayer(BayerPattern::Gbrg),
        "grbg" => ColorSpace::Bayer(BayerPattern::Grbg),
        _ => return Err(err(format!("unknown color space {s:?}"))),
    })
}

pub(crate) fn cspace_to_py<'py>(py: Python<'py>, cs: &ColorSpace) -> PyResult<Bound<'py, PyAny>> {
    let name = match cs {
        ColorSpace::Gray => "gray",
        ColorSpace::Rgb => "rgb",
        ColorSpace::Bayer(BayerPattern::Rggb) => "rggb",
        ColorSpace::Bayer(BayerPattern::Bggr) => "bggr",
        ColorSpace::Bayer(BayerPattern::Gbrg) => "gbrg",
        ColorSpace::Bayer(BayerPattern::Grbg) => "grbg",
        ColorSpace::Custom(n, name) => {
            return Ok((*n, name.as_str()).into_pyobject(py)?.into_any());
        }
        other => return Ok(PyString::new(py, &format!("{other:?}")).into_any()),
    };
    Ok(PyString::new(py, name).into_any())
}

/// A pixel type from `"u8"`, `"u10"`, `"u12"`, `"u14"`, `"u16"`, `"f32"`, or anything
/// `numpy.dtype` accepts (`np.uint8`, `"float32"`, a dtype, ...).
pub(crate) fn parse_pixel_type(obj: &Bound<'_, PyAny>) -> PyResult<PixelType> {
    let name: String = match obj.cast::<PyString>() {
        Ok(s) => s.to_cow()?.to_ascii_lowercase(),
        Err(_) => obj
            .py()
            .import("numpy")?
            .call_method1("dtype", (obj,))?
            .getattr("name")?
            .extract()?,
    };
    Ok(match name.as_str() {
        "u8" | "uint8" => PixelType::U8,
        "u10" => PixelType::U10,
        "u12" => PixelType::U12,
        "u14" => PixelType::U14,
        "u16" | "uint16" => PixelType::U16,
        "f32" | "float32" => PixelType::F32,
        _ => {
            return Err(err(format!(
                "unsupported pixel type {name:?} (use u8, u10, u12, u14, u16 or f32)"
            )));
        }
    })
}

pub(crate) fn pixel_type_name(pt: PixelType) -> &'static str {
    match pt {
        PixelType::U8 => "u8",
        PixelType::U10 => "u10",
        PixelType::U12 => "u12",
        PixelType::U14 => "u14",
        PixelType::U16 => "u16",
        _ => "f32",
    }
}

/// numpy's name for the storage dtype of `pt`.
pub(crate) fn dtype_name(pt: PixelType) -> &'static str {
    match pt.storage() {
        PixelType::U8 => "uint8",
        PixelType::U16 => "uint16",
        _ => "float32",
    }
}

pub(crate) fn parse_demosaic(s: &str) -> PyResult<DemosaicMethod> {
    Ok(match s {
        "none" => DemosaicMethod::None,
        "nearest" => DemosaicMethod::Nearest,
        "linear" => DemosaicMethod::Linear,
        "cubic" => DemosaicMethod::Cubic,
        _ => {
            return Err(err(format!(
                "unknown demosaic method {s:?} (use none, nearest, linear or cubic)"
            )));
        }
    })
}

pub(crate) fn parse_filter(s: &str) -> PyResult<ResizeFilter> {
    Ok(match s {
        "bilinear" | "linear" => ResizeFilter::Bilinear,
        "bicubic" | "cubic" => ResizeFilter::Bicubic,
        "lanczos3" | "lanczos" => ResizeFilter::Lanczos3,
        _ => {
            return Err(err(format!(
                "unknown resize filter {s:?} (use bilinear, bicubic or lanczos3)"
            )));
        }
    })
}

pub(crate) fn parse_interp(
    order: u8,
    mode: &str,
    cval: f64,
    prefilter: bool,
) -> PyResult<Interpolation> {
    let order = InterpOrder::from_u8(order).map_err(err)?;
    let mode: BoundaryMode = mode.parse().map_err(err)?;
    Ok(Interpolation::new(order, mode, cval).with_prefilter(prefilter))
}

pub(crate) fn parse_strategy(
    strategy: &str,
    tile_rows: usize,
    tile_cols: usize,
    parallel: bool,
) -> PyResult<Strategy> {
    match strategy {
        "sequential" => Ok(Strategy::Sequential),
        "tiled" => Ok(Strategy::Tiled {
            tile_rows,
            tile_cols,
            parallel,
        }),
        _ => Err(err(format!(
            "unknown strategy {strategy:?} (use \"tiled\" or \"sequential\")"
        ))),
    }
}

// --- numpy input ---------------------------------------------------------------

/// A read-only numpy array of one of the three storage dtypes.
pub(crate) enum NdPixels<'py> {
    U8(PyReadonlyArrayDyn<'py, u8>),
    U16(PyReadonlyArrayDyn<'py, u16>),
    F32(PyReadonlyArrayDyn<'py, f32>),
}

impl<'py> NdPixels<'py> {
    pub(crate) fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<'py, u8>>() {
            Ok(Self::U8(a))
        } else if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<'py, u16>>() {
            Ok(Self::U16(a))
        } else if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<'py, f32>>() {
            Ok(Self::F32(a))
        } else {
            Err(err(
                "expected a refimage.Image or a numpy array of dtype uint8, uint16 or float32",
            ))
        }
    }

    pub(crate) fn storage(&self) -> PixelType {
        match self {
            Self::U8(_) => PixelType::U8,
            Self::U16(_) => PixelType::U16,
            Self::F32(_) => PixelType::F32,
        }
    }

    /// `(height, width, channels)` of a `(H, W)` or `(H, W, C)` array.
    pub(crate) fn hwc(&self) -> PyResult<(usize, usize, usize)> {
        let shape = match self {
            Self::U8(a) => a.shape(),
            Self::U16(a) => a.shape(),
            Self::F32(a) => a.shape(),
        };
        match *shape {
            [h, w] => Ok((h, w, 1)),
            [h, w, c] => Ok((h, w, c)),
            _ => Err(err(format!(
                "image arrays must be (height, width) or (height, width, channels), not {shape:?}"
            ))),
        }
    }

    /// The samples as native-endian bytes: borrowed when the array is C-contiguous,
    /// copied (row-major) otherwise.
    pub(crate) fn bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        use std::borrow::Cow;
        macro_rules! bytes {
            ($a:expr) => {
                match $a.as_slice() {
                    Ok(s) => Cow::Borrowed(bytemuck::cast_slice(s)),
                    Err(_) => Cow::Owned(
                        bytemuck::cast_slice(&$a.as_array().iter().copied().collect::<Vec<_>>())
                            .to_vec(),
                    ),
                }
            };
        }
        match self {
            Self::U8(a) => bytes!(a),
            Self::U16(a) => bytes!(a),
            Self::F32(a) => bytes!(a),
        }
    }
}

/// The color space for `channels` channels: `given` when it agrees, else gray / RGB.
pub(crate) fn infer_cspace(channels: usize, given: Option<ColorSpace>) -> PyResult<ColorSpace> {
    match given {
        Some(cs) if cs.channels() as usize == channels => Ok(cs),
        Some(cs) => Err(err(format!(
            "color space {cs:?} has {} channel(s) but the array has {channels}",
            cs.channels()
        ))),
        None => match channels {
            1 => Ok(ColorSpace::Gray),
            3 => Ok(ColorSpace::Rgb),
            n => Err(err(format!(
                "cannot infer a color space for {n} channels; pass color_space=({n}, \"name\")"
            ))),
        },
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RefImageError", m.py().get_type::<RefImageError>())?;
    m.add_class::<image::Image>()?;
    m.add_class::<pipeline::Pipeline>()?;
    m.add_class::<pipeline::ImageSpec>()?;
    m.add_class::<pipeline::Runner>()?;
    m.add_class::<exposure::OptimumExposure>()?;
    m.add_class::<exposure::ExposureResult>()?;
    // The exchange format of refimage-pyo3 (other extension modules' images).
    m.add_function(wrap_pyfunction!(interop::_interop_protocol, m)?)?;
    m.add_function(wrap_pyfunction!(interop::_image_from_parts, m)?)?;
    m.add_function(wrap_pyfunction!(interop::_image_to_parts, m)?)?;
    m.add_function(wrap_pyfunction!(interop::_image_set_metadata, m)?)?;
    Ok(())
}
