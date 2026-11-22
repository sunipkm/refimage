//! The exchange format ("parts") between refimage images in Rust and the `refimage` Python
//! package: a plain Python dict holding a numpy array and bytes, which any extension module
//! can build and read without access to another module's Rust types.
//!
//! The same code runs on both sides: in `refimage._native` (whose `_image_from_parts`,
//! `_image_to_parts` and `_image_set_metadata` call it) and in every extension that links
//! this crate. [`PROTOCOL`](crate::PROTOCOL) names the format; both sides check that they
//! agree before exchanging anything.
//!
//! # The format (protocol 1)
//!
//! An **image** is a dict:
//!
//! | key | value |
//! |---|---|
//! | `"header"` | `bytes`: a [`Header`] (width, height, color space, pixel type, metadata), bincode-encoded |
//! | `"pixels"` | a numpy array of the storage type (`uint8`, `uint16` or `float32`), shaped `(height, width)` for one channel or `(height, width, channels)` |
//!
//! Metadata alone (`_image_set_metadata`) is a bincode-encoded
//! [`Metadata`](refimage::Metadata).
//!
//! The pixels are row-major with channels interleaved, in native byte order. A sender may
//! pass any such array; arrays that are C-contiguous, aligned and writable are adopted in
//! place by the receiver, anything else is copied.

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use refimage::pipeline::ImageSpec;
use refimage::{
    ColorSpace, DynamicImageOwned, GenericImageOwned, ImageOwned, ImageProps, Metadata, PixelType,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Everything about an image except its pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The color space (and so the channel count).
    pub color_space: ColorSpace,
    /// The pixel type, including the 10/12/14-bit tags of 16-bit storage.
    pub pixel_type: PixelType,
    /// Timestamp, exposure, frame id and metadata cards.
    pub metadata: Metadata,
}

impl Header {
    /// The header of an image with this `metadata` and the shape of `img`.
    pub fn new(metadata: Metadata, img: &impl ImageProps) -> Self {
        Self {
            width: img.width(),
            height: img.height(),
            color_space: img.color_space(),
            pixel_type: img.pixel_type(),
            metadata,
        }
    }

    /// Interleaved channels per pixel.
    pub fn channels(&self) -> usize {
        self.color_space.channels() as usize
    }

    /// The numpy shape of the pixels.
    pub fn shape(&self) -> Vec<usize> {
        match self.channels() {
            1 => vec![self.height, self.width],
            c => vec![self.height, self.width, c],
        }
    }

    /// The shape and type, as a pipeline sees it.
    pub fn spec(&self) -> ImageSpec {
        ImageSpec::new(
            self.width,
            self.height,
            self.color_space.clone(),
            self.pixel_type,
        )
    }
}

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("refimage parts: {e}"))
}

/// numpy's name for the dtype that stores `pt`.
pub fn dtype_name(pt: PixelType) -> &'static str {
    match pt.storage() {
        PixelType::U8 => "uint8",
        PixelType::U16 => "uint16",
        _ => "float32",
    }
}

/// `value` as bincode `bytes`.
///
/// # Errors
/// If `value` cannot be encoded.
pub fn encode<'py, T: Serialize>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = bincode::serialize(value).map_err(value_error)?;
    Ok(PyBytes::new(py, &bytes))
}

/// A value from bincode `bytes`.
///
/// # Errors
/// `TypeError` unless `bytes` is `bytes`; `ValueError` if it does not decode.
pub fn decode<T: DeserializeOwned>(bytes: &Bound<'_, PyAny>) -> PyResult<T> {
    let bytes = bytes
        .cast::<PyBytes>()
        .map_err(|_| PyTypeError::new_err("refimage parts: expected bytes"))?;
    bincode::deserialize(bytes.as_bytes()).map_err(value_error)
}

/// The parts dict of an image.
///
/// # Errors
/// If the header cannot be encoded.
pub fn write<'py>(
    py: Python<'py>,
    header: &Header,
    pixels: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("header", encode(py, header)?)?;
    d.set_item("pixels", pixels)?;
    Ok(d)
}

/// The header and pixel array of a parts dict, checked against each other: the pixels are
/// a numpy array of the header's storage dtype and shape.
///
/// # Errors
/// `TypeError` / `ValueError` for malformed parts.
pub fn read<'py>(parts: &Bound<'py, PyAny>) -> PyResult<(Header, Bound<'py, PyAny>)> {
    let d = parts
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("refimage parts must be a dict"))?;
    let item = |key: &str| {
        d.get_item(key)?
            .ok_or_else(|| value_error(format!("missing {key:?}")))
    };
    let header: Header = decode(&item("header")?)?;
    let pixels = item("pixels")?;
    let np = parts.py().import("numpy")?;
    if !pixels.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err(
            "refimage parts: pixels must be a numpy array",
        ));
    }
    let dtype: String = pixels.getattr("dtype")?.getattr("name")?.extract()?;
    if dtype != dtype_name(header.pixel_type) {
        return Err(value_error(format!(
            "pixels are {dtype}, but a {:?} image is stored as {}",
            header.pixel_type,
            dtype_name(header.pixel_type)
        )));
    }
    let shape: Vec<usize> = pixels.getattr("shape")?.extract()?;
    if shape != header.shape() {
        return Err(value_error(format!(
            "pixels are shaped {shape:?}, the header says {:?}",
            header.shape()
        )));
    }
    Ok((header, pixels))
}

/// An owned image from a header and samples.
fn assemble(header: Header, image: DynamicImageOwned) -> GenericImageOwned {
    GenericImageOwned::with_metadata(header.metadata, image)
}

fn bits(pt: PixelType) -> Option<u8> {
    (pt.storage() == PixelType::U16 && pt != PixelType::U16).then(|| pt.bit_depth())
}

/// The parts of `img`, moving its pixels into a numpy array (no copy: the array owns the
/// allocation, and frees it with this module's allocator).
///
/// # Errors
/// If the header cannot be encoded.
pub fn image_into_parts(py: Python<'_>, img: GenericImageOwned) -> PyResult<Bound<'_, PyDict>> {
    let (metadata, image) = img.into_parts();
    let header = Header::new(metadata, &image);
    let shape = header.shape();
    macro_rules! array {
        ($i:expr) => {{
            let flat = PyArray1::from_vec(py, $i.into_vec());
            flat.reshape(shape)?.into_any()
        }};
    }
    let pixels = match image {
        DynamicImageOwned::U8(i) => array!(i),
        DynamicImageOwned::U16(i) => array!(i),
        DynamicImageOwned::F32(i) => array!(i),
        _ => return Err(value_error("unsupported pixel storage")),
    };
    write(py, &header, pixels)
}

/// An owned image from parts, copying the pixels once (from any numpy layout).
///
/// # Errors
/// For malformed parts.
pub fn image_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<GenericImageOwned> {
    let (header, pixels) = read(parts)?;
    let (w, h, cs) = (header.width, header.height, header.color_space.clone());
    macro_rules! owned {
        ($t:ty) => {{
            let a = pixels.cast::<PyArrayDyn<$t>>()?.readonly();
            let v: Vec<$t> = match a.as_slice() {
                Ok(s) => s.to_vec(),
                Err(_) => a.as_array().iter().copied().collect(),
            };
            ImageOwned::from_owned(v, w, h, cs).map_err(value_error)?
        }};
    }
    let image: DynamicImageOwned = match header.pixel_type.storage() {
        PixelType::U8 => owned!(u8).into(),
        PixelType::U16 => owned!(u16).with_bit_depth(bits(header.pixel_type)).into(),
        _ => owned!(f32).into(),
    };
    Ok(assemble(header, image))
}

/// Whether `pixels` (already checked by [`read`]) can be used in place: a base-class numpy
/// array that is C-contiguous, aligned and writable.
///
/// # Errors
/// If numpy's attributes cannot be read.
pub fn adoptable(pixels: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = pixels.py().import("numpy")?;
    if !pixels.get_type().is(&np.getattr("ndarray")?) {
        return Ok(false);
    }
    let flags = pixels.getattr("flags")?;
    for flag in ["c_contiguous", "aligned", "writeable"] {
        if !flags.getattr(flag)?.is_truthy()? {
            return Ok(false);
        }
    }
    let native: bool = pixels.getattr("dtype")?.getattr("isnative")?.extract()?;
    Ok(native && pixels.cast::<numpy::PyUntypedArray>()?.len() > 0)
}

/// The 10/12/14-bit tag of a pixel type, for `with_bit_depth`.
#[must_use]
pub fn bit_depth_tag(pt: PixelType) -> Option<u8> {
    bits(pt)
}
