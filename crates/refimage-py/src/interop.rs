//! `refimage._native`'s side of the exchange format other extension modules use (the
//! `refimage-pyo3` crate): the same code as theirs, on refimage's own objects.
//!
//! Images cross as a numpy array plus a header. Into refimage, a suitable array is adopted
//! as the image's buffer (no copy); out of refimage, the image lends a writable numpy view
//! of its buffer (no copy), which also keeps the image from reallocating while it lives.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use refimage::Metadata;
use refimage_pyo3::parts;

use crate::image::Image;
use crate::store::Store;

/// The exchange format's version (see `refimage-pyo3`).
#[pyfunction]
pub(crate) fn _interop_protocol() -> u32 {
    refimage_pyo3::PROTOCOL
}

/// An `Image` from its parts: the pixel array adopted in place when it can be (C-contiguous,
/// aligned, writable), copied once otherwise.
#[pyfunction]
pub(crate) fn _image_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<Image> {
    let (header, pixels) = parts::read(parts)?;
    Ok(match Store::adopt(header, &pixels)? {
        Some(store) => Image::from_store(store),
        None => Image::from_inner(parts::image_from_parts(parts)?),
    })
}

/// An `Image`'s parts, its pixels lent (a writable numpy view of its buffer).
#[pyfunction]
pub(crate) fn _image_to_parts<'py>(img: &Bound<'py, Image>) -> PyResult<Bound<'py, PyDict>> {
    let py = img.py();
    let header = img.borrow().store.header();
    let pixels = py.import("numpy")?.call_method1("asarray", (img,))?;
    parts::write(py, &header, pixels)
}

/// Replace an `Image`'s metadata with bincode-encoded `Metadata`.
#[pyfunction]
pub(crate) fn _image_set_metadata(
    img: &Bound<'_, Image>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let metadata: Metadata = parts::decode(metadata)?;
    *img.borrow_mut().store.metadata_mut() = metadata;
    Ok(())
}
