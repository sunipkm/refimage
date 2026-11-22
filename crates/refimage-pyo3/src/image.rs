//! Images: [`RiImage`] (owned) and [`RiImageRef`] (a `refimage.Image`'s pixels, in place).

use numpy::{PyArrayDyn, PyArrayMethods, PyReadwriteArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use refimage::pipeline::{Frame, ImageSpec};
use refimage::{
    ColorSpace, DynamicImageRef, GenericImageOwned, GenericImageRef, ImageProps, ImageRef,
    Metadata, PixelType,
};

use crate::compat::refimage_module;
use crate::parts::{self, Header};

/// `refimage._native.<function>(args...)`.
fn call<'py>(
    py: Python<'py>,
    function: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    refimage_module(py)?.getattr(function)?.call1(args)
}

/// Whether `obj` is a `refimage.Image` (of the imported package, not a look-alike).
fn is_image(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.is_instance(&refimage_module(obj.py())?.getattr("Image")?)
}

fn not_an_image(obj: &Bound<'_, PyAny>) -> PyErr {
    let name = obj
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |n| n.to_string());
    PyTypeError::new_err(format!("expected a refimage.Image, not {name}"))
}

/// The parts of a `refimage.Image`, its pixels lent (a writable numpy view, no copy).
fn lend<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Header, Bound<'py, PyAny>)> {
    if !is_image(obj)? {
        return Err(not_an_image(obj));
    }
    parts::read(&call(obj.py(), "_image_to_parts", (obj,))?)
}

/// An image at the boundary with Python: `refimage.Image` there, a
/// [`GenericImageOwned`] here.
///
/// | direction | copies of the pixels |
/// |---|---|
/// | Rust → Python, by value | none: the pixel buffer moves into numpy and `refimage.Image` adopts it |
/// | Rust → Python, `&RiImage` | one |
/// | Python → Rust | one (use [`RiImageRef`] to read and write in place instead) |
#[derive(Debug, Clone, PartialEq)]
pub struct RiImage(pub GenericImageOwned);

impl<'py> IntoPyObject<'py> for RiImage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// Moves the pixels into numpy, which `refimage.Image` adopts: no copy.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        call(
            py,
            "_image_from_parts",
            (parts::image_into_parts(py, self.0)?,),
        )
    }
}

impl<'py> IntoPyObject<'py> for &RiImage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// Copies the pixels once, into numpy, which `refimage.Image` adopts.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        RiImage(self.0.clone()).into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for RiImage {
    type Error = PyErr;

    /// Copies the pixels once (the image lends a numpy view; the copy is this module's).
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        if !is_image(&obj)? {
            return Err(not_an_image(&obj));
        }
        let parts = call(obj.py(), "_image_to_parts", (&obj,))?;
        Ok(Self(parts::image_from_parts(&parts)?))
    }
}

impl From<GenericImageOwned> for RiImage {
    fn from(img: GenericImageOwned) -> Self {
        Self(img)
    }
}

/// The pixels a `refimage.Image` lends: a writable numpy view of its buffer.
#[derive(Debug)]
enum Lent<'py> {
    U8(PyReadwriteArrayDyn<'py, u8>),
    U16(PyReadwriteArrayDyn<'py, u16>),
    F32(PyReadwriteArrayDyn<'py, f32>),
}

/// A `refimage.Image` from Python whose pixels are read and written **in place**: no copy,
/// and writes are seen by Python (by the image and by numpy views of it).
///
/// The metadata is a copy taken when the argument was converted; change the image's with
/// [`set_metadata`](Self::set_metadata). While this value lives the image cannot change
/// shape (refimage refuses to reallocate pixels that numpy views), and numpy's borrow
/// checking keeps other Rust code from borrowing the same pixels mutably.
///
/// ```ignore
/// #[pyfunction]
/// fn subtract_dark(mut frame: RiImageRef<'_>, dark: u16) -> PyResult<()> {
///     if let DynamicImageRef::U16(img) = frame.image_mut()? {
///         img.as_mut_slice().iter_mut().for_each(|p| *p = p.saturating_sub(dark));
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct RiImageRef<'py> {
    object: Bound<'py, PyAny>,
    header: Header,
    pixels: Lent<'py>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for RiImageRef<'py> {
    type Error = PyErr;

    /// Borrows the image's pixels (no copy) and copies its metadata.
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let object = obj.to_owned();
        let (header, pixels) = lend(&object)?;
        let borrow_err = |e: numpy::BorrowError| {
            PyValueError::new_err(format!("cannot borrow the image's pixels mutably: {e}"))
        };
        let pixels = match header.pixel_type.storage() {
            PixelType::U8 => Lent::U8(
                pixels
                    .cast::<PyArrayDyn<u8>>()?
                    .try_readwrite()
                    .map_err(borrow_err)?,
            ),
            PixelType::U16 => Lent::U16(
                pixels
                    .cast::<PyArrayDyn<u16>>()?
                    .try_readwrite()
                    .map_err(borrow_err)?,
            ),
            _ => Lent::F32(
                pixels
                    .cast::<PyArrayDyn<f32>>()?
                    .try_readwrite()
                    .map_err(borrow_err)?,
            ),
        };
        let me = Self {
            object,
            header,
            pixels,
        };
        // The image lends its own, C-contiguous buffer; anything else is a protocol error.
        me.try_bytes()?;
        Ok(me)
    }
}

impl<'py> RiImageRef<'py> {
    /// The `refimage.Image` itself, e.g. to return it after changing it in place.
    pub fn object(&self) -> &Bound<'py, PyAny> {
        &self.object
    }

    /// Shape, color space, pixel type and (a copy of) the metadata.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// A copy of the image's metadata, taken when the argument was converted.
    pub fn metadata(&self) -> &Metadata {
        &self.header.metadata
    }

    /// The shape and type, as a pipeline sees it.
    pub fn spec(&self) -> ImageSpec {
        self.header.spec()
    }

    fn try_bytes(&self) -> PyResult<&[u8]> {
        let err = |_| PyValueError::new_err("refimage lent a non-contiguous pixel buffer");
        Ok(match &self.pixels {
            Lent::U8(a) => a.as_slice().map_err(err)?,
            Lent::U16(a) => bytemuck::cast_slice(a.as_slice().map_err(err)?),
            Lent::F32(a) => bytemuck::cast_slice(a.as_slice().map_err(err)?),
        })
    }

    /// The pixels as native-endian bytes, in place.
    pub fn bytes(&self) -> &[u8] {
        // Checked contiguous at extraction.
        self.try_bytes().unwrap_or(&[])
    }

    /// The pixels, writable in place.
    ///
    /// # Errors
    /// Only if the image's own buffer were inconsistent with its shape.
    pub fn image_mut(&mut self) -> PyResult<DynamicImageRef<'_>> {
        let (w, h) = (self.header.width, self.header.height);
        let cs: ColorSpace = self.header.color_space.clone();
        let bits = parts::bit_depth_tag(self.header.pixel_type);
        let err = |e: &dyn std::fmt::Display| PyValueError::new_err(e.to_string());
        let not_contiguous =
            |_| PyValueError::new_err("refimage lent a non-contiguous pixel buffer");
        Ok(match &mut self.pixels {
            Lent::U8(a) => ImageRef::new(a.as_slice_mut().map_err(not_contiguous)?, w, h, cs)
                .map_err(|e| err(&e))?
                .into(),
            Lent::U16(a) => ImageRef::new(a.as_slice_mut().map_err(not_contiguous)?, w, h, cs)
                .map_err(|e| err(&e))?
                .with_bit_depth(bits)
                .into(),
            Lent::F32(a) => ImageRef::new(a.as_slice_mut().map_err(not_contiguous)?, w, h, cs)
                .map_err(|e| err(&e))?
                .into(),
        })
    }

    /// The pixels in place with (a copy of) the metadata: what refimage's FITS writer,
    /// `Pipeline::apply` and the exposure calculator take.
    ///
    /// # Errors
    /// As [`image_mut`](Self::image_mut).
    pub fn generic_mut(&mut self) -> PyResult<GenericImageRef<'_>> {
        let meta = self.header.metadata.clone();
        Ok(GenericImageRef::with_metadata(meta, self.image_mut()?))
    }

    /// Replace the image's metadata (timestamp, exposure, frame id and cards), in Python
    /// as well as in this copy.
    ///
    /// # Errors
    /// If the metadata cannot be encoded, or `refimage` rejects it.
    pub fn set_metadata(&mut self, metadata: Metadata) -> PyResult<()> {
        let py = self.object.py();
        call(
            py,
            "_image_set_metadata",
            (&self.object, parts::encode(py, &metadata)?),
        )?;
        self.header.metadata = metadata;
        Ok(())
    }
}

impl ImageProps for RiImageRef<'_> {
    fn width(&self) -> usize {
        self.header.width
    }
    fn height(&self) -> usize {
        self.header.height
    }
    fn channels(&self) -> u8 {
        self.header.color_space.channels()
    }
    fn color_space(&self) -> ColorSpace {
        self.header.color_space.clone()
    }
    fn pixel_type(&self) -> PixelType {
        self.header.pixel_type
    }
    fn len(&self) -> usize {
        self.header.width * self.header.height * self.header.channels()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A lent image is a pipeline frame: `runner.run(&frame)` reads it in place.
impl Frame for RiImageRef<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.bytes()
    }
}
