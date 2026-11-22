//! Exchange [`refimage`] images and pipelines with the `refimage` Python package from
//! another PyO3 extension module.
//!
//! A library that produces or processes refimage images in Rust (a camera driver, an
//! acquisition system) can expose them to Python as the `refimage` package's own objects,
//! with no Python API of its own for images and without copying pixels. Its
//! `#[pyfunction]`s take and return the boundary types of this crate, which implement
//! PyO3's `FromPyObject` and `IntoPyObject`:
//!
//! | Rust | Python | type |
//! |---|---|---|
//! | [`GenericImageOwned`](refimage::GenericImageOwned) | `refimage.Image` | [`RiImage`] |
//! | the pixels of a `refimage.Image`, in place | `refimage.Image` | [`RiImageRef`] |
//! | [`Pipeline`](refimage::pipeline::Pipeline) | `refimage.Pipeline` | [`RiPipeline`] |
//!
//! ```ignore
//! use refimage_pyo3::{RiImage, RiImageRef, RiPipeline, refimage_error};
//! use refimage_pyo3::refimage::DynamicImageRef;
//! use pyo3::prelude::*;
//!
//! /// The next frame from the camera, as a `refimage.Image` (its pixels are not copied).
//! #[pyfunction]
//! fn capture(py: Python<'_>, camera: &Camera) -> PyResult<RiImage> {
//!     let frame = py.detach(|| camera.capture()).map_err(|e| refimage_error(py, e))?;
//!     Ok(RiImage(frame))   // a GenericImageOwned
//! }
//!
//! /// Subtract a dark level in place: Python sees the change, nothing is copied.
//! #[pyfunction]
//! fn subtract_dark(mut frame: RiImageRef<'_>, dark: u16) -> PyResult<()> {
//!     if let DynamicImageRef::U16(img) = frame.image_mut()? {
//!         img.as_mut_slice().iter_mut().for_each(|p| *p = p.saturating_sub(dark));
//!     }
//!     Ok(())
//! }
//!
//! /// Process a frame with a recipe built in Python.
//! #[pyfunction]
//! fn process(py: Python<'_>, p: RiPipeline, mut frame: RiImageRef<'_>) -> PyResult<RiImage> {
//!     let out = p.0.apply(&frame.generic_mut()?).map_err(|e| refimage_error(py, e))?;
//!     Ok(RiImage(out))
//! }
//! ```
//!
//! On the Python side these are ordinary `refimage` objects: `isinstance(x,
//! refimage.Image)` holds, and every `refimage` method and function accepts them.
//!
//! # The objects are `refimage`'s own
//!
//! PyO3 creates each `#[pyclass]` separately in every extension module that links it, so
//! an extension cannot use `refimage`'s classes as Rust types: its copies would not be the
//! classes `import refimage` loads (`isinstance` would fail, and refimage would reject
//! them). This crate therefore builds and reads objects only through the imported
//! `refimage` package: images cross as a numpy array plus a header (the [`parts`]
//! format), and `refimage._native` builds and reads its own objects from them, with the
//! same code as this crate.
//!
//! The first conversion imports `refimage` and checks that it is compatible with this
//! crate (the same release under Cargo's caret rule, and the same exchange
//! [`PROTOCOL`]); it raises `ImportError` otherwise. [`refimage_module`] runs the check
//! explicitly, for example when the extension module is imported.
//!
//! # Copies
//!
//! Each extension module has its own memory allocator, so memory never changes owner
//! between two extension modules. It need not: pixels cross as numpy arrays, which the
//! receiving side uses in place while keeping the array (and so the memory, freed by the
//! module that allocated it) alive.
//!
//! | direction | copies of the pixels |
//! |---|---|
//! | Rust → Python, by value ([`RiImage`]) | none: the pixel `Vec` moves into a numpy array, which `refimage.Image` adopts as its buffer |
//! | Rust → Python, by reference (`&RiImage`) | one |
//! | Python → Rust, in place ([`RiImageRef`]) | none: the image lends a writable view of its buffer |
//! | Python → Rust, owned ([`RiImage`]) | one |
//! | pipelines ([`RiPipeline`]) | the recipe crosses as JSON |
//!
//! Metadata (timestamp, exposure, frame id, cards) is small and always copied. An image
//! that adopted a buffer behaves like any other `refimage.Image`; when a runner writes a
//! result of a different shape into it, it switches to a buffer of its own.
//!
//! # Linking
//!
//! The extension module chooses PyO3's features (`abi3-*`, `extension-module`); this crate
//! sets none and defines no module or allocator. It needs the PyO3 and numpy versions it
//! is built with (re-exported here as [`pyo3`] and [`numpy`]), since an extension links
//! only one PyO3. The `refimage` package may itself be built with any PyO3: the two
//! modules only exchange Python objects.

pub mod compat;
mod image;
pub mod parts;
mod pipeline;

pub use compat::{PROTOCOL, VERSION, compatible, refimage_error, refimage_module};
pub use image::{RiImage, RiImageRef};
pub use parts::Header;
pub use pipeline::RiPipeline;

/// The numpy bindings this crate uses.
pub use numpy;
/// The PyO3 this crate uses.
pub use pyo3;
/// The refimage crate this one exchanges values of.
pub use refimage;
