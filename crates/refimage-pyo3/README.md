# refimage-pyo3

Exchange [refimage](https://docs.rs/refimage) images and pipelines with the `refimage`
Python package from another PyO3 extension module.

The objects are allocated by the imported package. `isinstance(x, refimage.Image)` holds,
and `refimage` methods accept them.

```rust,ignore
use pyo3::prelude::*;
use refimage_pyo3::refimage::DynamicImageRef;
use refimage_pyo3::{RiImage, RiImageRef, RiPipeline, refimage_error};

/// The next frame, as a `refimage.Image`: its pixel buffer is not copied.
#[pyfunction]
fn capture(py: Python<'_>, camera: &Camera) -> PyResult<RiImage> {
    let frame = py.detach(|| camera.capture()).map_err(|e| refimage_error(py, e))?;
    Ok(RiImage(frame)) // a refimage::GenericImageOwned
}

/// Subtract a dark level in place; Python sees the change.
#[pyfunction]
fn subtract_dark(mut frame: RiImageRef<'_>, dark: u16) -> PyResult<()> {
    if let DynamicImageRef::U16(mut img) = frame.image_mut()? {
        img.as_mut_slice().iter_mut().for_each(|p| *p = p.saturating_sub(dark));
    }
    Ok(())
}

/// Process a frame with a recipe built in Python.
#[pyfunction]
fn process(py: Python<'_>, p: RiPipeline, mut frame: RiImageRef<'_>) -> PyResult<RiImage> {
    let out = p.0.apply(&frame.generic_mut()?).map_err(|e| refimage_error(py, e))?;
    Ok(RiImage(out))
}
```

```python
import refimage, mycamera

img = mycamera.capture(cam)          # a refimage.Image, on the driver's own buffer
mycamera.subtract_dark(img, 64)      # in place
img.write_fits("frame.fits", "rice") # every refimage method works
```

## Copies

Each extension module has its own memory allocator, so memory never changes owner
between modules. Pixels cross as numpy arrays, which the receiving
side uses in place while keeping the array alive. The array, and so the memory, is
freed by the module that allocated it.

| Direction                         | Copies                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------ |
| Rust → Python, `RiImage` by value | none: the `Vec` moves into numpy, and `refimage.Image` adopts it as its buffer |
| Rust → Python, `&RiImage`         | one                                                                            |
| Python → Rust, `RiImageRef`       | none: the image lends a writable view of its buffer                            |
| Python → Rust, `RiImage`          | one                                                                            |
| pipelines (`RiPipeline`)          | the recipe crosses as JSON                                                     |

Metadata (timestamp, exposure, frame id and cards) is small and always copied. Use
`RiImageRef::set_metadata` to change a Python image's metadata from Rust.

An image that adopted a buffer behaves like any other `refimage.Image`, and runners write
into that buffer in place. Only a result of a different shape gives the image a buffer of
refimage's own, and refimage refuses that while numpy views of the old buffer exist.

## Checked

The first conversion imports `refimage` and checks two things: that the package is
compatible with this crate (Cargo's caret rule on the release), and that both speak
the same exchange protocol. If not, it raises `ImportError`. To fail at import time
instead, call `refimage_module(py)` from your module's init function.

## Linking

Use the PyO3 and numpy versions this crate is built with; they are re-exported as
`refimage_pyo3::pyo3` and `refimage_pyo3::numpy`. Your extension sets PyO3's features
(`abi3-*`, `extension-module`); this crate sets none, and defines no module and no
allocator. The `refimage` package itself may be built with any PyO3, because the two
modules exchange only Python objects.

## How it works

PyO3 creates each `#[pyclass]` separately in every extension module that links it, so a
downstream module cannot use refimage's Rust types as Python classes: `isinstance` would
fail. Instead, objects are built and read only by the imported `refimage._native`.
Images cross in a small versioned format, a header (bincode) plus a numpy pixel array,
and the same code in this crate runs on both sides (see the `parts` module). The test
extension in [`tests/ext`](tests/ext) and [`tests/test_interop.py`](tests/test_interop.py)
exercise every path. They check pointer identity to prove that nothing is copied, and
resident memory to prove that adopted buffers are freed.
