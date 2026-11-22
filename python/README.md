# refimage

Python bindings for the [`refimage`](https://crates.io/crates/refimage) Rust crate: images
with metadata, allocation-aware processing pipelines, and FITS output.

```python
import numpy as np
import refimage

raw = np.random.default_rng(0).integers(0, 4096, (1080, 1920), dtype=np.uint16)
img = refimage.Image(raw, "rggb", exposure=0.02, bit_depth=12)
img.set_key("CAMERA", "ZWO ASI2600", comment="camera model")

# A recipe: debayer, rotate 5 degrees about the centre (bicubic), luma, 8-bit.
s, c = np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5))
cx, cy = 959.5, 539.5
inverse = [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]]
pipe = (
    refimage.Pipeline()
    .debayer("linear")
    .warp(inverse, (1080, 1920), order=3, mode="nearest")
    .to_luma()
    .convert("u8")
)

runner = pipe.compile(img)          # validates the chain and allocates every buffer once
out = runner.run(img)               # a new Image, carrying img's metadata
view = np.asarray(out)              # zero-copy (1080, 1920) uint8 view
for frame in frames:                # numpy arrays or Images
    runner.run(frame, out=out)      # no pixel allocation; `view` sees each new frame

out.write_fits("frame.fits", "rice")
```

`Image.warp` / `Image.map_coordinates` (and the same `Pipeline` steps) reproduce
`skimage.transform.warp` with a matrix and `scipy.ndimage.map_coordinates` for spline
orders 0, 1 and 3 and every scipy boundary mode, on any pixel type and channel count.

Images serialize with `to_bytes()` / `Image.from_bytes()` works identically to Rust bincode
`bincode::serialize(&GenericImageOwned)` and pickle.

A Rust library that produces images, such as a camera driver, can return them from its own
PyO3 module as `refimage.Image` objects, with no Python API of its own and without copying
pixels. The [`refimage-pyo3`](https://crates.io/crates/refimage-pyo3) crate provides the
boundary types.
