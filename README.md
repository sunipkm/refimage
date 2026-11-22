# refimage

Image storage with rich metadata, allocation-aware processing pipelines (debayer, luma,
scaling, crop/ROI, flips and rotations, resize, homography warp, `map_coordinates`) and
FITS output — in Rust, with Python bindings and a C API.

| | |
|---|---|
| [`crates/refimage`](crates/refimage) | the Rust library ([crates.io](https://crates.io/crates/refimage)) |
| [`crates/refimage-py`](crates/refimage-py) | the PyO3 binding (`refimage._native`), not published on its own |
| [`crates/refimage-capi`](crates/refimage-capi) | the C API: `refimage.h` + `librefimage_c` (shared and static), pkg-config and CMake packages |
| [`python/`](python) | the `refimage` Python package, built with maturin |

```sh
cargo test                                  # the Rust crate (the default workspace member)
cargo test -p refimage-capi                 # the C API, including a C test program
cd python && maturin develop --release && pytest
```
