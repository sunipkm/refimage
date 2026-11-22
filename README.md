# refimage

Image storage with rich metadata, allocation-aware processing pipelines (debayer, luma,
scaling, crop/ROI, flips and rotations, resize, homography warp, `map_coordinates`) and
FITS output — in Rust, with Python bindings and a C API.

| | |
|---|---|
| [`crates/refimage`](crates/refimage) | the Rust library ([crates.io](https://crates.io/crates/refimage)) |
| [`crates/refimage-py`](crates/refimage-py) | the PyO3 binding (`refimage._native`), not published on its own |
| [`crates/refimage-capi`](crates/refimage-capi) | the C API: `refimage.h` + `librefimage_c` (shared and static), pkg-config and CMake packages |
| [`crates/refimage-pyo3`](crates/refimage-pyo3) | for other PyO3 extensions: take and return `refimage.Image` / `refimage.Pipeline` without copying pixels ([crates.io](https://crates.io/crates/refimage-pyo3)) |
| [`python/`](python) | the `refimage` Python package, built with maturin |

```sh
cargo test                                  # the Rust crate (the default workspace member)
cargo test -p refimage-capi                 # the C API, including a C test program
cd python && maturin develop --release && pytest
```

## Releasing

CI publishes everything; nothing is uploaded by hand. Push a tag `vX.Y.Z`, or
`vX.Y.Z-preN` for a pre-release:

```sh
git tag -a v1.0.0-pre7 -m "refimage 1.0.0-pre7"
git push origin v1.0.0-pre7
```

[`release.yml`](.github/workflows/release.yml) then does the following:

1. Runs the full CI ([`ci.yml`](.github/workflows/ci.yml)).
2. Builds the Python wheels for every platform, tests them on the oldest and newest
   supported Pythons, and builds the sdist.
3. Only once all of that passes, publishes the crates (`refimage`, then `refimage-pyo3`)
   to crates.io, then the `refimage` package to PyPI.
4. Creates the GitHub release. `-preN` tags are marked as pre-releases.

The version comes from the tag ([`tools/version.py`](tools/version.py)); it is never
edited by hand. The version committed in `Cargo.toml` is only the base for untagged
development builds (`1.0.0-pre7.dev.N+g<sha>`). Crate publishing
([`tools/publish_crates.py`](tools/publish_crates.py)) skips versions already on crates.io,
so re-running a failed release job finishes it.

One-time setup:

- a `CARGO_REGISTRY_TOKEN` repository secret;
- a PyPI trusted publisher for `refimage` with workflow `release.yml` and environment
  `pypi`.
