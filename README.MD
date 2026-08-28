# `refimage`
[![crates.io](https://img.shields.io/crates/v/refimage)](https://crates.io/crates/refimage)
[![Documentation](https://docs.rs/refimage/badge.svg)](https://docs.rs/refimage)

## A Serializable Image Container

This crate provides a type-erased image container (`GenericImageRef`), backed by a contiguous slice (owned or referenced) of primitive (`u8`, `u16`, `f32`) pixels, with arbitrary color space (grayscale, Bayer pattern, RGB, ...) and color channels support (max. 255).
Image sizes are limited to 65536 &times; 65536 for practical reasons.

`GenericImageRef` and `GenericImageOwned` carry metadata as (`key`, `value`) pairs with optional comments. Metadata keys are case-insensitive strings up to 80 characters; values are integers (stored as `i64`), 32-/64-bit floats (stored as `f64`), strings, `ColorSpace`, or `std::time::{SystemTime, Duration}`.

`GenericImageRef` supports serialization, and can be saved to the open [Flexible Image Transport System (FITS)](https://fits.gsfc.nasa.gov/fits_standard.html) through the `FitsWrite` trait. The FITS writer is pure Rust (no `cfitsio` linkage), always available, and compiles on every target including `wasm32`. It supports uncompressed output and the tile-compression convention with `GZIP_1` (any pixel type) and `RICE_1` (integer images); output is verified against `astropy.io.fits`.

`GenericImageRef` serializes to an internal representation which allows <i>de</i>serialization to `GenericImageOwned`.
`GenericImageOwned` contains owned data, and is freely <i>ser</i>ialized-<i>de</i>serialized. 

## The path to a `GenericImageRef`

A `GenericImageRef` is obtained from a `ImageRef` object, created with the
appropriate, contiguous, backing storage and image format:
```rust
use refimage::{BayerPattern, ImageRef, DemosaicMethod, DynamicImageRef, GenericImageOwned};
use refimage::pipeline::Pipeline;
use std::time::SystemTime;

let mut data = vec![0u8; 256]; // this is the backing store
// acquire(&mut data); // this function populates the backing store with the image pixels
let img = ImageRef::new(&mut data, 16, 16, BayerPattern::Grbg.into()).unwrap(); // 16x16 image backed by the vector
let img = DynamicImageRef::from(img); // convert the `ImageRef` object to `DynamicImageRef`
// All pixel conversions go through a `Pipeline`; `apply` runs it once and owns the result.
let img = Pipeline::new().debayer(DemosaicMethod::Nearest).apply(&img).expect("Could not debayer");
let mut img = GenericImageOwned::new(SystemTime::now(), img); // attach metadata
img.insert_key("CAMERA", ("Rust Test Program", "Name of the camera used to capture the image")).unwrap();
let json = serde_json::to_string(&img).unwrap(); // serialize the image to JSON
let rimg: GenericImageOwned = serde_json::from_str(&json).unwrap(); // deserialize to GenericImageOwned
assert_eq!(&img, &rimg); // Confirm that deserialized image matches the original
```
The intention behind `*ImageRef` is to minimize unnecessary allocations. A metadata-bearing
`GenericImageRef` can be run through a pipeline with `Pipeline::apply_meta`, which carries the
metadata onto the result.

## `GenericImageOwned` and other `ImageOwned` types
An image can be loaded using the [`image`](https://crates.io/crates/image) crate from disk, by enabling the `image` feature:
```rust,ignore
use refimage::DynamicImageOwned;
use image::open;

let img = open("/path/to/image.png").expect("Could not load image");
let img = DynamicImageOwned::try_from(img).expect("Could not convert image");
```

## Loading and storing a `GenericImageRef`
A `GenericImageRef` can be stored in a standard format such as [bincode](https://crates.io/crates/bincode) — which follows trivially from its `serde` implementation — or written to a FITS file through the `FitsWrite` trait. The FITS writer is pure Rust and always available.
```rust,ignore
use refimage::{FitsCompression, FitsWrite, GenericImageRef};
use std::path::Path;
let img: GenericImageRef = { todo!() }; // obtain a GenericImageRef

// RICE_1 tile compression; overwrite if it exists.
img.write_fits(Path::new("/path/to/image.fits"), FitsCompression::Rice, true)
    .expect("Could not write FITS file.");

// Or straight to bytes — handy on wasm, where there is no filesystem.
let bytes: Vec<u8> = img.fits_bytes(FitsCompression::Gzip).unwrap();
```

## Conversions

Debayering, luminance, pixel-type conversion, affine pixel scaling, cropping, ROI
extraction, flips and 90° rotations are all `Op`s on a [`Pipeline`](#reusable-processing-pipelines).
There is no separate per-type trait API — build a `Pipeline` and call `apply` (or
`apply_meta` to keep metadata, or compile a `Runner` for a frame stream).

## Reusable processing pipelines
Every conversion is a step in a `Pipeline`. For a one-off, `Pipeline::apply(&img)` compiles,
runs once, and returns an owned image. For acquisition loops that run the same chain over
every frame, compile a `Runner` once and reuse its buffers:
```rust
use refimage::{BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, ImageRef, PixelType};
use refimage::pipeline::{ImageSpec, Pipeline, Strategy};

// A declarative, cloneable, serializable recipe.
let recipe = Pipeline::new()
    .crop(0, 0, 64, 64)
    .debayer(DemosaicMethod::Linear)
    .to_luma()
    .scale(1.2, 4.0)
    .convert(PixelType::U8)
    .flip_vertical();

// Compile against a concrete frame format: validates the chain and
// pre-allocates every buffer it will ever need.
let spec = ImageSpec::new(64, 64, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
let mut runner = recipe.compile(spec, Strategy::tiled_parallel(16)).unwrap();

// Run it per frame; the result borrows the runner's internal buffer.
let mut frame = vec![0u16; 64 * 64];
let img = DynamicImageRef::from(
    ImageRef::new(&mut frame, 64, 64, ColorSpace::Bayer(BayerPattern::Rggb)).unwrap(),
);
let luma = runner.run(&img).unwrap();
assert_eq!(luma.color_space(), ColorSpace::Gray);
```
`Strategy::Sequential` uses one ping-pong buffer pair; `Strategy::tiled`/`tiled_parallel`/`tiled_2d`/`tiled_2d_parallel` cut the frame into cache-sized tiles (bit-identical output). A serial-tiled `Runner` does **zero** heap allocation per frame once compiled. See the module docs for how geometric ops interact with tiling and for `Runner::recompile` / the `grow` feature.

# FITS
Writing `GenericImageRef` / `GenericImageOwned` to FITS via the `FitsWrite` trait is a core, always-available feature — a pure-Rust implementation with no `cfitsio` linkage. It writes uncompressed images and the tile-compression convention (`FitsCompression::{None, Gzip, Rice}`); multi-channel images are stored planar, `u16` carries `BZERO = 32768`, and metadata becomes header cards (`HIERARCH` for long keys, `CONTINUE` for long strings). Output is checked against `astropy.io.fits` and `cfitsio` in the test suite. `FitsWrite::fits_bytes` serialises to memory when there is no filesystem.

# Optional Features
Features are available to extend the functionalities of the core `refimage` data types:
- `rayon`: Parallelizes luma / demosaic / cast inside the `pipeline`, and enables the parallel `pipeline` strategies (<b>enabled</b> by default).
- `image`: Enables `TryFrom` conversions between `image::DynamicImage` and `refimage::DynamicImageRef`, `refimage::DynamicImageOwned` (<b>disabled</b> by default).
- `grow`: Lets a compiled pipeline `Runner` reallocate its buffers on a frame-shape change instead of erroring (<b>disabled</b> by default).

# WebAssembly
The core of `refimage` — image construction, the `pipeline`, metadata, and `serde` — runs on `wasm32-unknown-unknown`. It never calls `SystemTime::now()`, spawns threads, or touches the filesystem itself; the caller supplies timestamps, and the pipeline falls back to its serial kernels when `rayon` is off. `tests/wasm.rs` is a `wasm-bindgen-test` suite covering this:

```sh
cargo install wasm-bindgen-cli
cargo test --target wasm32-unknown-unknown --no-default-features --tests
```

(`--tests` skips the doc-tests, which rustdoc does not cross-compile.)

FITS writing works on wasm too (it is pure Rust); `FitsWrite::fits_bytes` returns the file as a `Vec<u8>` since `wasm32-unknown-unknown` has no filesystem.