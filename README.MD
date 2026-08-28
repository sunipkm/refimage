# `refimage`
[![crates.io](https://img.shields.io/crates/v/refimage)](https://crates.io/crates/refimage)
[![Documentation](https://docs.rs/refimage/badge.svg)](https://docs.rs/refimage)

## A Serializable Image Container

This crate provides a type-erased image container (`GenericImageRef`), backed by a contiguous slice (owned or referenced) of primitive (`u8`, `u16`, `f32`) pixels, with arbitrary color space (grayscale, Bayer pattern, RGB, ...) and color channels support (max. 255).
Image sizes are limited to 65536 &times; 65536 for practical reasons.

`GenericImageRef` and `GenericImageOwned` are powerful, since these metadata in the form of (`key`, `value`) pairs, with optional comments. Valid metadata keys are case-insensitive, 80-character strings, and values are up to 64-bit integer types, 32- and 64-bit floating point numbers, strings up-to 4096 characters, or `std::time::{SystemTime, Duration}`.

`GenericImageRef` supports serialization, and, optionally can be saved in the open [Flexible Image Transport System (FITS)](https://fits.gsfc.nasa.gov/fits_standard.html) through the `FitsWrite` trait, exposed through the `fitsio` optional feature.
FITS files support lossless compression, and saving of arbitrary metadata.

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
```rust,no_run
use refimage::DynamicImageOwned;
use image::open;

let img = open("/path/to/image.png").expect("Could not load image");
let img = DynamicImageOwned::try_from(img).expect("Could not convert image");
```

## Loading and storing a `GenericImageRef`
A `GenericImageRef` is intended to be loaded and stored in a standard format, e.g. [bincode](https://crates.io/crates/bincode) - which follows trivially from the serialization of `GenericImageRef`. However, for more portable applications, with the `fitsio` feature, a `GenericImageRef` can be stored as a FITS file by importing the `FitsWrite` trait. 
The FITS file is stored using the [`fitsio`](https://crates.io/crates/fitsio) crate, which is a thin wrapper around the `cfitsio` library.
```rust,no_run
use refimage::{FitsCompression, FitsWrite, GenericImageRef}; // we need the FitsWrite trait to be able to use the `write_fits` method.
use std::path::Path;
let img: GenericImageRef = { todo!() }; // obtain a GenericImageRef
img.write_fits(Path::new("/path/to/fitsimage.fit"), FitsCompression::None, true) // no compression, overwrite
    .expect("Could not write FITS file.");
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

# Optional Features
Features are available to extend the functionalities of the core `refimage` data types:
- `rayon`: Parallelizes luma / demosaic / cast inside the `pipeline`, and enables the parallel `pipeline` strategies (<b>enabled</b> by default).
- `fitsio`: Exposes `FitsWrite` trait to write `GenericImageRef` and `GenericImageOwned` (<b>disabled</b> by default).
- `image`: Enables `TryFrom` conversions between `image::DynamicImage` and `refimage::DynamicImageRef`, `refimage::DynamicImageOwned` (<b>disabled</b> by default).
- `grow`: Lets a compiled pipeline `Runner` reallocate its buffers on a frame-shape change instead of erroring (<b>disabled</b> by default).

# WebAssembly
The core of `refimage` — image construction, the `pipeline`, metadata, and `serde` — runs on `wasm32-unknown-unknown`. It never calls `SystemTime::now()`, spawns threads, or touches the filesystem itself; the caller supplies timestamps, and the pipeline falls back to its serial kernels when `rayon` is off. `tests/wasm.rs` is a `wasm-bindgen-test` suite covering this:

```sh
cargo install wasm-bindgen-cli
cargo test --target wasm32-unknown-unknown --no-default-features --tests
```

(`--tests` skips the doc-tests, which rustdoc does not cross-compile.)

The `fitsio` feature (which links the C `cfitsio` library) is not available on wasm.