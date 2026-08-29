# `refimage`
[![crates.io](https://img.shields.io/crates/v/refimage)](https://crates.io/crates/refimage)
[![Documentation](https://docs.rs/refimage/badge.svg)](https://docs.rs/refimage)

## A Serializable Image Container

This crate provides a type-erased image container (`GenericImageRef`), backed by a contiguous slice (owned or referenced) of primitive (`u8`, `u16`, `f32`) pixels, with arbitrary color space (grayscale, Bayer pattern, RGB, ...) and color channels support (max. 255).
Image sizes are limited to 65536 &times; 65536 for practical reasons.

`GenericImageRef` and `GenericImageOwned` always carry a `Metadata` block: a mandatory typed core (`timestamp: chrono::DateTime<Utc>`, `exposure: Duration` — `Duration::ZERO` meaning "not applicable") plus an insertion-ordered map of extra (`key`, `value`) pairs with optional comments. Metadata keys are case-insensitive strings up to 80 characters (`TIMESTAMP` / `EXPOSURE` are reserved); values are integers (stored as `i64`), 32-/64-bit floats (stored as `f64`), strings, `ColorSpace`, `std::time::Duration`, or a UTC `chrono::DateTime`. `chrono` is re-exported as `refimage::chrono`.

`GenericImageRef` supports serialization, and can be saved to the open [Flexible Image Transport System (FITS)](https://fits.gsfc.nasa.gov/fits_standard.html) through the `FitsWrite` trait. The FITS writer is pure Rust (no `cfitsio` linkage), always available, and compiles on every target including `wasm32`. It supports uncompressed output (`FitsCompression::NONE`) and the tile-compression convention via the builder structs `Gzip` (lossless, any pixel type), `Rice` and `Hcompress` (both lossless for `u8`/`u16`; `f32` is quantized with `SUBTRACTIVE_DITHER_1`). Each builder exposes its settings — tile shape (`tile_rows` / `tile_dims`), quantization level, HCOMPRESS `scale` / `smooth`. Output is verified against `astropy.io.fits` and `cfitsio`.

`GenericImageRef` serializes to an internal representation which allows <i>de</i>serialization to `GenericImageOwned`.
`GenericImageOwned` contains owned data, and is freely <i>ser</i>ialized-<i>de</i>serialized. 

## The path to a `GenericImageRef`

A `GenericImageRef` is obtained from a `ImageRef` object, created with the
appropriate, contiguous, backing storage and image format:
```rust
use refimage::{BayerPattern, ImageRef, DemosaicMethod, DynamicImageRef, GenericImageOwned};
use refimage::pipeline::Pipeline;
use refimage::chrono::DateTime;
use std::time::Duration;

let mut data = vec![0u8; 256]; // this is the backing store
// acquire(&mut data); // this function populates the backing store with the image pixels
let img = ImageRef::new(&mut data, 16, 16, BayerPattern::Grbg.into()).unwrap(); // 16x16 image backed by the vector
let img = DynamicImageRef::from(img); // convert the `ImageRef` object to `DynamicImageRef`
// All pixel conversions go through a `Pipeline`; `apply` runs it once and owns the result.
let img = Pipeline::new().debayer(DemosaicMethod::Nearest).apply(&img).expect("Could not debayer");
// A UTC timestamp and an exposure (`Duration::ZERO` if not applicable) are mandatory.
// In application code the timestamp is usually `chrono::Utc::now()`.
let ts = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
let mut img = GenericImageOwned::new(ts, Duration::from_millis(20), img);
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
use refimage::{FitsCompression, FitsWrite, GenericImageRef, Gzip, Rice, Quantize};
use std::path::Path;
let img: GenericImageRef = { todo!() }; // obtain a GenericImageRef

// RICE_1 tile compression; overwrite if it exists.
img.write_fits(Path::new("/path/to/image.fits"), Rice::new(), true)
    .expect("Could not write FITS file.");

// Each algorithm is a builder: tile shape, quantization level, HCOMPRESS scale, ...
img.write_fits(
    Path::new("/path/to/image.fits"),
    Rice::new().tile_rows(16).quantize(Quantize::new().level(16.0)),
    true,
).unwrap();

// Or straight to bytes — handy on wasm, where there is no filesystem.
let bytes: Vec<u8> = img.fits_bytes(Gzip::new()).unwrap();
let plain: Vec<u8> = img.fits_bytes(FitsCompression::NONE).unwrap();
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
Writing `GenericImageRef` / `GenericImageOwned` to FITS via the `FitsWrite` trait is a core, always-available feature — a pure-Rust implementation with no `cfitsio` linkage. It writes uncompressed images (`FitsCompression::NONE`) and the tile-compression convention through the builder structs `Gzip`, `Rice` and `Hcompress` (anything `Into<FitsCompression>`). `Gzip` is lossless for any pixel type; `Rice` and `Hcompress` are lossless for `u8`/`u16` and quantize `f32` with `SUBTRACTIVE_DITHER_1`. Tile shape is set with `tile_rows(n)` or `tile_dims([nx, ny])` (mutually exclusive at compile time), defaulting to one image row per tile for `Rice`/`Gzip` and one whole channel plane for `Hcompress` (the H-transform + quadtree coder, min. tile 4×4); `Rice`/`Hcompress` also take a `Quantize` (level / dither seed) and `Hcompress` a `scale` / `smooth`. Multi-channel images are stored planar, `u16` carries `BZERO = 32768`, and metadata becomes header cards (`HIERARCH` for long keys, `CONTINUE` for long strings). Output is checked against `astropy.io.fits` and `cfitsio` in the test suite. Everything works without a filesystem: `FitsWrite::fits_bytes` returns a single image as a `Vec<u8>`, and `create_fits_to(sink, compression)` builds a multi-HDU file into any `Write` sink (`create_fits` is the on-disk shortcut).

# Optional Features
Features are available to extend the functionalities of the core `refimage` data types:
- `rayon`: Parallelizes luma / demosaic / cast inside the `pipeline`, and enables the parallel `pipeline` strategies (<b>enabled</b> by default).
- `image`: Enables `TryFrom` conversions between `image::DynamicImage` and `refimage::DynamicImageRef`, `refimage::DynamicImageOwned` (<b>disabled</b> by default).
- `grow`: Lets a compiled pipeline `Runner` reallocate its buffers on a frame-shape change instead of erroring (<b>disabled</b> by default).

# WebAssembly
The core of `refimage` — image construction, the `pipeline`, metadata, and `serde` — runs on `wasm32-unknown-unknown`. It never reads a system clock (`chrono` is pulled in without its `clock` feature), spawns threads, or touches the filesystem itself; the caller supplies timestamps, and the pipeline falls back to its serial kernels when `rayon` is off. `tests/wasm.rs` is a `wasm-bindgen-test` suite covering this:

```sh
cargo install wasm-bindgen-cli
cargo test --target wasm32-unknown-unknown --no-default-features --tests
```

(`--tests` skips the doc-tests, which rustdoc does not cross-compile.)

FITS writing works on wasm too (it is pure Rust): `FitsWrite::fits_bytes` returns a single image as a `Vec<u8>`, and `create_fits_to(Vec::new(), compression)` builds a multi-HDU file in memory — `wasm32-unknown-unknown` has no filesystem.