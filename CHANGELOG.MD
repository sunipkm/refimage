# Changelog for `refimage`

## Changes

### 0.8.0 (2024-09-22)
- Added `from_u8` methods to `ImageData` and `ImageOwned` to get these images from a slice of `u8` data in machine-endian order.

### 0.7.0 (2024-09-21)
- Alpha channel support has been added.
- API change: Bayer patterns moved to `BayerPattern` enum, `ColorSpace` now contains a variant `ColorSpace::Bayer(BayerPattern)`.
- Enhancement: `BayerPattern` implements the `BayerShift` trait to allow `BayerPattern` selection for smaller, shifted ROIs.

### 0.6.2 (2024-09-21)
- Bugfix: `into_luma()` errored on a grayscale image.

### 0.6.1 (2024-09-21)
- Added `EXPOSURE_KEY` to standardize setting of exposure time.

### 0.6.0 (2024-09-21)
- Added `CalcOptExp` trait to calculate optimum exposure for a given `DynamicImageData` or `ImageData`.
  This trait is also implemented for the equivalent Owned types.
- Changed the signature of `OptimumExposure::calculate` function to accept a mutable reference to an array instead of a `Vec<T>`.

### 0.5.1 (2024-09-14)
- Refactored `rayon` code, reduced code duplication.
- Added feature documentation.

### 0.5.0 (2024-09-13)
- Optimum exposure calculation internally casts to `f32`.
- Added `ColorSpace` metadata, removed `repr(u8)` requirement since `serde` is now required.
- Added `to_f32` and `from_f32` functions for `PixelStor` trait.
- Added `cast_u8` for `PixelStor`, allowing casting any value to [0, 255] range.
- Added owned data counterparts to `GenericImage`, `DynamicImageData` and `ImageData`, along with
  accompanying `To` and `From` implementations.
- Added `into_u8` (parallelized with `rayon`) functions to convert all images to owned images with `u8` pixels.

### 0.4.0 (2024-08-27)
- Exposed a `GenericValue` enum, with `From` and `TryInto` implementations to concrete types.
  `GenericLineItem` now does not include functions to access the underlying concrete types.
- Added optimum exposure calculation support.
- Added luminance conversion support.

### 0.3.0 (2024-08-23)
- Changed the signature of the closure taken by the `GenericImage::operate` function, and
  changed the behavior of the function to return a copy of the original image. Updated the
  relevant test.
- Updated the example to show the use of `DynamicImageData` and `GenericImage`.

### 0.2.0 (2024-08-21)
- Moved the `debayer` method to its own trait, and removed public `debayer` methods
  from `DynamicImageData` and `ImageData`.
- Renamed enum `Demosaic` to `DemosaicMethod`.
- Implemented `Debayer::debayer` for `GenericImage`.
- Implemented `GenericImage::operate()` that allows replacement of the underlying
  image while conserving the metadata.
- Moved the `GenericImage::write_fits` function (enabled with `fitsio` feature) to the `FitsWrite` trait.
- Added custom FITS compression support through `FitsCompression::Custom`.
- Formatted all files and documentation.
- Added this change log.

### 0.1.1 (2024-08-20)
- Updated documentation.

## Pre-publish steps
### Check documentation
First you add this to your `Cargo.toml`:

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```
This will make docs.rs compile your documentation with `RUSTFLAGS="--cfg docsrs"` set. We can use this to conditionally enable nightly features when building documentation, but not when building it normally.

Now you add this to your `lib.rs` file:

`#![cfg_attr(docsrs, feature(doc_cfg))]`
Then, on anything that needs such an annotation, you add the following

`#[cfg_attr(docsrs, doc(cfg(feature = "rt")))]`
This supports all of the syntax that `#[cfg(...)]` does, including stuff like platform-specifiers and any/all. Tokio often sets this annotation using the macros in [`src/macros/cfg.rs`](https://github.com/tokio-rs/tokio/blob/master/tokio/src/macros/cfg.rs). Note that the annotation works recursively, so putting it on a module also affects the stuff inside (sometimes, it's a bit flaky).

Finally, when building the documentation locally, you run the following command:
```sh
$ RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --all-features
```
### Run `cargo clippy`
```sh
$ cargo clippy --all-features
```

### Run `cargo fmt`