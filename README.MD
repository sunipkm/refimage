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
use refimage::{BayerPattern, ImageRef, Debayer, DemosaicMethod, DynamicImageRef, GenericImageRef, GenericImageOwned};
use std::time::SystemTime;

let mut data = vec![0u8; 256]; // this is the backing store
// acquire(&mut data); // this function populates the backing store with the image pixels
let img = ImageRef::new(&mut data, 16, 16, BayerPattern::Grbg.into()).unwrap(); // Create a 4x4 image backed by the vector
let img = DynamicImageRef::from(img); // convert the `ImageRef` object to `DynamicImageRef`
let img = img.debayer(DemosaicMethod::Nearest).expect("Could not debayer"); // debayer the image using nearest neighbor method
let mut img = GenericImageOwned::new(SystemTime::now(), img); // Convert to a GenericImageRef
// insert the camera information as metadata
img.insert_key("CAMERA", ("Rust Test Program", "Name of the camera used to capture the image"));
let json = serde_json::to_string(&img).unwrap(); // serialize the image to JSON
let rimg: GenericImageOwned = serde_json::from_str(&json).unwrap(); // deserialize to GenericImageOwned
assert_eq!(&img, &rimg); // Confirm that deserialized image matches the original
```
The intention behind `*ImageRef` is to minimize unnecessary allocations.

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

## Additional Features
`ToLuma` trait is implemented on all image functions.
Luminance calculation is done in-place.

# Optional Features
Features are available to extend the functionalities of the core `refimage` data types:
- `rayon`: Parallelizes `to_luma`, `to_luma_custom`, `to_u8` and `debayer` functions (<b>enabled</b> by default).
- `serde_flate`: Compresses the data using deflate during serialization (<b>enabled</b> by default).
- `fitsio`: Exposes `FitsWrite` trait to write `GenericImageRef` and `GenericImageOwned` (<b>disabled</b> by default).
- `image`: Enables `TryFrom` conversions between `image::DynamicImage` and `refimage::DynamicImageRef`, `refimage::DynamicImageOwned` (<b>disabled</b> by default).