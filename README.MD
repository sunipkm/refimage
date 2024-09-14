# `refimage`
[![crates.io](https://img.shields.io/crates/v/refimage)](https://crates.io/crates/refimage)
[![Documentation](https://docs.rs/refimage/badge.svg)](https://docs.rs/refimage)

## A Serializable Image Container

This crate provides a type-erased image container (`GenericImage`), backed by a contiguous
slice (owned or referenced) of primitive (`u8`, `u16`, `f32`) pixels, with arbitrary
color space (grayscale, Bayer pattern, RGB, ...) and color channels support (max. 255).
Image sizes are limited to 65536 &times; 65536 for practical reasons.

`GenericImage` and `GenericImageOwned` are powerful, since it supports metadata in the form of (`key`, `value`) pairs,
with optional comments. Valid metadata keys are case-insensitive, 80-character strings, and
values are up to 64-bit integer types, 32- and 64-bit floating point numbers, strings up-to 
4096 characters, or `std::time::{SystemTime, Duration}`.

`GenericImage` and `GenericImageOwned` also support serialization and deserialization, and, optionally can be saved
in the open [Flexible Image Transport System (FITS)](https://fits.gsfc.nasa.gov/fits_standard.html)
format by enabling the optional `fitsio` feature. FITS files support lossless compression,
and saving of arbitrary metadata.

`GenericImage` and `GenericImageOwned` are represented by the same underlying object
after serialization, and can be deserialized as either type at the deserializer end.

## The path to a `GenericImage`

A `GenericImage` can be obtained in a couple of ways. In the first case, the Rust program
itself generates the image. In that case, a `ImageData` object is first created with the
appropriate, contiguous, backing storage and image format:
```rust
use refimage::{ColorSpace, ImageData, Debayer, DemosaicMethod, DynamicImageData, GenericImage, GenericImageOwned};
use std::time::SystemTime;

let mut data = vec![0u8; 256]; // this is the backing store
// acquire(&mut data); // this function populates the backing store with the image pixels
let img = ImageData::from_mut_ref(&mut data, 16, 16, ColorSpace::Grbg).unwrap(); // Create a 4x4 image backed by the vector
let img = DynamicImageData::from(img); // convert the `ImageData` object to `DynamicImageData`
let img = img.debayer(DemosaicMethod::Nearest).expect("Could not debayer"); // debayer the image using nearest neighbor method
let mut img = GenericImage::new(SystemTime::now(), img); // Convert to a GenericImage
// insert the camera information as metadata
img.insert_key("CAMERA", ("Rust Test Program", "Name of the camera used to capture the image"));
let json = serde_json::to_string(&img).unwrap(); // serialize the image to JSON
let rimg: GenericImage = serde_json::from_str(&json).unwrap(); // deserialize to GenericImage
assert_eq!(&img, &rimg); // Confirm that deserialized image matches the original
let rimg: GenericImageOwned = rimg.into(); // convert to GenericImageOwned
let ownedjson = serde_json::to_string(&rimg).unwrap(); // serialize the image to JSON
assert_eq!(&json, &ownedjson); // the two representations should be identical
```

In the second case, an image can be loaded using the [`image`](https://crates.io/crates/image) crate from disk, by enabling the `image` feature:
```rust,no_run
use refimage::DynamicImageData;
use image::open;

let img = open("/path/to/image.png").expect("Could not load image");
let img = DynamicImageData::try_from(img).expect("Could not convert image");
```
Note, that the image being loaded must not contain an alpha channel for the conversion to be
successful in the current implementation. However, the `image` crate provides methods to
remove the alpha channel. A custom color space can be used for such an image, however,
the user must keep track of the custom color space information.

## Loading and storing a `GenericImage`
A `GenericImage` is intended to be loaded and stored in a standard format, e.g. [bincode](https://crates.io/crates/bincode) - which follows trivially from the serialization-
deserialization of `GenericImage`. However, for more portable applications, with the `fitsio`
feature, a `GenericImage` can be stored as a FITS file by importing the `FitsWrite` trait. 
The FITS file is stored using the [`fitsio`](https://crates.io/crates/fitsio) crate, which is a thin wrapper around the `cfitsio` library.
```rust,no_run
use refimage::{FitsCompression, FitsWrite, GenericImage}; // we need the FitsWrite trait to be able to use the `write_fits` method.
use std::path::Path;
let img: GenericImage = { todo!() }; // obtain a GenericImage
img.write_fits(Path::new("/path/to/fitsimage.fit"), FitsCompression::None, true) // no compression, overwrite
    .expect("Could not write FITS file.");
```
# Optional Features
Features are available to extend the functionalities of the core `refimage` data types:
- `rayon`: Parallelizes `into_luma`, `into_luma_custom`, `into_u8` and `debayer` functions (<b>enabled</b> by default).
- `serde_flate`: Compresses the data using deflate during serialization (<b>enabled</b> by default).
- `fitsio`: Exposes `FitsWrite` trait to write `GenericImage` and `GenericImageOwned` (<b>disabled</b> by default).
- `image`: Enables `TryFrom` conversions between `image::DynamicImage` and `refimage::DynamicImageData`, `refimage::DynamicImageOwned` (<b>disabled</b> by default).