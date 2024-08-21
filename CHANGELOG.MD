# Changelog for `refimage`

### Check documentation
```sh
$ RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --all-features
```

### 0.1.1 (2024-08-20)
- Updated documentation.

### 0.2.0 (2024-08-21)
- Moved the `debayer` method to its own trait, and removed public `debayer` methods
from `DynamicImageData` and `ImageData`.
- Renamed enum `Demosaic` to `DemosaicMethod`.
- Implemented `Debayer::debayer` for `GenericImage`.
- Implemented `GenericImage::operate()` that allows replacement of the underlying
image while conserving the metadata.
- Added this change log.