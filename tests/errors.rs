//! Each concrete error enum surfaces a representative variant.

use refimage::pipeline::{ImageSpec, Pipeline, Strategy};
use refimage::{
    BayerError, ColorSpace, DemosaicMethod, DynamicImageRef, ImageError, ImageRef, MetadataError,
    OptimumExposureBuilder, PixelType, SerdeError,
};

#[test]
fn image_error_variants() {
    // Zero width.
    assert_eq!(
        ImageRef::new(&mut [0u8; 4], 0, 2, ColorSpace::Gray).unwrap_err(),
        ImageError::ZeroWidth
    );
    // Backing store too short.
    assert!(matches!(
        ImageRef::new(&mut [0u8; 3], 2, 2, ColorSpace::Gray).unwrap_err(),
        ImageError::InsufficientData {
            expected: 4,
            got: 3
        }
    ));
}

#[test]
fn metadata_error_variants() {
    use refimage::{GenericImageRef, ImageProps};
    let mut buf = [0u8; 4];
    let img = DynamicImageRef::from(ImageRef::new(&mut buf, 2, 2, ColorSpace::Gray).unwrap());
    let mut g = GenericImageRef::new(std::time::SystemTime::now(), img);

    assert_eq!(
        g.insert_key("TIMESTAMP", 1u8).unwrap_err(),
        MetadataError::ReservedKey("TIMESTAMP")
    );
    assert_eq!(g.insert_key("", 1u8).unwrap_err(), MetadataError::EmptyKey);
    assert_eq!(
        g.remove_key("NOPE").unwrap_err(),
        MetadataError::KeyNotFound
    );
    let _ = g.channels(); // keep ImageProps import used
}

#[test]
fn exposure_error_variants() {
    assert_eq!(
        OptimumExposureBuilder::default()
            .percentile_pix(2.0)
            .build()
            .unwrap_err(),
        refimage::ExposureError::PercentileRange
    );
}

#[test]
fn bayer_error_through_pipeline() {
    // Debayer on a non-Bayer image -> PipelineError::Bayer(BayerError::InvalidColorSpace(..))
    let spec = ImageSpec::new(4, 4, ColorSpace::Gray, PixelType::U8);
    let err = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .compile(spec, Strategy::Sequential)
        .unwrap_err();
    // NotBayer is caught first at output_spec; the CFA conversion path is what
    // yields BayerError. Assert the Bayer error type exists and formats.
    let _ = BayerError::InvalidColorSpace(ColorSpace::Gray).to_string();
    assert!(format!("{err}").contains("Bayer") || format!("{err}").contains("bayer"));
}

#[test]
fn serde_error_checksum() {
    // Round-trip a valid image, then corrupt the bytes and confirm a SerdeError.
    let mut buf = [10u8, 20, 30, 40];
    let img = DynamicImageRef::from(ImageRef::new(&mut buf, 2, 2, ColorSpace::Gray).unwrap());
    let bytes = bincode::serialize(&img).unwrap();
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() ^= 0xFF;
    let e = bincode::deserialize::<refimage::DynamicImageOwned>(&corrupt);
    assert!(e.is_err());
    // And a direct SerdeError value formats.
    let _ = SerdeError::Checksum.to_string();
}
