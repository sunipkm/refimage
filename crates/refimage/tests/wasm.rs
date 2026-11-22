//! WebAssembly smoke tests.
//!
//! `refimage` links `std`, but never reaches for a system clock, threads, or the
//! filesystem on its own, so its core works on `wasm32-unknown-unknown`. These
//! tests pin that down: image construction, the whole pipeline (one-shot and
//! compiled-runner), metadata, and `serde` round-tripping.
//!
//! Run them with:
//!
//! ```text
//! cargo install wasm-bindgen-cli          # provides wasm-bindgen-test-runner
//! cargo test --target wasm32-unknown-unknown --no-default-features --tests
//! ```
//!
//! `--no-default-features` drops `rayon` — `wasm32-unknown-unknown` has no
//! threads, so the pipeline runs its serial kernels. `--tests` skips the
//! doc-tests, which rustdoc does not cross-compile. The file compiles to
//! nothing on every other target.
#![cfg(target_arch = "wasm32")]

use std::time::Duration;

use refimage::chrono::DateTime;
use refimage::pipeline::{ImageSpec, Pipeline, Strategy};
use refimage::{
    BayerPattern, ColorSpace, DemosaicMethod, DynamicImageOwned, DynamicImageRef, FitsCompression,
    FitsWrite, GenericImageOwned, Gzip, Hcompress, ImageOwned, ImageProps, ImageRef, PixelData,
    PixelType, Rice,
};
use wasm_bindgen_test::wasm_bindgen_test;

/// A deterministic 8x8 RGGB frame.
fn bayer_frame() -> Vec<u16> {
    (0..64u16).map(|i| i.wrapping_mul(37) % 4096).collect()
}

#[wasm_bindgen_test]
fn image_ref_reports_its_shape() {
    let mut data = bayer_frame();
    let img = ImageRef::new(&mut data, 8, 8, ColorSpace::Bayer(BayerPattern::Rggb)).unwrap();
    assert_eq!(img.width(), 8);
    assert_eq!(img.height(), 8);
    assert_eq!(img.channels(), 1);
    assert_eq!(img.color_space(), ColorSpace::Bayer(BayerPattern::Rggb));
}

#[wasm_bindgen_test]
fn pipeline_apply_debayers_to_gray_u8() {
    let mut data = bayer_frame();
    let img = DynamicImageRef::from(
        ImageRef::new(&mut data, 8, 8, ColorSpace::Bayer(BayerPattern::Rggb)).unwrap(),
    );

    let out = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .to_luma()
        .scale(1.1, 3.0)
        .convert(PixelType::U8)
        .flip_vertical()
        .apply(&img)
        .expect("pipeline runs on wasm");

    assert_eq!(out.width(), 8);
    assert_eq!(out.height(), 8);
    assert_eq!(out.color_space(), ColorSpace::Gray);
    assert_eq!(out.pixel_type(), PixelType::U8);
}

#[wasm_bindgen_test]
fn compiled_runner_is_reusable_across_frames() {
    let spec = ImageSpec::new(8, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::None)
        .to_luma()
        .convert(PixelType::U8)
        .compile(spec, Strategy::Sequential)
        .expect("compile on wasm");

    let mut first = None;
    for _ in 0..4 {
        let mut data = bayer_frame();
        let img = DynamicImageRef::from(
            ImageRef::new(&mut data, 8, 8, ColorSpace::Bayer(BayerPattern::Rggb)).unwrap(),
        );
        let out = runner.run(&img).expect("run");
        let bytes = out.as_raw_u8().to_vec();
        assert_eq!(*first.get_or_insert_with(|| bytes.clone()), bytes);
    }
}

#[wasm_bindgen_test]
fn dynamic_image_serde_round_trips() {
    let mut data = bayer_frame();
    let img = DynamicImageRef::from(ImageRef::new(&mut data, 8, 8, ColorSpace::Gray).unwrap());

    let bytes = bincode::serialize(&img).expect("serialize");
    let back: DynamicImageOwned = bincode::deserialize(&bytes).expect("deserialize");

    assert_eq!(back.width(), 8);
    assert_eq!(back.height(), 8);
    assert_eq!(back.as_raw_u8(), img.as_raw_u8());
}

#[wasm_bindgen_test]
fn generic_image_carries_metadata() {
    let mut data = vec![0u8; 16];
    let img = DynamicImageOwned::from(&DynamicImageRef::from(
        ImageRef::new(&mut data, 4, 4, ColorSpace::Gray).unwrap(),
    ));

    // A caller-supplied timestamp — no `SystemTime::now()`, which panics on
    // `wasm32-unknown-unknown`.
    let stamp = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    let mut g = GenericImageOwned::new(stamp, Duration::ZERO, img);
    g.insert_key("GAIN", 42u16).unwrap();

    assert_eq!(g.timestamp(), stamp);
    assert_eq!(
        g.key("GAIN")
            .and_then(|it| it.value().value_u16()),
        Some(42)
    );
}

#[wasm_bindgen_test]
fn fits_bytes_on_wasm() {
    let data: Vec<u16> = (0..16 * 16).map(|i| (i as u16).wrapping_mul(257)).collect();
    let img =
        DynamicImageOwned::from(ImageOwned::from_owned(data, 16, 16, ColorSpace::Gray).unwrap());
    let stamp = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    let g = GenericImageOwned::new(stamp, Duration::from_millis(250), img);

    for comp in [
        FitsCompression::NONE,
        Gzip::new().into(),
        Rice::new().into(),
        Hcompress::new().into(),
    ] {
        let bytes = g.fits_bytes(&comp).expect("fits_bytes");
        assert_eq!(bytes.len() % 2880, 0, "{comp:?} not block-aligned");
        assert!(
            bytes.starts_with(b"SIMPLE  =                    T"),
            "{comp:?}"
        );
    }

    // f32 + Rice exercises the quantization path (seeded RNG table) on wasm.
    let f: Vec<f32> = (0..16 * 16)
        .map(|i| (i as f32 * 0.1).sin() * 40.0 + 100.0)
        .collect();
    let fimg =
        DynamicImageOwned::from(ImageOwned::from_owned(f, 16, 16, ColorSpace::Gray).unwrap());
    let gf = GenericImageOwned::new(stamp, Duration::ZERO, fimg);
    for comp in [FitsCompression::from(Rice::new()), Hcompress::new().into()] {
        let bytes = gf.fits_bytes(&comp).expect("f32 compress");
        assert_eq!(bytes.len() % 2880, 0);
    }

    // In-memory multi-HDU file — no filesystem on wasm.
    let mut w = refimage::create_fits_to(Vec::new(), Rice::new()).expect("create_fits_to");
    g.append_fits(&mut w).expect("append 1");
    g.append_fits(&mut w).expect("append 2");
    let file = w.finish().expect("finish");
    assert_eq!(file.len() % 2880, 0);
    assert!(file.starts_with(b"SIMPLE  =                    T"));
}
