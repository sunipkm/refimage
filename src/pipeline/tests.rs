use super::*;
use crate::PixelStor;

fn sample16(n: usize) -> Vec<u16> {
    (0..n).map(|i| ((i * 7 + 3) % 251) as u16).collect()
}

fn bayer_frame(w: usize, h: usize, pat: BayerPattern, data: &mut [u16]) -> DynamicImageRef<'_> {
    DynamicImageRef::from(ImageRef::new(data, w, h, ColorSpace::Bayer(pat)).unwrap())
}

/// A growing in-place conversion (`u16 -> f32`) rewrites its buffer back to
/// front and must still match an element-wise reference.
#[test]
fn growing_convert_in_place_matches_reference() {
    let (w, h) = (7, 5);
    let src = sample16(w * h);
    let expected: Vec<f32> = src.iter().map(|&v| v.cast_f32()).collect();

    let spec = ImageSpec::new(w, h, ColorSpace::Gray, PixelType::U16);
    let mut runner = Pipeline::new()
        .convert(PixelType::F32)
        .compile(spec, Strategy::Sequential)
        .unwrap();
    let mut frame = src.clone();
    let img = DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Gray).unwrap());
    let out = runner.run(&img).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(out.as_raw_u8()),
        &expected[..]
    );
}

fn gray16(w: usize, h: usize, data: &mut [u16]) -> DynamicImageRef<'_> {
    DynamicImageRef::from(ImageRef::new(data, w, h, ColorSpace::Gray).unwrap())
}

/// Each geometric op relocates pixels according to its coordinate map.
#[test]
fn geo_ops_sequential_reference() {
    let (w, h) = (8usize, 6usize);
    let src: Vec<u16> = (0..(w * h) as u16).collect();
    let at = |d: &[u16], c: usize, r: usize, rw: usize| d[r * rw + c];

    let run = |p: Pipeline| {
        let mut runner = p
            .compile(
                ImageSpec::new(w, h, ColorSpace::Gray, PixelType::U16),
                Strategy::Sequential,
            )
            .unwrap();
        let mut f = src.clone();
        let out = runner.run(&gray16(w, h, &mut f)).unwrap();
        (
            out.width(),
            out.height(),
            bytemuck::cast_slice::<u8, u16>(out.as_raw_u8()).to_vec(),
        )
    };

    let (ow, oh, d) = run(Pipeline::new().flip_horizontal());
    assert_eq!((ow, oh), (w, h));
    for r in 0..h {
        for c in 0..w {
            assert_eq!(at(&d, c, r, w), at(&src, w - 1 - c, r, w));
        }
    }

    let (_, _, d) = run(Pipeline::new().flip_vertical());
    for r in 0..h {
        for c in 0..w {
            assert_eq!(at(&d, c, r, w), at(&src, c, h - 1 - r, w));
        }
    }

    let (_, _, d) = run(Pipeline::new().rotate_180());
    for r in 0..h {
        for c in 0..w {
            assert_eq!(at(&d, c, r, w), at(&src, w - 1 - c, h - 1 - r, w));
        }
    }

    let (ow, oh, d) = run(Pipeline::new().rotate_90());
    assert_eq!((ow, oh), (h, w));
    for r in 0..w {
        for c in 0..h {
            assert_eq!(at(&d, c, r, h), at(&src, r, h - 1 - c, w));
        }
    }

    let (ow, oh, d) = run(Pipeline::new().rotate_270());
    assert_eq!((ow, oh), (h, w));
    for r in 0..w {
        for c in 0..h {
            assert_eq!(at(&d, c, r, h), at(&src, w - 1 - r, c, w));
        }
    }

    let (ow, oh, d) = run(Pipeline::new().crop(2, 1, 4, 3));
    assert_eq!((ow, oh), (4, 3));
    for r in 0..3 {
        for c in 0..4 {
            assert_eq!(at(&d, c, r, 4), at(&src, 2 + c, 1 + r, w));
        }
    }
}

/// A geometric tail on a tiled debayer chain stays byte-identical to
/// `Sequential`, and a leading crop still lets the pixel run tile.
#[test]
fn geo_with_tiling_matches_sequential() {
    let (w, h) = (32usize, 24usize);
    let pat = BayerPattern::Rggb;
    let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);

    let chains: Vec<(&str, Pipeline)> = vec![
        (
            "trailing flip_v",
            Pipeline::new()
                .debayer(DemosaicMethod::Linear)
                .to_luma()
                .convert(PixelType::U8)
                .flip_vertical(),
        ),
        (
            "trailing rotate_90",
            Pipeline::new()
                .debayer(DemosaicMethod::Cubic)
                .to_luma()
                .rotate_90(),
        ),
        (
            "trailing rotate_270 + crop",
            Pipeline::new()
                .debayer(DemosaicMethod::Nearest)
                .to_luma()
                .rotate_270()
                .crop(1, 2, 10, 20),
        ),
        (
            "leading crop",
            Pipeline::new()
                .crop(4, 2, 20, 16)
                .debayer(DemosaicMethod::Linear)
                .to_luma()
                .convert(PixelType::U8),
        ),
        (
            "leading crop + trailing rot180",
            Pipeline::new()
                .crop(2, 4, 24, 16)
                .debayer(DemosaicMethod::Cubic)
                .to_luma()
                .convert(PixelType::U8)
                .rotate_180(),
        ),
    ];

    for (name, p) in chains {
        let mut seq = p.clone().compile(spec(), Strategy::Sequential).unwrap();
        let mut f0 = sample16(w * h);
        let want = seq
            .run(&bayer_frame(w, h, pat, &mut f0))
            .unwrap()
            .as_raw_u8()
            .to_vec();

        for strat in [
            Strategy::tiled(4),
            Strategy::tiled_parallel(5),
            Strategy::tiled_2d(6, 8),
        ] {
            let mut r = p.clone().compile(spec(), strat).unwrap();
            let mut f = sample16(w * h);
            let got = r
                .run(&bayer_frame(w, h, pat, &mut f))
                .unwrap()
                .as_raw_u8()
                .to_vec();
            assert_eq!(got, want, "{name} / {strat:?}");
        }
    }
}

#[test]
fn geo_validation_errors() {
    let bayer = ImageSpec::new(8, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    assert!(matches!(
        Pipeline::new()
            .rotate_90()
            .compile(bayer.clone(), Strategy::Sequential)
            .unwrap_err(),
        PipelineError::RotateOnBayer
    ));
    assert!(matches!(
        Pipeline::new()
            .crop(4, 4, 6, 2)
            .compile(bayer.clone(), Strategy::Sequential)
            .unwrap_err(),
        PipelineError::CropOutOfBounds { .. }
    ));
    // An ROI origin outside the image is still an error...
    assert!(matches!(
        Pipeline::new()
            .roi(8, 0, 2, 2)
            .compile(bayer, Strategy::Sequential)
            .unwrap_err(),
        PipelineError::RoiOutOfBounds { .. }
    ));
}

/// `Op::Roi` reproduces the old `SelectRoi`: clamp the overlap, zero-fill the
/// overhang, error only on an out-of-image origin.
#[test]
fn roi_matches_select_roi_reference() {
    let mut data: Vec<u16> = vec![0, 1, 2, 3, 4, 6, 5, 7, 8, 9]; // 5x2, row-major
    let img = gray16(5, 2, &mut data);
    let out = Pipeline::new().roi(1, 0, 2, 3).apply(&img).unwrap();
    let got: &[u16] = bytemuck::cast_slice(out.as_raw_u8());
    assert_eq!(got, &[1, 2, 5, 7, 0, 0]);
    assert_eq!((out.width(), out.height()), (2, 3));
}

/// `Runner::run_into` writes into a caller buffer — the old `CopyRoi::copy_to`.
#[test]
fn run_into_reproduces_copy_to() {
    let spec = ImageSpec::new(5, 2, ColorSpace::Gray, PixelType::U16);
    let mut r = Pipeline::new()
        .roi(1, 0, 2, 3)
        .compile(spec, Strategy::Sequential)
        .unwrap();

    // Pre-sized destination: filled in place, no realloc.
    let mut dest = DynamicImageOwned::from(
        crate::ImageOwned::from_owned(vec![0u16; 6], 2, 3, ColorSpace::Gray).unwrap(),
    );
    let mut data: Vec<u16> = vec![0, 1, 2, 3, 4, 6, 5, 7, 8, 9];
    r.run_into(&gray16(5, 2, &mut data), &mut dest).unwrap();
    assert_eq!(
        dest.as_raw_u8_checked()
            .map(bytemuck::cast_slice::<u8, u16>),
        Some(&[1, 2, 5, 7, 0, 0][..])
    );

    // Wrong-shaped destination: rebuilt.
    let mut dest2 = DynamicImageOwned::from(
        crate::ImageOwned::from_owned(vec![9u16; 4], 2, 2, ColorSpace::Gray).unwrap(),
    );
    let mut data2: Vec<u16> = vec![0, 1, 2, 3, 4, 6, 5, 7, 8, 9];
    r.run_into(&gray16(5, 2, &mut data2), &mut dest2).unwrap();
    assert_eq!((dest2.width(), dest2.height()), (2, 3));
}

/// `apply_meta` carries a `GenericImageRef`'s metadata onto the owned result.
#[test]
fn apply_meta_preserves_metadata() {
    use std::time::Duration;
    let mut data: Vec<u16> = (0..24).collect();
    let dynref = gray16(4, 6, &mut data);
    let ts = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    let mut gen = GenericImageRef::new(ts, Duration::from_millis(50), dynref);
    gen.insert_key("CAMERA", "test-cam").unwrap();

    let out = Pipeline::new()
        .convert(PixelType::U8)
        .apply_meta(&gen)
        .unwrap();
    assert_eq!(out.get_metadata(), gen.get_metadata());
    assert_eq!(out.get_timestamp(), ts);
    assert_eq!(out.get_exposure(), Duration::from_millis(50));
    assert_eq!(out.pixel_type(), PixelType::U8);
}

/// Every tiling of a debayer chain must be byte-identical to `Sequential`.
#[test]
fn tiled_matches_sequential_debayer_chain() {
    let (w, h) = (24, 20);
    let build = |m| Pipeline::new().debayer(m).to_luma().convert(PixelType::U8);

    for pat in [
        BayerPattern::Rggb,
        BayerPattern::Bggr,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
    ] {
        for method in [
            DemosaicMethod::None,
            DemosaicMethod::Nearest,
            DemosaicMethod::Linear,
            DemosaicMethod::Cubic,
        ] {
            let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);
            let mut seq = build(method).compile(spec(), Strategy::Sequential).unwrap();
            let mut f0 = sample16(w * h);
            let want = seq
                .run(&bayer_frame(w, h, pat, &mut f0))
                .unwrap()
                .as_raw_u8()
                .to_vec();

            let strategies = [
                Strategy::tiled(1),
                Strategy::tiled(3),
                Strategy::tiled(7),
                Strategy::tiled_parallel(4),
                Strategy::tiled_2d(4, 6),
                Strategy::tiled_2d(3, 8),
                Strategy::tiled_2d_parallel(5, 7),
                Strategy::tiled_2d(1, 1),
            ];
            for strat in strategies {
                let mut r = build(method).compile(spec(), strat).unwrap();
                let mut f = sample16(w * h);
                let got = r
                    .run(&bayer_frame(w, h, pat, &mut f))
                    .unwrap()
                    .as_raw_u8()
                    .to_vec();
                assert_eq!(got, want, "pat={pat:?} method={method:?} strat={strat:?}");
            }
        }
    }
}

#[test]
fn tiled_matches_sequential_row_local_chain() {
    let (w, h) = (10, 40);
    let build = || Pipeline::new().to_luma().convert(PixelType::U8);
    let spec = || ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U16);
    let mut seq = build().compile(spec(), Strategy::Sequential).unwrap();
    let mut f0 = sample16(w * h * 3);
    let img0 = DynamicImageRef::from(ImageRef::new(&mut f0, w, h, ColorSpace::Rgb).unwrap());
    let want = seq.run(&img0).unwrap().as_raw_u8().to_vec();

    for strat in [
        Strategy::tiled_parallel(6),
        Strategy::tiled_2d(6, 4),
        Strategy::tiled(1),
    ] {
        let mut r = build().compile(spec(), strat).unwrap();
        let mut f = sample16(w * h * 3);
        let img = DynamicImageRef::from(ImageRef::new(&mut f, w, h, ColorSpace::Rgb).unwrap());
        assert_eq!(r.run(&img).unwrap().as_raw_u8(), want.as_slice());
    }
}

/// A shrinking chain sizes its two buffers independently.
#[test]
fn independent_buffer_sizing() {
    // u16 Bayer 100x100 -> debayer  (swaps A->B; B = 100*100*3*2 = 60000)
    //                   -> luma     (in place in B; 100*100*2 = 20000)
    //                   -> u8       (in place in B; 100*100*1 = 10000)
    // A only ever holds the input: 20000 B
    // B peaks at the RGB intermediate: 60000 B
    let spec = ImageSpec::new(
        100,
        100,
        ColorSpace::Bayer(BayerPattern::Rggb),
        PixelType::U16,
    );
    let runner = Pipeline::new()
        .debayer(DemosaicMethod::Nearest)
        .to_luma()
        .convert(PixelType::U8)
        .compile(spec, Strategy::Sequential)
        .unwrap();
    // Naive sizing would be 2 * 60000 = 120000 B; the walk gets 80000 B.
    assert!(
        runner.scratch_bytes() <= 80_000,
        "scratch {} B, expected <= 80000",
        runner.scratch_bytes()
    );
}

/// A chain with no debayer never swaps, so the second buffer stays unused.
#[test]
fn no_swap_chain_uses_one_buffer() {
    let spec = ImageSpec::new(64, 64, ColorSpace::Rgb, PixelType::U16);
    let runner = Pipeline::new()
        .scale(1.5, 0.0)
        .to_luma()
        .convert(PixelType::U8)
        .compile(spec, Strategy::Sequential)
        .unwrap();
    // buf_a holds the rgb u16 input (64*64*3*2 = 24576 B); buf_b is a stub.
    assert!(
        runner.scratch_bytes() <= 24_576 + 16,
        "scratch {} B",
        runner.scratch_bytes()
    );
}

#[test]
fn scale_tiles_identically() {
    let (w, h) = (16, 12);
    let pat = BayerPattern::Rggb;
    let build = || {
        Pipeline::new()
            .debayer(DemosaicMethod::Linear)
            .scale(0.5, 10.0)
            .to_luma()
            .convert(PixelType::U8)
    };
    let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);

    let mut seq = build().compile(spec(), Strategy::Sequential).unwrap();
    let mut f0 = sample16(w * h);
    let want = seq
        .run(&bayer_frame(w, h, pat, &mut f0))
        .unwrap()
        .as_raw_u8()
        .to_vec();

    for strat in [
        Strategy::tiled(3),
        Strategy::tiled_parallel(4),
        Strategy::tiled_2d(4, 6),
    ] {
        let mut r = build().compile(spec(), strat).unwrap();
        let mut f = sample16(w * h);
        let got = r
            .run(&bayer_frame(w, h, pat, &mut f))
            .unwrap()
            .as_raw_u8()
            .to_vec();
        assert_eq!(got, want, "strat={strat:?}");
    }
}

/// `Scale` saturates instead of panicking when the affine map drives a
/// value out of the type's range (negative offset, gain > 1).
#[test]
fn scale_saturates_out_of_range() {
    let (w, h) = (8, 4);
    let mut data: Vec<u16> = (0..(w * h) as u16).map(|v| v * 1000).collect();

    let mut r = Pipeline::new()
        .scale(4.0, -5000.0)
        .compile(
            ImageSpec::new(w, h, ColorSpace::Gray, PixelType::U16),
            Strategy::Sequential,
        )
        .unwrap();
    let got: Vec<u16> = {
        let out = r.run(&gray16(w, h, &mut data)).unwrap();
        bytemuck::cast_slice::<u8, u16>(out.as_raw_u8()).to_vec()
    };

    for (i, &g) in got.iter().enumerate() {
        let want = u16::from_f64((i as f64 * 1000.0) * 4.0 - 5000.0);
        assert_eq!(g, want);
    }
    // First few pixels underflow to 0, last few clamp to u16::MAX.
    assert_eq!(got[0], 0);
    assert_eq!(*got.last().unwrap(), u16::MAX);
}

#[test]
fn recompile_adapts_to_new_shape() {
    let pat = BayerPattern::Rggb;
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .to_luma()
        .compile(
            ImageSpec::new(16, 16, ColorSpace::Bayer(pat), PixelType::U16),
            Strategy::tiled(4),
        )
        .unwrap();

    let mut f16 = sample16(16 * 16);
    assert!(runner.run(&bayer_frame(16, 16, pat, &mut f16)).is_ok());

    runner
        .recompile(ImageSpec::new(
            32,
            24,
            ColorSpace::Bayer(pat),
            PixelType::U16,
        ))
        .unwrap();
    assert_eq!(runner.input_spec().width, 32);

    let mut f32 = sample16(32 * 24);
    let out = runner.run(&bayer_frame(32, 24, pat, &mut f32)).unwrap();
    assert_eq!((out.width(), out.height()), (32, 24));
}

#[test]
fn pipeline_round_trips_through_json() {
    let p = Pipeline::new()
        .crop(1, 1, 6, 6)
        .debayer(DemosaicMethod::Cubic)
        .scale(1.25, -3.0)
        .to_luma_custom(vec![0.2, 0.7, 0.1])
        .convert(PixelType::U8)
        .rotate_90();
    let json = serde_json::to_string(&p).unwrap();
    let back: Pipeline = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);

    // and it still compiles + runs
    let spec = ImageSpec::new(8, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    assert!(back.compile(spec, Strategy::Sequential).is_ok());
}

#[test]
#[cfg(not(feature = "grow"))]
fn rejects_wrong_input_shape() {
    let spec = ImageSpec::new(8, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Nearest)
        .compile(spec, Strategy::Sequential)
        .unwrap();
    let mut frame = vec![0u16; 36];
    let img = bayer_frame(6, 6, BayerPattern::Rggb, &mut frame);
    assert!(matches!(
        runner.run(&img),
        Err(PipelineError::InputMismatch { .. })
    ));
}

#[test]
#[cfg(feature = "grow")]
fn grow_feature_auto_recompiles() {
    let pat = BayerPattern::Rggb;
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Nearest)
        .compile(
            ImageSpec::new(8, 8, ColorSpace::Bayer(pat), PixelType::U16),
            Strategy::Sequential,
        )
        .unwrap();
    let mut frame = sample16(6 * 6);
    let out = runner.run(&bayer_frame(6, 6, pat, &mut frame)).unwrap();
    assert_eq!((out.width(), out.height()), (6, 6));
    assert_eq!(runner.input_spec().width, 6);
}

#[test]
fn tiny_frame_falls_back_to_sequential() {
    let spec = ImageSpec::new(8, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    let runner = Pipeline::new()
        .debayer(DemosaicMethod::Cubic)
        .compile(spec, Strategy::tiled(2))
        .unwrap();
    assert!(!runner.is_tiled());
}

#[test]
fn rejects_invalid_chain_at_compile() {
    let spec = ImageSpec::new(8, 8, ColorSpace::Gray, PixelType::U8);
    let err = Pipeline::new()
        .debayer(DemosaicMethod::Nearest)
        .compile(spec, Strategy::Sequential)
        .unwrap_err();
    assert!(matches!(err, PipelineError::NotBayer));
}
