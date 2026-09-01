use super::*;
use crate::{
    BayerPattern, ColorSpace, DemosaicMethod, DynamicImageOwned, DynamicImageRef, GenericImageRef,
    ImageProps, ImageRef, PixelStor, PixelType,
};

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

/// `apply` on a `GenericImageRef` carries its metadata onto the owned result.
#[test]
fn apply_preserves_metadata() {
    use std::time::Duration;
    let mut data: Vec<u16> = (0..24).collect();
    let dynref = gray16(4, 6, &mut data);
    let ts = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
    let mut gen = GenericImageRef::new(ts, Duration::from_millis(50), dynref);
    gen.insert_key("CAMERA", "test-cam").unwrap();

    let out = Pipeline::new().convert(PixelType::U8).apply(&gen).unwrap();
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

fn mk_strat(tile_rows: usize, tile_cols: usize, parallel: bool) -> Strategy {
    match (tile_cols, parallel) {
        (0, false) => Strategy::tiled(tile_rows),
        (0, true) => Strategy::tiled_parallel(tile_rows),
        (c, false) => Strategy::tiled_2d(tile_rows, c),
        (c, true) => Strategy::tiled_2d_parallel(tile_rows, c),
    }
}

/// Genuinely-tiled execution (asserted with [`Runner::is_tiled`]) must stay
/// byte-identical to `Sequential` when the tile size divides neither the width nor the
/// height. Dimensions are primes / a mix of odd and even; every tile size below is
/// coprime with all of them, so every band and every column tile has a ragged edge.
#[test]
fn tiled_matches_sequential_ragged_dims() {
    let dims = [(53, 71), (53, 64), (64, 53), (47, 47)];
    // (tile_rows, tile_cols); tile_cols == 0 -> full-width bands. All large enough to
    // clear the `2*halo + 6` fallback threshold even for `Cubic` (halo 3).
    let tilings = [(16, 0), (19, 0), (11, 17), (23, 29), (13, 0)];

    let mut tiled = 0usize;
    let mut check = |strat: Strategy, is_tiled: bool, got: &[u8], want: &[u8], ctx: &str| {
        assert!(is_tiled, "expected genuine tiling for {ctx} {strat:?}");
        assert_eq!(got, want, "{ctx} {strat:?}");
        tiled += 1;
    };

    for &(w, h) in &dims {
        // (a) row-local chain on RGB (halo 0).
        {
            let build = || Pipeline::new().to_luma().convert(PixelType::U8);
            let spec = || ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U16);
            let mut f0 = sample16(w * h * 3);
            let want = build()
                .compile(spec(), Strategy::Sequential)
                .unwrap()
                .run(&DynamicImageRef::from(
                    ImageRef::new(&mut f0, w, h, ColorSpace::Rgb).unwrap(),
                ))
                .unwrap()
                .as_raw_u8()
                .to_vec();

            for &(tr, tc) in &tilings {
                for parallel in [false, true] {
                    let strat = mk_strat(tr, tc, parallel);
                    let mut r = build().compile(spec(), strat).unwrap();
                    let is_tiled = r.is_tiled();
                    let mut f = sample16(w * h * 3);
                    let got = r
                        .run(&DynamicImageRef::from(
                            ImageRef::new(&mut f, w, h, ColorSpace::Rgb).unwrap(),
                        ))
                        .unwrap()
                        .as_raw_u8()
                        .to_vec();
                    check(strat, is_tiled, &got, &want, &format!("row-local {w}x{h}"));
                }
            }
        }

        // (b) debayer chain — halo > 0, even-snapped tile edges. `Cubic` (halo 3) is
        // the widest and stresses the edge padding the most.
        for pat in [BayerPattern::Rggb, BayerPattern::Gbrg] {
            for method in [DemosaicMethod::Linear, DemosaicMethod::Cubic] {
                let build = || {
                    Pipeline::new()
                        .debayer(method)
                        .to_luma()
                        .convert(PixelType::U8)
                };
                let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);
                let mut f0 = sample16(w * h);
                let want = build()
                    .compile(spec(), Strategy::Sequential)
                    .unwrap()
                    .run(&bayer_frame(w, h, pat, &mut f0))
                    .unwrap()
                    .as_raw_u8()
                    .to_vec();

                for &(tr, tc) in &tilings {
                    for parallel in [false, true] {
                        let strat = mk_strat(tr, tc, parallel);
                        let mut r = build().compile(spec(), strat).unwrap();
                        let is_tiled = r.is_tiled();
                        let mut f = sample16(w * h);
                        let got = r
                            .run(&bayer_frame(w, h, pat, &mut f))
                            .unwrap()
                            .as_raw_u8()
                            .to_vec();
                        check(
                            strat,
                            is_tiled,
                            &got,
                            &want,
                            &format!("debayer {w}x{h} {pat:?} {method:?}"),
                        );
                    }
                }
            }
        }
    }
    assert_eq!(tiled, dims.len() * (5 * 2 + 2 * 2 * 5 * 2));
}

/// The documented fall-backs to `Sequential`: a frame shorter than `2*halo + 6`, or a
/// tile that spans the whole frame. The output must still be correct.
#[test]
fn tiling_falls_back_to_sequential_when_it_cannot_help() {
    let build = || Pipeline::new().debayer(DemosaicMethod::Cubic).to_luma();
    // 8 rows < 2*3 + 6 for Cubic -> no Y tiling; full-width bands -> no X tiling.
    let spec = ImageSpec::new(40, 8, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
    for strat in [
        Strategy::tiled(2),
        Strategy::tiled_parallel(2),
        Strategy::tiled_2d(2, 4),
    ] {
        let mut r = build().compile(spec.clone(), strat).unwrap();
        assert!(!r.is_tiled(), "{strat:?} should have fallen back");
        let mut f = sample16(40 * 8);
        let mut s = build().compile(spec.clone(), Strategy::Sequential).unwrap();
        let mut f2 = sample16(40 * 8);
        assert_eq!(
            r.run(&bayer_frame(40, 8, BayerPattern::Rggb, &mut f))
                .unwrap()
                .as_raw_u8(),
            s.run(&bayer_frame(40, 8, BayerPattern::Rggb, &mut f2))
                .unwrap()
                .as_raw_u8(),
        );
    }

    // A tall frame but tile_rows >= height -> one band, no column tiling -> fall back.
    let tall = ImageSpec::new(20, 100, ColorSpace::Rgb, PixelType::U16);
    let r = Pipeline::new()
        .to_luma()
        .compile(tall, Strategy::tiled(200))
        .unwrap();
    assert!(!r.is_tiled());
}

/// A realistic sensor frame (1944x1472): tiled output stays bit-identical to
/// `Sequential` for tile grids that divide the frame cleanly, leave a ragged edge on
/// one axis, or on both — including the auto tile height.
///
/// `#[ignore]`d because a ~2.9 MP debayer in a debug build is slow; run it with
/// `cargo test --all-features -- --ignored full_sensor_frame`.
#[test]
#[ignore = "multi-megapixel sweep; slow in debug"]
fn tiled_matches_sequential_full_sensor_frame() {
    let (w, h) = (1944usize, 1472usize);
    // 1944 = 2^3 * 3^5, 1472 = 2^6 * 23.
    let tilings = [
        (0, 0),     // auto height, full-width bands
        (256, 0),   // 1472 = 256*5 + 192  -> ragged Y
        (200, 0),   // 1472 = 200*7 + 72   -> ragged Y
        (64, 0),    // 1472 = 64*23        -> clean Y
        (256, 512), // 1472 % 256 != 0, 1944 % 512 != 0 -> ragged both
        (180, 300), // ragged both
        (64, 8),    // 64*23, 8*243        -> clean both
        (64, 64),   // 64*23, 64*30 + 24   -> ragged X
    ];

    // Row-local: RGB -> luma -> u8 (halo 0). Cheap; also exercises pure banding.
    {
        let build = || Pipeline::new().to_luma().convert(PixelType::U8);
        let spec = || ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U16);
        let mut f0 = sample16(w * h * 3);
        let want = build()
            .compile(spec(), Strategy::Sequential)
            .unwrap()
            .run(&DynamicImageRef::from(
                ImageRef::new(&mut f0, w, h, ColorSpace::Rgb).unwrap(),
            ))
            .unwrap()
            .as_raw_u8()
            .to_vec();
        for &(tr, tc) in &tilings {
            for parallel in [false, true] {
                let strat = mk_strat(tr, tc, parallel);
                let mut r = build().compile(spec(), strat).unwrap();
                assert!(r.is_tiled(), "row-local {strat:?}");
                let mut f = sample16(w * h * 3);
                let got = r
                    .run(&DynamicImageRef::from(
                        ImageRef::new(&mut f, w, h, ColorSpace::Rgb).unwrap(),
                    ))
                    .unwrap()
                    .as_raw_u8()
                    .to_vec();
                assert_eq!(got, want, "row-local {strat:?}");
            }
        }
    }

    // Full debayer chain: Bayer u16 -> debayer(Linear) -> luma -> u8 (halo 1).
    {
        let pat = BayerPattern::Rggb;
        let build = || {
            Pipeline::new()
                .debayer(DemosaicMethod::Linear)
                .to_luma()
                .convert(PixelType::U8)
        };
        let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);
        let mut f0 = sample16(w * h);
        let want = build()
            .compile(spec(), Strategy::Sequential)
            .unwrap()
            .run(&bayer_frame(w, h, pat, &mut f0))
            .unwrap()
            .as_raw_u8()
            .to_vec();
        for &(tr, tc) in &tilings {
            for parallel in [false, true] {
                let strat = mk_strat(tr, tc, parallel);
                let mut r = build().compile(spec(), strat).unwrap();
                assert!(r.is_tiled(), "debayer {strat:?}");
                let mut f = sample16(w * h);
                let got = r
                    .run(&bayer_frame(w, h, pat, &mut f))
                    .unwrap()
                    .as_raw_u8()
                    .to_vec();
                assert_eq!(got, want, "debayer {strat:?}");
            }
        }
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

const FILTERS: [ResizeFilter; 3] = [
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
];

/// `resize_to_fit` keeps the aspect ratio, never exceeds either bound, and
/// enlarges an image smaller than the box.
#[test]
fn resize_to_fit_computes_bounded_aspect_dims() {
    let cases = [
        // (src_w, src_h, box_w, box_h, want_w, want_h)
        (100, 40, 50, 50, 50, 20),      // width-bound shrink
        (40, 100, 50, 50, 20, 50),      // height-bound shrink
        (30, 30, 50, 80, 50, 50),       // enlarge to the tighter bound
        (200, 100, 200, 100, 200, 100), // already exact -> unchanged
        (7, 3, 4, 4, 4, 2),             // rounding, min side stays >= 1
    ];
    for (sw, sh, bw, bh, ww, wh) in cases {
        let runner = Pipeline::new()
            .resize_to_fit(bw, bh, ResizeFilter::Bilinear)
            .compile(
                ImageSpec::new(sw, sh, ColorSpace::Gray, PixelType::U8),
                Strategy::Sequential,
            )
            .unwrap();
        let out = runner.output_spec();
        assert_eq!(
            (out.width, out.height),
            (ww, wh),
            "{sw}x{sh} -> box {bw}x{bh}"
        );
        assert!(out.width <= bw && out.height <= bh);
    }
}

/// A box equal to the current size resamples at scale 1.0, which every filter
/// here reduces to the identity — the output is bit-identical to the input.
#[test]
fn resize_to_fit_scale_one_is_identity() {
    let (w, h) = (9, 7);
    for cs in [ColorSpace::Gray, ColorSpace::Rgb] {
        let ch = if cs == ColorSpace::Rgb { 3 } else { 1 };
        for filter in FILTERS {
            let src = sample16(w * h * ch);
            let mut frame = src.clone();
            let img = DynamicImageRef::from(ImageRef::new(&mut frame, w, h, cs.clone()).unwrap());
            let mut runner = Pipeline::new()
                .resize_to_fit(w, h, filter)
                .compile(
                    ImageSpec::new(w, h, cs.clone(), PixelType::U16),
                    Strategy::Sequential,
                )
                .unwrap();
            let out = runner.run(&img).unwrap();
            assert_eq!(
                bytemuck::cast_slice::<u8, u16>(out.as_raw_u8()),
                &src[..],
                "{cs:?} / {filter:?}"
            );
        }
    }
}

/// A flat image stays flat after any resize — weights sum to 1 everywhere,
/// including the clamped edges.
#[test]
fn resize_to_fit_preserves_constant_image() {
    let (w, h) = (37, 29);
    for (bw, bh) in [(20, 16), (80, 62), (37, 29)] {
        for filter in FILTERS {
            let mut frame = vec![1234u16; w * h];
            let img = gray16(w, h, &mut frame);
            let mut runner = Pipeline::new()
                .resize_to_fit(bw, bh, filter)
                .compile(
                    ImageSpec::new(w, h, ColorSpace::Gray, PixelType::U16),
                    Strategy::Sequential,
                )
                .unwrap();
            let out = runner.run(&img).unwrap();
            let px = bytemuck::cast_slice::<u8, u16>(out.as_raw_u8());
            assert!(
                px.iter().all(|&v| v == 1234),
                "box {bw}x{bh} / {filter:?}: {:?}",
                &px[..px.len().min(8)]
            );
        }
    }
}

/// Downscaling an antisymmetric horizontal gradient gives a monotonic,
/// range-safe, antisymmetric result.
#[test]
fn resize_to_fit_downscale_gradient() {
    let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    let mut frame = src.clone();
    let img = DynamicImageRef::from(ImageRef::new(&mut frame, 16, 1, ColorSpace::Gray).unwrap());
    let mut runner = Pipeline::new()
        .resize_to_fit(4, 1, ResizeFilter::Bilinear)
        .compile(
            ImageSpec::new(16, 1, ColorSpace::Gray, PixelType::F32),
            Strategy::Sequential,
        )
        .unwrap();
    let out = runner.run(&img).unwrap();
    let px = bytemuck::cast_slice::<u8, f32>(out.as_raw_u8());
    assert_eq!(px.len(), 4);
    assert!(px.windows(2).all(|w| w[0] < w[1]), "monotonic: {px:?}");
    assert!(px.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!((px[0] + px[3] - 1.0).abs() < 1e-3, "antisymmetric: {px:?}");
}

/// A parallel resample is deterministic: repeated runs of a large, non-trivial
/// downscale are byte-identical (each output element is summed by one task in a
/// fixed order, regardless of the pool).
#[test]
fn resize_to_fit_is_deterministic() {
    let (w, h) = (300usize, 220usize);
    let src = sample16(w * h * 3);
    let build = || {
        Pipeline::new()
            .resize_to_fit(97, 71, ResizeFilter::Lanczos3)
            .compile(
                ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U16),
                Strategy::Sequential,
            )
            .unwrap()
    };
    let mut first = build();
    let mut f0 = src.clone();
    let want = first
        .run(&DynamicImageRef::from(
            ImageRef::new(&mut f0, w, h, ColorSpace::Rgb).unwrap(),
        ))
        .unwrap()
        .as_raw_u8()
        .to_vec();
    for _ in 0..8 {
        let mut r = build();
        let mut f = src.clone();
        let got = r
            .run(&DynamicImageRef::from(
                ImageRef::new(&mut f, w, h, ColorSpace::Rgb).unwrap(),
            ))
            .unwrap()
            .as_raw_u8()
            .to_vec();
        assert_eq!(got, want);
    }
}

/// Resize composes with the other geometric ops in any order: `resize` then
/// `rotate`/`flip`/`crop` runs as one sequential tail, tracking dims correctly.
#[test]
fn resize_to_fit_composes_with_geometry() {
    let (w, h) = (40usize, 24usize);
    let src: Vec<u16> = (0..(w * h) as u16).collect();
    let run = |p: Pipeline| {
        let mut r = p
            .compile(
                ImageSpec::new(w, h, ColorSpace::Gray, PixelType::U16),
                Strategy::Sequential,
            )
            .unwrap();
        let mut f = src.clone();
        let out = r.run(&gray16(w, h, &mut f)).unwrap();
        (out.width(), out.height())
    };

    // resize (40x24 -> fits 20x20 => 20x12) then a quarter turn swaps the axes.
    assert_eq!(
        run(Pipeline::new()
            .resize_to_fit(20, 20, ResizeFilter::Bicubic)
            .rotate_90()),
        (12, 20)
    );
    // ...and resize can also come after geometry.
    assert_eq!(
        run(Pipeline::new().flip_vertical().rotate_90().resize_to_fit(
            10,
            30,
            ResizeFilter::Bilinear
        )),
        (10, 17) // 24x40 -> scale 10/24 -> round(24*.4167)=10, round(40*.4167)=17
    );
    // resize between two geometric ops is fine too.
    assert_eq!(
        run(Pipeline::new()
            .crop(4, 0, 32, 24)
            .resize_to_fit(16, 16, ResizeFilter::Lanczos3)
            .flip_horizontal()),
        (16, 12)
    );
}

/// Resize is rejected on a Bayer image; debayering first makes it legal.
#[test]
fn resize_to_fit_rejects_bayer() {
    let bayer = ImageSpec::new(
        16,
        16,
        ColorSpace::Bayer(BayerPattern::Rggb),
        PixelType::U16,
    );
    let err = Pipeline::new()
        .resize_to_fit(8, 8, ResizeFilter::Bicubic)
        .compile(bayer.clone(), Strategy::Sequential)
        .unwrap_err();
    assert!(matches!(err, PipelineError::ResizeOnBayer));

    assert!(Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .resize_to_fit(8, 8, ResizeFilter::Bicubic)
        .compile(bayer, Strategy::Sequential)
        .is_ok());
}

/// Resize runs as the sequential tail after a tiled pixel-op body.
#[test]
fn resize_to_fit_runs_after_tiled_body() {
    let (w, h) = (64, 48);
    let mut frame = sample16(w * h);
    let img = bayer_frame(w, h, BayerPattern::Rggb, &mut frame);
    let mut runner = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .to_luma()
        .resize_to_fit(20, 20, ResizeFilter::Lanczos3)
        .convert(PixelType::U8)
        .compile(
            ImageSpec::new(w, h, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16),
            Strategy::tiled_parallel(16),
        )
        .unwrap();
    assert!(runner.is_tiled());
    let out = runner.run(&img).unwrap();
    assert_eq!((out.width(), out.height()), (20, 15));
    assert_eq!(out.color_space(), ColorSpace::Gray);
    assert_eq!(out.pixel_type(), PixelType::U8);
}

#[test]
fn resize_op_round_trips_through_serde() {
    let p = Pipeline::new().resize_to_fit(320, 240, ResizeFilter::Lanczos3);
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(p, serde_json::from_str::<Pipeline>(&json).unwrap());
}

/// A geometric op no longer strands the rest of the chain in a single
/// whole-frame pass: the pixel run past it retiles against its own dimensions.
#[test]
fn geometry_splits_the_chain_into_multiple_tiled_passes() {
    let (w, h) = (96usize, 72usize);
    let spec = || ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U16);

    // No geo op: one tiled pass.
    let r = Pipeline::new()
        .to_luma()
        .convert(PixelType::U8)
        .compile(spec(), Strategy::tiled(8))
        .unwrap();
    assert_eq!(r.tiled_pass_count(), 1);

    // resize in the middle: tiles before *and* after.
    let r = Pipeline::new()
        .scale(1.1, 0.0)
        .resize_to_fit(60, 60, ResizeFilter::Bilinear)
        .to_luma()
        .convert(PixelType::U8)
        .compile(spec(), Strategy::tiled(8))
        .unwrap();
    assert_eq!(r.tiled_pass_count(), 2);

    // two geo ops, three pixel runs: three tiled passes.
    let r = Pipeline::new()
        .scale(1.0, 1.0)
        .flip_vertical()
        .to_luma()
        .rotate_180()
        .convert(PixelType::U8)
        .compile(spec(), Strategy::tiled(8))
        .unwrap();
    assert_eq!(r.tiled_pass_count(), 3);

    // Sequential strategy: never tiled.
    let r = Pipeline::new()
        .resize_to_fit(60, 60, ResizeFilter::Bilinear)
        .to_luma()
        .compile(spec(), Strategy::Sequential)
        .unwrap();
    assert_eq!(r.tiled_pass_count(), 0);
}

/// Every segmented tiled plan stays byte-identical to `Sequential`, across
/// serial/parallel and 1-D/2-D tilings, with geo ops (including resize) mid-chain.
#[test]
fn segmented_tiling_matches_sequential() {
    let (w, h) = (64usize, 48usize);
    let pat = BayerPattern::Rggb;
    let spec = || ImageSpec::new(w, h, ColorSpace::Bayer(pat), PixelType::U16);

    let chains: Vec<(&str, Pipeline)> = vec![
        (
            "debayer luma | flip_v | scale convert",
            Pipeline::new()
                .debayer(DemosaicMethod::Linear)
                .to_luma()
                .flip_vertical()
                .scale(1.3, 2.0)
                .convert(PixelType::U8),
        ),
        (
            "debayer luma | resize | scale convert",
            Pipeline::new()
                .debayer(DemosaicMethod::Cubic)
                .to_luma()
                .resize_to_fit(40, 40, ResizeFilter::Lanczos3)
                .scale(0.9, 1.0)
                .convert(PixelType::U8),
        ),
        (
            "debayer | rotate_90 | luma convert | crop",
            Pipeline::new()
                .debayer(DemosaicMethod::Nearest)
                .rotate_90()
                .to_luma()
                .convert(PixelType::U8)
                .crop(2, 3, 30, 40),
        ),
        (
            "crop | debayer luma | resize | convert | rotate_180",
            Pipeline::new()
                .crop(4, 2, 52, 40)
                .debayer(DemosaicMethod::Linear)
                .to_luma()
                .resize_to_fit(24, 24, ResizeFilter::Bicubic)
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
            Strategy::tiled(6),
            Strategy::tiled(13),
            Strategy::tiled_parallel(8),
            Strategy::tiled_2d(10, 20),
            Strategy::tiled_2d_parallel(7, 15),
        ] {
            let mut r = p.clone().compile(spec(), strat).unwrap();
            let got = {
                let mut f = sample16(w * h);
                r.run(&bayer_frame(w, h, pat, &mut f))
                    .unwrap()
                    .as_raw_u8()
                    .to_vec()
            };
            assert_eq!(got, want, "{name} / {strat:?}");
        }
    }
}
