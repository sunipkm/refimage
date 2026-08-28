//! Cross-check: files written by the pure-Rust writer are read back correctly by
//! cfitsio (via the `fitsio` crate). A second, Python-free reader next to
//! `tests/fits_astropy.rs`.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Duration;

use chrono::DateTime;
use fitsio::hdu::HduInfo;
use fitsio::FitsFile;
use refimage::{
    ColorSpace, DynamicImageOwned, FitsCompression, FitsWrite, GenericImageOwned, ImageOwned,
    ImageProps,
};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("refimg_xchk_{}_{name}.fits", std::process::id()))
}

/// First HDU that holds a non-empty image (primary for uncompressed, extension 1 for
/// compressed — cfitsio decompresses transparently).
fn image_hdu(f: &mut FitsFile) -> fitsio::hdu::FitsHdu {
    for i in 0.. {
        let Ok(hdu) = f.hdu(i) else { break };
        if let HduInfo::ImageInfo { ref shape, .. } = hdu.info {
            if !shape.is_empty() {
                return hdu;
            }
        }
    }
    panic!("no image HDU");
}

fn wrap(img: DynamicImageOwned) -> GenericImageOwned {
    GenericImageOwned::new(
        DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
        Duration::from_millis(500),
        img,
    )
}

#[test]
fn cfitsio_reads_gray_u16() {
    let src: Vec<u16> = (0..32 * 24).map(|i| (i * 613 % 64000) as u16).collect();
    let g = wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(src.clone(), 32, 24, ColorSpace::Gray).unwrap(),
    ));

    for comp in [
        FitsCompression::None,
        FitsCompression::Gzip,
        FitsCompression::Rice,
        FitsCompression::Hcompress,
    ] {
        let path = tmp(&format!("u16_{comp:?}"));
        g.write_fits(&path, comp, true).unwrap();

        let mut f = FitsFile::open(&path).unwrap();
        let hdu = image_hdu(&mut f);
        let data: Vec<u16> = hdu.read_image(&mut f).unwrap();
        assert_eq!(data, src, "{comp:?}");

        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn cfitsio_reads_rgb_u8_planar() {
    // interleaved source
    let src: Vec<u8> = (0..3 * 20 * 16).map(|i| (i * 37 % 255) as u8).collect();
    let g = wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(src.clone(), 20, 16, ColorSpace::Rgb).unwrap(),
    ));

    for comp in [FitsCompression::None, FitsCompression::Gzip] {
        let path = tmp(&format!("rgb_{comp:?}"));
        g.write_fits(&path, comp, true).unwrap();

        let mut f = FitsFile::open(&path).unwrap();
        let hdu = image_hdu(&mut f);
        let planar: Vec<u8> = hdu.read_image(&mut f).unwrap();

        // planar [c][y][x] -> interleaved [y][x][c]
        let (w, h, ch) = (20usize, 16usize, 3usize);
        let mut interleaved = vec![0u8; w * h * ch];
        for c in 0..ch {
            for p in 0..w * h {
                interleaved[p * ch + c] = planar[c * w * h + p];
            }
        }
        assert_eq!(interleaved, src, "{comp:?}");

        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn cfitsio_sees_headers() {
    let g = wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(vec![1u16; 12], 4, 3, ColorSpace::Gray).unwrap(),
    ));
    let path = tmp("hdr");
    g.write_fits(&path, FitsCompression::None, true).unwrap();

    let mut f = FitsFile::open(&path).unwrap();
    let hdu = f.primary_hdu().unwrap();
    let naxis1: i64 = hdu.read_key(&mut f, "NAXIS1").unwrap();
    let naxis2: i64 = hdu.read_key(&mut f, "NAXIS2").unwrap();
    assert_eq!((naxis1, naxis2), (4, 3));
    let _ = g.get_image().width();

    std::fs::remove_file(&path).ok();
}

#[test]
fn cfitsio_reads_hcompressed_u8_and_noise() {
    // High-entropy data pushes the quadtree into its direct-bitmap fallback path.
    let mut rng = 0x2545_f491u32;
    let src: Vec<u8> = (0..48 * 40)
        .map(|_| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 24) as u8
        })
        .collect();
    let g = wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(src.clone(), 48, 40, ColorSpace::Gray).unwrap(),
    ));

    let path = tmp("hcomp_u8");
    g.write_fits(&path, FitsCompression::Hcompress, true)
        .unwrap();

    let mut f = FitsFile::open(&path).unwrap();
    let hdu = image_hdu(&mut f);
    let back: Vec<u8> = hdu.read_image(&mut f).unwrap();
    assert_eq!(back, src, "HCOMPRESS scale=0 must be lossless for u8");

    std::fs::remove_file(&path).ok();
}

#[test]
fn cfitsio_reads_quantized_float_rice() {
    // Gradient + small deterministic noise so quantization is meaningful.
    let mut rng = 0x9e37_79b9u32;
    let src: Vec<f32> = (0..40 * 30)
        .map(|i| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (i as f32 * 0.05).cos() * 500.0 + 800.0 + ((rng >> 29) as f32 - 3.5)
        })
        .collect();
    let g = wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(src.clone(), 40, 30, ColorSpace::Gray).unwrap(),
    ));

    let path = tmp("f32_rice");
    g.write_fits(&path, FitsCompression::Rice, true).unwrap();

    let mut f = FitsFile::open(&path).unwrap();
    let hdu = image_hdu(&mut f);
    let back: Vec<f32> = hdu.read_image(&mut f).unwrap();

    assert_eq!(back.len(), src.len());
    let max_err = back
        .iter()
        .zip(&src)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 2.0,
        "cfitsio dequantized with max error {max_err}"
    );

    std::fs::remove_file(&path).ok();
}
