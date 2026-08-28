//! Cross-check: files written by the pure-Rust writer are read back correctly by
//! cfitsio (via the `fitsio` crate). A second, Python-free reader next to
//! `tests/fits_astropy.rs`.

#![cfg(not(target_arch = "wasm32"))]

use std::time::{Duration, UNIX_EPOCH};

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
    GenericImageOwned::new(UNIX_EPOCH + Duration::from_secs(1_700_000_000), img)
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
