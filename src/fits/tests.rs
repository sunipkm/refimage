//! Unit tests for the FITS writer. Structural checks only — `astropy` round-trips live
//! in `tests/fits_astropy.rs`, cfitsio cross-checks in `tests/fits_roundtrip.rs`.

use std::time::Duration;

use chrono::{DateTime, Utc};

use super::*;
use crate::{ColorSpace, DynamicImageOwned, GenericImageOwned, ImageOwned};

/// A fixed, deterministic UTC timestamp for tests.
fn ts() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).unwrap()
}

fn gray_u16(w: usize, h: usize) -> GenericImageOwned {
    let data: Vec<u16> = (0..w * h).map(|i| (i as u16).wrapping_mul(97)).collect();
    let img =
        DynamicImageOwned::from(ImageOwned::from_owned(data, w, h, ColorSpace::Gray).unwrap());
    let mut g = GenericImageOwned::new(ts(), Duration::from_millis(1500), img);
    g.insert_key("CAMERA", "Test Cam").unwrap();
    g.insert_key("GAIN", 3u16).unwrap();
    g
}

fn is_block_aligned(b: &[u8]) -> bool {
    b.len().is_multiple_of(2880)
}

fn find_card<'a>(bytes: &'a [u8], key: &str) -> Option<&'a str> {
    // Search only header blocks (stop at first block with no '=' cards is overkill;
    // just scan everything in 80-byte steps until END).
    for card in bytes.chunks(80) {
        if card.len() < 80 {
            break;
        }
        let Ok(text) = std::str::from_utf8(card) else {
            continue;
        };
        if text.starts_with(&format!("{key:<8}=")) || text.starts_with(&format!("HIERARCH {key} ="))
        {
            return Some(text);
        }
    }
    None
}

#[test]
fn uncompressed_primary_structure() {
    let g = gray_u16(8, 4);
    let bytes = g.fits_bytes(FitsCompression::None).unwrap();
    assert!(is_block_aligned(&bytes));
    assert!(bytes.starts_with(b"SIMPLE  =                    T"));
    assert!(find_card(&bytes, "BITPIX").unwrap().contains("16"));
    assert!(find_card(&bytes, "NAXIS1").unwrap().contains("8"));
    assert!(find_card(&bytes, "NAXIS2").unwrap().contains("4"));
    assert!(find_card(&bytes, "BZERO").unwrap().contains("32768"));
    assert!(find_card(&bytes, "DATE-OBS").is_some());
    assert!(find_card(&bytes, "COLORSPC").unwrap().contains("GRAY"));
    // Duration split into _S / _NS plus a float base card.
    assert!(find_card(&bytes, "EXPOSURE_S").unwrap().contains("1"));
    assert!(find_card(&bytes, "EXPOSURE_NS")
        .unwrap()
        .contains("500000000"));
    // header + 8*4*2 = 64 data bytes -> 2 blocks
    assert_eq!(bytes.len(), 2880 * 2);
}

#[test]
fn compressed_is_bintable_extension() {
    let g = gray_u16(16, 16);
    for comp in [FitsCompression::Gzip, FitsCompression::Rice] {
        let bytes = g.fits_bytes(comp).unwrap();
        assert!(is_block_aligned(&bytes), "{comp:?}");
        // primary first
        assert!(bytes.starts_with(b"SIMPLE  =                    T"));
        assert!(find_card(&bytes, "XTENSION").unwrap().contains("BINTABLE"));
        assert!(find_card(&bytes, "ZIMAGE").unwrap().contains('T'));
        let zc = find_card(&bytes, "ZCMPTYPE").unwrap();
        assert!(zc.contains(if comp == FitsCompression::Rice {
            "RICE_1"
        } else {
            "GZIP_1"
        }));
        assert!(find_card(&bytes, "ZNAXIS1").unwrap().contains("16"));
    }
}

#[test]
fn hcompress_is_bintable_with_scale_cards() {
    let g = gray_u16(20, 16);
    let bytes = g.fits_bytes(FitsCompression::Hcompress).unwrap();
    assert!(is_block_aligned(&bytes));
    assert!(bytes.starts_with(b"SIMPLE  =                    T"));
    assert!(find_card(&bytes, "ZCMPTYPE")
        .unwrap()
        .contains("HCOMPRESS_1"));
    assert!(find_card(&bytes, "ZNAME1").unwrap().contains("SCALE"));
    assert!(find_card(&bytes, "ZNAME2").unwrap().contains("SMOOTH"));
    // Whole image is one tile: ZTILE1 == ZNAXIS1.
    assert!(find_card(&bytes, "ZTILE1").unwrap().contains("20"));
    assert!(find_card(&bytes, "ZTILE2").unwrap().contains("16"));
}

#[test]
fn hcompress_rejects_tiny_images() {
    let data = vec![0u16; 9];
    let img =
        DynamicImageOwned::from(ImageOwned::from_owned(data, 3, 3, ColorSpace::Gray).unwrap());
    let g = GenericImageOwned::new(ts(), Duration::ZERO, img);
    assert!(matches!(
        g.fits_bytes(FitsCompression::Hcompress),
        Err(FitsError::HcompressTooSmall)
    ));
}

#[test]
fn float_compresses_both_ways() {
    let data: Vec<f32> = (0..64 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
    let img =
        DynamicImageOwned::from(ImageOwned::from_owned(data, 64, 8, ColorSpace::Gray).unwrap());
    let g = GenericImageOwned::new(ts(), Duration::ZERO, img);

    // Gzip: lossless raw float bytes, no quantization keywords.
    let gz = g.fits_bytes(FitsCompression::Gzip).unwrap();
    assert!(find_card(&gz, "ZQUANTIZ").is_none());

    // Rice: quantized, so ZQUANTIZ / ZDITHER0 / ZSCALE column appear.
    let rc = g.fits_bytes(FitsCompression::Rice).unwrap();
    assert!(find_card(&rc, "ZQUANTIZ")
        .unwrap()
        .contains("SUBTRACTIVE_DITHER_1"));
    assert!(find_card(&rc, "ZDITHER0").is_some());
    assert!(find_card(&rc, "TTYPE2").unwrap().contains("ZSCALE"));
    assert!(find_card(&rc, "ZBITPIX").unwrap().contains("-32"));
}

#[test]
fn rgb_is_planar_cube() {
    let data: Vec<u8> = (0..3 * 4 * 5).map(|i| i as u8).collect();
    let img = DynamicImageOwned::from(ImageOwned::from_owned(data, 5, 4, ColorSpace::Rgb).unwrap());
    let g = GenericImageOwned::new(ts(), Duration::ZERO, img);
    let bytes = g.fits_bytes(FitsCompression::None).unwrap();
    assert!(find_card(&bytes, "NAXIS").unwrap().contains('3'));
    assert!(find_card(&bytes, "NAXIS3").unwrap().contains('3'));
}

#[test]
fn reserved_metadata_key_errors() {
    let data = vec![0u8; 4];
    let img =
        DynamicImageOwned::from(ImageOwned::from_owned(data, 2, 2, ColorSpace::Gray).unwrap());
    let mut g = GenericImageOwned::new(ts(), Duration::ZERO, img);
    g.insert_key("NAXIS1", 5u16).unwrap();
    assert!(matches!(
        g.fits_bytes(FitsCompression::None),
        Err(FitsError::ReservedKeyword(_))
    ));
}

#[test]
fn multi_hdu_file() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("refimage_fits_test_{}.fits", std::process::id()));
    let g = gray_u16(8, 8);
    {
        let mut w = create_fits(&path, FitsCompression::Rice, true).unwrap();
        g.append_fits(&mut w).unwrap();
        g.append_fits(&mut w).unwrap();
        w.finish().unwrap();
    }
    let bytes = std::fs::read(&path).unwrap();
    assert!(is_block_aligned(&bytes));
    let n_xtension = bytes
        .chunks(80)
        .filter(|c| c.starts_with(b"XTENSION= 'BINTABLE'"))
        .count();
    assert_eq!(n_xtension, 2);
    std::fs::remove_file(&path).ok();
}

#[test]
fn in_memory_multi_hdu_file() {
    let g = gray_u16(8, 8);

    // Owned Vec sink, recovered from finish().
    let mut w = create_fits_to(Vec::new(), FitsCompression::Rice).unwrap();
    g.append_fits(&mut w).unwrap();
    g.append_fits(&mut w).unwrap();
    assert_eq!(w.hdu_count(), 3);
    let bytes = w.finish().unwrap();

    assert!(is_block_aligned(&bytes));
    assert!(bytes.starts_with(b"SIMPLE  =                    T"));
    let n_ext = bytes
        .chunks(80)
        .filter(|c| c.starts_with(b"XTENSION= 'BINTABLE'"))
        .count();
    assert_eq!(n_ext, 2);

    // Borrowed Vec sink: caller keeps ownership, same bytes.
    let mut buf = Vec::new();
    {
        let mut w = create_fits_to(&mut buf, FitsCompression::Rice).unwrap();
        g.append_fits(&mut w).unwrap();
        g.append_fits(&mut w).unwrap();
        w.finish().unwrap();
    }
    assert_eq!(buf, bytes);
}

#[test]
fn bincode_roundtrip_still_works() {
    // Part A regression: numeric-collapsed GenericValue serialises.
    let g = gray_u16(4, 4);
    let ser = bincode::serialize(&g).unwrap();
    let de: GenericImageOwned = bincode::deserialize(&ser).unwrap();
    assert_eq!(g.get_metadata(), de.get_metadata());
}
