//! End-to-end check that `astropy.io.fits` reads what the pure-Rust writer produces.
//!
//! Skipped (not failed) when `python3` or `astropy` is unavailable.

#![cfg(not(target_arch = "wasm32"))]

use std::io::Write;
use std::process::Command;
use std::time::{Duration, UNIX_EPOCH};

use refimage::{
    ColorSpace, DynamicImageOwned, FitsCompression, FitsWrite, GenericImageOwned, ImageOwned,
};

fn astropy_available() -> bool {
    Command::new("python3")
        .args(["-c", "import astropy.io.fits"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Python: open `fits_path`, undo the planar transpose for RGB, compare raw bytes to
/// `raw_path`. Exits 0 on match, 1 (with a message) otherwise.
const CHECK: &str = r#"
import sys, numpy as np
from astropy.io import fits

fits_path, raw_path, dtype, channels = sys.argv[1:5]
channels = int(channels)
want = np.fromfile(raw_path, dtype=dtype)

with fits.open(fits_path) as hdul:
    hdu = next(h for h in hdul if h.data is not None)
    data = hdu.data

if channels > 1:
    # astropy gives (channels, h, w); refimage raw is interleaved (h, w, channels)
    data = np.moveaxis(data, 0, -1)

got = np.ascontiguousarray(data).astype(dtype).ravel()

if got.shape != want.shape or not np.array_equal(got, want):
    print(f"MISMATCH dtype={dtype} ch={channels}: got {got[:8]} want {want[:8]} "
          f"(shapes {got.shape} vs {want.shape})")
    sys.exit(1)
"#;

fn roundtrip(g: &GenericImageOwned, dtype: &str, channels: usize, comp: FitsCompression) {
    let tmp = std::env::temp_dir();
    let tag = format!("{}_{:?}_{}", std::process::id(), comp, dtype);
    let fits_path = tmp.join(format!("refimg_{tag}.fits"));
    let raw_path = tmp.join(format!("refimg_{tag}.raw"));

    g.write_fits(&fits_path, comp, true).expect("write_fits");
    std::fs::File::create(&raw_path)
        .unwrap()
        .write_all(g.get_image().as_raw_u8())
        .unwrap();

    let out = Command::new("python3")
        .arg("-c")
        .arg(CHECK)
        .arg(&fits_path)
        .arg(&raw_path)
        .arg(dtype)
        .arg(channels.to_string())
        .output()
        .expect("run python");

    std::fs::remove_file(&fits_path).ok();
    std::fs::remove_file(&raw_path).ok();

    assert!(
        out.status.success(),
        "{comp:?} {dtype}: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

fn gray_u8() -> GenericImageOwned {
    let d: Vec<u8> = (0..64 * 48).map(|i| (i * 7 % 251) as u8).collect();
    wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(d, 64, 48, ColorSpace::Gray).unwrap(),
    ))
}

fn gray_u16() -> GenericImageOwned {
    let d: Vec<u16> = (0..64 * 48).map(|i| (i * 613 % 65000) as u16).collect();
    wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(d, 64, 48, ColorSpace::Gray).unwrap(),
    ))
}

fn gray_f32() -> GenericImageOwned {
    let d: Vec<f32> = (0..64 * 48).map(|i| (i as f32).sin().abs()).collect();
    wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(d, 64, 48, ColorSpace::Gray).unwrap(),
    ))
}

fn rgb_u8() -> GenericImageOwned {
    let d: Vec<u8> = (0..3 * 40 * 30).map(|i| (i * 11 % 255) as u8).collect();
    wrap(DynamicImageOwned::from(
        ImageOwned::from_owned(d, 40, 30, ColorSpace::Rgb).unwrap(),
    ))
}

fn wrap(img: DynamicImageOwned) -> GenericImageOwned {
    let mut g = GenericImageOwned::new(UNIX_EPOCH + Duration::from_secs(1_700_000_000), img);
    g.insert_key("CAMERA", "refimage self-test").unwrap();
    g.insert_key("GAIN", 5u16).unwrap();
    g.insert_key("EXPOSURE", Duration::from_micros(1_234_567))
        .unwrap();
    g
}

#[test]
fn astropy_reads_every_combination() {
    if !astropy_available() {
        eprintln!("skipping: python3 / astropy not available");
        return;
    }

    for comp in [
        FitsCompression::None,
        FitsCompression::Gzip,
        FitsCompression::Rice,
    ] {
        roundtrip(&gray_u8(), "<u1", 1, comp);
        roundtrip(&gray_u16(), "<u2", 1, comp);
        roundtrip(&rgb_u8(), "<u1", 3, comp);
        if comp != FitsCompression::Rice {
            roundtrip(&gray_f32(), "<f4", 1, comp);
        }
    }
}

#[test]
fn astropy_reads_metadata_cards() {
    if !astropy_available() {
        eprintln!("skipping: python3 / astropy not available");
        return;
    }
    let tmp = std::env::temp_dir();
    let path = tmp.join(format!("refimg_meta_{}.fits", std::process::id()));
    gray_u16()
        .write_fits(&path, FitsCompression::None, true)
        .unwrap();

    let script = r#"
import sys
from astropy.io import fits
h = fits.open(sys.argv[1])[0].header
assert h['CAMERA'] == 'refimage self-test', h['CAMERA']
assert h['GAIN'] == 5, h['GAIN']
assert abs(h['EXPOSURE'] - 1.234567) < 1e-9, h['EXPOSURE']
assert h['EXPOSURE_S'] == 1 and h['EXPOSURE_NS'] == 234567000
assert h['DATE-OBS'].startswith('2023-11-14T22:13:20'), h['DATE-OBS']
assert h['COLORSPC'] == 'GRAY'
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(&path).ok();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
