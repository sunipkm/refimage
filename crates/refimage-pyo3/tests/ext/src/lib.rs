//! A downstream extension module, as a library that hands out refimage images would write
//! one: it defines no image class of its own, only functions over refimage-pyo3's boundary
//! types.

use std::time::Duration;

use pyo3::prelude::*;
use refimage::chrono::DateTime;
use refimage::pipeline::Strategy;
use refimage::{BayerPattern, ColorSpace, DynamicImageRef, GenericImageOwned, ImageOwned};
use refimage_pyo3::{RiImage, RiImageRef, RiPipeline, refimage_error, refimage_module};

/// The installed refimage's version, after the compatibility check.
#[pyfunction]
fn check(py: Python<'_>) -> PyResult<String> {
    refimage_module(py)?.getattr("__version__")?.extract()
}

/// A synthetic camera frame made in Rust, and the address of its pixel buffer, so the
/// tests can see that Python ends up using that very buffer.
#[pyfunction]
fn capture(
    py: Python<'_>,
    width: usize,
    height: usize,
    kind: &str,
    frame_id: u32,
) -> PyResult<(RiImage, usize)> {
    let n = width * height;
    let ts = DateTime::from_timestamp(1_700_000_000, 123_000_000).expect("valid timestamp");
    let bad = |e: refimage::ImageError| refimage_error(py, e);
    let (image, ptr): (refimage::DynamicImageOwned, usize) = match kind {
        "u8" => {
            let v: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let p = v.as_ptr() as usize;
            (
                ImageOwned::from_owned(v, width, height, ColorSpace::Gray)
                    .map_err(bad)?
                    .into(),
                p,
            )
        }
        "u12" => {
            let v: Vec<u16> = (0..n).map(|i| (i * 7 % 4096) as u16).collect();
            let p = v.as_ptr() as usize;
            let cs = ColorSpace::Bayer(BayerPattern::Rggb);
            (
                ImageOwned::from_owned(v, width, height, cs)
                    .map_err(bad)?
                    .with_bit_depth(12)
                    .into(),
                p,
            )
        }
        "rgbf32" => {
            let v: Vec<f32> = (0..n * 3).map(|i| (i % 100) as f32 / 100.0).collect();
            let p = v.as_ptr() as usize;
            (
                ImageOwned::from_owned(v, width, height, ColorSpace::Rgb)
                    .map_err(bad)?
                    .into(),
                p,
            )
        }
        other => return Err(refimage_error(py, format!("unknown kind {other:?}"))),
    };
    let mut img = GenericImageOwned::new(ts, Duration::from_millis(250), image);
    img.set_frame_id(frame_id);
    img.insert_key("CAMERA", ("ext", "made in Rust"))
        .map_err(|e| refimage_error(py, e))?;
    img.insert_key("GAIN", 120i64)
        .map_err(|e| refimage_error(py, e))?;
    Ok((RiImage(img), ptr))
}

/// The address of an image's pixels as lent to Rust (no copy).
#[pyfunction]
fn data_ptr(img: RiImageRef<'_>) -> usize {
    img.bytes().as_ptr() as usize
}

/// Invert the pixels in place, within the image's bit depth.
#[pyfunction]
fn invert(mut img: RiImageRef<'_>) -> PyResult<()> {
    let max = (1u32 << img.header().pixel_type.bit_depth().min(16)) - 1;
    match img.image_mut()? {
        DynamicImageRef::U8(mut i) => i.as_mut_slice().iter_mut().for_each(|p| *p = 255 - *p),
        DynamicImageRef::U16(mut i) => i
            .as_mut_slice()
            .iter_mut()
            .for_each(|p| *p = max as u16 - *p),
        DynamicImageRef::F32(mut i) => i.as_mut_slice().iter_mut().for_each(|p| *p = 1.0 - *p),
        _ => {}
    }
    Ok(())
}

/// The sum of the pixels of an owned copy.
#[pyfunction]
fn total(img: RiImage) -> f64 {
    use refimage::PixelData;
    let i = &img.0;
    if let Some(s) = i.as_slice_u8() {
        s.iter().map(|&p| f64::from(p)).sum()
    } else if let Some(s) = i.as_slice_u16() {
        s.iter().map(|&p| f64::from(p)).sum()
    } else {
        i.as_slice_f32()
            .map_or(0.0, |s| s.iter().map(|&p| f64::from(p)).sum())
    }
}

/// An image into Rust (one copy) and back out (no copy).
#[pyfunction]
fn echo(img: RiImage) -> RiImage {
    img
}

/// An image into Rust and back out by reference (one more copy).
#[pyfunction]
fn echo_by_ref<'py>(py: Python<'py>, img: RiImage) -> PyResult<Bound<'py, PyAny>> {
    (&img).into_pyobject(py)
}

/// Apply a pipeline built in Python to an image read in place.
#[pyfunction]
fn process(py: Python<'_>, p: RiPipeline, mut img: RiImageRef<'_>) -> PyResult<RiImage> {
    let out =
        p.0.apply(&img.generic_mut()?)
            .map_err(|e| refimage_error(py, e))?;
    Ok(RiImage(out))
}

/// Compile a pipeline and run it over a lent image as a frame (no copy of the input).
#[pyfunction]
fn run(py: Python<'_>, p: RiPipeline, img: RiImageRef<'_>) -> PyResult<RiImage> {
    let mut runner =
        p.0.compile(img.spec(), Strategy::tiled(0))
            .map_err(|e| refimage_error(py, e))?;
    let out = runner.run(&img).map_err(|e| refimage_error(py, e))?;
    Ok(RiImage(GenericImageOwned::with_metadata(
        img.metadata().clone(),
        &out,
    )))
}

/// A pipeline through Rust, with one more op.
#[pyfunction]
fn flipped(p: RiPipeline) -> RiPipeline {
    RiPipeline(p.0.flip_horizontal())
}

/// Set the frame id and a card on a Python image, from Rust.
#[pyfunction]
fn stamp(py: Python<'_>, mut img: RiImageRef<'_>, frame_id: u32) -> PyResult<()> {
    let mut meta = img.metadata().clone();
    meta.set_frame_id(frame_id);
    meta.insert("STAMPED", true as i64)
        .map_err(|e| refimage_error(py, e))?;
    img.set_metadata(meta)
}

/// Raise refimage's own exception from this module.
#[pyfunction]
fn fail(py: Python<'_>) -> PyResult<()> {
    Err(refimage_error(py, "raised by another extension"))
}

#[pymodule]
fn refimage_pyo3_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(capture, m)?)?;
    m.add_function(wrap_pyfunction!(data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(total, m)?)?;
    m.add_function(wrap_pyfunction!(echo, m)?)?;
    m.add_function(wrap_pyfunction!(echo_by_ref, m)?)?;
    m.add_function(wrap_pyfunction!(process, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(flipped, m)?)?;
    m.add_function(wrap_pyfunction!(stamp, m)?)?;
    m.add_function(wrap_pyfunction!(fail, m)?)?;
    Ok(())
}
