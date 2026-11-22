//! Resize a real photo — down to a thumbnail, and a small crop up to 8x — with
//! each [`ResizeFilter`], through a compiled [`Runner`].
//!
//! Run with:
//! `cargo run --release --example resize_filters --features image [-- OUT_DIR]`
//!
//! Reads `examples/assets/chelsea.png` (CC0; see that folder's README) and writes
//! `source.png`, `down_<filter>.png` and `zoom_<filter>.png` to `OUT_DIR`
//! (default: the current directory).

use std::path::PathBuf;

use image::DynamicImage;
use refimage::pipeline::{ImageSpec, Pipeline, ResizeFilter, Runner, Strategy};
use refimage::{ColorSpace, DynamicImageRef, ImageRef, PixelType};

/// Load the bundled test photo as interleaved RGB8 + its dimensions.
fn load_source() -> (Vec<u8>, usize, usize) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/chelsea.png");
    let rgb = image::open(path).expect("open chelsea.png").to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    (rgb.into_raw(), w, h)
}

/// Compile `pipeline` for a `w`x`h` RGB8 frame and run it once over `rgb`.
fn run(pipeline: Pipeline, rgb: &[u8], w: usize, h: usize) -> (DynamicImage, Runner) {
    let mut buf = rgb.to_vec();
    let frame =
        DynamicImageRef::from(ImageRef::new(&mut buf, w, h, ColorSpace::Rgb).expect("frame"));
    let mut runner = pipeline
        .compile(
            ImageSpec::new(w, h, ColorSpace::Rgb, PixelType::U8),
            Strategy::tiled_parallel(64),
        )
        .expect("compile");
    let out = runner.run(&frame).expect("run");
    (DynamicImage::try_from(out).expect("to image"), runner)
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    std::fs::create_dir_all(&dir).unwrap();

    let (src, w, h) = load_source();
    println!("source: {w}x{h}");
    {
        let mut s = src.clone();
        let img = DynamicImageRef::from(ImageRef::new(&mut s, w, h, ColorSpace::Rgb).unwrap());
        DynamicImage::try_from(img)
            .unwrap()
            .save(dir.join("source.png"))
            .unwrap();
    }

    for (name, filter) in [
        ("bilinear", ResizeFilter::Bilinear),
        ("bicubic", ResizeFilter::Bicubic),
        ("lanczos3", ResizeFilter::Lanczos3),
    ] {
        // Whole-image thumbnail — a ~4x downscale (stresses anti-aliasing).
        let (down, _) = run(Pipeline::new().resize_to_fit(110, 110, filter), &src, w, h);
        down.save(dir.join(format!("down_{name}.png"))).unwrap();

        // Whole-image ~2.7x upscale (stretch).
        let (up, _) = run(
            Pipeline::new().resize_to_fit(1200, 1200, filter),
            &src,
            w,
            h,
        );
        up.save(dir.join(format!("up_{name}.png"))).unwrap();

        // A 60x45 patch around the cat's eye, blown up 8x (stresses interpolation
        // — ringing, blocking, sharpness). `crop` then `resize` in one Runner;
        // the leading crop folds into the frame read.
        let (zoom, r) = run(
            Pipeline::new()
                .crop(150, 82, 60, 45)
                .resize_to_fit(480, 480, filter),
            &src,
            w,
            h,
        );
        zoom.save(dir.join(format!("zoom_{name}.png"))).unwrap();

        println!(
            "{name:>9}: down {}x{} | zoom {}x{} | zoom runner: tiled={}, passes={}",
            down.width(),
            down.height(),
            zoom.width(),
            zoom.height(),
            r.is_tiled(),
            r.tiled_pass_count(),
        );
    }
    println!("wrote PNGs to {}", dir.display());
}
