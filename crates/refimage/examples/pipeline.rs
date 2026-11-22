//! Build a processing recipe once, then run it over many frames with no
//! per-frame heap allocation.
//!
//! Run with: `cargo run --release --example pipeline`

use std::time::Instant;

use refimage::pipeline::{ImageSpec, Pipeline, Strategy};
use refimage::{
    BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, ImageProps, ImageRef, PixelData,
    PixelType,
};

fn main() {
    let (w, h) = (1024usize, 768usize);
    let pattern = BayerPattern::Rggb;

    // A declarative, cloneable, serializable recipe: raw Bayer -> RGB ->
    // gray -> gain/offset -> 8-bit -> flipped vertically.
    let recipe = Pipeline::new()
        .debayer(DemosaicMethod::Linear)
        .to_luma()
        .scale(1.15, 2.0)
        .convert(PixelType::U8)
        .flip_vertical();

    // It round-trips through JSON, so a recipe can live in a config file.
    let json = serde_json::to_string_pretty(&recipe).unwrap();
    println!("recipe:\n{json}\n");

    // Compile against the concrete frame format. This validates the whole
    // chain and pre-allocates every buffer the runner will ever need.
    let spec = ImageSpec::new(w, h, ColorSpace::Bayer(pattern), PixelType::U16);
    let mut runner = recipe
        .compile(spec, Strategy::tiled_parallel(64))
        .expect("valid chain");

    println!(
        "compiled: tiled={}, scratch={} KiB, output {:?} {}x{}",
        runner.is_tiled(),
        runner.scratch_bytes() / 1024,
        runner.output_spec().cspace,
        runner.output_spec().width,
        runner.output_spec().height,
    );

    // A fake acquisition loop. The backing buffer is reused; so is every
    // buffer inside the runner.
    let mut frame: Vec<u16> = (0..w * h).map(|i| ((i * 7 + 3) % 4093) as u16).collect();

    let n = 200;
    let start = Instant::now();
    for _ in 0..n {
        let img = DynamicImageRef::from(
            ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pattern)).unwrap(),
        );
        let out = runner.run(&img).expect("run");
        assert_eq!(out.color_space(), ColorSpace::Gray);
        assert_eq!(out.pixel_type(), PixelType::U8);
        std::hint::black_box(out.as_raw_u8()[0]);
    }
    let per = start.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("ran {n} frames, {per:.3} ms/frame");

    // One-shot ROI extraction: `Pipeline::apply` compiles, runs once, and hands
    // back an owned image. `Op::Roi` zero-fills any overhang past the edge.
    let img =
        DynamicImageRef::from(ImageRef::new(&mut frame, w, h, ColorSpace::Bayer(pattern)).unwrap());
    let thumb = Pipeline::new()
        .roi(16, 16, 128, 128)
        .debayer(DemosaicMethod::Linear)
        .to_luma()
        .convert(PixelType::U8)
        .apply(&img)
        .expect("roi");
    println!("thumb: {}x{}", thumb.width(), thumb.height());
}
