//! Reusable, allocation-aware processing pipelines for pixel
//! conversion in this crate (debayer, luminance, pixel-type conversion, affine
//! pixel scaling, crop, ROI, flips, 90° rotations, aspect-preserving resize).
//!
//! A [`Pipeline`] is a declarative, cloneable, serializable list of [`Op`]s.
//! [`Pipeline::compile`] pairs it with a concrete [`ImageSpec`], validates the
//! whole chain, and allocates every buffer it needs, producing a
//! [`Runner`]. [`Runner::run`] then executes the chain against successive frames,
//! reusing those buffers instead of allocating per frame.
//!
//! For a one-off conversion, [`Pipeline::apply`] compiles with
//! [`Strategy::Sequential`], runs once, and returns an owned image. It accepts any
//! image type ([`ApplyInput`]): a [`DynamicImageRef`] or [`DynamicImageOwned`] gives
//! back a [`DynamicImageOwned`]; a [`GenericImageRef`] or [`GenericImageOwned`] gives
//! back a [`GenericImageOwned`] with its metadata carried through unchanged.
//! [`Runner::run_into`] renders into a caller-supplied [`DynamicImageOwned`]
//! (allocation-free when it is already the output shape) — with a leading
//! [`Op::Roi`] that is a "blit region into a pre-sized buffer".
//!
//! ```
//! use refimage::{
//!     BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, ImageRef, PixelType,
//! };
//! use refimage::pipeline::{ImageSpec, Pipeline, Strategy};
//!
//! let spec = ImageSpec::new(64, 64, ColorSpace::Bayer(BayerPattern::Rggb), PixelType::U16);
//! let mut runner = Pipeline::new()
//!     .debayer(DemosaicMethod::Linear)
//!     .to_luma()
//!     .convert(PixelType::U8)
//!     .compile(spec, Strategy::tiled_parallel(16))
//!     .expect("valid chain");
//!
//! for _ in 0..3 {
//!     let mut frame = vec![0u16; 64 * 64];
//!     let img = DynamicImageRef::from(
//!         ImageRef::new(&mut frame, 64, 64, ColorSpace::Bayer(BayerPattern::Rggb)).unwrap(),
//!     );
//!     let luma = runner.run(&img).expect("run");
//!     assert_eq!(luma.color_space(), ColorSpace::Gray);
//! }
//! ```
//!
//! # Execution strategies
//! - [`Strategy::Sequential`] — one ping-pong buffer pair, every stage sweeps the
//!   whole frame.
//! - [`Strategy::Tiled`] — the output is cut into tiles: horizontal row bands
//!   (the unit of parallelism) optionally subdivided into column tiles (to bound
//!   the working-set width). Each tile pulls a slightly larger, even-snapped input
//!   region — an automatic *halo* sized from the demosaic kernels — and runs the
//!   whole chain fused with a **serial** demosaic kernel, so a tile stays hot in
//!   cache and there is no nested parallelism. Tiling silently falls back to
//!   `Sequential` when the frame is too small to be tiled.
//!
//! # Allocation
//! A serial-tiled [`Runner`] (`Strategy::tiled`) does **zero** heap allocation per
//! frame once compiled — including a chain with geometric segments, which reuse a
//! pair of full-frame buffers already sized for the largest intermediate.
//! Parallel-tiled allocates one scratch set per worker thread on first use
//! (warm-up only). `Sequential` uses the internally-parallel demosaic kernel,
//! which allocates a padded frame copy per call; [`Op::ResizeToFit`] allocates a
//! small strip per output-row band as it resamples. The two
//! tile-scratch buffers are sized independently by a liveness walk over the
//! chain, so a `raw → debayer → luma → u8` pipeline does not pay for the fat RGB
//! intermediate twice. Luma, scale, and pixel-type conversion all rewrite their
//! buffer in place; only debayer swaps.
//!
//! # Adapting to a new frame shape
//! By default [`Runner::run`] rejects a frame whose shape differs from the
//! compiled [`ImageSpec`]. [`Runner::recompile`] rebuilds the plan and buffers for
//! a new shape (this allocates). With the `grow` feature, `run` calls `recompile`
//! automatically on a shape change.
//!
//! # Geometric ops and tiling
//! [`Op::Crop`], [`Op::Roi`], the flips, the rotations, and [`Op::ResizeToFit`]
//! relocate (or resample) pixels, so a tile can't carry the context they need.
//! The chain is split into **segments** at each such op:
//! - **Leading crops** fold into the frame read — `crop → debayer → …` tiles the
//!   pixel run at the cropped size.
//! - Each geometric op runs as one whole-frame pass between segments — cheap
//!   index copies for the flips, rotations, crop and ROI; a row-parallel
//!   resample for [`Op::ResizeToFit`].
//! - Every pixel-op run between geometric ops is retiled against its own
//!   (post-transform) dimensions and the compiled [`Strategy`], so `debayer luma
//!   → resize → scale convert` tiles on *both* sides of the resize.
//!   [`Runner::tiled_pass_count`] reports how many such passes a plan has.
//!
//! # Limitations
//! `Op` covers debayer / luminance / affine pixel scale / pixel-type conversion
//! / crop / ROI / flip / 90° rotation / aspect-preserving resize. It does not allow
//! arbitrary-angle rotation and does not support exact-size (aspect-changing) resampling.
//!
//! # Source layout
//! - `spec` — [`ImageSpec`], the static shape/type description.
//! - `op` — [`Op`] and its compile-time shape inference.
//! - `strategy` — [`Strategy`].
//! - `builder` — [`Pipeline`]: the op list, `compile`, and `apply`.
//! - `optimize` — [`Pipeline::optimize`]: output-preserving peephole cleanup of
//!   the op list (drop no-ops, fold flip/rotation runs, merge nested crops).
//! - `apply` — the [`Frame`] / [`ApplyInput`] input traits.
//! - `plan` — lowering to steps, the buffer-liveness walk, and the tiling math.
//! - `runner` — [`Runner`]: the compiled pipeline plus its scratch.
//! - `exec` — `run_chain` / `fill_tiled`, the actual per-frame work.
#[allow(unused_imports)]
use crate::{DynamicImageOwned, DynamicImageRef, GenericImageOwned, GenericImageRef};

mod apply;
mod builder;
mod error;
mod exec;
mod geom;
mod kernels;
mod operations;
mod optimize;
mod plan;
mod resample;
mod runner;
mod spec;
mod strategy;

pub use apply::{ApplyInput, Frame};
pub use builder::Pipeline;
pub use error::PipelineError;
pub use operations::{Op, ScaleFactor};
pub use resample::ResizeFilter;
pub use runner::Runner;
pub use spec::ImageSpec;
pub use strategy::Strategy;

#[cfg(test)]
mod tests;
