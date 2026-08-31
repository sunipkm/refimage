//! Reusable, allocation-aware processing pipelines — the one path for every pixel
//! conversion in this crate (debayer, luminance, pixel-type conversion, affine
//! pixel scaling, crop, ROI, flips, 90° rotations, aspect-preserving resize).
//!
//! A [`Pipeline`] is a declarative, cloneable, serializable list of [`Op`]s.
//! [`Pipeline::compile`] pairs it with a concrete [`ImageSpec`], validates the
//! whole chain, and allocates every buffer it will ever need, producing a
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
//!   cache and there is no nested parallelism. The result is **bit-identical** to
//!   `Sequential`. Tiling silently falls back to `Sequential` when the frame is
//!   too small to benefit.
//!
//! # Allocation
//! A serial-tiled [`Runner`] (`Strategy::tiled`) does **zero** heap allocation per
//! frame once compiled. Parallel-tiled allocates one scratch set per worker
//! thread on first use (warm-up only). `Sequential` uses the internally-parallel
//! demosaic kernel, which allocates a padded frame copy per call; a trailing
//! [`Op::ResizeToFit`] likewise allocates one scratch plane and resamples its
//! rows in parallel. The two
//! ping-pong buffers are sized independently by a liveness walk over the chain,
//! so a `raw → debayer → luma → u8` pipeline does not pay for the fat RGB
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
//! Tiling handles them at the chain's edges:
//! - **Leading crops** fold into the frame read — `crop → debayer → …` still
//!   tiles the pixel run, at the cropped size.
//! - Everything from the **first non-leading geometric op** onward runs as one
//!   whole-frame pass after the tiled body — cheap index copies for the flips,
//!   rotations, crop and ROI; a row-parallel resample for [`Op::ResizeToFit`].
//!
//! A geometric op sandwiched between pixel ops (`debayer → crop → luma`) just
//! means the tiled body stops at the crop and the rest is that sequential tail;
//! reorder to `crop → debayer → luma` to keep the whole run tiled.
//!
//! # Limitations
//! - `Op` covers debayer / luminance / affine pixel scale / pixel-type conversion
//!   / crop / ROI / flip / 90° rotation / aspect-preserving resize. No
//!   arbitrary-angle rotation and no exact-size (aspect-changing) resampling.

mod error;
mod geom;
mod kernels;
mod resample;

use bytemuck::{cast_slice, cast_slice_mut};
use serde::{Deserialize, Serialize};

pub use error::PipelineError;
pub use resample::ResizeFilter;

use geom::{geo_crop, geo_flip, geo_roi, geo_rot90};
use kernels::{convert_inplace, debayer_into, luma_inplace, scale_inplace, Demosaic};
use resample::{geo_resize, resize_dims};

use crate::demosaic::demosaic_serial_scratch_len;
use crate::{
    BayerPattern, BayerShift, ColorSpace, DemosaicMethod, DynamicImageOwned, DynamicImageRef,
    GenericImageOwned, GenericImageRef, ImageProps, ImageRef, PixelType,
};

/// Static description of an image's shape and element type.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpec {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of interleaved channels per pixel.
    pub channels: u8,
    /// Color space.
    pub cspace: ColorSpace,
    /// Primitive element type.
    pub pixel_type: PixelType,
}

impl ImageSpec {
    /// Build a spec, deriving `channels` from `cspace`.
    pub fn new(width: usize, height: usize, cspace: ColorSpace, pixel_type: PixelType) -> Self {
        let channels = match &cspace {
            ColorSpace::Gray | ColorSpace::Bayer(_) => 1,
            ColorSpace::Rgb => 3,
            ColorSpace::Custom(c, _) => *c,
        };
        Self {
            width,
            height,
            channels,
            cspace,
            pixel_type,
        }
    }

    /// Snapshot the shape of a live [`DynamicImageRef`].
    pub fn from_dynamic<I: ImageProps + ?Sized>(img: &I) -> Self {
        Self {
            width: img.width(),
            height: img.height(),
            channels: img.channels(),
            cspace: img.color_space(),
            pixel_type: img.pixel_type(),
        }
    }

    /// Element count (`width * height * channels`).
    pub fn elems(&self) -> usize {
        self.width * self.height * self.channels as usize
    }

    /// Byte count of a tightly-packed buffer for this spec.
    pub fn bytes(&self) -> Result<usize, PipelineError> {
        Ok(self.elems() * pixel_size(self.pixel_type)?)
    }

    /// Bytes per pixel (`channels * pixel_size`).
    fn bpp(&self) -> Result<usize, PipelineError> {
        Ok(self.channels as usize * pixel_size(self.pixel_type)?)
    }

    /// Bytes for a `rows * cols` tile in this spec's channels / element type.
    fn tile_bytes(&self, rows: usize, cols: usize) -> Result<usize, PipelineError> {
        Ok(rows * cols * self.bpp()?)
    }

    fn validate(&self) -> Result<(), PipelineError> {
        if self.width == 0 || self.height == 0 || self.width > 65535 || self.height > 65535 {
            return Err(PipelineError::BadDimensions);
        }
        pixel_size(self.pixel_type)?;
        Ok(())
    }
}

pub(crate) fn pixel_size(pt: PixelType) -> Result<usize, PipelineError> {
    match pt {
        PixelType::U8 => Ok(1),
        PixelType::U16 => Ok(2),
        PixelType::F32 => Ok(4),
        other => Err(PipelineError::UnsupportedPixelType(other)),
    }
}

/// A single processing stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Op {
    /// Demosaic a single-channel Bayer image to 3-channel RGB.
    Debayer(DemosaicMethod),
    /// Rec.601 luma (`0.299, 0.587, 0.114`); RGB (or `Custom`) becomes `Gray`.
    ToLuma,
    /// Luma with custom per-channel weights; length must equal the channel count.
    ToLumaCustom(Vec<f64>),
    /// Affine per-pixel remap `y = x * gain + offset`, evaluated on the raw
    /// stored value and saturated back into the current type (for `f32`, into
    /// `[0.0, 1.0]`). Shape, channels, and type are unchanged.
    Scale {
        /// Multiplicative factor.
        gain: f64,
        /// Additive term, in raw stored units.
        offset: f64,
    },
    /// Rescale every pixel into a different primitive type.
    Convert(PixelType),
    /// Extract the sub-rectangle with top-left `(x, y)` and size `width * height`.
    /// The rectangle must lie fully inside the image (else [`PipelineError::CropOutOfBounds`]).
    /// On a Bayer image the pattern is re-phased for an odd origin, so a crop may
    /// precede a debayer.
    Crop {
        /// Left edge, in pixels from the current origin.
        x: usize,
        /// Top edge, in pixels from the current origin.
        y: usize,
        /// Output width.
        width: usize,
        /// Output height.
        height: usize,
    },
    /// Region of interest: like [`Op::Crop`], but a `width`/`height` overhang past
    /// the image edge is legal — the missing pixels come out zero. Errors only if
    /// the origin `(x, y)` itself is outside the image
    /// ([`PipelineError::RoiOutOfBounds`]). Bayer patterns are re-phased.
    Roi {
        /// Left edge, in pixels from the current origin.
        x: usize,
        /// Top edge, in pixels from the current origin.
        y: usize,
        /// Output width.
        width: usize,
        /// Output height.
        height: usize,
    },
    /// Mirror left-to-right. Bayer patterns are re-phased.
    FlipHorizontal,
    /// Mirror top-to-bottom. Bayer patterns are re-phased.
    FlipVertical,
    /// Rotate 90° clockwise; width and height swap. Not valid on a Bayer image.
    Rotate90,
    /// Rotate 180°. Bayer patterns are re-phased.
    Rotate180,
    /// Rotate 90° counter-clockwise; width and height swap. Not valid on Bayer.
    Rotate270,
    /// Resample to the largest size that fits within `max_width` x `max_height`
    /// at the original aspect ratio, enlarging the image if it is smaller than
    /// the box. Each side of the result is at least 1 px and never exceeds its
    /// bound. Not valid on a Bayer image ([`PipelineError::ResizeOnBayer`]) —
    /// debayer first. Runs as one whole-frame pass (after any tiled pixel-op
    /// body) that allocates a scratch plane sized to the horizontal-pass
    /// intermediate (output-width x source-height); with the `rayon` feature the
    /// pass is fanned out over its rows, independent of thread count.
    ResizeToFit {
        /// Width bound in pixels; the result is never wider than this.
        max_width: usize,
        /// Height bound in pixels; the result is never taller than this.
        max_height: usize,
        /// Resampling filter.
        filter: ResizeFilter,
    },
}

/// Re-phase a Bayer pattern through a geometric transform; leave other color
/// spaces untouched.
fn rephase(cspace: &ColorSpace, f: impl Fn(BayerPattern) -> BayerPattern) -> ColorSpace {
    match cspace {
        ColorSpace::Bayer(p) => ColorSpace::Bayer(f(*p)),
        other => other.clone(),
    }
}

impl Op {
    fn output_spec(&self, input: &ImageSpec) -> Result<ImageSpec, PipelineError> {
        match self {
            Op::Debayer(_) => {
                if !matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::NotBayer);
                }
                if input.channels != 1 {
                    return Err(PipelineError::DebayerChannels(input.channels));
                }
                Ok(ImageSpec {
                    channels: 3,
                    cspace: ColorSpace::Rgb,
                    ..input.clone()
                })
            }
            Op::ToLuma => luma_output(input, 3),
            Op::ToLumaCustom(w) => luma_output(input, w.len()),
            Op::Scale { .. } => {
                pixel_size(input.pixel_type)?;
                Ok(input.clone())
            }
            Op::Convert(pt) => {
                pixel_size(*pt)?;
                Ok(ImageSpec {
                    pixel_type: *pt,
                    ..input.clone()
                })
            }
            Op::Crop {
                x,
                y,
                width,
                height,
            } => {
                if *width == 0 || *height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if x + width > input.width || y + height > input.height {
                    return Err(PipelineError::CropOutOfBounds {
                        rect: (*x, *y, *width, *height),
                        image: (input.width, input.height),
                    });
                }
                Ok(ImageSpec {
                    width: *width,
                    height: *height,
                    cspace: rephase(&input.cspace, |p| p.shift(*x, *y)),
                    ..input.clone()
                })
            }
            Op::Roi {
                x,
                y,
                width,
                height,
            } => {
                if *width == 0 || *height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if *x >= input.width || *y >= input.height {
                    return Err(PipelineError::RoiOutOfBounds {
                        origin: (*x, *y),
                        image: (input.width, input.height),
                    });
                }
                Ok(ImageSpec {
                    width: *width,
                    height: *height,
                    cspace: rephase(&input.cspace, |p| p.shift(*x, *y)),
                    ..input.clone()
                })
            }
            Op::FlipHorizontal => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_horizontal()),
                ..input.clone()
            }),
            Op::FlipVertical => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_vertical()),
                ..input.clone()
            }),
            Op::Rotate180 => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_horizontal().flip_vertical()),
                ..input.clone()
            }),
            Op::Rotate90 | Op::Rotate270 => {
                if matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::RotateOnBayer);
                }
                Ok(ImageSpec {
                    width: input.height,
                    height: input.width,
                    ..input.clone()
                })
            }
            Op::ResizeToFit {
                max_width,
                max_height,
                filter: _,
            } => {
                if *max_width == 0 || *max_height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::ResizeOnBayer);
                }
                pixel_size(input.pixel_type)?;
                let (width, height) =
                    resize_dims(input.width, input.height, *max_width, *max_height);
                Ok(ImageSpec {
                    width,
                    height,
                    ..input.clone()
                })
            }
        }
    }

    /// Rows/cols of vertical/horizontal context this op reads on each side of an
    /// output pixel; drives tile halos. Row-local and geometric ops are 0.
    fn halo(&self) -> usize {
        match self {
            Op::Debayer(DemosaicMethod::None) => 0,
            Op::Debayer(DemosaicMethod::Nearest) => 1,
            Op::Debayer(DemosaicMethod::Linear) => 1,
            Op::Debayer(DemosaicMethod::Cubic) => 3,
            Op::ToLuma
            | Op::ToLumaCustom(_)
            | Op::Scale { .. }
            | Op::Convert(_)
            | Op::Crop { .. }
            | Op::Roi { .. }
            | Op::FlipHorizontal
            | Op::FlipVertical
            | Op::Rotate90
            | Op::Rotate180
            | Op::Rotate270
            | Op::ResizeToFit { .. } => 0,
        }
    }

    /// A "pixel op" keeps every output pixel at its input `(x, y)` — so a run of
    /// them can be fused into one tiled pass. Geometric ops relocate pixels.
    fn is_pixel(&self) -> bool {
        matches!(
            self,
            Op::Debayer(_) | Op::ToLuma | Op::ToLumaCustom(_) | Op::Scale { .. } | Op::Convert(_)
        )
    }

    /// A crop with no other effect — used to fold leading crops into an input
    /// offset so a following pixel run can still tile.
    fn as_crop(&self) -> Option<(usize, usize)> {
        match self {
            Op::Crop { x, y, .. } => Some((*x, *y)),
            _ => None,
        }
    }
}

fn luma_output(input: &ImageSpec, ncoeffs: usize) -> Result<ImageSpec, PipelineError> {
    match input.cspace {
        // Matches `ImageRef::to_luma`: already-gray is a no-op passthrough.
        ColorSpace::Gray => Ok(input.clone()),
        ColorSpace::Rgb | ColorSpace::Custom(..) => {
            if input.channels as usize != ncoeffs {
                return Err(PipelineError::LumaCoeffMismatch {
                    channels: input.channels,
                    coeffs: ncoeffs,
                });
            }
            Ok(ImageSpec {
                channels: 1,
                cspace: ColorSpace::Gray,
                ..input.clone()
            })
        }
        ColorSpace::Bayer(_) => Err(PipelineError::LumaOnBayer),
    }
}

/// A declarative, reusable list of [`Op`]s. Cheap to clone; serializable, so a
/// processing recipe can live in a config file or image header.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Pipeline {
    ops: Vec<Op>,
}

impl Pipeline {
    /// An empty pipeline (identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append [`Op::Debayer`].
    pub fn debayer(mut self, method: DemosaicMethod) -> Self {
        self.ops.push(Op::Debayer(method));
        self
    }

    /// Append [`Op::ToLuma`].
    pub fn to_luma(mut self) -> Self {
        self.ops.push(Op::ToLuma);
        self
    }

    /// Append [`Op::ToLumaCustom`].
    pub fn to_luma_custom(mut self, weights: impl Into<Vec<f64>>) -> Self {
        self.ops.push(Op::ToLumaCustom(weights.into()));
        self
    }

    /// Append [`Op::Scale`] (`y = x * gain + offset`).
    pub fn scale(mut self, gain: f64, offset: f64) -> Self {
        self.ops.push(Op::Scale { gain, offset });
        self
    }

    /// Append [`Op::Convert`].
    pub fn convert(mut self, pixel_type: PixelType) -> Self {
        self.ops.push(Op::Convert(pixel_type));
        self
    }

    /// Append [`Op::Crop`].
    pub fn crop(mut self, x: usize, y: usize, width: usize, height: usize) -> Self {
        self.ops.push(Op::Crop {
            x,
            y,
            width,
            height,
        });
        self
    }

    /// Append [`Op::Roi`] (crop that zero-fills any overhang past the edge).
    pub fn roi(mut self, x: usize, y: usize, width: usize, height: usize) -> Self {
        self.ops.push(Op::Roi {
            x,
            y,
            width,
            height,
        });
        self
    }

    /// Append [`Op::FlipHorizontal`].
    pub fn flip_horizontal(mut self) -> Self {
        self.ops.push(Op::FlipHorizontal);
        self
    }

    /// Append [`Op::FlipVertical`].
    pub fn flip_vertical(mut self) -> Self {
        self.ops.push(Op::FlipVertical);
        self
    }

    /// Append [`Op::Rotate90`] (90° clockwise).
    pub fn rotate_90(mut self) -> Self {
        self.ops.push(Op::Rotate90);
        self
    }

    /// Append [`Op::Rotate180`].
    pub fn rotate_180(mut self) -> Self {
        self.ops.push(Op::Rotate180);
        self
    }

    /// Append [`Op::Rotate270`] (90° counter-clockwise).
    pub fn rotate_270(mut self) -> Self {
        self.ops.push(Op::Rotate270);
        self
    }

    /// Append [`Op::ResizeToFit`]: resample to the largest size fitting within
    /// `max_width` x `max_height` at the original aspect ratio.
    pub fn resize_to_fit(
        mut self,
        max_width: usize,
        max_height: usize,
        filter: ResizeFilter,
    ) -> Self {
        self.ops.push(Op::ResizeToFit {
            max_width,
            max_height,
            filter,
        });
        self
    }

    /// Append an arbitrary [`Op`].
    pub fn push(mut self, op: Op) -> Self {
        self.ops.push(op);
        self
    }

    /// The ops, in execution order.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Validate the chain against a concrete input and allocate all buffers.
    pub fn compile(&self, input: ImageSpec, strategy: Strategy) -> Result<Runner, PipelineError> {
        input.validate()?;
        let plan = Plan::build(&self.ops, &input)?;
        let n = plan.steps.len();

        // Fold a leading run of crops into an input offset, so `crop -> debayer
        // -> ...` still tiles. `plan.specs[prefix_lo]` already carries the
        // cropped dims and re-phased Bayer pattern.
        let mut prefix_lo = 0;
        let (mut in_off_x, mut in_off_y) = (0usize, 0usize);
        while prefix_lo < n {
            if let Some((x, y)) = self.ops[prefix_lo].as_crop() {
                in_off_x += x;
                in_off_y += y;
                prefix_lo += 1;
            } else {
                break;
            }
        }
        // The tiled body is the pixel-op run right after the leading crops.
        let mut prefix_hi = prefix_lo;
        while prefix_hi < n && self.ops[prefix_hi].is_pixel() {
            prefix_hi += 1;
        }

        let body = &self.ops[prefix_lo..prefix_hi];
        let body_debayer = body.iter().any(|op| matches!(op, Op::Debayer(_)));
        let halo: usize = body.iter().map(Op::halo).sum();
        let body_in = &plan.specs[prefix_lo];
        let (bw, bh) = (body_in.width, body_in.height);

        // Only tile when there is a pixel-op body to tile.
        let tile = if prefix_hi > prefix_lo {
            resolve_exec(strategy, bw, bh, halo, body_debayer)
        } else {
            None
        };

        let (cap_a, cap_b, out_cap, demo_cap, tail_cap) = match &tile {
            None => {
                let (a, b) = plan.buf_caps(0, n, |s| s.bytes())?;
                (f32_cap(a), f32_cap(b), 0, 0, 0)
            }
            Some(rt) => {
                let cols = if rt.tile_cols == 0 { bw } else { rt.tile_cols };
                let pr = (rt.tile_rows + 2 * halo + 6).min(bh);
                let pc = (cols + 2 * halo + 6).min(bw);
                let (a, b) = plan.buf_caps(prefix_lo, prefix_hi, |s| s.tile_bytes(pr, pc))?;
                let demo = if body_debayer {
                    demosaic_serial_scratch_len(bw)
                } else {
                    0
                };
                // `out_buf` receives the tiled body's output and is then the
                // first ping-pong buffer of the sequential geo tail; `tail_buf`
                // is the second. Both span the largest spec the tail touches.
                let tail_max = plan.max_bytes(prefix_hi, n)?;
                let tail_cap = if prefix_hi < n { f32_cap(tail_max) } else { 0 };
                (f32_cap(a), f32_cap(b), f32_cap(tail_max), demo, tail_cap)
            }
        };

        let exec = match &tile {
            None => Exec::Sequential,
            Some(rt) => Exec::Tiled {
                tile_rows: rt.tile_rows,
                tile_cols: rt.tile_cols,
                halo: rt.halo,
                even: rt.even,
                parallel: rt.parallel,
                in_off_x,
                in_off_y,
                prefix_lo,
                prefix_hi,
            },
        };

        Ok(Runner {
            pipeline: self.clone(),
            strategy,
            steps: plan.steps,
            coeffs: plan.coeffs,
            specs: plan.specs,
            out_spec: plan.out_spec,
            exec,
            buf_a: vec![0.0; cap_a],
            buf_b: vec![0.0; cap_b],
            out_buf: vec![0.0; out_cap],
            tail_buf: vec![0.0; tail_cap],
            demosaic_scratch: vec![0.0; demo_cap],
        })
    }

    /// Run the chain once against `img` and return an owned result.
    ///
    /// The output mirrors the input: a [`DynamicImageRef`] / [`DynamicImageOwned`]
    /// yields a [`DynamicImageOwned`], and a metadata-bearing [`GenericImageRef`] /
    /// [`GenericImageOwned`] yields a [`GenericImageOwned`] carrying the same
    /// [`Metadata`](crate::Metadata) unchanged.
    ///
    /// Compiles with [`Strategy::Sequential`] and discards the [`Runner`] — for a
    /// stream of frames, keep a [`Runner`] from [`compile`](Pipeline::compile)
    /// instead so the buffers are reused.
    pub fn apply<I: ApplyInput + ?Sized>(&self, img: &I) -> Result<I::Output, PipelineError> {
        img.run_pipeline(self)
    }
}

/// A frame the pipeline reads from: its shape (via [`ImageProps`]) plus the pixel
/// data as native-endian bytes. Implemented for [`DynamicImageRef`] and
/// [`DynamicImageOwned`]; it is the input to [`Runner::run`] / [`Runner::run_into`]
/// and, through [`ApplyInput`], to [`Pipeline::apply`].
pub trait Frame: ImageProps {
    /// The pixel data as a native-endian byte slice.
    fn as_bytes(&self) -> &[u8];
}

impl Frame for DynamicImageRef<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.as_raw_u8()
    }
}

impl Frame for DynamicImageOwned {
    fn as_bytes(&self) -> &[u8] {
        self.as_raw_u8()
    }
}

/// An image a [`Pipeline`] can be applied to: a [`DynamicImageRef`] /
/// [`DynamicImageOwned`], or a metadata-bearing [`GenericImageRef`] /
/// [`GenericImageOwned`]. [`Output`](ApplyInput::Output) is the corresponding owned
/// image type — a `Generic*` input keeps its metadata.
pub trait ApplyInput {
    /// The owned image [`Pipeline::apply`] produces for this input.
    type Output;

    #[doc(hidden)]
    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError>;
}

/// Run `pipeline` once over `frame`, returning a fresh owned image.
fn apply_dynamic<F: Frame + ?Sized>(
    pipeline: &Pipeline,
    frame: &F,
) -> Result<DynamicImageOwned, PipelineError> {
    let mut runner = pipeline.compile(ImageSpec::from_dynamic(frame), Strategy::Sequential)?;
    let out = runner.run(frame)?;
    Ok(DynamicImageOwned::from(&out))
}

impl ApplyInput for DynamicImageRef<'_> {
    type Output = DynamicImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        apply_dynamic(pipeline, self)
    }
}

impl ApplyInput for DynamicImageOwned {
    type Output = DynamicImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        apply_dynamic(pipeline, self)
    }
}

impl ApplyInput for GenericImageRef<'_> {
    type Output = GenericImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        Ok(GenericImageOwned {
            metadata: self.metadata.clone(),
            image: apply_dynamic(pipeline, self.get_image())?,
        })
    }
}

impl ApplyInput for GenericImageOwned {
    type Output = GenericImageOwned;

    fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Self::Output, PipelineError> {
        Ok(GenericImageOwned {
            metadata: self.metadata.clone(),
            image: apply_dynamic(pipeline, self.get_image())?,
        })
    }
}

/// The per-step execution plan plus every intermediate shape.
struct Plan {
    steps: Vec<Step>,
    coeffs: Vec<Box<[f64]>>,
    specs: Vec<ImageSpec>, // len == steps.len() + 1
    out_spec: ImageSpec,
}

impl Plan {
    fn build(ops: &[Op], input: &ImageSpec) -> Result<Self, PipelineError> {
        let mut cur = input.clone();
        let mut specs = Vec::with_capacity(ops.len() + 1);
        specs.push(cur.clone());
        let mut steps = Vec::with_capacity(ops.len());
        let mut coeffs: Vec<Box<[f64]>> = Vec::new();

        for op in ops {
            let next = op.output_spec(&cur)?;
            let mut step = Step {
                kind: StepKind::Convert,
                in_pt: cur.pixel_type,
                out_pt: next.pixel_type,
                in_channels: cur.channels,
                bayer: None,
                coeff_idx: 0,
                luma_identity: false,
            };
            match op {
                Op::Debayer(m) => {
                    let ColorSpace::Bayer(pat) = cur.cspace else {
                        unreachable!("checked by output_spec")
                    };
                    step.kind = StepKind::Debayer(*m);
                    step.bayer = Some(pat);
                }
                Op::ToLuma | Op::ToLumaCustom(_) => {
                    step.kind = StepKind::Luma;
                    if matches!(cur.cspace, ColorSpace::Gray) {
                        step.luma_identity = true;
                    } else {
                        let w: Box<[f64]> = match op {
                            Op::ToLumaCustom(v) => v.clone().into_boxed_slice(),
                            _ => Box::new([0.299, 0.587, 0.114]),
                        };
                        step.coeff_idx = coeffs.len();
                        coeffs.push(w);
                    }
                }
                Op::Scale { gain, offset } => {
                    step.kind = StepKind::Scale {
                        gain: *gain,
                        offset: *offset,
                    }
                }
                Op::Convert(_) => step.kind = StepKind::Convert,
                Op::Crop { x, y, .. } => {
                    step.kind = StepKind::Crop {
                        x: *x,
                        y: *y,
                        w: next.width,
                        h: next.height,
                    }
                }
                Op::Roi { x, y, .. } => {
                    step.kind = StepKind::Roi {
                        x: *x,
                        y: *y,
                        w: next.width,
                        h: next.height,
                    }
                }
                Op::FlipHorizontal => {
                    step.kind = StepKind::Flip {
                        horizontal: true,
                        vertical: false,
                    }
                }
                Op::FlipVertical => {
                    step.kind = StepKind::Flip {
                        horizontal: false,
                        vertical: true,
                    }
                }
                Op::Rotate180 => {
                    step.kind = StepKind::Flip {
                        horizontal: true,
                        vertical: true,
                    }
                }
                Op::Rotate90 => step.kind = StepKind::Rot90 { ccw: false },
                Op::Rotate270 => step.kind = StepKind::Rot90 { ccw: true },
                Op::ResizeToFit { filter, .. } => {
                    step.kind = StepKind::Resize {
                        w: next.width,
                        h: next.height,
                        filter: *filter,
                    }
                }
            }
            steps.push(step);
            specs.push(next.clone());
            cur = next;
        }

        Ok(Plan {
            steps,
            coeffs,
            out_spec: cur,
            specs,
        })
    }

    /// Walk `steps[lo..hi]` the way [`run_chain`] does — buffer `A` holds
    /// `specs[lo]`, swapping steps flip the home buffer, in-place steps keep it —
    /// and return `(max bytes ever in A, max bytes ever in B)` under `cell` (a
    /// per-spec size: full-frame for `Sequential`, padded-tile for `Tiled`).
    fn buf_caps(
        &self,
        lo: usize,
        hi: usize,
        cell: impl Fn(&ImageSpec) -> Result<usize, PipelineError>,
    ) -> Result<(usize, usize), PipelineError> {
        let mut in_b = false;
        let mut a = cell(&self.specs[lo])?;
        let mut b = 0usize;
        for i in lo..hi {
            if self.steps[i].swaps() {
                in_b = !in_b;
            }
            let c = cell(&self.specs[i + 1])?;
            if in_b {
                b = b.max(c);
            } else {
                a = a.max(c);
            }
        }
        Ok((a.max(1), b.max(1)))
    }

    /// Largest full-frame buffer any spec in `specs[lo..=hi]` needs.
    fn max_bytes(&self, lo: usize, hi: usize) -> Result<usize, PipelineError> {
        let mut m = 0;
        for s in &self.specs[lo..=hi] {
            m = m.max(s.bytes()?);
        }
        Ok(m.max(1))
    }
}

/// f32 elements needed to hold `bytes` (rounded up), at least 1.
fn f32_cap(bytes: usize) -> usize {
    bytes.div_ceil(4).max(1)
}

/// Auto tile height aims for roughly this working-set size per band.
const TILE_TARGET_BYTES: usize = 256 * 1024;

/// Tiling parameters resolved from a [`Strategy`] against concrete dimensions.
struct ResolvedTile {
    tile_rows: usize,
    tile_cols: usize,
    halo: usize,
    even: bool,
    parallel: bool,
}

/// Turn a [`Strategy`] + the *tiled body's* dimensions into concrete tiling
/// parameters, or `None` when tiling cannot help and the body should run
/// sequentially.
fn resolve_exec(
    strategy: Strategy,
    w: usize,
    h: usize,
    halo: usize,
    even: bool,
) -> Option<ResolvedTile> {
    let Strategy::Tiled {
        tile_rows,
        tile_cols,
        parallel,
    } = strategy
    else {
        return None;
    };

    // The halo math needs a few rows of slack; below that, don't tile in Y.
    let min_dim = 2 * halo + 6;
    let mut cols = if tile_cols == 0 || tile_cols >= w {
        0 // full width
    } else {
        tile_cols
    };
    if cols != 0 && cols < min_dim {
        cols = 0; // too narrow to tile in X usefully
    }
    // Auto row height targets a working set; when the bands are column-tiled it
    // is `cols` wide, not the full frame.
    let band_w = if cols != 0 { cols } else { w };
    let rows = if tile_rows == 0 {
        // aim for TILE_TARGET_BYTES over a u16 RGB row (6 B/px worst case)
        (TILE_TARGET_BYTES / (band_w * 6).max(1)).clamp(1, h)
    } else {
        tile_rows
    };

    let one_band = rows >= h;
    let one_col = cols == 0;
    if (one_band && one_col) || h < min_dim {
        return None;
    }

    Some(ResolvedTile {
        tile_rows: rows.min(h),
        tile_cols: cols,
        halo,
        even,
        parallel,
    })
}

/// How a [`Runner`] executes its steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Strategy {
    /// One ping-pong buffer pair; every stage runs over the whole frame in order.
    Sequential,
    /// Tiled execution; see the [module docs](self#execution-strategies).
    Tiled {
        /// Output rows per band before the halo. `0` picks an automatic height.
        tile_rows: usize,
        /// Output columns per tile before the halo. `0` (or `>= width`) keeps
        /// bands full-width — cheaper, and all that wide-enough frames need.
        tile_cols: usize,
        /// Fan bands out across a rayon pool (needs the `rayon` feature; a plain
        /// serial sweep otherwise).
        parallel: bool,
    },
}

impl Strategy {
    /// Serial row-band tiling. `tile_rows == 0` auto-sizes.
    pub fn tiled(tile_rows: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols: 0,
            parallel: false,
        }
    }

    /// Row-band tiling fanned out over a rayon pool. `tile_rows == 0` auto-sizes.
    pub fn tiled_parallel(tile_rows: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols: 0,
            parallel: true,
        }
    }

    /// 2-D tiling: `tile_rows` bands, each split into `tile_cols`-wide tiles.
    pub fn tiled_2d(tile_rows: usize, tile_cols: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols,
            parallel: false,
        }
    }

    /// 2-D tiling with the bands fanned out over a rayon pool.
    pub fn tiled_2d_parallel(tile_rows: usize, tile_cols: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols,
            parallel: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Exec {
    /// Run every step over the whole frame, in order.
    Sequential,
    /// Tile `steps[prefix_lo..prefix_hi]` (all pixel ops) into `out_buf`, reading
    /// the frame from `(in_off_x, in_off_y)` (folded leading crops). Any steps
    /// past `prefix_hi` — geometric ops and whatever follows — then run as a
    /// sequential pass over `out_buf`/`tail_buf`.
    Tiled {
        tile_rows: usize,
        tile_cols: usize, // 0 == full width
        halo: usize,
        /// Snap tile origins to even rows/cols (keeps the Bayer phase); set when
        /// the tiled body contains a debayer.
        even: bool,
        parallel: bool,
        in_off_x: usize,
        in_off_y: usize,
        prefix_lo: usize,
        prefix_hi: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct Step {
    kind: StepKind,
    in_pt: PixelType,
    out_pt: PixelType,
    in_channels: u8,
    bayer: Option<BayerPattern>,
    coeff_idx: usize,
    luma_identity: bool,
}

impl Step {
    /// Does this step read one buffer and write the other? Debayer and every
    /// geometric op do; luma, scale, and convert rewrite their buffer in place.
    fn swaps(&self) -> bool {
        matches!(
            self.kind,
            StepKind::Debayer(_)
                | StepKind::Crop { .. }
                | StepKind::Roi { .. }
                | StepKind::Flip { .. }
                | StepKind::Rot90 { .. }
                | StepKind::Resize { .. }
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum StepKind {
    Debayer(DemosaicMethod),
    Luma,
    Scale {
        gain: f64,
        offset: f64,
    },
    Convert,
    /// Origin `(x, y)` into the current image; output size `(w, h)`.
    Crop {
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    },
    /// Like [`StepKind::Crop`] but zero-fills any part of `(w, h)` that runs off
    /// the source edge.
    Roi {
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    },
    Flip {
        horizontal: bool,
        vertical: bool,
    },
    /// Quarter turns clockwise, 1 or 3 (2 is folded to `Flip`).
    Rot90 {
        ccw: bool,
    },
    /// Resample the current image to `w` x `h` with `filter`.
    Resize {
        w: usize,
        h: usize,
        filter: ResizeFilter,
    },
}

/// A compiled [`Pipeline`]: recipe + locked input spec + owned scratch.
///
/// Build one per input format, keep it alive, and call [`run`](Runner::run) per
/// frame.
#[derive(Debug)]
pub struct Runner {
    pipeline: Pipeline,
    strategy: Strategy,
    steps: Vec<Step>,
    coeffs: Vec<Box<[f64]>>,
    // Every intermediate shape; `specs[0]` is the input, `specs.last()` the
    // output. `len == steps.len() + 1`.
    specs: Vec<ImageSpec>,
    out_spec: ImageSpec,
    exec: Exec,
    // `f32`-backed for 4-byte alignment; reinterpreted per stage. Sized
    // independently by the liveness walk. Sequential: full-frame ping-pong.
    // Tiled (serial): per-tile ping-pong.
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    // Tiled: the assembled tiled-body output, then the first ping-pong buffer of
    // the sequential geo tail. Empty for Sequential.
    out_buf: Vec<f32>,
    // Tiled with a geometric tail: the tail's second ping-pong buffer.
    tail_buf: Vec<f32>,
    // Tiled + debayer, serial: pooled working buffer for the serial kernel.
    // Empty otherwise (the parallel path allocates one per worker instead).
    demosaic_scratch: Vec<f32>,
}

impl Runner {
    /// The input spec this runner is currently compiled for.
    pub fn input_spec(&self) -> &ImageSpec {
        &self.specs[0]
    }

    /// The spec of the image [`run`](Runner::run) returns.
    pub fn output_spec(&self) -> &ImageSpec {
        &self.out_spec
    }

    /// `true` if this runner tiles the frame.
    pub fn is_tiled(&self) -> bool {
        matches!(self.exec, Exec::Tiled { .. })
    }

    /// Total scratch held, in bytes.
    pub fn scratch_bytes(&self) -> usize {
        (self.buf_a.len()
            + self.buf_b.len()
            + self.out_buf.len()
            + self.tail_buf.len()
            + self.demosaic_scratch.len())
            * 4
    }

    /// Rebuild the plan and buffers for a new input shape. Allocates.
    ///
    /// Use this when the acquired frame size changes. Errors exactly as
    /// [`Pipeline::compile`] would for the new spec.
    pub fn recompile(&mut self, input: ImageSpec) -> Result<(), PipelineError> {
        *self = self.pipeline.compile(input, self.strategy)?;
        Ok(())
    }

    /// Run the chain against one frame.
    ///
    /// The frame's shape must equal [`input_spec`](Runner::input_spec) — unless the
    /// `grow` feature is enabled, in which case a mismatch triggers an automatic
    /// [`recompile`](Runner::recompile). The result borrows the runner's internal
    /// buffer and stays valid until the next call.
    pub fn run<'r, F: Frame + ?Sized>(
        &'r mut self,
        frame: &F,
    ) -> Result<DynamicImageRef<'r>, PipelineError> {
        let got = ImageSpec::from_dynamic(frame);
        if got != self.specs[0] {
            #[cfg(feature = "grow")]
            {
                self.recompile(got)?;
            }
            #[cfg(not(feature = "grow"))]
            {
                return Err(PipelineError::InputMismatch {
                    expected: Box::new(self.specs[0].clone()),
                    got: Box::new(got),
                });
            }
        }

        let raw = frame.as_bytes();

        match self.exec {
            Exec::Sequential => {
                let (w, h) = (self.specs[0].width, self.specs[0].height);
                cast_slice_mut::<f32, u8>(&mut self.buf_a)[..raw.len()].copy_from_slice(raw);
                let cur_a = run_chain(
                    &self.steps,
                    &self.coeffs,
                    &mut self.buf_a,
                    &mut self.buf_b,
                    w,
                    h,
                    Demosaic::Alloc,
                )?;
                let buf = current(&mut self.buf_a, &mut self.buf_b, cur_a);
                view(buf, &self.out_spec)
            }
            Exec::Tiled {
                tile_rows,
                tile_cols,
                halo,
                even,
                parallel,
                in_off_x,
                in_off_y,
                prefix_lo,
                prefix_hi,
            } => {
                let body_in = self.specs[prefix_lo].clone();
                let body_out = self.specs[prefix_hi].clone();
                let frame_stride = self.specs[0].width * self.specs[0].bpp()?;
                let frame_off = in_off_y * frame_stride + in_off_x * body_in.bpp()?;
                let has_tail = prefix_hi < self.steps.len();
                let (bo_w, bo_h) = (body_out.width, body_out.height);

                fill_tiled(
                    &self.steps[prefix_lo..prefix_hi],
                    &self.coeffs,
                    &body_in,
                    &body_out,
                    raw,
                    frame_stride,
                    frame_off,
                    &mut self.out_buf,
                    &mut self.buf_a,
                    &mut self.buf_b,
                    &mut self.demosaic_scratch,
                    TileGeom {
                        tile_rows,
                        tile_cols,
                        halo,
                        even,
                    },
                    parallel,
                )?;

                if has_tail {
                    let cur_a = run_chain(
                        &self.steps[prefix_hi..],
                        &self.coeffs,
                        &mut self.out_buf,
                        &mut self.tail_buf,
                        bo_w,
                        bo_h,
                        Demosaic::Alloc,
                    )?;
                    let buf = current(&mut self.out_buf, &mut self.tail_buf, cur_a);
                    view(buf, &self.out_spec)
                } else {
                    view(&mut self.out_buf, &self.out_spec)
                }
            }
        }
    }

    /// Run the chain, then copy the output into `dest`.
    ///
    /// When `dest` already has the output shape
    /// ([`output_spec`](Runner::output_spec)) this is a plain byte copy with no
    /// allocation — the pattern for a hot loop that owns its destination. When
    /// the shape differs, `dest` is rebuilt (which allocates).
    ///
    /// With an [`Op::Roi`] first in the chain this reproduces a "blit region into
    /// a pre-sized buffer" — the old `CopyRoi::copy_to`.
    pub fn run_into<F: Frame + ?Sized>(
        &mut self,
        frame: &F,
        dest: &mut DynamicImageOwned,
    ) -> Result<(), PipelineError> {
        let spec = self.out_spec.clone();
        let fits = dest.width() == spec.width
            && dest.height() == spec.height
            && dest.channels() == spec.channels
            && dest.color_space() == spec.cspace
            && dest.pixel_type() == spec.pixel_type;
        let out = self.run(frame)?;
        if fits {
            dest.as_mut_raw_u8().copy_from_slice(out.as_raw_u8());
        } else {
            *dest = DynamicImageOwned::from(&out);
        }
        Ok(())
    }
}

/// Execute every step over an image whose input (shape `w * h`) already sits in
/// `buf_a`. Geometric steps update the running dimensions. Returns `true` if the
/// result ended up in `buf_a`, `false` if in `buf_b`.
fn run_chain(
    steps: &[Step],
    coeffs: &[Box<[f64]>],
    buf_a: &mut [f32],
    buf_b: &mut [f32],
    w: usize,
    h: usize,
    mut demosaic: Demosaic<'_>,
) -> Result<bool, PipelineError> {
    let mut cur_a = true;
    let (mut cw, mut ch) = (w, h);
    for step in steps {
        let n = cw * ch * step.in_channels as usize;
        let bpp = step.in_channels as usize * pixel_size(step.in_pt)?;
        match step.kind {
            StepKind::Debayer(method) => {
                let pat = step.bayer.expect("debayer step carries a pattern");
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                debayer_into(src, dst, step.in_pt, cw, ch, pat, method, &mut demosaic)?;
                cur_a = !cur_a;
            }
            StepKind::Convert => {
                let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                convert_inplace(buf, step.in_pt, step.out_pt, n)?;
            }
            StepKind::Scale { gain, offset } => {
                let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                scale_inplace(buf, step.in_pt, n, gain, offset)?;
            }
            StepKind::Luma => {
                if !step.luma_identity {
                    let cf = &coeffs[step.coeff_idx];
                    let buf = current(&mut *buf_a, &mut *buf_b, cur_a);
                    luma_inplace(buf, step.in_pt, step.in_channels as usize, n, cf)?;
                }
            }
            StepKind::Crop { x, y, w: ow, h: oh } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_crop(src, dst, cw, bpp, x, y, ow, oh);
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
            StepKind::Roi { x, y, w: ow, h: oh } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_roi(src, dst, cw, ch, bpp, x, y, ow, oh);
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
            StepKind::Flip {
                horizontal,
                vertical,
            } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_flip(src, dst, cw, ch, bpp, horizontal, vertical);
                cur_a = !cur_a;
            }
            StepKind::Rot90 { ccw } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_rot90(src, dst, cw, ch, bpp, ccw);
                std::mem::swap(&mut cw, &mut ch);
                cur_a = !cur_a;
            }
            StepKind::Resize {
                w: ow,
                h: oh,
                filter,
            } => {
                let (src, dst) = pick(&mut *buf_a, &mut *buf_b, cur_a);
                geo_resize(
                    src,
                    dst,
                    cw,
                    ch,
                    step.in_channels as usize,
                    step.in_pt,
                    ow,
                    oh,
                    filter,
                )?;
                cw = ow;
                ch = oh;
                cur_a = !cur_a;
            }
        }
    }
    Ok(cur_a)
}

/// Expand an output range `[lo, hi)` to a source range that (a) starts on an even
/// boundary when `even` (keeps the Bayer phase), (b) carries `halo` real
/// rows/cols of context on each side of the kept region, and (c) is at least
/// `2*halo + 3` wide so the serial kernel has room. Never shrinks past
/// `[lo, hi)`; clamps to `[0, limit)`.
fn halo_range(lo: usize, hi: usize, limit: usize, halo: usize, even: bool) -> (usize, usize) {
    if halo == 0 && !even {
        return (lo, hi);
    }
    let (mut a, mut b) = if even {
        (
            lo.saturating_sub(halo) & !1,
            ((hi + halo + 1) & !1).min(limit),
        )
    } else {
        (lo.saturating_sub(halo), (hi + halo).min(limit))
    };
    let step = if even { 2 } else { 1 };
    let min_span = 2 * halo + 3;
    while b - a < min_span {
        if b < limit {
            b = (b + step).min(limit);
        } else if a >= step {
            a -= step;
        } else {
            break;
        }
    }
    (a, b)
}

/// Tile geometry shared by every tile in one `fill_tiled` call.
#[derive(Clone, Copy)]
struct TileGeom {
    tile_rows: usize,
    tile_cols: usize, // 0 == full width
    halo: usize,
    even: bool,
}

/// Fill `out_buf` by running the chain independently on each tile. Bands (row
/// strips) are the unit of parallelism; each band is swept left-to-right in
/// `tile_cols`-wide column tiles.
#[allow(clippy::too_many_arguments)]
fn fill_tiled(
    steps: &[Step],
    coeffs: &[Box<[f64]>],
    in_spec: &ImageSpec,
    out_spec: &ImageSpec,
    frame: &[u8],
    // Byte stride of one frame row, and the byte offset of the tiled body's
    // origin into `frame` (folded leading crops). For an un-cropped input these
    // are `in_spec.width * in_bpp` and `0`.
    frame_stride: usize,
    frame_off: usize,
    out_buf: &mut [f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    demosaic_scratch: &mut [f32],
    geom: TileGeom,
    parallel: bool,
) -> Result<(), PipelineError> {
    let TileGeom {
        tile_rows,
        tile_cols,
        halo,
        even,
    } = geom;
    let w = in_spec.width;
    let h = in_spec.height;
    let cols = if tile_cols == 0 || tile_cols >= w {
        w
    } else {
        tile_cols
    };
    let in_bpp = in_spec.bpp()?;
    let out_bpp = out_spec.bpp()?;
    let out_row = w * out_bpp;
    let out_bytes = h * out_row;
    let n_cols = w.div_ceil(cols);
    let cap_a = scratch_a.len();
    let cap_b = scratch_b.len();
    let demo_len = demosaic_scratch.len();

    // One band = one contiguous `[y0, y1)` slab of the output.
    let do_band = |band_idx: usize,
                   band: &mut [u8],
                   sa: &mut [f32],
                   sb: &mut [f32],
                   demo: &mut [f32]|
     -> Result<(), PipelineError> {
        let y0 = band_idx * tile_rows;
        if y0 >= h {
            return Ok(());
        }
        let y1 = (y0 + tile_rows).min(h);
        let (ry0, ry1) = halo_range(y0, y1, h, halo, even);
        let sh = ry1 - ry0;

        for col in 0..n_cols {
            let x0 = col * cols;
            let x1 = (x0 + cols).min(w);
            let (rx0, rx1) = halo_range(x0, x1, w, halo, even);
            let sw = rx1 - rx0;

            // Assemble the padded input sub-rect into `sa`, packed at width `sw`.
            let sa_u8 = cast_slice_mut::<f32, u8>(&mut *sa);
            let s_in_row = sw * in_bpp;
            for r in 0..sh {
                let src = frame_off + (ry0 + r) * frame_stride + rx0 * in_bpp;
                sa_u8[r * s_in_row..r * s_in_row + s_in_row]
                    .copy_from_slice(&frame[src..src + s_in_row]);
            }

            // Run the chain on the tile with a serial demosaic kernel.
            let dm = if demo.is_empty() {
                Demosaic::Alloc
            } else {
                Demosaic::Pooled(&mut *demo)
            };
            let cur_a = run_chain(steps, coeffs, &mut *sa, &mut *sb, sw, sh, dm)?;
            let res: &[f32] = if cur_a { &*sa } else { &*sb };
            let res_u8 = cast_slice::<f32, u8>(res);
            let s_out_row = sw * out_bpp;

            // Scatter the valid center rect into the band.
            let y_skip = y0 - ry0;
            if x0 == 0 && x1 == w && sw == w {
                let take = (y1 - y0) * out_row;
                let s0 = y_skip * s_out_row;
                band[..take].copy_from_slice(&res_u8[s0..s0 + take]);
            } else {
                let x_skip_b = (x0 - rx0) * out_bpp;
                let span_b = (x1 - x0) * out_bpp;
                for r in 0..(y1 - y0) {
                    let d = r * out_row + x0 * out_bpp;
                    let s = (y_skip + r) * s_out_row + x_skip_b;
                    band[d..d + span_b].copy_from_slice(&res_u8[s..s + span_b]);
                }
            }
        }
        Ok(())
    };

    #[cfg(feature = "rayon")]
    if parallel {
        use rayon::prelude::*;
        let out_u8 = &mut cast_slice_mut::<f32, u8>(out_buf)[..out_bytes];
        return out_u8
            .par_chunks_mut(tile_rows * out_row)
            .enumerate()
            .try_for_each_init(
                || {
                    (
                        vec![0.0f32; cap_a],
                        vec![0.0f32; cap_b],
                        vec![0.0f32; demo_len],
                    )
                },
                |(sa, sb, demo), (idx, band)| do_band(idx, band, sa, sb, demo),
            );
    }
    #[cfg(not(feature = "rayon"))]
    let _ = (parallel, cap_a, cap_b, demo_len);

    let out_u8 = &mut cast_slice_mut::<f32, u8>(out_buf)[..out_bytes];
    for (idx, band) in out_u8.chunks_mut(tile_rows * out_row).enumerate() {
        do_band(
            idx,
            band,
            &mut *scratch_a,
            &mut *scratch_b,
            &mut *demosaic_scratch,
        )?;
    }
    Ok(())
}

fn pick<'x>(a: &'x mut [f32], b: &'x mut [f32], cur_a: bool) -> (&'x mut [f32], &'x mut [f32]) {
    if cur_a {
        (a, b)
    } else {
        (b, a)
    }
}

fn current<'x>(a: &'x mut [f32], b: &'x mut [f32], cur_a: bool) -> &'x mut [f32] {
    if cur_a {
        a
    } else {
        b
    }
}

fn view<'a>(buf: &'a mut [f32], spec: &ImageSpec) -> Result<DynamicImageRef<'a>, PipelineError> {
    let n = spec.elems();
    let (w, h) = (spec.width, spec.height);
    let cs = spec.cspace.clone();
    Ok(match spec.pixel_type {
        PixelType::U8 => {
            let d = &mut cast_slice_mut::<f32, u8>(buf)[..n];
            DynamicImageRef::from(ImageRef::<u8>::create(d, w, h, cs)?)
        }
        PixelType::U16 => {
            let d = &mut cast_slice_mut::<f32, u16>(buf)[..n];
            DynamicImageRef::from(ImageRef::<u16>::create(d, w, h, cs)?)
        }
        PixelType::F32 => {
            let d = &mut buf[..n];
            DynamicImageRef::from(ImageRef::<f32>::create(d, w, h, cs)?)
        }
        other => return Err(PipelineError::UnsupportedPixelType(other)),
    })
}

#[cfg(test)]
mod tests;
