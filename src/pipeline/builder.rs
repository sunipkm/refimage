//! [`Pipeline`] — the declarative op list and the `compile` (planning + buffer
//! allocation) and `apply` (one-shot) functions.

use serde::{Deserialize, Serialize};

use crate::demosaic::demosaic_serial_scratch_len;
use crate::{DemosaicMethod, PixelType};

use super::plan::{build_tail_phases, f32_cap, resolve_exec, Exec, Plan, TailPhase};
use super::{ApplyInput, ImageSpec, Op, PipelineError, ResizeFilter, Runner, Strategy};

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

        let (cap_a, cap_b, out_cap, demo_cap, tail_cap, tail_phases) = match &tile {
            None => {
                let (a, b) = plan.buf_caps(0, n, |s| s.bytes())?;
                (f32_cap(a), f32_cap(b), 0, 0, 0, Vec::new())
            }
            Some(rt) => {
                let cols = if rt.tile_cols == 0 { bw } else { rt.tile_cols };
                let pr = (rt.tile_rows + 2 * halo + 6).min(bh);
                let pc = (cols + 2 * halo + 6).min(bw);
                let (mut a, mut b) =
                    plan.buf_caps(prefix_lo, prefix_hi, |s| s.tile_bytes(pr, pc))?;
                let demo = if body_debayer {
                    demosaic_serial_scratch_len(bw)
                } else {
                    0
                };
                // Split `steps[prefix_hi..]` into whole-frame / retiled passes.
                // Each retiled pass needs its own padded-tile scratch, folded
                // into `buf_a`/`buf_b` (halo 0 there — no debayer past the body).
                let tail_phases = build_tail_phases(&plan.steps, &plan.specs, strategy, prefix_hi);
                for ph in &tail_phases {
                    if let TailPhase::Tiled {
                        lo,
                        hi,
                        tile_rows,
                        tile_cols,
                        ..
                    } = *ph
                    {
                        let s = &plan.specs[lo];
                        let c = if tile_cols == 0 { s.width } else { tile_cols };
                        let ppr = (tile_rows + 6).min(s.height);
                        let ppc = (c + 6).min(s.width);
                        let (sa, sb) = plan.buf_caps(lo, hi, |sp| sp.tile_bytes(ppr, ppc))?;
                        a = a.max(sa);
                        b = b.max(sb);
                    }
                }
                // `out_buf`/`tail_buf` are the full-frame ping-pong for the
                // remainder; both span the largest spec from `prefix_hi` on.
                let tail_max = plan.max_bytes(prefix_hi, n)?;
                let tail_cap = if prefix_hi < n { f32_cap(tail_max) } else { 0 };
                (
                    f32_cap(a),
                    f32_cap(b),
                    f32_cap(tail_max),
                    demo,
                    tail_cap,
                    tail_phases,
                )
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
            tail_phases,
        })
    }

    /// Run the chain once against `img` and return an owned result.
    ///
    /// The output mirrors the input: a [`DynamicImageRef`](crate::DynamicImageRef)
    /// / [`DynamicImageOwned`](crate::DynamicImageOwned) yields a
    /// [`DynamicImageOwned`](crate::DynamicImageOwned), and a metadata-bearing
    /// [`GenericImageRef`](crate::GenericImageRef) /
    /// [`GenericImageOwned`](crate::GenericImageOwned) yields a
    /// [`GenericImageOwned`](crate::GenericImageOwned) carrying the same
    /// [`Metadata`](crate::Metadata) unchanged.
    ///
    /// Compiles with [`Strategy::Sequential`] and discards the [`Runner`] — for a
    /// stream of frames, keep a [`Runner`] from [`compile`](Pipeline::compile)
    /// instead so the buffers are reused.
    pub fn apply<I: ApplyInput + ?Sized>(&self, img: &I) -> Result<I::Output, PipelineError> {
        img.run_pipeline(self)
    }
}
