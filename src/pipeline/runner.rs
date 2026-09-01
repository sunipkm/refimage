//! [`Runner`] — a compiled, reusable [`Pipeline`] containing scratch buffers, executed per
//! frame.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{DynamicImageOwned, DynamicImageRef, ImageProps};

use super::exec::{current, fill_tiled, run_chain, view, TileGeom};
use super::kernels::Demosaic;
use super::plan::{Exec, Step, TailPhase};
use super::{Frame, ImageSpec, Pipeline, PipelineError, Strategy};

/// A compiled [`Pipeline`]: recipe + locked input spec + owned scratch.
///
/// Build one per input format, keep it alive, and call [`run`](Runner::run) per
/// frame.
#[derive(Debug)]
pub struct Runner {
    pub(super) pipeline: Pipeline,
    pub(super) strategy: Strategy,
    pub(super) steps: Vec<Step>,
    pub(super) coeffs: Vec<Box<[f64]>>,
    // Every intermediate shape; `specs[0]` is the input, `specs.last()` the
    // output. `len == steps.len() + 1`.
    pub(super) specs: Vec<ImageSpec>,
    pub(super) out_spec: ImageSpec,
    pub(super) exec: Exec,
    // `f32`-backed for 4-byte alignment; reinterpreted per stage. Sized
    // independently by the liveness walk. Sequential: full-frame ping-pong.
    // Tiled (serial): per-tile ping-pong.
    pub(super) buf_a: Vec<f32>,
    pub(super) buf_b: Vec<f32>,
    // Tiled: the assembled tiled-body output, then a full-frame ping-pong buffer
    // for the segmented remainder. Empty for Sequential.
    pub(super) out_buf: Vec<f32>,
    // Tiled with anything past the leading body: the other full-frame ping-pong
    // buffer.
    pub(super) tail_buf: Vec<f32>,
    // Tiled + debayer, serial: pooled working buffer for the serial kernel.
    // Empty otherwise (the parallel path allocates one per worker instead).
    pub(super) demosaic_scratch: Vec<f32>,
    // Tiled: how `steps[prefix_hi..]` is split into whole-frame / retiled passes
    // over `out_buf`/`tail_buf`. Empty for `Sequential` or a bare tiled body.
    pub(super) tail_phases: Vec<TailPhase>,
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

    /// How many independently-tiled passes the plan runs. Zero when the runner
    /// is not tiled at all; one for a plain tiled chain; more when a geometric
    /// op (a rotation, a mid-chain crop, a [`resize`](super::Op::ResizeToFit))
    /// splits the chain and the pixel run on its far side retiles — e.g. a resize
    /// in the middle of a chain tiles on both sides and reports `2`.
    pub fn tiled_pass_count(&self) -> usize {
        if !self.is_tiled() {
            return 0;
        }
        1 + self
            .tail_phases
            .iter()
            .filter(|p| matches!(p, TailPhase::Tiled { .. }))
            .count()
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
                let Runner {
                    ref steps,
                    ref coeffs,
                    ref specs,
                    ref out_spec,
                    ref mut out_buf,
                    ref mut tail_buf,
                    ref mut buf_a,
                    ref mut buf_b,
                    ref mut demosaic_scratch,
                    ref tail_phases,
                    ..
                } = *self;

                let body_in = specs[prefix_lo].clone();
                let body_out = specs[prefix_hi].clone();
                let frame_stride = specs[0].width * specs[0].bpp()?;
                let frame_off = in_off_y * frame_stride + in_off_x * body_in.bpp()?;

                // Leading body: tiled straight from the (crop-folded) raw frame.
                fill_tiled(
                    &steps[prefix_lo..prefix_hi],
                    coeffs,
                    &body_in,
                    &body_out,
                    raw,
                    frame_stride,
                    frame_off,
                    out_buf,
                    buf_a,
                    buf_b,
                    demosaic_scratch,
                    TileGeom {
                        tile_rows,
                        tile_cols,
                        halo,
                        even,
                    },
                    parallel,
                )?;

                // The remainder: ping-pong the two full-frame buffers, each
                // phase either a whole-frame `run_chain` or its own tiled pass.
                // `out_holds` tracks which buffer has the live image.
                let mut out_holds = true;
                for phase in tail_phases {
                    let (src, dst): (&mut [f32], &mut [f32]) = if out_holds {
                        (out_buf.as_mut_slice(), tail_buf.as_mut_slice())
                    } else {
                        (tail_buf.as_mut_slice(), out_buf.as_mut_slice())
                    };
                    match *phase {
                        TailPhase::Whole { lo, hi } => {
                            let s = &specs[lo];
                            let ended_in_src = run_chain(
                                &steps[lo..hi],
                                coeffs,
                                src,
                                dst,
                                s.width,
                                s.height,
                                Demosaic::Alloc,
                            )?;
                            if !ended_in_src {
                                out_holds = !out_holds;
                            }
                        }
                        TailPhase::Tiled {
                            lo,
                            hi,
                            tile_rows,
                            tile_cols,
                            parallel,
                        } => {
                            let seg_in = specs[lo].clone();
                            let seg_out = specs[hi].clone();
                            let seg_stride = seg_in.width * seg_in.bpp()?;
                            fill_tiled(
                                &steps[lo..hi],
                                coeffs,
                                &seg_in,
                                &seg_out,
                                cast_slice::<f32, u8>(src),
                                seg_stride,
                                0,
                                dst,
                                buf_a,
                                buf_b,
                                demosaic_scratch,
                                TileGeom {
                                    tile_rows,
                                    tile_cols,
                                    halo: 0,
                                    even: false,
                                },
                                parallel,
                            )?;
                            out_holds = !out_holds;
                        }
                    }
                }

                let buf = if out_holds { out_buf } else { tail_buf };
                view(buf, out_spec)
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
    /// With an [`Op::Roi`](super::Op::Roi) first in the chain this reproduces a
    /// "blit region into a pre-sized buffer" — the old `CopyRoi::copy_to`.
    pub fn run_into<F: Frame + ?Sized>(
        &mut self,
        frame: &F,
        dest: &mut DynamicImageOwned,
    ) -> Result<(), PipelineError> {
        let spec = self.out_spec.clone();
        let fits = dest.width() == spec.width
            && dest.height() == spec.height
            && dest.channels() == spec.cspace.channels()
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
