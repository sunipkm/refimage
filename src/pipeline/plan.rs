//! Compile-time planning: lowering [`Op`]s to [`Step`]s, the per-buffer liveness
//! walk, and the tiling math that turns a [`Strategy`] into concrete tile sizes.

use crate::{BayerPattern, ColorSpace, DemosaicMethod, PixelType};

use super::{ImageSpec, Op, PipelineError, ResizeFilter, Strategy};

/// The per-step execution plan plus every intermediate shape.
pub(super) struct Plan {
    pub(super) steps: Vec<Step>,
    pub(super) coeffs: Vec<Box<[f64]>>,
    pub(super) specs: Vec<ImageSpec>, // len == steps.len() + 1
    pub(super) out_spec: ImageSpec,
}

impl Plan {
    pub(super) fn build(ops: &[Op], input: &ImageSpec) -> Result<Self, PipelineError> {
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
                in_channels: cur.cspace.channels(),
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

    /// Walk `steps[lo..hi]` the way [`run_chain`](super::exec::run_chain) does —
    /// buffer `A` holds `specs[lo]`, swapping steps flip the home buffer, in-place
    /// steps keep it — and return `(max bytes ever in A, max bytes ever in B)`
    /// under `cell` (a per-spec size: full-frame for `Sequential`, padded-tile for
    /// `Tiled`).
    pub(super) fn buf_caps(
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
    pub(super) fn max_bytes(&self, lo: usize, hi: usize) -> Result<usize, PipelineError> {
        let mut m = 0;
        for s in &self.specs[lo..=hi] {
            m = m.max(s.bytes()?);
        }
        Ok(m.max(1))
    }
}

/// f32 elements needed to hold `bytes` (rounded up), at least 1.
pub(super) fn f32_cap(bytes: usize) -> usize {
    bytes.div_ceil(4).max(1)
}

/// Auto tile height aims for roughly this working-set size per band.
const TILE_TARGET_BYTES: usize = 256 * 1024;

/// Tiling parameters resolved from a [`Strategy`] against concrete dimensions.
pub(super) struct ResolvedTile {
    pub(super) tile_rows: usize,
    pub(super) tile_cols: usize,
    pub(super) halo: usize,
    pub(super) even: bool,
    pub(super) parallel: bool,
}

/// One phase of the post-body remainder: a maximal run of steps that either
/// tiles on its own or runs as a single whole-frame [`run_chain`](super::exec::run_chain)
/// pass. Phases ping-pong between the two full-frame buffers.
#[derive(Debug, Clone, Copy)]
pub(super) enum TailPhase {
    /// Tile `steps[lo..hi]` — a pixel-op run past a geometric op, retiled
    /// against its own (post-transform) dimensions. Halo is always 0 here (the
    /// only haloed pixel op, debayer, can only appear in the leading body).
    Tiled {
        lo: usize,
        hi: usize,
        tile_rows: usize,
        tile_cols: usize,
        parallel: bool,
    },
    /// Run `steps[lo..hi]` whole-frame: the connecting geometric ops, and any
    /// pixel run too small to tile. Adjacent `Whole` phases are merged.
    Whole { lo: usize, hi: usize },
}

/// Append `steps[lo..hi]` as a [`TailPhase::Whole`], extending the previous one
/// when they abut.
fn push_whole(phases: &mut Vec<TailPhase>, lo: usize, hi: usize) {
    if let Some(TailPhase::Whole { hi: prev_hi, .. }) = phases.last_mut()
        && *prev_hi == lo {
            *prev_hi = hi;
            return;
        }
    phases.push(TailPhase::Whole { lo, hi });
}

/// Split `steps[start..]` into phases at every geometric op: a geometric run is
/// whole-frame; a pixel run tiles when [`resolve_exec`] says it is worth it
/// (against that run's own dimensions), else it is whole-frame too.
pub(super) fn build_tail_phases(
    steps: &[Step],
    specs: &[ImageSpec],
    strategy: Strategy,
    start: usize,
) -> Vec<TailPhase> {
    let n = steps.len();
    let mut phases = Vec::new();
    let mut i = start;
    while i < n {
        let geo = steps[i].kind.is_geometric();
        let mut j = i + 1;
        while j < n && steps[j].kind.is_geometric() == geo {
            j += 1;
        }
        let s = &specs[i];
        match (geo, resolve_exec(strategy, s.width, s.height, 0, false)) {
            (false, Some(rt)) => phases.push(TailPhase::Tiled {
                lo: i,
                hi: j,
                tile_rows: rt.tile_rows,
                tile_cols: rt.tile_cols,
                parallel: rt.parallel,
            }),
            _ => push_whole(&mut phases, i, j),
        }
        i = j;
    }
    phases
}

/// Turn a [`Strategy`] + the *tiled body's* dimensions into concrete tiling
/// parameters, or `None` when tiling cannot help and the body should run
/// sequentially.
pub(super) fn resolve_exec(
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

/// How a [`Runner`](super::Runner) executes its steps.
#[derive(Debug, Clone, Copy)]
pub(super) enum Exec {
    /// Run every step over the whole frame, in order.
    Sequential,
    /// Tile the leading pixel-op run `steps[prefix_lo..prefix_hi]` into `out_buf`,
    /// reading the frame from `(in_off_x, in_off_y)` (folded leading crops).
    /// `steps[prefix_hi..]` is then run as [`Runner::tail_phases`](super::Runner)
    /// — whole-frame geometric ops and retiled pixel runs, ping-ponging
    /// `out_buf`/`tail_buf`.
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
pub(super) struct Step {
    pub(super) kind: StepKind,
    pub(super) in_pt: PixelType,
    pub(super) out_pt: PixelType,
    pub(super) in_channels: u8,
    pub(super) bayer: Option<BayerPattern>,
    pub(super) coeff_idx: usize,
    pub(super) luma_identity: bool,
}

impl Step {
    /// Does this step read one buffer and write the other? Debayer and every
    /// geometric op do; luma, scale, and convert rewrite their buffer in place.
    pub(super) fn swaps(&self) -> bool {
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
pub(super) enum StepKind {
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

impl StepKind {
    /// Relocates or resamples pixels (crop, ROI, flip, rotate, resize), as
    /// opposed to the per-pixel kinds (debayer, luma, scale, convert). A
    /// geometric step ends one tiled segment and starts the next.
    pub(super) fn is_geometric(&self) -> bool {
        matches!(
            self,
            StepKind::Crop { .. }
                | StepKind::Roi { .. }
                | StepKind::Flip { .. }
                | StepKind::Rot90 { .. }
                | StepKind::Resize { .. }
        )
    }
}
