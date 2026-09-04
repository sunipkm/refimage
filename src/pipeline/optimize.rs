//! [`Pipeline::optimize`](super::Pipeline::optimize) — peephole simplification of
//! an [`Op`] list before [`compile`](super::Pipeline::compile).
//!
//! Every rewrite here is *output-preserving* for any input the original chain
//! accepts: it only drops no-ops, folds a run of coordinate transforms into the
//! shortest equivalent one, and merges stages that compose exactly (nested
//! crops). Arithmetic stages ([`Op::Scale`], [`Op::ScalePixels`],
//! [`Op::Convert`]) are **not** fused across each other — the intermediate
//! rounding and saturation between them is observable — only exact adjacent
//! duplicates are dropped.

use super::Op;

/// Run every peephole rewrite to a fixed point. See
/// [`Pipeline::optimize`](super::Pipeline::optimize).
pub(super) fn simplify(mut ops: Vec<Op>) -> Vec<Op> {
    // Each pass is idempotent and non-growing, so this settles in a couple of
    // rounds; the cap only guards against a future non-monotone rewrite.
    for _ in 0..16 {
        let next = simplify_once(&ops);
        if next == ops {
            return ops;
        }
        ops = next;
    }
    ops
}

fn simplify_once(ops: &[Op]) -> Vec<Op> {
    let no_nops: Vec<Op> = ops.iter().filter(|o| !matches!(o, Op::Nop)).cloned().collect();
    let folded = coalesce_dihedral(no_nops);
    peephole(folded)
}

// --- Dihedral (flip / 90° rotation) folding ------------------------------------

/// A signed 2x2 integer matrix acting on image coordinates (`+x` right, `+y`
/// down): `x' = a·x + b·y`, `y' = c·x + d·y`. The eight elements of the square's
/// symmetry group `D4` are exactly the flips and quarter-turn rotations, and they
/// compose by matrix multiplication — so any run of them collapses to one.
#[derive(Clone, Copy, PartialEq, Eq)]
struct D4 {
    a: i8,
    b: i8,
    c: i8,
    d: i8,
}

impl D4 {
    /// The transform of a single geometric [`Op`], or `None` if it is not a
    /// flip / 90° rotation.
    fn of(op: &Op) -> Option<D4> {
        Some(match op {
            Op::FlipHorizontal => D4 {
                a: -1,
                b: 0,
                c: 0,
                d: 1,
            },
            Op::FlipVertical => D4 {
                a: 1,
                b: 0,
                c: 0,
                d: -1,
            },
            Op::Rotate90 => D4 {
                a: 0,
                b: -1,
                c: 1,
                d: 0,
            },
            Op::Rotate180 => D4 {
                a: -1,
                b: 0,
                c: 0,
                d: -1,
            },
            Op::Rotate270 => D4 {
                a: 0,
                b: 1,
                c: -1,
                d: 0,
            },
            _ => return None,
        })
    }

    /// Apply `self` first, then `rhs` (i.e. the matrix product `rhs · self`).
    fn then(self, rhs: D4) -> D4 {
        D4 {
            a: rhs.a * self.a + rhs.b * self.c,
            b: rhs.a * self.b + rhs.b * self.d,
            c: rhs.c * self.a + rhs.d * self.c,
            d: rhs.c * self.b + rhs.d * self.d,
        }
    }

    /// The shortest op sequence for this transform: 0 ops (identity), 1 op (a
    /// flip or single rotation), or 2 ops (the two diagonal mirrors, which have
    /// no dedicated [`Op`]).
    fn to_ops(self) -> Vec<Op> {
        match (self.a, self.b, self.c, self.d) {
            (1, 0, 0, 1) => vec![],
            (-1, 0, 0, 1) => vec![Op::FlipHorizontal],
            (1, 0, 0, -1) => vec![Op::FlipVertical],
            (-1, 0, 0, -1) => vec![Op::Rotate180],
            (0, -1, 1, 0) => vec![Op::Rotate90],
            (0, 1, -1, 0) => vec![Op::Rotate270],
            // Transpose (mirror on the main diagonal): flip, then quarter-turn.
            (0, 1, 1, 0) => vec![Op::FlipVertical, Op::Rotate90],
            // Anti-transpose (mirror on the other diagonal).
            (0, -1, -1, 0) => vec![Op::FlipHorizontal, Op::Rotate90],
            _ => unreachable!("D4 is closed under composition"),
        }
    }
}

/// Replace every maximal run of two or more flips / 90° rotations with the
/// shortest equivalent sequence.
fn coalesce_dihedral(ops: Vec<Op>) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        let Some(first) = D4::of(&ops[i]) else {
            out.push(ops[i].clone());
            i += 1;
            continue;
        };
        let start = i;
        let mut m = first;
        i += 1;
        while i < ops.len() {
            match D4::of(&ops[i]) {
                Some(d) => {
                    m = m.then(d);
                    i += 1;
                }
                None => break,
            }
        }
        if i - start >= 2 {
            out.extend(m.to_ops());
        } else {
            out.push(ops[start].clone());
        }
    }
    out
}

// --- Adjacent-pair peepholes --------------------------------------------------

fn is_luma(op: &Op) -> bool {
    matches!(op, Op::ToLuma | Op::ToLumaCustom(_))
}

/// What to do with `op` given the op already emitted before it.
enum Act {
    /// Replace the previous op with this one.
    Fuse(Op),
    /// Drop `op` (it is a no-op after the previous one).
    Drop,
    /// Keep `op` as-is.
    Keep,
}

fn decide(prev: Option<&Op>, op: &Op) -> Act {
    match (prev, op) {
        // Nested crops compose exactly: the inner origin is relative to the
        // outer one, and the inner size is the final size.
        (
            Some(Op::Crop {
                x: ox, y: oy, ..
            }),
            Op::Crop {
                x: ix,
                y: iy,
                width,
                height,
            },
        ) => Act::Fuse(Op::Crop {
            x: ox + ix,
            y: oy + iy,
            width: *width,
            height: *height,
        }),
        // A second luminance pass runs on already-gray data — a passthrough.
        (Some(p), _) if is_luma(p) && is_luma(op) => Act::Drop,
        // Re-converting to the type just produced is a no-op.
        (Some(Op::Convert(a)), Op::Convert(b)) if a == b => Act::Drop,
        _ => Act::Keep,
    }
}

fn peephole(ops: Vec<Op>) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    for op in ops {
        match decide(out.last(), &op) {
            Act::Fuse(n) => {
                out.pop();
                out.push(n);
            }
            Act::Drop => {}
            Act::Keep => out.push(op),
        }
    }
    out
}
