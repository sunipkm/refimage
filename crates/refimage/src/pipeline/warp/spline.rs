//! Cubic B-spline prefilter (poles and boundary initialisation as in SciPy), ported from
//! `imwarp` to work in place on an interleaved plane with a caller-supplied line buffer.

use super::boundary::FilterExtension;

/// The single pole of the cubic B-spline filter, `sqrt(3) - 2`, with SciPy's literal.
#[allow(clippy::excessive_precision)]
const POLE: f64 = -0.267_949_192_431_122_706_472_553_658_494_127_633;

/// Convert the samples of an `ny * nx`, `ch`-interleaved plane to cubic B-spline
/// coefficients, along both axes (columns first, then rows, as `imwarp` does), channel by
/// channel. `line` must hold at least `max(ny, nx)` values; nothing is allocated.
pub(super) fn spline_filter(
    plane: &mut [f64],
    ny: usize,
    nx: usize,
    ch: usize,
    extension: FilterExtension,
    line: &mut [f64],
) {
    // Vertical lines: one per (column, channel).
    if ny >= 2 {
        let line = &mut line[..ny];
        for x in 0..nx {
            for c in 0..ch {
                for (y, v) in line.iter_mut().enumerate() {
                    *v = plane[(y * nx + x) * ch + c];
                }
                filter_line(line, extension);
                for (y, v) in line.iter().enumerate() {
                    plane[(y * nx + x) * ch + c] = *v;
                }
            }
        }
    }
    // Horizontal lines: one per (row, channel).
    if nx >= 2 {
        let line = &mut line[..nx];
        for y in 0..ny {
            let row = &mut plane[y * nx * ch..(y + 1) * nx * ch];
            for c in 0..ch {
                for (x, v) in line.iter_mut().enumerate() {
                    *v = row[x * ch + c];
                }
                filter_line(line, extension);
                for (x, v) in line.iter().enumerate() {
                    row[x * ch + c] = *v;
                }
            }
        }
    }
}

pub(super) fn filter_line(c: &mut [f64], extension: FilterExtension) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = POLE;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    match extension {
        FilterExtension::Mirror => init_causal_mirror(c, z),
        FilterExtension::Reflect => init_causal_reflect(c, z),
    }
    // Causal pass: each sample adds a multiple of the already-updated one before it.
    let mut previous = c[0];
    for v in &mut c[1..] {
        *v += z * previous;
        previous = *v;
    }
    match extension {
        FilterExtension::Mirror => c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0),
        FilterExtension::Reflect => c[n - 1] *= z / (z - 1.0),
    }
    // Anti-causal pass, from the end: each sample uses the already-updated one after it.
    let mut next = c[n - 1];
    for v in c[..n - 1].iter_mut().rev() {
        *v = z * (next - *v);
        next = *v;
    }
}

fn init_causal_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n_1 = z.powi(i32::try_from(n - 1).unwrap_or(i32::MAX));
    let (first, rest) = c.split_at_mut(1);
    let last = rest[rest.len() - 1];
    first[0] += z_n_1 * last;
    // Samples 1..n-1 paired with their mirror images n-2..=1.
    let inner = &rest[..rest.len() - 1];
    for (&near, &far) in inner.iter().zip(inner.iter().rev()) {
        first[0] += z_i * (near + z_n_1 * far);
        z_i *= z;
    }
    first[0] /= 1.0 - z_n_1 * z_n_1;
}

fn init_causal_reflect(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n = z.powi(i32::try_from(n).unwrap_or(i32::MAX));
    let (first, rest) = c.split_at_mut(1);
    let origin = first[0];
    let mut sum = origin + z_n * rest[rest.len() - 1];
    // Samples 1..n paired with their mirror images n-2..=0: the last pair reaches back to
    // the (not yet updated) first sample.
    let mirrored = rest[..rest.len() - 1]
        .iter()
        .rev()
        .copied()
        .chain(std::iter::once(origin));
    for (&near, far) in rest.iter().zip(mirrored) {
        sum += z_i * (near + z_n * far);
        z_i *= z;
    }
    first[0] += sum * z / (1.0 - z_n * z_n);
}
