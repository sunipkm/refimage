//! Spline interpolation at arbitrary coordinates, for [`Op::Warp`](super::Op::Warp) and
//! [`Op::MapCoordinates`](super::Op::MapCoordinates).
//!
//! The numerics are ported from the `imwarp` crate and reproduce
//! `scipy.ndimage.map_coordinates` (orders 0, 1 and 3, with SciPy's B-spline prefilter and
//! boundary modes) and `skimage.transform.warp` with a homography matrix. Unlike `imwarp`
//! they run on the pipeline's own typed buffers: samples are read as `f64`, every
//! interpolation is evaluated in `f64`, and the result is rounded (integer types) and
//! saturated back into the image's pixel type. The cubic prefilter's coefficient plane
//! lives in scratch the [`Runner`](super::Runner) allocates at compile time, so a warp
//! does not allocate per frame.

mod boundary;
mod spline;

use bytemuck::{cast_slice, cast_slice_mut};
use serde::{Deserialize, Serialize};

use crate::{PixelStor, PixelType, U10, U12, U14};

pub use boundary::BoundaryMode;

use super::PipelineError;
use super::spec::pixel_size;

/// Spline interpolation order for [`Interpolation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum InterpOrder {
    /// Nearest neighbour (SciPy order 0).
    Nearest,
    /// Bilinear (SciPy order 1).
    #[default]
    Linear,
    /// Bicubic B-spline (SciPy order 3), prefiltered by default.
    Cubic,
}

impl InterpOrder {
    /// From SciPy's numeric order. Orders 2, 4 and 5 are not implemented.
    pub fn from_u8(order: u8) -> Result<Self, PipelineError> {
        match order {
            0 => Ok(Self::Nearest),
            1 => Ok(Self::Linear),
            3 => Ok(Self::Cubic),
            other => Err(PipelineError::UnsupportedOrder(other)),
        }
    }

    /// SciPy's numeric order.
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Nearest => 0,
            Self::Linear => 1,
            Self::Cubic => 3,
        }
    }

    const fn degree(self) -> usize {
        self.as_u8() as usize
    }
}

/// How [`Op::Warp`](super::Op::Warp) / [`Op::MapCoordinates`](super::Op::MapCoordinates)
/// interpolate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interpolation {
    /// Interpolation order.
    pub order: InterpOrder,
    /// Behaviour beyond the edge of the image.
    pub mode: BoundaryMode,
    /// Value used outside the image in [`BoundaryMode::Constant`], in raw stored units
    /// (e.g. `255.0` is white on a `u8` image, `1.0` on an `f32` one). Saturated into the
    /// pixel type like any other output sample.
    pub cval: f64,
    /// Convert samples to spline coefficients first. Only meaningful for
    /// [`InterpOrder::Cubic`]; defaults to `true` for it, like SciPy.
    pub prefilter: bool,
}

impl Interpolation {
    /// Interpolate with `order`, treating the outside as `mode`, filling with `cval` for
    /// [`BoundaryMode::Constant`]. Cubic order prefilters by default, like SciPy.
    pub fn new(order: InterpOrder, mode: BoundaryMode, cval: f64) -> Self {
        Self {
            order,
            mode,
            cval,
            prefilter: order == InterpOrder::Cubic,
        }
    }

    /// Turn the spline prefilter on or off (only cubic order uses it).
    #[must_use]
    pub fn with_prefilter(mut self, prefilter: bool) -> Self {
        self.prefilter = prefilter;
        self
    }

    fn filtering(&self) -> bool {
        self.order == InterpOrder::Cubic && self.prefilter
    }

    /// Edge samples padded on each side before prefiltering.
    fn pad(&self) -> usize {
        if self.filtering() && self.mode == BoundaryMode::Nearest {
            NEAREST_PAD
        } else {
            0
        }
    }
}

impl Default for Interpolation {
    /// Linear, constant boundary, fill 0.
    fn default() -> Self {
        Self::new(InterpOrder::Linear, BoundaryMode::Constant, 0.0)
    }
}

/// Number of samples edge-padded before cubic prefiltering in `nearest` mode.
const NEAREST_PAD: usize = 12;

/// `f64` scratch one warp over a `sw * sh`, `ch`-channel image needs: the padded
/// coefficient plane plus one filter line, or nothing when there is no prefilter.
pub(super) fn scratch_len(interp: &Interpolation, sw: usize, sh: usize, ch: usize) -> usize {
    if !interp.filtering() {
        return 0;
    }
    let pad = interp.pad();
    let (ny, nx) = (sh + 2 * pad, sw + 2 * pad);
    ny * nx * ch + ny.max(nx)
}

/// Where each output pixel samples the input, as `(row, column)`.
#[derive(Debug, Clone, Copy)]
pub(super) enum Coords<'a> {
    /// Row-major 3x3 matrix taking `(column, row, 1)` of an output pixel to
    /// `(column, row, w)` in the input (scikit-image's convention).
    Homography(&'a [f64]),
    /// One `(row, column)` per output pixel, row-major over the output.
    Map { rows: &'a [f64], cols: &'a [f64] },
}

impl Coords<'_> {
    #[inline]
    fn at(&self, x: usize, y: usize, ow: usize) -> (f64, f64) {
        match *self {
            Coords::Homography(m) => {
                let (x, y) = (x as f64, y as f64);
                let w = m[6] * x + m[7] * y + m[8];
                let src_x = (m[0] * x + m[1] * y + m[2]) / w;
                let src_y = (m[3] * x + m[4] * y + m[5]) / w;
                (src_y, src_x)
            }
            Coords::Map { rows, cols } => {
                let i = y * ow + x;
                (rows[i], cols[i])
            }
        }
    }
}

/// Interpolation footprint along one axis: up to four taps.
struct Taps {
    index: [usize; 4],
    weight: [f64; 4],
    len: usize,
}

/// Footprint for coordinate `coord` along an axis of length `len` (of the possibly padded
/// sample plane), or `None` when the point is outside the image and takes the fill value.
fn taps(coord: f64, len: usize, order: InterpOrder, mode: BoundaryMode, pad: usize) -> Option<Taps> {
    // NaN takes the fill value as in SciPy; so does +-inf (a homography's `w == 0`),
    // which SciPy would turn into garbage indices.
    if !coord.is_finite() {
        return None;
    }
    let mut cc = coord + pad as f64;
    if mode != BoundaryMode::Nearest {
        cc = mode.map_coordinate(cc, len);
    }
    if cc <= -1.0 && mode != BoundaryMode::Nearest {
        return None;
    }
    let degree = order.degree();
    let start = if degree % 2 == 1 {
        cc.floor()
    } else {
        (cc + 0.5).floor()
    } as i64
        - (degree / 2) as i64;

    let weight = weights(cc, order);
    let spline_mode = mode.spline_mode();
    let mut taps = Taps {
        index: [0; 4],
        weight,
        len: degree + 1,
    };
    for (ll, slot) in taps.index.iter_mut().enumerate().take(degree + 1) {
        let idx = start + ll as i64;
        *slot = if (0..len as i64).contains(&idx) {
            idx as usize
        } else {
            spline_mode.map_coordinate(idx as f64, len).max(0.0) as usize
        };
    }
    Some(taps)
}

/// B-spline weights of the taps for coordinate `cc`.
fn weights(cc: f64, order: InterpOrder) -> [f64; 4] {
    let mut w = [0.0; 4];
    match order {
        InterpOrder::Nearest => w[0] = 1.0,
        InterpOrder::Linear => {
            let x = cc - cc.floor();
            w[0] = 1.0 - x;
            w[1] = 1.0 - w[0];
        }
        InterpOrder::Cubic => {
            let x = cc - cc.floor();
            let (y, z) = (x, 1.0 - x);
            w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
            w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
            w[0] = z * z * z / 6.0;
            w[3] = 1.0 - w[0] - w[1] - w[2];
        }
    }
    w
}

/// A sample plane the taps index: `get((y * nx + x) * ch + c)`.
struct Plane<S> {
    get: S,
    ny: usize,
    nx: usize,
    ch: usize,
    pad: usize,
}

impl<S: Fn(usize) -> f64> Plane<S> {
    /// Interpolate every channel at `(r, c)`, handing each to `put(channel, value)`.
    #[inline]
    fn eval(&self, r: f64, c: f64, interp: &Interpolation, mut put: impl FnMut(usize, f64)) {
        let (order, mode) = (interp.order, interp.mode);
        match (
            taps(r, self.ny, order, mode, self.pad),
            taps(c, self.nx, order, mode, self.pad),
        ) {
            (Some(ty), Some(tx)) => {
                for ch in 0..self.ch {
                    let mut total = 0.0;
                    for a in 0..ty.len {
                        let row = ty.index[a] * self.nx;
                        for b in 0..tx.len {
                            // Coefficient times the weight of each axis, in that order.
                            total += (self.get)((row + tx.index[b]) * self.ch + ch)
                                * ty.weight[a]
                                * tx.weight[b];
                        }
                    }
                    put(ch, total);
                }
            }
            _ => (0..self.ch).for_each(|ch| put(ch, interp.cval)),
        }
    }
}

/// Interpolate the `sw * sh`, `ch`-interleaved image of type `pt` in `src` at the
/// coordinates of every pixel of the `ow * oh` output in `dst`. `scratch` must hold
/// [`scratch_len`] values. Rows fan out over rayon when `parallel` (and the `rayon`
/// feature) is on.
#[allow(clippy::too_many_arguments)]
pub(super) fn geo_warp(
    src: &[f32],
    dst: &mut [f32],
    sw: usize,
    sh: usize,
    ch: usize,
    pt: PixelType,
    ow: usize,
    oh: usize,
    interp: &Interpolation,
    coords: Coords<'_>,
    scratch: &mut [f64],
    parallel: bool,
) -> Result<(), PipelineError> {
    pixel_size(pt)?;
    let g = Geom {
        sw,
        sh,
        ch,
        ow,
        oh,
        parallel,
    };
    macro_rules! typed {
        ($t:ty) => {
            warp_typed::<$t>(cast_slice(src), cast_slice_mut(dst), g, interp, coords, scratch)
        };
    }
    match pt {
        PixelType::U8 => typed!(u8),
        // `U10`/`U12`/`U14`: cubic overshoot saturates against the sensor's true range.
        PixelType::U10 => typed!(U10),
        PixelType::U12 => typed!(U12),
        PixelType::U14 => typed!(U14),
        PixelType::U16 => typed!(u16),
        PixelType::F32 => warp_typed::<f32>(src, dst, g, interp, coords, scratch),
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Geom {
    sw: usize,
    sh: usize,
    ch: usize,
    ow: usize,
    oh: usize,
    parallel: bool,
}

fn warp_typed<T: PixelStor>(
    src: &[T],
    dst: &mut [T],
    g: Geom,
    interp: &Interpolation,
    coords: Coords<'_>,
    scratch: &mut [f64],
) {
    let Geom { sw, sh, ch, .. } = g;
    if interp.filtering() {
        let pad = interp.pad();
        let (ny, nx) = (sh + 2 * pad, sw + 2 * pad);
        let (plane, line) = scratch[..scratch_len(interp, sw, sh, ch)].split_at_mut(ny * nx * ch);
        // Edge-pad (a no-op copy when `pad == 0`) into the coefficient plane.
        for y in 0..ny {
            let sy = y.saturating_sub(pad).min(sh - 1);
            for x in 0..nx {
                let sx = x.saturating_sub(pad).min(sw - 1);
                let (d, s) = ((y * nx + x) * ch, (sy * sw + sx) * ch);
                for c in 0..ch {
                    plane[d + c] = src[s + c].to_f64();
                }
            }
        }
        spline::spline_filter(plane, ny, nx, ch, interp.mode.filter_extension(), line);
        let plane: &[f64] = plane;
        let p = Plane {
            get: |i: usize| plane[i],
            ny,
            nx,
            ch,
            pad,
        };
        render(dst, &p, g, interp, coords);
    } else {
        let p = Plane {
            get: |i: usize| src[i].to_f64(),
            ny: sh,
            nx: sw,
            ch,
            pad: 0,
        };
        render(dst, &p, g, interp, coords);
    }
}

fn render<T: PixelStor, S: Fn(usize) -> f64 + Sync>(
    dst: &mut [T],
    plane: &Plane<S>,
    g: Geom,
    interp: &Interpolation,
    coords: Coords<'_>,
) {
    let Geom { ch, ow, oh, .. } = g;
    let round = T::PIXEL_TYPE != PixelType::F32;
    let out_row = ow * ch;
    let do_row = |y: usize, row: &mut [T]| {
        for x in 0..ow {
            let (r, c) = coords.at(x, y, ow);
            let px = &mut row[x * ch..(x + 1) * ch];
            plane.eval(r, c, interp, |k, v| {
                px[k] = T::from_f64(if round { v.round() } else { v });
            });
        }
    };
    let out = &mut dst[..oh * out_row];

    #[cfg(feature = "rayon")]
    if g.parallel {
        use rayon::prelude::*;
        out.par_chunks_mut(out_row)
            .enumerate()
            .for_each(|(y, row)| do_row(y, row));
        return;
    }
    #[cfg(not(feature = "rayon"))]
    let _ = g.parallel;
    for (y, row) in out.chunks_mut(out_row).enumerate() {
        do_row(y, row);
    }
}

#[cfg(test)]
mod tests {
    //! `map_coordinates` against SciPy (`tests/data/scipy.json`, produced by `imwarp`'s
    //! `tools/make_goldens.py`), evaluated in `f64` exactly as the pipeline does before it
    //! narrows to the pixel type.
    use super::*;
    use serde_json::Value;

    const MODES: [(&str, BoundaryMode); 5] = [
        ("constant", BoundaryMode::Constant),
        ("nearest", BoundaryMode::Nearest),
        ("mirror", BoundaryMode::Mirror),
        ("reflect", BoundaryMode::Reflect),
        ("wrap", BoundaryMode::Wrap),
    ];
    const ORDERS: [(u8, InterpOrder); 3] = [
        (0, InterpOrder::Nearest),
        (1, InterpOrder::Linear),
        (3, InterpOrder::Cubic),
    ];
    const RTOL: f64 = 1e-11;
    const ATOL: f64 = 1e-11;

    fn numbers(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("an array")
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    fn assert_close(label: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= ATOL + RTOL * w.abs();
            assert!(ok, "{label}[{i}]: got {g}, want {w}");
        }
    }

    /// The same path `warp_typed` takes, but kept in `f64` end to end.
    fn sample(data: &[f64], ny: usize, nx: usize, coords: &[f64], interp: &Interpolation) -> Vec<f64> {
        let n = coords.len() / 2;
        let (rows, cols) = coords.split_at(n);
        let mut scratch = vec![0.0; scratch_len(interp, nx, ny, 1)];
        let mut out = vec![0.0; n];
        let mut eval = |p: &Plane<&dyn Fn(usize) -> f64>| {
            for i in 0..n {
                p.eval(rows[i], cols[i], interp, |_, v| out[i] = v);
            }
        };
        if interp.filtering() {
            let pad = interp.pad();
            let (py, px) = (ny + 2 * pad, nx + 2 * pad);
            let (plane, line) = scratch.split_at_mut(py * px);
            for y in 0..py {
                for x in 0..px {
                    plane[y * px + x] =
                        data[y.saturating_sub(pad).min(ny - 1) * nx + x.saturating_sub(pad).min(nx - 1)];
                }
            }
            spline::spline_filter(plane, py, px, 1, interp.mode.filter_extension(), line);
            let plane: &[f64] = plane;
            let get = |i: usize| plane[i];
            eval(&Plane { get: &get, ny: py, nx: px, ch: 1, pad });
        } else {
            let get = |i: usize| data[i];
            eval(&Plane { get: &get, ny, nx, ch: 1, pad: 0 });
        }
        out
    }

    #[test]
    fn matches_scipy_for_every_order_mode_and_shape() {
        // Read at run time: `tests/` is not part of the published crate, so a test run
        // from the crates.io source skips this instead of failing to compile.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/scipy.json");
        let Ok(text) = std::fs::read_to_string(path) else {
            eprintln!("skipping: SciPy goldens not found at {path}");
            return;
        };
        let golden: Value = serde_json::from_str(&text).expect("golden");
        let mut compared = 0;
        for case in golden["cases"].as_array().expect("cases") {
            let shape = numbers(&case["shape"]);
            let (ny, nx) = (shape[0] as usize, shape[1] as usize);
            let data = numbers(&case["data"]);
            let coords = numbers(&case["coords"]);
            let expected = &case["expected"];
            for (o, order) in ORDERS {
                for (name, mode) in MODES {
                    let got = sample(&data, ny, nx, &coords, &Interpolation::new(order, mode, -7.5));
                    let key = format!("o{o}_{name}");
                    assert_close(&format!("{key} {ny}x{nx}"), &got, &numbers(&expected[&key]));
                    compared += 1;
                }
            }
            let nan_fill = Interpolation::new(InterpOrder::Linear, BoundaryMode::Constant, f64::NAN);
            assert_close(
                "nan cval",
                &sample(&data, ny, nx, &coords, &nan_fill),
                &numbers(&expected["o1_constant_nan"]),
            );
            let unfiltered = Interpolation::new(InterpOrder::Linear, BoundaryMode::Constant, 0.0)
                .with_prefilter(false);
            assert_close(
                "o1 noprefilter",
                &sample(&data, ny, nx, &coords, &unfiltered),
                &numbers(&expected["o1_noprefilter_constant"]),
            );
            let cubic_raw = Interpolation::new(InterpOrder::Cubic, BoundaryMode::Mirror, 0.0)
                .with_prefilter(false);
            assert_close(
                "o3 noprefilter",
                &sample(&data, ny, nx, &coords, &cubic_raw),
                &numbers(&expected["o3_noprefilter_mirror"]),
            );
        }
        assert_eq!(compared, 75);
    }

    #[test]
    fn non_finite_coordinates_take_the_fill_value() {
        let data = [1.0; 9];
        let interp = Interpolation::new(InterpOrder::Linear, BoundaryMode::Reflect, -2.0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(sample(&data, 3, 3, &[bad, 0.0], &interp), [-2.0]);
        }
    }

    #[test]
    fn orders_and_modes_parse() {
        assert!(InterpOrder::from_u8(2).is_err());
        assert_eq!(InterpOrder::from_u8(3).unwrap(), InterpOrder::Cubic);
        assert_eq!("edge".parse::<BoundaryMode>().unwrap(), BoundaryMode::Nearest);
        assert_eq!("symmetric".parse::<BoundaryMode>().unwrap(), BoundaryMode::Mirror);
        assert!("bogus".parse::<BoundaryMode>().is_err());
    }
}
