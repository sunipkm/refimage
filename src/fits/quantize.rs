//! Lossy quantization of floating-point tiles for `RICE_1` (the `ZQUANTIZ`
//! convention). Ported from cfitsio's `quantize.c` / `imcompress.c`.
//!
//! Only `SUBTRACTIVE_DITHER_1` is implemented. The scale (`ZSCALE`) is derived once
//! from a 3rd-order MAD noise estimate over the whole image; the zero point (`ZZERO`)
//! is per-tile. The subtractive-dither pseudo-random sequence is reproduced exactly so
//! `astropy.io.fits` reconstructs the same values.

use std::sync::OnceLock;

/// `N_RANDOM` in cfitsio — do not change.
const N_RANDOM: usize = 10_000;
/// Reserved integer codes at the bottom of the range (`N_RESERVED_VALUES`).
const N_RESERVED: f64 = 10.0;

/// The seeded random sequence from `fits_init_randoms`.
///
/// `seed_{i+1} = (16807 * seed_i) mod 2147483647`, `seed_0 = 1`, stored as
/// `f32(seed / 2147483647)`. The 10000th seed is `1043618065` when correct.
fn rand_table() -> &'static [f32; N_RANDOM] {
    static TABLE: OnceLock<[f32; N_RANDOM]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let a = 16807.0f64;
        let m = 2_147_483_647.0f64;
        let mut seed = 1.0f64;
        let mut t = [0.0f32; N_RANDOM];
        for slot in t.iter_mut() {
            let temp = a * seed;
            seed = temp - m * (temp / m).trunc();
            *slot = (seed / m) as f32;
        }
        debug_assert_eq!(seed as i64, 1_043_618_065);
        t
    })
}

/// C's `NINT`: round half away from zero.
fn nint(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

/// The dither index walk: given a starting `(iseed, nextrand)`, yield successive
/// `fits_rand_value[nextrand]` and advance.
struct Dither {
    iseed: usize,
    nextrand: usize,
}

impl Dither {
    /// `row` is 0-based `tile_index + ZDITHER0 - 1` (cfitsio's `irow - 1`).
    fn new(row: usize) -> Self {
        let table = rand_table();
        let iseed = row % N_RANDOM;
        let nextrand = (table[iseed] * 500.0) as usize;
        Self { iseed, nextrand }
    }

    fn value(&self) -> f64 {
        rand_table()[self.nextrand] as f64
    }

    fn advance(&mut self) {
        self.nextrand += 1;
        if self.nextrand == N_RANDOM {
            self.iseed += 1;
            if self.iseed == N_RANDOM {
                self.iseed = 0;
            }
            self.nextrand = (rand_table()[self.iseed] * 500.0) as usize;
        }
    }
}

/// `ZDITHER0`: a hash of the first tile's raw bytes, in `1..=10000` (cfitsio's
/// "negative request" method — deterministic for a given image).
pub(super) fn dither_seed(first_tile: &[f32]) -> u32 {
    let sum: u64 = first_tile
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .map(u64::from)
        .sum();
    (sum % 10_000) as u32 + 1
}

/// 3rd-order MAD noise, plus min/max, over `data` laid out as `nx` columns by `ny`
/// rows (`FnNoise3_float` with `nullcheck = 0`).
pub(super) fn noise3(data: &[f32], mut nx: usize, mut ny: usize) -> (f64, f64, f64) {
    if nx < 5 {
        nx *= ny;
        ny = 1;
    }
    let mut minv = f32::INFINITY;
    let mut maxv = f32::NEG_INFINITY;

    if nx < 5 {
        for &v in data {
            minv = minv.min(v);
            maxv = maxv.max(v);
        }
        return (minv as f64, maxv as f64, 0.0);
    }

    let mut row_medians: Vec<f64> = Vec::with_capacity(ny);
    let mut diffs: Vec<f32> = Vec::with_capacity(nx);

    for row in data.chunks_exact(nx).take(ny) {
        for &v in row {
            minv = minv.min(v);
            maxv = maxv.max(v);
        }
        let (mut v1, mut v2, mut v3, mut v4) = (row[0], row[1], row[2], row[3]);
        diffs.clear();
        for &v5 in &row[4..] {
            if !(v1 == v2 && v2 == v3 && v3 == v4 && v4 == v5) {
                diffs.push(((2.0 * v3 as f64) - v1 as f64 - v5 as f64).abs() as f32);
            }
            v1 = v2;
            v2 = v3;
            v3 = v4;
            v4 = v5;
        }
        match diffs.len() {
            0 => {}
            1 => row_medians.push(diffs[0] as f64),
            _ => row_medians.push(quick_select(&mut diffs) as f64),
        }
    }

    let noise = match row_medians.len() {
        0 => 0.0,
        1 => row_medians[0],
        n => {
            row_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (row_medians[(n - 1) / 2] + row_medians[n / 2]) / 2.0
        }
    };
    (minv as f64, maxv as f64, 0.605_269_7 * noise)
}

/// Median via Hoare's quickselect (cfitsio's `quick_select_float`), in place.
fn quick_select(arr: &mut [f32]) -> f32 {
    let n = arr.len();
    let (mut low, mut high) = (0usize, n - 1);
    let median = n / 2;
    loop {
        if high <= low {
            return arr[median];
        }
        if high == low + 1 {
            if arr[low] > arr[high] {
                arr.swap(low, high);
            }
            return arr[median];
        }
        let middle = (low + high) / 2;
        if arr[middle] > arr[high] {
            arr.swap(middle, high);
        }
        if arr[low] > arr[high] {
            arr.swap(low, high);
        }
        if arr[middle] > arr[low] {
            arr.swap(middle, low);
        }
        arr.swap(middle, low + 1);

        let mut ll = low + 1;
        let mut hh = high;
        loop {
            loop {
                ll += 1;
                if arr[low] <= arr[ll] {
                    break;
                }
            }
            loop {
                hh -= 1;
                if arr[hh] <= arr[low] {
                    break;
                }
            }
            if hh < ll {
                break;
            }
            arr.swap(ll, hh);
        }
        arr.swap(low, hh);
        if hh <= median {
            low = ll;
        }
        if hh >= median {
            high = hh - 1;
        }
    }
}

/// The result of quantizing one tile.
pub(super) struct QuantTile {
    pub idata: Vec<i32>,
    pub zzero: f64,
}

/// Quantize `fdata` (one tile) with the given global `delta` (= `ZSCALE`).
///
/// `tile_index` is 0-based; `zdither0` is the header `ZDITHER0`. The zero point is
/// snapped to a multiple of `delta` near the tile minimum (cfitsio's `iqfactor`
/// fudge), which keeps the scaling stable across repeated compress/decompress cycles.
pub(super) fn quantize_tile(
    fdata: &[f32],
    delta: f64,
    tile_index: usize,
    zdither0: u32,
) -> QuantTile {
    let minval = fdata.iter().copied().fold(f32::INFINITY, f32::min) as f64;

    let iqfactor = (minval / delta + 0.5) as i64;
    let zzero = iqfactor as f64 * delta;

    let row = tile_index + zdither0 as usize - 1;
    let mut dither = Dither::new(row);

    let idata = fdata
        .iter()
        .map(|&f| {
            let q = nint((f as f64 - zzero) / delta + dither.value() - 0.5);
            dither.advance();
            q
        })
        .collect();

    QuantTile { idata, zzero }
}

/// Choose the global quantization step from a whole-image noise estimate.
/// `qlevel` is `ZVAL` for `QUANTIZE_LEVEL` (cfitsio default 4).
pub(super) fn global_delta(data: &[f32], nx: usize, ny: usize, qlevel: f64) -> f64 {
    let (minv, maxv, noise) = noise3(data, nx, ny);
    let delta = if noise > 0.0 {
        noise / qlevel
    } else if maxv > minv {
        // constant-noise (or tiny) image: spread the range over most of the int axis
        (maxv - minv) / (2.0 * i32::MAX as f64 - N_RESERVED - 1.0)
    } else {
        1.0
    };
    delta.max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_table_matches_cfitsio_checksum() {
        // Forces the debug_assert in `rand_table` and checks a couple of values.
        let t = rand_table();
        assert!((0.0..1.0).contains(&t[0]));
        assert_eq!(t.len(), N_RANDOM);
    }

    #[test]
    fn quantize_dequantize_is_close() {
        let n = 64usize;
        let fdata: Vec<f32> = (0..n * n)
            .map(|i| ((i as f32) * 0.013).sin() * 100.0 + 500.0)
            .collect();
        let delta = global_delta(&fdata, n, n, 4.0);
        assert!(delta > 0.0);

        // Quantize row-tiles, then dequantize with the same dither and compare.
        for (t, row) in fdata.chunks_exact(n).enumerate() {
            let zdither0 = 137u32;
            let q = quantize_tile(row, delta, t, zdither0);

            let seed_row = t + zdither0 as usize - 1;
            let mut d = Dither::new(seed_row);
            for (k, &iv) in q.idata.iter().enumerate() {
                let back = (iv as f64 - d.value() + 0.5) * delta + q.zzero;
                d.advance();
                assert!(
                    (back - row[k] as f64).abs() <= delta,
                    "row {t} px {k}: {back} vs {}",
                    row[k]
                );
            }
        }
    }
}
