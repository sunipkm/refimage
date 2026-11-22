//! Rice compression for the `RICE_1` tile-compression type.
//!
//! Ported from the encode and decode paths of cfitsio's public-domain `ricecomp.c`
//! (R. White, STScI): 8-bit (`fits_rcomp_byte`), 16-bit (`fits_rcomp_short`) and
//! 32-bit (`fits_rcomp`), the last used for quantized floating-point tiles. The block
//! size is fixed at 32 (`ZVAL1 = BLOCKSIZE = 32`).
//!
//! The decoder is retained alongside the encoder so unit tests can round-trip a tile
//! without a reference reader; it is otherwise unused until a `FitsRead` exists.
#![allow(dead_code)] // `decode_*` are exercised only by the round-trip tests for now.
#![allow(clippy::int_plus_one)] // kept 1:1 with cfitsio `ricecomp.c`

const NBLOCK: usize = 32;

const NONZERO_COUNT: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut i = 1usize;
    while i < 256 {
        let mut v = i;
        let mut n = 0i32;
        while v != 0 {
            v >>= 1;
            n += 1;
        }
        t[i] = n;
        i += 1;
    }
    t
};

/// Parameters that vary with pixel width.
struct Params {
    fsbits: i32,
    fsmax: i32,
    /// bits for the raw first pixel and for direct (high-entropy) coding.
    nbits: i32,
    /// `& mask` applied to the block sum before choosing the split (`u16`/`u8`/`u32`
    /// cast in C).
    psum_mask: u32,
    /// truncate a difference to the pixel width (`d as i16 as i32`, etc.).
    trunc_bits: u32,
}

const BYTE: Params = Params {
    fsbits: 3,
    fsmax: 6,
    nbits: 8,
    psum_mask: 0xff,
    trunc_bits: 8,
};
const SHORT: Params = Params {
    fsbits: 4,
    fsmax: 14,
    nbits: 16,
    psum_mask: 0xffff,
    trunc_bits: 16,
};
const INT: Params = Params {
    fsbits: 5,
    fsmax: 25,
    nbits: 32,
    psum_mask: u32::MAX,
    trunc_bits: 32,
};

/// Rice-encode 8-bit (FITS-native) pixel values.
pub(super) fn encode_byte(a: &[i8]) -> Vec<u8> {
    encode(&a.iter().map(|&x| x as i32).collect::<Vec<_>>(), &BYTE)
}

/// Rice-encode 16-bit (FITS-native) pixel values.
pub(super) fn encode_short(a: &[i16]) -> Vec<u8> {
    encode(&a.iter().map(|&x| x as i32).collect::<Vec<_>>(), &SHORT)
}

/// Rice-encode 32-bit values (quantized floating-point tiles).
pub(super) fn encode_int(a: &[i32]) -> Vec<u8> {
    encode(a, &INT)
}

/// Rice-decode into 8-bit values.
pub(super) fn decode_byte(c: &[u8], nx: usize) -> Vec<u8> {
    decode(c, nx, &BYTE).into_iter().map(|v| v as u8).collect()
}

/// Rice-decode into 16-bit values.
pub(super) fn decode_short(c: &[u8], nx: usize) -> Vec<u16> {
    decode(c, nx, &SHORT)
        .into_iter()
        .map(|v| v as u16)
        .collect()
}

/// Rice-decode into 32-bit values.
pub(super) fn decode_int(c: &[u8], nx: usize) -> Vec<i32> {
    decode(c, nx, &INT).into_iter().map(|v| v as i32).collect()
}

/// Truncate `d` to `bits` low bits, sign-extended back to `i32` (C's `(short) d` etc.).
fn trunc(d: i32, bits: u32) -> i32 {
    if bits >= 32 {
        d
    } else {
        let sh = 32 - bits;
        (d << sh) >> sh
    }
}

fn mask(n: i32) -> u32 {
    if n <= 0 {
        0
    } else if n >= 32 {
        u32::MAX
    } else {
        (1u32 << n) - 1
    }
}

/// `x << k`, yielding 0 when `k >= 32` (C's `int` shift wraps; the callers only care
/// about the low 8 bits they subsequently mask).
fn shl(x: u32, k: i32) -> u32 {
    if !(0..32).contains(&k) {
        0
    } else {
        x << k
    }
}

struct BitWriter {
    buf: Vec<u8>,
    acc: u32,
    free: i32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            free: 8,
        }
    }

    /// `output_nbits` from cfitsio (`n <= 32`).
    fn put(&mut self, bits: u32, mut n: i32) {
        if self.free + n > 32 {
            // Large n: flush the top `free` bits first (0 < free <= 8).
            self.acc = shl(self.acc, self.free);
            self.acc |= (bits >> (n - self.free)) & mask(self.free);
            self.buf.push((self.acc & 0xff) as u8);
            n -= self.free;
            self.free = 8;
        }
        self.acc = shl(self.acc, n);
        self.acc |= bits & mask(n);
        self.free -= n;
        while self.free <= 0 {
            self.buf
                .push((self.acc.wrapping_shr((-self.free) as u32) & 0xff) as u8);
            self.free += 8;
        }
    }

    fn done(&mut self) {
        if self.free < 8 {
            self.buf.push((shl(self.acc, self.free) & 0xff) as u8);
        }
    }
}

fn encode(a: &[i32], p: &Params) -> Vec<u8> {
    let nx = a.len();
    let mut w = BitWriter::new();
    if nx == 0 {
        return w.buf;
    }

    w.put(a[0] as u32 & mask(p.nbits), p.nbits);
    let mut lastpix = a[0];

    let bbits = 1i32 << p.fsbits;
    let mut diff = [0u32; NBLOCK];

    let mut i = 0;
    while i < nx {
        let thisblock = (nx - i).min(NBLOCK);
        let mut pixelsum = 0.0f64;
        for j in 0..thisblock {
            let nextpix = a[i + j];
            let pdiff = trunc(nextpix.wrapping_sub(lastpix), p.trunc_bits);
            let shifted = pdiff.wrapping_shl(1);
            let mapped = if pdiff < 0 { !shifted } else { shifted } as u32;
            diff[j] = mapped;
            pixelsum += mapped as f64;
            lastpix = nextpix;
        }

        let mut dpsum = (pixelsum - (thisblock / 2) as f64 - 1.0) / thisblock as f64;
        if dpsum < 0.0 {
            dpsum = 0.0;
        }
        let mut psum = (dpsum as u32 & p.psum_mask) >> 1;
        let mut fs = 0i32;
        while psum > 0 {
            fs += 1;
            psum >>= 1;
        }

        if fs >= p.fsmax {
            w.put((p.fsmax + 1) as u32, p.fsbits);
            for &d in &diff[..thisblock] {
                w.put(d, bbits);
            }
        } else if fs == 0 && pixelsum == 0.0 {
            w.put(0, p.fsbits);
        } else {
            w.put((fs + 1) as u32, p.fsbits);
            let fsmask = mask(fs);
            for &v in &diff[..thisblock] {
                let mut top = (v >> fs) as i32;
                if w.free >= top + 1 {
                    w.acc = shl(w.acc, top + 1) | 1;
                    w.free -= top + 1;
                } else {
                    w.acc = shl(w.acc, w.free);
                    w.buf.push((w.acc & 0xff) as u8);
                    top -= w.free;
                    while top >= 8 {
                        w.buf.push(0);
                        top -= 8;
                    }
                    w.acc = 1;
                    w.free = 7 - top;
                }
                if fs > 0 {
                    w.acc = shl(w.acc, fs) | (v & fsmask);
                    w.free -= fs;
                    while w.free <= 0 {
                        w.buf
                            .push((w.acc.wrapping_shr((-w.free) as u32) & 0xff) as u8);
                        w.free += 8;
                    }
                }
            }
        }
        i += NBLOCK;
    }
    w.done();
    w.buf
}

fn decode(c: &[u8], nx: usize, p: &Params) -> Vec<u32> {
    let mut out = vec![0u32; nx];
    if nx == 0 {
        return out;
    }
    let bbits = 1i32 << p.fsbits;

    let head = (p.nbits / 8) as usize;
    let mut lastpix: u32 = 0;
    for &byte in &c[..head] {
        lastpix = (lastpix << 8) | byte as u32;
    }
    let mut pos = head;

    let mut b = c[pos] as u32;
    pos += 1;
    let mut nbits = 8i32;

    let mut i = 0usize;
    while i < nx {
        nbits -= p.fsbits;
        while nbits < 0 {
            b = (b << 8) | c[pos] as u32;
            pos += 1;
            nbits += 8;
        }
        let fs = (b >> nbits) as i32 - 1;
        b &= mask(nbits);

        let imax = (i + NBLOCK).min(nx);
        if fs < 0 {
            for slot in out.iter_mut().take(imax).skip(i) {
                *slot = lastpix;
            }
            i = imax;
        } else if fs == p.fsmax {
            while i < imax {
                let mut k = bbits - nbits;
                let mut d = shl(b, k);
                k -= 8;
                while k >= 0 {
                    b = c[pos] as u32;
                    pos += 1;
                    d |= shl(b, k);
                    k -= 8;
                }
                if nbits > 0 {
                    b = c[pos] as u32;
                    pos += 1;
                    d |= b >> (-k);
                    b &= mask(nbits);
                } else {
                    b = 0;
                }
                let d = if d & 1 == 0 { d >> 1 } else { !(d >> 1) };
                lastpix = d.wrapping_add(lastpix);
                out[i] = lastpix;
                i += 1;
            }
        } else {
            while i < imax {
                while b == 0 {
                    nbits += 8;
                    b = c[pos] as u32;
                    pos += 1;
                }
                let nzero = nbits - NONZERO_COUNT[b as usize];
                nbits -= nzero + 1;
                b ^= 1u32 << nbits;
                nbits -= fs;
                while nbits < 0 {
                    b = (b << 8) | c[pos] as u32;
                    pos += 1;
                    nbits += 8;
                }
                let d = (shl(nzero as u32, fs)) | (b >> nbits);
                b &= mask(nbits);
                let d = if d & 1 == 0 { d >> 1 } else { !(d >> 1) };
                lastpix = d.wrapping_add(lastpix);
                out[i] = lastpix;
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt_byte(v: &[i8]) {
        let dec = decode_byte(&encode_byte(v), v.len());
        let want: Vec<u8> = v.iter().map(|&x| x as u8).collect();
        assert_eq!(dec, want, "input {v:?}");
    }

    fn rt_short(v: &[i16]) {
        let dec = decode_short(&encode_short(v), v.len());
        let want: Vec<u16> = v.iter().map(|&x| x as u16).collect();
        assert_eq!(dec, want, "input {v:?}");
    }

    fn rt_int(v: &[i32]) {
        let dec = decode_int(&encode_int(v), v.len());
        assert_eq!(dec, v, "input {v:?}");
    }

    #[test]
    fn byte_roundtrips() {
        rt_byte(&[]);
        rt_byte(&[42]);
        rt_byte(&[0i8; 80]);
        rt_byte(&(-128..127i8).collect::<Vec<_>>());
        rt_byte(&(0..300i32).map(|i| (i * 97) as i8).collect::<Vec<_>>());
    }

    #[test]
    fn short_roundtrips() {
        rt_short(&[]);
        rt_short(&[0]);
        rt_short(&[7; 1]);
        rt_short(&[0i16; 100]);
        rt_short(&(0..200i16).collect::<Vec<_>>());
        rt_short(&(0..500i32).map(|i| (i * 277) as i16).collect::<Vec<_>>());
        rt_short(&[i16::MIN, i16::MAX, 0, -1, 1, i16::MIN, i16::MAX]);
    }

    #[test]
    fn int_roundtrips() {
        rt_int(&[]);
        rt_int(&[123_456]);
        rt_int(&[0i32; 100]);
        rt_int(&(0..200i32).collect::<Vec<_>>());
        rt_int(
            &(0..500i32)
                .map(|i| i.wrapping_mul(1_234_577))
                .collect::<Vec<_>>(),
        );
        rt_int(&[
            i32::MIN,
            i32::MAX,
            0,
            -1,
            1,
            i32::MIN,
            i32::MAX,
            -12345,
            67890,
        ]);
    }
}
