//! `HCOMPRESS_1` tile compression.
//!
//! A faithful port of the encoder in cfitsio's public-domain `fits_hcompress.c`
//! (R. White, STScI): the H-transform (`htrans`), digitization (`digitize`) and the
//! binary-quadtree bit-plane coder (`encode` / `doencode` / `qtree_encode`). Only the
//! 32-bit integer path is ported — floating-point tiles are quantized to integers
//! first (see [`quantize`](super::quantize)), exactly as cfitsio does.
//!
//! `scale = 0` (lossless) is the only mode this crate exposes. The whole image (one
//! plane) is compressed as a single tile.
//!
//! The matching decoder lives in `astropy` / `cfitsio`; round-trips are verified in the
//! integration tests rather than here.
#![allow(unused_assignments)] // pointer bookkeeping kept 1:1 with cfitsio `fits_hcompress.c`
#![allow(clippy::manual_div_ceil)] // `(n + 1) / 2` kept verbatim from `fits_hcompress.c`

/// Huffman code values for the 16 quadtree nybble symbols.
const CODE: [u32; 16] = [
    0x3e, 0x00, 0x01, 0x08, 0x02, 0x09, 0x1a, 0x1b, 0x03, 0x1c, 0x0a, 0x1d, 0x0b, 0x1e, 0x3f, 0x0c,
];
/// Bit length of each Huffman code above.
const NCODE: [i32; 16] = [6, 3, 3, 4, 3, 4, 5, 5, 3, 5, 4, 5, 4, 5, 6, 4];

/// The two-byte stream signature (`code_magic`).
const MAGIC: [u8; 2] = [0xDD, 0x99];

/// Compress one plane (`width * height` row-major `i32` values, FITS-native) with the
/// H-compress algorithm. `a` is consumed as scratch (the transform runs in place).
pub(super) fn compress(a: &mut [i32], width: usize, height: usize, scale: i32) -> Vec<u8> {
    debug_assert_eq!(a.len(), width * height);
    // cfitsio names the fast (column / NAXIS1) axis `ny` and the slow axis `nx`.
    let nx = height;
    let ny = width;
    htrans(a, nx, ny);
    digitize(a, nx, ny, scale);
    let mut out = Out::new();
    encode(&mut out, a, nx, ny, scale);
    out.buf
}

/// `log2` of `n`, rounded up to the next power of two — reproduces cfitsio's
/// `(int)(log((float) n)/log(2.0) + 0.5)` plus its correction step.
fn log2_ceil(n: usize) -> i32 {
    let mut log2n = (((n as f32) as f64).ln() / std::f64::consts::LN_2 + 0.5) as i32;
    if n > (1usize << log2n.max(0)) {
        log2n += 1;
    }
    log2n
}

/* ------------------------------------------------------------------------- */
/* htrans.c                                                                  */
/* ------------------------------------------------------------------------- */

fn htrans(a: &mut [i32], nx: usize, ny: usize) {
    let nmax = nx.max(ny);
    let log2n = log2_ceil(nmax);
    let mut tmp = vec![0i32; (nmax + 1) / 2];

    let mut shift = 0i32;
    let mut mask = -2i32;
    let mut mask2 = mask.wrapping_shl(1);
    let mut prnd = 1i32;
    let mut prnd2 = prnd.wrapping_shl(1);
    let mut nrnd2 = prnd2.wrapping_sub(1);

    let mut nxtop = nx;
    let mut nytop = ny;

    let round = |h: i32, p: i32, m: i32| (if h >= 0 { h.wrapping_add(p) } else { h }) & m;
    let round0 = |h: i32, p2: i32, n2: i32, m2: i32| {
        (if h >= 0 {
            h.wrapping_add(p2)
        } else {
            h.wrapping_add(n2)
        }) & m2
    };

    for _ in 0..log2n {
        let oddx = nxtop % 2;
        let oddy = nytop % 2;

        let mut i = 0usize;
        while i + oddx < nxtop {
            let mut s00 = i * ny;
            let mut s10 = s00 + ny;
            let mut j = 0usize;
            while j + oddy < nytop {
                let (p00, p01, p10, p11) = (a[s00], a[s00 + 1], a[s10], a[s10 + 1]);
                let h0 = p11.wrapping_add(p10).wrapping_add(p01).wrapping_add(p00) >> shift;
                let hx = p11.wrapping_add(p10).wrapping_sub(p01).wrapping_sub(p00) >> shift;
                let hy = p11.wrapping_sub(p10).wrapping_add(p01).wrapping_sub(p00) >> shift;
                let hc = p11.wrapping_sub(p10).wrapping_sub(p01).wrapping_add(p00) >> shift;
                a[s10 + 1] = hc;
                a[s10] = round(hx, prnd, mask);
                a[s00 + 1] = round(hy, prnd, mask);
                a[s00] = round0(h0, prnd2, nrnd2, mask2);
                s00 += 2;
                s10 += 2;
                j += 2;
            }
            if oddy == 1 {
                let sh = (1 - shift) as u32;
                let h0 = a[s10].wrapping_add(a[s00]).wrapping_shl(sh);
                let hx = a[s10].wrapping_sub(a[s00]).wrapping_shl(sh);
                a[s10] = round(hx, prnd, mask);
                a[s00] = round0(h0, prnd2, nrnd2, mask2);
                s00 += 1;
                s10 += 1;
            }
            i += 2;
        }
        if oddx == 1 {
            let mut s00 = i * ny;
            let mut j = 0usize;
            while j + oddy < nytop {
                let sh = (1 - shift) as u32;
                let h0 = a[s00 + 1].wrapping_add(a[s00]).wrapping_shl(sh);
                let hy = a[s00 + 1].wrapping_sub(a[s00]).wrapping_shl(sh);
                a[s00 + 1] = round(hy, prnd, mask);
                a[s00] = round0(h0, prnd2, nrnd2, mask2);
                s00 += 2;
                j += 2;
            }
            if oddy == 1 {
                let h0 = a[s00].wrapping_shl((2 - shift) as u32);
                a[s00] = round0(h0, prnd2, nrnd2, mask2);
            }
        }

        for row in 0..nxtop {
            shuffle(a, ny * row, nytop, 1, &mut tmp);
        }
        for col in 0..nytop {
            shuffle(a, col, nxtop, ny, &mut tmp);
        }

        nxtop = (nxtop + 1) >> 1;
        nytop = (nytop + 1) >> 1;
        shift = 1;
        mask = mask2;
        prnd = prnd2;
        mask2 = mask2.wrapping_shl(1);
        prnd2 = prnd2.wrapping_shl(1);
        nrnd2 = prnd2.wrapping_sub(1);
    }
}

/// Group coefficients by order: even-indexed elements to the front, odd to the back.
fn shuffle(a: &mut [i32], off: usize, n: usize, n2: usize, tmp: &mut [i32]) {
    let mut pt = 0usize;
    let mut p1 = off + n2;
    let mut i = 1usize;
    while i < n {
        tmp[pt] = a[p1];
        pt += 1;
        p1 += n2 + n2;
        i += 2;
    }

    p1 = off + n2;
    let mut p2 = off + n2 + n2;
    i = 2;
    while i < n {
        a[p1] = a[p2];
        p1 += n2;
        p2 += n2 + n2;
        i += 2;
    }

    pt = 0;
    i = 1;
    while i < n {
        a[p1] = tmp[pt];
        p1 += n2;
        pt += 1;
        i += 2;
    }
}

/* ------------------------------------------------------------------------- */
/* digitize.c                                                                */
/* ------------------------------------------------------------------------- */

fn digitize(a: &mut [i32], nx: usize, ny: usize, scale: i32) {
    if scale <= 1 {
        return;
    }
    let d = (scale + 1) / 2 - 1;
    for p in a[..nx * ny].iter_mut() {
        let v = if *p > 0 {
            p.wrapping_add(d)
        } else {
            p.wrapping_sub(d)
        };
        *p = v / scale;
    }
}

/* ------------------------------------------------------------------------- */
/* encode.c / bit_output.c                                                   */
/* ------------------------------------------------------------------------- */

/// The output byte stream plus the MSB-first bit buffer (`bit_output.c`'s `buffer2` /
/// `bits_to_go2`).
struct Out {
    buf: Vec<u8>,
    buffer2: u32,
    bits_to_go2: i32,
}

impl Out {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            buffer2: 0,
            bits_to_go2: 8,
        }
    }

    fn qwrite(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn writeint(&mut self, a: i32) {
        self.qwrite(&a.to_be_bytes());
    }

    fn writelonglong(&mut self, a: i64) {
        self.qwrite(&a.to_be_bytes());
    }

    fn start_outputing_bits(&mut self) {
        self.buffer2 = 0;
        self.bits_to_go2 = 8;
    }

    /// `output_nbits` — `n` must be `<= 8`.
    fn output_nbits(&mut self, bits: u32, n: i32) {
        const MASK: [u32; 9] = [0, 1, 3, 7, 15, 31, 63, 127, 255];
        self.buffer2 = self.buffer2.wrapping_shl(n as u32) | (bits & MASK[n as usize]);
        self.bits_to_go2 -= n;
        if self.bits_to_go2 <= 0 {
            self.buf
                .push((self.buffer2 >> (-self.bits_to_go2) as u32 & 0xff) as u8);
            self.bits_to_go2 += 8;
        }
    }

    fn output_nybble(&mut self, bits: u32) {
        self.buffer2 = self.buffer2.wrapping_shl(4) | (bits & 15);
        self.bits_to_go2 -= 4;
        if self.bits_to_go2 <= 0 {
            self.buf
                .push((self.buffer2 >> (-self.bits_to_go2) as u32 & 0xff) as u8);
            self.bits_to_go2 += 8;
        }
    }

    fn output_huffman(&mut self, c: usize) {
        self.output_nbits(CODE[c], NCODE[c]);
    }

    /// Pack the low 4 bits of every `array` element (`output_nnybble`).
    fn output_nnybble(&mut self, n: usize, array: &[u8]) {
        if n == 1 {
            self.output_nybble(array[0] as u32);
            return;
        }
        let mut kk = 0usize;
        if self.bits_to_go2 <= 4 {
            self.output_nybble(array[0] as u32);
            kk += 1;
            if n == 2 {
                self.output_nybble(array[1] as u32);
                return;
            }
        }
        let shift = (8 - self.bits_to_go2) as u32;
        let jj = (n - kk) / 2;
        if self.bits_to_go2 == 8 {
            self.buffer2 = 0;
            for _ in 0..jj {
                self.buf
                    .push(((array[kk] & 15) << 4) | (array[kk + 1] & 15));
                kk += 2;
            }
        } else {
            for _ in 0..jj {
                self.buffer2 = self.buffer2.wrapping_shl(8)
                    | (((array[kk] & 15) as u32) << 4)
                    | (array[kk + 1] & 15) as u32;
                kk += 2;
                self.buf.push((self.buffer2 >> shift & 0xff) as u8);
            }
        }
        if kk != n {
            self.output_nybble(array[n - 1] as u32);
        }
    }

    fn done_outputing_bits(&mut self) {
        if self.bits_to_go2 < 8 {
            self.buf
                .push((self.buffer2.wrapping_shl(self.bits_to_go2 as u32) & 0xff) as u8);
        }
    }
}

fn encode(out: &mut Out, a: &mut [i32], nx: usize, ny: usize, scale: i32) {
    let nel = nx * ny;

    out.qwrite(&MAGIC);
    out.writeint(nx as i32);
    out.writeint(ny as i32);
    out.writeint(scale);
    // Sum of all pixels — the one value that does not compress well.
    out.writelonglong(a[0] as i64);
    a[0] = 0;

    // Sign bits, 8 per byte; replace each element by its absolute value.
    let mut signbits = vec![0u8; (nel + 7) / 8];
    let mut nsign = 0usize;
    let mut bits_to_go = 8i32;
    for v in a[..nel].iter_mut() {
        if *v > 0 {
            signbits[nsign] <<= 1;
            bits_to_go -= 1;
        } else if *v < 0 {
            signbits[nsign] = (signbits[nsign] << 1) | 1;
            bits_to_go -= 1;
            *v = v.wrapping_neg();
        }
        if bits_to_go == 0 {
            bits_to_go = 8;
            nsign += 1;
        }
    }
    if bits_to_go != 8 {
        signbits[nsign] <<= bits_to_go;
        nsign += 1;
    }

    // Maximum absolute value, and hence bit-plane count, per quadrant.
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;
    let mut vmax = [0i32; 3];
    let (mut col, mut row) = (0usize, 0usize);
    for &v in &a[..nel] {
        let q = (col >= ny2) as usize + (row >= nx2) as usize;
        if vmax[q] < v {
            vmax[q] = v;
        }
        col += 1;
        if col >= ny {
            col = 0;
            row += 1;
        }
    }
    let mut nbitplanes = [0u8; 3];
    for q in 0..3 {
        let mut v = vmax[q];
        while v > 0 {
            v >>= 1;
            nbitplanes[q] += 1;
        }
    }
    out.qwrite(&nbitplanes);

    doencode(out, a, nx, ny, nbitplanes);

    if nsign > 0 {
        out.qwrite(&signbits[..nsign]);
    }
}

/* ------------------------------------------------------------------------- */
/* doencode.c                                                                */
/* ------------------------------------------------------------------------- */

fn doencode(out: &mut Out, a: &[i32], nx: usize, ny: usize, nbitplanes: [u8; 3]) {
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;
    out.start_outputing_bits();
    qtree_encode(out, a, 0, ny, nx2, ny2, nbitplanes[0] as i32);
    qtree_encode(out, a, ny2, ny, nx2, ny / 2, nbitplanes[1] as i32);
    qtree_encode(out, a, ny * nx2, ny, nx / 2, ny2, nbitplanes[1] as i32);
    qtree_encode(
        out,
        a,
        ny * nx2 + ny2,
        ny,
        nx / 2,
        ny / 2,
        nbitplanes[2] as i32,
    );
    out.output_nybble(0); // EOF symbol
    out.done_outputing_bits();
}

/* ------------------------------------------------------------------------- */
/* qtree_encode.c                                                            */
/* ------------------------------------------------------------------------- */

/// Bit buffer used while Huffman-coding quadtree codes (`bitbuffer` / `bits_to_go3`).
struct QTree {
    bitbuffer: u32,
    bits_to_go3: i32,
}

impl QTree {
    /// Copy non-zero Huffman codes from `a` into `buffer`; returns `true` if the
    /// quadtree is expanding the data and the caller should fall back to a bitmap.
    fn bufcopy(
        &mut self,
        a: &[u8],
        n: usize,
        buffer: &mut [u8],
        b: &mut usize,
        bmax: usize,
    ) -> bool {
        for &code in &a[..n] {
            if code != 0 {
                let c = code as usize;
                self.bitbuffer |= CODE[c] << self.bits_to_go3;
                self.bits_to_go3 += NCODE[c];
                if self.bits_to_go3 >= 8 {
                    buffer[*b] = (self.bitbuffer & 0xff) as u8;
                    *b += 1;
                    if *b >= bmax {
                        return true;
                    }
                    self.bitbuffer >>= 8;
                    self.bits_to_go3 -= 8;
                }
            }
        }
        false
    }
}

fn qtree_encode(
    out: &mut Out,
    a: &[i32],
    off: usize,
    n: usize,
    nqx: usize,
    nqy: usize,
    nbitplanes: i32,
) {
    let nqmax = nqx.max(nqy);
    let log2n = log2_ceil(nqmax);

    let nqx2 = (nqx + 1) / 2;
    let nqy2 = (nqy + 1) / 2;
    let bmax = ((nqx2 * nqy2 + 1) / 2).max(1);

    let mut scratch = vec![0u8; (nqx2 * nqy2).max(1)];
    let mut buffer = vec![0u8; bmax];

    for bit in (0..nbitplanes).rev() {
        let mut b = 0usize;
        let mut q = QTree {
            bitbuffer: 0,
            bits_to_go3: 0,
        };

        qtree_onebit(a, off, n, nqx, nqy, &mut scratch, bit);
        let mut nx = (nqx + 1) >> 1;
        let mut ny = (nqy + 1) >> 1;

        let mut bailed = q.bufcopy(&scratch, nx * ny, &mut buffer, &mut b, bmax);
        if !bailed {
            for _ in 1..log2n {
                qtree_reduce(&mut scratch, ny, nx, ny);
                nx = (nx + 1) >> 1;
                ny = (ny + 1) >> 1;
                if q.bufcopy(&scratch, nx * ny, &mut buffer, &mut b, bmax) {
                    bailed = true;
                    break;
                }
            }
        }

        if bailed {
            write_bdirect(out, a, off, n, nqx, nqy, bit);
            continue;
        }

        out.output_nybble(0xF);
        if q.bits_to_go3 > 0 {
            out.output_nbits(q.bitbuffer & ((1u32 << q.bits_to_go3) - 1), q.bits_to_go3);
        } else if b == 0 {
            out.output_huffman(0);
        }
        if b > 0 {
            for i in (0..b).rev() {
                out.output_nbits(buffer[i] as u32, 8);
            }
        }
    }
}

/// First quadtree reduction on bit `bit` of `a`: four source bits packed per nybble.
fn qtree_onebit(a: &[i32], off: usize, n: usize, nx: usize, ny: usize, b: &mut [u8], bit: i32) {
    let bit = bit as u32;
    let b0 = 1u32.wrapping_shl(bit);
    let b1 = b0.wrapping_shl(1);
    let b2 = b0.wrapping_shl(2);
    let b3 = b0.wrapping_shl(3);
    let g = |idx: usize| a[off + idx] as u32;

    let mut k = 0usize;
    let mut i = 0usize;
    while i + 1 < nx {
        let mut s00 = n * i;
        let mut s10 = s00 + n;
        let mut j = 0usize;
        while j + 1 < ny {
            b[k] = ((g(s10 + 1) & b0)
                | (g(s10).wrapping_shl(1) & b1)
                | (g(s00 + 1).wrapping_shl(2) & b2)
                | (g(s00).wrapping_shl(3) & b3))
                .wrapping_shr(bit) as u8;
            k += 1;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny {
            b[k] = ((g(s10).wrapping_shl(1) & b1) | (g(s00).wrapping_shl(3) & b3)).wrapping_shr(bit)
                as u8;
            k += 1;
        }
        i += 2;
    }
    if i < nx {
        let mut s00 = n * i;
        let mut j = 0usize;
        while j + 1 < ny {
            b[k] = ((g(s00 + 1).wrapping_shl(2) & b2) | (g(s00).wrapping_shl(3) & b3))
                .wrapping_shr(bit) as u8;
            k += 1;
            s00 += 2;
            j += 2;
        }
        if j < ny {
            b[k] = (g(s00).wrapping_shl(3) & b3).wrapping_shr(bit) as u8;
            k += 1;
        }
    }
}

/// One further quadtree reduction: OR of the four child nybbles, one bit each. Runs in
/// place (`b` aliases `a` in cfitsio); the write index `k` always trails `s00 <= s10`,
/// so reading the four source cells into locals before the store is enough.
fn qtree_reduce(b: &mut [u8], n: usize, nx: usize, ny: usize) {
    let mut k = 0usize;
    let mut i = 0usize;
    while i + 1 < nx {
        let mut s00 = n * i;
        let mut s10 = s00 + n;
        let mut j = 0usize;
        while j + 1 < ny {
            let (v00, v01, v10, v11) = (b[s00], b[s00 + 1], b[s10], b[s10 + 1]);
            b[k] = (v11 != 0) as u8
                | (((v10 != 0) as u8) << 1)
                | (((v01 != 0) as u8) << 2)
                | (((v00 != 0) as u8) << 3);
            k += 1;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny {
            let (v00, v10) = (b[s00], b[s10]);
            b[k] = (((v10 != 0) as u8) << 1) | (((v00 != 0) as u8) << 3);
            k += 1;
        }
        i += 2;
    }
    if i < nx {
        let mut s00 = n * i;
        let mut j = 0usize;
        while j + 1 < ny {
            let (v00, v01) = (b[s00], b[s00 + 1]);
            b[k] = (((v01 != 0) as u8) << 2) | (((v00 != 0) as u8) << 3);
            k += 1;
            s00 += 2;
            j += 2;
        }
        if j < ny {
            let v00 = b[s00];
            b[k] = ((v00 != 0) as u8) << 3;
            k += 1;
        }
    }
}

/// The quadtree expanded the data: emit the direct-bitmap warning code and dump the
/// bit plane as raw nybbles.
fn write_bdirect(out: &mut Out, a: &[i32], off: usize, n: usize, nqx: usize, nqy: usize, bit: i32) {
    out.output_nybble(0x0);
    let count = ((nqx + 1) / 2) * ((nqy + 1) / 2);
    let mut scratch = vec![0u8; count.max(1)];
    qtree_onebit(a, off, n, nqx, nqy, &mut scratch, bit);
    out.output_nnybble(count, &scratch);
}
