//! Turning a `DynamicImage*` into FITS image bytes.
//!
//! Multi-channel images are written **planar** (`NAXIS3 = channels`, all of channel 0,
//! then channel 1, …); `refimage` stores channels interleaved, so this module
//! de-interleaves. Unsigned 16-bit data is offset to signed (`BZERO = 32768`), which is
//! how `astropy.io.fits` recognises it as `uint16`.

use crate::{ColorSpace, DynamicImageOwned, DynamicImageRef, ImageProps};

/// A borrowed view of image pixels, independent of ref/owned.
pub(super) struct ImageView<'a> {
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    pub cspace: ColorSpace,
    pixels: Pixels<'a>,
}

enum Pixels<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

/// One image tile (a single row of one plane) in its FITS-native form.
pub(super) enum Tile {
    /// Signed 16-bit (`u16` image, already offset by `-32768`).
    I16(Vec<i16>),
    /// Signed 8-bit (`u8` image reinterpreted).
    I8(Vec<i8>),
    /// 32-bit float values (`f32` image).
    F32(Vec<f32>),
}

impl Tile {
    /// FITS-native big-endian bytes for this tile.
    pub(super) fn to_be_bytes(&self) -> Vec<u8> {
        match self {
            Tile::I8(v) => v.iter().map(|&x| x as u8).collect(),
            Tile::I16(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            Tile::F32(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
        }
    }
}

impl<'a> ImageView<'a> {
    pub(super) fn from_ref(img: &'a DynamicImageRef<'_>) -> Self {
        let (w, h, ch) = (img.width(), img.height(), img.channels() as usize);
        let cspace = img.color_space();
        let pixels = match img {
            DynamicImageRef::U8(r) => Pixels::U8(r.as_slice()),
            DynamicImageRef::U16(r) => Pixels::U16(r.as_slice()),
            DynamicImageRef::F32(r) => Pixels::F32(r.as_slice()),
        };
        Self {
            w,
            h,
            ch,
            cspace,
            pixels,
        }
    }

    pub(super) fn from_owned(img: &'a DynamicImageOwned) -> Self {
        let (w, h, ch) = (img.width(), img.height(), img.channels() as usize);
        let cspace = img.color_space();
        let pixels = match img {
            DynamicImageOwned::U8(r) => Pixels::U8(r.as_slice()),
            DynamicImageOwned::U16(r) => Pixels::U16(r.as_slice()),
            DynamicImageOwned::F32(r) => Pixels::F32(r.as_slice()),
        };
        Self {
            w,
            h,
            ch,
            cspace,
            pixels,
        }
    }

    /// FITS `BITPIX` for the image data.
    pub(super) fn bitpix(&self) -> i64 {
        match self.pixels {
            Pixels::U8(_) => 8,
            Pixels::U16(_) => 16,
            Pixels::F32(_) => -32,
        }
    }

    /// `BZERO` needed to express the data as signed FITS values, if any.
    pub(super) fn bzero(&self) -> Option<i64> {
        matches!(self.pixels, Pixels::U16(_)).then_some(32768)
    }

    pub(super) fn is_float(&self) -> bool {
        matches!(self.pixels, Pixels::F32(_))
    }

    /// FITS axis lengths, fastest first: `[w, h]` or `[w, h, ch]`.
    pub(super) fn axes(&self) -> Vec<usize> {
        if self.ch > 1 {
            vec![self.w, self.h, self.ch]
        } else {
            vec![self.w, self.h]
        }
    }

    /// Bytes per pixel of the compressed representation (`ZVAL2 = BYTEPIX`).
    pub(super) fn bytepix(&self) -> usize {
        match self.pixels {
            Pixels::U8(_) => 1,
            Pixels::U16(_) => 2,
            Pixels::F32(_) => 4,
        }
    }

    /// Number of row tiles: one per row of each plane.
    pub(super) fn n_tiles(&self) -> usize {
        self.h * self.ch
    }

    /// The value at interleaved position `(plane, row, col)`.
    fn interleaved_index(&self, plane: usize, row: usize, col: usize) -> usize {
        (row * self.w + col) * self.ch + plane
    }

    /// Tile `t` (row `t % h` of plane `t / h`) in FITS-native form.
    pub(super) fn tile(&self, t: usize) -> Tile {
        let plane = t / self.h;
        let row = t % self.h;
        let idx = |col| self.interleaved_index(plane, row, col);
        match self.pixels {
            Pixels::U8(p) => Tile::I8((0..self.w).map(|c| p[idx(c)] as i8).collect()),
            Pixels::U16(p) => Tile::I16(
                (0..self.w)
                    .map(|c| (p[idx(c)] as i32 - 32768) as i16)
                    .collect(),
            ),
            Pixels::F32(p) => Tile::F32((0..self.w).map(|c| p[idx(c)]).collect()),
        }
    }

    /// One whole plane, row-major, as FITS-native `i32` — for HCOMPRESS, which is 2-D.
    /// `u8` stays unsigned `0..=255`; `u16` is offset by `-32768`. `f32` is quantized
    /// separately, so it is not handled here.
    pub(super) fn plane_i32(&self, plane: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.w * self.h);
        for row in 0..self.h {
            for col in 0..self.w {
                let idx = self.interleaved_index(plane, row, col);
                out.push(match self.pixels {
                    Pixels::U8(p) => p[idx] as i32,
                    Pixels::U16(p) => p[idx] as i32 - 32768,
                    Pixels::F32(_) => unreachable!("f32 planes are quantized before HCOMPRESS"),
                });
            }
        }
        out
    }

    /// One whole `f32` plane, row-major (for per-tile HCOMPRESS quantization).
    pub(super) fn plane_f32(&self, plane: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.w * self.h);
        for row in 0..self.h {
            for col in 0..self.w {
                let idx = self.interleaved_index(plane, row, col);
                match self.pixels {
                    Pixels::F32(p) => out.push(p[idx]),
                    _ => unreachable!("plane_f32 on a non-float image"),
                }
            }
        }
        out
    }

    /// The whole `f32` image, planar order (for the global noise estimate).
    pub(super) fn planar_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.w * self.h * self.ch);
        for t in 0..self.n_tiles() {
            if let Tile::F32(v) = self.tile(t) {
                out.extend_from_slice(&v);
            }
        }
        out
    }

    /// The whole image as one planar, big-endian, FITS-native byte blob (uncompressed
    /// data section, before block padding).
    pub(super) fn native_be(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * self.ch * self.bytepix());
        for t in 0..self.n_tiles() {
            out.extend_from_slice(&self.tile(t).to_be_bytes());
        }
        out
    }
}

/// The FITS string for a colour space, for the `COLORSPC` card.
pub(super) fn colorspace_str(cs: &ColorSpace) -> String {
    use crate::BayerPattern::*;
    match cs {
        ColorSpace::Gray => "GRAY".into(),
        ColorSpace::Rgb => "RGB".into(),
        ColorSpace::Bayer(Bggr) => "BGGR".into(),
        ColorSpace::Bayer(Gbrg) => "GBRG".into(),
        ColorSpace::Bayer(Grbg) => "GRBG".into(),
        ColorSpace::Bayer(Rggb) => "RGGB".into(),
        ColorSpace::Custom(n, name) => format!("C({n},{name})"),
    }
}

/// The `BAYERPAT` value for a Bayer colour space, if applicable.
pub(super) fn bayer_pattern(cs: &ColorSpace) -> Option<&'static str> {
    use crate::BayerPattern::*;
    match cs {
        ColorSpace::Bayer(Bggr) => Some("BGGR"),
        ColorSpace::Bayer(Gbrg) => Some("GBRG"),
        ColorSpace::Bayer(Grbg) => Some("GRBG"),
        ColorSpace::Bayer(Rggb) => Some("RGGB"),
        _ => None,
    }
}
