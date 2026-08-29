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

    /// The value at interleaved position `(plane, row, col)`.
    fn interleaved_index(&self, plane: usize, row: usize, col: usize) -> usize {
        (row * self.w + col) * self.ch + plane
    }

    /// Interleaved indices for a `tw`×`th` rectangle at `(x0, y0)` of `plane`, row-major.
    fn rect_indices(
        &self,
        plane: usize,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        (0..th).flat_map(move |ry| {
            (0..tw).map(move |rx| self.interleaved_index(plane, y0 + ry, x0 + rx))
        })
    }

    /// A rectangular tile of one plane, row-major, in FITS-native form. `u8` is
    /// reinterpreted signed, `u16` is offset by `-32768`, `f32` passes through.
    pub(super) fn rect_tile(
        &self,
        plane: usize,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
    ) -> Tile {
        let idx = self.rect_indices(plane, x0, y0, tw, th);
        match self.pixels {
            Pixels::U8(p) => Tile::I8(idx.map(|i| p[i] as i8).collect()),
            Pixels::U16(p) => Tile::I16(idx.map(|i| (p[i] as i32 - 32768) as i16).collect()),
            Pixels::F32(p) => Tile::F32(idx.map(|i| p[i]).collect()),
        }
    }

    /// A rectangular tile as FITS-native `i32` — for HCOMPRESS, which is 2-D. `u8` stays
    /// unsigned `0..=255`; `u16` is offset by `-32768`. `f32` is quantized separately.
    pub(super) fn rect_i32(
        &self,
        plane: usize,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
    ) -> Vec<i32> {
        self.rect_indices(plane, x0, y0, tw, th)
            .map(|i| match self.pixels {
                Pixels::U8(p) => p[i] as i32,
                Pixels::U16(p) => p[i] as i32 - 32768,
                Pixels::F32(_) => unreachable!("f32 tiles are quantized before HCOMPRESS"),
            })
            .collect()
    }

    /// A rectangular `f32` tile, row-major (for per-tile quantization).
    pub(super) fn rect_f32(
        &self,
        plane: usize,
        x0: usize,
        y0: usize,
        tw: usize,
        th: usize,
    ) -> Vec<f32> {
        match self.pixels {
            Pixels::F32(p) => self
                .rect_indices(plane, x0, y0, tw, th)
                .map(|i| p[i])
                .collect(),
            _ => unreachable!("rect_f32 on a non-float image"),
        }
    }

    /// The whole `f32` image, planar order (for the global noise estimate).
    pub(super) fn planar_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.w * self.h * self.ch);
        for plane in 0..self.ch {
            out.extend(self.rect_f32(plane, 0, 0, self.w, self.h));
        }
        out
    }

    /// The whole image as one planar, big-endian, FITS-native byte blob (uncompressed
    /// data section, before block padding).
    pub(super) fn native_be(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * self.ch * self.bytepix());
        for plane in 0..self.ch {
            out.extend_from_slice(&self.rect_tile(plane, 0, 0, self.w, self.h).to_be_bytes());
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
