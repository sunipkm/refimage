//! The tile-compressed `BINTABLE` HDU (`ZIMAGE` convention).
//!
//! For `Gzip` / `Rice`, tiling is one image row per tile (`ZTILE = [w, 1, …]`), matching
//! the cfitsio / `astropy` default. `Hcompress` is inherently 2-D, so it uses one whole
//! channel plane per tile (`ZTILE = [w, h, 1, …]`). Each tile is compressed
//! independently; the compressed bytes are concatenated into the table heap and pointed
//! at by a `1PB` variable-length-array descriptor per row.
//!
//! Integer images (`u8` / `u16`) compress losslessly. `f32` + `Rice` / `Hcompress`
//! quantizes each tile to 32-bit integers first ([`quantize`]); `ZSCALE` is constant and
//! `ZZERO` is per-tile, both written as `1D` columns alongside `COMPRESSED_DATA`.

use super::hdu::{ImageView, Tile};
use super::{gzip, hcompress, quantize, rice, FitsCompression, FitsError, FitsResult};

/// cfitsio's default quantization level (`ZVAL` for `QUANTIZE_LEVEL`).
const QUANTIZE_LEVEL: f64 = 4.0;

/// Everything `mod.rs` needs to emit a compressed image HDU.
pub(super) struct Compressed {
    pub zcmptype: &'static str,
    pub zbitpix: i64,
    pub zaxes: Vec<usize>,
    pub ztiles: Vec<usize>,
    pub bytepix: usize,
    pub is_rice: bool,
    /// `true` for `HCOMPRESS_1` — emits the `SCALE` / `SMOOTH` `ZNAME` cards.
    pub is_hcompress: bool,
    /// HCOMPRESS scale factor (`ZVAL1`); `0` = lossless.
    pub hscale: i64,
    /// `Some` when `f32` data was quantized: carries `ZDITHER0`.
    pub quant: Option<u32>,
    pub naxis2: usize,
    pub pcount: usize,
    /// `NAXIS1` — table row width in bytes (8, or 24 when quantized).
    pub row_bytes: usize,
    /// Main table (fixed columns + descriptors) followed by the heap; not block-padded.
    pub data: Vec<u8>,
}

pub(super) fn build(view: &ImageView<'_>, compression: FitsCompression) -> FitsResult<Compressed> {
    if compression == FitsCompression::Hcompress {
        return build_hcompress(view);
    }

    let is_rice = compression == FitsCompression::Rice;
    let zcmptype = match compression {
        FitsCompression::Gzip => "GZIP_1",
        FitsCompression::Rice => "RICE_1",
        FitsCompression::Hcompress => unreachable!("handled above"),
        FitsCompression::None => unreachable!("caller handles None"),
    };

    let axes = view.axes();
    let ztiles: Vec<usize> = axes
        .iter()
        .enumerate()
        .map(|(i, &n)| if i == 0 { n } else { 1 })
        .collect();
    let n_tiles = view.n_tiles();

    // f32 + Rice: quantize. f32 + Gzip stays lossless (raw float bytes).
    let quant = if is_rice && view.is_float() {
        let planar = view.planar_f32();
        let first = match view.tile(0) {
            Tile::F32(v) => v,
            _ => unreachable!(),
        };
        Some((
            quantize::global_delta(&planar, view.w, view.h * view.ch, QUANTIZE_LEVEL),
            quantize::dither_seed(&first),
        ))
    } else {
        None
    };

    let mut heap = Vec::new();
    let mut table = Vec::with_capacity(n_tiles * if quant.is_some() { 24 } else { 8 });

    for t in 0..n_tiles {
        let tile = view.tile(t);
        let (compressed, zzero) = match (&quant, compression, tile) {
            (Some((delta, seed)), _, Tile::F32(v)) => {
                let q = quantize::quantize_tile(&v, *delta, t, *seed);
                (rice::encode_int(&q.idata), Some(q.zzero))
            }
            (None, FitsCompression::Rice, Tile::I16(v)) => (rice::encode_short(&v), None),
            (None, FitsCompression::Rice, Tile::I8(v)) => (rice::encode_byte(&v), None),
            (None, FitsCompression::Rice, Tile::F32(_)) => unreachable!("handled by quant"),
            (None, FitsCompression::Gzip, tile) => (gzip::gzip(&tile.to_be_bytes()), None),
            (_, FitsCompression::None, _) => unreachable!(),
            (_, FitsCompression::Hcompress, _) => unreachable!("handled by build_hcompress"),
            (Some(_), _, _) => unreachable!("quant implies float tiles"),
        };

        let offset = heap.len() as i32;
        let nbytes = compressed.len() as i32;
        table.extend_from_slice(&nbytes.to_be_bytes());
        table.extend_from_slice(&offset.to_be_bytes());
        if let (Some((delta, _)), Some(zzero)) = (&quant, zzero) {
            table.extend_from_slice(&delta.to_be_bytes());
            table.extend_from_slice(&zzero.to_be_bytes());
        }
        heap.extend_from_slice(&compressed);
    }

    let pcount = heap.len();
    let row_bytes = if quant.is_some() { 24 } else { 8 };
    let mut data = table;
    data.extend_from_slice(&heap);

    // BYTEPIX for Rice: 4 for quantized floats, else the pixel width.
    let bytepix = if quant.is_some() { 4 } else { view.bytepix() };

    Ok(Compressed {
        zcmptype,
        zbitpix: view.bitpix(),
        zaxes: axes,
        ztiles,
        bytepix,
        is_rice,
        is_hcompress: false,
        hscale: 0,
        quant: quant.map(|(_, seed)| seed),
        naxis2: n_tiles,
        pcount,
        row_bytes,
        data,
    })
}

/// `HCOMPRESS_1`: the image is inherently 2-D, so each channel plane is compressed as
/// one whole tile (`ZTILE = [w, h, 1…]`). `f32` planes are quantized to integers first
/// (same `ZSCALE` / `ZZERO` / `ZDITHER0` machinery as `RICE_1`), then H-compressed
/// losslessly (`scale = 0`).
fn build_hcompress(view: &ImageView<'_>) -> FitsResult<Compressed> {
    if view.w < 4 || view.h < 4 {
        return Err(FitsError::HcompressTooSmall);
    }

    let axes = view.axes();
    let ztiles: Vec<usize> = axes
        .iter()
        .enumerate()
        .map(|(i, &n)| if i < 2 { n } else { 1 })
        .collect();
    let n_planes = view.ch;
    let scale = 0i32;

    // f32 planes are quantized; integer planes go straight through.
    let quant = if view.is_float() {
        let planar = view.planar_f32();
        Some((
            quantize::global_delta(&planar, view.w, view.h * view.ch, QUANTIZE_LEVEL),
            quantize::dither_seed(&view.plane_f32(0)),
        ))
    } else {
        None
    };

    let mut heap = Vec::new();
    let mut table = Vec::with_capacity(n_planes * if quant.is_some() { 24 } else { 8 });

    for p in 0..n_planes {
        let (mut idata, zzero) = match &quant {
            Some((delta, seed)) => {
                let q = quantize::quantize_tile(&view.plane_f32(p), *delta, p, *seed);
                (q.idata, Some(q.zzero))
            }
            None => (view.plane_i32(p), None),
        };
        let compressed = hcompress::compress(&mut idata, view.w, view.h, scale);

        let offset = heap.len() as i32;
        let nbytes = compressed.len() as i32;
        table.extend_from_slice(&nbytes.to_be_bytes());
        table.extend_from_slice(&offset.to_be_bytes());
        if let (Some((delta, _)), Some(z)) = (&quant, zzero) {
            table.extend_from_slice(&delta.to_be_bytes());
            table.extend_from_slice(&z.to_be_bytes());
        }
        heap.extend_from_slice(&compressed);
    }

    let pcount = heap.len();
    let row_bytes = if quant.is_some() { 24 } else { 8 };
    let mut data = table;
    data.extend_from_slice(&heap);

    Ok(Compressed {
        zcmptype: "HCOMPRESS_1",
        zbitpix: view.bitpix(),
        zaxes: axes,
        ztiles,
        bytepix: view.bytepix(),
        is_rice: false,
        is_hcompress: true,
        hscale: scale as i64,
        quant: quant.map(|(_, seed)| seed),
        naxis2: n_planes,
        pcount,
        row_bytes,
        data,
    })
}
