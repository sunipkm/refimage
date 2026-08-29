//! The tile-compressed `BINTABLE` HDU (`ZIMAGE` convention).
//!
//! Tiling is resolved here from the caller's [`Tiling`] choice and the concrete image
//! size: the default is one image row per tile for `GZIP_1` / `RICE_1` and one whole
//! channel plane for `HCOMPRESS_1`, but [`Rice::tile_rows`](super::Rice::tile_rows) /
//! [`Rice::tile_dims`](super::Rice::tile_dims) (and the same on the other builders) pick
//! any rectangular grid. Tiles are numbered fastest-FITS-axis-first; edge tiles are
//! clipped, not padded. Each tile is compressed independently; the bytes are
//! concatenated into the table heap and pointed at by a `1PB` variable-length-array
//! descriptor per row.
//!
//! Integer images (`u8` / `u16`) compress losslessly. `f32` + `Rice` / `Hcompress`
//! quantizes each tile to 32-bit integers first ([`quantize`]); `ZSCALE` is constant and
//! `ZZERO` is per-tile, both written as `1D` columns alongside `COMPRESSED_DATA`.

use super::config::{DitherSeed, Method, Quantize, Tiling};
use super::hdu::{ImageView, Tile};
use super::{gzip, hcompress, quantize, rice, FitsError, FitsResult};

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
    /// HCOMPRESS `SMOOTH` flag (`ZVAL2`).
    pub hsmooth: bool,
    /// `Some` when `f32` data was quantized: carries `ZDITHER0`.
    pub quant: Option<u32>,
    pub naxis2: usize,
    pub pcount: usize,
    /// `NAXIS1` — table row width in bytes (8, or 24 when quantized).
    pub row_bytes: usize,
    /// Main table (fixed columns + descriptors) followed by the heap; not block-padded.
    pub data: Vec<u8>,
}

/// The chosen algorithm plus its resolved settings.
enum Algo {
    Gzip,
    Rice(Quantize),
    Hcompress {
        scale: i32,
        smooth: bool,
        quantize: Quantize,
    },
}

impl Algo {
    fn is_hcompress(&self) -> bool {
        matches!(self, Algo::Hcompress { .. })
    }
    fn hscale(&self) -> i32 {
        match self {
            Algo::Hcompress { scale, .. } => *scale,
            _ => 0,
        }
    }
    fn quantize(&self) -> Option<Quantize> {
        match self {
            Algo::Gzip => None,
            Algo::Rice(q) => Some(*q),
            Algo::Hcompress { quantize, .. } => Some(*quantize),
        }
    }
}

pub(super) fn build(view: &ImageView<'_>, method: &Method) -> FitsResult<Compressed> {
    let (tiling, algo) = match method {
        Method::None => unreachable!("caller handles None"),
        Method::Gzip { tiling } => (tiling, Algo::Gzip),
        Method::Rice { tiling, quantize } => (tiling, Algo::Rice(*quantize)),
        Method::Hcompress {
            tiling,
            scale,
            smooth,
            quantize,
        } => (
            tiling,
            Algo::Hcompress {
                scale: *scale,
                smooth: *smooth,
                quantize: *quantize,
            },
        ),
    };
    build_tiled(view, tiling, &algo)
}

/// The tile grid, resolved against the image size.
struct Grid {
    tx: usize,
    ty: usize,
    ntx: usize,
    nty: usize,
    /// `ZTILE1..ZTILEn`.
    ztile: Vec<usize>,
}

fn resolve_grid(view: &ImageView<'_>, tiling: &Tiling, hcompress: bool) -> FitsResult<Grid> {
    let axes = view.axes();
    let (mut tx, mut ty) = match tiling {
        Tiling::Default => {
            if hcompress {
                (view.w, view.h)
            } else {
                (view.w, 1)
            }
        }
        Tiling::Rows(n) => (view.w, (*n).max(1)),
        Tiling::Dims(d) => {
            if d.is_empty() || d.iter().take(2).any(|&v| v == 0) {
                return Err(FitsError::InvalidTiling(
                    "tile dimensions must be non-zero".into(),
                ));
            }
            if d.len() > axes.len() || d.iter().skip(2).any(|&v| v != 1) {
                return Err(FitsError::InvalidTiling(
                    "only the first two tile dimensions may exceed 1".into(),
                ));
            }
            (d[0], d.get(1).copied().unwrap_or(1))
        }
    };
    tx = tx.min(view.w).max(1);
    ty = ty.min(view.h).max(1);

    let ntx = view.w.div_ceil(tx);
    let nty = view.h.div_ceil(ty);

    if hcompress {
        let last_w = view.w - (ntx - 1) * tx;
        let last_h = view.h - (nty - 1) * ty;
        if tx < 4 || ty < 4 || last_w < 4 || last_h < 4 {
            return Err(FitsError::HcompressTooSmall);
        }
    }

    let mut ztile = vec![1usize; axes.len()];
    ztile[0] = tx;
    ztile[1] = ty;

    Ok(Grid {
        tx,
        ty,
        ntx,
        nty,
        ztile,
    })
}

fn build_tiled(view: &ImageView<'_>, tiling: &Tiling, algo: &Algo) -> FitsResult<Compressed> {
    let grid = resolve_grid(view, tiling, algo.is_hcompress())?;

    // Global quantization parameters (only for `f32` + Rice/Hcompress).
    let quant = match algo.quantize() {
        Some(q) if view.is_float() => {
            let planar = view.planar_f32();
            let delta = quantize::global_delta(&planar, view.w, view.h * view.ch, q.level);
            let seed = match q.seed {
                DitherSeed::Auto => {
                    let tw = grid.tx.min(view.w);
                    let th = grid.ty.min(view.h);
                    quantize::dither_seed(&view.rect_f32(0, 0, 0, tw, th))
                }
                DitherSeed::Fixed(n) => n.clamp(1, 10_000),
            };
            Some((delta, seed))
        }
        _ => None,
    };

    let quantized = quant.is_some();
    let row_bytes = if quantized { 24 } else { 8 };
    let n_tiles = grid.ntx * grid.nty * view.ch;

    let mut heap = Vec::new();
    let mut table = Vec::with_capacity(n_tiles * row_bytes);

    let mut tile_index = 0usize;
    for plane in 0..view.ch {
        for iy in 0..grid.nty {
            for ix in 0..grid.ntx {
                let x0 = ix * grid.tx;
                let y0 = iy * grid.ty;
                let tw = grid.tx.min(view.w - x0);
                let th = grid.ty.min(view.h - y0);

                let (compressed, zzero) = if let Some((delta, seed)) = quant {
                    let f = view.rect_f32(plane, x0, y0, tw, th);
                    let q = quantize::quantize_tile(&f, delta, tile_index, seed);
                    let bytes = if algo.is_hcompress() {
                        let mut d = q.idata;
                        hcompress::compress(&mut d, tw, th, algo.hscale())
                    } else {
                        rice::encode_int(&q.idata)
                    };
                    (bytes, Some(q.zzero))
                } else {
                    let bytes = match algo {
                        Algo::Gzip => {
                            gzip::gzip(&view.rect_tile(plane, x0, y0, tw, th).to_be_bytes())
                        }
                        Algo::Rice(_) => match view.rect_tile(plane, x0, y0, tw, th) {
                            Tile::I16(v) => rice::encode_short(&v),
                            Tile::I8(v) => rice::encode_byte(&v),
                            Tile::F32(_) => unreachable!("float is quantized"),
                        },
                        Algo::Hcompress { scale, .. } => {
                            let mut d = view.rect_i32(plane, x0, y0, tw, th);
                            hcompress::compress(&mut d, tw, th, *scale)
                        }
                    };
                    (bytes, None)
                };

                let offset = heap.len() as i32;
                let nbytes = compressed.len() as i32;
                table.extend_from_slice(&nbytes.to_be_bytes());
                table.extend_from_slice(&offset.to_be_bytes());
                if let Some((delta, _)) = quant {
                    table.extend_from_slice(&delta.to_be_bytes());
                    table.extend_from_slice(&zzero.unwrap_or(0.0).to_be_bytes());
                }
                heap.extend_from_slice(&compressed);
                tile_index += 1;
            }
        }
    }

    let pcount = heap.len();
    let mut data = table;
    data.extend_from_slice(&heap);

    let (zcmptype, is_rice, is_hcompress, hscale, hsmooth) = match algo {
        Algo::Gzip => ("GZIP_1", false, false, 0i64, false),
        Algo::Rice(_) => ("RICE_1", true, false, 0, false),
        Algo::Hcompress { scale, smooth, .. } => {
            ("HCOMPRESS_1", false, true, *scale as i64, *smooth)
        }
    };
    // BYTEPIX for Rice: 4 for quantized floats, else the pixel width.
    let bytepix = if quantized { 4 } else { view.bytepix() };

    Ok(Compressed {
        zcmptype,
        zbitpix: view.bitpix(),
        zaxes: view.axes(),
        ztiles: grid.ztile,
        bytepix,
        is_rice,
        is_hcompress,
        hscale,
        hsmooth,
        quant: quant.map(|(_, seed)| seed),
        naxis2: n_tiles,
        pcount,
        row_bytes,
        data,
    })
}
