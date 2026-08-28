//! The tile-compressed `BINTABLE` HDU (`ZIMAGE` convention).
//!
//! Tiling is one image row per tile (`ZTILE = [w, 1, …]`), matching the cfitsio /
//! `astropy` default. Each tile is compressed independently; the compressed bytes are
//! concatenated into the table heap and pointed at by a `1PB` variable-length-array
//! descriptor per row.

use super::hdu::{ImageView, Tile};
use super::{gzip, rice, FitsCompression, FitsError, FitsResult};

/// Everything `mod.rs` needs to emit a compressed image HDU.
pub(super) struct Compressed {
    pub zcmptype: &'static str,
    pub zbitpix: i64,
    pub zaxes: Vec<usize>,
    pub ztiles: Vec<usize>,
    pub bytepix: usize,
    pub is_rice: bool,
    pub naxis2: usize,
    pub pcount: usize,
    /// Main table (descriptors) followed by the heap; not yet block-padded.
    pub data: Vec<u8>,
}

pub(super) fn build(view: &ImageView<'_>, compression: FitsCompression) -> FitsResult<Compressed> {
    let is_rice = compression == FitsCompression::Rice;
    if is_rice && view.is_float() {
        return Err(FitsError::CompressionUnsupported {
            pixel_type: crate::PixelType::F32,
            compression,
        });
    }

    let zcmptype = match compression {
        FitsCompression::Gzip => "GZIP_1",
        FitsCompression::Rice => "RICE_1",
        FitsCompression::None => unreachable!("caller handles None"),
    };

    let axes = view.axes();
    let ztiles: Vec<usize> = axes
        .iter()
        .enumerate()
        .map(|(i, &n)| if i == 0 { n } else { 1 })
        .collect();

    let n_tiles = view.n_tiles();
    let mut heap = Vec::new();
    let mut descriptors = Vec::with_capacity(n_tiles * 8);

    for t in 0..n_tiles {
        let compressed = match (compression, view.tile(t)) {
            (FitsCompression::Rice, Tile::I16(v)) => rice::encode_short(&v),
            (FitsCompression::Rice, Tile::I8(v)) => rice::encode_byte(&v),
            (FitsCompression::Rice, Tile::Be(_)) => unreachable!("float + rice rejected above"),
            (FitsCompression::Gzip, tile) => gzip::gzip(&tile_bytes(tile)),
            (FitsCompression::None, _) => unreachable!(),
        };
        let offset = heap.len() as i32;
        let nbytes = compressed.len() as i32;
        descriptors.extend_from_slice(&nbytes.to_be_bytes());
        descriptors.extend_from_slice(&offset.to_be_bytes());
        heap.extend_from_slice(&compressed);
    }

    let pcount = heap.len();
    let mut data = descriptors;
    data.extend_from_slice(&heap);

    Ok(Compressed {
        zcmptype,
        zbitpix: view.bitpix(),
        zaxes: axes,
        ztiles,
        bytepix: view.bytepix(),
        is_rice,
        naxis2: n_tiles,
        pcount,
        data,
    })
}

/// FITS-native big-endian bytes for one tile (used by the GZIP path).
fn tile_bytes(tile: Tile) -> Vec<u8> {
    match tile {
        Tile::I8(v) => v.iter().map(|&x| x as u8).collect(),
        Tile::I16(v) => {
            let mut b = Vec::with_capacity(v.len() * 2);
            for x in v {
                b.extend_from_slice(&x.to_be_bytes());
            }
            b
        }
        Tile::Be(v) => v,
    }
}
