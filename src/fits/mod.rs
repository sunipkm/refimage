//! A pure-Rust FITS writer.
//!
//! No `cfitsio` linkage: this compiles on every target, `wasm32` included. It writes
//! plain (uncompressed) image HDUs and the tile-compression convention with `GZIP_1`
//! (lossless, any pixel type), `RICE_1` and `HCOMPRESS_1` (both lossless for integers;
//! `f32` is quantized with `SUBTRACTIVE_DITHER_1`). Output is written to be read
//! cleanly by `astropy.io.fits`.
//!
//! Every path can target a file or an in-memory buffer: [`FitsWrite::fits_bytes`] /
//! [`FitsWrite::write_fits_to`] for a single image, [`create_fits_to`] for a multi-HDU
//! file (both work on `wasm32`, which has no filesystem).
//!
//! # Layout
//!
//! - A single uncompressed image goes in the primary HDU.
//! - A single compressed image, and every file built with [`create_fits`] /
//!   [`create_fits_to`] / [`FitsWrite::append_fits`], gets an empty primary HDU followed
//!   by image extensions.
//! - Multi-channel images are stored **planar** (`NAXIS3 = channels`).
//! - `u16` data carries `BZERO = 32768` so readers see `uint16`.
//!
//! # Metadata
//!
//! The image timestamp is written as the standard `DATE-OBS`, and a non-zero exposure
//! as `EXPOSURE_S` / `EXPOSURE_NS` / `EXPOSURE`. Each extra
//! [`GenericLineItem`](crate::GenericLineItem) then becomes a header card, in insertion
//! order (`HIERARCH` for long keys, `CONTINUE` for long strings); [`Duration`] and
//! UTC timestamp values are written as an exact `<KEY>_S` / `<KEY>_NS` integer pair
//! plus a convenient base card (`f64` seconds, or an ISO-8601 string).

mod card;
mod compress;
mod datetime;
mod gzip;
mod hcompress;
mod hdu;
mod quantize;
mod rice;

use chrono::{DateTime, Utc};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::{GenericLineItem, GenericValue, Metadata, EXPOSURE_KEY};

use card::Header;
use hdu::{bayer_pattern, colorspace_str, ImageView};

/// Compression applied to the image data of a FITS HDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FitsCompression {
    /// No compression — a plain IMAGE HDU.
    None,
    /// `GZIP_1`: each row tile is a gzip stream. Lossless for every pixel type.
    Gzip,
    /// `RICE_1`: Rice coding of each row tile. Lossless for `u8` / `u16`; `f32` is
    /// quantized first (`ZQUANTIZ = 'SUBTRACTIVE_DITHER_1'`, lossy).
    Rice,
    /// `HCOMPRESS_1`: the H-transform + quadtree coder, each channel plane compressed
    /// as one whole tile. Lossless (`scale = 0`) for `u8` / `u16`; `f32` is quantized
    /// first (`ZQUANTIZ = 'SUBTRACTIVE_DITHER_1'`, lossy). Needs a 2-D image at least
    /// 4×4.
    Hcompress,
}

/// Errors produced while writing a FITS file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FitsError {
    /// Underlying I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A metadata key collides with a structural FITS keyword.
    #[error("metadata key {0:?} is a reserved FITS keyword")]
    ReservedKeyword(String),
    /// A metadata key does not fit in a FITS keyword, even as `HIERARCH`.
    #[error("metadata key {0:?} is too long for a FITS keyword")]
    KeywordTooLong(String),
    /// A metadata string value is too long to encode.
    #[error("metadata string value is too long")]
    MetadataValueTooLong,
    /// `HCOMPRESS` was requested for an image smaller than 4×4 (or 1-D).
    #[error("HCOMPRESS needs a 2-D image at least 4x4 pixels")]
    HcompressTooSmall,
}

/// `Result` alias for [`FitsError`].
pub type FitsResult<T> = Result<T, FitsError>;

/// A multi-HDU FITS file open for writing, over any [`Write`] sink (a file by default,
/// or an in-memory buffer — handy on `wasm32`, which has no filesystem).
///
/// Append image HDUs with [`FitsWrite::append_fits`], then call [`FitsWriter::finish`]
/// to flush and recover the sink.
pub struct FitsWriter<W: Write = BufWriter<std::fs::File>> {
    out: W,
    compress: FitsCompression,
    n_hdus: usize,
}

impl<W: Write> FitsWriter<W> {
    /// Flush the sink and return it (a `Vec<u8>` now holds the whole file, a file
    /// handle is fully written).
    pub fn finish(mut self) -> FitsResult<W> {
        self.out.flush()?;
        Ok(self.out)
    }

    /// Number of HDUs written so far (the empty primary counts as one).
    pub fn hdu_count(&self) -> usize {
        self.n_hdus
    }
}

/// Create a FITS file on disk with an empty primary HDU, ready for
/// [`FitsWrite::append_fits`]. `compress` is the compression each appended image uses.
pub fn create_fits<P: AsRef<Path>>(
    path: P,
    compress: FitsCompression,
    overwrite: bool,
) -> FitsResult<FitsWriter> {
    let out = BufWriter::new(open(path.as_ref(), overwrite)?);
    create_fits_to(out, compress)
}

/// Like [`create_fits`], but writes to any [`Write`] sink instead of a path — e.g. a
/// `Vec<u8>` for an in-memory multi-HDU FITS file. The empty primary HDU is written
/// immediately.
pub fn create_fits_to<W: Write>(
    mut sink: W,
    compress: FitsCompression,
) -> FitsResult<FitsWriter<W>> {
    sink.write_all(&primary_empty())?;
    Ok(FitsWriter {
        out: sink,
        compress,
        n_hdus: 1,
    })
}

fn open(path: &Path, overwrite: bool) -> FitsResult<std::fs::File> {
    if path.is_dir() {
        return Err(FitsError::Io(std::io::Error::new(
            std::io::ErrorKind::IsADirectory,
            "path is a directory",
        )));
    }
    Ok(OpenOptions::new()
        .write(true)
        .create(true)
        .create_new(!overwrite)
        .truncate(true)
        .open(path)?)
}

/// Write an image, with its metadata, to a FITS file.
pub trait FitsWrite {
    /// Write the image to `path` (created, or truncated if `overwrite`). Returns `path`.
    fn write_fits<P: AsRef<Path>>(
        &self,
        path: P,
        compress: FitsCompression,
        overwrite: bool,
    ) -> FitsResult<PathBuf>;

    /// Write the image to any [`Write`] sink (no filesystem needed).
    fn write_fits_to<W: Write>(&self, sink: W, compress: FitsCompression) -> FitsResult<()>;

    /// Serialise the image to an in-memory FITS file.
    fn fits_bytes(&self, compress: FitsCompression) -> FitsResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.write_fits_to(&mut buf, compress)?;
        Ok(buf)
    }

    /// Append the image as a new HDU of an open [`FitsWriter`] (file- or memory-backed).
    fn append_fits<W: Write>(&self, writer: &mut FitsWriter<W>) -> FitsResult<()>;
}

macro_rules! impl_fitswrite {
    ($t:ty) => {
        impl FitsWrite for $t {
            fn write_fits<P: AsRef<Path>>(
                &self,
                path: P,
                compress: FitsCompression,
                overwrite: bool,
            ) -> FitsResult<PathBuf> {
                let path = path.as_ref().to_path_buf();
                let file = open(&path, overwrite)?;
                self.write_fits_to(BufWriter::new(file), compress)?;
                Ok(path)
            }

            fn write_fits_to<W: Write>(
                &self,
                mut sink: W,
                compress: FitsCompression,
            ) -> FitsResult<()> {
                let view = ImageView::from_ref_like(self.get_image());
                let meta = self.get_metadata();
                if compress == FitsCompression::None {
                    sink.write_all(&serialize_image(&view, meta, true)?)?;
                } else {
                    sink.write_all(&primary_empty())?;
                    sink.write_all(&serialize_compressed(&view, meta, compress)?)?;
                }
                Ok(())
            }

            fn append_fits<Sink: Write>(&self, writer: &mut FitsWriter<Sink>) -> FitsResult<()> {
                let view = ImageView::from_ref_like(self.get_image());
                let meta = self.get_metadata();
                let bytes = if writer.compress == FitsCompression::None {
                    serialize_image(&view, meta, false)?
                } else {
                    serialize_compressed(&view, meta, writer.compress)?
                };
                writer.out.write_all(&bytes)?;
                writer.n_hdus += 1;
                Ok(())
            }
        }
    };
}

impl_fitswrite!(crate::GenericImageRef<'_>);
impl_fitswrite!(crate::GenericImageOwned);

impl FitsWrite for crate::GenericImage<'_> {
    fn write_fits<P: AsRef<Path>>(
        &self,
        path: P,
        compress: FitsCompression,
        overwrite: bool,
    ) -> FitsResult<PathBuf> {
        match self {
            crate::GenericImage::Ref(i) => i.write_fits(path, compress, overwrite),
            crate::GenericImage::Own(i) => i.write_fits(path, compress, overwrite),
        }
    }

    fn write_fits_to<W: Write>(&self, sink: W, compress: FitsCompression) -> FitsResult<()> {
        match self {
            crate::GenericImage::Ref(i) => i.write_fits_to(sink, compress),
            crate::GenericImage::Own(i) => i.write_fits_to(sink, compress),
        }
    }

    fn append_fits<W: Write>(&self, writer: &mut FitsWriter<W>) -> FitsResult<()> {
        match self {
            crate::GenericImage::Ref(i) => i.append_fits(writer),
            crate::GenericImage::Own(i) => i.append_fits(writer),
        }
    }
}

/// An empty primary HDU (`NAXIS = 0`), block-padded.
fn primary_empty() -> Vec<u8> {
    let mut h = Header::new();
    h.logical("SIMPLE", true, Some("conforms to FITS standard"))
        .unwrap();
    h.integer("BITPIX", 8, None).unwrap();
    h.integer("NAXIS", 0, None).unwrap();
    h.logical("EXTEND", true, None).unwrap();
    h.finish()
}

/// A plain (uncompressed) image HDU.
fn serialize_image(view: &ImageView<'_>, meta: &Metadata, primary: bool) -> FitsResult<Vec<u8>> {
    let axes = view.axes();
    let mut h = Header::new();

    if primary {
        h.logical("SIMPLE", true, Some("conforms to FITS standard"))?;
        h.integer("BITPIX", view.bitpix(), None)?;
        h.integer("NAXIS", axes.len() as i64, None)?;
        for (i, &n) in axes.iter().enumerate() {
            h.integer(&format!("NAXIS{}", i + 1), n as i64, None)?;
        }
        h.logical("EXTEND", true, None)?;
    } else {
        h.string("XTENSION", "IMAGE", Some("image extension"))?;
        h.integer("BITPIX", view.bitpix(), None)?;
        h.integer("NAXIS", axes.len() as i64, None)?;
        for (i, &n) in axes.iter().enumerate() {
            h.integer(&format!("NAXIS{}", i + 1), n as i64, None)?;
        }
        h.integer("PCOUNT", 0, None)?;
        h.integer("GCOUNT", 1, None)?;
    }

    if let Some(bz) = view.bzero() {
        h.integer("BZERO", bz, Some("offset for unsigned integers"))?;
        h.integer("BSCALE", 1, None)?;
    }

    write_common_cards(&mut h, view, meta.timestamp())?;
    write_metadata(&mut h, meta)?;

    let mut out = h.finish();
    let mut data = view.native_be();
    card::pad_data(&mut data);
    out.extend_from_slice(&data);
    Ok(out)
}

/// A tile-compressed image HDU (`BINTABLE` with `ZIMAGE = T`).
fn serialize_compressed(
    view: &ImageView<'_>,
    meta: &Metadata,
    compression: FitsCompression,
) -> FitsResult<Vec<u8>> {
    let c = compress::build(view, compression)?;
    let quantized = c.quant.is_some();
    let mut h = Header::new();

    h.string("XTENSION", "BINTABLE", Some("binary table extension"))?;
    h.integer("BITPIX", 8, None)?;
    h.integer("NAXIS", 2, None)?;
    h.integer(
        "NAXIS1",
        c.row_bytes as i64,
        Some("width of table row (bytes)"),
    )?;
    h.integer("NAXIS2", c.naxis2 as i64, Some("number of tiles"))?;
    h.integer("PCOUNT", c.pcount as i64, Some("heap size (bytes)"))?;
    h.integer("GCOUNT", 1, None)?;
    h.integer("TFIELDS", if quantized { 3 } else { 1 }, None)?;
    h.string("TTYPE1", "COMPRESSED_DATA", None)?;
    h.string("TFORM1", "1PB", Some("variable-length array of bytes"))?;
    if quantized {
        h.string("TTYPE2", "ZSCALE", None)?;
        h.string("TFORM2", "1D", None)?;
        h.string("TTYPE3", "ZZERO", None)?;
        h.string("TFORM3", "1D", None)?;
    }

    h.logical("ZIMAGE", true, Some("tile-compressed image"))?;
    h.string("ZCMPTYPE", c.zcmptype, Some("compression algorithm"))?;
    h.integer("ZBITPIX", c.zbitpix, Some("data type of original image"))?;
    h.integer("ZNAXIS", c.zaxes.len() as i64, None)?;
    for (i, &n) in c.zaxes.iter().enumerate() {
        h.integer(&format!("ZNAXIS{}", i + 1), n as i64, None)?;
    }
    for (i, &n) in c.ztiles.iter().enumerate() {
        h.integer(&format!("ZTILE{}", i + 1), n as i64, None)?;
    }
    if c.is_rice {
        h.string("ZNAME1", "BLOCKSIZE", None)?;
        h.integer("ZVAL1", 32, None)?;
        h.string("ZNAME2", "BYTEPIX", None)?;
        h.integer("ZVAL2", c.bytepix as i64, None)?;
    }
    if c.is_hcompress {
        h.string("ZNAME1", "SCALE", Some("HCOMPRESS scale factor"))?;
        h.real("ZVAL1", c.hscale as f64, Some("HCOMPRESS scale factor"))?;
        h.string("ZNAME2", "SMOOTH", Some("HCOMPRESS smooth option"))?;
        h.integer("ZVAL2", 0, Some("HCOMPRESS smooth option"))?;
    }
    if let Some(zdither0) = c.quant {
        h.string(
            "ZQUANTIZ",
            "SUBTRACTIVE_DITHER_1",
            Some("lossy float quantization"),
        )?;
        h.integer("ZDITHER0", zdither0 as i64, Some("dithering offset"))?;
    }

    if let Some(bz) = view.bzero() {
        h.integer("BZERO", bz, Some("offset for unsigned integers"))?;
        h.integer("BSCALE", 1, None)?;
    }

    write_common_cards(&mut h, view, meta.timestamp())?;
    write_metadata(&mut h, meta)?;

    let mut out = h.finish();
    let mut data = c.data;
    card::pad_data(&mut data);
    out.extend_from_slice(&data);
    Ok(out)
}

/// `DATE-OBS`, `COLORSPC`, `BAYERPAT` — the image's own descriptive cards.
fn write_common_cards(h: &mut Header, view: &ImageView<'_>, ts: DateTime<Utc>) -> FitsResult<()> {
    h.string(
        "DATE-OBS",
        &datetime::to_iso8601(ts),
        Some("UTC of exposure start"),
    )?;
    h.string(
        "COLORSPC",
        &colorspace_str(&view.cspace),
        Some("refimage colour space"),
    )?;
    if let Some(p) = bayer_pattern(&view.cspace) {
        h.string("BAYERPAT", p, Some("Bayer mosaic pattern"))?;
    }
    Ok(())
}

/// Emit the typed exposure plus one or more cards for every extra metadata entry,
/// in insertion order.
fn write_metadata(h: &mut Header, meta: &Metadata) -> FitsResult<()> {
    let exp = meta.exposure();
    if !exp.is_zero() {
        h.integer(
            &format!("{EXPOSURE_KEY}_S"),
            exp.as_secs() as i64,
            Some("[s]"),
        )?;
        h.integer(
            &format!("{EXPOSURE_KEY}_NS"),
            exp.subsec_nanos() as i64,
            Some("[ns]"),
        )?;
        h.real(EXPOSURE_KEY, exp.as_secs_f64(), Some("[s] exposure time"))?;
    }
    for (key, item) in meta.iter() {
        if card::is_reserved(key) {
            return Err(FitsError::ReservedKeyword(key.clone()));
        }
        write_item(h, key, item)?;
    }
    Ok(())
}

fn write_item(h: &mut Header, key: &str, item: &GenericLineItem) -> FitsResult<()> {
    let comment = item.get_comment();
    match item.get_value() {
        GenericValue::Integer(v) => h.integer(key, *v, comment)?,
        GenericValue::Real(v) => h.real(key, *v, comment)?,
        GenericValue::String(s) => {
            if s.len() > 65_536 {
                return Err(FitsError::MetadataValueTooLong);
            }
            h.string(key, s, comment)?;
        }
        GenericValue::ColorSpace(cs) => h.string(key, &colorspace_str(cs), comment)?,
        GenericValue::Duration(d) => {
            h.integer(&format!("{key}_S"), d.as_secs() as i64, Some("[s]"))?;
            h.integer(&format!("{key}_NS"), d.subsec_nanos() as i64, Some("[ns]"))?;
            h.real(key, d.as_secs_f64(), comment.or(Some("[s]")))?;
        }
        GenericValue::Timestamp(t) => {
            // The image's own timestamp is emitted as `DATE-OBS` by
            // `write_common_cards`; this arm is for any *other* timestamp value.
            let (secs, nanos) = datetime::epoch_parts(*t);
            h.integer(&format!("{key}_S"), secs, Some("[s] since 1970-01-01"))?;
            h.integer(&format!("{key}_NS"), nanos as i64, Some("[ns]"))?;
            h.string(key, &datetime::to_iso8601(*t), comment)?;
        }
    }
    Ok(())
}

// Bridge so the macro can accept either a `&DynamicImageRef` or `&DynamicImageOwned`.
impl<'a> ImageView<'a> {
    fn from_ref_like<T: AsImageView>(v: &'a T) -> Self {
        v.as_image_view()
    }
}

trait AsImageView {
    fn as_image_view(&self) -> ImageView<'_>;
}

impl AsImageView for crate::DynamicImageRef<'_> {
    fn as_image_view(&self) -> ImageView<'_> {
        ImageView::from_ref(self)
    }
}

impl AsImageView for crate::DynamicImageOwned {
    fn as_image_view(&self) -> ImageView<'_> {
        ImageView::from_owned(self)
    }
}

#[cfg(test)]
mod tests;
