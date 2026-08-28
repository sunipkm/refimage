//! A pure-Rust FITS writer.
//!
//! No `cfitsio` linkage: this compiles on every target, `wasm32` included. It writes
//! plain (uncompressed) image HDUs and the tile-compression convention with `GZIP_1`
//! (any pixel type) and `RICE_1` (integer images). Output is written to be read cleanly
//! by `astropy.io.fits`.
//!
//! # Layout
//!
//! - A single uncompressed image goes in the primary HDU.
//! - A single compressed image, and every file built with [`create_fits`] /
//!   [`FitsWrite::append_fits`], gets an empty primary HDU followed by image extensions.
//! - Multi-channel images are stored **planar** (`NAXIS3 = channels`).
//! - `u16` data carries `BZERO = 32768` so readers see `uint16`.
//!
//! # Metadata
//!
//! Each [`GenericLineItem`](crate::GenericLineItem) becomes a header card (`HIERARCH`
//! for long keys, `CONTINUE` for long strings). [`Duration`] and [`SystemTime`] values
//! are written as an exact `<KEY>_S` / `<KEY>_NS` integer pair plus a convenient base
//! card (`f64` seconds, or an ISO-8601 string). The image timestamp is also written as
//! the standard `DATE-OBS`.

mod card;
mod compress;
mod datetime;
mod gzip;
mod hdu;
mod rice;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use thiserror::Error;

use crate::{GenericLineItem, GenericValue, MetaCollection, PixelType};

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
    /// `RICE_1`: Rice coding of each row tile. Integer images only (`u8` / `u16`).
    Rice,
}

/// Errors produced while writing a FITS file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FitsError {
    /// Underlying I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The image carries no `TIMESTAMP` metadata.
    #[error("image has no TIMESTAMP metadata")]
    MissingTimestamp,
    /// A timestamp or duration value predates the Unix epoch.
    #[error("timestamp predates the Unix epoch")]
    TimestampBeforeEpoch,
    /// A metadata key collides with a structural FITS keyword.
    #[error("metadata key {0:?} is a reserved FITS keyword")]
    ReservedKeyword(String),
    /// A metadata key does not fit in a FITS keyword, even as `HIERARCH`.
    #[error("metadata key {0:?} is too long for a FITS keyword")]
    KeywordTooLong(String),
    /// A metadata string value is too long to encode.
    #[error("metadata string value is too long")]
    MetadataValueTooLong,
    /// The requested compression cannot encode this pixel type.
    #[error("{compression:?} compression does not support {pixel_type:?} images")]
    CompressionUnsupported {
        /// The image's pixel type.
        pixel_type: PixelType,
        /// The requested compression.
        compression: FitsCompression,
    },
}

/// `Result` alias for [`FitsError`].
pub type FitsResult<T> = Result<T, FitsError>;

/// A FITS file open for writing. Append image HDUs with [`FitsWrite::append_fits`], then
/// call [`FitsWriter::finish`] (or just drop it).
pub struct FitsWriter {
    out: BufWriter<std::fs::File>,
    compress: FitsCompression,
    n_hdus: usize,
}

impl FitsWriter {
    /// Flush the file. Dropping the writer does this too, but without surfacing errors.
    pub fn finish(mut self) -> FitsResult<()> {
        self.out.flush()?;
        Ok(())
    }

    /// Number of HDUs written so far (the empty primary counts as one).
    pub fn hdu_count(&self) -> usize {
        self.n_hdus
    }
}

impl Drop for FitsWriter {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}

/// Create a FITS file with an empty primary HDU, ready for
/// [`FitsWrite::append_fits`]. `compress` is the compression each appended image uses.
pub fn create_fits<P: AsRef<Path>>(
    path: P,
    compress: FitsCompression,
    overwrite: bool,
) -> FitsResult<FitsWriter> {
    let mut out = BufWriter::new(open(path.as_ref(), overwrite)?);
    out.write_all(&primary_empty())?;
    Ok(FitsWriter {
        out,
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

    /// Append the image as a new HDU of an open [`FitsWriter`].
    fn append_fits(&self, writer: &mut FitsWriter) -> FitsResult<()>;
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
                let ts = self.get_timestamp();
                let meta = self.get_metadata();
                if compress == FitsCompression::None {
                    sink.write_all(&serialize_image(&view, meta, ts, true)?)?;
                } else {
                    sink.write_all(&primary_empty())?;
                    sink.write_all(&serialize_compressed(&view, meta, ts, compress)?)?;
                }
                Ok(())
            }

            fn append_fits(&self, writer: &mut FitsWriter) -> FitsResult<()> {
                let view = ImageView::from_ref_like(self.get_image());
                let ts = self.get_timestamp();
                let meta = self.get_metadata();
                let bytes = if writer.compress == FitsCompression::None {
                    serialize_image(&view, meta, ts, false)?
                } else {
                    serialize_compressed(&view, meta, ts, writer.compress)?
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

    fn append_fits(&self, writer: &mut FitsWriter) -> FitsResult<()> {
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
fn serialize_image(
    view: &ImageView<'_>,
    meta: &MetaCollection,
    ts: SystemTime,
    primary: bool,
) -> FitsResult<Vec<u8>> {
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

    write_common_cards(&mut h, view, ts)?;
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
    meta: &MetaCollection,
    ts: SystemTime,
    compression: FitsCompression,
) -> FitsResult<Vec<u8>> {
    let c = compress::build(view, compression)?;
    let mut h = Header::new();

    h.string("XTENSION", "BINTABLE", Some("binary table extension"))?;
    h.integer("BITPIX", 8, None)?;
    h.integer("NAXIS", 2, None)?;
    h.integer("NAXIS1", 8, Some("width of table row (bytes)"))?;
    h.integer("NAXIS2", c.naxis2 as i64, Some("number of tiles"))?;
    h.integer("PCOUNT", c.pcount as i64, Some("heap size (bytes)"))?;
    h.integer("GCOUNT", 1, None)?;
    h.integer("TFIELDS", 1, None)?;
    h.string("TTYPE1", "COMPRESSED_DATA", None)?;
    h.string("TFORM1", "1PB", Some("variable-length array of bytes"))?;

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

    if let Some(bz) = view.bzero() {
        h.integer("BZERO", bz, Some("offset for unsigned integers"))?;
        h.integer("BSCALE", 1, None)?;
    }

    write_common_cards(&mut h, view, ts)?;
    write_metadata(&mut h, meta)?;

    let mut out = h.finish();
    let mut data = c.data;
    card::pad_data(&mut data);
    out.extend_from_slice(&data);
    Ok(out)
}

/// `DATE-OBS`, `COLORSPC`, `BAYERPAT` — the image's own descriptive cards.
fn write_common_cards(h: &mut Header, view: &ImageView<'_>, ts: SystemTime) -> FitsResult<()> {
    h.string(
        "DATE-OBS",
        &datetime::to_iso8601(ts)?,
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

/// Emit one or more cards for every metadata entry (keys sorted for determinism).
fn write_metadata(h: &mut Header, meta: &MetaCollection) -> FitsResult<()> {
    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for key in keys {
        let item = &meta[key];
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
        GenericValue::SystemTime(t) => {
            // The `TIMESTAMP` entry is also emitted as the standard `DATE-OBS` by
            // `write_common_cards`; here it just gets the same treatment as any other
            // `SystemTime` value.
            let (secs, nanos) = datetime::epoch_parts(*t)?;
            h.integer(&format!("{key}_S"), secs, Some("[s] since 1970-01-01"))?;
            h.integer(&format!("{key}_NS"), nanos as i64, Some("[ns]"))?;
            h.string(key, &datetime::to_iso8601(*t)?, comment)?;
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
