//! `RefImage`: pixels plus metadata, serialization and FITS output.

use std::ffi::{c_char, c_void};
use std::time::Duration;

use refimage::chrono::{DateTime, Utc};
use refimage::{
    DynamicImageOwned, DynamicImageRef, FitsCompression, FitsError, FitsWrite, GenericImageOwned,
    GenericValue, Gzip, Hcompress, ImageOwned, ImageProps, InsertValue, Metadata, PixelData,
    PixelType, Quantize, Rice,
};

use crate::RefStatus::{self, *};
use crate::frame::{RefFrame, RefImageInfo};
use crate::{
    Context, Error, REF_FITS_GZIP, REF_FITS_HCOMPRESS, REF_FITS_NONE, REF_FITS_RICE,
    REF_VALUE_COLOR_SPACE, REF_VALUE_DURATION, REF_VALUE_INT, REF_VALUE_REAL, REF_VALUE_STRING,
    REF_VALUE_TIMESTAMP, RefBuffer, RefFitsCompression, RefValueKind, Result, as_mut, as_ref,
    cstr, ffi, free_handle, opt_cstr, put, put_buffer, put_handle, put_str, slice,
};

/// An image: pixels plus metadata (timestamp, exposure, frame id and named
/// FITS-style cards). Opaque; see `ref_image_*`.
pub struct RefImage {
    pub(crate) inner: GenericImageOwned,
}

impl RefImage {
    pub(crate) fn new(inner: GenericImageOwned) -> Self {
        Self { inner }
    }
}

pub(crate) fn fresh_metadata() -> Metadata {
    Metadata::new(Utc::now(), Duration::ZERO)
}

/// Copy `src` into `dst`'s pixels, reusing the buffer (no allocation) when the shape
/// and storage type already match; otherwise replace it.
pub(crate) fn store(dst: &mut GenericImageOwned, src: &DynamicImageRef<'_>) {
    let d = dst.image();
    let fits = d.width() == src.width()
        && d.height() == src.height()
        && d.color_space() == src.color_space()
        && d.pixel_type().storage() == src.pixel_type().storage();
    if fits {
        let bytes = src.as_raw_u8();
        dst.image_mut().as_mut_raw_u8()[..bytes.len()].copy_from_slice(bytes);
        if let DynamicImageOwned::U16(i) = dst.image_mut() {
            i.set_bit_depth(src.pixel_type().bit_depth());
        }
    } else {
        *dst.image_mut() = DynamicImageOwned::from(src);
    }
}

/// An owned image of `spec` holding `bytes` (packed), or zeros when `bytes` is `None`.
pub(crate) fn owned(spec: &refimage::pipeline::ImageSpec, bytes: Option<&[u8]>) -> Result<DynamicImageOwned> {
    let (w, h, cs) = (spec.width, spec.height, spec.cspace.clone());
    let n = spec.elems();
    let fill = |dst: &mut [u8]| {
        if let Some(b) = bytes {
            dst.copy_from_slice(b);
        }
    };
    Ok(match spec.pixel_type.storage() {
        PixelType::U8 => {
            let mut v = vec![0u8; n];
            fill(&mut v);
            ImageOwned::from_owned(v, w, h, cs).status(REF_ERR_IMAGE)?.into()
        }
        PixelType::U16 => {
            let mut v = vec![0u16; n];
            fill(bytemuck::cast_slice_mut(&mut v));
            let bits = (spec.pixel_type != PixelType::U16).then(|| spec.pixel_type.bit_depth());
            ImageOwned::from_owned(v, w, h, cs)
                .status(REF_ERR_IMAGE)?
                .with_bit_depth(bits)
                .into()
        }
        _ => {
            let mut v = vec![0f32; n];
            fill(bytemuck::cast_slice_mut(&mut v));
            ImageOwned::from_owned(v, w, h, cs).status(REF_ERR_IMAGE)?.into()
        }
    })
}

pub(crate) fn duration(seconds: f64, what: &str) -> Result<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| Error::invalid(format!("{what} must be finite and non-negative, not {seconds}")))
}

fn timestamp(ns: i64) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp(ns.div_euclid(1_000_000_000), ns.rem_euclid(1_000_000_000) as u32)
        .ok_or_else(|| Error::invalid(format!("timestamp {ns} ns is out of range")))
}

fn timestamp_ns(t: DateTime<Utc>) -> Result<i64> {
    t.timestamp_nanos_opt()
        .ok_or_else(|| Error::invalid(format!("timestamp {t} does not fit in int64 nanoseconds")))
}

// --- Lifecycle ------------------------------------------------------------------

/// Create an image from `frame`, copying its pixels (or zero-filled when
/// `frame->data` is `NULL`). The timestamp is now, the exposure 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_new(frame: *const RefFrame, out: *mut *mut RefImage) -> RefStatus {
    ffi! {
        let frame = as_ref(frame, "frame")?;
        let spec = frame.spec()?;
        let mut staging = Vec::new();
        let bytes = if frame.data.is_null() {
            RefFrame::packed_len(&spec)?;
            None
        } else {
            Some(frame.packed(&spec, &mut staging)?)
        };
        let img = owned(&spec, bytes)?;
        put_handle(out, RefImage::new(GenericImageOwned::with_metadata(fresh_metadata(), img)), "out")
    }
}

/// A deep copy (pixels and metadata).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_clone(img: *const RefImage, out: *mut *mut RefImage) -> RefStatus {
    ffi! {
        let img = as_ref(img, "img")?;
        put_handle(out, RefImage::new(img.inner.clone()), "out")
    }
}

/// Free an image (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_free(img: *mut RefImage) {
    unsafe { free_handle(img) }
}

/// Copy `src`'s pixels and metadata into `dst`, reusing `dst`'s pixel buffer when the
/// shape and type match (no allocation).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_copy_from(dst: *mut RefImage, src: *const RefImage) -> RefStatus {
    ffi! {
        if std::ptr::eq(dst, src) {
            return Ok(());
        }
        let src = as_ref(src, "src")?;
        let dst = as_mut(dst, "dst")?;
        let (s, d) = (src.inner.image(), dst.inner.image_mut());
        if d.width() == s.width()
            && d.height() == s.height()
            && d.color_space() == s.color_space()
            && d.pixel_type() == s.pixel_type()
        {
            d.as_mut_raw_u8().copy_from_slice(s.as_raw_u8());
        } else {
            *d = s.clone();
        }
        dst.inner.metadata_mut().clone_from(src.inner.metadata());
        Ok(())
    }
}

/// `*out = true` when both images have identical pixels and metadata.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_equal(a: *const RefImage, b: *const RefImage, out: *mut bool) -> RefStatus {
    ffi! {
        let (a, b) = (as_ref(a, "a")?, as_ref(b, "b")?);
        put(out, a.inner == b.inner, "out")
    }
}

// --- Pixels and shape -----------------------------------------------------------

/// Shape and pixel type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_info(img: *const RefImage, out: *mut RefImageInfo) -> RefStatus {
    ffi! {
        let i = &as_ref(img, "img")?.inner;
        put(out, RefImageInfo::of(i.width(), i.height(), &i.color_space(), i.pixel_type()), "out")
    }
}

/// The color space's name: `"gray"`, `"rgb"`, `"rggb"`, `"bggr"`, `"gbrg"`, `"grbg"`,
/// or a custom space's own name. String-out convention.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_color_name(
    img: *const RefImage,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> RefStatus {
    ffi! {
        let cs = as_ref(img, "img")?.inner.color_space();
        put_str(&crate::color_name(&cs), buf, buf_len, out_len)
    }
}

/// A writable pointer to the packed pixels and their size in bytes. Valid until the
/// image is freed or reshaped (by a run or copy that changes its shape or type).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_data(img: *mut RefImage, data: *mut *mut c_void, nbytes: *mut usize) -> RefStatus {
    ffi! {
        let raw = as_mut(img, "img")?.inner.image_mut().as_mut_raw_u8();
        let (p, n) = (raw.as_mut_ptr(), raw.len());
        put(data, p.cast(), "data")?;
        if !nbytes.is_null() {
            nbytes.write(n);
        }
        Ok(())
    }
}

/// A read-only pointer to the packed pixels and their size in bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_data_const(
    img: *const RefImage,
    data: *mut *const c_void,
    nbytes: *mut usize,
) -> RefStatus {
    ffi! {
        let raw = as_ref(img, "img")?.inner.image().as_raw_u8();
        put(data, raw.as_ptr().cast(), "data")?;
        if !nbytes.is_null() {
            nbytes.write(raw.len());
        }
        Ok(())
    }
}

/// Re-tag 16-bit storage as `REF_PIXEL_U10` / `U12` / `U14` / `U16` (values are not
/// changed). Fails on 8-bit and float images.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_pixel_type(img: *mut RefImage, pixel_type: crate::RefPixelType) -> RefStatus {
    ffi! {
        let pt = crate::pixel_type(pixel_type)?;
        let img = as_mut(img, "img")?;
        match (img.inner.image_mut(), pt.storage()) {
            (DynamicImageOwned::U16(i), PixelType::U16) => {
                i.set_bit_depth(pt.bit_depth());
                Ok(())
            }
            _ => Err(Error::invalid("only 16-bit images can be re-tagged, and only to another 16-bit type")),
        }
    }
}

// --- Standard metadata ----------------------------------------------------------

/// Acquisition time, in nanoseconds since 1970-01-01T00:00:00Z.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_timestamp(img: *const RefImage, ns: *mut i64) -> RefStatus {
    ffi! { put(ns, timestamp_ns(as_ref(img, "img")?.inner.timestamp())?, "ns") }
}

/// Set the acquisition time (nanoseconds since the Unix epoch).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_timestamp(img: *mut RefImage, ns: i64) -> RefStatus {
    ffi! {
        let t = timestamp(ns)?;
        as_mut(img, "img")?.inner.metadata_mut().set_timestamp(t);
        Ok(())
    }
}

/// Exposure in seconds (`0` when unknown).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_exposure(img: *const RefImage, seconds: *mut f64) -> RefStatus {
    ffi! { put(seconds, as_ref(img, "img")?.inner.exposure().as_secs_f64(), "seconds") }
}

/// Set the exposure (finite, non-negative seconds).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_exposure(img: *mut RefImage, seconds: f64) -> RefStatus {
    ffi! {
        let d = duration(seconds, "exposure")?;
        as_mut(img, "img")?.inner.set_exposure(d);
        Ok(())
    }
}

/// The acquisition frame counter; `REF_ERR_NOT_FOUND` when none was set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_frame_id(img: *const RefImage, id: *mut u32) -> RefStatus {
    ffi! {
        let v = as_ref(img, "img")?
            .inner
            .frame_id()
            .ok_or_else(|| Error::new(REF_ERR_NOT_FOUND, "the image has no frame id"))?;
        put(id, v, "id")
    }
}

/// Set the acquisition frame counter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_frame_id(img: *mut RefImage, id: u32) -> RefStatus {
    ffi! {
        as_mut(img, "img")?.inner.set_frame_id(id);
        Ok(())
    }
}

// --- Metadata cards -------------------------------------------------------------

fn card<'a>(img: &'a RefImage, key: &str) -> Result<&'a GenericValue> {
    img.inner
        .key(key)
        .map(|item| item.value())
        .ok_or_else(|| Error::new(REF_ERR_NOT_FOUND, format!("no metadata key {key:?}")))
}

fn wrong_type(key: &str, want: &str) -> Error {
    Error::new(REF_ERR_TYPE, format!("metadata key {key:?} is not {want}"))
}

fn insert<T: InsertValue>(meta: &mut Metadata, key: &str, value: T) -> Result<()> {
    if meta.get(key).is_some() {
        meta.replace(key, value).status(REF_ERR_METADATA)?;
    } else {
        meta.insert(key, value).status(REF_ERR_METADATA)?;
    }
    Ok(())
}

/// Insert or replace a card, with an optional comment.
unsafe fn set_card<T: InsertValue>(
    img: *mut RefImage,
    key: *const c_char,
    comment: *const c_char,
    value: T,
) -> Result<()>
where
    for<'c> (T, &'c str): InsertValue,
{
    let img = unsafe { as_mut(img, "img") }?;
    let key = unsafe { cstr(key, "key") }?;
    let meta = img.inner.metadata_mut();
    match unsafe { opt_cstr(comment, "comment") }? {
        Some(c) => insert(meta, key, (value, c)),
        None => insert(meta, key, value),
    }
}

/// Number of metadata cards (not counting timestamp, exposure and frame id).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_key_count(img: *const RefImage, count: *mut usize) -> RefStatus {
    ffi! { put(count, as_ref(img, "img")?.inner.metadata().len(), "count") }
}

/// The name of card `index` (`0 .. ref_image_key_count`), in insertion order.
/// String-out convention; `REF_ERR_NOT_FOUND` past the end.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_key_at(
    img: *const RefImage,
    index: usize,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> RefStatus {
    ffi! {
        let img = as_ref(img, "img")?;
        let (k, _) = img
            .inner
            .metadata()
            .iter()
            .nth(index)
            .ok_or_else(|| Error::new(REF_ERR_NOT_FOUND, format!("no metadata card at index {index}")))?;
        put_str(k, buf, buf_len, out_len)
    }
}

/// The type of card `key` (keys are case-insensitive). `REF_ERR_NOT_FOUND` if absent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_value_kind(
    img: *const RefImage,
    key: *const c_char,
    kind: *mut RefValueKind,
) -> RefStatus {
    ffi! {
        let k = match card(as_ref(img, "img")?, cstr(key, "key")?)? {
            GenericValue::Integer(_) => REF_VALUE_INT,
            GenericValue::Real(_) => REF_VALUE_REAL,
            GenericValue::String(_) => REF_VALUE_STRING,
            GenericValue::Timestamp(_) => REF_VALUE_TIMESTAMP,
            GenericValue::Duration(_) => REF_VALUE_DURATION,
            _ => REF_VALUE_COLOR_SPACE,
        };
        put(kind, k, "kind")
    }
}

/// An integer card. `REF_ERR_TYPE` if it holds anything else.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_int(img: *const RefImage, key: *const c_char, value: *mut i64) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        match card(as_ref(img, "img")?, key)? {
            GenericValue::Integer(i) => put(value, *i, "value"),
            _ => Err(wrong_type(key, "an integer")),
        }
    }
}

/// A real card; integer cards are converted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_real(img: *const RefImage, key: *const c_char, value: *mut f64) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        match card(as_ref(img, "img")?, key)? {
            GenericValue::Real(f) => put(value, *f, "value"),
            GenericValue::Integer(i) => put(value, *i as f64, "value"),
            _ => Err(wrong_type(key, "a number")),
        }
    }
}

/// A string card (or a color-space card, as its name). String-out convention.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_string(
    img: *const RefImage,
    key: *const c_char,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        match card(as_ref(img, "img")?, key)? {
            GenericValue::String(s) => put_str(s, buf, buf_len, out_len),
            GenericValue::ColorSpace(cs) => put_str(&crate::color_name(cs), buf, buf_len, out_len),
            _ => Err(wrong_type(key, "a string")),
        }
    }
}

/// A timestamp card, in nanoseconds since the Unix epoch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_timestamp(img: *const RefImage, key: *const c_char, ns: *mut i64) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        match card(as_ref(img, "img")?, key)? {
            GenericValue::Timestamp(t) => put(ns, timestamp_ns(*t)?, "ns"),
            _ => Err(wrong_type(key, "a timestamp")),
        }
    }
}

/// A duration card, in seconds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_duration(img: *const RefImage, key: *const c_char, seconds: *mut f64) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        match card(as_ref(img, "img")?, key)? {
            GenericValue::Duration(d) => put(seconds, d.as_secs_f64(), "seconds"),
            _ => Err(wrong_type(key, "a duration")),
        }
    }
}

/// The comment of card `key`. String-out convention; an empty string when the card
/// has no comment, `REF_ERR_NOT_FOUND` when the card does not exist.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_get_comment(
    img: *const RefImage,
    key: *const c_char,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> RefStatus {
    ffi! {
        let key = cstr(key, "key")?;
        let item = as_ref(img, "img")?
            .inner
            .key(key)
            .ok_or_else(|| Error::new(REF_ERR_NOT_FOUND, format!("no metadata key {key:?}")))?;
        put_str(item.comment().unwrap_or(""), buf, buf_len, out_len)
    }
}

/// Insert or replace an integer card. Keys are stored upper-case, at most 80
/// characters; `TIMESTAMP`, `EXPOSURE` and `FRAMEID` are reserved (use the dedicated
/// setters). `comment` may be `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_int(
    img: *mut RefImage,
    key: *const c_char,
    value: i64,
    comment: *const c_char,
) -> RefStatus {
    ffi! { set_card(img, key, comment, value) }
}

/// Insert or replace a real card (see `ref_image_set_int`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_real(
    img: *mut RefImage,
    key: *const c_char,
    value: f64,
    comment: *const c_char,
) -> RefStatus {
    ffi! { set_card(img, key, comment, value) }
}

/// Insert or replace a string card (see `ref_image_set_int`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_string(
    img: *mut RefImage,
    key: *const c_char,
    value: *const c_char,
    comment: *const c_char,
) -> RefStatus {
    ffi! {
        let value = cstr(value, "value")?;
        set_card(img, key, comment, value)
    }
}

/// Insert or replace a timestamp card, nanoseconds since the Unix epoch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_timestamp_key(
    img: *mut RefImage,
    key: *const c_char,
    ns: i64,
    comment: *const c_char,
) -> RefStatus {
    ffi! { set_card(img, key, comment, timestamp(ns)?) }
}

/// Insert or replace a duration card, in seconds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_set_duration(
    img: *mut RefImage,
    key: *const c_char,
    seconds: f64,
    comment: *const c_char,
) -> RefStatus {
    ffi! { set_card(img, key, comment, duration(seconds, "duration")?) }
}

/// Remove card `key`. `REF_ERR_NOT_FOUND` if absent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_remove_key(img: *mut RefImage, key: *const c_char) -> RefStatus {
    ffi! {
        let img = as_mut(img, "img")?;
        let key = cstr(key, "key")?;
        if img.inner.key(key).is_none() {
            return Err(Error::new(REF_ERR_NOT_FOUND, format!("no metadata key {key:?}")));
        }
        img.inner.remove_key(key).status(REF_ERR_METADATA)?;
        Ok(())
    }
}

// --- Serialization --------------------------------------------------------------

/// Serialize pixels and metadata (bincode; the same bytes as Rust's
/// `bincode::serialize(&GenericImageOwned)` and Python's `Image.to_bytes()`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_to_bytes(img: *const RefImage, out: *mut RefBuffer) -> RefStatus {
    ffi! {
        let bytes = bincode::serialize(&as_ref(img, "img")?.inner).status(REF_ERR_IMAGE)?;
        put_buffer(out, bytes)
    }
}

/// Decode an image written by `ref_image_to_bytes` (or the Rust / Python equivalents).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_from_bytes(data: *const u8, len: usize, out: *mut *mut RefImage) -> RefStatus {
    ffi! {
        let bytes = slice(data, len, "data")?;
        let inner: GenericImageOwned = bincode::deserialize(bytes)
            .map_err(|e| Error::new(REF_ERR_DECODE, format!("not a serialized refimage image: {e}")))?;
        put_handle(out, RefImage::new(inner), "out")
    }
}

// --- FITS -----------------------------------------------------------------------

/// FITS output settings. Start from `ref_fits_options_default` and change fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefFitsOptions {
    /// A `REF_FITS_*` constant.
    pub compression: RefFitsCompression,
    /// GZIP level 0-9; `-1` for the default.
    pub gzip_level: i32,
    /// Rows per compressed tile; `0` picks automatically.
    pub tile_rows: usize,
    /// HCOMPRESS scale (`0` = lossless).
    pub hcompress_scale: i32,
    /// HCOMPRESS smoothing on decompression.
    pub hcompress_smooth: bool,
    /// Float quantization level for RICE / HCOMPRESS; `0` for the default.
    pub quantize_level: f64,
}

/// Uncompressed, with every tuning field at its default.
#[unsafe(no_mangle)]
pub extern "C" fn ref_fits_options_default() -> RefFitsOptions {
    RefFitsOptions {
        compression: REF_FITS_NONE,
        gzip_level: -1,
        tile_rows: 0,
        hcompress_scale: 0,
        hcompress_smooth: false,
        quantize_level: 0.0,
    }
}

fn fits_compression(o: Option<&RefFitsOptions>) -> Result<FitsCompression> {
    let Some(o) = o else {
        return Ok(FitsCompression::NONE);
    };
    let q = if o.quantize_level > 0.0 {
        Quantize::new().level(o.quantize_level)
    } else if o.quantize_level == 0.0 {
        Quantize::new()
    } else {
        return Err(Error::invalid(format!("quantize_level must be >= 0, not {}", o.quantize_level)));
    };
    let tiles = (o.tile_rows > 0).then_some(o.tile_rows);
    Ok(match o.compression {
        REF_FITS_NONE => FitsCompression::NONE,
        REF_FITS_GZIP => {
            let g = match o.gzip_level {
                -1 => Gzip::new(),
                l @ 0..=9 => Gzip::new().level(l as u8),
                l => return Err(Error::invalid(format!("gzip_level must be 0-9 or -1, not {l}"))),
            };
            match tiles {
                Some(t) => g.tile_rows(t).into(),
                None => g.into(),
            }
        }
        REF_FITS_RICE => {
            let r = Rice::new().quantize(q);
            match tiles {
                Some(t) => r.tile_rows(t).into(),
                None => r.into(),
            }
        }
        REF_FITS_HCOMPRESS => {
            let h = Hcompress::new()
                .scale(o.hcompress_scale)
                .smooth(o.hcompress_smooth)
                .quantize(q);
            match tiles {
                Some(t) => h.tile_rows(t).into(),
                None => h.into(),
            }
        }
        c => return Err(Error::invalid(format!("unknown FITS compression {c}"))),
    })
}

fn fits_err(e: FitsError) -> Error {
    match e {
        FitsError::Io(io) => Error::new(REF_ERR_IO, io.to_string()),
        other => Error::new(REF_ERR_FITS, other.to_string()),
    }
}

/// Write the image and its metadata to a FITS file. `options` may be `NULL`
/// (uncompressed). Without `overwrite`, an existing file is an `REF_ERR_IO` error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_write_fits(
    img: *const RefImage,
    path: *const c_char,
    options: *const RefFitsOptions,
    overwrite: bool,
) -> RefStatus {
    ffi! {
        let img = as_ref(img, "img")?;
        let path = cstr(path, "path")?;
        let c = fits_compression(options.as_ref())?;
        img.inner.write_fits(path, c, overwrite).map_err(fits_err)?;
        Ok(())
    }
}

/// The FITS file `ref_image_write_fits` would write, in memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_image_fits_bytes(
    img: *const RefImage,
    options: *const RefFitsOptions,
    out: *mut RefBuffer,
) -> RefStatus {
    ffi! {
        let img = as_ref(img, "img")?;
        let c = fits_compression(options.as_ref())?;
        let bytes = img.inner.fits_bytes(c).map_err(fits_err)?;
        put_buffer(out, bytes)
    }
}
