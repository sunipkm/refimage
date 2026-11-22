//! C API for `refimage`: `librefimage_c` (shared and static) and `refimage.h`.
//!
//! The header is generated from this crate by cbindgen and committed at
//! `include/refimage.h`; `cargo test -p refimage-capi` fails when it is stale, and
//! `REFIMAGE_BLESS=1 cargo test -p refimage-capi --test header` rewrites it.
//!
//! # Conventions
//!
//! - **Handles.** `RefImage`, `RefPipeline` and `RefRunner` are opaque. Each is
//!   created by a `ref_*_new` / `ref_*_compile` / `ref_*_from_*` function that writes
//!   it through an out-pointer, and released with the matching `ref_*_free` (which
//!   accepts `NULL`). Handles may be moved between threads but not used from two
//!   threads at once; distinct handles are independent.
//! - **Errors.** Every fallible function returns a `RefStatus`; `REF_OK` is `0`. On
//!   failure, `ref_last_error` returns a message for the calling thread, valid until
//!   that thread's next failing call. Out-parameters are only written on success.
//!   Rust panics are caught at the boundary and reported as `REF_ERR_PANIC`.
//! - **Enumerations** are `uint32_t` typedefs with `REF_*` constants, so an
//!   out-of-range value is reported as `REF_ERR_INVALID_ARG` rather than being
//!   undefined behaviour.
//! - **Strings** in are NUL-terminated UTF-8. Strings out are copied into a caller
//!   buffer: `(char *buf, size_t buf_len, size_t *out_len)`. `*out_len` (when non-null)
//!   receives the length without the terminator; pass `buf = NULL, buf_len = 0` to
//!   query it. A buffer shorter than `*out_len + 1` gives `REF_ERR_BUFFER_TOO_SMALL`.
//! - **Byte blobs** out (bincode, FITS, JSON) come back as a `RefBuffer` that the
//!   caller releases with `ref_buffer_free`.
//! - **Pixels** are row-major with channels interleaved, in native byte order:
//!   `uint8_t` for `REF_PIXEL_U8`, `uint16_t` for `REF_PIXEL_U10` ... `REF_PIXEL_U16`
//!   (the 10/12/14-bit types are right-aligned in 16 bits) and `float` for
//!   `REF_PIXEL_F32`.
//!
//! # Safety
//!
//! Every pointer argument must be `NULL` or valid for the access the function
//! documents, and handles must come from this library and not have been freed. The
//! header marks each pointer `_Nonnull` / `_Nullable` (Clang's `_Nonnull` /
//! `_Nullable`); a `NULL` passed to a `_Nonnull` parameter is still detected and
//! reported as `REF_ERR_NULL`. A parameter is nullable exactly when its docs say
//! "`name` may be `NULL`" (see tests/header.rs).

#![allow(clippy::missing_safety_doc)]
mod exposure;
mod frame;
mod image;
mod pipeline;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

use refimage::pipeline::{BoundaryMode, InterpOrder, Interpolation, ResizeFilter, Strategy};
use refimage::{BayerPattern, ColorSpace, DemosaicMethod, PixelType};

// Everything public is part of the C API (and usable from Rust through the rlib).
pub use exposure::*;
pub use frame::*;
pub use image::*;
pub use pipeline::*;

// --- Status and errors ----------------------------------------------------------

/// Result of every fallible call. `REF_OK` (0) is success.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum RefStatus {
    /// Success.
    REF_OK = 0,
    /// A required pointer argument was `NULL`.
    REF_ERR_NULL = 1,
    /// An argument is out of range or inconsistent (unknown enum value, bad size, ...).
    REF_ERR_INVALID_ARG = 2,
    /// A string argument is not valid UTF-8.
    REF_ERR_UTF8 = 3,
    /// The caller's buffer is too small; the required size was written to `out_len`.
    REF_ERR_BUFFER_TOO_SMALL = 4,
    /// The metadata key (or frame id) does not exist.
    REF_ERR_NOT_FOUND = 5,
    /// The metadata value exists but has a different type.
    REF_ERR_TYPE = 6,
    /// The pixel data does not describe a valid image.
    REF_ERR_IMAGE = 7,
    /// The pipeline is invalid for this input, or failed to run.
    REF_ERR_PIPELINE = 8,
    /// A metadata key or value was rejected (reserved, too long, ...).
    REF_ERR_METADATA = 9,
    /// The image cannot be encoded as FITS with these options.
    REF_ERR_FITS = 10,
    /// An operating-system I/O error.
    REF_ERR_IO = 11,
    /// The bytes are not a serialized image / the JSON is not a pipeline.
    REF_ERR_DECODE = 12,
    /// The optimum-exposure calculation failed.
    REF_ERR_EXPOSURE = 13,
    /// A bug: Rust panicked inside the library. The message says where.
    REF_ERR_PANIC = 255,
}

use RefStatus::*;

pub(crate) struct Error {
    status: RefStatus,
    msg: String,
}

impl Error {
    pub(crate) fn new(status: RefStatus, msg: impl Into<String>) -> Self {
        Self {
            status,
            msg: msg.into(),
        }
    }

    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Self::new(REF_ERR_INVALID_ARG, msg)
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

/// `map_err` adapter: wrap any displayable error with `status`.
pub(crate) fn with(status: RefStatus) -> impl FnOnce(&dyn std::fmt::Display) -> Error {
    move |e| Error::new(status, e.to_string())
}

/// Shorthand for `.map_err(|e| with(status)(&e))`.
pub(crate) trait Context<T> {
    fn status(self, status: RefStatus) -> Result<T>;
}

impl<T, E: std::fmt::Display> Context<T> for std::result::Result<T, E> {
    fn status(self, status: RefStatus) -> Result<T> {
        self.map_err(|e| with(status)(&e))
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg.replace('\0', "\\0")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Run an FFI body: convert its error or panic into a status and the thread's last
/// error message. Success leaves the last error untouched (no allocation).
pub(crate) fn guard(f: impl FnOnce() -> Result<()>) -> RefStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => REF_OK,
        Ok(Err(e)) => {
            set_last_error(&e.msg);
            e.status
        }
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            set_last_error(&format!("internal error (panic): {what}"));
            REF_ERR_PANIC
        }
    }
}

/// `guard` around an `unsafe` body.
macro_rules! ffi {
    ($($body:tt)*) => {
        $crate::guard(|| {
            #[allow(unused_unsafe)]
            unsafe { $($body)* }
        })
    };
}
pub(crate) use ffi;

// --- Pointer helpers ------------------------------------------------------------

pub(crate) unsafe fn as_ref<'a, T>(p: *const T, what: &str) -> Result<&'a T> {
    // SAFETY: the caller guarantees `p` is null or valid for reads.
    unsafe { p.as_ref() }.ok_or_else(|| Error::new(REF_ERR_NULL, format!("{what} is NULL")))
}

pub(crate) unsafe fn as_mut<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    // SAFETY: the caller guarantees `p` is null or valid for reads and writes.
    unsafe { p.as_mut() }.ok_or_else(|| Error::new(REF_ERR_NULL, format!("{what} is NULL")))
}

/// Write `v` through an out-pointer.
pub(crate) unsafe fn put<T>(p: *mut T, v: T, what: &str) -> Result<()> {
    if p.is_null() {
        return Err(Error::new(REF_ERR_NULL, format!("{what} is NULL")));
    }
    // SAFETY: non-null, and the caller guarantees it is valid for writes. `write`
    // because the target may be uninitialized.
    unsafe { p.write(v) };
    Ok(())
}

pub(crate) unsafe fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        return Err(Error::new(REF_ERR_NULL, format!("{what} is NULL")));
    }
    // SAFETY: the caller guarantees a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| Error::new(REF_ERR_UTF8, format!("{what} is not valid UTF-8")))
}

pub(crate) unsafe fn opt_cstr<'a>(p: *const c_char, what: &str) -> Result<Option<&'a str>> {
    if p.is_null() {
        Ok(None)
    } else {
        unsafe { cstr(p, what) }.map(Some)
    }
}

/// Copy `s` into a caller buffer (see the crate docs for the convention).
pub(crate) unsafe fn put_str(
    s: &str,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> Result<()> {
    if !out_len.is_null() {
        // SAFETY: non-null; the caller guarantees it is writable.
        unsafe { out_len.write(s.len()) };
    }
    if buf.is_null() && buf_len == 0 {
        return Ok(());
    }
    if buf.is_null() {
        return Err(Error::new(REF_ERR_NULL, "buf is NULL but buf_len is not 0"));
    }
    if buf_len <= s.len() {
        return Err(Error::new(
            REF_ERR_BUFFER_TOO_SMALL,
            format!(
                "buffer of {buf_len} bytes cannot hold {} bytes plus NUL",
                s.len()
            ),
        ));
    }
    // SAFETY: `buf` is valid for `buf_len > s.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf.cast::<u8>(), s.len());
        buf.add(s.len()).write(0);
    }
    Ok(())
}

/// A slice from a C pointer and element count (`NULL` allowed when `len == 0`).
pub(crate) unsafe fn slice<'a, T>(p: *const T, len: usize, what: &str) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(Error::new(REF_ERR_NULL, format!("{what} is NULL")));
    }
    // SAFETY: the caller guarantees `len` readable elements.
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

/// Box a handle into an out-pointer.
pub(crate) unsafe fn put_handle<T>(out: *mut *mut T, v: T, what: &str) -> Result<()> {
    if out.is_null() {
        return Err(Error::new(REF_ERR_NULL, format!("{what} is NULL")));
    }
    // SAFETY: checked non-null; caller guarantees writable.
    unsafe { out.write(Box::into_raw(Box::new(v))) };
    Ok(())
}

/// Free a boxed handle (`NULL` is a no-op).
pub(crate) unsafe fn free_handle<T>(p: *mut T) {
    if !p.is_null() {
        // SAFETY: `p` came from `put_handle` and has not been freed.
        drop(unsafe { Box::from_raw(p) });
    }
}

// --- Library info ---------------------------------------------------------------

/// Version of the C API's ABI. Incremented on any incompatible change to a function
/// signature, struct layout or constant; compare with `ref_abi_version` at run time.
pub const REF_ABI_VERSION: u32 = 1;

/// The library version string (e.g. `"1.0.0-pre6"`). Static; do not free.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// The ABI version the library was built with (`REF_ABI_VERSION` of its header).
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_abi_version() -> u32 {
    REF_ABI_VERSION
}

/// The message of the calling thread's most recent error, or `""` if there was none.
/// Valid until that thread's next failing call; do not free.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => c"".as_ptr(),
    })
}

/// The constant's name for a status code (`"REF_ERR_IO"`, ...). Static; do not free.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn ref_status_name(status: c_int) -> *const c_char {
    let s: &CStr = match status {
        0 => c"REF_OK",
        1 => c"REF_ERR_NULL",
        2 => c"REF_ERR_INVALID_ARG",
        3 => c"REF_ERR_UTF8",
        4 => c"REF_ERR_BUFFER_TOO_SMALL",
        5 => c"REF_ERR_NOT_FOUND",
        6 => c"REF_ERR_TYPE",
        7 => c"REF_ERR_IMAGE",
        8 => c"REF_ERR_PIPELINE",
        9 => c"REF_ERR_METADATA",
        10 => c"REF_ERR_FITS",
        11 => c"REF_ERR_IO",
        12 => c"REF_ERR_DECODE",
        13 => c"REF_ERR_EXPOSURE",
        255 => c"REF_ERR_PANIC",
        _ => c"unknown status",
    };
    s.as_ptr()
}

// --- Byte buffers ---------------------------------------------------------------

/// A library-allocated byte blob. `data[len]` is always a NUL byte, so text (JSON)
/// can be used as a C string directly. Release with `ref_buffer_free`.
#[repr(C)]
#[derive(Debug)]
pub struct RefBuffer {
    /// The bytes. May be `NULL` (after `ref_buffer_free`).
    pub data: *mut u8,
    /// Number of bytes, not counting the trailing NUL.
    pub len: usize,
}

impl RefBuffer {
    pub(crate) fn from_vec(mut v: Vec<u8>) -> Self {
        let len = v.len();
        v.push(0);
        let data = Box::into_raw(v.into_boxed_slice()).cast::<u8>();
        Self { data, len }
    }
}

/// Free a `RefBuffer`'s bytes and reset it to `{NULL, 0}`. `NULL` / an already
/// freed buffer is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ref_buffer_free(buf: *mut RefBuffer) {
    // SAFETY: the caller passes NULL or a buffer this library filled.
    let Some(b) = (unsafe { buf.as_mut() }) else {
        return;
    };
    if !b.data.is_null() {
        // SAFETY: `data` is a boxed slice of `len + 1` bytes from `from_vec`.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.data, b.len + 1)) });
    }
    b.data = std::ptr::null_mut();
    b.len = 0;
}

pub(crate) unsafe fn put_buffer(out: *mut RefBuffer, v: Vec<u8>) -> Result<()> {
    if out.is_null() {
        return Err(Error::new(REF_ERR_NULL, "out is NULL"));
    }
    // SAFETY: checked non-null; the caller guarantees it is writable.
    unsafe { out.write(RefBuffer::from_vec(v)) };
    Ok(())
}

// --- Enumerations ---------------------------------------------------------------

/// Sample type (see the pixel layout in the module docs).
pub type RefPixelType = u32;
/// 8-bit unsigned (`uint8_t`).
pub const REF_PIXEL_U8: RefPixelType = 0;
/// 10-bit unsigned, right-aligned in a `uint16_t`.
pub const REF_PIXEL_U10: RefPixelType = 1;
/// 12-bit unsigned, right-aligned in a `uint16_t`.
pub const REF_PIXEL_U12: RefPixelType = 2;
/// 14-bit unsigned, right-aligned in a `uint16_t`.
pub const REF_PIXEL_U14: RefPixelType = 3;
/// 16-bit unsigned (`uint16_t`).
pub const REF_PIXEL_U16: RefPixelType = 4;
/// 32-bit float (`float`), nominally in `[0, 1]`.
pub const REF_PIXEL_F32: RefPixelType = 5;

/// Color space. Gray and Bayer images have one channel, RGB three; a custom color
/// space carries its own channel count and name.
pub type RefColor = u32;
/// Single-channel intensity.
pub const REF_COLOR_GRAY: RefColor = 0;
/// Three interleaved channels, R G B.
pub const REF_COLOR_RGB: RefColor = 1;
/// Single-channel Bayer mosaic, top-left 2x2 cell `R G / G B`.
pub const REF_COLOR_BAYER_RGGB: RefColor = 2;
/// Bayer mosaic `B G / G R`.
pub const REF_COLOR_BAYER_BGGR: RefColor = 3;
/// Bayer mosaic `G B / R G`.
pub const REF_COLOR_BAYER_GBRG: RefColor = 4;
/// Bayer mosaic `G R / B G`.
pub const REF_COLOR_BAYER_GRBG: RefColor = 5;
/// Any other layout: `channels` interleaved channels and a free-form name.
pub const REF_COLOR_CUSTOM: RefColor = 6;

/// Demosaicing method for `ref_pipeline_debayer`.
pub type RefDemosaic = u32;
/// Replicate each sample into its 2x2 cell (no interpolation).
pub const REF_DEMOSAIC_NONE: RefDemosaic = 0;
/// Nearest-neighbour.
pub const REF_DEMOSAIC_NEAREST: RefDemosaic = 1;
/// Bilinear.
pub const REF_DEMOSAIC_LINEAR: RefDemosaic = 2;
/// Bicubic.
pub const REF_DEMOSAIC_CUBIC: RefDemosaic = 3;

/// Resampling filter for `ref_pipeline_resize_to_fit`.
pub type RefResizeFilter = u32;
/// Triangle filter: fast, mildly soft.
pub const REF_RESIZE_BILINEAR: RefResizeFilter = 0;
/// Catmull-Rom cubic: sharper, slight overshoot.
pub const REF_RESIZE_BICUBIC: RefResizeFilter = 1;
/// Lanczos, a = 3: highest fidelity, most ringing.
pub const REF_RESIZE_LANCZOS3: RefResizeFilter = 2;

/// How warps sample outside the input (`scipy.ndimage` names).
pub type RefBoundary = u32;
/// Outside is `cval`.
pub const REF_BOUNDARY_CONSTANT: RefBoundary = 0;
/// Repeat the edge pixel (scikit-image `edge`).
pub const REF_BOUNDARY_NEAREST: RefBoundary = 1;
/// Reflect about the edge pixel's centre: `d c b | a b c d | c b a` (scikit-image `symmetric`).
pub const REF_BOUNDARY_MIRROR: RefBoundary = 2;
/// Reflect about the edge: `d c b a | a b c d | d c b a`.
pub const REF_BOUNDARY_REFLECT: RefBoundary = 3;
/// Wrap around.
pub const REF_BOUNDARY_WRAP: RefBoundary = 4;

/// Runner execution strategy (see `RefStrategy`).
pub type RefStrategyKind = u32;
/// Cut frames into row bands (optionally column tiles), optionally in parallel.
pub const REF_STRATEGY_TILED: RefStrategyKind = 0;
/// Sweep every step over the whole frame in turn.
pub const REF_STRATEGY_SEQUENTIAL: RefStrategyKind = 1;

/// FITS compression algorithm (see `RefFitsOptions`).
pub type RefFitsCompression = u32;
/// Uncompressed primary HDU.
pub const REF_FITS_NONE: RefFitsCompression = 0;
/// Tile-compressed with GZIP.
pub const REF_FITS_GZIP: RefFitsCompression = 1;
/// Tile-compressed with Rice (lossless for integers; floats are quantized).
pub const REF_FITS_RICE: RefFitsCompression = 2;
/// Tile-compressed with HCOMPRESS (tiles at least 4x4).
pub const REF_FITS_HCOMPRESS: RefFitsCompression = 3;

/// Type of a metadata value.
pub type RefValueKind = u32;
/// `int64_t`.
pub const REF_VALUE_INT: RefValueKind = 0;
/// `double`.
pub const REF_VALUE_REAL: RefValueKind = 1;
/// UTF-8 string.
pub const REF_VALUE_STRING: RefValueKind = 2;
/// UTC timestamp, read as nanoseconds since the Unix epoch.
pub const REF_VALUE_TIMESTAMP: RefValueKind = 3;
/// Duration, read as seconds.
pub const REF_VALUE_DURATION: RefValueKind = 4;
/// A color space (read with `ref_image_get_string` as its name).
pub const REF_VALUE_COLOR_SPACE: RefValueKind = 5;

// --- Conversions ----------------------------------------------------------------

pub(crate) fn pixel_type(v: RefPixelType) -> Result<PixelType> {
    Ok(match v {
        REF_PIXEL_U8 => PixelType::U8,
        REF_PIXEL_U10 => PixelType::U10,
        REF_PIXEL_U12 => PixelType::U12,
        REF_PIXEL_U14 => PixelType::U14,
        REF_PIXEL_U16 => PixelType::U16,
        REF_PIXEL_F32 => PixelType::F32,
        _ => return Err(Error::invalid(format!("unknown pixel type {v}"))),
    })
}

pub(crate) fn pixel_type_code(pt: PixelType) -> RefPixelType {
    match pt {
        PixelType::U8 => REF_PIXEL_U8,
        PixelType::U10 => REF_PIXEL_U10,
        PixelType::U12 => REF_PIXEL_U12,
        PixelType::U14 => REF_PIXEL_U14,
        PixelType::U16 => REF_PIXEL_U16,
        _ => REF_PIXEL_F32,
    }
}

/// Bytes per stored sample.
pub(crate) fn item_size(pt: PixelType) -> usize {
    match pt.storage() {
        PixelType::U8 => 1,
        PixelType::U16 => 2,
        _ => 4,
    }
}

pub(crate) fn color_space(kind: RefColor, channels: u8, name: Option<&str>) -> Result<ColorSpace> {
    Ok(match kind {
        REF_COLOR_GRAY => ColorSpace::Gray,
        REF_COLOR_RGB => ColorSpace::Rgb,
        REF_COLOR_BAYER_RGGB => ColorSpace::Bayer(BayerPattern::Rggb),
        REF_COLOR_BAYER_BGGR => ColorSpace::Bayer(BayerPattern::Bggr),
        REF_COLOR_BAYER_GBRG => ColorSpace::Bayer(BayerPattern::Gbrg),
        REF_COLOR_BAYER_GRBG => ColorSpace::Bayer(BayerPattern::Grbg),
        REF_COLOR_CUSTOM => {
            if channels == 0 {
                return Err(Error::invalid(
                    "a custom color space needs at least one channel",
                ));
            }
            ColorSpace::Custom(channels, name.unwrap_or("custom").to_owned())
        }
        _ => return Err(Error::invalid(format!("unknown color space {kind}"))),
    })
}

pub(crate) fn color_code(cs: &ColorSpace) -> RefColor {
    match cs {
        ColorSpace::Gray => REF_COLOR_GRAY,
        ColorSpace::Rgb => REF_COLOR_RGB,
        ColorSpace::Bayer(BayerPattern::Rggb) => REF_COLOR_BAYER_RGGB,
        ColorSpace::Bayer(BayerPattern::Bggr) => REF_COLOR_BAYER_BGGR,
        ColorSpace::Bayer(BayerPattern::Gbrg) => REF_COLOR_BAYER_GBRG,
        ColorSpace::Bayer(BayerPattern::Grbg) => REF_COLOR_BAYER_GRBG,
        _ => REF_COLOR_CUSTOM,
    }
}

/// `"gray"`, `"rgb"`, `"rggb"`, ... or a custom space's own name.
pub(crate) fn color_name(cs: &ColorSpace) -> std::borrow::Cow<'_, str> {
    match cs {
        ColorSpace::Gray => "gray".into(),
        ColorSpace::Rgb => "rgb".into(),
        ColorSpace::Bayer(BayerPattern::Rggb) => "rggb".into(),
        ColorSpace::Bayer(BayerPattern::Bggr) => "bggr".into(),
        ColorSpace::Bayer(BayerPattern::Gbrg) => "gbrg".into(),
        ColorSpace::Bayer(BayerPattern::Grbg) => "grbg".into(),
        ColorSpace::Custom(_, name) => name.as_str().into(),
        other => format!("{other:?}").into(),
    }
}

pub(crate) fn demosaic(v: RefDemosaic) -> Result<DemosaicMethod> {
    Ok(match v {
        REF_DEMOSAIC_NONE => DemosaicMethod::None,
        REF_DEMOSAIC_NEAREST => DemosaicMethod::Nearest,
        REF_DEMOSAIC_LINEAR => DemosaicMethod::Linear,
        REF_DEMOSAIC_CUBIC => DemosaicMethod::Cubic,
        _ => return Err(Error::invalid(format!("unknown demosaic method {v}"))),
    })
}

pub(crate) fn resize_filter(v: RefResizeFilter) -> Result<ResizeFilter> {
    Ok(match v {
        REF_RESIZE_BILINEAR => ResizeFilter::Bilinear,
        REF_RESIZE_BICUBIC => ResizeFilter::Bicubic,
        REF_RESIZE_LANCZOS3 => ResizeFilter::Lanczos3,
        _ => return Err(Error::invalid(format!("unknown resize filter {v}"))),
    })
}

pub(crate) fn boundary(v: RefBoundary) -> Result<BoundaryMode> {
    Ok(match v {
        REF_BOUNDARY_CONSTANT => BoundaryMode::Constant,
        REF_BOUNDARY_NEAREST => BoundaryMode::Nearest,
        REF_BOUNDARY_MIRROR => BoundaryMode::Mirror,
        REF_BOUNDARY_REFLECT => BoundaryMode::Reflect,
        REF_BOUNDARY_WRAP => BoundaryMode::Wrap,
        _ => return Err(Error::invalid(format!("unknown boundary mode {v}"))),
    })
}

pub(crate) fn interpolation(i: Option<&RefInterpolation>) -> Result<Interpolation> {
    let Some(i) = i else {
        return Ok(Interpolation::default());
    };
    let order = InterpOrder::from_u8(i.order).status(REF_ERR_INVALID_ARG)?;
    Ok(Interpolation::new(order, boundary(i.mode)?, i.cval).with_prefilter(i.prefilter))
}

pub(crate) fn strategy(s: Option<&RefStrategy>) -> Result<Strategy> {
    let s = s
        .copied()
        .unwrap_or_else(|| pipeline::ref_strategy_default());
    match s.kind {
        REF_STRATEGY_TILED => Ok(Strategy::Tiled {
            tile_rows: s.tile_rows,
            tile_cols: s.tile_cols,
            parallel: s.parallel,
        }),
        REF_STRATEGY_SEQUENTIAL => Ok(Strategy::Sequential),
        k => Err(Error::invalid(format!("unknown strategy {k}"))),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn soname_matches_abi_version() {
        assert_eq!(
            env!("REFIMAGE_SONAME_ABI"),
            super::REF_ABI_VERSION.to_string()
        );
    }
}
