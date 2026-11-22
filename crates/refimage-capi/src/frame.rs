//! `RefFrame`: borrowed caller pixels, and `RefImageInfo`: a shape report.

use std::ffi::{c_char, c_void};

use refimage::pipeline::{Frame, ImageSpec};
use refimage::{ColorSpace, ImageProps, PixelType};

use crate::RefStatus::*;
use crate::{
    Error, RefColor, RefPixelType, Result, color_code, item_size, opt_cstr, pixel_type_code,
};

/// Pixels owned by the caller, described for the library to read (or, with
/// `data = NULL`, just a shape to allocate or compile for).
///
/// The library never keeps `data` beyond the call it is passed to. The pointer needs
/// no particular alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RefFrame {
    /// First byte of the top row. May be `NULL` when only the shape is needed.
    pub data: *const c_void,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Bytes from the start of one row to the next; `0` means tightly packed
    /// (`width * channels * sample size`). Padded rows are copied into a packed staging
    /// buffer, which the runner reuses between calls.
    pub row_stride: usize,
    /// A `REF_PIXEL_*` constant.
    pub pixel_type: RefPixelType,
    /// A `REF_COLOR_*` constant.
    pub color: RefColor,
    /// Channels per pixel; read only for `REF_COLOR_CUSTOM`.
    pub channels: u8,
    /// Name of a `REF_COLOR_CUSTOM` color space; otherwise ignored. May be `NULL`
    /// (meaning `"custom"`).
    pub color_name: *const c_char,
}

/// The shape and type of an image or a runner's input/output.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RefImageInfo {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Interleaved channels per pixel.
    pub channels: u8,
    /// Meaningful bits per sample (10/12/14 for the sub-16-bit types, 32 for float).
    pub bit_depth: u8,
    /// A `REF_PIXEL_*` constant.
    pub pixel_type: RefPixelType,
    /// A `REF_COLOR_*` constant.
    pub color: RefColor,
    /// Bytes per packed row: `width * channels * sample size`.
    pub row_bytes: usize,
    /// Bytes of pixel data: `row_bytes * height`.
    pub nbytes: usize,
}

impl RefImageInfo {
    pub(crate) fn of(width: usize, height: usize, cs: &ColorSpace, pt: PixelType) -> Self {
        let channels = cs.channels();
        let row_bytes = width * channels as usize * item_size(pt);
        Self {
            width,
            height,
            channels,
            bit_depth: pt.bit_depth(),
            pixel_type: pixel_type_code(pt),
            color: color_code(cs),
            row_bytes,
            nbytes: row_bytes * height,
        }
    }

    pub(crate) fn of_spec(s: &ImageSpec) -> Self {
        Self::of(s.width, s.height, &s.cspace, s.pixel_type)
    }
}

impl RefFrame {
    /// The frame's shape and type.
    pub(crate) unsafe fn spec(&self) -> Result<ImageSpec> {
        let pt = crate::pixel_type(self.pixel_type)?;
        let name = unsafe { opt_cstr(self.color_name, "frame.color_name") }?;
        let cs = crate::color_space(self.color, self.channels, name)?;
        if self.width == 0 || self.height == 0 {
            return Err(Error::new(
                REF_ERR_INVALID_ARG,
                "frame width and height must be non-zero",
            ));
        }
        Ok(ImageSpec::new(self.width, self.height, cs, pt))
    }

    /// Packed row length in bytes, and total packed length.
    pub(crate) fn packed_len(spec: &ImageSpec) -> Result<(usize, usize)> {
        let row = spec
            .width
            .checked_mul(spec.cspace.channels() as usize)
            .and_then(|n| n.checked_mul(item_size(spec.pixel_type)))
            .ok_or_else(|| Error::invalid("frame row size overflows"))?;
        let total = row
            .checked_mul(spec.height)
            .ok_or_else(|| Error::invalid("frame size overflows"))?;
        Ok((row, total))
    }

    /// The pixels, packed: borrowed straight from `data` when the rows are already
    /// contiguous, otherwise copied into `staging` (reused; it only grows).
    pub(crate) unsafe fn packed<'a>(
        &'a self,
        spec: &ImageSpec,
        staging: &'a mut Vec<u8>,
    ) -> Result<&'a [u8]> {
        let (row, total) = Self::packed_len(spec)?;
        if self.data.is_null() {
            return Err(Error::new(REF_ERR_NULL, "frame.data is NULL"));
        }
        let src = self.data.cast::<u8>();
        if self.row_stride == 0 || self.row_stride == row {
            // SAFETY: the caller guarantees `height` packed rows are readable.
            return Ok(unsafe { std::slice::from_raw_parts(src, total) });
        }
        if self.row_stride < row {
            return Err(Error::invalid(format!(
                "frame.row_stride {} is shorter than a row ({row} bytes)",
                self.row_stride
            )));
        }
        staging.resize(total, 0);
        for (y, dst) in staging.chunks_exact_mut(row).enumerate() {
            // SAFETY: row `y` starts `y * row_stride` bytes in and is `row` bytes long,
            // all within the buffer the caller describes.
            let line = unsafe { std::slice::from_raw_parts(src.add(y * self.row_stride), row) };
            dst.copy_from_slice(line);
        }
        Ok(&staging[..])
    }
}

/// Packed caller bytes presented to the pipeline as a frame (no copy).
pub(crate) struct CFrame<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) spec: &'a ImageSpec,
}

impl ImageProps for CFrame<'_> {
    fn width(&self) -> usize {
        self.spec.width
    }
    fn height(&self) -> usize {
        self.spec.height
    }
    fn channels(&self) -> u8 {
        self.spec.cspace.channels()
    }
    fn color_space(&self) -> ColorSpace {
        self.spec.cspace.clone()
    }
    fn pixel_type(&self) -> PixelType {
        self.spec.pixel_type
    }
    fn len(&self) -> usize {
        self.spec.elems()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Frame for CFrame<'_> {
    fn as_bytes(&self) -> &[u8] {
        self.bytes
    }
}
