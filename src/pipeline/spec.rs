//! [`ImageSpec`] — the static shape/type description a [`Pipeline`](super::Pipeline)
//! is compiled against.

use crate::{ColorSpace, ImageProps, PixelType};

use super::PipelineError;

/// Static description of an image's shape and element type.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpec {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Color space.
    pub cspace: ColorSpace,
    /// Primitive element type.
    pub pixel_type: PixelType,
}

impl ImageSpec {
    /// Build a spec. The channel count is taken from `cspace`.
    pub fn new(width: usize, height: usize, cspace: ColorSpace, pixel_type: PixelType) -> Self {
        Self {
            width,
            height,
            cspace,
            pixel_type,
        }
    }

    /// Snapshot the shape of a live [`DynamicImageRef`](crate::DynamicImageRef).
    pub fn from_dynamic<I: ImageProps + ?Sized>(img: &I) -> Self {
        Self {
            width: img.width(),
            height: img.height(),
            cspace: img.color_space(),
            pixel_type: img.pixel_type(),
        }
    }

    /// Element count (`width * height * channels`).
    pub fn elems(&self) -> usize {
        self.width * self.height * self.cspace.channels() as usize
    }

    /// Byte count of a tightly-packed buffer for this spec.
    pub fn bytes(&self) -> Result<usize, PipelineError> {
        Ok(self.elems() * pixel_size(self.pixel_type)?)
    }

    /// Bytes per pixel (`channels * pixel_size`).
    pub(super) fn bpp(&self) -> Result<usize, PipelineError> {
        Ok(self.cspace.channels() as usize * pixel_size(self.pixel_type)?)
    }

    /// Bytes for a `rows * cols` tile in this spec's channels / element type.
    pub(super) fn tile_bytes(&self, rows: usize, cols: usize) -> Result<usize, PipelineError> {
        Ok(rows * cols * self.bpp()?)
    }

    pub(super) fn validate(&self) -> Result<(), PipelineError> {
        if self.width == 0 || self.height == 0 || self.width > 65535 || self.height > 65535 {
            return Err(PipelineError::BadDimensions);
        }
        pixel_size(self.pixel_type)?;
        Ok(())
    }
}

pub(crate) fn pixel_size(pt: PixelType) -> Result<usize, PipelineError> {
    match pt {
        PixelType::U8 => Ok(1),
        PixelType::U16 => Ok(2),
        PixelType::F32 => Ok(4),
        other => Err(PipelineError::UnsupportedPixelType(other)),
    }
}
