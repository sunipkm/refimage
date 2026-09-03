//! [`Op`] — one declarative stage of a [`Pipeline`](super::Pipeline).
//! This also drives the compile-time buffer allocation inference.

use serde::{Deserialize, Serialize};

use crate::{BayerPattern, BayerShift, ColorSpace, DemosaicMethod, PixelType};

use super::resample::resize_dims;
use super::spec::pixel_size;
use super::{ImageSpec, PipelineError, ResizeFilter};

/// A single processing stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Op {
    /// Demosaic a single-channel Bayer image to 3-channel RGB.
    Debayer(DemosaicMethod),
    /// Rec.601 luma (`0.299, 0.587, 0.114`); RGB (or `Custom`) becomes `Gray`.
    ToLuma,
    /// Luma with custom per-channel weights; length must equal the channel count.
    ToLumaCustom(Vec<f64>),
    /// Affine per-pixel remap `y = x * gain + offset`, evaluated on the raw
    /// stored value and saturated back into the current type (for `f32`, into
    /// `[0.0, 1.0]`). Shape, channels, and type are unchanged.
    Scale {
        /// Multiplicative factor.
        gain: f64,
        /// Additive term, in raw stored units.
        offset: f64,
    },
    /// Rescale every pixel into a different primitive type.
    Convert(PixelType),
    /// Extract the sub-rectangle with top-left `(x, y)` and size `width * height`.
    /// The rectangle must lie fully inside the image (else [`PipelineError::CropOutOfBounds`]).
    /// On a Bayer image the pattern is re-phased for an odd origin, so a crop may
    /// precede a debayer.
    Crop {
        /// Left edge, in pixels from the current origin.
        x: usize,
        /// Top edge, in pixels from the current origin.
        y: usize,
        /// Output width.
        width: usize,
        /// Output height.
        height: usize,
    },
    /// Region of interest: like [`Op::Crop`], but a `width`/`height` overhang past
    /// the image edge is legal — the missing pixels come out zero. Errors only if
    /// the origin `(x, y)` itself is outside the image
    /// ([`PipelineError::RoiOutOfBounds`]). Bayer patterns are re-phased.
    Roi {
        /// Left edge, in pixels from the current origin.
        x: usize,
        /// Top edge, in pixels from the current origin.
        y: usize,
        /// Output width.
        width: usize,
        /// Output height.
        height: usize,
    },
    /// Mirror left-to-right. Bayer patterns are re-phased.
    FlipHorizontal,
    /// Mirror top-to-bottom. Bayer patterns are re-phased.
    FlipVertical,
    /// Rotate 90° clockwise; width and height swap. Not applicable on a Bayer image.
    Rotate90,
    /// Rotate 180°. Bayer patterns are re-phased.
    Rotate180,
    /// Rotate 90° counter-clockwise; width and height swap. Not applicable on Bayer.
    Rotate270,
    /// Resample to the largest size that fits within `max_width` x `max_height`
    /// at the original aspect ratio, enlarging the image if it is smaller than
    /// the box. Each side of the result is at least 1 px and never exceeds its
    /// bound. Not valid on a Bayer image ([`PipelineError::ResizeOnBayer`]) —
    /// debayer first. Runs as one whole-frame pass between tiled segments,
    /// fused and cache-blocked into bands of output rows (each allocates a small
    /// strip, not a full intermediate plane) that fan out over the `rayon` pool,
    /// independent of band size and thread count.
    ResizeToFit {
        /// Width bound in pixels; the result is never wider than this.
        max_width: usize,
        /// Height bound in pixels; the result is never taller than this.
        max_height: usize,
        /// Resampling filter.
        filter: ResizeFilter,
    },
    /// Not an operation
    Nop,
}

/// Re-phase a Bayer pattern through a geometric transform; leave other color
/// spaces untouched.
fn rephase(cspace: &ColorSpace, f: impl Fn(BayerPattern) -> BayerPattern) -> ColorSpace {
    match cspace {
        ColorSpace::Bayer(p) => ColorSpace::Bayer(f(*p)),
        other => other.clone(),
    }
}

impl Op {
    pub(super) fn output_spec(&self, input: &ImageSpec) -> Result<ImageSpec, PipelineError> {
        match self {
            Op::Debayer(_) => {
                if !matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::NotBayer);
                }
                if input.cspace.channels() != 1 {
                    return Err(PipelineError::DebayerChannels(input.cspace.channels()));
                }
                Ok(ImageSpec {
                    cspace: ColorSpace::Rgb,
                    ..input.clone()
                })
            }
            Op::ToLuma => luma_output(input, 3),
            Op::ToLumaCustom(w) => luma_output(input, w.len()),
            Op::Scale { .. } => {
                pixel_size(input.pixel_type)?;
                Ok(input.clone())
            }
            Op::Convert(pt) => {
                pixel_size(*pt)?;
                Ok(ImageSpec {
                    pixel_type: *pt,
                    ..input.clone()
                })
            }
            Op::Crop {
                x,
                y,
                width,
                height,
            } => {
                if *width == 0 || *height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if x + width > input.width || y + height > input.height {
                    return Err(PipelineError::CropOutOfBounds {
                        rect: (*x, *y, *width, *height),
                        image: (input.width, input.height),
                    });
                }
                Ok(ImageSpec {
                    width: *width,
                    height: *height,
                    cspace: rephase(&input.cspace, |p| p.shift(*x, *y)),
                    ..input.clone()
                })
            }
            Op::Roi {
                x,
                y,
                width,
                height,
            } => {
                if *width == 0 || *height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if *x >= input.width || *y >= input.height {
                    return Err(PipelineError::RoiOutOfBounds {
                        origin: (*x, *y),
                        image: (input.width, input.height),
                    });
                }
                Ok(ImageSpec {
                    width: *width,
                    height: *height,
                    cspace: rephase(&input.cspace, |p| p.shift(*x, *y)),
                    ..input.clone()
                })
            }
            Op::FlipHorizontal => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_horizontal()),
                ..input.clone()
            }),
            Op::FlipVertical => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_vertical()),
                ..input.clone()
            }),
            Op::Rotate180 => Ok(ImageSpec {
                cspace: rephase(&input.cspace, |p| p.flip_horizontal().flip_vertical()),
                ..input.clone()
            }),
            Op::Rotate90 | Op::Rotate270 => {
                if matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::RotateOnBayer);
                }
                Ok(ImageSpec {
                    width: input.height,
                    height: input.width,
                    ..input.clone()
                })
            }
            Op::ResizeToFit {
                max_width,
                max_height,
                filter: _,
            } => {
                if *max_width == 0 || *max_height == 0 {
                    return Err(PipelineError::BadDimensions);
                }
                if matches!(input.cspace, ColorSpace::Bayer(_)) {
                    return Err(PipelineError::ResizeOnBayer);
                }
                pixel_size(input.pixel_type)?;
                let (width, height) =
                    resize_dims(input.width, input.height, *max_width, *max_height);
                Ok(ImageSpec {
                    width,
                    height,
                    ..input.clone()
                })
            }
            Op::Nop => Ok(input.clone()),
        }
    }

    /// Rows/cols of vertical/horizontal context this op reads on each side of an
    /// output pixel; drives tile halos. Row-local and geometric ops are 0.
    pub(super) fn halo(&self) -> usize {
        match self {
            Op::Debayer(DemosaicMethod::None) => 0,
            Op::Debayer(DemosaicMethod::Nearest) => 1,
            Op::Debayer(DemosaicMethod::Linear) => 1,
            Op::Debayer(DemosaicMethod::Cubic) => 3,
            Op::ToLuma
            | Op::ToLumaCustom(_)
            | Op::Scale { .. }
            | Op::Convert(_)
            | Op::Crop { .. }
            | Op::Roi { .. }
            | Op::FlipHorizontal
            | Op::FlipVertical
            | Op::Rotate90
            | Op::Rotate180
            | Op::Rotate270
            | Op::ResizeToFit { .. }
            | Op::Nop => 0,
        }
    }

    /// A "pixel op" keeps every output pixel at its input `(x, y)` — so a run of
    /// them can be fused into one tiled pass. Geometric ops relocate pixels.
    pub(super) fn is_pixel(&self) -> bool {
        matches!(
            self,
            Op::Debayer(_) | Op::ToLuma | Op::ToLumaCustom(_) | Op::Scale { .. } | Op::Convert(_)
        )
    }

    /// A crop with no other effect — used to fold leading crops into an input
    /// offset so a following pixel run can still tile.
    pub(super) fn as_crop(&self) -> Option<(usize, usize)> {
        match self {
            Op::Crop { x, y, .. } => Some((*x, *y)),
            _ => None,
        }
    }
}

fn luma_output(input: &ImageSpec, ncoeffs: usize) -> Result<ImageSpec, PipelineError> {
    match input.cspace {
        // Matches `ImageRef::to_luma`: already-gray is a no-op passthrough.
        ColorSpace::Gray => Ok(input.clone()),
        ColorSpace::Rgb | ColorSpace::Custom(..) => {
            if input.cspace.channels() as usize != ncoeffs {
                return Err(PipelineError::LumaCoeffMismatch {
                    channels: input.cspace.channels(),
                    coeffs: ncoeffs,
                });
            }
            Ok(ImageSpec {
                cspace: ColorSpace::Gray,
                ..input.clone()
            })
        }
        ColorSpace::Bayer(_) => Err(PipelineError::LumaOnBayer),
    }
}
