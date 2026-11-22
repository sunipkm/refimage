//! How coordinates and indices outside an image are treated (ported from `imwarp`).

use serde::{Deserialize, Serialize};

use crate::pipeline::PipelineError;

/// Behaviour beyond the edge of the image for [`Op::Warp`](crate::pipeline::Op::Warp) and
/// [`Op::MapCoordinates`](crate::pipeline::Op::MapCoordinates).
///
/// The names follow `scipy.ndimage`; [`FromStr`](std::str::FromStr) also accepts
/// scikit-image's `edge` (for [`Nearest`](Self::Nearest)) and `symmetric` (for
/// [`Reflect`](Self::Reflect)). Note that scikit-image's own `reflect` is scipy's
/// `mirror`; the plain names here always mean scipy's modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BoundaryMode {
    /// Everything outside the image is the fill value; no interpolation across the edge.
    #[default]
    Constant,
    /// Repeat the edge pixel.
    Nearest,
    /// Reflect about the edge pixel's centre: `d c b | a b c d | c b a`.
    Mirror,
    /// Reflect about the edge: `d c b a | a b c d | d c b a`.
    Reflect,
    /// Wrap around with the last pixel coinciding with the first.
    Wrap,
}

impl std::str::FromStr for BoundaryMode {
    type Err = PipelineError;

    fn from_str(s: &str) -> Result<Self, PipelineError> {
        match s {
            "constant" => Ok(Self::Constant),
            "nearest" | "edge" => Ok(Self::Nearest),
            "mirror" => Ok(Self::Mirror),
            "reflect" | "symmetric" => Ok(Self::Reflect),
            "wrap" => Ok(Self::Wrap),
            other => Err(PipelineError::UnknownBoundaryMode(other.to_owned())),
        }
    }
}

/// Which boundary extension the spline prefilter assumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FilterExtension {
    Mirror,
    Reflect,
}

impl BoundaryMode {
    /// Mode used to map the taps of the interpolation footprint that fall off the image.
    ///
    /// Constant and wrap fall back to mirror extension, as in SciPy.
    pub(super) const fn spline_mode(self) -> Self {
        match self {
            Self::Constant | Self::Wrap | Self::Mirror => Self::Mirror,
            Self::Nearest => Self::Nearest,
            Self::Reflect => Self::Reflect,
        }
    }

    /// Boundary condition of the cubic prefilter.
    pub(super) const fn filter_extension(self) -> FilterExtension {
        match self {
            Self::Constant | Self::Mirror | Self::Wrap => FilterExtension::Mirror,
            Self::Nearest | Self::Reflect => FilterExtension::Reflect,
        }
    }

    /// Map a (possibly fractional) index into `0..=len-1` according to this mode.
    ///
    /// For [`Self::Constant`] a coordinate outside the image maps to `-1`.
    pub(super) fn map_coordinate(self, mut x: f64, len: usize) -> f64 {
        let len_f = len as f64;
        if x < 0.0 {
            match self {
                Self::Mirror if len <= 1 => x = 0.0,
                Self::Mirror => {
                    let sz2 = (2 * len - 2) as f64;
                    x += sz2 * (-x / sz2).trunc();
                    x = if x <= 1.0 - len_f { x + sz2 } else { -x };
                }
                Self::Reflect if len <= 1 => x = 0.0,
                Self::Reflect => {
                    let sz2 = 2.0 * len_f;
                    if x < -sz2 {
                        x += sz2 * (-x / sz2).trunc();
                    }
                    // The 1e-15 guard avoids (-x - 1) == -1 for tiny negative x.
                    x = if x < -len_f {
                        x + sz2
                    } else {
                        (if x > -1e-15 { 1e-15 } else { -x }) - 1.0
                    };
                }
                Self::Wrap if len <= 1 => x = 0.0,
                Self::Wrap => {
                    let sz = (len - 1) as f64;
                    x += sz * ((-x / sz).trunc() + 1.0);
                }
                Self::Nearest => x = 0.0,
                Self::Constant => x = -1.0,
            }
        } else if x > len_f - 1.0 {
            match self {
                Self::Mirror if len <= 1 => x = 0.0,
                Self::Mirror => {
                    let sz2 = (2 * len - 2) as f64;
                    x -= sz2 * (x / sz2).trunc();
                    if x >= len_f {
                        x = sz2 - x;
                    }
                }
                Self::Reflect if len <= 1 => x = 0.0,
                Self::Reflect => {
                    let sz2 = 2.0 * len_f;
                    x -= sz2 * (x / sz2).trunc();
                    if x >= len_f {
                        x = sz2 - x - 1.0;
                    }
                }
                Self::Wrap if len <= 1 => x = 0.0,
                Self::Wrap => {
                    let sz = (len - 1) as f64;
                    x -= sz * (x / sz).trunc();
                }
                Self::Nearest => x = len_f - 1.0,
                Self::Constant => x = -1.0,
            }
        }
        x
    }
}
