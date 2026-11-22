//! The public, builder-style compression API.
//!
//! Each algorithm is its own struct — [`Gzip`], [`Rice`], [`Hcompress`] — with chainable
//! setters. They all convert into the opaque [`FitsCompression`] that the write methods
//! take (`impl Into<FitsCompression>`), so a bare builder can be passed directly:
//!
//! ```no_run
//! # use refimage::{FitsWrite, GenericImageRef, Rice, Gzip, Hcompress, Quantize};
//! # fn f(img: GenericImageRef<'_>) -> std::io::Result<()> {
//! # let path = std::path::Path::new("x.fits");
//! img.write_fits(path, Gzip::new(), true).ok();
//! img.write_fits(path, Gzip::new().level(9), true).ok();
//! img.write_fits(path, Rice::new().tile_rows(16), true).ok();
//! img.write_fits(
//!     path,
//!     Hcompress::new().scale(4).smooth(true).quantize(Quantize::new().level(16.0)),
//!     true,
//! ).ok();
//! # Ok(()) }
//! ```
//!
//! The tile selectors [`tile_rows`](Rice::tile_rows) and [`tile_dims`](Rice::tile_dims)
//! are mutually exclusive *at compile time*: calling either moves the builder from
//! `…<`[`AutoTile`]`>` to `…<`[`FixedTile`]`>`, and the second selector does not exist on
//! the latter.

use std::marker::PhantomData;

/// How a compressed image is cut into independently-compressed tiles (resolved against
/// the concrete image size at write time).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tiling {
    /// Algorithm default: one image row per tile for `GZIP_1` / `RICE_1`; one whole
    /// channel plane for `HCOMPRESS_1`.
    Default,
    /// `n` full image rows per tile (`ZTILE = [w, n, 1, …]`).
    Rows(usize),
    /// Explicit tile side lengths, fastest FITS axis first (`[nx, ny]`). Dimensions
    /// past the second must be `1` — the channel axis is never tiled.
    Dims(Vec<usize>),
}

/// Type-state marker: the tile shape has not been chosen yet, so both
/// [`tile_rows`](Rice::tile_rows) and [`tile_dims`](Rice::tile_dims) are available.
#[derive(Debug, Clone, Copy, Default)]
pub struct AutoTile;

/// Type-state marker: the tile shape is fixed, so neither tile selector is in scope any
/// more (you cannot combine `tile_rows` and `tile_dims`).
#[derive(Debug, Clone, Copy, Default)]
pub struct FixedTile;

/// Lossy quantization of `f32` images, shared by [`Rice`] and [`Hcompress`]. Integer
/// images ignore it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantize {
    pub(crate) level: f64,
    pub(crate) seed: DitherSeed,
}

impl Default for Quantize {
    fn default() -> Self {
        // cfitsio's default `QUANTIZE_LEVEL`.
        Self {
            level: 4.0,
            seed: DitherSeed::Auto,
        }
    }
}

impl Quantize {
    /// A default quantization (`level = 4.0`, data-derived dither seed).
    pub fn new() -> Self {
        Self::default()
    }

    /// The quantization level `q` (cfitsio's `QUANTIZE_LEVEL`, historically
    /// "noise bits"). The step is `noise / q`, so a larger `q` keeps more precision and
    /// compresses less. cfitsio's default is `4.0`.
    pub fn level(mut self, q: f64) -> Self {
        self.level = q;
        self
    }

    /// How the subtractive-dither offset (`ZDITHER0`) is chosen.
    pub fn seed(mut self, seed: DitherSeed) -> Self {
        self.seed = seed;
        self
    }
}

/// Choice of subtractive-dither seed (`ZDITHER0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DitherSeed {
    /// Derive it deterministically from the pixel data (cfitsio's "negative seed"
    /// method) — the same image always yields the same file.
    Auto,
    /// A fixed value in `1..=10000`, for byte-reproducible output across images.
    Fixed(u32),
}

macro_rules! tile_selectors {
    ($name:ident $(, $extra:ident : $ety:ty)*) => {
        impl $name<AutoTile> {
            /// Use tiles of `rows` full image rows each (`ZTILE = [w, rows, 1, …]`).
            ///
            /// Mutually exclusive with [`tile_dims`](Self::tile_dims): once either is
            /// called the other is no longer available.
            pub fn tile_rows(self, rows: usize) -> $name<FixedTile> {
                $name {
                    tiling: Tiling::Rows(rows),
                    $($extra: self.$extra,)*
                    _tile: PhantomData,
                }
            }

            /// Use rectangular tiles of the given side lengths, fastest FITS axis first
            /// (`[nx, ny]`). Values are clamped to the image size; dimensions past the
            /// second must be `1`.
            ///
            /// Mutually exclusive with [`tile_rows`](Self::tile_rows).
            pub fn tile_dims<const N: usize>(self, dims: [usize; N]) -> $name<FixedTile> {
                $name {
                    tiling: Tiling::Dims(dims.to_vec()),
                    $($extra: self.$extra,)*
                    _tile: PhantomData,
                }
            }
        }
    };
}

/// cfitsio (and zlib) compress at effort 6 by default.
const GZIP_DEFAULT_LEVEL: u8 = 6;

/// `GZIP_1` tile compression — lossless for every pixel type.
#[derive(Debug, Clone)]
pub struct Gzip<Tile = AutoTile> {
    tiling: Tiling,
    level: u8,
    _tile: PhantomData<Tile>,
}

impl Default for Gzip<AutoTile> {
    fn default() -> Self {
        Self {
            tiling: Tiling::Default,
            level: GZIP_DEFAULT_LEVEL,
            _tile: PhantomData,
        }
    }
}

impl Gzip<AutoTile> {
    /// `GZIP_1` with the default tiling (one image row per tile) and DEFLATE
    /// effort 6.
    pub fn new() -> Self {
        Self::default()
    }
}

tile_selectors!(Gzip, level: u8);

impl<Tile> Gzip<Tile> {
    /// DEFLATE effort, `0` (store, no compression) to `9` (best); `6` is the
    /// default, matching zlib and cfitsio. Higher values shrink the file at the
    /// cost of CPU and never change what a reader gets back. Values above `9` are
    /// clamped.
    pub fn level(mut self, level: u8) -> Self {
        self.level = level.min(9);
        self
    }
}

impl<Tile> From<Gzip<Tile>> for FitsCompression {
    fn from(g: Gzip<Tile>) -> Self {
        FitsCompression(Method::Gzip {
            tiling: g.tiling,
            level: g.level,
        })
    }
}

/// `RICE_1` tile compression — lossless for `u8` / `u16`; `f32` is quantized first
/// (see [`quantize`](Self::quantize)).
#[derive(Debug, Clone)]
pub struct Rice<Tile = AutoTile> {
    tiling: Tiling,
    quantize: Quantize,
    _tile: PhantomData<Tile>,
}

impl Default for Rice<AutoTile> {
    fn default() -> Self {
        Self {
            tiling: Tiling::Default,
            quantize: Quantize::default(),
            _tile: PhantomData,
        }
    }
}

impl Rice<AutoTile> {
    /// `RICE_1` with the default tiling (one image row per tile) and default
    /// quantization for `f32`.
    pub fn new() -> Self {
        Self::default()
    }
}

tile_selectors!(Rice, quantize: Quantize);

impl<Tile> Rice<Tile> {
    /// Quantization applied to `f32` images (ignored for `u8` / `u16`).
    pub fn quantize(mut self, quantize: Quantize) -> Self {
        self.quantize = quantize;
        self
    }
}

impl<Tile> From<Rice<Tile>> for FitsCompression {
    fn from(r: Rice<Tile>) -> Self {
        FitsCompression(Method::Rice {
            tiling: r.tiling,
            quantize: r.quantize,
        })
    }
}

/// `HCOMPRESS_1` tile compression (H-transform + quadtree coder). Lossless at
/// `scale = 0`; `f32` is quantized first. Each tile must be at least 4×4; the default
/// tiling is one whole channel plane per tile.
#[derive(Debug, Clone)]
pub struct Hcompress<Tile = AutoTile> {
    tiling: Tiling,
    scale: i32,
    smooth: bool,
    quantize: Quantize,
    _tile: PhantomData<Tile>,
}

impl Default for Hcompress<AutoTile> {
    fn default() -> Self {
        Self {
            tiling: Tiling::Default,
            scale: 0,
            smooth: false,
            quantize: Quantize::default(),
            _tile: PhantomData,
        }
    }
}

impl Hcompress<AutoTile> {
    /// `HCOMPRESS_1`, lossless (`scale = 0`), one plane per tile.
    pub fn new() -> Self {
        Self::default()
    }
}

tile_selectors!(Hcompress, scale: i32, smooth: bool, quantize: Quantize);

impl<Tile> Hcompress<Tile> {
    /// The H-transform scale divisor. `0` (the default) is lossless; larger values
    /// discard low-order bits for a smaller file.
    pub fn scale(mut self, scale: i32) -> Self {
        self.scale = scale;
        self
    }

    /// Store the `SMOOTH` flag so a reader applies image smoothing when decompressing a
    /// scaled image (no effect at `scale = 0`).
    pub fn smooth(mut self, smooth: bool) -> Self {
        self.smooth = smooth;
        self
    }

    /// Quantization applied to `f32` images (ignored for `u8` / `u16`).
    pub fn quantize(mut self, quantize: Quantize) -> Self {
        self.quantize = quantize;
        self
    }
}

impl<Tile> From<Hcompress<Tile>> for FitsCompression {
    fn from(hc: Hcompress<Tile>) -> Self {
        FitsCompression(Method::Hcompress {
            tiling: hc.tiling,
            scale: hc.scale,
            smooth: hc.smooth,
            quantize: hc.quantize,
        })
    }
}

/// The resolved, private compression selector.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Method {
    None,
    Gzip {
        tiling: Tiling,
        /// DEFLATE effort, 0..=9.
        level: u8,
    },
    Rice {
        tiling: Tiling,
        quantize: Quantize,
    },
    Hcompress {
        tiling: Tiling,
        scale: i32,
        smooth: bool,
        quantize: Quantize,
    },
}

/// An opaque, resolved compression choice, built from [`Gzip`] / [`Rice`] /
/// [`Hcompress`] (or [`FitsCompression::NONE`]) and handed to the write methods.
#[derive(Clone, PartialEq)]
pub struct FitsCompression(pub(crate) Method);

impl FitsCompression {
    /// Write a plain, uncompressed image HDU.
    pub const NONE: FitsCompression = FitsCompression(Method::None);

    /// Which algorithm this selects.
    pub fn kind(&self) -> FitsCompressionKind {
        match self.0 {
            Method::None => FitsCompressionKind::None,
            Method::Gzip { .. } => FitsCompressionKind::Gzip,
            Method::Rice { .. } => FitsCompressionKind::Rice,
            Method::Hcompress { .. } => FitsCompressionKind::Hcompress,
        }
    }
}

impl Default for FitsCompression {
    fn default() -> Self {
        Self::NONE
    }
}

impl std::fmt::Debug for FitsCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

impl From<&FitsCompression> for FitsCompression {
    fn from(c: &FitsCompression) -> Self {
        c.clone()
    }
}

/// The algorithm behind a [`FitsCompression`], without its settings — handy for
/// branching on what a value selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FitsCompressionKind {
    /// Uncompressed.
    None,
    /// `GZIP_1`.
    Gzip,
    /// `RICE_1`.
    Rice,
    /// `HCOMPRESS_1`.
    Hcompress,
}
