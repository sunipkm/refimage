//! [`Strategy`] — how a [`Runner`](super::Runner) sweeps the frame.

use serde::{Deserialize, Serialize};

/// How a [`Runner`](super::Runner) executes its steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Strategy {
    /// One ping-pong buffer pair; every stage runs over the whole frame in order.
    Sequential,
    /// Tiled execution; see the [module docs](super#execution-strategies).
    Tiled {
        /// Output rows per band before the halo. `0` picks an automatic height.
        tile_rows: usize,
        /// Output columns per tile before the halo. `0` (or `>= width`) keeps
        /// bands full-width — cheaper, and all that wide-enough frames need.
        tile_cols: usize,
        /// Fan bands out across a rayon pool (needs the `rayon` feature; a plain
        /// serial sweep otherwise).
        parallel: bool,
    },
}

impl Strategy {
    /// Serial row-band tiling. `tile_rows == 0` auto-sizes.
    pub fn tiled(tile_rows: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols: 0,
            parallel: false,
        }
    }

    /// Row-band tiling fanned out over a rayon pool. `tile_rows == 0` auto-sizes.
    pub fn tiled_parallel(tile_rows: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols: 0,
            parallel: true,
        }
    }

    /// 2-D tiling: `tile_rows` bands, each split into `tile_cols`-wide tiles.
    pub fn tiled_2d(tile_rows: usize, tile_cols: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols,
            parallel: false,
        }
    }

    /// 2-D tiling with the bands fanned out over a rayon pool.
    pub fn tiled_2d_parallel(tile_rows: usize, tile_cols: usize) -> Self {
        Strategy::Tiled {
            tile_rows,
            tile_cols,
            parallel: true,
        }
    }
}
