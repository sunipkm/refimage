//! Demosaicing using linear interpolation.
//!
//! ```text
//!   green_kernel = (1 / 4) *
//!       [ 0 1 0
//!       ; 1 4 1
//!       ; 0 1 0 ];
//!
//!   red/blue_kernel = (1 / 4) *
//!       [ 1 2 1
//!       ; 2 4 2
//!       ; 1 2 1 ];
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::demosaic::border_replicate::*;
use crate::demosaic::{BayerError, BayerRead, BayerResult, ColorFilterArray, RasterMut};
use crate::traits::{get_mean, Enlargeable};
use crate::{ImageData, Primitive};

const PADDING: usize = 1;

pub fn run<T>(
    src: &ImageData<'_, T>,
    cfa: ColorFilterArray,
    dst: &mut RasterMut<'_, T>,
) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
{
    if src.width < 2 || src.height < 2 {
        return Err(BayerError::WrongResolution);
    }

    debayer(src.as_slice(), cfa, dst)
}

macro_rules! apply_kernel_c {
    ($T:ty; $row:ident, $prev:expr, $curr:expr, $next:expr, $cfa:expr, $i:expr) => {{
        // current = B/R, diagonal = R/B.
        let (c, d) = if $cfa == ColorFilterArray::Bggr {
            (2, 0)
        } else {
            (0, 2)
        };
        let j = $i + PADDING;

        $row[3 * $i + c] = $curr[j];
        $row[3 * $i + 1] = get_mean(&[$prev[j], $curr[j - 1], $curr[j + 1], $next[j]]);
        $row[3 * $i + d] = get_mean(&[$prev[j - 1], $prev[j + 1], $next[j - 1], $next[j + 1]]);
    }};
}

macro_rules! apply_kernel_g {
    ($T:ty; $row:ident, $prev:expr, $curr:expr, $next:expr, $cfa:expr, $i:expr) => {{
        // horizontal = B/R, vertical = R/G.
        let (h, v) = if $cfa == ColorFilterArray::Gbrg {
            (2, 0)
        } else {
            (0, 2)
        };
        let j = $i + PADDING;

        $row[3 * $i + h] = get_mean(&[$curr[j - 1], $curr[j + 1]]);
        $row[3 * $i + 1] = $curr[j];
        $row[3 * $i + v] = get_mean(&[$prev[j], $next[j]]);
    }};
}

macro_rules! apply_kernel_row {
    ($T:ty; $row:ident, $prev:expr, $curr:expr, $next:expr, $cfa:expr, $w:expr) => {{
        let (mut i, cfa_c, cfa_g) =
            if $cfa == ColorFilterArray::Bggr || $cfa == ColorFilterArray::Rggb {
                (0, $cfa, $cfa.next_x())
            } else {
                apply_kernel_g!($T; $row, $prev, $curr, $next, $cfa, 0);
                (1, $cfa.next_x(), $cfa)
            };

        while i + 1 < $w {
            apply_kernel_c!($T; $row, $prev, $curr, $next, cfa_c, i);
            apply_kernel_g!($T; $row, $prev, $curr, $next, cfa_g, i + 1);
            i += 2;
        }

        if i < $w {
            apply_kernel_c!($T; $row, $prev, $curr, $next, cfa_c, i);
        }
    }}
}

/*--------------------------------------------------------------*/
/* Rayon                                                        */
/*--------------------------------------------------------------*/

#[cfg(feature = "rayon")]
fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
{
    let (w, h) = (dst.w, dst.h);
    let mut data = vec![T::zero(); (2 * PADDING + w) * (2 * PADDING + h)];

    // Read all data.
    {
        let stride = 2 * PADDING + w;
        let rdr = BorderReplicate::new(w, PADDING);

        for row in data.chunks_mut(stride).skip(PADDING).take(h) {
            rdr.read_row(r, row)?;
        }

        {
            let (top, src) = data.split_at_mut(stride * PADDING);
            top[0..stride].copy_from_slice(&src[stride..(stride * 2)]);
        }

        {
            let (src, bottom) = data.split_at_mut(stride * (h + PADDING));
            let yy = PADDING + h;
            bottom[0..stride].copy_from_slice(&src[(stride * (yy - 2))..(stride * (yy - 1))]);
        }
    }

    dst.buf
        .par_chunks_mut(dst.stride)
        .enumerate()
        .for_each(|(y, row)| {
            let stride = 2 * PADDING + w;
            let prev = &data[(stride * (PADDING + y - 1))..(stride * (PADDING + y))];
            let curr = &data[(stride * (PADDING + y))..(stride * (PADDING + y + 1))];
            let next = &data[(stride * (PADDING + y + 1))..(stride * (PADDING + y + 2))];
            let cfa_y = if y % 2 == 0 { cfa } else { cfa.next_y() };

            apply_kernel_row!(T; row, prev, curr, next, cfa_y, w);
        });

    Ok(())
}

/*--------------------------------------------------------------*/
/* Naive                                                        */
/*--------------------------------------------------------------*/

#[cfg(not(feature = "rayon"))]
fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
{
    let (w, h) = (dst.w, dst.h);
    let mut prev = vec![T::zero(); 2 * PADDING + w];
    let mut curr = vec![T::zero(); 2 * PADDING + w];
    let mut next = vec![T::zero(); 2 * PADDING + w];
    let mut cfa = cfa;

    let rdr = BorderReplicate::new(w, PADDING);
    rdr.read_row(r, &mut curr)?;
    rdr.read_row(r, &mut next)?;

    {
        // y = 0.
        let row = dst.borrow_row_mut(0);
        apply_kernel_row!(T; row, next, curr, next, cfa, w);
        cfa = cfa.next_y();
    }

    for y in 1..(h - 1) {
        rotate!(prev <- curr <- next);
        rdr.read_row(r, &mut next)?;

        let row = dst.borrow_row_mut(y);
        apply_kernel_row!(T; row, prev, curr, next, cfa, w);
        cfa = cfa.next_y();
    }

    {
        // y = h - 1.
        let row = dst.borrow_row_mut(h - 1);
        apply_kernel_row!(T; row, curr, next, curr, cfa, w);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::debayer;
    use crate::demosaic::{ColorFilterArray, RasterMut};

    #[test]
    fn test_even() {
        // R: set.seed(0); matrix(floor(runif(n=16, min=0, max=256)), nrow=4, byrow=TRUE)
        let src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
        ];

        let expected = [
            229, 149, 51, 162, 67, 51, 95, 167, 146, 95, 146, 241, 199, 232, 51, 127, 172, 51, 55,
            229, 146, 55, 164, 241, 169, 149, 113, 92, 161, 113, 15, 135, 166, 15, 52, 219, 169,
            45, 175, 92, 116, 175, 15, 98, 186, 15, 75, 197,
        ];

        const IMG_W: usize = 4;
        const IMG_H: usize = 4;
        let mut dst = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(
            &src,
            ColorFilterArray::Rggb,
            &mut RasterMut::new(IMG_W, IMG_H, &mut dst),
        );
        assert!(res.is_ok());
        assert_eq!(&dst[..], &expected[..]);
    }

    #[test]
    fn test_odd() {
        // R: set.seed(0); matrix(floor(runif(n=9, min=0, max=256)), nrow=3, byrow=TRUE)
        let src = [229, 67, 95, 146, 232, 51, 229, 241, 169];

        let expected = [
            229, 106, 232, 162, 67, 232, 95, 59, 232, 229, 146, 232, 180, 126, 232, 132, 51, 232,
            229, 193, 232, 199, 241, 232, 169, 146, 232,
        ];

        const IMG_W: usize = 3;
        const IMG_H: usize = 3;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(
            &src,
            ColorFilterArray::Rggb,
            &mut RasterMut::new(IMG_W, IMG_H, &mut buf),
        );
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }
}
