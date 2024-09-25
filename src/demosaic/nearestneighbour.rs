//! Demosaicing using nearest neighbour interpolation.

use crate::demosaic::RasterMut;
use crate::demosaic::{BayerError, BayerRead, BayerResult, ColorFilterArray};
use crate::{ImageRef, ImageOwned, PixelStor};

use super::border_replicate::BorderReplicate;

const PADDING: usize = 1;

pub fn run_imagedata<T>(
    src: &ImageRef<'_, T>,
    cfa: ColorFilterArray,
    dst: &mut RasterMut<'_, T>,
) -> BayerResult<()>
where
    T: PixelStor,
{
    if src.width() < 2 || src.height() < 2 {
        return Err(BayerError::WrongResolution);
    }

    debayer(src.as_slice(), cfa, dst)
}

pub fn run_imageowned<T>(
    src: &ImageOwned<T>,
    cfa: ColorFilterArray,
    dst: &mut RasterMut<'_, T>,
) -> BayerResult<()>
where
    T: PixelStor,
{
    if src.width() < 2 || src.height() < 2 {
        return Err(BayerError::WrongResolution);
    }

    debayer(src.as_slice(), cfa, dst)
}

macro_rules! apply_kernel_row {
    ($row:ident, $prev:expr, $curr:expr, $cfa:expr, $w:expr) => {{
        let (mut i, cfa_c, cfa_g) =
            if $cfa == ColorFilterArray::Bggr || $cfa == ColorFilterArray::Rggb {
                (0, $cfa, $cfa.next_x())
            } else {
                apply_kernel_g!($row, $prev, $curr, $cfa, 0);
                (1, $cfa.next_x(), $cfa)
            };

        while i + 1 < $w {
            apply_kernel_c!($row, $prev, $curr, cfa_c, i);
            apply_kernel_g!($row, $prev, $curr, cfa_g, i + 1);
            i += 2;
        }

        if i < $w {
            apply_kernel_c!($row, $prev, $curr, cfa_c, i);
        }
    }};
}

macro_rules! apply_kernel_c {
    ($row:ident, $prev:expr, $curr:expr, $cfa:expr, $i:expr) => {{
        // current = B/R, diagonal = R/B.
        let (c, d) = if $cfa == ColorFilterArray::Bggr {
            (2, 0)
        } else {
            (0, 2)
        };
        let j = $i + PADDING;

        $row[3 * $i + c] = $curr[j];
        $row[3 * $i + 1] = $curr[j - 1];
        $row[3 * $i + d] = $prev[j - 1];
    }};
}

macro_rules! apply_kernel_g {
    ($row:ident, $prev:expr, $curr:expr, $cfa:expr, $i:expr) => {{
        // horizontal = B/R, vertical = R/G.
        let (h, v) = if $cfa == ColorFilterArray::Gbrg {
            (2, 0)
        } else {
            (0, 2)
        };
        let j = $i + PADDING;

        $row[3 * $i + h] = $curr[j - 1];
        $row[3 * $i + 1] = $curr[j];
        $row[3 * $i + v] = $prev[j];
    }};
}

/*--------------------------------------------------------------*/

fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: PixelStor,
{
    let (w, h) = (dst.w, dst.h);
    let mut prev = vec![T::zero(); 2 * PADDING + w];
    let mut curr = vec![T::zero(); 2 * PADDING + w];
    let mut cfa = cfa;

    let rdr = BorderReplicate::new(w, PADDING);
    rdr.read_row(r, &mut prev)?;
    rdr.read_row(r, &mut curr)?;

    {
        // y = 0.
        let row = dst.borrow_row_mut(0);
        apply_kernel_row!(row, curr, prev, cfa, w);
        cfa = cfa.next_y();
    }

    {
        // y = 1.
        let row = dst.borrow_row_mut(1);
        apply_kernel_row!(row, prev, curr, cfa, w);
        cfa = cfa.next_y();
    }

    for y in 2..h {
        rotate!(prev <- curr);
        rdr.read_row(r, &mut curr)?;

        let row = dst.borrow_row_mut(y);
        apply_kernel_row!(row, prev, curr, cfa, w);
        cfa = cfa.next_y();
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
            229, 67, 51, 229, 67, 51, 95, 67, 51, 95, 146, 241, 229, 232, 51, 229, 232, 51, 95,
            229, 51, 95, 229, 241, 169, 161, 51, 169, 161, 51, 15, 161, 51, 15, 52, 241, 169, 45,
            175, 169, 45, 175, 15, 98, 175, 15, 98, 197,
        ];

        const IMG_W: usize = 4;
        const IMG_H: usize = 4;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(
            &src,
            ColorFilterArray::Rggb,
            &mut RasterMut::new(IMG_W, IMG_H, &mut buf),
        );
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn test_odd() {
        // R: set.seed(0); matrix(floor(runif(n=9, min=0, max=256)), nrow=3, byrow=TRUE)
        let src = [229, 67, 95, 146, 232, 51, 229, 241, 169];

        let expected = [
            229, 67, 232, 229, 67, 232, 95, 67, 232, 229, 146, 232, 229, 146, 232, 95, 51, 232,
            229, 241, 232, 229, 241, 232, 169, 241, 232,
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
