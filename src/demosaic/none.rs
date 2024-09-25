//! Demosaicing without any interpolation.
use crate::demosaic::RasterMut;
use crate::demosaic::{BayerError, BayerRead, BayerResult, ColorFilterArray};
use crate::{ImageRef, ImageOwned, PixelStor};

use super::border_none::BorderNone;

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
    ($row:ident, $curr:expr, $cfa:expr, $w:expr) => {{
        for e in $row.iter_mut() {
            *e = T::zero();
        }

        let (mut i, cfa_c) = if $cfa == ColorFilterArray::Bggr || $cfa == ColorFilterArray::Rggb {
            (0, $cfa)
        } else {
            apply_kernel_g!($row, $curr, 0);
            (1, $cfa.next_x())
        };

        while i + 1 < $w {
            apply_kernel_c!($row, $curr, cfa_c, i);
            apply_kernel_g!($row, $curr, i + 1);
            i += 2;
        }

        if i < $w {
            apply_kernel_c!($row, $curr, cfa_c, i);
        }
    }};
}

macro_rules! apply_kernel_c {
    ($row:ident, $curr:expr, $cfa:expr, $i:expr) => {{
        if $cfa == ColorFilterArray::Bggr {
            $row[3 * $i + 2] = $curr[$i];
        } else {
            $row[3 * $i + 0] = $curr[$i];
        }
    }};
}

macro_rules! apply_kernel_g {
    ($row:ident, $curr:expr, $i:expr) => {{
        $row[3 * $i + 1] = $curr[$i];
    }};
}

fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: PixelStor,
{
    let (w, h) = (dst.w, dst.h);
    let mut curr = vec![T::zero(); w];
    let mut cfa = cfa;

    let rdr = BorderNone::new();

    for y in 0..h {
        let row = dst.borrow_row_mut(y);
        rdr.read_row(r, &mut curr)?;
        apply_kernel_row!(row, curr, cfa, w);
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
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
            169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
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
            229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 0, 232, 0, 51, 0, 229, 0, 0, 0, 241, 0,
            169, 0, 0,
        ];

        const IMG_W: usize = 3;
        const IMG_H: usize = 3;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(
            &src[..],
            ColorFilterArray::Rggb,
            &mut RasterMut::new(IMG_W, IMG_H, &mut buf),
        );
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }
}
