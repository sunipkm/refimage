//! Demosaicing using cubic interpolation.
//!
//! ```text
//!   green_kernel = (1 / 256) *
//!       [   0   0   0   1   0   0   0
//!       ;   0   0  -9   0  -9   0   0
//!       ;   0  -9   0  81   0  -9   0
//!       ;   1   0  81 256  81   0   1
//!       ;   0  -9   0  81   0  -9   0
//!       ;   0   0  -9   0  -9   0   0
//!       ;   0   0   0   1   0   0   0 ];
//!
//!   red/blue_kernel = (1 / 256) *
//!       [   1   0  -9 -16  -9   0   1
//!       ;   0   0   0   0   0   0   0
//!       ;  -9   0  81 144  81   0  -9
//!       ; -16   0 144 256 144   0 -16
//!       ;  -9   0  81 144  81   0  -9
//!       ;   0   0   0   0   0   0   0
//!       ;   1   0  -9 -16  -9   0   1 ];
//! ```
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::demosaic::{BayerError, BayerRead, BayerResult, RasterMut, ColorFilterArray};
use crate::traits::{
    do_div_float, do_prod, get_mean,
    large_to_f64, Enlargeable,
};

#[cfg(feature = "rayon")]
use crate::demosaic::border_mirror::BorderMirror;
use crate::{ImageData, Primitive};

const PADDING: usize = 3;

pub fn run<T>(src: &ImageData<'_, T>, cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
{
    if dst.w < 4 || dst.h < 4 {
        return Err(BayerError::WrongResolution);
    }

    debayer(src.as_slice(), cfa, dst)
}

macro_rules! apply_kernel_row {
    ($T:ident; $row:ident,
            $prv3:expr, $prv2:expr, $prv1:expr, $curr:expr,
            $nxt1:expr, $nxt2:expr, $nxt3:expr,
            $cfa:expr, $w:expr) => {{
        let (mut i, cfa_c, cfa_g) =
            if $cfa == ColorFilterArray::Bggr || $cfa == ColorFilterArray::Rggb {
                (0, $cfa, $cfa.next_x())
            } else {
                apply_kernel_g!($T; $row, $w, $prv3, $prv2, $prv1, $curr, $nxt1, $nxt2, $nxt3, $cfa, 0);
                (1, $cfa.next_x(), $cfa)
            };

        while i + 1 < $w {
            apply_kernel_c!($T; $row, $w, $prv3, $prv2, $prv1, $curr, $nxt1, $nxt2, $nxt3, cfa_c, i);
            apply_kernel_g!($T; $row, $w, $prv3, $prv2, $prv1, $curr, $nxt1, $nxt2, $nxt3, cfa_g, i + 1);
            i += 2;
        }

        if i < $w {
            apply_kernel_c!($T; $row, $w, $prv3, $prv2, $prv1, $curr, $nxt1, $nxt2, $nxt3, cfa_c, i);
        }
    }}
}

macro_rules! apply_kernel_c {
    ($T:ident; $row:ident, $w:expr,
            $prv3:expr, $prv2:expr, $prv1:expr, $curr:expr,
            $nxt1:expr, $nxt2:expr, $nxt3:expr,
            $cfa:expr, $i:expr) => {{
        // current = B/R, diagonal = R/B.
        let (c, d) = if $cfa == ColorFilterArray::Bggr { (2, 0) } else { (0, 2) };
        let j = $i + PADDING;

        let g_pos = do_prod(
            get_mean(&[$prv1[j], $curr[j - 1], $curr[j + 1], $nxt1[j]]),
            81 * 4,
        ) + do_prod(
            get_mean(&[$prv3[j], $curr[j - 3], $curr[j + 3], $nxt3[j]]),
            4,
        );
        let g_neg = do_prod(
            get_mean(&[
                $prv2[j - 1],
                $prv2[j + 1],
                $prv1[j - 2],
                $prv1[j + 2],
                $nxt1[j - 2],
                $nxt1[j + 2],
                $nxt2[j - 1],
                $nxt2[j + 1],
            ]),
            9 * 8,
        );

        let d_pos = do_prod(
            get_mean(&[$prv1[j - 1], $prv1[j + 1], $nxt1[j - 1], $nxt1[j + 1]]),
            81 * 4,
        ) + do_prod(
            get_mean(&[$prv3[j - 3], $prv3[j + 3], $nxt3[j - 3], $nxt3[j + 3]]),
            4,
        );
        let d_neg = do_prod(
            get_mean(&[
                $prv3[j - 1],
                $prv3[j + 1],
                $prv1[j - 3],
                $prv1[j + 3],
                $nxt1[j - 3],
                $nxt1[j + 3],
                $nxt3[j - 1],
                $nxt3[j + 1],
            ]),
            9 * 8,
        );

        $row[3 * $i + c] = $curr[j];
        let x = large_to_f64(g_pos) - large_to_f64(g_neg);
        $row[3 * $i + 1] = do_div_float(x, 256);
        // $row[3 * $i + 1] = do_div(g_pos - g_neg, 256);
        let x = large_to_f64(d_pos) - large_to_f64(d_neg);
        $row[3 * $i + d] = do_div_float(x, 256);
    }};
}

macro_rules! apply_kernel_g {
    ($T:ident; $row:ident, $w:expr,
            $prv3:expr, $prv2:expr, $prv1:expr, $curr:expr,
            $nxt1:expr, $nxt2:expr, $nxt3:expr,
            $cfa:expr, $i:expr) => {{
        // horizontal = B/R, vertical = R/G.
        let (h, v) = if $cfa == ColorFilterArray::Gbrg { (2, 0) } else { (0, 2) };
        let j = $i + PADDING;

        let h_pos = do_prod(get_mean(&[$curr[j - 1], $curr[j + 1]]), 18);
        let h_neg = do_prod(get_mean(&[$curr[j - 3], $curr[j + 3]]), 2);
        let v_pos = do_prod(get_mean(&[$prv1[j], $nxt1[j]]), 18);
        let v_neg = do_prod(get_mean(&[$prv3[j], $nxt3[j]]), 2);

        let x = large_to_f64(h_pos) - large_to_f64(h_neg);
        $row[3 * $i + h] = do_div_float(x, 16);
        $row[3 * $i + 1] = $curr[j];
        let x = large_to_f64(v_pos) - large_to_f64(v_neg);
        $row[3 * $i + v] = do_div_float(x, 16);
    }};
}

/*--------------------------------------------------------------*/
/* Rayon                                                        */
/*--------------------------------------------------------------*/

#[cfg(feature = "rayon")]
#[allow(unused_parens)]
fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
    {
    let (w, h) = (dst.w, dst.h);
    let mut data = vec![T::zero(); (2 * PADDING + w) * (2 * PADDING + h)];

    // Read all data.
    {
        let stride = 2 * PADDING + w;
        let rdr = BorderMirror::new(w, PADDING);
        for row in data.chunks_mut(stride).skip(PADDING).take(h) {
            rdr.read_row(r, row)?;
        }

        {
            let (top, src) = data.split_at_mut(stride * PADDING);
            top[0..stride].copy_from_slice(&src[(stride * 3)..(stride * 4)]);
            top[stride..(stride * 2)].copy_from_slice(&src[(stride * 2)..(stride * 3)]);
            top[(stride * 2)..(stride * 3)].copy_from_slice(&src[stride..(stride * 2)]);
        }

        {
            let (src, bottom) = data.split_at_mut(stride * (h + PADDING));
            let yy = PADDING + h;
            bottom[0..stride]
                .copy_from_slice(&src[(stride * (yy - 2))..(stride * (yy - 1))]);
            bottom[stride..(stride * 2)]
                .copy_from_slice(&src[(stride * (yy - 3))..(stride * (yy - 2))]);
            bottom[(stride * 2)..(stride * 3)]
                .copy_from_slice(&src[(stride * (yy - 4))..(stride * (yy - 3))]);
        }
    }

    dst.buf
        .par_chunks_mut(dst.stride)
        .enumerate()
        .for_each(|(y, row)| {
            let stride = 2 * PADDING + w;
            let prv3 = &data[(stride * (PADDING + y - 3))..(stride * (PADDING + y - 2))];
            let prv2 = &data[(stride * (PADDING + y - 2))..(stride * (PADDING + y - 1))];
            let prv1 = &data[(stride * (PADDING + y - 1))..(stride * PADDING + y)];
            let curr = &data[(stride * PADDING + y)..(stride * (PADDING + y + 1))];
            let nxt1 = &data[(stride * (PADDING + y + 1))..(stride * (PADDING + y + 2))];
            let nxt2 = &data[(stride * (PADDING + y + 2))..(stride * (PADDING + y + 3))];
            let nxt3 = &data[(stride * (PADDING + y + 3))..(stride * (PADDING + y + 4))];
            let cfa_y = if y % 2 == 0 { cfa } else { cfa.next_y() };

            apply_kernel_row!(u8; row, prv3, prv2, prv1, curr, nxt1, nxt2, nxt3, cfa_y, w);
        });

    Ok(())
}

/*--------------------------------------------------------------*/
/* Naive                                                        */
/*--------------------------------------------------------------*/

#[cfg(not(feature = "rayon"))]
#[allow(unused_parens)]
fn debayer<T>(r: &[T], cfa: ColorFilterArray, dst: &mut RasterMut<'_, T>) -> BayerResult<()>
where
    T: Primitive + Enlargeable,
{
    use super::border_mirror::BorderMirror;

    let (w, h) = (dst.w, dst.h);
    let mut prv3 = vec![T::zero(); 2 * PADDING + w];
    let mut prv2 = vec![T::zero(); 2 * PADDING + w];
    let mut prv1 = vec![T::zero(); 2 * PADDING + w];
    let mut curr = vec![T::zero(); 2 * PADDING + w];
    let mut nxt1 = vec![T::zero(); 2 * PADDING + w];
    let mut nxt2 = vec![T::zero(); 2 * PADDING + w];
    let mut nxt3 = vec![T::zero(); 2 * PADDING + w];
    let mut cfa = cfa;

    let rdr = BorderMirror::new(w, PADDING);
    rdr.read_row(r, &mut curr)?;
    rdr.read_row(r, &mut nxt1)?;
    rdr.read_row(r, &mut nxt2)?;
    rdr.read_row(r, &mut nxt3)?;

    prv1.copy_from_slice(&nxt1);
    prv2.copy_from_slice(&nxt2);
    prv3.copy_from_slice(&nxt3);

    {
        // y = 0.
        let row = dst.borrow_row_mut(0);
        apply_kernel_row!(u8; row, nxt3, nxt2, nxt1, curr, nxt1, nxt2, nxt3, cfa, w);
        cfa = cfa.next_y();
    }

    for y in 1..(h - 3) {
        rotate!(prv3 <- prv2 <- prv1 <- curr <- nxt1 <- nxt2 <- nxt3);
        rdr.read_row(r, &mut nxt3)?;

        let row = dst.borrow_row_mut(y);
        apply_kernel_row!(u8; row, prv3, prv2, prv1, curr, nxt1, nxt2, nxt3, cfa, w);
        cfa = cfa.next_y();
    }

    {
        // y = h - 3.
        let row = dst.borrow_row_mut(h - 3);
        apply_kernel_row!(u8; row, prv2, prv1, curr, nxt1, nxt2, nxt3, nxt2, cfa, w);
        cfa = cfa.next_y();
    }

    {
        // y = h - 2.
        let row = dst.borrow_row_mut(h - 2);
        apply_kernel_row!(u8; row, prv1, curr, nxt1, nxt2, nxt3, nxt2, nxt1, cfa, w);
        cfa = cfa.next_y();
    }

    {
        // y = h - 1.
        let row = dst.borrow_row_mut(h - 1);
        apply_kernel_row!(u8; row, curr, nxt1, nxt2, nxt3, nxt2, nxt1, curr, cfa, w);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::debayer;
    use crate::demosaic::{RasterMut, ColorFilterArray};

    #[test]
    fn test_even() {
        // color_backtrace::install();
        // R: set.seed(0); matrix(floor(runif(n=64, min=0, max=256)), nrow=8, byrow=TRUE)
        let src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197, 127, 183, 253,
            97, 199, 239, 54, 166, 32, 68, 98, 3, 97, 222, 87, 123, 153, 126, 47, 211, 171, 203,
            27, 185, 105, 210, 165, 200, 141, 135, 202, 5, 122, 187, 177, 122, 220, 112, 62, 18,
            25, 80, 132, 169, 104, 233, 75, 117,
        ];

        let expected = [
            229, 122, 186, 161, 67, 172, 95, 42, 108, 154, 146, 58, 232, 60, 103, 238, 51, 169,
            229, 117, 196, 228, 241, 206, 182, 169, 174, 177, 110, 161, 177, 15, 98, 201, 69, 52,
            218, 45, 104, 188, 104, 175, 153, 98, 195, 145, 158, 197, 127, 158, 116, 185, 183, 105,
            253, 95, 48, 243, 97, 14, 199, 120, 106, 122, 239, 203, 54, 153, 194, 35, 166, 167,
            135, 32, 76, 141, 102, 68, 151, 98, 21, 175, 120, 3, 179, 97, 114, 104, 158, 222, 26,
            87, 179, 7, 115, 123, 153, 80, 146, 98, 126, 141, 47, 157, 115, 111, 211, 99, 171, 168,
            143, 106, 203, 174, 27, 179, 110, 9, 185, 52, 138, 105, 211, 114, 153, 210, 99, 165,
            209, 152, 166, 200, 193, 141, 174, 124, 178, 135, 42, 202, 57, 23, 159, 5, 122, 118,
            139, 142, 187, 145, 177, 157, 170, 211, 122, 194, 220, 109, 200, 143, 112, 184, 62, 93,
            113, 42, 18, 60, 118, 25, 68, 148, 128, 80, 193, 132, 120, 223, 111, 169, 226, 104,
            213, 148, 95, 233, 66, 75, 171, 46, 16, 117,
        ];

        const IMG_W: usize = 8;
        const IMG_H: usize = 8;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(&src, ColorFilterArray::Rggb, &mut RasterMut::new(IMG_W, IMG_H, &mut buf));
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn test_odd() {
        // R: set.seed(0); matrix(floor(runif(n=49, min=0, max=256)), nrow=7, byrow=TRUE)
        let src = [
            229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197, 127, 183, 253,
            97, 199, 239, 54, 166, 32, 68, 98, 3, 97, 222, 87, 123, 153, 126, 47, 211, 171, 203,
            27, 185, 105, 210, 165, 200, 141, 135, 202, 5, 122,
        ];

        let expected = [
            229, 147, 204, 161, 67, 183, 95, 122, 96, 154, 146, 12, 232, 52, 14, 238, 51, 38, 229,
            123, 41, 171, 241, 188, 136, 163, 169, 111, 161, 90, 176, 143, 15, 246, 52, 20, 243,
            92, 45, 225, 175, 48, 98, 235, 113, 102, 197, 103, 127, 184, 60, 195, 183, 23, 253, 95,
            41, 230, 97, 70, 199, 99, 76, 84, 239, 56, 87, 208, 54, 105, 166, 38, 160, 129, 32,
            201, 68, 63, 159, 53, 98, 116, 3, 106, 97, 231, 114, 88, 222, 104, 87, 178, 63, 126,
            123, 30, 153, 125, 62, 97, 126, 104, 47, 124, 113, 135, 211, 189, 121, 215, 171, 114,
            203, 94, 148, 165, 27, 173, 185, 57, 124, 138, 105, 79, 210, 114, 165, 202, 205, 150,
            200, 185, 141, 184, 101, 174, 135, 26, 202, 115, 56, 160, 5, 105, 122, 92, 115,
        ];

        const IMG_W: usize = 7;
        const IMG_H: usize = 7;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(&src, ColorFilterArray::Rggb, &mut RasterMut::new(IMG_W, IMG_H, &mut buf));
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn test_overflow() {
        // color_backtrace::install();
        let src = [
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
            255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        ];

        let expected = [
            255, 255, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 190, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 111, 174,
            255, 0, 111, 255, 111, 174, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 190, 255,
            255, 0, 111, 255, 255, 0, 255, 0, 111, 255, 190, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 111, 174, 255, 0, 111, 255, 111, 174, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 190, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 251,
        ];

        const IMG_W: usize = 7;
        const IMG_H: usize = 7;
        let mut buf = [0u8; 3 * IMG_W * IMG_H];

        let res = debayer(&src, ColorFilterArray::Rggb, &mut RasterMut::new(IMG_W, IMG_H, &mut buf));
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }
}
