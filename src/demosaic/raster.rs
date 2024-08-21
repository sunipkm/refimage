//! Raster implementation.
use crate::PixelStor;

use super::RasterMut;

impl<'a, T: PixelStor> RasterMut<'a, T> {
    /// Allocate a new raster for the given destination buffer slice.
    pub fn new(w: usize, h: usize, buf: &'a mut [T]) -> Self {
        let stride = w.checked_mul(3).expect("overflow");
        Self::with_offset(0, 0, w, h, stride, buf)
    }

    /// Allocate a new raster for the given destination buffer slice.
    /// Stride is in number of bytes.
    pub fn with_offset(
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        stride: usize,
        buf: &'a mut [T],
    ) -> Self {
        let x1 = x.checked_add(w).expect("overflow");
        let y1 = y.checked_add(h).expect("overflow");
        assert!(x < x1 && x1.checked_mul(3).expect("overflow") <= stride && h > 0);
        assert!(stride.checked_mul(y1).expect("overflow") <= buf.len());
        assert_eq!(stride % 3, 0);

        RasterMut {
            x,
            y,
            w,
            h,
            stride,
            buf,
        }
    }

    /// Borrow a mutable row slice.
    ///
    /// # Panics
    ///
    /// Panics if the raster is not 8-bpp.
    pub fn borrow_row_mut(&mut self, y: usize) -> &mut [T] {
        assert!(y < self.h);

        let bytes_per_pixel = 3;
        let start = self.stride * (self.y + y) + bytes_per_pixel * self.x;
        let end = start + bytes_per_pixel * self.w;
        &mut self.buf[start..end]
    }

    /// Get a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::RasterMut;

    #[test]
    #[should_panic]
    fn test_raster_mut_overflow() {
        let mut buf = [0; 1];
        let _ = RasterMut::new(usize::MAX, usize::MAX, &mut buf);
    }

    #[test]
    fn test_borrow_row_u16_mut() {
        let expected = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB];

        const IMG_W: usize = 4;
        const IMG_H: usize = 1;
        let mut buf = [0u16; 3 * IMG_W * IMG_H];

        {
            let mut dst = RasterMut::new(IMG_W, IMG_H, &mut buf);
            let row = dst.borrow_row_mut(0);

            for (i, elt) in row.iter_mut().enumerate() {
                *elt = i as _;
            }
        }

        assert_eq!(&buf[0..3 * IMG_W * IMG_H], &expected[..]);
    }
}
