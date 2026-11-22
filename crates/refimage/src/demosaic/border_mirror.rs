//! Bayer reader that mirrors pixels on the border.
//!
//! If the raw data is given by the unprimed values shown below, this
//! reader will produce the following row, where the primed values
//! have the same value as the unprimed values.
//!
//! ```text
//!   r2' g1' r1' g0' | r0 g0 r1 g1 r2 g2 ... rl gl rm gm rn gn | rn' gm' rm' gl'
//! ```

use std::cell::RefCell;

use crate::demosaic::BayerResult;

use super::bayer::BayerRead;

/// Tuple structs (x1, x2, x3) designating the different sub-regions
/// of the output lines.
///
/// ```text
///    0 .. x1 => left border
///   x1 .. x2 => raw data
///   x2 .. x3 => right border
/// ```
pub struct BorderMirror(usize, usize, usize, usize, RefCell<usize>);

macro_rules! fill_row {
    ($dst:ident, $x1:expr, $x2:expr, $x3:expr) => {{
        let mut i;
        let mut j;

        // Left border.
        i = $x1;
        j = $x1 + 1;
        while i > 0 {
            $dst[i - 1] = $dst[j];
            i -= 1;
            j += 1;
        }

        // Right border.
        i = $x2;
        j = $x2 - 2;
        while i < $x3 {
            $dst[i] = $dst[j];
            i += 1;
            j -= 1;
        }
    }};
}

impl BorderMirror {
    pub fn new(width: usize, padding: usize) -> Self {
        let x1 = padding;
        let x2 = x1.checked_add(width).expect("overflow");
        let x3 = x2.checked_add(padding).expect("overflow");
        assert!(width > padding);

        BorderMirror(x1, x2, x3, width, RefCell::new(0))
    }

    fn unpack(&self) -> (usize, usize, usize, usize, usize) {
        let BorderMirror(x1, x2, x3, width, _) = *self;
        let start = *self.4.borrow();
        (x1, x2, x3, width, start)
    }

    fn update(&self, end: usize) {
        self.4.replace(end);
    }
}

impl<T: Copy> BayerRead<T> for BorderMirror {
    fn read_row(&self, r: &[T], dst: &mut [T]) -> BayerResult<()> {
        let (x1, x2, x3, width, start) = self.unpack();
        let end = start.checked_add(width).expect("overflow");
        dst[x1..x2].copy_from_slice(&r[start..end]);
        self.update(end);
        fill_row!(dst, x1, x2, x3);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BorderMirror;
    use crate::demosaic::border_mirror::BayerRead;

    #[test]
    fn test_mirror_even() {
        let src = [1, 2, 3, 4, 5, 6];

        let expected = [
            5, 4, 3, 2, /*-----*/ 1, 2, 3, 4, 5, 6, /*--------------------*/ 5, 4, 3, 2,
        ];

        let rdr = BorderMirror::new(6, 4);
        let mut buf = [0u8; 4 + 6 + 4];

        let res = rdr.read_row(&src[..], &mut buf);
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn test_mirror_odd() {
        let src = [1, 2, 3, 4, 5];

        let expected = [
            4, 3, 2, /*---*/ 1, 2, 3, 4, 5, /*---------------*/ 4, 3, 2,
        ];

        let rdr = BorderMirror::new(5, 3);
        let mut buf = [0u8; 3 + 5 + 3];

        let res = rdr.read_row(&src[..], &mut buf);
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }
}
