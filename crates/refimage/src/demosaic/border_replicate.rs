//! Bayer reader that replicates pixels on the border.
//!
//! If the raw data is given by the unprimed values shown below, this
//! reader will produce the following row, where the primed values
//! have the same value as the unprimed values.
//!
//! ```text
//!   r0' g0' r0' g0' | r0 g0 r1 g1 r2 g2 ... rl gl rm gm rn gn | rn' gn' rn' gn'
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
pub struct BorderReplicate(usize, usize, usize, usize, RefCell<usize>);

macro_rules! fill_row {
    ($dst:ident, $x1:expr, $x2:expr, $x3:expr) => {{
        let mut i;

        // Left border.
        let r0 = $dst[$x1 + 0];
        let g0 = $dst[$x1 + 1];
        i = 0;
        if $x1 % 2 == 1 {
            $dst[0] = g0;
            i = 1;
        }
        while i < $x1 {
            $dst[i + 0] = r0;
            $dst[i + 1] = g0;
            i += 2;
        }

        // Right border.
        let r0 = $dst[$x2 - 2];
        let g0 = $dst[$x2 - 1];
        i = $x2;
        while i + 1 < $x3 {
            $dst[i + 0] = r0;
            $dst[i + 1] = g0;
            i += 2;
        }
        if i == $x3 - 1 {
            $dst[i] = r0;
        }
    }};
}

impl BorderReplicate {
    pub fn new(width: usize, padding: usize) -> Self {
        let x1 = padding;
        let x2 = x1.checked_add(width).expect("overflow");
        let x3 = x2.checked_add(padding).expect("overflow");
        assert!(width >= 2);

        BorderReplicate(x1, x2, x3, width, RefCell::new(0))
    }

    pub fn unpack(&self) -> (usize, usize, usize, usize, usize) {
        let BorderReplicate(x1, x2, x3, width, _) = *self;
        let items_read = *self.4.borrow();
        (x1, x2, x3, width, items_read)
    }

    pub fn update(&self, items_read: usize) {
        self.4.replace(items_read);
    }
}

impl<T: Copy> BayerRead<T> for BorderReplicate {
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
    use super::BorderReplicate;
    use crate::demosaic::border_replicate::BayerRead;

    #[test]
    fn test_replicate_even() {
        let src = [1, 2, 3, 4, 5, 6];

        let expected = [
            1, 2, 1, 2, /*-----*/ 1, 2, 3, 4, 5, 6, /*--------------------*/ 5, 6, 5, 6,
        ];

        let rdr = BorderReplicate::new(6, 4);
        let mut buf = [0u8; 4 + 6 + 4];

        let res = rdr.read_row(&src[..], &mut buf);
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn test_replicate_odd() {
        let src = [1, 2, 3, 4, 5];

        let expected = [
            2, 1, 2, /*---*/ 1, 2, 3, 4, 5, /*---------------*/ 4, 5, 4,
        ];

        let rdr = BorderReplicate::new(5, 3);
        let mut buf = [0u8; 3 + 5 + 3];

        let res = rdr.read_row(&src[..], &mut buf);
        assert!(res.is_ok());
        assert_eq!(&buf[..], &expected[..]);
    }
}
