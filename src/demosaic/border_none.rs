//! Bayer reader without any additional border logic.

use std::cell::RefCell;

use crate::demosaic::BayerResult;

use super::bayer::BayerRead;

pub struct BorderNone(RefCell<usize>);

impl BorderNone {
    pub fn new() -> Self {
        BorderNone(RefCell::new(0))
    }
}

impl<T: Copy> BayerRead<T> for BorderNone {
    fn read_row(&self, r: &[T], dst: &mut [T]) -> BayerResult<()> {
        let len = dst.len();
        let start = *self.0.borrow();
        let end = start.checked_add(len).expect("overflow");
        dst.copy_from_slice(&r[start..end]);
        self.0.replace(end);
        Ok(())
    }
}
