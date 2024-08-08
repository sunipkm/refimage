//! Bayer error codes.

use quick_error::quick_error;

pub type BayerResult<T> = Result<T, BayerError>;

quick_error! {

#[derive(Debug)]
pub enum BayerError {
    // Generic failure.  Please try to make something more meaningful.
    NoGood {
        display("No good")
    }

    WrongResolution {
        display("Wrong resolution")
    }
    WrongDepth {
        display("Wrong depth")
    }
}

}
