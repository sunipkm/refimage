//! Bayer error codes.

use quick_error::quick_error;

pub type BayerResult<T> = Result<T, BayerError>;

quick_error! {

#[derive(Debug)]
/// Error codes for the Bayer demosaicing.
pub enum BayerError {
    /// Generic failure.  Please try to make something more meaningful.
    NoGood {
        display("No good")
    }
    /// The image is not the right size.
    WrongResolution {
        display("Wrong resolution")
    }
    /// The image is not the right depth.
    WrongDepth {
        display("Wrong depth")
    }
}

}
