use std::{path::Path, time::SystemTime};

use refimage::{DynamicImageData, FitsCompression, GenericImage};
use image::open;

fn main() {
    color_backtrace::install();
    let img = open("examples/test2.jpeg").unwrap();
    let dyn_img = DynamicImageData::try_from(img).unwrap();
    let mut img = GenericImage::new(
        SystemTime::now(),
        dyn_img
    );
    img.insert_key("TEST_I8", 10i8).expect("Failed to insert key");
    img.insert_key("TEST_STR", "Hello, world!".to_string()).expect("Failed to insert key");
    img.write_fits(Path::new("test.fits"), FitsCompression::None, true).expect("Failed to write FITS file");
}