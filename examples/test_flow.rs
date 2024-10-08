use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use image::DynamicImage;
use refimage::{
    BayerPattern, ColorSpace, DemosaicMethod, DynamicImageRef, FitsCompression, FitsWrite,
    GenericImageRef, ImageProps, ImageRef, ToLuma,
};
use refimage::{CalcOptExp, GenericImage, OptimumExposureBuilder};

fn main() {
    // color_backtrace::install();
    let mut src = [
        229u8, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
    ];
    let expected = [
        229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241, 169,
        0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
    ];
    let img = ImageRef::new(&mut src, 4, 4, BayerPattern::Rggb.into())
        .expect("Failed to create ImageRef");
    let img = DynamicImageRef::from(img);
    let mut img = GenericImageRef::new(SystemTime::now(), img);
    img.insert_key("Camera", "Canon EOS 5D Mark III")
        .expect("Failed to insert key");
    img.insert_key("Lens", "EF24-70mm f/2.8L II USM")
        .expect("Failed to insert key");
    let img = GenericImage::from(img);
    let mut debayered = img
        .debayer(DemosaicMethod::None)
        .expect("Failed to debayer");
    assert!(debayered.channels() == 3);
    assert!(debayered.width() == 4);
    assert!(debayered.height() == 4);
    assert!(debayered.color_space() == ColorSpace::Rgb);
    let ptr = debayered.as_slice_u8().unwrap();
    assert_eq!(ptr, &expected);
    debayered
        .write_fits(&PathBuf::from("./test.fits"), FitsCompression::None, true)
        .expect("Failed to write FITS");
    let dimg: DynamicImage = debayered
        .clone()
        .try_into()
        .expect("Failed to convert to DynamicImage");
    dimg.save("test.png").expect("Failed to save image");
    debayered.to_luma().expect("Failed to convert to luma");
    let eval = OptimumExposureBuilder::default()
        .pixel_exclusion(1)
        .build()
        .expect("Failed to build OptimumExposure");
    let (exp, _) = debayered
        .calc_opt_exp(&eval, Duration::from_secs(1), 1)
        .expect("Failed to calculate optimum exposure");
    println!("Optimum exposure: {exp:?}");
}
