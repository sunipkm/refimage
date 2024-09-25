use std::time::SystemTime;

use refimage::{
    BayerPattern, ColorSpace, Debayer, DemosaicMethod, DynamicImageOwned, GenericImageOwned,
    ImageOwned,
};

fn main() {
    // color_backtrace::install();
    let src = [
        229u8, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
    ];
    let expected = [
        229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241, 169,
        0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
    ];
    let img = ImageOwned::from_owned(src.into(), 4, 4, BayerPattern::Rggb.into())
        .expect("Failed to create ImageRef");
    let a = img.debayer(DemosaicMethod::None);
    assert!(a.is_ok());
    let a = a.unwrap(); // at this point, a is an ImageRef struct
    assert!(a.channels() == 3);
    assert!(a.width() == 4);
    assert!(a.height() == 4);
    assert!(a.color_space() == ColorSpace::Rgb);
    assert_eq!(a.as_slice(), &expected);
    let img = DynamicImageOwned::from(a);
    let mut gimg = GenericImageOwned::new(SystemTime::now(), img.clone());
    gimg.insert_key("Camera", "Canon EOS 5D Mark III")
        .expect("Failed to insert key");
    gimg.insert_key("Lens", "EF24-70mm f/2.8L II USM")
        .expect("Failed to insert key");
    println!("{:?}", gimg);
    assert_eq!(&img, gimg.get_image());
}
