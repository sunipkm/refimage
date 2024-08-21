use refimage::{ColorSpace, Debayer, DemosaicMethod, ImageData};

fn main() {
    // color_backtrace::install();
    let src = [
        229, 67, 95, 146, 232, 51, 229, 241, 169, 161, 15, 52, 45, 175, 98, 197,
    ];
    let expected = [
        229, 0, 0, 0, 67, 0, 95, 0, 0, 0, 146, 0, 0, 232, 0, 0, 0, 51, 0, 229, 0, 0, 0, 241,
        169, 0, 0, 0, 161, 0, 15, 0, 0, 0, 52, 0, 0, 45, 0, 0, 0, 175, 0, 98, 0, 0, 0, 197,
    ];
    let img = ImageData::from_owned(
        src.into(),
        4,
        4,
        crate::ColorSpace::Rggb,
    ).expect("Failed to create ImageData");
    let a = img.debayer(DemosaicMethod::None);
    assert!(a.is_ok());
    let a = a.unwrap(); // at this point, a is an ImageData struct
    assert!(a.channels() == 3);
    assert!(a.width() == 4);
    assert!(a.height() == 4);
    assert!(a.color_space() == ColorSpace::Rgb);
    assert_eq!(a.as_slice(), &expected);
}