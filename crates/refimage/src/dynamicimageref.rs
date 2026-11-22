use std::time::Duration;

use crate::{
    CalcOptExp, ColorSpace, DynamicImageRef, ExposureResult, ImageProps, ImageRef, OptimumExposure,
    OptimumExposureResult, PixelData, PixelType,
};

macro_rules! dynamic_map(
    ($dynimage: expr, $image: pat => $action: expr) => ({
        use DynamicImageRef::*;
        match $dynimage {
            U8($image) => U8($action),
            U16($image) => U16($action),
            F32($image) => F32($action),
        }
    });

    ($dynimage: expr, $image:pat_param, $action: expr) => (
        match $dynimage {
            DynamicImageRef::U8($image) => $action,
            DynamicImageRef::U16($image) => $action,
            DynamicImageRef::F32($image) => $action,
        }
    );
);

impl DynamicImageRef<'_> {
    /// Get the width of the image.
    pub fn width(&self) -> usize {
        dynamic_map!(self, image, { image.width() })
    }

    /// Get the height of the image.
    pub fn height(&self) -> usize {
        dynamic_map!(self, image, { image.height() })
    }

    /// Get the number of channels in the image.
    pub fn channels(&self) -> u8 {
        dynamic_map!(self, image, { image.channels() })
    }

    /// Get the color space of the image.
    pub fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, image, { image.color_space() })
    }
}

impl ImageProps for DynamicImageRef<'_> {
    fn width(&self) -> usize {
        dynamic_map!(self, image, { image.width() })
    }

    fn height(&self) -> usize {
        dynamic_map!(self, image, { image.height() })
    }

    fn channels(&self) -> u8 {
        dynamic_map!(self, image, { image.channels() })
    }

    fn color_space(&self) -> ColorSpace {
        dynamic_map!(self, image, { image.color_space() })
    }

    fn pixel_type(&self) -> PixelType {
        dynamic_map!(self, image, { image.pixel_type() })
    }

    fn len(&self) -> usize {
        dynamic_map!(self, image, { image.len() })
    }

    fn is_empty(&self) -> bool {
        dynamic_map!(self, image, { image.is_empty() })
    }
}

impl From<&DynamicImageRef<'_>> for PixelType {
    fn from(data: &DynamicImageRef<'_>) -> Self {
        // Delegates to the inner image so a `u16` buffer tagged 10-/12-bit
        // reports `U10` / `U12` rather than `U16`.
        ImageProps::pixel_type(data)
    }
}

macro_rules! tryfrom_dynimgdata_imgdata {
    ($type:ty, $variant:path) => {
        impl<'a> TryFrom<DynamicImageRef<'a>> for ImageRef<'a, $type> {
            type Error = crate::ImageError;

            fn try_from(data: DynamicImageRef<'a>) -> Result<Self, Self::Error> {
                match data {
                    $variant(data) => Ok(data),
                    _ => Err(crate::ImageError::PixelTypeMismatch),
                }
            }
        }
    };
}

tryfrom_dynimgdata_imgdata!(u8, DynamicImageRef::U8);
tryfrom_dynimgdata_imgdata!(u16, DynamicImageRef::U16);
tryfrom_dynimgdata_imgdata!(f32, DynamicImageRef::F32);

macro_rules! from_imgdata_dynimg {
    ($type:ty, $variant:path) => {
        impl<'a> From<ImageRef<'a, $type>> for DynamicImageRef<'a> {
            fn from(data: ImageRef<'a, $type>) -> Self {
                $variant(data)
            }
        }
    };
}

from_imgdata_dynimg!(u8, DynamicImageRef::U8);
from_imgdata_dynimg!(u16, DynamicImageRef::U16);
from_imgdata_dynimg!(f32, DynamicImageRef::F32);

impl PixelData for DynamicImageRef<'_> {
    fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, image, { image.as_u8_slice() })
    }

    fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, image, { image.as_u8_slice_checked() })
    }

    fn as_slice_u8(&self) -> Option<&[u8]> {
        match self {
            DynamicImageRef::U8(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        match self {
            DynamicImageRef::U8(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageRef::U16(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        match self {
            DynamicImageRef::U16(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageRef::F32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        match self {
            DynamicImageRef::F32(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }
}

impl CalcOptExp for DynamicImageRef<'_> {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        use DynamicImageRef::*;
        match self {
            U8(img) => eval.calculate(img.as_mut_slice(), exposure, bin),
            U16(img) => eval.calculate(img.as_mut_slice(), exposure, bin),
            F32(img) => eval.calculate(img.as_mut_slice(), exposure, bin),
        }
    }
}

mod test {
    #[test]
    fn test_optimum_exposure() {
        use crate::CalcOptExp;
        let opt_exp = crate::OptimumExposureBuilder::default()
            .pixel_exclusion(1)
            .build()
            .unwrap();
        let mut img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let img = crate::ImageRef::new(img.as_mut_slice(), 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let mut img = crate::DynamicImageRef::from(img);
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);
    }
}
