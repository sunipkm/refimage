use std::time::Duration;

use crate::DynamicImageRef;
use crate::{
    CalcOptExp, ColorSpace, DynamicImageOwned, ExposureResult, ImageOwned, ImageProps,
    OptimumExposure, OptimumExposureResult, PixelData, PixelType,
};

macro_rules! dynamic_map(
    ($dynimage: expr, $image: pat => $action: expr) => ({
        use DynamicImageOwned::*;
        match $dynimage {
            U8($image) => U8($action),
            U16($image) => U16($action),
            F32($image) => F32($action),
        }
    });

    ($dynimage: expr, $image:pat_param, $action: expr) => (
        match $dynimage {
            DynamicImageOwned::U8($image) => $action,
            DynamicImageOwned::U16($image) => $action,
            DynamicImageOwned::F32($image) => $action,
        }
    );
);

impl ImageProps for DynamicImageOwned {
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

impl From<&DynamicImageOwned> for PixelType {
    fn from(data: &DynamicImageOwned) -> Self {
        crate::ImageProps::pixel_type(data)
    }
}

macro_rules! tryfrom_dynimgdata_imgdata {
    ($type:ty, $variant:path) => {
        impl<'a> TryFrom<DynamicImageOwned> for ImageOwned<$type> {
            type Error = crate::ImageError;

            fn try_from(data: DynamicImageOwned) -> Result<Self, Self::Error> {
                match data {
                    $variant(data) => Ok(data),
                    _ => Err(crate::ImageError::PixelTypeMismatch),
                }
            }
        }
    };
}

tryfrom_dynimgdata_imgdata!(u8, DynamicImageOwned::U8);
tryfrom_dynimgdata_imgdata!(u16, DynamicImageOwned::U16);
tryfrom_dynimgdata_imgdata!(f32, DynamicImageOwned::F32);

macro_rules! from_imgdata_dynimg {
    ($type:ty, $variant:path) => {
        impl<'a> From<ImageOwned<$type>> for DynamicImageOwned {
            fn from(data: ImageOwned<$type>) -> Self {
                $variant(data)
            }
        }
    };
}

from_imgdata_dynimg!(u8, DynamicImageOwned::U8);
from_imgdata_dynimg!(u16, DynamicImageOwned::U16);
from_imgdata_dynimg!(f32, DynamicImageOwned::F32);

impl DynamicImageOwned {
    /// The whole sample buffer as a mutable `&mut [u8]`, whatever the element
    /// type (native endianness).
    pub fn as_mut_raw_u8(&mut self) -> &mut [u8] {
        dynamic_map!(self, image, image.as_mut_u8_slice())
    }
}

impl PixelData for DynamicImageOwned {
    fn as_raw_u8(&self) -> &[u8] {
        dynamic_map!(self, image, { image.as_u8_slice() })
    }

    fn as_raw_u8_checked(&self) -> Option<&[u8]> {
        dynamic_map!(self, image, { image.as_u8_slice_checked() })
    }

    fn as_slice_u8(&self) -> Option<&[u8]> {
        match self {
            DynamicImageOwned::U8(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_u8(&mut self) -> Option<&mut [u8]> {
        match self {
            DynamicImageOwned::U8(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    fn as_slice_u16(&self) -> Option<&[u16]> {
        match self {
            DynamicImageOwned::U16(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_u16(&mut self) -> Option<&mut [u16]> {
        match self {
            DynamicImageOwned::U16(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }

    fn as_slice_f32(&self) -> Option<&[f32]> {
        match self {
            DynamicImageOwned::F32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn as_mut_slice_f32(&mut self) -> Option<&mut [f32]> {
        match self {
            DynamicImageOwned::F32(data) => Some(data.as_mut_slice()),
            _ => None,
        }
    }
}

impl From<&DynamicImageRef<'_>> for DynamicImageOwned {
    fn from(data: &DynamicImageRef<'_>) -> Self {
        match data {
            DynamicImageRef::U8(data) => DynamicImageOwned::U8(data.into()),
            DynamicImageRef::U16(data) => DynamicImageOwned::U16(data.into()),
            DynamicImageRef::F32(data) => DynamicImageOwned::F32(data.into()),
        }
    }
}

impl CalcOptExp for DynamicImageOwned {
    fn calc_opt_exp(
        &mut self,
        eval: &OptimumExposure,
        exposure: Duration,
        bin: u16,
    ) -> ExposureResult<OptimumExposureResult> {
        use DynamicImageOwned::*;
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
        let img = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let img = crate::ImageOwned::from_owned(img, 5, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageOwned");
        let mut img = crate::DynamicImageOwned::from(img);
        let res = img
            .calc_opt_exp(&opt_exp, std::time::Duration::from_secs(10), 1)
            .unwrap();
        assert_eq!(res.exposure, std::time::Duration::from_secs(10));
        assert_eq!(res.bin, 1);
    }
}
