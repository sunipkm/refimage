use bytemuck::NoUninit;
use num_traits::{Bounded, Num, NumCast, ToPrimitive, Zero};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::ops::AddAssign;

use crate::PixelType;

extern crate paste;
macro_rules! impl_cast_floor {
    ($to:ty) => {
        ::paste::paste! {
            #[doc = "Cast the value to [`" $to " `], by scaling the value to requisite range."]
            #[inline(always)]
            fn [<cast_ $to>](self) -> $to {
                let mut val: f32 = NumCast::from(self).unwrap();
                let min: f32 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
                let max: f32 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
                val -= min;
                val /= max - min;
                val *= (<$to>::DEFAULT_MAX_VALUE as f32 + <$to>::DEFAULT_MIN_VALUE as f32);
                val -= <$to>::DEFAULT_MIN_VALUE as f32;
                val.round() as $to
            }

            #[doc = "Cast the value to [`" $to " `], by scaling the value to the requisite range. Floors the value in the end."]
            #[inline(always)]
            fn [<floor_ $to>](self) -> $to {
                let mut val: f32 = NumCast::from(self).unwrap();
                let min: f32 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
                let max: f32 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
                val -= min;
                val /= max - min;
                val *= (<$to>::DEFAULT_MAX_VALUE as f32 + <$to>::DEFAULT_MIN_VALUE as f32);
                val -= <$to>::DEFAULT_MIN_VALUE as f32;
                val.floor() as $to
            }
        }
    }
}

/// The type of each channel in a pixel. For example, this can be `u8`, `u16`, `f32`.
pub trait PixelStor:
    Copy + NumCast + Num + PartialOrd<Self> + Clone + Bounded + Send + Sync + NoUninit
{
    /// The maximum value for this type of primitive within the context of color.
    /// For floats, the maximum is `1.0`, whereas the integer types inherit their usual maximum values.
    const DEFAULT_MAX_VALUE: Self;

    /// The minimum value for this type of primitive within the context of color.
    /// For floats, the minimum is `0.0`, whereas the integer types inherit their usual minimum values.
    const DEFAULT_MIN_VALUE: Self;

    /// The pixel type of the primitive.
    const PIXEL_TYPE: PixelType;

    /// Convert to f64.
    #[inline(always)]
    fn to_f64(self) -> f64 {
        NumCast::from(self).unwrap()
    }

    /// Convert from f64.
    /// This function will clamp the value to the range of the type.
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        let v = NumCast::from(v).unwrap();
        if v > Self::DEFAULT_MAX_VALUE {
            Self::DEFAULT_MAX_VALUE
        } else if v < Self::DEFAULT_MIN_VALUE {
            Self::DEFAULT_MIN_VALUE
        } else {
            v
        }
    }

    /// Convert to f32.
    #[inline(always)]
    fn to_f32(self) -> f32 {
        NumCast::from(self).unwrap()
    }

    /// Convert from f32.
    /// This function will clamp the value to the range of the type.
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        let v = NumCast::from(v).unwrap();
        if v > Self::DEFAULT_MAX_VALUE {
            Self::DEFAULT_MAX_VALUE
        } else if v < Self::DEFAULT_MIN_VALUE {
            Self::DEFAULT_MIN_VALUE
        } else {
            v
        }
    }

    impl_cast_floor!(u8);
    impl_cast_floor!(u16);
    impl_cast_floor!(u32);
    impl_cast_floor!(i8);
    impl_cast_floor!(i16);
    impl_cast_floor!(i32);

    /// Cast the value to `f32`, by scaling the value to requisite range.
    #[inline(always)]
    fn cast_f32(self) -> f32 {
        let mut val: f32 = NumCast::from(self).unwrap();
        let min: f32 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
        let max: f32 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
        val -= min;
        val /= max - min;
        val
    }

    /// Cast the value to `f64`, by scaling the value to the requisite range.
    #[inline(always)]
    fn cast_f64(self) -> f64 {
        let mut val: f64 = NumCast::from(self).unwrap();
        let min: f64 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
        let max: f64 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
        val -= min;
        val /= max - min;
        val
    }
}

macro_rules! declare_pixelstor {
    ($base:ty: ($from:expr)..$to:expr, $pty: path) => {
        impl PixelStor for $base {
            const DEFAULT_MAX_VALUE: Self = $to;
            const DEFAULT_MIN_VALUE: Self = $from;
            const PIXEL_TYPE: PixelType = $pty;
        }
    };
}

declare_pixelstor!(u8: (0)..Self::MAX, PixelType::U8);
declare_pixelstor!(u16: (0)..Self::MAX, PixelType::U16);
declare_pixelstor!(u32: (0)..Self::MAX, PixelType::U32);

declare_pixelstor!(i8: (Self::MIN)..Self::MAX, PixelType::I8);
declare_pixelstor!(i16: (Self::MIN)..Self::MAX, PixelType::I16);
declare_pixelstor!(i32: (Self::MIN)..Self::MAX, PixelType::I32);

declare_pixelstor!(f32: (0.0)..1.0,  PixelType::F32);
declare_pixelstor!(f64: (0.0)..1.0, PixelType::F64);

/// An `Enlargable::Larger` value should be enough to calculate
/// the sum (average) of a few hundred or thousand Enlargeable values.
pub trait Enlargeable: Sized + Bounded + NumCast + Copy {
    /// The larger type that can hold the sum of `Self` values.
    type Larger: Copy
        + NumCast
        + Num
        + PartialOrd<Self::Larger>
        + Clone
        + Bounded
        + AddAssign
        + Zero;

    /// Clamp a larger value to the range of the smaller type.
    #[inline(always)]
    fn clamp_larger(n: Self::Larger) -> Self {
        if n > Self::max_value().make_larger() {
            Self::max_value()
        } else if n < Self::min_value().make_larger() {
            Self::min_value()
        } else {
            NumCast::from(n).expect("Failed to cast to Self")
        }
    }

    /// Convert the value to a larger type.
    #[inline(always)]
    fn make_larger(self) -> Self::Larger {
        NumCast::from(self).unwrap()
    }
}

pub(crate) fn get_mean<T>(values: &[T]) -> T
where
    T: PixelStor + Enlargeable,
{
    let sum = values
        .iter()
        .fold(T::Larger::zero(), |acc, &x| acc + x.make_larger());
    let n = NumCast::from(values.len()).unwrap();
    let mean = sum / n;
    T::clamp_larger(mean)
}

#[allow(dead_code)]
pub(crate) fn get_clamp<T>(value: T) -> T
where
    T: PixelStor + Enlargeable,
{
    T::clamp_larger(value.make_larger())
}

#[inline(always)]
pub(crate) fn do_prod<T>(v1: T, v2: i32) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    v1.make_larger() * NumCast::from(v2).unwrap()
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn do_prod2<T>(v1: T, v2: T) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    v1.make_larger() * v2.make_larger()
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn do_sum<T>(src: &[T]) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    src.iter()
        .fold(T::Larger::zero(), |acc, &x| acc + x.make_larger())
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn do_div<T>(v1: T::Larger, v2: i32) -> T
where
    T: PixelStor + Enlargeable,
{
    let div = v1 / NumCast::from(v2).unwrap();
    T::clamp_larger(div)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn do_div2<T>(v1: T, v2: i32) -> T
where
    T: PixelStor + Enlargeable,
{
    let div = v1.make_larger() / NumCast::from(v2).unwrap();
    T::clamp_larger(div)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn do_sub<T>(v1: T::Larger, v2: T::Larger) -> T
where
    T: PixelStor + Enlargeable,
{
    let sub = v1 - v2;
    T::clamp_larger(sub)
}

#[inline(always)]
pub(crate) fn large_to_f64<T>(v: T) -> f64
where
    T: Copy + ToPrimitive,
{
    NumCast::from(v).unwrap()
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn f64_to_larger<T>(v: f64) -> T::Larger
where
    T: Enlargeable,
{
    NumCast::from(v).unwrap()
}

#[inline(always)]
pub(crate) fn do_div_float<T>(v1: f64, v2: i32) -> T
where
    T: PixelStor + Enlargeable,
{
    NumCast::from(v1 / v2 as f64).unwrap_or(T::max_value())
}

impl Enlargeable for u8 {
    type Larger = u32;
}
impl Enlargeable for u16 {
    type Larger = u32;
}
impl Enlargeable for u32 {
    type Larger = u64;
}
impl Enlargeable for i8 {
    type Larger = i32;
}
impl Enlargeable for i16 {
    type Larger = i32;
}
impl Enlargeable for i32 {
    type Larger = i64;
}
impl Enlargeable for f32 {
    type Larger = f64;
}
impl Enlargeable for f64 {
    type Larger = f64;
}

macro_rules! impl_pixelstor_cast {
    ($to: ty) => {
        ::paste::paste! {
            #[doc = "Cast a slice of T to a slice of [`" $to " `], by scaling the value to requisite range."]
            #[inline(never)]
            pub(crate) fn [<cast_ $to>]<T: PixelStor>(data: &[T]) -> Vec<$to> {
                #[cfg(not(feature = "rayon"))]
                {
                    data.iter().map(|&x| x.[<cast_ $to>]()).collect()
                }
                #[cfg(feature = "rayon")]
                {
                    data.par_iter().map(|&x| x.[<cast_ $to>]()).collect()
                }
            }
        }
    }
}

impl_pixelstor_cast!(u8);
impl_pixelstor_cast!(u16);
impl_pixelstor_cast!(f32);

mod test {
    #[test]
    fn test_pixelstor() {
        use crate::coretraits::PixelStor;
        let v = 0.5f32;
        let u = v.cast_u8();
        assert_eq!(u, 128);
        let v = 0.4f32;
        let u = v.cast_u8();
        assert_eq!(u, 102); // f32::round(v * 255.0) as u8);
    }
}
