use bytemuck::NoUninit;
use num_traits::{Bounded, Num, NumCast, ToPrimitive, Zero};
#[cfg(feature = "rayon")]
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::ops::AddAssign;

use crate::BayerPattern;

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

    /// Convert to f64.
    fn to_f64(self) -> f64 {
        NumCast::from(self).unwrap()
    }

    /// Convert from f64.
    /// This function will clamp the value to the range of the type.
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
    fn to_f32(self) -> f32 {
        NumCast::from(self).unwrap()
    }

    /// Convert from f32.
    /// This function will clamp the value to the range of the type.
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

    /// Cast the value to [`u8`], by scaling the value to the range `[0, 255]`.
    fn cast_u8(self) -> u8 {
        let mut val: f32 = NumCast::from(self).unwrap();
        let min: f32 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
        let max: f32 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
        val -= min;
        val /= max - min;
        val *= 255.0;
        val.round() as u8
    }

    /// Cast the value to [`u8`], by scaling the value to the range `[0, 255]`. Floors the value at the end.
    fn floor_u8(self) -> u8 {
        let mut val: f32 = NumCast::from(self).unwrap();
        let min: f32 = NumCast::from(Self::DEFAULT_MIN_VALUE).unwrap();
        let max: f32 = NumCast::from(Self::DEFAULT_MAX_VALUE).unwrap();
        val -= min;
        val /= max - min;
        val *= 255.0;
        val.floor() as u8
    }
}

macro_rules! declare_pixelstor {
    ($base:ty: ($from:expr)..$to:expr) => {
        impl PixelStor for $base {
            const DEFAULT_MAX_VALUE: Self = $to;
            const DEFAULT_MIN_VALUE: Self = $from;
        }
    };
}

declare_pixelstor!(u8: (0)..Self::MAX);
declare_pixelstor!(u16: (0)..Self::MAX);
declare_pixelstor!(u32: (0)..Self::MAX);

declare_pixelstor!(i8: (Self::MIN)..Self::MAX);
declare_pixelstor!(i16: (Self::MIN)..Self::MAX);
declare_pixelstor!(i32: (Self::MIN)..Self::MAX);

declare_pixelstor!(f32: (0.0)..1.0);
declare_pixelstor!(f64: (0.0)..1.0);

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

pub(crate) fn do_prod<T>(v1: T, v2: i32) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    v1.make_larger() * NumCast::from(v2).unwrap()
}

#[allow(dead_code)]
pub(crate) fn do_prod2<T>(v1: T, v2: T) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    v1.make_larger() * v2.make_larger()
}

#[allow(dead_code)]
pub(crate) fn do_sum<T>(src: &[T]) -> T::Larger
where
    T: PixelStor + Enlargeable,
{
    src.iter()
        .fold(T::Larger::zero(), |acc, &x| acc + x.make_larger())
}

#[allow(dead_code)]
pub(crate) fn do_div<T>(v1: T::Larger, v2: i32) -> T
where
    T: PixelStor + Enlargeable,
{
    let div = v1 / NumCast::from(v2).unwrap();
    T::clamp_larger(div)
}

#[allow(dead_code)]
pub(crate) fn do_div2<T>(v1: T, v2: i32) -> T
where
    T: PixelStor + Enlargeable,
{
    let div = v1.make_larger() / NumCast::from(v2).unwrap();
    T::clamp_larger(div)
}

#[allow(dead_code)]
pub(crate) fn do_sub<T>(v1: T::Larger, v2: T::Larger) -> T
where
    T: PixelStor + Enlargeable,
{
    let sub = v1 - v2;
    T::clamp_larger(sub)
}

pub(crate) fn large_to_f64<T>(v: T) -> f64
where
    T: Copy + ToPrimitive,
{
    NumCast::from(v).unwrap()
}

#[allow(dead_code)]
pub(crate) fn f64_to_larger<T>(v: f64) -> T::Larger
where
    T: Enlargeable,
{
    NumCast::from(v).unwrap()
}

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

/// Cast a slice of `T` to a slice of `u8`.
#[inline(never)]
pub(crate) fn cast_u8<T: PixelStor>(data: &[T]) -> Vec<u8> {
    #[cfg(not(feature = "rayon"))]
    {
        data.iter().map(|&x| x.cast_u8()).collect()
    }
    #[cfg(feature = "rayon")]
    {
        data.par_iter().map(|&x| x.cast_u8()).collect()
    }
}

/// Run the luminance conversion on a slice of pixel data.
#[inline(never)]
pub(crate) fn run_luma<T: PixelStor>(data: &[T], wts: &[f64]) -> Vec<T> {
    let channels = wts.len();
    #[cfg(not(feature = "rayon"))]
    {
        data.chunks_exact(channels)
            .map(|chunk| {
                T::from_f64(
                    chunk
                        .iter()
                        .zip(wts.iter())
                        .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
                )
            })
            .collect::<Vec<T>>()
    }
    #[cfg(feature = "rayon")]
    {
        data.par_chunks_exact(channels)
            .map(|chunk| {
                T::from_f64(
                    chunk
                        .iter()
                        .zip(wts.iter())
                        .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
                )
            })
            .collect::<Vec<T>>()
    }
}

mod test {
    #[test]
    fn test_pixelstor() {
        use crate::traits::PixelStor;
        let v = 0.5f32;
        let u = v.cast_u8();
        assert_eq!(u, 128);
        let v = 0.4f32;
        let u = v.cast_u8();
        assert_eq!(u, 102); // f32::round(v * 255.0) as u8);
    }
}

/// A trait for shifting Bayer patterns.
pub trait BayerShift {
    /// Shift the Bayer pattern by `x` and `y` pixels.
    fn shift(&self, x: usize, y: usize) -> Self;
}

impl BayerShift for BayerPattern {
    fn shift(&self, x: usize, y: usize) -> Self {
        match self {
            BayerPattern::Rggb => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Rggb,
                (1, 0) => BayerPattern::Gbrg,
                (0, 1) => BayerPattern::Grbg,
                (1, 1) => BayerPattern::Bggr,
                _ => unreachable!(),
            },
            BayerPattern::Gbrg => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Gbrg,
                (1, 0) => BayerPattern::Rggb,
                (0, 1) => BayerPattern::Bggr,
                (1, 1) => BayerPattern::Grbg,
                _ => unreachable!(),
            },
            BayerPattern::Grbg => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Grbg,
                (1, 0) => BayerPattern::Bggr,
                (0, 1) => BayerPattern::Rggb,
                (1, 1) => BayerPattern::Gbrg,
                _ => unreachable!(),
            },
            BayerPattern::Bggr => match (x % 2, y % 2) {
                (0, 0) => BayerPattern::Bggr,
                (1, 0) => BayerPattern::Grbg,
                (0, 1) => BayerPattern::Gbrg,
                (1, 1) => BayerPattern::Rggb,
                _ => unreachable!(),
            },
        }
    }
}
