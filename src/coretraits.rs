use bytemuck::NoUninit;
use num_traits::{Bounded, Num, NumCast, One, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

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
    ///
    /// The value is clamped into `[DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE]` (NaN
    /// maps to the minimum) *before* it is narrowed, so out-of-range and
    /// non-finite inputs saturate instead of panicking.
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        let min = Self::DEFAULT_MIN_VALUE.to_f64();
        let max = Self::DEFAULT_MAX_VALUE.to_f64();
        let v = if v.is_nan() { min } else { v.clamp(min, max) };
        NumCast::from(v).unwrap_or(Self::DEFAULT_MIN_VALUE)
    }

    /// Convert to f32.
    #[inline(always)]
    fn to_f32(self) -> f32 {
        NumCast::from(self).unwrap()
    }

    /// Convert from f32.
    ///
    /// The value is clamped into `[DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE]` (NaN
    /// maps to the minimum) *before* it is narrowed, so out-of-range and
    /// non-finite inputs saturate instead of panicking.
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        let min = Self::DEFAULT_MIN_VALUE.to_f32();
        let max = Self::DEFAULT_MAX_VALUE.to_f32();
        let v = if v.is_nan() { min } else { v.clamp(min, max) };
        NumCast::from(v).unwrap_or(Self::DEFAULT_MIN_VALUE)
    }

    impl_cast_floor!(u8);
    impl_cast_floor!(u16);

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
declare_pixelstor!(f32: (0.0)..1.0, PixelType::F32);

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
impl Enlargeable for f32 {
    type Larger = f64;
}

/// Declares a sub-container `PixelStor`: a `#[repr(transparent)]` newtype over
/// `u16` whose `DEFAULT_MIN/MAX_VALUE` (and so every cast/saturate/interpolate
/// that goes through them) is the sensor's true bit depth rather than the full
/// `u16` range. See [`U10`] for the shared rationale.
macro_rules! declare_subrange_pixelstor {
    ($name:ident, $max:literal, $doc:literal) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
        pub struct $name(pub u16);

        // SAFETY: `#[repr(transparent)]` over `u16`, which is `Pod` — identical
        // layout, no padding, every bit pattern valid.
        unsafe impl bytemuck::Zeroable for $name {}
        unsafe impl bytemuck::Pod for $name {}

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
        impl Mul for $name {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }
        impl Div for $name {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }
        impl Rem for $name {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self {
                Self(self.0 % rhs.0)
            }
        }

        impl Zero for $name {
            fn zero() -> Self {
                Self(0)
            }
            fn is_zero(&self) -> bool {
                self.0 == 0
            }
        }
        impl One for $name {
            fn one() -> Self {
                Self(1)
            }
        }
        impl Bounded for $name {
            fn min_value() -> Self {
                Self(0)
            }
            fn max_value() -> Self {
                Self($max)
            }
        }
        impl ToPrimitive for $name {
            fn to_i64(&self) -> Option<i64> {
                self.0.to_i64()
            }
            fn to_u64(&self) -> Option<u64> {
                self.0.to_u64()
            }
        }
        impl NumCast for $name {
            fn from<F: ToPrimitive>(n: F) -> Option<Self> {
                <u16 as NumCast>::from(n).map(Self)
            }
        }
        impl Num for $name {
            type FromStrRadixErr = <u16 as Num>::FromStrRadixErr;
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <u16 as Num>::from_str_radix(s, radix).map(Self)
            }
        }

        impl PixelStor for $name {
            const DEFAULT_MAX_VALUE: Self = Self($max);
            const DEFAULT_MIN_VALUE: Self = Self(0);
            const PIXEL_TYPE: PixelType = PixelType::$name;
        }

        impl Enlargeable for $name {
            type Larger = u32;
        }
    };
}

declare_subrange_pixelstor!(
    U10,
    1023,
    "A 10-bit sample stored right-aligned in a `u16` (see [`PixelType::U10`]).\n\n\
     It is bit-identical to `u16`, but every [`PixelStor`] cast and saturation \
     on `U10` (`from_f64`, `cast_u8`, …) is relative to `0..=1023` instead of \
     the full `u16` range. A [`pipeline`](crate::pipeline) stage dispatches to \
     `U10` instead of plain `u16`, so e.g. converting to `u8` \
     scales against the sensor's true range rather than clipping almost \
     everything to black."
);
declare_subrange_pixelstor!(
    U12,
    4095,
    "A 12-bit sample stored right-aligned in a `u16` (see [`PixelType::U12`]); \
     see [`U10`] for the rationale. The only difference is the saturating \
     range, `0..=4095`."
);
declare_subrange_pixelstor!(
    U14,
    16383,
    "A 14-bit sample stored right-aligned in a `u16` (see [`PixelType::U14`]); \
     see [`U10`] for the rationale. The only difference is the saturating \
     range, `0..=16383`."
);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pixelstor() {
        let v = 0.5f32;
        let u = v.cast_u8();
        assert_eq!(u, 128);
        let v = 0.4f32;
        let u = v.cast_u8();
        assert_eq!(u, 102); // f32::round(v * 255.0) as u8);
    }

    #[test]
    fn subrange_pixelstor_saturates_at_true_range() {
        assert_eq!(U10::DEFAULT_MAX_VALUE, U10(1023));
        assert_eq!(U12::DEFAULT_MAX_VALUE, U12(4095));
        assert_eq!(U14::DEFAULT_MAX_VALUE, U14(16383));
        assert_eq!(U10::PIXEL_TYPE, PixelType::U10);
        assert_eq!(U12::PIXEL_TYPE, PixelType::U12);
        assert_eq!(U14::PIXEL_TYPE, PixelType::U14);

        // `from_f64` clamps into the *tagged* range, not the full `u16` one.
        assert_eq!(U12::from_f64(5000.0), U12(4095));
        assert_eq!(U12::from_f64(-5.0), U12(0));
        assert_eq!(U12::from_f64(2000.0), U12(2000));

        // The max value of each range casts to the top of any target type.
        assert_eq!(U10(1023).cast_u8(), u8::MAX);
        assert_eq!(U12(4095).cast_u8(), u8::MAX);
        assert_eq!(U14(16383).cast_u8(), u8::MAX);
        assert_eq!(U12(4095).cast_u16(), u16::MAX);
        assert_eq!(U12(4095).cast_f32(), 1.0);

        // The `u16` storage representation is a public field.
        assert_eq!(U12(4000).0, 4000u16);

        // `bytemuck` reinterprets a `u16` buffer as any of the sub-ranges.
        let mut buf = [500u16, 1000];
        let tagged: &mut [U12] = bytemuck::cast_slice_mut(&mut buf);
        assert_eq!(tagged, &[U12(500), U12(1000)]);
    }
}
