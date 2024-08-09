use num_traits::{Bounded, Num, NumCast, ToPrimitive, Zero};
use std::ops::AddAssign;

/// The type of each channel in a pixel. For example, this can be `u8`, `u16`, `f32`.
// TODO rename to `PixelComponent`? Split up into separate traits? Seal?
pub trait Primitive:
    Copy + NumCast + Num + PartialOrd<Self> + Clone + Bounded + Send + Sync
{
    /// The maximum value for this type of primitive within the context of color.
    /// For floats, the maximum is `1.0`, whereas the integer types inherit their usual maximum values.
    const DEFAULT_MAX_VALUE: Self;

    /// The minimum value for this type of primitive within the context of color.
    /// For floats, the minimum is `0.0`, whereas the integer types inherit their usual minimum values.
    const DEFAULT_MIN_VALUE: Self;
}

macro_rules! declare_primitive {
    ($base:ty: ($from:expr)..$to:expr) => {
        impl Primitive for $base {
            const DEFAULT_MAX_VALUE: Self = $to;
            const DEFAULT_MIN_VALUE: Self = $from;
        }
    };
}

declare_primitive!(u8: (0)..Self::MAX);
declare_primitive!(u16: (0)..Self::MAX);
declare_primitive!(u32: (0)..Self::MAX);

declare_primitive!(i8: (Self::MIN)..Self::MAX);
declare_primitive!(i16: (Self::MIN)..Self::MAX);
declare_primitive!(i32: (Self::MIN)..Self::MAX);
declare_primitive!(f32: (0.0)..1.0);
declare_primitive!(f64: (0.0)..1.0);

/// An `Enlargable::Larger` value should be enough to calculate
/// the sum (average) of a few hundred or thousand Enlargeable values.
pub trait Enlargeable: Sized + Bounded + NumCast + Copy {
    type Larger: Copy
        + NumCast
        + Num
        + PartialOrd<Self::Larger>
        + Clone
        + Bounded
        + AddAssign
        + Zero;

    /// Clamp a larger value to the range of the smaller type.
    fn clamp_from(n: Self::Larger) -> Self {
        if n > Self::max_value().to_larger() {
            Self::max_value()
        } else if n < Self::min_value().to_larger() {
            Self::min_value()
        } else {
            NumCast::from(n).expect("Failed to cast to Self")
        }
    }

    /// Convert the value to a larger type.
    fn to_larger(self) -> Self::Larger {
        NumCast::from(self).unwrap()
    }
}

pub(crate) fn get_mean<T>(values: &[T]) -> T
where
    T: Primitive + Enlargeable,
{
    let sum = values
        .iter()
        .fold(T::Larger::zero(), |acc, &x| acc + x.to_larger());
    let n = NumCast::from(values.len()).unwrap();
    let mean = sum / n;
    T::clamp_from(mean)
}

#[allow(dead_code)]
pub(crate) fn get_clamp<T>(value: T) -> T
where
    T: Primitive + Enlargeable,
{
    T::clamp_from(value.to_larger())
}

pub(crate) fn do_prod<T>(v1: T, v2: i32) -> T::Larger
where
    T: Primitive + Enlargeable,
{
    v1.to_larger() * NumCast::from(v2).unwrap()
}

#[allow(dead_code)]
pub(crate) fn do_prod2<T>(v1: T, v2: T) -> T::Larger
where
    T: Primitive + Enlargeable,
{
    v1.to_larger() * v2.to_larger()
}

#[allow(dead_code)]
pub(crate) fn do_sum<T>(src: &[T]) -> T::Larger
where
    T: Primitive + Enlargeable,
{
    src.iter()
        .fold(T::Larger::zero(), |acc, &x| acc + x.to_larger())
}

#[allow(dead_code)]
pub(crate) fn do_div<T>(v1: T::Larger, v2: i32) -> T
where
    T: Primitive + Enlargeable,
{
    let div = v1 / NumCast::from(v2).unwrap();
    T::clamp_from(div)
}

#[allow(dead_code)]
pub(crate) fn do_div2<T>(v1: T, v2: i32) -> T
where
    T: Primitive + Enlargeable,
{
    let div = v1.to_larger() / NumCast::from(v2).unwrap();
    T::clamp_from(div)
}

#[allow(dead_code)]
pub(crate) fn do_sub<T>(v1: T::Larger, v2: T::Larger) -> T
where
    T: Primitive + Enlargeable,
{
    let sub = v1 - v2;
    T::clamp_from(sub)
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
    T: Primitive + Enlargeable,
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

/// Linear interpolation without involving floating numbers.
pub trait Lerp: Bounded + NumCast {
    /// The type used to represent the ratio between two values.
    type Ratio: Primitive;

    /// Linear interpolation between two values.
    fn lerp(a: Self, b: Self, ratio: Self::Ratio) -> Self {
        let a = <Self::Ratio as NumCast>::from(a).unwrap();
        let b = <Self::Ratio as NumCast>::from(b).unwrap();

        let res = a + (b - a) * ratio;

        if res > NumCast::from(Self::max_value()).unwrap() {
            Self::max_value()
        } else if res < NumCast::from(0).unwrap() {
            NumCast::from(0).unwrap()
        } else {
            NumCast::from(res).unwrap()
        }
    }
}

impl Lerp for u8 {
    type Ratio = f32;
}

impl Lerp for u16 {
    type Ratio = f32;
}

impl Lerp for u32 {
    type Ratio = f64;
}

impl Lerp for f32 {
    type Ratio = f32;

    fn lerp(a: Self, b: Self, ratio: Self::Ratio) -> Self {
        a + (b - a) * ratio
    }
}
