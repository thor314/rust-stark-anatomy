use std::{ops::*, str::Bytes};
// full impl: https://github.com/arkworks-rs/algebra/blob/master/ff/src/fields/mod.rs#L161
pub trait Field:
  Add<Output = Self>
  + AddAssign
  + Sub<Output = Self>
  + SubAssign
  + Mul<Output = Self>
  + MulAssign
  + Div<Output = Self>
  + DivAssign
  + Neg<Output = Self>
  + Sized
  + Copy
  + Clone
  + PartialEq
  + Eq
  + PartialOrd
  + Ord
  + BitXor<Output = Self>
  + std::fmt::Debug
  + std::fmt::Display
  + Into<Vec<u8>> {
  const ZERO: Self;
  const ONE: Self;
  fn is_zero(&self) -> bool { self == &Self::ZERO }
  fn inverse(&self) -> Self;
  fn power(&self, n: usize) -> Self {
    let mut acc = Self::ONE;
    let mut val = *self;
    let mut n = n;
    while n > 0 {
      if n & 1 == 1 {
        acc *= val;
      }
      val *= val;
      n >>= 1;
    }
    acc
  }
}
