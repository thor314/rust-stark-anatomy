#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use anyhow::Result;

mod error;
#[cfg(test)] mod tests;

mod math {
  /// extended Euclidean algorithm for multiplicative inverses
  /// returns (a, b, gcd(x, y)) such that ax + by = gcd(x, y)
  pub fn xgcd<F: Field>(x: F, y: F) -> (F, F, F) {
    let zero = F::ZERO;
    let one = F::ONE;
    let (mut old_r, mut r) = (x, y);
    let (mut old_s, mut s) = (one, zero);
    let (mut old_t, mut t) = (zero, one);
    while r != zero {
      let quotient = old_r; // r
      (old_r, r) = (r, old_r - quotient * r);
      (old_s, s) = (s, old_s - quotient * s);
      (old_t, t) = (t, old_t - quotient * t);
    }

    (old_s, old_t, old_r)
  }

  use std::ops::*;
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
    // + Rem<Output = Self>
    // + RemAssign
    + Sized
    + Copy
    + Clone
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + From<usize>
    + std::fmt::Debug
    + std::fmt::Display
     {
    const ZERO : Self;
    const ONE : Self;
    fn inverse(rhs: &Self) -> Self{
      let (x, _, gcd) = xgcd(*rhs, Self::ONE);
      assert_eq!(gcd, Self::ONE);
      x
    }
    fn xor(&self, rhs: &Self) -> Self;
    fn power(&self, n: usize) -> Self{
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
    fn as_bytes(&self) -> &[u8];
    fn is_zero(&self) -> bool {
      self == &Self::ZERO
    }
  }
}

mod test_field {
  use crate::math::Field;

  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
  pub struct MyField {
    x: u128,
  }

  impl MyField {
    const MODULUS: u128 = 1 + 407 * (1 << 119);

    pub fn new(x: u128) -> Self { Self { x } }
  }

  impl Field for MyField {
    const ONE: Self = Self { x: 1 };
    const ZERO: Self = Self { x: 0 };

    fn as_bytes(&self) -> &[u8] { unimplemented!() }

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

    fn is_zero(&self) -> bool { self == &Self::ZERO }

    fn inverse(rhs: &Self) -> Self {
      let (x, _, gcd) = crate::math::xgcd(*rhs, Self::ONE);
      assert_eq!(gcd, Self::ONE);
      x
    }

    fn xor(&self, rhs: &Self) -> Self { Self { x: self.x ^ rhs.x } }
  }

  impl std::fmt::Display for MyField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.x) }
  }

  impl std::convert::From<usize> for MyField {
    fn from(val: usize) -> Self { Self { x: val as u128 } }
  }

  impl std::ops::Neg for MyField {
    type Output = Self;

    fn neg(self) -> Self::Output { Self { x: Self::MODULUS - self.x } }
  }

  impl std::ops::AddAssign for MyField {
    fn add_assign(&mut self, rhs: Self) { self.x += rhs.x; }
  }

  impl std::ops::Add for MyField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output { Self { x: self.x + rhs.x } }
  }

  impl std::ops::SubAssign for MyField {
    fn sub_assign(&mut self, rhs: Self) { self.x -= rhs.x; }
  }

  impl std::ops::Sub for MyField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output { Self { x: self.x - rhs.x } }
  }

  impl std::ops::MulAssign for MyField {
    fn mul_assign(&mut self, rhs: Self) { self.x *= rhs.x; }
  }

  impl std::ops::Mul for MyField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output { Self { x: self.x * rhs.x } }
  }

  impl std::ops::DivAssign for MyField {
    fn div_assign(&mut self, rhs: Self) { self.x /= rhs.x; }
  }

  impl std::ops::Div for MyField {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output { Self { x: self.x / rhs.x } }
  }
}
