/// extended Euclidean algorithm for multiplicative inverses
/// returns (a, b, gcd(x, y)) such that ax + by = gcd(x, y)
pub fn xgcd<F: Field>(x: F, y: F) -> (F, F, F) {
  let (n0, n1) = (F::ZERO, F::ONE);
  let (mut old_r, mut r) = (std::cmp::max(x, y), std::cmp::min(x, y));
  let (mut old_s, mut s) = (n1, n0);
  let (mut old_t, mut t) = (n0, n1);
  while r != n0 {
    let q = old_r / r;
    dbg!(r, old_r - q * r);
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
    (old_t, t) = (t, old_t - q * t);
    // dbg!("after", old_r, r);
  }

  (old_s, old_t, old_r)
}

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
  + From<usize>
  + std::fmt::Debug
  + std::fmt::Display
  + Into<Vec<u8>> {
  const ZERO: Self;
  const ONE: Self;
  // this will actually be a prime field trait, tee hee
  const GENERATOR: Self;
  const MODULUS: Self;
  fn is_zero(&self) -> bool { self == &Self::ZERO }
  fn inverse(&self) -> Self {
    let (x, _, gcd) = xgcd(*self, Self::ONE);
    assert_eq!(gcd, Self::ONE);
    x
  }
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
