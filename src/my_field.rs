use std::{
  cmp::{max, min},
  fmt::{self, Debug},
  ops::*,
  u32,
};

use rand::Rng;

use crate::field::Field;

/// A field with a large subgroup of power-of-two-order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MyField {
  x: u32,
}

impl MyField {
  const GENERATOR: u32 = 3;
  // see generate_field.py on how modulus and generator were chosen.
  /// chosen to contain a 2^22 order subgroup
  const MODULUS: u32 = 104857601;
  const ORDER: u32 = 22;

  pub fn new(x: u32) -> Self { Self { x } }

  /// obtain a generator of order 2**n for generating power-of-two subgroups.
  pub fn primitive_nth_root(n: u32) -> Self {
    assert!(n <= Self::ORDER);
    let mut n = n;
    let mut g: u64 = Self::GENERATOR as u64;
    let p = Self::MODULUS as u64;

    while n > 0 {
      g = g * g % p;
      n -= 1;
    }

    Self::new(g as u32)
  }

  /// obtain a random element of the field
  fn sample(rng: &mut impl Rng) -> Self { Self::new(rng.gen_range(0..Self::MODULUS)) }

  /// extended Euclidean algorithm for multiplicative inverses
  /// returns (a, b, gcd(x, y)) such that ax + by = gcd(x, y)
  fn xgcd(x: Self, y: Self) -> (Self, Self, Self) {
    let (n0, n1) = (Self::ZERO, Self::ONE);
    let (mut r_, mut r) = (max(x, y), min(x, y));
    let (mut s_, mut s) = (n0, n0);
    let (mut t_, mut t) = (n0, n1);

    while r != n0 {
      let q = MyField::new(r_.x / r.x);
      (r_, r) = (r, r_ - q * r);
      (s_, s) = (s, s_ - q * s);
      (t_, t) = (t, t_ - q * t);
    }

    // if ax + by = gcd(x, y)
    // (a, b, gcd(x, y))
    (s_, t_, r_)
  }
}

impl Field for MyField {
  const ONE: Self = Self { x: 1 };
  const ZERO: Self = Self { x: 0 };

  fn inverse(&self) -> Self {
    let (_a, b, _gcd) = Self::xgcd(Self::MODULUS.into(), *self);
    b
  }
}

impl From<MyField> for u32 {
  fn from(val: MyField) -> Self { val.x }
}

impl fmt::Display for MyField {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.x) }
}

impl From<u32> for MyField {
  fn from(val: u32) -> Self { Self { x: val } }
}

impl From<MyField> for Vec<u8> {
  fn from(val: MyField) -> Vec<u8> { val.x.to_be_bytes().to_vec() }
}

impl Rem<u32> for MyField {
  type Output = Self;

  fn rem(self, rhs: u32) -> Self::Output { Self { x: self.x % rhs } }
}

impl Rem for MyField {
  type Output = Self;

  fn rem(self, rhs: Self) -> Self::Output { Self { x: self.x % rhs.x } }
}

impl Neg for MyField {
  type Output = Self;

  fn neg(self) -> Self::Output { Self { x: ((Self::MODULUS - self.x) % Self::MODULUS) } }
}

impl Add for MyField {
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output { Self { x: (self.x + rhs.x) % Self::MODULUS } }
}

impl AddAssign for MyField {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl Sub for MyField {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output { self + (-rhs) }
}

impl SubAssign for MyField {
  fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl BitXor for MyField {
  type Output = Self;

  fn bitxor(self, rhs: Self) -> Self::Output { Self { x: self.x ^ rhs.x } }
}

impl BitXorAssign for MyField {
  fn bitxor_assign(&mut self, rhs: Self) { *self = *self ^ rhs; }
}

impl Mul for MyField {
  type Output = Self;

  /// naive double and add, not accelerated or anything
  fn mul(self, rhs: Self) -> Self::Output {
    let (mut a, mut b) = (max(self, rhs), min(self, rhs));
    let mut result = Self::ZERO;

    while b > Self::ZERO {
      if b & 1 == 1 {
        result += a;
      }
      a <<= 1 % Self::MODULUS;
      b >>= 1;
    }

    result % Self::MODULUS
  }
}

impl Shl<u32> for MyField {
  type Output = Self;

  fn shl(self, rhs: u32) -> Self::Output { Self { x: self.x << rhs } }
}

impl ShlAssign<u32> for MyField {
  fn shl_assign(&mut self, rhs: u32) { *self = *self << rhs; }
}

impl Shr<u32> for MyField {
  type Output = Self;

  fn shr(self, rhs: u32) -> Self::Output { Self { x: self.x >> rhs } }
}

impl ShrAssign<u32> for MyField {
  fn shr_assign(&mut self, rhs: u32) { *self = *self >> rhs; }
}

impl Mul<u32> for MyField {
  type Output = Self;

  fn mul(self, rhs: u32) -> Self::Output { self * Self::new(rhs) }
}

impl MulAssign for MyField {
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs }
}

impl Div<MyField> for MyField {
  type Output = Self;

  #[allow(clippy::suspicious_arithmetic_impl)]
  fn div(self, rhs: MyField) -> Self::Output { self * rhs.inverse() }
}

impl DivAssign for MyField {
  fn div_assign(&mut self, rhs: Self) { *self = *self / rhs }
}

impl DivAssign<u32> for MyField {
  fn div_assign(&mut self, rhs: u32) { *self /= Self::new(rhs) }
}

impl BitAnd<u32> for MyField {
  type Output = u32;

  fn bitand(self, rhs: u32) -> Self::Output { self.x & rhs }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_xgcd() {
    let (a, b, gcd) = MyField::xgcd(MyField::new(123), MyField::new(72));
    assert_eq!(a, MyField::new(0));
    assert_eq!(b, MyField::new(12));
    assert_eq!(gcd, MyField::new(3));
    let (a, b, gcd) = MyField::xgcd(MyField::new(20), MyField::new(33));
    assert_eq!(a, MyField::new(0));
    assert_eq!(b, MyField::new(5));
    assert_eq!(gcd, MyField::new(1));
  }

  #[test]
  fn test_pow() {
    let felt = MyField::new(2);
    let feltp = felt.power(3);
    assert!(feltp.x == 8);
    let feltp = felt.power(0);
    assert!(feltp.x == 1);
  }

  #[test]
  fn test_inv() {
    let felt = MyField::new(2);
    let felti = felt.inverse();
    assert_eq!(felti * felt, MyField::new(1));
  }

  #[test]
  fn test_primitive_nth_root() {
    let p = MyField::MODULUS as u128;
    let root = MyField::primitive_nth_root(MyField::ORDER);
    let root5 = (root.x as u128).pow(5) % p;
    let root25 = (root5 * root5 * root5 % p) * root5 * root5 % p;
    assert_eq!(root25, 1);
  }

  #[test]
  fn test_arith() {
    let a = MyField::new(123);
    let b = MyField::new(72);
    assert_eq!(a + b, MyField::new(195));
    assert_eq!(MyField::new(MyField::MODULUS) + a, a);
    assert_eq!(MyField::new(MyField::MODULUS) + 0.into(), 0.into());
    assert_eq!(a - b, MyField::new(51));
    assert_eq!(a * b, MyField::new(8856));
    assert_eq!(a % b, MyField::new(51));
    assert_eq!(-a, (MyField::MODULUS - 123).into());
    assert_eq!(a ^ b, MyField::new(51));
    let mut c = a;
    c += b;
    assert_eq!(c, MyField::new(195));
    c -= b;
    assert_eq!(c, 123.into());
    c *= b;
    assert_eq!(c, 8856.into());
    // let a_inv = a.inverse();
    // assert_eq!(a * a_inv, 1.into());
    // c /= b;
    // assert_eq!(c, MyField::new(123));
  }
}
