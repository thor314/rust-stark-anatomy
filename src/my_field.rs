use std::{fmt, ops::*};

use rand::Rng;

use crate::field::Field;

/// A field with a large subgroup of power-of-two-order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MyField {
  x: u128,
}

impl MyField {
  pub fn new(x: u128) -> Self { Self { x } }

  /// obtain a generator for power-of-two subgroups.
  pub fn primitive_nth_root(n: u128) -> Self {
    assert!(n.is_power_of_two());
    let mut root = Self::GENERATOR;
    let mut order = 1u128 << 119;

    while order != n {
      root *= root;
      order /= 2;
    }

    root
  }

  /// obtain a random element of the field
  fn sample(rng: &mut impl Rng) -> Self { Self::new(rng.gen_range(0..Self::MODULUS.x)) }
}

impl Field for MyField {
  /// note: unchecked
  const GENERATOR: Self = Self { x: 85408008396924667383611388730472331217 };
  /// chosen to contain a 2^119 order subgroup
  const MODULUS: Self = Self { x: 1 + 407 * (1 << 119) };
  const ONE: Self = Self { x: 1 };
  const ZERO: Self = Self { x: 0 };
}

impl fmt::Display for MyField {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.x) }
}

impl From<usize> for MyField {
  fn from(val: usize) -> Self { Self { x: val as u128 } }
}

impl From<MyField> for Vec<u8> {
  fn from(val: MyField) -> Vec<u8> { val.x.to_be_bytes().to_vec() }
}

impl Rem for MyField {
  type Output = Self;

  fn rem(self, rhs: Self) -> Self::Output { Self { x: self.x % rhs.x } }
}

impl Neg for MyField {
  type Output = Self;

  fn neg(self) -> Self::Output { Self { x: (Self::MODULUS.x - self.x) } }
}

impl Add for MyField {
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output { Self { x: (self.x + rhs.x) % Self::MODULUS.x } }
}

impl AddAssign for MyField {
  fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl Sub for MyField {
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output { Self::MODULUS + self + (-rhs) }
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

impl std::ops::Mul for MyField {
  type Output = Self;

  /// naive double and add, not accelerated or anything
  fn mul(self, rhs: Self) -> Self::Output {
    let (mut a, mut b) = (self, rhs);
    let (n0, n1, n2) = (Self::ZERO, Self::ONE, Self::new(2));
    let mut result = n0;

    while b > n0 {
      if b % n2 == n1 {
        result += a;
      }
      a = (n2 * a) % Self::MODULUS;
      b /= n2;
    }

    result
  }
}

impl MulAssign for MyField {
  fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs }
}

impl Div for MyField {
  type Output = Self;

  #[allow(clippy::suspicious_arithmetic_impl)]
  fn div(self, rhs: Self) -> Self::Output { self * rhs.inverse() }
}

impl DivAssign for MyField {
  fn div_assign(&mut self, rhs: Self) { *self = *self / rhs }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_xgcd() {
    let (a, b, gcd) = crate::field::xgcd(MyField::new(2), MyField::new(3));
    // assert_eq!(a, MyField::new(2));
    // assert_eq!(b, MyField::new(1));
    // assert_eq!(gcd, MyField::new(1));
    // let (a, b, gcd) = crate::field::xgcd(MyField::new(22), MyField::new(33));
    // assert_eq!(a, MyField::new(11));
    // assert_eq!(b, MyField::new(11));
    // assert_eq!(gcd, MyField::new(11));
  }

  #[test]
  fn test_pow_and_inv() {
    let felt = MyField::new(2);
    let feltp = felt.power(3);
    assert!(feltp.x == 8);
    let feltp = felt.power(0);
    assert!(feltp.x == 1);
    let felti = felt.inverse();
    assert!(felti * felt == MyField::new(1));
  }

  #[test]
  fn test_primitive_nth_root() {
    let root = MyField::primitive_nth_root(1 << 118);
    assert_eq!(root * MyField::new(2), MyField::ONE);
    let root = MyField::primitive_nth_root(1 << 119);
    assert_eq!(root, MyField::ONE);
  }
}
