//! A translation of Alan Szepiniac's Anatomy of a Stark blog post
//! https://aszepieniec.github.io/stark-anatomy/
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use anyhow::Result;

// mod error;
#[cfg(test)] mod tests;

mod field {
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
    fn is_zero(&self) -> bool { self == &Self::ZERO }
  }
}

mod my_field {
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
}

mod poly {
  use std::{
    iter::{once, repeat},
    ops::*,
  };

  use crate::field::Field;

  #[derive(Debug, Clone, PartialEq, Eq)]
  struct Polynomial<F: Field> {
    coeffs: Vec<F>,
  }

  impl<F: Field> Polynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self { Self { coeffs } }

    pub fn degree(&self) -> usize {
      if self.is_zero() {
        0
      } else {
        self.coeffs.len() - 1
      }
    }

    pub fn is_zero(&self) -> bool { self.coeffs.iter().all(|x| x.is_zero()) }

    pub fn leading_coefficient(&self) -> F { self.coeffs[self.degree()] }

    /// If `rhs` is zero, returns `None`. Otherwise, returns `(quotient, remainder)`.
    fn div_remainder(self, rhs: Self) -> Option<(Self /* quotient */, Self /* remainder */)> {
      if rhs.is_zero() {
        return None;
      }
      if self.degree() < rhs.degree() {
        return Some((Self::new(vec![]), self));
      }
      let rem = Self::new(self.coeffs.clone());
      let mut quot_coeffs: Vec<F> =
        repeat(F::ZERO).take(self.degree() - rhs.degree() + 1).collect();

      for i in 0..(self.degree() - rhs.degree() + 1) {
        if rem.degree() < rhs.degree() {
          break;
        }
        let coefficient = rem.leading_coefficient() / rhs.leading_coefficient();
        let shift = rem.degree() - rhs.degree();
        let subtractee =
          Self::new(repeat(F::ZERO).take(shift).chain(once(coefficient)).collect()) * rhs.clone();
        quot_coeffs[shift] = coefficient;
        let remainder = rem.clone() - subtractee;
      }

      let quot = Self::new(quot_coeffs);
      Some((quot, rem))
    }

    pub fn pow(&self, n: usize) -> Self {
      if n == 0 {
        return Self::new(vec![F::ONE]);
      }
      let mut acc = Self::new(vec![F::ONE]);
      let mut val = self.clone();
      let mut n = n;
      while n > 0 {
        if n & 1 == 1 {
          acc *= val.clone();
        }
        val *= val.clone();
        n >>= 1;
      }
      acc
    }

    // exploit horner's method to evaluate the polynomial
    pub fn evaluate(&self, x: F) -> F {
      let mut acc = F::ZERO;
      for i in (0..=self.degree()).rev() {
        acc *= x;
        acc += self.coeffs[i];
      }
      acc
    }

    /// evaluate the polynomial over some domain of points, returning the evaluations
    pub fn evaluate_domain(&self, domain: &[F]) -> Vec<F> {
      domain.iter().map(|x| self.evaluate(*x)).collect()
    }

    ///
    pub fn interpolate_domain(domain: &[F], values: &[F]) -> Self {
      assert_eq!(domain.len(), values.len());
      // let mut acc = Self::new(vec![F::ZERO]);
      for i in 0..domain.len() {
        // let mut term = Self::new(vec![F::ONE]);
        // for j in 0..domain.len() {
        //   if i == j {
        //     continue;
        //   }
        //   let mut tmp = Self::new(vec![F::ONE, -domain[j]]);
        //   tmp *= F::inverse(&(domain[i] - domain[j]));
        //   term *= tmp;
        // }
        // term *= values[i];
        // acc += term;
      }
      // acc
      todo!()
    }
  }

  impl<F: Field> Add for Polynomial<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
      let mut coeffs = vec![F::ZERO; std::cmp::max(self.coeffs.len(), rhs.coeffs.len())];
      for i in 0..self.coeffs.len() {
        coeffs[i] += self.coeffs[i];
      }
      for i in 0..rhs.coeffs.len() {
        coeffs[i] += rhs.coeffs[i];
      }
      Self::new(coeffs)
    }
  }

  impl<F: Field> Sub for Polynomial<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
      let mut coeffs = vec![F::ZERO; std::cmp::max(self.coeffs.len(), rhs.coeffs.len())];
      for i in 0..self.coeffs.len() {
        coeffs[i] += self.coeffs[i];
      }
      for i in 0..rhs.coeffs.len() {
        coeffs[i] -= rhs.coeffs[i];
      }
      Self::new(coeffs)
    }
  }

  impl<F: Field> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
      let coeffs = self.coeffs.into_iter().map(|x| -x).collect::<Vec<_>>();
      Self::new(coeffs)
    }
  }

  impl<F: Field> Mul for Polynomial<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
      let mut coeffs = vec![F::ZERO; self.coeffs.len() + rhs.coeffs.len() - 1];
      for i in 0..self.coeffs.len() {
        for j in 0..rhs.coeffs.len() {
          coeffs[i + j] += self.coeffs[i] * rhs.coeffs[j];
        }
      }
      Self::new(coeffs)
    }
  }

  impl<F: Field> MulAssign for Polynomial<F> {
    fn mul_assign(&mut self, rhs: Self) { *self = self.clone() * rhs; }
  }

  impl<F: Field> AsRef<[F]> for Polynomial<F> {
    fn as_ref(&self) -> &[F] { &self.coeffs }
  }

  impl<F: Field> Div for Polynomial<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
      let (q, r) = self.div_remainder(rhs).expect("failed to divide");
      assert!(r.is_zero(), "non-zero remainder");
      q
    }
  }

  impl<F: Field> Rem for Polynomial<F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
      let (q, r) = self.div_remainder(rhs).expect("failed to divide");
      assert!(r.is_zero(), "non-zero remainder");
      r
    }
  }
}
