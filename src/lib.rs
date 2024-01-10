//! A translation of Alan Szepiniac's Anatomy of a Stark blog post
//! https://aszepieniec.github.io/stark-anatomy/
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use anyhow::Result;

mod error;
#[cfg(test)] mod tests;

mod field {
  /// extended Euclidean algorithm for multiplicative inverses
  /// returns (a, b, gcd(x, y)) such that ax + by = gcd(x, y)
  pub fn xgcd<F: Field>(x: &F, y: &F) -> (F, F, F) {
    let zero = F::ZERO;
    let one = F::ONE;
    let (mut old_r, mut r) = (x.clone(), y.clone());
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
    fn inverse(&self) -> Self{
      let (x, _, gcd) = xgcd(self, &Self::ONE);
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

mod my_field {
  use std::ops::{Deref, DerefMut};

  use rand::Rng;

  use crate::field::Field;

  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
  pub struct MyField {
    x: u128,
  }

  impl MyField {
    const GENERATOR: u128 = 85408008396924667383611388730472331217u128;
    const MODULUS: u128 = 1 + 407 * (1 << 119);

    pub fn new(x: u128) -> Self { Self { x } }

    /// obtain a generator for power-of-two subgroups
    pub fn primitive_nth_root(n: u128) -> Self {
      assert!(n.is_power_of_two());
      let mut root = Self::GENERATOR;
      let mut order = 1u128 << 119;
      while order != n {
        root *= root;
        order /= 2;
      }
      Self::new(root)
    }

    fn sample(rng: &mut impl Rng) -> Self { Self::new(rng.gen_range(0..Self::MODULUS)) }
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

    fn inverse(&self) -> Self {
      let (x, _, gcd) = crate::field::xgcd(self, &Self::ONE);
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
    fn sub_assign(&mut self, rhs: Self) { self.x = self.x - rhs.x; }
  }

  impl std::ops::Sub for MyField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
      if self.x < rhs.x {
        Self { x: Self::MODULUS - (rhs.x - self.x) }
      } else {
        Self { x: self.x - rhs.x }
      }
    }
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

  #[cfg(test)]
  mod test {
    use super::*;
    #[test]
    fn test_xgcd() {
      let (a, b, gcd) = crate::field::xgcd(&MyField::new(2), &MyField::new(3));
      assert_eq!(a, MyField::new(2));
      assert_eq!(b, MyField::new(1));
      assert_eq!(gcd, MyField::new(1));
      let (a, b, gcd) = crate::field::xgcd(&MyField::new(22), &MyField::new(33));
      assert_eq!(a, MyField::new(11));
      assert_eq!(b, MyField::new(11));
      assert_eq!(gcd, MyField::new(11));
    }
    #[test]
    fn test_pow_and_inv() {
      let felt = MyField::new(2);
      let feltp = felt.power(3);
      assert!(feltp == MyField::new(8));
      let feltp = felt.power(0);
      assert!(feltp == MyField::new(1));
      let felti = felt.inverse();
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
      let mut acc = Self::new(vec![F::ZERO]);
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
      acc
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
