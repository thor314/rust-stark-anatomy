//! A translation of Alan Szepiniac's Anatomy of a Stark blog post
//! https://aszepieniec.github.io/stark-anatomy/
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use anyhow::Result;

// mod error;
#[cfg(test)] mod tests;
mod field;
mod my_field;

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
