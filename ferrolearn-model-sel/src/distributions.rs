//! Distributions for [`RandomizedSearchCV`](crate::RandomizedSearchCV).
//!
//! This module defines:
//!
//! - [`Distribution`] — a trait for objects that can sample a [`ParamValue`].
//! - [`Uniform`] — uniform sampling over a continuous `[low, high)` range.
//! - [`LogUniform`] — log-uniform (log-scale) sampling.
//! - [`IntUniform`] — uniform sampling over an integer `[low, high]` range.
//! - [`Choice`] — uniform sampling from a fixed list of values.
//!
//! # Example
//!
//! ```rust
//! use ferrolearn_model_sel::distributions::{Distribution, Uniform, Choice};
//! use ferrolearn_model_sel::ParamValue;
//! use rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! let mut rng = SmallRng::seed_from_u64(0);
//! let dist = Uniform::new(0.0, 1.0);
//! let val = dist.sample(&mut rng);
//! if let ParamValue::Float(v) = val {
//!     assert!(v >= 0.0 && v < 1.0);
//! }
//! ```

use rand::Rng;
use rand::rngs::SmallRng;

use crate::ParamValue;

// ---------------------------------------------------------------------------
// Distribution trait
// ---------------------------------------------------------------------------

/// A trait for objects that can sample a [`ParamValue`] from some distribution.
///
/// Implementors produce a single random [`ParamValue`] each time
/// [`sample`](Distribution::sample) is called.
///
/// The method takes a concrete [`SmallRng`] reference so that the trait is
/// [dyn-compatible] and can be used as `Box<dyn Distribution>`.
///
/// [dyn-compatible]: https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility
pub trait Distribution {
    /// Draw one sample from this distribution.
    fn sample(&self, rng: &mut SmallRng) -> ParamValue;
}

// ---------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------

/// Uniform continuous distribution over `[low, high)`.
///
/// Samples a [`ParamValue::Float`].
///
/// # Panics
///
/// Panics at construction if `low >= high`.
#[derive(Debug, Clone)]
pub struct Uniform {
    low: f64,
    high: f64,
}

impl Uniform {
    /// Create a new [`Uniform`] distribution over `[low, high)`.
    ///
    /// # Panics
    ///
    /// Panics if `low >= high`.
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low < high, "Uniform: low ({low}) must be < high ({high})");
        Self { low, high }
    }
}

impl Distribution for Uniform {
    fn sample(&self, rng: &mut SmallRng) -> ParamValue {
        let v: f64 = rng.random_range(self.low..self.high);
        ParamValue::Float(v)
    }
}

// ---------------------------------------------------------------------------
// LogUniform
// ---------------------------------------------------------------------------

/// Log-uniform distribution over `[low, high)` in log space.
///
/// Equivalent to drawing `exp(Uniform(log(low), log(high)))`. Samples a
/// [`ParamValue::Float`].
///
/// Useful for hyperparameters like learning rates or regularisation strengths
/// that span several orders of magnitude.
///
/// # Panics
///
/// Panics at construction if `low <= 0` or `low >= high`.
#[derive(Debug, Clone)]
pub struct LogUniform {
    log_low: f64,
    log_high: f64,
}

impl LogUniform {
    /// Create a new [`LogUniform`] distribution over `[low, high)`.
    ///
    /// # Panics
    ///
    /// Panics if `low <= 0.0` or `low >= high`.
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low > 0.0, "LogUniform: low ({low}) must be > 0");
        assert!(
            low < high,
            "LogUniform: low ({low}) must be < high ({high})"
        );
        Self {
            log_low: low.ln(),
            log_high: high.ln(),
        }
    }
}

impl Distribution for LogUniform {
    fn sample(&self, rng: &mut SmallRng) -> ParamValue {
        let log_val: f64 = rng.random_range(self.log_low..self.log_high);
        ParamValue::Float(log_val.exp())
    }
}

// ---------------------------------------------------------------------------
// IntUniform
// ---------------------------------------------------------------------------

/// Uniform integer distribution over `[low, high]` (inclusive on both ends).
///
/// Samples a [`ParamValue::Int`].
///
/// # Panics
///
/// Panics at construction if `low > high`.
#[derive(Debug, Clone)]
pub struct IntUniform {
    low: i64,
    high: i64,
}

impl IntUniform {
    /// Create a new [`IntUniform`] distribution over `[low, high]`.
    ///
    /// # Panics
    ///
    /// Panics if `low > high`.
    pub fn new(low: i64, high: i64) -> Self {
        assert!(
            low <= high,
            "IntUniform: low ({low}) must be <= high ({high})"
        );
        Self { low, high }
    }
}

impl Distribution for IntUniform {
    fn sample(&self, rng: &mut SmallRng) -> ParamValue {
        // rand 0.9: random_range for inclusive range uses `..=`.
        let v: i64 = rng.random_range(self.low..=self.high);
        ParamValue::Int(v)
    }
}

// ---------------------------------------------------------------------------
// Choice
// ---------------------------------------------------------------------------

/// Uniform discrete distribution — sample uniformly from a fixed list.
///
/// Samples a clone of one of the values in the provided list.
///
/// # Panics
///
/// Panics at construction if the value list is empty.
#[derive(Debug, Clone)]
pub struct Choice {
    values: Vec<ParamValue>,
}

impl Choice {
    /// Create a new [`Choice`] distribution.
    ///
    /// # Panics
    ///
    /// Panics if `values` is empty.
    pub fn new(values: Vec<ParamValue>) -> Self {
        assert!(!values.is_empty(), "Choice: values must not be empty");
        Self { values }
    }
}

impl Distribution for Choice {
    fn sample(&self, rng: &mut SmallRng) -> ParamValue {
        let idx = rng.random_range(0..self.values.len());
        self.values[idx].clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // -- Uniform ------------------------------------------------------------

    #[test]
    fn test_uniform_samples_in_range() {
        let mut rng = seeded_rng();
        let dist = Uniform::new(1.0, 5.0);
        for _ in 0..100 {
            if let ParamValue::Float(v) = dist.sample(&mut rng) {
                assert!((1.0..5.0).contains(&v), "value {v} out of [1, 5)");
            } else {
                panic!("expected Float");
            }
        }
    }

    #[test]
    fn test_uniform_produces_float() {
        let mut rng = seeded_rng();
        let dist = Uniform::new(0.0, 1.0);
        assert!(matches!(dist.sample(&mut rng), ParamValue::Float(_)));
    }

    #[test]
    #[should_panic]
    fn test_uniform_panics_low_ge_high() {
        let _ = Uniform::new(5.0, 1.0);
    }

    // -- LogUniform ---------------------------------------------------------

    #[test]
    fn test_log_uniform_samples_in_range() {
        let mut rng = seeded_rng();
        let dist = LogUniform::new(1e-4, 1e-1);
        for _ in 0..100 {
            if let ParamValue::Float(v) = dist.sample(&mut rng) {
                assert!((1e-4..1e-1).contains(&v), "value {v} out of [1e-4, 1e-1)");
            } else {
                panic!("expected Float");
            }
        }
    }

    #[test]
    fn test_log_uniform_produces_float() {
        let mut rng = seeded_rng();
        let dist = LogUniform::new(0.001, 10.0);
        assert!(matches!(dist.sample(&mut rng), ParamValue::Float(_)));
    }

    #[test]
    #[should_panic]
    fn test_log_uniform_panics_low_zero() {
        let _ = LogUniform::new(0.0, 1.0);
    }

    // -- IntUniform ---------------------------------------------------------

    #[test]
    fn test_int_uniform_samples_in_range() {
        let mut rng = seeded_rng();
        let dist = IntUniform::new(1, 10);
        for _ in 0..100 {
            if let ParamValue::Int(v) = dist.sample(&mut rng) {
                assert!((1..=10).contains(&v), "value {v} out of [1, 10]");
            } else {
                panic!("expected Int");
            }
        }
    }

    #[test]
    fn test_int_uniform_produces_int() {
        let mut rng = seeded_rng();
        let dist = IntUniform::new(0, 5);
        assert!(matches!(dist.sample(&mut rng), ParamValue::Int(_)));
    }

    #[test]
    fn test_int_uniform_single_value() {
        let mut rng = seeded_rng();
        let dist = IntUniform::new(7, 7);
        for _ in 0..10 {
            assert_eq!(dist.sample(&mut rng), ParamValue::Int(7));
        }
    }

    // -- Choice -------------------------------------------------------------

    #[test]
    fn test_choice_samples_from_list() {
        let mut rng = seeded_rng();
        let values = vec![
            ParamValue::Float(0.1),
            ParamValue::Float(1.0),
            ParamValue::Float(10.0),
        ];
        let dist = Choice::new(values.clone());
        for _ in 0..50 {
            let sampled = dist.sample(&mut rng);
            assert!(values.contains(&sampled), "sampled value not in list");
        }
    }

    #[test]
    fn test_choice_single_value() {
        let mut rng = seeded_rng();
        let values = vec![ParamValue::Bool(true)];
        let dist = Choice::new(values);
        for _ in 0..10 {
            assert_eq!(dist.sample(&mut rng), ParamValue::Bool(true));
        }
    }

    #[test]
    #[should_panic]
    fn test_choice_panics_empty() {
        let _ = Choice::new(vec![]);
    }
}
