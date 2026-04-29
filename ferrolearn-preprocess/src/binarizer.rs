//! Binarizer: threshold features to binary values.
//!
//! Values strictly greater than the threshold are set to `1.0`; all other
//! values are set to `0.0`.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// Binarizer
// ---------------------------------------------------------------------------

/// A stateless feature binarizer.
///
/// Values strictly greater than `threshold` become `1.0`; all other values
/// become `0.0`. The default threshold is `0.0`.
///
/// This transformer is stateless — no fitting is needed. Call
/// [`Transform::transform`] directly.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::binarizer::Binarizer;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let binarizer = Binarizer::<f64>::new(0.5);
/// let x = array![[0.0, 0.5, 1.0]];
/// let out = binarizer.transform(&x).unwrap();
/// // out = [[0.0, 0.0, 1.0]]
/// ```
#[derive(Debug, Clone)]
pub struct Binarizer<F> {
    /// The threshold value. Values strictly greater than this become 1.0.
    pub(crate) threshold: F,
}

impl<F: Float + Send + Sync + 'static> Binarizer<F> {
    /// Create a new `Binarizer` with the given threshold.
    #[must_use]
    pub fn new(threshold: F) -> Self {
        Self { threshold }
    }

    /// Return the configured threshold.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }
}

impl<F: Float + Send + Sync + 'static> Default for Binarizer<F> {
    fn default() -> Self {
        Self::new(F::zero())
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for Binarizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the threshold: values > threshold become `1.0`, others become `0.0`.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error for well-formed inputs.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let out = x.mapv(|v| {
            if v > self.threshold {
                F::one()
            } else {
                F::zero()
            }
        });
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_binarizer_default_threshold() {
        let b = Binarizer::<f64>::default();
        assert_eq!(b.threshold(), 0.0);
        let x = array![[-1.0, 0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -1 <= 0
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0 not > 0
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 0.5 > 0
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 1.0 > 0
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let b = Binarizer::<f64>::new(0.5);
        let x = array![[0.0, 0.5, 1.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 0.0 not > 0.5
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // 0.5 not > 0.5 (strict)
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // 1.0 > 0.5
    }

    #[test]
    fn test_binarizer_all_zeros() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[0.0, 0.0, 0.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_all_ones() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[1.0, 2.0, 3.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarizer_negative_threshold() {
        let b = Binarizer::<f64>::new(-1.0);
        let x = array![[-2.0, -1.0, -0.5, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // -2 <= -1
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10); // -1 not > -1
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10); // -0.5 > -1
        assert_abs_diff_eq!(out[[0, 3]], 1.0, epsilon = 1e-10); // 0.0 > -1
    }

    #[test]
    fn test_binarizer_multiple_rows() {
        let b = Binarizer::<f64>::new(2.0);
        let x = array![[1.0, 3.0], [2.0, 4.0], [5.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // 1 <= 2
        assert_abs_diff_eq!(out[[0, 1]], 1.0, epsilon = 1e-10); // 3 > 2
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10); // 2 not > 2
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-10); // 4 > 2
        assert_abs_diff_eq!(out[[2, 0]], 1.0, epsilon = 1e-10); // 5 > 2
        assert_abs_diff_eq!(out[[2, 1]], 0.0, epsilon = 1e-10); // 0 <= 2
    }

    #[test]
    fn test_binarizer_preserves_shape() {
        let b = Binarizer::<f64>::default();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = b.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_binarizer_f32() {
        let b = Binarizer::<f32>::new(0.0f32);
        let x: Array2<f32> = array![[1.0f32, -1.0, 0.0]];
        let out = b.transform(&x).unwrap();
        assert!((out[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.0f32).abs() < 1e-6);
        assert!((out[[0, 2]] - 0.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_output_values_are_zero_or_one() {
        let b = Binarizer::<f64>::new(0.0);
        let x = array![[-5.0, -1.0, 0.0, 0.001, 1.0, 100.0]];
        let out = b.transform(&x).unwrap();
        for v in &out {
            assert!(*v == 0.0 || *v == 1.0, "expected 0 or 1, got {v}");
        }
    }
}
