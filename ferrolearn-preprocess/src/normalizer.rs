//! Normalizer: scale each sample (row) to unit norm.
//!
//! Unlike column-wise scalers, the `Normalizer` operates row-wise: each
//! sample is scaled independently so that its chosen norm equals 1.
//!
//! Supported norms:
//! - **L1**: divide by the sum of absolute values
//! - **L2**: divide by the Euclidean norm (default)
//! - **Max**: divide by the maximum absolute value
//!
//! Samples that already have a zero norm are left unchanged.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::Transform;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// NormType
// ---------------------------------------------------------------------------

/// The norm used by [`Normalizer`] when scaling each sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    /// L1 norm: sum of absolute values.
    L1,
    /// L2 norm: Euclidean norm (square root of sum of squares). This is the default.
    #[default]
    L2,
    /// Max norm: maximum absolute value in the sample.
    Max,
}

// ---------------------------------------------------------------------------
// Normalizer
// ---------------------------------------------------------------------------

/// A stateless row-wise normalizer.
///
/// Each sample (row) is independently scaled so that its chosen norm equals 1.
/// Samples with a zero norm are left unchanged.
///
/// This transformer is stateless — no [`Fit`](ferrolearn_core::traits::Fit)
/// step is needed. Call [`Transform::transform`] directly.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::normalizer::{Normalizer, NormType};
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let normalizer = Normalizer::<f64>::new(NormType::L2);
/// let x = array![[3.0, 4.0], [1.0, 0.0]];
/// let out = normalizer.transform(&x).unwrap();
/// // Row 0: [3/5, 4/5], Row 1: [1.0, 0.0]
/// ```
#[derive(Debug, Clone)]
pub struct Normalizer<F> {
    /// The norm to use for normalisation.
    pub(crate) norm: NormType,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> Normalizer<F> {
    /// Create a new `Normalizer` with the specified norm type.
    #[must_use]
    pub fn new(norm: NormType) -> Self {
        Self {
            norm,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new `Normalizer` using the default L2 norm.
    #[must_use]
    pub fn l2() -> Self {
        Self::new(NormType::L2)
    }

    /// Create a new `Normalizer` using the L1 norm.
    #[must_use]
    pub fn l1() -> Self {
        Self::new(NormType::L1)
    }

    /// Create a new `Normalizer` using the Max norm.
    #[must_use]
    pub fn max() -> Self {
        Self::new(NormType::Max)
    }

    /// Return the configured norm type.
    #[must_use]
    pub fn norm(&self) -> NormType {
        self.norm
    }
}

impl<F: Float + Send + Sync + 'static> Default for Normalizer<F> {
    fn default() -> Self {
        Self::new(NormType::L2)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for Normalizer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Normalize each row of `x` to unit norm.
    ///
    /// Rows with a zero norm value are left unchanged.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error for well-formed inputs, but
    /// returns `Ok(...)` to satisfy the trait contract.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let mut out = x.to_owned();
        for mut row in out.rows_mut() {
            let norm_val =
                match self.norm {
                    NormType::L1 => row.iter().copied().fold(F::zero(), |acc, v| acc + v.abs()),
                    NormType::L2 => row
                        .iter()
                        .copied()
                        .fold(F::zero(), |acc, v| acc + v * v)
                        .sqrt(),
                    NormType::Max => row.iter().copied().fold(F::zero(), |acc, v| {
                        if v.abs() > acc { v.abs() } else { acc }
                    }),
                };
            if norm_val == F::zero() {
                // Zero-norm row: leave unchanged.
                continue;
            }
            for v in row.iter_mut() {
                *v = *v / norm_val;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for Normalizer<F> {
    /// Fit the normalizer using the pipeline interface.
    ///
    /// Because `Normalizer` is stateless, this simply boxes `self` as a
    /// [`FittedPipelineTransformer`].
    ///
    /// # Errors
    ///
    /// This implementation never returns an error.
    fn fit_pipeline(
        &self,
        _x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        Ok(Box::new(self.clone()))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for Normalizer<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
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
    fn test_l2_norm_basic() {
        let norm = Normalizer::<f64>::l2();
        // Row [3, 4] has L2 norm 5.
        let x = array![[3.0, 4.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_unit_norm_after_transform() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = norm.transform(&x).unwrap();
        for row in out.rows() {
            let row_norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l1_norm_basic() {
        let norm = Normalizer::<f64>::l1();
        // Row [1, 2, 3] has L1 norm 6.
        let x = array![[1.0, 2.0, 3.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 3.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_unit_norm_after_transform() {
        let norm = Normalizer::<f64>::l1();
        let x = array![[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]];
        let out = norm.transform(&x).unwrap();
        for row in out.rows() {
            let row_norm: f64 = row.iter().map(|v| v.abs()).sum();
            assert_abs_diff_eq!(row_norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_max_norm_basic() {
        let norm = Normalizer::<f64>::max();
        // Row [-5, 3, 1] has max norm 5.
        let x = array![[-5.0, 3.0, 1.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_row_unchanged() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let out = norm.transform(&x).unwrap();
        // Zero row stays zero
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 2]], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_negative_values_l2() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[-3.0, -4.0]];
        let out = norm.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], -0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_default_is_l2() {
        let norm = Normalizer::<f64>::default();
        assert_eq!(norm.norm(), NormType::L2);
    }

    #[test]
    fn test_multiple_rows_independent() {
        let norm = Normalizer::<f64>::l2();
        let x = array![[3.0, 4.0], [0.0, 5.0]];
        let out = norm.transform(&x).unwrap();
        // Row 0: L2 norm = 5
        assert_abs_diff_eq!(out[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.8, epsilon = 1e-10);
        // Row 1: L2 norm = 5
        assert_abs_diff_eq!(out[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let norm = Normalizer::<f64>::l2();
        let x = array![[3.0, 4.0], [0.0, 2.0]];
        let y = Array1::zeros(2);
        let fitted = norm.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_abs_diff_eq!(result[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_normalizer() {
        let norm = Normalizer::<f32>::l2();
        let x: Array2<f32> = array![[3.0f32, 4.0]];
        let out = norm.transform(&x).unwrap();
        assert!((out[[0, 0]] - 0.6f32).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.8f32).abs() < 1e-6);
    }
}
