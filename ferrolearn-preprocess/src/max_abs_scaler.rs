//! Max-absolute scaler: scale each feature by its maximum absolute value.
//!
//! Each feature is transformed as `x_scaled = x / max(|x|)` so that values
//! fall within `[-1, 1]`. This scaler does not shift the data (no centering),
//! making it suitable for sparse data.
//!
//! Columns where `max_abs = 0` (all-zero features) are left unchanged.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// MaxAbsScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted max-absolute scaler.
///
/// Calling [`Fit::fit`] learns the per-column maximum absolute values and
/// returns a [`FittedMaxAbsScaler`] that can transform new data.
///
/// Columns where the maximum absolute value is zero are left unchanged after
/// transformation.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::MaxAbsScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = MaxAbsScaler::<f64>::new();
/// let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// // All values now in [-1, 1]
/// ```
#[derive(Debug, Clone)]
pub struct MaxAbsScaler<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> MaxAbsScaler<F> {
    /// Create a new `MaxAbsScaler`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for MaxAbsScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedMaxAbsScaler
// ---------------------------------------------------------------------------

/// A fitted max-absolute scaler holding per-column maximum absolute values.
///
/// Created by calling [`Fit::fit`] on a [`MaxAbsScaler`].
#[derive(Debug, Clone)]
pub struct FittedMaxAbsScaler<F> {
    /// Per-column maximum absolute values learned during fitting.
    pub(crate) max_abs: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedMaxAbsScaler<F> {
    /// Return the per-column maximum absolute values learned during fitting.
    #[must_use]
    pub fn max_abs(&self) -> &Array1<F> {
        &self.max_abs
    }

    /// Inverse-transform scaled data back to the original space.
    ///
    /// Applies `x_orig = x_scaled * max_abs` per column.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    pub fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.max_abs.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMaxAbsScaler::inverse_transform".into(),
            });
        }
        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let ma = self.max_abs[j];
            if ma == F::zero() {
                continue;
            }
            for v in col.iter_mut() {
                *v = *v * ma;
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MaxAbsScaler<F> {
    type Fitted = FittedMaxAbsScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column maximum absolute values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMaxAbsScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MaxAbsScaler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut max_abs = Array1::zeros(n_features);

        for j in 0..n_features {
            let col_max_abs = x
                .column(j)
                .iter()
                .copied()
                .map(|v| v.abs())
                .fold(F::zero(), |acc, v| if v > acc { v } else { acc });
            max_abs[j] = col_max_abs;
        }

        Ok(FittedMaxAbsScaler { max_abs })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMaxAbsScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by dividing each feature by its maximum absolute value.
    ///
    /// Columns where `max_abs = 0` are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.max_abs.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMaxAbsScaler::transform".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let ma = self.max_abs[j];
            if ma == F::zero() {
                // All-zero column: leave unchanged.
                continue;
            }
            for v in col.iter_mut() {
                *v = *v / ma;
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for MaxAbsScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedMaxAbsScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "MaxAbsScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for MaxAbsScaler<F> {
    type FitError = FerroError;

    /// Fit the scaler on `x` and return the scaled output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g., zero rows).
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for MaxAbsScaler<F> {
    /// Fit the scaler using the pipeline interface.
    ///
    /// The `y` argument is ignored; it exists only for API compatibility.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedMaxAbsScaler<F> {
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
    fn test_max_abs_scaler_basic() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        // col0: max_abs = 3.0, col1: max_abs = 4.0
        assert_abs_diff_eq!(fitted.max_abs()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.max_abs()[1], 4.0, epsilon = 1e-10);

        let scaled = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_values_in_range() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-10.0, 5.0], [3.0, -8.0], [7.0, 2.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        for v in scaled.iter() {
            assert!(
                *v >= -1.0 - 1e-10 && *v <= 1.0 + 1e-10,
                "value {v} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn test_zero_column_unchanged() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.max_abs()[0], 0.0, epsilon = 1e-15);
        let scaled = fitted.transform(&x).unwrap();
        // All-zero column stays 0.0
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[-3.0, 1.0], [0.0, -2.0], [2.0, 4.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&scaled).unwrap();
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let via_fit_transform = scaler.fit_transform(&x).unwrap();
        let fitted = scaler.fit(&x, &()).unwrap();
        let via_separate = fitted.transform(&x).unwrap();
        for (a, b) in via_fit_transform.iter().zip(via_separate.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(scaler.fit(&x, &()).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[1.0, 2.0]];
        assert!(scaler.transform(&x).is_err());
    }

    #[test]
    fn test_negative_values() {
        let scaler = MaxAbsScaler::<f64>::new();
        // All negative values
        let x = array![[-5.0], [-3.0], [-1.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.max_abs()[0], 5.0, epsilon = 1e-10);
        let scaled = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(scaled[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[1, 0]], -0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[[2, 0]], -0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let scaler = MaxAbsScaler::<f64>::new();
        let x = array![[2.0, 4.0], [1.0, -2.0]];
        let y = Array1::zeros(2);
        let fitted = scaler.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_scaler() {
        let scaler = MaxAbsScaler::<f32>::new();
        let x: Array2<f32> = array![[2.0f32, -4.0], [1.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        let scaled = fitted.transform(&x).unwrap();
        assert!((scaled[[0, 0]] - 1.0f32).abs() < 1e-6);
        assert!((scaled[[0, 1]] - (-1.0f32)).abs() < 1e-6);
    }
}
