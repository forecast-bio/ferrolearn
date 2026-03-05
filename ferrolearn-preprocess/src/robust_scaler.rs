//! Robust scaler: median and IQR-based scaling.
//!
//! Each feature is transformed as `(x - median) / IQR` where
//! `IQR = Q75 - Q25`. This scaler is robust to outliers.
//!
//! Columns where IQR = 0 are left unchanged after transformation.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Helper: compute quantile of a sorted slice
// ---------------------------------------------------------------------------

/// Compute the `q`-th quantile (0.0–1.0) of a sorted slice using linear interpolation.
///
/// Panics if `sorted` is empty.
fn quantile_sorted<F: Float>(sorted: &[F], q: f64) -> F {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = F::from(idx - lo as f64).unwrap_or(F::zero());
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

// ---------------------------------------------------------------------------
// RobustScaler (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted robust scaler.
///
/// Calling [`Fit::fit`] learns the per-column medians and interquartile ranges
/// (IQR = Q75 − Q25) and returns a [`FittedRobustScaler`] that can transform
/// new data.
///
/// Columns with IQR = 0 are left unchanged after transformation.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::RobustScaler;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let scaler = RobustScaler::<f64>::new();
/// let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]];
/// let fitted = scaler.fit(&x, &()).unwrap();
/// let scaled = fitted.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustScaler<F> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> RobustScaler<F> {
    /// Create a new `RobustScaler`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for RobustScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedRobustScaler
// ---------------------------------------------------------------------------

/// A fitted robust scaler holding per-column medians and IQRs.
///
/// Created by calling [`Fit::fit`] on a [`RobustScaler`].
#[derive(Debug, Clone)]
pub struct FittedRobustScaler<F> {
    /// Per-column medians learned during fitting.
    pub(crate) median: Array1<F>,
    /// Per-column interquartile ranges (Q75 − Q25) learned during fitting.
    pub(crate) iqr: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedRobustScaler<F> {
    /// Return the per-column medians learned during fitting.
    #[must_use]
    pub fn median(&self) -> &Array1<F> {
        &self.median
    }

    /// Return the per-column IQR values learned during fitting.
    #[must_use]
    pub fn iqr(&self) -> &Array1<F> {
        &self.iqr
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for RobustScaler<F> {
    type Fitted = FittedRobustScaler<F>;
    type Error = FerroError;

    /// Fit the scaler by computing per-column medians and IQRs.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedRobustScaler<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RobustScaler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut median_arr = Array1::zeros(n_features);
        let mut iqr_arr = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut col: Vec<F> = x.column(j).iter().copied().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let med = quantile_sorted(&col, 0.5);
            let q25 = quantile_sorted(&col, 0.25);
            let q75 = quantile_sorted(&col, 0.75);

            median_arr[j] = med;
            iqr_arr[j] = q75 - q25;
        }

        Ok(FittedRobustScaler {
            median: median_arr,
            iqr: iqr_arr,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedRobustScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by subtracting the median and dividing by the IQR.
    ///
    /// Columns with IQR = 0 are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.median.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedRobustScaler::transform".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let med = self.median[j];
            let iqr = self.iqr[j];
            if iqr == F::zero() {
                // Zero-IQR column: leave unchanged.
                continue;
            }
            for v in col.iter_mut() {
                *v = (*v - med) / iqr;
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted scaler to satisfy the `FitTransform: Transform`
/// supertrait bound. Calling `transform` on an unfitted scaler always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for RobustScaler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the scaler must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedRobustScaler`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "RobustScaler".into(),
            reason: "scaler must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for RobustScaler<F> {
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

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for RobustScaler<F> {
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedRobustScaler<F> {
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
    fn test_robust_scaler_basic() {
        let scaler = RobustScaler::<f64>::new();
        // Symmetric distribution: median = 3, Q25 = 2, Q75 = 4, IQR = 2
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.median()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.iqr()[0], 2.0, epsilon = 1e-10);

        let scaled = fitted.transform(&x).unwrap();
        // Median should be 0 after scaling
        assert_abs_diff_eq!(scaled[[2, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_iqr_column_unchanged() {
        let scaler = RobustScaler::<f64>::new();
        // Column 0 is constant: IQR = 0
        let x = array![[7.0, 1.0], [7.0, 2.0], [7.0, 3.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.iqr()[0], 0.0, epsilon = 1e-15);
        let scaled = fitted.transform(&x).unwrap();
        // Constant column should remain 7.0
        for i in 0..3 {
            assert_abs_diff_eq!(scaled[[i, 0]], 7.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_outlier_robustness() {
        let scaler = RobustScaler::<f64>::new();
        // Add a large outlier; median should not shift much
        let x = array![[1.0], [2.0], [3.0], [4.0], [1000.0]];
        let fitted = scaler.fit(&x, &()).unwrap();
        // Median of sorted [1,2,3,4,1000] = 3.0
        assert_abs_diff_eq!(fitted.median()[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let scaler = RobustScaler::<f64>::new();
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
        let scaler = RobustScaler::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = scaler.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let scaler = RobustScaler::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(scaler.fit(&x, &()).is_err());
    }
}
