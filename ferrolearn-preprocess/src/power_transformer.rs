//! Power transformer: apply a power transform to make data more Gaussian.
//!
//! Implements the **Yeo-Johnson** transformation, which works for both positive
//! and negative values. An optimal lambda per feature is estimated via a simple
//! grid search that maximises the log-likelihood of the transformed column
//! following a normal distribution.
//!
//! After transformation, the data can optionally be standardized (zero mean,
//! unit variance). Standardization is enabled by default, matching the
//! scikit-learn default.
//!
//! # Yeo-Johnson definition
//!
//! ```text
//! y ≥ 0, λ ≠ 0:  ((y + 1)^λ - 1) / λ
//! y ≥ 0, λ = 0:  ln(y + 1)
//! y < 0, λ ≠ 2:  -((1 - y)^(2-λ) - 1) / (2 - λ)
//! y < 0, λ = 2:  -ln(1 - y)
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply the Yeo-Johnson transform to a single value with parameter `lambda`.
fn yeo_johnson<F: Float>(y: F, lambda: F) -> F {
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    let eps = F::from(1e-10_f64).unwrap_or(F::epsilon());

    if y >= zero {
        if (lambda - zero).abs() < eps {
            // λ ≈ 0: ln(y + 1)
            (y + one).ln()
        } else {
            // ((y + 1)^λ - 1) / λ
            ((y + one).powf(lambda) - one) / lambda
        }
    } else {
        // y < 0
        let two_minus_lambda = two - lambda;
        if (two_minus_lambda).abs() < eps {
            // λ ≈ 2: -ln(1 - y)
            -(one - y).ln()
        } else {
            // -((1 - y)^(2-λ) - 1) / (2 - λ)
            -((one - y).powf(two_minus_lambda) - one) / two_minus_lambda
        }
    }
}

/// Compute the log-likelihood of a zero-mean, unit-variance normal distribution
/// for the transformed data. This is used as the optimisation criterion for
/// finding the optimal lambda.
///
/// For a column `col` transformed with `lambda`, the log-likelihood contribution
/// from the Yeo-Johnson Jacobian is:
/// `(λ - 1) * sum(sign(y) * ln(|y| + 1))` for each sample.
/// We then add the normal log-likelihood of the transformed values.
fn log_likelihood_yj<F: Float>(col: &[F], lambda: F) -> F {
    let n = F::from(col.len()).unwrap_or(F::one());
    let one = F::one();
    let two = one + one;
    let pi2 = F::from(std::f64::consts::TAU).unwrap_or(F::one()); // 2π

    // Transform each value
    let transformed: Vec<F> = col
        .iter()
        .copied()
        .map(|v| yeo_johnson(v, lambda))
        .collect();

    // Compute mean and variance of transformed values
    let mean = transformed
        .iter()
        .copied()
        .fold(F::zero(), |acc, v| acc + v)
        / n;
    let variance = transformed
        .iter()
        .copied()
        .map(|v| (v - mean) * (v - mean))
        .fold(F::zero(), |acc, v| acc + v)
        / n;

    if variance <= F::zero() {
        return F::neg_infinity();
    }

    // Normal log-likelihood: -n/2 * ln(2π) - n/2 * ln(var) - 1/(2*var)*sum((t-mean)^2)
    // Simplified: -n/2 * ln(2π*var) - n/2
    let normal_ll = -n / two * (pi2 * variance).ln() - n / two;

    // Jacobian contribution from Yeo-Johnson
    // For both y >= 0 and y < 0, the Jacobian term is ln(|y| + 1).
    let lambda_minus_1 = lambda - one;
    let jacobian: F = col
        .iter()
        .copied()
        .fold(F::zero(), |acc, y| acc + (y.abs() + one).ln());
    let jacobian_ll = lambda_minus_1 * jacobian;

    normal_ll + jacobian_ll
}

// ---------------------------------------------------------------------------
// PowerTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted power transformer using the Yeo-Johnson method.
///
/// Calling [`Fit::fit`] estimates an optimal lambda per feature (via grid
/// search over a range of lambda values) and returns a [`FittedPowerTransformer`]
/// that can transform new data.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::PowerTransformer;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let pt = PowerTransformer::<f64>::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = pt.fit(&x, &()).unwrap();
/// let transformed = fitted.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PowerTransformer<F> {
    /// Whether to standardize the output (zero mean, unit variance).
    pub(crate) standardize: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PowerTransformer<F> {
    /// Create a new `PowerTransformer` with standardization enabled (default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            standardize: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new `PowerTransformer` with standardization disabled.
    #[must_use]
    pub fn without_standardize() -> Self {
        Self {
            standardize: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Whether standardization is enabled.
    #[must_use]
    pub fn standardize(&self) -> bool {
        self.standardize
    }
}

impl<F: Float + Send + Sync + 'static> Default for PowerTransformer<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedPowerTransformer
// ---------------------------------------------------------------------------

/// A fitted power transformer holding per-column lambda values and optional
/// standardisation parameters.
///
/// Created by calling [`Fit::fit`] on a [`PowerTransformer`].
#[derive(Debug, Clone)]
pub struct FittedPowerTransformer<F> {
    /// Per-column optimal lambda values.
    pub(crate) lambdas: Array1<F>,
    /// Per-column means of the transformed data (used for standardization).
    pub(crate) means: Option<Array1<F>>,
    /// Per-column standard deviations of the transformed data (used for standardization).
    pub(crate) stds: Option<Array1<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedPowerTransformer<F> {
    /// Return the per-column lambda values learned during fitting.
    #[must_use]
    pub fn lambdas(&self) -> &Array1<F> {
        &self.lambdas
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PowerTransformer<F> {
    type Fitted = FittedPowerTransformer<F>;
    type Error = FerroError;

    /// Fit the transformer by estimating the optimal lambda per feature.
    ///
    /// Uses a grid search over lambda values in `[-3, 3]` with 201 candidate
    /// values, selecting the lambda that maximises the log-likelihood of the
    /// Yeo-Johnson transformed column following a normal distribution.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPowerTransformer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "PowerTransformer::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut lambdas = Array1::zeros(n_features);

        // Grid search: 201 candidate lambdas in [-3, 3]
        let n_candidates = 201_usize;
        let lambda_min = F::from(-3.0_f64).unwrap_or(F::zero());
        let lambda_max = F::from(3.0_f64).unwrap_or(F::one());
        let step = (lambda_max - lambda_min) / F::from(n_candidates - 1).unwrap_or(F::one());

        for j in 0..n_features {
            let col: Vec<F> = x.column(j).iter().copied().collect();

            let mut best_ll = F::neg_infinity();
            let mut best_lambda = F::one(); // default lambda

            for k in 0..n_candidates {
                let lambda = lambda_min + step * F::from(k).unwrap_or(F::zero());
                let ll = log_likelihood_yj(&col, lambda);
                if ll > best_ll {
                    best_ll = ll;
                    best_lambda = lambda;
                }
            }

            lambdas[j] = best_lambda;
        }

        // If standardize, compute mean and std of transformed data
        let (means, stds) = if self.standardize {
            let n = F::from(n_samples).unwrap_or(F::one());
            let mut means_arr = Array1::zeros(n_features);
            let mut stds_arr = Array1::zeros(n_features);
            for j in 0..n_features {
                let lambda = lambdas[j];
                let transformed: Vec<F> = x
                    .column(j)
                    .iter()
                    .copied()
                    .map(|v| yeo_johnson(v, lambda))
                    .collect();
                let mean = transformed
                    .iter()
                    .copied()
                    .fold(F::zero(), |acc, v| acc + v)
                    / n;
                let variance = transformed
                    .iter()
                    .copied()
                    .map(|v| (v - mean) * (v - mean))
                    .fold(F::zero(), |acc, v| acc + v)
                    / n;
                means_arr[j] = mean;
                stds_arr[j] = variance.sqrt();
            }
            (Some(means_arr), Some(stds_arr))
        } else {
            (None, None)
        };

        Ok(FittedPowerTransformer {
            lambdas,
            means,
            stds,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPowerTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the Yeo-Johnson transform and optionally standardize.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.lambdas.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPowerTransformer::transform".into(),
            });
        }

        let mut out = x.to_owned();
        for (j, mut col) in out.columns_mut().into_iter().enumerate() {
            let lambda = self.lambdas[j];
            for v in col.iter_mut() {
                *v = yeo_johnson(*v, lambda);
            }

            // Standardize if requested
            if let (Some(means), Some(stds)) = (&self.means, &self.stds) {
                let m = means[j];
                let s = stds[j];
                if s > F::zero() {
                    for v in col.iter_mut() {
                        *v = (*v - m) / s;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted transformer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for PowerTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the transformer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "PowerTransformer".into(),
            reason: "transformer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for PowerTransformer<F> {
    type FitError = FerroError;

    /// Fit the transformer on `x` and return the transformed output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PowerTransformer<F> {
    /// Fit the transformer using the pipeline interface.
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedPowerTransformer<F> {
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
    fn test_yeo_johnson_identity_at_lambda_one() {
        // At λ=1: y≥0 -> ((y+1)^1 - 1)/1 = y.  So identity for non-negative.
        let one = 1.0_f64;
        for v in [0.0, 0.5, 1.0, 2.0, 5.0] {
            let out = yeo_johnson(v, one);
            assert_abs_diff_eq!(out, v, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_log_at_lambda_zero() {
        // At λ=0, y≥0: ln(y+1)
        let zero = 0.0_f64;
        for v in [0.0, 0.5, 1.0, 2.0] {
            let expected = (v + 1.0).ln();
            assert_abs_diff_eq!(yeo_johnson(v, zero), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_yeo_johnson_negative_at_lambda_two() {
        // At λ=2, y<0: -ln(1-y)
        let two = 2.0_f64;
        for v in [-0.5, -1.0, -2.0] {
            let expected = -(1.0 - v).ln();
            assert_abs_diff_eq!(yeo_johnson(v, two), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_power_transformer_fit_basic() {
        let pt = PowerTransformer::<f64>::new();
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = pt.fit(&x, &()).unwrap();
        // Lambda should be within [-3, 3]
        let lambda = fitted.lambdas()[0];
        assert!(lambda >= -3.0 && lambda <= 3.0);
    }

    #[test]
    fn test_power_transformer_transform_shape() {
        let pt = PowerTransformer::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_standardize_produces_zero_mean() {
        let pt = PowerTransformer::<f64>::new(); // standardize=true
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = pt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        let mean: f64 = out.column(0).iter().sum::<f64>() / out.nrows() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_without_standardize() {
        let pt = PowerTransformer::<f64>::without_standardize();
        assert!(!pt.standardize());
        let x = array![[1.0], [2.0], [3.0]];
        let fitted = pt.fit(&x, &()).unwrap();
        assert!(fitted.means.is_none());
        assert!(fitted.stds.is_none());
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_fit_transform_equivalence() {
        let pt = PowerTransformer::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let via_ft = pt.fit_transform(&x).unwrap();
        let fitted = pt.fit(&x, &()).unwrap();
        let via_sep = fitted.transform(&x).unwrap();
        for (a, b) in via_ft.iter().zip(via_sep.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let pt = PowerTransformer::<f64>::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = pt.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let pt = PowerTransformer::<f64>::new();
        let x: Array2<f64> = Array2::zeros((0, 2));
        assert!(pt.fit(&x, &()).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let pt = PowerTransformer::<f64>::new();
        let x = array![[1.0, 2.0]];
        assert!(pt.transform(&x).is_err());
    }

    #[test]
    fn test_negative_values_supported() {
        let pt = PowerTransformer::<f64>::without_standardize();
        // Yeo-Johnson supports negative values
        let x = array![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
        let fitted = pt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Should not panic and produce finite values
        for v in out.iter() {
            assert!(v.is_finite(), "got non-finite value: {v}");
        }
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let pt = PowerTransformer::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = Array1::zeros(3);
        let fitted = pt.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_eq!(result.shape(), x.shape());
    }
}
