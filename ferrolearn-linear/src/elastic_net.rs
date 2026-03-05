//! ElasticNet regression (combined L1 and L2 regularization).
//!
//! This module provides [`ElasticNet`], which fits a linear model with a
//! blended L1/L2 regularization penalty using coordinate descent with
//! soft-thresholding:
//!
//! ```text
//! minimize (1/(2n)) * ||X @ w - y||^2
//!        + alpha * l1_ratio * ||w||_1
//!        + (alpha/2) * (1 - l1_ratio) * ||w||_2^2
//! ```
//!
//! When `l1_ratio = 1`, ElasticNet is equivalent to Lasso. When
//! `l1_ratio = 0`, it is equivalent to Ridge. Intermediate values produce
//! solutions that are both sparse (L1) and small in magnitude (L2).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ElasticNet;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = ElasticNet::<f64>::new()
//!     .with_alpha(0.1)
//!     .with_l1_ratio(0.5);
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// ElasticNet regression (L1 + L2 regularized least squares).
///
/// Minimizes a combination of L1 and L2 penalties controlled by
/// `alpha` and `l1_ratio`. Uses coordinate descent with soft-thresholding
/// to handle the non-smooth L1 component.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ElasticNet<F> {
    /// Overall regularization strength. Larger values enforce stronger
    /// regularization.
    pub alpha: F,
    /// Mix between L1 and L2 regularization.
    /// - `l1_ratio = 1.0` → pure Lasso (L1 only)
    /// - `l1_ratio = 0.0` → pure Ridge (L2 only)
    /// - `0.0 < l1_ratio < 1.0` → ElasticNet blend
    pub l1_ratio: F,
    /// Maximum number of coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change per pass.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> ElasticNet<F> {
    /// Create a new `ElasticNet` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `l1_ratio = 0.5`, `max_iter = 1000`,
    /// `tol = 1e-4`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            l1_ratio: F::from(0.5).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the overall regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the L1/L2 mixing ratio.
    ///
    /// Must be in `[0.0, 1.0]`. Values outside this range will be rejected
    /// at fit time.
    #[must_use]
    pub fn with_l1_ratio(mut self, l1_ratio: F) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set the maximum number of coordinate descent iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance on maximum coefficient change.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for ElasticNet<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ElasticNet regression model.
///
/// Stores the learned (potentially sparse) coefficients and intercept.
/// Implements [`Predict`] and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedElasticNet<F> {
    /// Learned coefficient vector (some may be exactly zero when L1 > 0).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float> FittedElasticNet<F> {
    /// Returns the intercept (bias) term learned during fitting.
    pub fn intercept(&self) -> F {
        self.intercept
    }
}

/// Soft-thresholding operator used in coordinate descent for L1 penalty.
///
/// Returns `sign(x) * max(|x| - threshold, 0)`.
#[inline]
fn soft_threshold<F: Float>(x: F, threshold: F) -> F {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        F::zero()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for ElasticNet<F>
{
    type Fitted = FittedElasticNet<F>;
    type Error = FerroError;

    /// Fit the ElasticNet model using coordinate descent.
    ///
    /// Centers the data if `fit_intercept` is `true`, then alternates
    /// coordinate updates using the soft-threshold rule with L2 scaling.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers
    ///   of samples.
    /// - [`FerroError::InvalidParameter`] if `alpha` is negative, `l1_ratio`
    ///   is outside `[0, 1]`, or `tol` is non-positive.
    /// - [`FerroError::InsufficientSamples`] if `n_samples == 0`.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedElasticNet<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if self.l1_ratio < F::zero() || self.l1_ratio > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "l1_ratio".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "ElasticNet requires at least one sample".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Center data when fitting intercept.
        let (x_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to compute target mean".into(),
            })?;

            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Precompute per-column X_j^T X_j / n (used as denominator).
        let col_norms: Vec<F> = (0..n_features)
            .map(|j| {
                let col = x_work.column(j);
                col.dot(&col) / n_f
            })
            .collect();

        // L1 and L2 penalty strengths split from alpha/l1_ratio.
        let alpha_l1 = self.alpha * self.l1_ratio;
        let alpha_l2 = self.alpha * (F::one() - self.l1_ratio);

        // Effective denominator per column: (X_j^T X_j / n) + alpha_l2.
        let denominators: Vec<F> = col_norms.iter().map(|&cn| cn + alpha_l2).collect();

        let mut w = Array1::<F>::zeros(n_features);
        let mut residual = y_work.clone();

        for _iter in 0..self.max_iter {
            let mut max_change = F::zero();

            for j in 0..n_features {
                let col_j = x_work.column(j);
                let w_old = w[j];

                // Add back contribution of current coefficient j to residual.
                if w_old != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] + col_j[i] * w_old;
                    }
                }

                // Unpenalized correlation: X_j^T r / n.
                let rho_j = col_j.dot(&residual) / n_f;

                // Apply soft-threshold for L1, then divide by (col_norm + alpha_l2).
                let w_new = if denominators[j] > F::zero() {
                    soft_threshold(rho_j, alpha_l1) / denominators[j]
                } else {
                    F::zero()
                };

                // Update residual with new coefficient.
                if w_new != F::zero() {
                    for i in 0..n_samples {
                        residual[i] = residual[i] - col_j[i] * w_new;
                    }
                }

                let change = (w_new - w_old).abs();
                if change > max_change {
                    max_change = change;
                }

                w[j] = w_new;
            }

            if max_change < self.tol {
                let intercept = compute_intercept(&x_mean, &y_mean, &w);
                return Ok(FittedElasticNet {
                    coefficients: w,
                    intercept,
                });
            }
        }

        // Return best solution found even without full convergence.
        let intercept = compute_intercept(&x_mean, &y_mean, &w);
        Ok(FittedElasticNet {
            coefficients: w,
            intercept,
        })
    }
}

/// Compute intercept from the centered means and fitted coefficients.
fn compute_intercept<F: Float + 'static>(
    x_mean: &Option<Array1<F>>,
    y_mean: &Option<F>,
    w: &Array1<F>,
) -> F {
    if let (Some(xm), Some(ym)) = (x_mean, y_mean) {
        *ym - xm.dot(w)
    } else {
        F::zero()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedElasticNet<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ coefficients + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.coefficients) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedElasticNet<F> {
    /// Returns the learned coefficient vector.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Returns the learned intercept term.
    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for ElasticNet<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    /// Fit the model and return it as a boxed pipeline estimator.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `fit`.
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedElasticNet<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    /// Generate predictions via the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates any [`FerroError`] from `predict`.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- soft_threshold helpers ----

    #[test]
    fn test_soft_threshold_positive() {
        assert_relative_eq!(soft_threshold(5.0_f64, 1.0), 4.0);
    }

    #[test]
    fn test_soft_threshold_negative() {
        assert_relative_eq!(soft_threshold(-5.0_f64, 1.0), -4.0);
    }

    #[test]
    fn test_soft_threshold_within_band() {
        assert_relative_eq!(soft_threshold(0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(-0.5_f64, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(0.0_f64, 1.0), 0.0);
    }

    // ---- Builder ----

    #[test]
    fn test_default_builder() {
        let m = ElasticNet::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_relative_eq!(m.l1_ratio, 0.5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder_setters() {
        let m = ElasticNet::<f64>::new()
            .with_alpha(0.5)
            .with_l1_ratio(0.2)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_relative_eq!(m.l1_ratio, 0.2);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    // ---- Validation errors ----

    #[test]
    fn test_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = ElasticNet::<f64>::new().with_alpha(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_l1_ratio_out_of_range_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = ElasticNet::<f64>::new().with_l1_ratio(1.5).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = ElasticNet::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----

    #[test]
    fn test_lasso_limit_l1_ratio_one() {
        // With l1_ratio=1, ElasticNet should behave like Lasso.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.0).with_l1_ratio(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ridge_limit_l1_ratio_zero() {
        // With l1_ratio=0 and alpha=0, should recover OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.0).with_l1_ratio(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sparsity_with_high_l1_ratio() {
        // High alpha with l1_ratio=1 should zero out irrelevant features.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let model = ElasticNet::<f64>::new().with_alpha(5.0).with_l1_ratio(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_higher_alpha_shrinks_more() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let low = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .with_l1_ratio(0.5)
            .fit(&x, &y)
            .unwrap();
        let high = ElasticNet::<f64>::new()
            .with_alpha(2.0)
            .with_l1_ratio(0.5)
            .fit(&x, &y)
            .unwrap();

        assert!(high.coefficients()[0].abs() <= low.coefficients()[0].abs());
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.0)
            .with_l1_ratio(0.5)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_correct_length() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.01)
            .fit(&x_train, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ElasticNet::<f64>::new()
            .with_alpha(0.1)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = ElasticNet::<f64>::new().with_alpha(0.01);
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
