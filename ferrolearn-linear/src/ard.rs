//! Automatic Relevance Determination (ARD) Regression.
//!
//! This module provides [`ARDRegression`], a Bayesian linear regression model
//! with per-feature weight precision priors. Features whose precision
//! (`lambda_i`) exceeds a threshold are pruned — their weights are driven to
//! zero, achieving automatic feature selection.
//!
//! # Algorithm
//!
//! Starting from initial alpha (noise precision) and per-feature lambda_i
//! (weight precision) values, the model iterates:
//!
//! 1. Solve the regularised posterior: `w = (alpha * X^T X + diag(lambda))^{-1} alpha X^T y`.
//! 2. Update gamma_i (effective degrees of freedom): `gamma_i = 1 - lambda_i * Sigma_{ii}`.
//! 3. Update alpha: `alpha = (n - sum(gamma)) / ||y - Xw||^2`.
//! 4. Update lambda_i: `lambda_i = gamma_i / w_i^2`.
//!
//! Features where `lambda_i > threshold_lambda` are pruned.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ard::ARDRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = ARDRegression::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Automatic Relevance Determination Regression.
///
/// Bayesian linear regression with per-feature precision priors. Features
/// with high precision (small variance) are pruned, achieving sparsity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ARDRegression<F> {
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in alpha/lambda.
    pub tol: F,
    /// Shape hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_1: F,
    /// Rate hyperparameter for the alpha (noise precision) Gamma prior.
    pub alpha_2: F,
    /// Shape hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_1: F,
    /// Rate hyperparameter for the lambda (weight precision) Gamma prior.
    pub lambda_2: F,
    /// Features with `lambda_i > threshold_lambda` are pruned.
    pub threshold_lambda: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> ARDRegression<F> {
    /// Create a new `ARDRegression` with default settings.
    ///
    /// Defaults: `max_iter = 300`, `tol = 1e-3`, `alpha_1 = alpha_2 = 1e-6`,
    /// `lambda_1 = lambda_2 = 1e-6`, `threshold_lambda = 1e4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-3).unwrap(),
            alpha_1: F::from(1e-6).unwrap(),
            alpha_2: F::from(1e-6).unwrap(),
            lambda_1: F::from(1e-6).unwrap(),
            lambda_2: F::from(1e-6).unwrap(),
            threshold_lambda: F::from(1e4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the alpha shape hyperparameter.
    #[must_use]
    pub fn with_alpha_1(mut self, alpha_1: F) -> Self {
        self.alpha_1 = alpha_1;
        self
    }

    /// Set the alpha rate hyperparameter.
    #[must_use]
    pub fn with_alpha_2(mut self, alpha_2: F) -> Self {
        self.alpha_2 = alpha_2;
        self
    }

    /// Set the lambda shape hyperparameter.
    #[must_use]
    pub fn with_lambda_1(mut self, lambda_1: F) -> Self {
        self.lambda_1 = lambda_1;
        self
    }

    /// Set the lambda rate hyperparameter.
    #[must_use]
    pub fn with_lambda_2(mut self, lambda_2: F) -> Self {
        self.lambda_2 = lambda_2;
        self
    }

    /// Set the pruning threshold for feature lambda values.
    #[must_use]
    pub fn with_threshold_lambda(mut self, threshold_lambda: F) -> Self {
        self.threshold_lambda = threshold_lambda;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for ARDRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ARD Regression model.
///
/// Stores the posterior mean coefficients, intercept, estimated noise
/// precision (`alpha`), per-feature weight precisions (`lambda`), and
/// the diagonal of the posterior covariance.
#[derive(Debug, Clone)]
pub struct FittedARDRegression<F> {
    /// Posterior mean coefficient vector.
    coefficients: Array1<F>,
    /// Intercept (bias) term.
    intercept: F,
    /// Estimated noise precision (1 / noise_variance).
    alpha: F,
    /// Per-feature weight precisions.
    lambda: Array1<F>,
    /// Diagonal of the posterior covariance matrix.
    sigma: Array1<F>,
}

impl<F: Float> FittedARDRegression<F> {
    /// Returns the estimated noise precision (alpha = 1/sigma^2_noise).
    #[must_use]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the per-feature weight precisions.
    #[must_use]
    pub fn lambda(&self) -> &Array1<F> {
        &self.lambda
    }

    /// Returns the diagonal of the posterior covariance matrix.
    #[must_use]
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }
}

/// Solve the ARD system: `(alpha * X^T X + diag(lambda)) w = alpha * X^T y`.
///
/// Returns `(w, diag(Sigma))`.
fn ard_solve<F: Float + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    alpha: F,
    lambda: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_features = x.ncols();
    let xt = x.t();
    let mut xtx = xt.dot(x);

    // Scale by alpha, then add diag(lambda).
    for i in 0..n_features {
        for j in 0..n_features {
            xtx[[i, j]] = xtx[[i, j]] * alpha;
        }
        xtx[[i, i]] = xtx[[i, i]] + lambda[i];
    }

    let xty = xt.dot(y);
    let xty_scaled: Array1<F> = xty.mapv(|v| v * alpha);

    // Cholesky solve.
    let n = n_features;
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = xtx[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "ARD: matrix not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    // Forward substitution.
    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = xty_scaled[i];
        for j in 0..i {
            s = s - l[[i, j]] * z[j];
        }
        z[i] = s / l[[i, i]];
    }

    // Back substitution.
    let mut w = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s = s - l[[j, i]] * w[j];
        }
        w[i] = s / l[[i, i]];
    }

    // Compute diagonal of posterior covariance: diag((alpha * X^T X + diag(lambda))^{-1}).
    let mut sigma_diag = Array1::<F>::zeros(n);
    for col in 0..n {
        let mut z_inv = Array1::<F>::zeros(n);
        z_inv[col] = F::one() / l[[col, col]];
        for i in (col + 1)..n {
            let mut s = F::zero();
            for k in col..i {
                s = s + l[[i, k]] * z_inv[k];
            }
            z_inv[i] = -s / l[[i, i]];
        }
        for i in 0..n {
            sigma_diag[i] = sigma_diag[i] + z_inv[i] * z_inv[i];
        }
    }

    Ok((w, sigma_diag))
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for ARDRegression<F>
{
    type Fitted = FittedARDRegression<F>;
    type Error = FerroError;

    /// Fit the ARD model via iterative evidence maximization.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] — numerical failure in solver.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedARDRegression<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "ARDRegression requires at least 2 samples".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Center data for intercept.
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

        let mut alpha = F::one();
        let mut lambda = Array1::<F>::from_elem(n_features, F::one());
        let clamp_max = F::from(1e10).unwrap();
        let clamp_min = F::from(1e-10).unwrap();

        let mut w = Array1::<F>::zeros(n_features);
        let mut sigma_diag = Array1::<F>::ones(n_features);

        for _iter in 0..self.max_iter {
            let alpha_old = alpha;
            let lambda_old = lambda.clone();

            // E-step: compute posterior.
            let (w_new, sd_new) = ard_solve(&x_work, &y_work, alpha, &lambda)?;

            // Compute gamma_i = 1 - lambda_i * Sigma_ii.
            let gamma: Array1<F> = Array1::from_shape_fn(n_features, |i| {
                F::one() - lambda[i] * sd_new[i]
            });

            let gamma_sum: F = gamma.iter().fold(F::zero(), |a, &b| a + b);

            // Update alpha: (n - sum(gamma) + 2*alpha_1) / (||y - Xw||^2 + 2*alpha_2).
            let residual = &y_work - x_work.dot(&w_new);
            let sse = residual.dot(&residual);
            let two = F::from(2.0).unwrap();
            let new_alpha = (n_f - gamma_sum + two * self.alpha_1)
                / (sse + two * self.alpha_2).max(F::from(1e-300).unwrap());

            // Update lambda_i: (gamma_i + 2*lambda_1) / (w_i^2 + 2*lambda_2).
            let mut new_lambda = Array1::<F>::zeros(n_features);
            for i in 0..n_features {
                let wi_sq = w_new[i] * w_new[i];
                new_lambda[i] = (gamma[i] + two * self.lambda_1)
                    / (wi_sq + two * self.lambda_2).max(F::from(1e-300).unwrap());
            }

            // Clamp.
            alpha = new_alpha.min(clamp_max).max(clamp_min);
            for i in 0..n_features {
                new_lambda[i] = new_lambda[i].min(clamp_max).max(clamp_min);
            }
            lambda = new_lambda;

            w = w_new;
            sigma_diag = sd_new;

            // Check convergence.
            let delta_alpha =
                (alpha - alpha_old).abs() / (alpha_old.abs() + F::from(1e-10).unwrap());
            let mut max_delta_lambda = F::zero();
            for i in 0..n_features {
                let delta = (lambda[i] - lambda_old[i]).abs()
                    / (lambda_old[i].abs() + F::from(1e-10).unwrap());
                if delta > max_delta_lambda {
                    max_delta_lambda = delta;
                }
            }

            if delta_alpha < self.tol && max_delta_lambda < self.tol {
                break;
            }
        }

        // Prune features with lambda > threshold.
        for i in 0..n_features {
            if lambda[i] > self.threshold_lambda {
                w[i] = F::zero();
            }
        }

        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedARDRegression {
            coefficients: w,
            intercept,
            alpha,
            lambda,
            sigma: sigma_diag,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedARDRegression<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values using the posterior mean coefficients.
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedARDRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for ARDRegression<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedARDRegression<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_default_constructor() {
        let m = ARDRegression::<f64>::new();
        assert_eq!(m.max_iter, 300);
        assert!(m.fit_intercept);
        assert_relative_eq!(m.alpha_1, 1e-6);
    }

    #[test]
    fn test_builder_setters() {
        let m = ARDRegression::<f64>::new()
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_alpha_1(1e-3)
            .with_alpha_2(1e-3)
            .with_lambda_1(1e-3)
            .with_lambda_2(1e-3)
            .with_threshold_lambda(1e5)
            .with_fit_intercept(false);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
        assert_relative_eq!(m.threshold_lambda, 1e5);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];
        let result = ARDRegression::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        // Should recover roughly y = 2x + 1.
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1.5);
    }

    #[test]
    fn test_alpha_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert!(fitted.alpha() > 0.0);
    }

    #[test]
    fn test_lambda_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.lambda().iter() {
            assert!(v > 0.0, "lambda must be positive, got {v}");
        }
    }

    #[test]
    fn test_sigma_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        for &v in fitted.sigma().iter() {
            assert!(v > 0.0, "sigma diagonal must be positive, got {v}");
        }
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = ARDRegression::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparsity_on_irrelevant_features() {
        // y depends only on x1, x2 is noise-free irrelevant.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]; // y = 2 * x1

        let fitted = ARDRegression::<f64>::new()
            .with_max_iter(1000)
            .fit(&x, &y)
            .unwrap();

        // The model should learn that x1 is relevant.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 3.0, 0.0, 1.5, 4.0, 1.0, 2.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = ARDRegression::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
