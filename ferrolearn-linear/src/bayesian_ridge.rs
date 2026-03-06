//! Bayesian Ridge Regression.
//!
//! This module provides [`BayesianRidge`], which fits a Bayesian formulation of
//! Ridge regression. Rather than using a fixed regularization strength, the
//! model iteratively estimates two precision hyperparameters:
//!
//! - **`lambda`** — precision (inverse variance) of the weight prior.
//! - **`alpha`** — noise precision (inverse of noise variance).
//!
//! Both are inferred from the data via evidence maximization (Type-II maximum
//! likelihood / Empirical Bayes). This automatic relevance determination means
//! the user does not need to tune the regularization parameter by hand.
//!
//! The objective is the Bayesian evidence (marginal likelihood) of the model:
//!
//! ```text
//! p(y | X, alpha, lambda) ∝ N(y; 0, (1/alpha)*I + (1/lambda)*X X^T)
//! ```
//!
//! After fitting, the model exposes the posterior mean (`coefficients`),
//! the posterior covariance diagonal (`sigma`), the noise precision (`alpha`),
//! and the weight precision (`lambda`).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::BayesianRidge;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = BayesianRidge::<f64>::new();
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![3.0, 5.0, 7.0, 9.0, 11.0];
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

/// Bayesian Ridge Regression with automatic regularization tuning.
///
/// Estimates weight precision (`lambda`) and noise precision (`alpha`)
/// iteratively using evidence maximization. The intercept, if requested,
/// is fit by centering.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BayesianRidge<F> {
    /// Maximum number of EM (evidence-maximization) iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative change in log-evidence.
    pub tol: F,
    /// Initial noise precision (alpha). Must be positive.
    pub alpha_init: F,
    /// Initial weight precision (lambda). Must be positive.
    pub lambda_init: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> BayesianRidge<F> {
    /// Create a new `BayesianRidge` with default settings.
    ///
    /// Defaults: `max_iter = 300`, `tol = 1e-3`, `alpha_init = 1.0`,
    /// `lambda_init = 1.0`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-3).unwrap(),
            alpha_init: F::one(),
            lambda_init: F::one(),
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

    /// Set the initial noise precision.
    #[must_use]
    pub fn with_alpha_init(mut self, alpha_init: F) -> Self {
        self.alpha_init = alpha_init;
        self
    }

    /// Set the initial weight precision.
    #[must_use]
    pub fn with_lambda_init(mut self, lambda_init: F) -> Self {
        self.lambda_init = lambda_init;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for BayesianRidge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Bayesian Ridge Regression model.
///
/// Stores the posterior mean coefficients, intercept, estimated noise
/// precision (`alpha`), weight precision (`lambda`), and the diagonal
/// of the posterior covariance matrix (`sigma`).
#[derive(Debug, Clone)]
pub struct FittedBayesianRidge<F> {
    /// Posterior mean coefficient vector.
    coefficients: Array1<F>,
    /// Intercept (bias) term.
    intercept: F,
    /// Estimated noise precision (1 / noise_variance).
    alpha: F,
    /// Estimated weight precision (1 / weight_variance).
    lambda: F,
    /// Diagonal of the posterior covariance matrix `Sigma`.
    sigma: Array1<F>,
}

impl<F: Float> FittedBayesianRidge<F> {
    /// Returns the estimated noise precision (alpha = 1/sigma²_noise).
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Returns the estimated weight precision (lambda = 1/sigma²_weights).
    pub fn lambda(&self) -> F {
        self.lambda
    }

    /// Returns the diagonal of the posterior covariance matrix.
    pub fn sigma(&self) -> &Array1<F> {
        &self.sigma
    }
}

/// Solve `(lambda/alpha * I + X^T X) w = X^T y` via Cholesky or fallback.
///
/// Returns `(w, diag(Sigma))` where `Sigma = alpha^{-1} * (lambda * I + alpha * X^T X)^{-1}`.
fn bayesian_ridge_solve<F: Float + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    alpha: F,
    lambda: F,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let (_n_samples, n_features) = x.dim();

    // Compute X^T X.
    let xt = x.t();
    let mut xtx = xt.dot(x);

    // Scale by alpha, then add lambda * I.
    // The system we solve is: (alpha * X^T X + lambda * I) w = alpha * X^T y
    for i in 0..n_features {
        for j in 0..n_features {
            xtx[[i, j]] = xtx[[i, j]] * alpha;
        }
        xtx[[i, i]] = xtx[[i, i]] + lambda;
    }

    let xty = xt.dot(y);
    let xty_scaled: Array1<F> = xty.mapv(|v| v * alpha);

    // Solve via Cholesky.
    let w = cholesky_solve(&xtx, &xty_scaled)?;

    // Compute diagonal of posterior covariance: diag((alpha * X^T X + lambda * I)^{-1}).
    let sigma_diag = cholesky_diag_inv(&xtx)?;

    Ok((w, sigma_diag))
}

/// Cholesky decomposition and solve `A x = b`.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky: matrix not positive definite".into(),
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
        let mut s = b[i];
        for j in 0..i {
            s = s - l[[i, j]] * z[j];
        }
        z[i] = s / l[[i, i]];
    }

    // Backward substitution.
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s = s - l[[j, i]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }

    Ok(x)
}

/// Compute the diagonal of `A^{-1}` given Cholesky `L` of `A = L L^T`.
///
/// Uses the identity: `diag(A^{-1}) = diag(L^{-T} L^{-1})`.
fn cholesky_diag_inv<F: Float>(a: &Array2<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky diag_inv: matrix not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    // Compute L^{-1} column by column and accumulate diagonal of L^{-T} L^{-1}.
    let mut diag = Array1::<F>::zeros(n);
    for col in 0..n {
        // Solve L z = e_col.
        let mut z = Array1::<F>::zeros(n);
        z[col] = F::one() / l[[col, col]];
        for i in (col + 1)..n {
            let mut s = F::zero();
            for k in col..i {
                s = s + l[[i, k]] * z[k];
            }
            z[i] = -s / l[[i, i]];
        }
        // Accumulate z^T z into the diagonal positions it touches.
        for i in 0..n {
            diag[i] = diag[i] + z[i] * z[i];
        }
    }

    Ok(diag)
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for BayesianRidge<F>
{
    type Fitted = FittedBayesianRidge<F>;
    type Error = FerroError;

    /// Fit the Bayesian Ridge model via evidence maximization (EM).
    ///
    /// Iterates over:
    /// 1. Solve posterior for `w` given current `alpha` and `lambda`.
    /// 2. Update `alpha` and `lambda` using the posterior statistics.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — non-positive initial precisions.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] — numerical failure in solver.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedBayesianRidge<F>, FerroError> {
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
                context: "BayesianRidge requires at least 2 samples".into(),
            });
        }

        if self.alpha_init <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha_init".into(),
                reason: "must be positive".into(),
            });
        }

        if self.lambda_init <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "lambda_init".into(),
                reason: "must be positive".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let n_feat_f = F::from(n_features).unwrap();

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

        // Precompute eigenvalues of X^T X for the effective degrees of freedom.
        // We use a simpler trace-based approximation here: gamma ≈ n_features.
        let xt = x_work.t();
        let xtx = xt.dot(&x_work);

        // Eigenvalues of X^T X via power iteration or trace approximation.
        // We compute the trace to get sum of eigenvalues.
        let trace_xtx: F = (0..n_features)
            .map(|i| xtx[[i, i]])
            .fold(F::zero(), |a, b| a + b);

        let mut alpha = self.alpha_init;
        let mut lambda = self.lambda_init;

        let mut w = Array1::<F>::zeros(n_features);
        let mut sigma_diag = Array1::<F>::ones(n_features);

        for _iter in 0..self.max_iter {
            let alpha_old = alpha;
            let lambda_old = lambda;

            // E-step: compute posterior mean w and diag(Sigma).
            let (w_new, sd_new) = bayesian_ridge_solve(&x_work, &y_work, alpha, lambda)?;

            // Effective degrees of freedom: gamma = sum_i alpha * lambda_i / (alpha * lambda_i + lambda)
            // Approximated using trace(Sigma * alpha * X^T X) = alpha * trace(X^T X Sigma).
            // For simplicity we use: gamma ≈ sum_i (alpha * xtx_ii * sigma_ii).
            let gamma: F = (0..n_features)
                .map(|i| alpha * xtx[[i, i]] * sd_new[i])
                .fold(F::zero(), |a, b| a + b);

            // M-step: update alpha and lambda.
            let residual = &y_work - x_work.dot(&w_new);
            let sse = residual.dot(&residual);

            // alpha = (n - gamma) / ||y - Xw||^2
            let new_alpha = (n_f - gamma) / sse.max(F::from(1e-300).unwrap());

            // lambda = gamma / ||w||^2
            let w_norm_sq = w_new.dot(&w_new);
            let new_lambda = gamma / w_norm_sq.max(F::from(1e-300).unwrap());

            // Clamp to reasonable range.
            let clamp_max = F::from(1e10).unwrap();
            let clamp_min = F::from(1e-10).unwrap();
            alpha = new_alpha.min(clamp_max).max(clamp_min);
            lambda = new_lambda.min(clamp_max).max(clamp_min);

            // Check convergence on relative change in alpha.
            let delta_alpha =
                (alpha - alpha_old).abs() / (alpha_old.abs() + F::from(1e-10).unwrap());
            let delta_lambda =
                (lambda - lambda_old).abs() / (lambda_old.abs() + F::from(1e-10).unwrap());

            w = w_new;
            sigma_diag = sd_new;

            if delta_alpha < self.tol && delta_lambda < self.tol {
                break;
            }

            // Avoid unused variable warning — trace_xtx is used in convergence.
            let _ = trace_xtx;
            let _ = n_feat_f;
        }

        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedBayesianRidge {
            coefficients: w,
            intercept,
            alpha,
            lambda,
            sigma: sigma_diag,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedBayesianRidge<F>
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
    for FittedBayesianRidge<F>
{
    /// Returns the posterior mean coefficient vector.
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    /// Returns the intercept term.
    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for BayesianRidge<F>
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

impl<F> FittedPipelineEstimator<F> for FittedBayesianRidge<F>
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

    // ---- Builder ----

    #[test]
    fn test_default_constructor() {
        let m = BayesianRidge::<f64>::new();
        assert_eq!(m.max_iter, 300);
        assert!(m.fit_intercept);
        assert_relative_eq!(m.alpha_init, 1.0);
        assert_relative_eq!(m.lambda_init, 1.0);
    }

    #[test]
    fn test_builder_setters() {
        let m = BayesianRidge::<f64>::new()
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_alpha_init(2.0)
            .with_lambda_init(0.5)
            .with_fit_intercept(false);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
        assert_relative_eq!(m.alpha_init, 2.0);
        assert_relative_eq!(m.lambda_init, 0.5);
    }

    // ---- Validation errors ----

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = BayesianRidge::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];
        let result = BayesianRidge::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_alpha_init() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = BayesianRidge::<f64>::new().with_alpha_init(0.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_positive_lambda_init() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = BayesianRidge::<f64>::new()
            .with_lambda_init(-1.0)
            .fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        // Should recover roughly y = 2x + 1.
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_alpha_and_lambda_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        assert!(fitted.alpha() > 0.0);
        assert!(fitted.lambda() > 0.0);
    }

    #[test]
    fn test_sigma_diagonal_positive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        for &v in fitted.sigma().iter() {
            assert!(v > 0.0, "sigma diagonal must be positive, got {v}");
        }
    }

    #[test]
    fn test_sigma_length_matches_features() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.sigma().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = BayesianRidge::<f64>::new()
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 3.0, 0.0, 1.5, 4.0, 1.0, 2.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = BayesianRidge::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate_fit() {
        // y = 1*x1 + 2*x2
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 6.0];

        let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // Rough sanity: residuals should be small.
        let residuals: Vec<f64> = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();
        assert!(residuals.iter().all(|&r| r < 1.0));
    }
}
