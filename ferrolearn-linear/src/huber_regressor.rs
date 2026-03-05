//! Huber Regressor — robust regression via IRLS.
//!
//! This module provides [`HuberRegressor`], a robust regression estimator
//! that uses the Huber loss function. Unlike OLS (which uses squared loss),
//! the Huber loss is quadratic for residuals smaller than `epsilon` and
//! linear (i.e., MAE-like) for larger residuals. This makes it substantially
//! less sensitive to outliers.
//!
//! The Huber loss for a single residual `r` is:
//!
//! ```text
//! L(r) = { (1/2) * r²           if |r| <= epsilon
//!         { epsilon*(|r| - ε/2)  if |r| > epsilon
//! ```
//!
//! The model is fitted via **Iteratively Reweighted Least Squares (IRLS)**:
//! each iteration solves a weighted Ridge problem where samples with large
//! residuals receive reduced weight.
//!
//! An L2 penalty (`alpha`) on the coefficients is also supported.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::HuberRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = HuberRegressor::<f64>::new();
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![3.0, 5.0, 7.0, 9.0, 100.0]; // last point is an outlier
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

/// Huber Regressor — robust regression less sensitive to outliers.
///
/// Fits by iteratively reweighted least squares (IRLS): samples whose
/// residuals exceed `epsilon` are down-weighted by `epsilon / |r|`, reducing
/// their influence on the fit.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct HuberRegressor<F> {
    /// Threshold between quadratic and linear Huber loss regions.
    /// Typically around 1.35 (the default), which gives ~95% efficiency
    /// for Gaussian-distributed errors.
    pub epsilon: F,
    /// L2 regularization strength applied to the coefficients.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> HuberRegressor<F> {
    /// Create a new `HuberRegressor` with default settings.
    ///
    /// Defaults: `epsilon = 1.35`, `alpha = 0.0001`, `max_iter = 100`,
    /// `tol = 1e-5`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            epsilon: F::from(1.35).unwrap(),
            alpha: F::from(1e-4).unwrap(),
            max_iter: 100,
            tol: F::from(1e-5).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the Huber threshold `epsilon`.
    ///
    /// Must be strictly greater than 1.0.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for HuberRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Huber Regressor model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// and [`HasCoefficients`].
#[derive(Debug, Clone)]
pub struct FittedHuberRegressor<F> {
    /// Learned coefficient vector.
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

/// Solve the weighted ridge system `(X^T W X + alpha I) w = X^T W y`.
///
/// `weights` are the diagonal of `W`. Uses Cholesky or Gaussian fallback.
fn weighted_ridge_solve<F: Float + FromPrimitive>(
    x: &Array2<F>,
    y: &Array1<F>,
    weights: &Array1<F>,
    alpha: F,
) -> Result<Array1<F>, FerroError> {
    let (_n_samples, n_features) = x.dim();

    // Build X^T W X and X^T W y by iterating over samples.
    let mut xtwx = Array2::<F>::zeros((n_features, n_features));
    let mut xtwy = Array1::<F>::zeros(n_features);

    for i in 0.._n_samples {
        let wi = weights[i];
        let xi = x.row(i);
        // Outer product contribution: wi * xi * xi^T
        for r in 0..n_features {
            xtwy[r] = xtwy[r] + wi * xi[r] * y[i];
            for c in 0..n_features {
                xtwx[[r, c]] = xtwx[[r, c]] + wi * xi[r] * xi[c];
            }
        }
    }

    // Add L2 regularization.
    for i in 0..n_features {
        xtwx[[i, i]] = xtwx[[i, i]] + alpha;
    }

    cholesky_solve(&xtwx, &xtwy).or_else(|_| gaussian_solve(n_features, &xtwx, &xtwy))
}

/// Cholesky solve for `A x = b`.
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

    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s = s - l[[i, j]] * z[j];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s = s - l[[j, i]] * x_sol[j];
        }
        x_sol[i] = s / l[[i, i]];
    }

    Ok(x_sol)
}

/// Gaussian elimination with partial pivoting fallback.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or(F::epsilon()) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in Gaussian elimination".into(),
            });
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s = s - aug[[i, j]] * x_sol[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or(F::epsilon()) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in back substitution".into(),
            });
        }
        x_sol[i] = s / aug[[i, i]];
    }

    Ok(x_sol)
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for HuberRegressor<F>
{
    type Fitted = FittedHuberRegressor<F>;
    type Error = FerroError;

    /// Fit the Huber Regressor via Iteratively Reweighted Least Squares (IRLS).
    ///
    /// Each IRLS iteration:
    /// 1. Computes residuals `r = y - X w - intercept`.
    /// 2. Assigns Huber weights: `w_i = 1` if `|r_i| <= epsilon`,
    ///    else `w_i = epsilon / |r_i|`.
    /// 3. Solves the weighted Ridge system.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `epsilon <= 1.0` or negative `alpha`.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::NumericalInstability`] — numerical failure.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedHuberRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.epsilon <= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "epsilon".into(),
                reason: "must be strictly greater than 1.0".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "HuberRegressor requires at least one sample".into(),
            });
        }

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

        // Initialize coefficients to zero.
        let mut w = Array1::<F>::zeros(n_features);
        // Uniform weights to start.
        let mut weights = Array1::<F>::ones(n_samples);

        let one = F::one();
        let min_weight = F::from(1e-10).unwrap();

        for _iter in 0..self.max_iter {
            let w_old = w.clone();

            // Solve weighted ridge.
            w = weighted_ridge_solve(&x_work, &y_work, &weights, self.alpha)?;

            // Recompute residuals.
            let residuals = &y_work - x_work.dot(&w);

            // Update Huber weights.
            for i in 0..n_samples {
                let abs_r = residuals[i].abs();
                weights[i] = if abs_r <= self.epsilon {
                    one
                } else {
                    (self.epsilon / abs_r).max(min_weight)
                };
            }

            // Check convergence.
            let max_change = w
                .iter()
                .zip(w_old.iter())
                .map(|(&wn, &wo)| (wn - wo).abs())
                .fold(F::zero(), |a, b| if b > a { b } else { a });

            if max_change < self.tol {
                break;
            }
        }

        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedHuberRegressor {
            coefficients: w,
            intercept,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedHuberRegressor<F>
{
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedHuberRegressor<F>
{
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
impl<F> PipelineEstimator<F> for HuberRegressor<F>
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

impl<F> FittedPipelineEstimator<F> for FittedHuberRegressor<F>
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
        let m = HuberRegressor::<f64>::new();
        assert_relative_eq!(m.epsilon, 1.35);
        assert_relative_eq!(m.alpha, 1e-4);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder_setters() {
        let m = HuberRegressor::<f64>::new()
            .with_epsilon(2.0)
            .with_alpha(0.1)
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.epsilon, 2.0);
        assert_relative_eq!(m.alpha, 0.1);
        assert_eq!(m.max_iter, 50);
        assert!(!m.fit_intercept);
    }

    // ---- Validation errors ----

    #[test]
    fn test_epsilon_too_small_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_epsilon(0.5).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_epsilon_exactly_one_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_epsilon(1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let result = HuberRegressor::<f64>::new().with_alpha(-1.0).fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        let result = HuberRegressor::<f64>::new().fit(&x, &y);
        assert!(result.is_err());
    }

    // ---- Correctness ----

    #[test]
    fn test_fits_clean_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_robust_to_outliers() {
        // 9 inliers following y = 2x, 1 large outlier.
        // With majority inliers, Huber should be much more robust than OLS.
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y_clean = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let y_outlier = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 200.0];

        let fitted_clean = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y_clean)
            .unwrap();

        let fitted_huber = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y_outlier)
            .unwrap();

        // OLS on the outlier data.
        let ols_coef = {
            // Manual OLS: slope = (sum xi*yi - n*xmean*ymean) / (sum xi^2 - n*xmean^2)
            // x = [1..10], y_outlier = [2,4,...,18,200]
            // Just verify the Huber is more robust: |huber - clean| < |ols - clean|.
            // For OLS the outlier pulls the slope significantly higher.
            // y_outlier OLS slope ≈ much larger than 2.0
            let n = 10.0_f64;
            let xv: Vec<f64> = (1..=10).map(|i| i as f64).collect();
            let yv = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 200.0];
            let xmean = xv.iter().sum::<f64>() / n;
            let ymean = yv.iter().sum::<f64>() / n;
            let num: f64 = xv
                .iter()
                .zip(yv.iter())
                .map(|(xi, yi)| xi * yi)
                .sum::<f64>()
                - n * xmean * ymean;
            let den: f64 = xv.iter().map(|xi| xi * xi).sum::<f64>() - n * xmean * xmean;
            num / den
        };

        let huber_coef = fitted_huber.coefficients()[0];
        let clean_coef = fitted_clean.coefficients()[0];

        // The Huber coefficient should be closer to the clean slope than OLS is.
        let huber_err = (huber_coef - clean_coef).abs();
        let ols_err = (ols_coef - clean_coef).abs();
        assert!(
            huber_err < ols_err,
            "Huber error {huber_err:.4} should be less than OLS error {ols_err:.4} \
             (huber coef={huber_coef:.4}, ols coef={ols_coef:.4}, clean coef={clean_coef:.4})"
        );
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients_length() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.5, 0.2, 2.0, 1.0, 0.4, 3.0, 1.5, 0.6, 4.0, 2.0, 0.8],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let fitted = HuberRegressor::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 3);
    }

    #[test]
    fn test_large_epsilon_approaches_ols() {
        // With very large epsilon, all residuals fall in the quadratic zone
        // so Huber ≈ WLS with uniform weights ≈ OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_epsilon(1000.0)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 0.5);
    }

    #[test]
    fn test_l2_regularization_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let low = HuberRegressor::<f64>::new()
            .with_alpha(0.0001)
            .fit(&x, &y)
            .unwrap();
        let high = HuberRegressor::<f64>::new()
            .with_alpha(100.0)
            .fit(&x, &y)
            .unwrap();

        assert!(
            high.coefficients()[0].abs() <= low.coefficients()[0].abs(),
            "higher alpha should shrink more"
        );
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = HuberRegressor::<f64>::new();
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.5]).unwrap();
        let y = array![1.0, 2.0, 3.0, 3.0];

        let fitted = HuberRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
