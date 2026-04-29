//! Quantile Regression via IRLS on the pinball loss.
//!
//! This module provides [`QuantileRegressor`], which estimates conditional
//! quantiles of the response variable. The default `quantile = 0.5`
//! corresponds to the conditional median, which is more robust to outliers
//! than the conditional mean (OLS).
//!
//! The pinball (check) loss for quantile `q` is:
//!
//! ```text
//! L_q(r) = q * max(r, 0) + (1 - q) * max(-r, 0)
//! ```
//!
//! The model is fitted via IRLS with weights `w_i = 1 / (2 * max(|r_i|, eps))`
//! and optional L1 regularization (`alpha`).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::QuantileRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = QuantileRegressor::<f64>::new(); // median regression
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

/// Quantile Regressor — conditional quantile estimation via IRLS.
///
/// Minimises the pinball loss with optional L1 regularization. The IRLS
/// weights are `w_i = 1 / (2 * max(|r_i|, eps))`, which gives the
/// iteratively reweighted least absolute deviations procedure.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct QuantileRegressor<F> {
    /// The quantile to estimate (must be in (0, 1)). Default 0.5 (median).
    pub quantile: F,
    /// L1 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> QuantileRegressor<F> {
    /// Create a new `QuantileRegressor` with default settings.
    ///
    /// Defaults: `quantile = 0.5`, `alpha = 1.0`, `max_iter = 1000`,
    /// `tol = 1e-5`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantile: F::from(0.5).unwrap(),
            alpha: F::one(),
            max_iter: 1000,
            tol: F::from(1e-5).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the quantile to estimate.
    ///
    /// Must be strictly between 0 and 1.
    #[must_use]
    pub fn with_quantile(mut self, quantile: F) -> Self {
        self.quantile = quantile;
        self
    }

    /// Set the L1 regularization strength.
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

impl<F: Float + FromPrimitive> Default for QuantileRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Quantile Regressor model.
///
/// Stores the learned coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedQuantileRegressor<F> {
    /// Learned coefficient vector.
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
        for k in 0..i {
            s = s - l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s = s - l[[k, i]] * x_sol[k];
        }
        x_sol[i] = s / l[[i, i]];
    }

    Ok(x_sol)
}

/// Gaussian elimination with partial pivoting.
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

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
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
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in back substitution".into(),
            });
        }
        x_sol[i] = s / aug[[i, i]];
    }

    Ok(x_sol)
}

/// Solve the weighted least-squares problem with L1 penalty approximation.
///
/// `(X^T W X + alpha * diag) w = X^T W y`
///
/// For the quantile regression IRLS, the L1 penalty is linearised around
/// the current coefficients.
fn weighted_l1_solve<F: Float + FromPrimitive>(
    x: &Array2<F>,
    y: &Array1<F>,
    weights: &Array1<F>,
    alpha: F,
    prev_coef: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();
    let eps = F::from(1e-8).unwrap();

    let mut xtwx = Array2::<F>::zeros((n_features, n_features));
    let mut xtwy = Array1::<F>::zeros(n_features);

    for i in 0..n_samples {
        let wi = weights[i];
        let xi = x.row(i);
        for r in 0..n_features {
            xtwy[r] = xtwy[r] + wi * xi[r] * y[i];
            for c in 0..n_features {
                xtwx[[r, c]] = xtwx[[r, c]] + wi * xi[r] * xi[c];
            }
        }
    }

    // Add L1 penalty via IRLS approximation: penalise with alpha / max(|w_j|, eps).
    for j in 0..n_features {
        let pen = alpha / prev_coef[j].abs().max(eps);
        xtwx[[j, j]] = xtwx[[j, j]] + pen;
    }

    cholesky_solve(&xtwx, &xtwy).or_else(|_| gaussian_solve(n_features, &xtwx, &xtwy))
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for QuantileRegressor<F>
{
    type Fitted = FittedQuantileRegressor<F>;
    type Error = FerroError;

    /// Fit the quantile regression model via IRLS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — quantile outside (0, 1) or
    ///   negative alpha.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedQuantileRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "QuantileRegressor requires at least one sample".into(),
            });
        }

        if self.quantile <= F::zero() || self.quantile >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "quantile".into(),
                reason: "must be strictly between 0 and 1".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        let eps = F::from(1e-8).unwrap();
        let one = F::one();
        let q = self.quantile;

        // Center data if fitting intercept.
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

        // Initialise coefficients to zero.
        let mut w = Array1::<F>::zeros(n_features);
        // Initialise with small values for L1 linearisation.
        let mut w_prev = Array1::from_elem(n_features, eps);

        for _iter in 0..self.max_iter {
            let w_old = w.clone();

            // Compute residuals.
            let residuals = &y_work - x_work.dot(&w);

            // Compute IRLS weights for pinball loss.
            // weight_i = asymmetric_weight_i / (2 * max(|r_i|, eps))
            let mut weights = Array1::<F>::zeros(n_samples);
            for i in 0..n_samples {
                let abs_r = residuals[i].abs().max(eps);
                // Asymmetric weight: q for positive residuals, (1-q) for negative.
                let asym = if residuals[i] >= F::zero() { q } else { one - q };
                weights[i] = asym / abs_r;
            }

            // Working response is y_work itself (we re-solve for w directly).
            w = weighted_l1_solve(&x_work, &y_work, &weights, self.alpha, &w_prev)?;
            w_prev = w.mapv(|v| v.abs().max(eps));

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

        Ok(FittedQuantileRegressor {
            coefficients: w,
            intercept,
        })
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedQuantileRegressor<F>
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
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coefficients) + self.intercept)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedQuantileRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for QuantileRegressor<F>
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

impl<F> FittedPipelineEstimator<F> for FittedQuantileRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_defaults() {
        let m = QuantileRegressor::<f64>::new();
        assert_relative_eq!(m.quantile, 0.5);
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = QuantileRegressor::<f64>::new()
            .with_quantile(0.9)
            .with_alpha(0.5)
            .with_max_iter(500)
            .with_tol(1e-8)
            .with_fit_intercept(false);
        assert_relative_eq!(m.quantile, 0.9);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(QuantileRegressor::<f64>::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_quantile_zero() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(QuantileRegressor::<f64>::new()
            .with_quantile(0.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_invalid_quantile_one() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(QuantileRegressor::<f64>::new()
            .with_quantile(1.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(QuantileRegressor::<f64>::new()
            .with_alpha(-1.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_median_regression_clean_data() {
        // On clean linear data, median regression should approximate OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(2000)
            .fit(&x, &y)
            .unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1.0);
    }

    #[test]
    fn test_predict_length() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = QuantileRegressor::<f64>::new().fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = QuantileRegressor::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = QuantileRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_high_quantile_higher_prediction() {
        // A higher quantile should generally yield higher predicted values.
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        // y with some noise.
        let y = array![2.5, 3.8, 6.2, 7.9, 10.5, 12.1, 14.3, 15.8, 18.2, 20.5];

        let fitted_low = QuantileRegressor::<f64>::new()
            .with_quantile(0.1)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();
        let fitted_high = QuantileRegressor::<f64>::new()
            .with_quantile(0.9)
            .with_alpha(0.0)
            .fit(&x, &y)
            .unwrap();

        let x_test = Array2::from_shape_vec((1, 1), vec![5.5]).unwrap();
        let pred_low = fitted_low.predict(&x_test).unwrap()[0];
        let pred_high = fitted_high.predict(&x_test).unwrap()[0];

        assert!(
            pred_high >= pred_low,
            "q=0.9 prediction ({pred_high}) should be >= q=0.1 prediction ({pred_low})"
        );
    }

    #[test]
    fn test_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = QuantileRegressor::<f64>::new().with_alpha(0.0);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
