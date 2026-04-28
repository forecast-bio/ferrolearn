//! Lasso regression with built-in cross-validation for alpha selection.
//!
//! This module provides [`LassoCV`], which automatically selects the best
//! regularization parameter `alpha` using k-fold cross-validation. When no
//! explicit alpha grid is provided, the module generates a logarithmically
//! spaced sequence from `alpha_max` (the smallest alpha that zeros all
//! coefficients) down to `alpha_max * epsilon`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::LassoCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = LassoCV::<f64>::new();
//! let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
//! let y = Array1::from_iter((1..=10).map(|i| 2.0 * i as f64 + 1.0));
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 10);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::Lasso;

/// Lasso regression with built-in cross-validation for alpha selection.
///
/// Evaluates a grid of `alpha` values (either user-specified or automatically
/// generated) using k-fold cross-validation, selects the alpha with the
/// lowest mean squared error, and refits the Lasso on the full data.
///
/// # Auto-generated alpha grid
///
/// When `alphas` is `None`, the grid is computed as follows:
///
/// 1. `alpha_max = max(|X^T y|) / n_samples` — the smallest alpha that sets
///    all Lasso coefficients to zero.
/// 2. Generate `n_alphas` values log-spaced from `alpha_max` down to
///    `alpha_max * 1e-3`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LassoCV<F> {
    /// Explicit alpha candidates. `None` means auto-generate.
    alphas: Option<Vec<F>>,
    /// Number of alphas to generate when `alphas` is `None`.
    n_alphas: usize,
    /// Number of cross-validation folds.
    cv: usize,
    /// Maximum number of coordinate descent iterations per Lasso fit.
    max_iter: usize,
    /// Convergence tolerance for coordinate descent.
    tol: F,
    /// Whether to fit an intercept (bias) term.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> LassoCV<F> {
    /// Create a new `LassoCV` with default settings.
    ///
    /// Defaults: `alphas = None` (auto), `n_alphas = 100`, `cv = 5`,
    /// `max_iter = 1000`, `tol = 1e-4`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alphas: None,
            n_alphas: 100,
            cv: 5,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Provide an explicit list of alpha candidates.
    ///
    /// Each value must be non-negative. When set, `n_alphas` is ignored.
    #[must_use]
    pub fn with_alphas(mut self, alphas: Vec<F>) -> Self {
        self.alphas = Some(alphas);
        self
    }

    /// Set the number of alphas to generate on the log-spaced grid.
    ///
    /// Only used when `alphas` is `None`.
    #[must_use]
    pub fn with_n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas;
        self
    }

    /// Set the number of cross-validation folds.
    ///
    /// Must be at least 2.
    #[must_use]
    pub fn with_cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Set the maximum number of coordinate descent iterations.
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

impl<F: Float + FromPrimitive> Default for LassoCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Lasso regression model with cross-validated alpha.
///
/// Stores the selected alpha, the full alpha grid that was evaluated, the
/// learned coefficients, and the intercept.
#[derive(Debug, Clone)]
pub struct FittedLassoCV<F> {
    /// The alpha that achieved the lowest CV error.
    best_alpha: F,
    /// The full grid of alphas that was evaluated.
    alphas: Vec<F>,
    /// Learned coefficient vector (some may be exactly zero).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float> FittedLassoCV<F> {
    /// Returns the alpha value selected by cross-validation.
    #[must_use]
    pub fn best_alpha(&self) -> F {
        self.best_alpha
    }

    /// Returns the full grid of alpha values that was evaluated.
    #[must_use]
    pub fn alphas(&self) -> &[F] {
        &self.alphas
    }
}

/// Split sample indices into `k` roughly equal folds.
fn kfold_indices(n_samples: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for i in 0..n_samples {
        folds[i % k].push(i);
    }
    folds
}

/// Compute mean squared error between two arrays.
fn mse<F: Float + FromPrimitive + 'static>(y_true: &Array1<F>, y_pred: &Array1<F>) -> F {
    let n = F::from(y_true.len()).unwrap();
    let diff = y_true - y_pred;
    diff.dot(&diff) / n
}

/// Gather rows from a 2-D array by index.
fn select_rows<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let ncols = x.ncols();
    let mut out = Array2::<F>::zeros((indices.len(), ncols));
    for (out_row, &idx) in indices.iter().enumerate() {
        out.row_mut(out_row).assign(&x.row(idx));
    }
    out
}

/// Gather elements from a 1-D array by index.
fn select_elements<F: Float>(y: &Array1<F>, indices: &[usize]) -> Array1<F> {
    Array1::from_iter(indices.iter().map(|&i| y[i]))
}

/// Compute `alpha_max = max(|X^T y_centered|) / n_samples`.
///
/// This is the smallest alpha for which the Lasso solution is all zeros
/// (assuming the data is centered).
fn compute_alpha_max<F: Float + FromPrimitive + ScalarOperand>(
    x: &Array2<F>,
    y: &Array1<F>,
    fit_intercept: bool,
) -> F {
    let n = F::from(x.nrows()).unwrap();
    let y_work = if fit_intercept {
        let y_mean = y.mean().unwrap_or(F::zero());
        y - y_mean
    } else {
        y.clone()
    };

    let x_work = if fit_intercept {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        x - &x_mean
    } else {
        x.clone()
    };

    // X^T y_centered
    let xty = x_work.t().dot(&y_work);
    let mut max_abs = F::zero();
    for &v in xty.iter() {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }

    max_abs / n
}

/// Generate `n` log-spaced values from `high` down to `high * eps_ratio`.
fn logspace<F: Float + FromPrimitive>(high: F, eps_ratio: F, n: usize) -> Vec<F> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![high];
    }

    let log_high = high.ln();
    let log_low = (high * eps_ratio).ln();
    let step = (log_low - log_high) / F::from(n - 1).unwrap();

    (0..n)
        .map(|i| (log_high + step * F::from(i).unwrap()).exp())
        .collect()
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for LassoCV<F>
{
    type Fitted = FittedLassoCV<F>;
    type Error = FerroError;

    /// Fit the `LassoCV` model.
    ///
    /// Generates or uses the provided alpha grid, runs k-fold CV for each
    /// alpha, picks the best one, and refits on the full data.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` sizes differ.
    /// - [`FerroError::InvalidParameter`] if an alpha is negative, `cv < 2`,
    ///   or the alpha list (when explicit) is empty.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < cv`.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLassoCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: "number of folds must be at least 2".into(),
            });
        }

        if n_samples < self.cv {
            return Err(FerroError::InsufficientSamples {
                required: self.cv,
                actual: n_samples,
                context: "LassoCV requires at least as many samples as folds".into(),
            });
        }

        // Build alpha grid.
        let alpha_grid: Vec<F> = match &self.alphas {
            Some(user_alphas) => {
                if user_alphas.is_empty() {
                    return Err(FerroError::InvalidParameter {
                        name: "alphas".into(),
                        reason: "must contain at least one candidate".into(),
                    });
                }
                for &a in user_alphas {
                    if a < F::zero() {
                        return Err(FerroError::InvalidParameter {
                            name: "alphas".into(),
                            reason: "all alpha values must be non-negative".into(),
                        });
                    }
                }
                user_alphas.clone()
            }
            None => {
                if self.n_alphas == 0 {
                    return Err(FerroError::InvalidParameter {
                        name: "n_alphas".into(),
                        reason: "must be at least 1".into(),
                    });
                }
                let alpha_max = compute_alpha_max(x, y, self.fit_intercept);
                if alpha_max <= F::zero() {
                    // Degenerate case: y is all zeros or X columns are zero.
                    vec![F::from(1e-6).unwrap(); self.n_alphas]
                } else {
                    logspace(alpha_max, F::from(1e-3).unwrap(), self.n_alphas)
                }
            }
        };

        let folds = kfold_indices(n_samples, self.cv);

        let mut best_alpha = alpha_grid[0];
        let mut best_mse = F::infinity();

        for &alpha in &alpha_grid {
            let mut total_mse = F::zero();

            for fold_idx in 0..self.cv {
                let test_indices = &folds[fold_idx];
                let train_indices: Vec<usize> = folds
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != fold_idx)
                    .flat_map(|(_, v)| v.iter().copied())
                    .collect();

                let x_train = select_rows(x, &train_indices);
                let y_train = select_elements(y, &train_indices);
                let x_test = select_rows(x, test_indices);
                let y_test = select_elements(y, test_indices);

                let model = Lasso::<F>::new()
                    .with_alpha(alpha)
                    .with_max_iter(self.max_iter)
                    .with_tol(self.tol)
                    .with_fit_intercept(self.fit_intercept);

                let fitted = model.fit(&x_train, &y_train)?;
                let preds = fitted.predict(&x_test)?;
                total_mse = total_mse + mse(&y_test, &preds);
            }

            let avg_mse = total_mse / F::from(self.cv).unwrap();

            if avg_mse < best_mse {
                best_mse = avg_mse;
                best_alpha = alpha;
            }
        }

        // Refit on full data with the best alpha.
        let final_model = Lasso::<F>::new()
            .with_alpha(best_alpha)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);
        let final_fitted = final_model.fit(x, y)?;

        Ok(FittedLassoCV {
            best_alpha,
            alphas: alpha_grid,
            coefficients: final_fitted.coefficients().clone(),
            intercept: final_fitted.intercept(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLassoCV<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLassoCV<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_lasso_cv_default_builder() {
        let m = LassoCV::<f64>::new();
        assert!(m.alphas.is_none());
        assert_eq!(m.n_alphas, 100);
        assert_eq!(m.cv, 5);
        assert_eq!(m.max_iter, 1000);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_lasso_cv_builder_setters() {
        let m = LassoCV::<f64>::new()
            .with_alphas(vec![0.1, 1.0])
            .with_n_alphas(50)
            .with_cv(3)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert!(m.alphas.is_some());
        assert_eq!(m.n_alphas, 50);
        assert_eq!(m.cv, 3);
        assert_eq!(m.max_iter, 500);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_lasso_cv_auto_alpha_grid() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((1..=20).map(|i| 2.0 * i as f64 + 1.0));

        let model = LassoCV::<f64>::new().with_n_alphas(10).with_cv(3);

        let fitted = model.fit(&x, &y).unwrap();

        // The auto grid should have generated 10 alphas.
        assert_eq!(fitted.alphas().len(), 10);
        assert!(fitted.best_alpha() > 0.0);
    }

    #[test]
    fn test_lasso_cv_explicit_alphas() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((1..=20).map(|i| 2.0 * i as f64 + 1.0));

        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let model = LassoCV::<f64>::new().with_alphas(alphas.clone()).with_cv(3);

        let fitted = model.fit(&x, &y).unwrap();

        // Best alpha must be one of the supplied candidates.
        assert!(
            alphas
                .iter()
                .any(|&a| (a - fitted.best_alpha()).abs() < 1e-12)
        );
    }

    #[test]
    fn test_lasso_cv_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * i as f64 + 1.0));

        let model = LassoCV::<f64>::new()
            .with_alphas(vec![0.001, 0.01, 0.1])
            .with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);

        for i in 0..10 {
            assert_relative_eq!(preds[i], y[i], epsilon = 2.0);
        }
    }

    #[test]
    fn test_lasso_cv_has_coefficients() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(|i| i as f64));

        let model = LassoCV::<f64>::new()
            .with_alphas(vec![0.01, 0.1])
            .with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_lasso_cv_empty_alphas_error() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = LassoCV::<f64>::new().with_alphas(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_negative_alpha_error() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = LassoCV::<f64>::new().with_alphas(vec![1.0, -0.5]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = LassoCV::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = LassoCV::<f64>::new().with_cv(5);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_cv_too_small() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| i as f64));

        let model = LassoCV::<f64>::new().with_cv(1);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_cv_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(|i| i as f64));

        let fitted = LassoCV::<f64>::new()
            .with_alphas(vec![0.01, 0.1])
            .with_cv(3)
            .fit(&x_train, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_logspace_basic() {
        let vals = logspace(100.0_f64, 0.01, 5);
        assert_eq!(vals.len(), 5);
        // First should be approximately 100, last approximately 1.
        assert_relative_eq!(vals[0], 100.0, epsilon = 1e-8);
        assert_relative_eq!(vals[4], 1.0, epsilon = 1e-8);
        // Should be monotonically decreasing.
        for i in 0..4 {
            assert!(vals[i] > vals[i + 1]);
        }
    }

    #[test]
    fn test_logspace_single() {
        let vals = logspace(10.0_f64, 0.01, 1);
        assert_eq!(vals.len(), 1);
        assert_relative_eq!(vals[0], 10.0, epsilon = 1e-8);
    }

    #[test]
    fn test_logspace_empty() {
        let vals = logspace(10.0_f64, 0.01, 0);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_compute_alpha_max() {
        // For y = [1, 2, 3, 4, 5] and x = [[1],[2],[3],[4],[5]],
        // after centering: x_c = [-2,-1,0,1,2], y_c = [-2,-1,0,1,2]
        // X^T y_c = 10, n = 5 => alpha_max = 2.0
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let amax = compute_alpha_max(&x, &y, true);
        assert_relative_eq!(amax, 2.0, epsilon = 1e-8);
    }
}
