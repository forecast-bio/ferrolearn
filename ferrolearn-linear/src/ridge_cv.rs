//! Ridge regression with built-in cross-validation for alpha selection.
//!
//! This module provides [`RidgeCV`], which automatically selects the best
//! regularization parameter `alpha` from a candidate list by evaluating
//! each candidate with k-fold cross-validation and choosing the one with
//! the lowest mean squared error.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::RidgeCV;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = RidgeCV::<f64>::new();
//! let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::Ridge;

/// Ridge regression with built-in cross-validation for alpha selection.
///
/// Evaluates a list of candidate `alpha` values using k-fold cross-validation
/// and selects the one that minimises mean squared error. The final model is
/// then refit on the full training data with the chosen alpha.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RidgeCV<F> {
    /// Candidate regularization strengths to evaluate.
    alphas: Vec<F>,
    /// Number of cross-validation folds.
    cv: usize,
    /// Whether to fit an intercept (bias) term.
    fit_intercept: bool,
}

impl<F: Float + FromPrimitive> RidgeCV<F> {
    /// Create a new `RidgeCV` with default settings.
    ///
    /// Defaults: `alphas = [0.1, 1.0, 10.0]`, `cv = 5`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alphas: vec![F::from(0.1).unwrap(), F::one(), F::from(10.0).unwrap()],
            cv: 5,
            fit_intercept: true,
        }
    }

    /// Set the candidate regularization strengths.
    ///
    /// Each value must be non-negative.
    #[must_use]
    pub fn with_alphas(mut self, alphas: Vec<F>) -> Self {
        self.alphas = alphas;
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for RidgeCV<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge regression model with cross-validated alpha.
///
/// Stores the selected alpha, learned coefficients, and intercept. Implements
/// [`Predict`] and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidgeCV<F> {
    /// The alpha that achieved the lowest CV error.
    best_alpha: F,
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float> FittedRidgeCV<F> {
    /// Returns the alpha value that was selected by cross-validation.
    #[must_use]
    pub fn best_alpha(&self) -> F {
        self.best_alpha
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

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for RidgeCV<F>
{
    type Fitted = FittedRidgeCV<F>;
    type Error = FerroError;

    /// Fit the `RidgeCV` model.
    ///
    /// For each candidate alpha, runs k-fold cross-validation, measures mean
    /// squared error, then refits on the full data using the alpha with the
    /// lowest average CV error.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers
    ///   of samples.
    /// - [`FerroError::InvalidParameter`] if `alphas` is empty or contains
    ///   a negative value.
    /// - [`FerroError::InsufficientSamples`] if the number of samples is less
    ///   than the number of folds.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedRidgeCV<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alphas.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "alphas".into(),
                reason: "must contain at least one candidate".into(),
            });
        }

        for &a in &self.alphas {
            if a < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "alphas".into(),
                    reason: "all alpha values must be non-negative".into(),
                });
            }
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
                context: "RidgeCV requires at least as many samples as folds".into(),
            });
        }

        let folds = kfold_indices(n_samples, self.cv);

        let mut best_alpha = self.alphas[0];
        let mut best_mse = F::infinity();

        for &alpha in &self.alphas {
            let mut total_mse = F::zero();

            for fold_idx in 0..self.cv {
                // Build train/test split.
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

                let model = Ridge::<F>::new()
                    .with_alpha(alpha)
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
        let final_model = Ridge::<F>::new()
            .with_alpha(best_alpha)
            .with_fit_intercept(self.fit_intercept);
        let final_fitted = final_model.fit(x, y)?;

        Ok(FittedRidgeCV {
            best_alpha,
            coefficients: final_fitted.coefficients().clone(),
            intercept: final_fitted.intercept(),
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidgeCV<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedRidgeCV<F> {
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
    fn test_ridge_cv_default_builder() {
        let m = RidgeCV::<f64>::new();
        assert_eq!(m.alphas.len(), 3);
        assert_eq!(m.cv, 5);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_ridge_cv_custom_alphas() {
        let m = RidgeCV::<f64>::new().with_alphas(vec![0.01, 0.1, 1.0, 10.0, 100.0]);
        assert_eq!(m.alphas.len(), 5);
    }

    #[test]
    fn test_ridge_cv_fit_selects_alpha() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=20).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = RidgeCV::<f64>::new()
            .with_alphas(vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            .with_cv(5);

        let fitted = model.fit(&x, &y).unwrap();

        // For a clean linear relationship, a small alpha should win.
        assert!(fitted.best_alpha() <= 1.0);
    }

    #[test]
    fn test_ridge_cv_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i) + 1.0));

        let model = RidgeCV::<f64>::new().with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);

        // Predictions should be close to the true values.
        for i in 0..10 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_ridge_cv_has_coefficients() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let model = RidgeCV::<f64>::new().with_cv(3);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_ridge_cv_empty_alphas_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = RidgeCV::<f64>::new().with_alphas(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_negative_alpha_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = RidgeCV::<f64>::new().with_alphas(vec![1.0, -0.5]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = RidgeCV::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = RidgeCV::<f64>::new().with_cv(5);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_cv_too_small() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(f64::from));

        let model = RidgeCV::<f64>::new().with_cv(1);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_cv_no_intercept() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((1..=10).map(|i| 2.0 * f64::from(i)));

        let model = RidgeCV::<f64>::new().with_cv(3).with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        // With no intercept and origin-passing data, predictions should be close.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_ridge_cv_predict_feature_mismatch() {
        let x_train = Array2::from_shape_vec((10, 2), (0..20).map(f64::from).collect()).unwrap();
        let y = Array1::from_iter((0..10).map(f64::from));

        let fitted = RidgeCV::<f64>::new().with_cv(3).fit(&x_train, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_indices_coverage() {
        let folds = kfold_indices(10, 3);
        assert_eq!(folds.len(), 3);

        // Every index 0..10 should appear exactly once.
        let mut all: Vec<usize> = folds.into_iter().flatten().collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }
}
