//! Orthogonal Matching Pursuit (OMP).
//!
//! This module provides [`OrthogonalMatchingPursuit`], a greedy feature
//! selection algorithm that iteratively selects the feature most correlated
//! with the current residual, adds it to a support set, solves OLS on
//! the support, and updates the residual. The process repeats until the
//! desired number of non-zero coefficients is reached or the residual
//! tolerance is met.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::OrthogonalMatchingPursuit;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 3), vec![
//!     1.0, 0.0, 0.0,
//!     2.0, 0.1, 0.0,
//!     3.0, 0.0, 0.1,
//!     4.0, 0.1, 0.0,
//!     5.0, 0.0, 0.1,
//! ]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(1);
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

/// Orthogonal Matching Pursuit.
///
/// A greedy sparse approximation algorithm that selects features one at a
/// time. At each iteration it picks the feature most correlated with the
/// residual, adds it to the support, solves OLS on the support set, and
/// re-computes the residual.
///
/// Termination is controlled by either `n_nonzero_coefs` (maximum
/// support size) or `tol` (residual norm threshold), whichever is reached
/// first.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct OrthogonalMatchingPursuit<F> {
    /// Maximum number of non-zero coefficients. Defaults to `None` (use
    /// all features or stop at `tol`).
    pub n_nonzero_coefs: Option<usize>,
    /// Residual norm tolerance. If the squared residual norm drops below
    /// this threshold the algorithm terminates. Defaults to `None`.
    pub tol: Option<F>,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> OrthogonalMatchingPursuit<F> {
    /// Create a new `OrthogonalMatchingPursuit` with default settings.
    ///
    /// Defaults: `n_nonzero_coefs = None`, `tol = None`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: None,
            fit_intercept: true,
        }
    }

    /// Set the maximum number of non-zero coefficients.
    #[must_use]
    pub fn with_n_nonzero_coefs(mut self, n: usize) -> Self {
        self.n_nonzero_coefs = Some(n);
        self
    }

    /// Set the residual norm tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = Some(tol);
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for OrthogonalMatchingPursuit<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Orthogonal Matching Pursuit model.
///
/// Stores the learned (sparse) coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedOMP<F> {
    /// Learned coefficient vector (many entries may be zero).
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

/// Solve OLS on the active columns, returning the full-length coefficient vector.
fn ols_active<F: Float + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    support: &[usize],
    n_features: usize,
) -> Result<Array1<F>, FerroError> {
    let n_samples = x.nrows();
    let k = support.len();

    let mut xa = Array2::<F>::zeros((n_samples, k));
    for (col_idx, &j) in support.iter().enumerate() {
        for i in 0..n_samples {
            xa[[i, col_idx]] = x[[i, j]];
        }
    }

    let xat = xa.t();
    let xtx = xat.dot(&xa);
    let xty = xat.dot(y);

    let w_active =
        cholesky_solve(&xtx, &xty).or_else(|_| gaussian_solve(k, &xtx, &xty))?;

    let mut w = Array1::<F>::zeros(n_features);
    for (col_idx, &j) in support.iter().enumerate() {
        w[j] = w_active[col_idx];
    }
    Ok(w)
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for OrthogonalMatchingPursuit<F>
{
    type Fitted = FittedOMP<F>;
    type Error = FerroError;

    /// Fit the OMP model.
    ///
    /// Greedily selects features by correlation with the residual and
    /// solves OLS on the growing support set.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — `n_nonzero_coefs` exceeds features,
    ///   or neither `n_nonzero_coefs` nor `tol` is set.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedOMP<F>, FerroError> {
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
                context: "OMP requires at least one sample".into(),
            });
        }

        // At least one stopping criterion must be set.
        if self.n_nonzero_coefs.is_none() && self.tol.is_none() {
            return Err(FerroError::InvalidParameter {
                name: "n_nonzero_coefs / tol".into(),
                reason: "at least one stopping criterion must be set".into(),
            });
        }

        let max_k = self
            .n_nonzero_coefs
            .unwrap_or(n_features)
            .min(n_features);

        if let Some(n) = self.n_nonzero_coefs {
            if n > n_features {
                return Err(FerroError::InvalidParameter {
                    name: "n_nonzero_coefs".into(),
                    reason: format!(
                        "cannot exceed number of features ({n_features})"
                    ),
                });
            }
        }

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

        let mut support: Vec<usize> = Vec::with_capacity(max_k);
        let mut in_support = vec![false; n_features];
        let mut w = Array1::<F>::zeros(n_features);
        let mut residual = y_work.clone();

        for _step in 0..max_k {
            // Check residual tolerance.
            if let Some(tol_val) = self.tol {
                let res_norm_sq = residual.dot(&residual);
                if res_norm_sq < tol_val {
                    break;
                }
            }

            // Find feature most correlated with residual.
            let mut best_j = None;
            let mut best_corr = F::zero();
            for (j, &is_in_support) in in_support.iter().enumerate() {
                if is_in_support {
                    continue;
                }
                let corr = x_work.column(j).dot(&residual).abs();
                if corr > best_corr {
                    best_corr = corr;
                    best_j = Some(j);
                }
            }

            let j = match best_j {
                Some(j) => j,
                None => break,
            };

            support.push(j);
            in_support[j] = true;

            // OLS on support set.
            w = ols_active(&x_work, &y_work, &support, n_features)?;

            // Update residual.
            residual = &y_work - x_work.dot(&w);
        }

        let intercept = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            *ym - xm.dot(&w)
        } else {
            F::zero()
        };

        Ok(FittedOMP {
            coefficients: w,
            intercept,
        })
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedOMP<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedOMP<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F> PipelineEstimator<F> for OrthogonalMatchingPursuit<F>
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

impl<F> FittedPipelineEstimator<F> for FittedOMP<F>
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
        let m = OrthogonalMatchingPursuit::<f64>::new();
        assert!(m.n_nonzero_coefs.is_none());
        assert!(m.tol.is_none());
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(3)
            .with_tol(1e-4)
            .with_fit_intercept(false);
        assert_eq!(m.n_nonzero_coefs, Some(3));
        assert_relative_eq!(m.tol.unwrap(), 1e-4);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_no_stopping_criterion() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(OrthogonalMatchingPursuit::<f64>::new().fit(&x, &y).is_err());
    }

    #[test]
    fn test_n_nonzero_exceeds_features() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(5)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_simple_linear() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sparsity() {
        // With n_nonzero_coefs=1, only one coefficient should be non-zero.
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.3, 0.03, 4.0, 0.4, 0.04,
                5.0, 0.5, 0.05, 6.0, 0.6, 0.06, 7.0, 0.7, 0.07, 8.0, 0.8, 0.08,
                9.0, 0.9, 0.09, 10.0, 1.0, 0.10,
            ],
        )
        .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let nonzero = fitted
            .coefficients()
            .iter()
            .filter(|&&c| c.abs() > 1e-10)
            .count();
        assert_eq!(nonzero, 1);
    }

    #[test]
    fn test_tol_stopping() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect linear

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_tol(1e-10)
            .fit(&x, &y)
            .unwrap();
        // Should find perfect fit with 1 feature.
        let preds = fitted.predict(&x).unwrap();
        for (pred, actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(pred, actual, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .fit(&x, &y)
            .unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(2)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(1)
            .with_fit_intercept(false)
            .fit(&x, &y)
            .unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];
        let model = OrthogonalMatchingPursuit::<f64>::new().with_n_nonzero_coefs(1);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_multivariate_recovery() {
        // y = 1*x1 + 3*x2, OMP with n_nonzero_coefs=2 should recover both.
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 1.0, 1.0, 0.1, 2.0, 0.0, 0.8, 0.0, 2.0, 0.4,
            ],
        )
        .unwrap();
        let y = array![1.0, 3.0, 4.0, 2.0, 6.0]; // = x1 + 3*x2

        let fitted = OrthogonalMatchingPursuit::<f64>::new()
            .with_n_nonzero_coefs(2)
            .fit(&x, &y)
            .unwrap();

        // The third feature should remain approximately zero.
        assert!(
            fitted.coefficients()[2].abs() < 0.5,
            "irrelevant feature should have near-zero coefficient"
        );
    }
}
