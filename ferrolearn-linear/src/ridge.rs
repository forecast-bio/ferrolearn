//! Ridge regression (L2-regularized linear regression).
//!
//! This module provides [`Ridge`], which fits a linear model with L2
//! regularization using the closed-form solution:
//!
//! ```text
//! w = (X^T X + alpha * I)^{-1} X^T y
//! ```
//!
//! The regularization parameter `alpha` controls the strength of the
//! L2 penalty, shrinking coefficients toward zero.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::Ridge;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = Ridge::<f64>::new();
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

use crate::linalg;

/// Ridge regression (L2-regularized least squares).
///
/// Adds an L2 penalty to the ordinary least squares objective, which
/// shrinks coefficients toward zero and can help with multicollinearity.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Ridge<F> {
    /// Regularization strength. Larger values specify stronger
    /// regularization.
    pub alpha: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> Ridge<F> {
    /// Create a new `Ridge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            fit_intercept: true,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for Ridge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge regression model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// to generate predictions and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedRidge<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for Ridge<F>
{
    type Fitted = FittedRidge<F>;
    type Error = FerroError;

    /// Fit the Ridge regression model using Cholesky decomposition.
    ///
    /// Solves `(X^T X + alpha * I)^{-1} X^T y`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedRidge<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Ridge requires at least one sample".into(),
            });
        }

        if self.fit_intercept {
            // Center the data to handle the intercept.
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y.mean().ok_or_else(|| FerroError::NumericalInstability {
                message: "failed to compute target mean".into(),
            })?;

            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;

            let w = linalg::solve_ridge(&x_centered, &y_centered, self.alpha)?;
            let intercept = y_mean - x_mean.dot(&w);

            Ok(FittedRidge {
                coefficients: w,
                intercept,
            })
        } else {
            let w = linalg::solve_ridge(x, y, self.alpha)?;

            Ok(FittedRidge {
                coefficients: w,
                intercept: F::zero(),
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedRidge<F> {
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedRidge<F> {
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for Ridge<F>
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

impl<F> FittedPipelineEstimator<F> for FittedRidge<F>
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
    fn test_ridge_no_regularization() {
        // With alpha=0, Ridge should behave like OLS.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = Ridge::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model_low = Ridge::<f64>::new().with_alpha(0.01);
        let model_high = Ridge::<f64>::new().with_alpha(100.0);

        let fitted_low = model_low.fit(&x, &y).unwrap();
        let fitted_high = model_high.fit(&x, &y).unwrap();

        // Higher alpha should shrink coefficients more.
        assert!(fitted_high.coefficients()[0].abs() < fitted_low.coefficients()[0].abs());
    }

    #[test]
    fn test_ridge_no_intercept() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = Ridge::<f64>::new()
            .with_alpha(0.0)
            .with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ridge_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = Ridge::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 6.0];

        let model = Ridge::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ridge_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = Ridge::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }
}
