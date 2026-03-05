//! Ordinary Least Squares linear regression.
//!
//! This module provides [`LinearRegression`], which fits a linear model
//! using QR decomposition (via `faer`) to solve the least squares problem:
//!
//! ```text
//! minimize ||X @ w - y||^2
//! ```
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::LinearRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = LinearRegression::<f64>::new();
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

/// Ordinary least squares linear regression.
///
/// Solves the normal equations using QR decomposition for numerical
/// stability. The `fit_intercept` option controls whether a bias
/// (intercept) term is included.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearRegression<F> {
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> LinearRegression<F> {
    /// Create a new `LinearRegression` with default settings.
    ///
    /// Defaults: `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for LinearRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted ordinary least squares linear regression model.
///
/// Stores the learned coefficients and intercept. Implements [`Predict`]
/// to generate predictions and [`HasCoefficients`] for introspection.
#[derive(Debug, Clone)]
pub struct FittedLinearRegression<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float + Send + Sync + ScalarOperand + num_traits::FromPrimitive + 'static>
    Fit<Array2<F>, Array1<F>> for LinearRegression<F>
{
    type Fitted = FittedLinearRegression<F>;
    type Error = FerroError;

    /// Fit the linear regression model.
    ///
    /// Uses the centering trick with Cholesky normal equations for speed.
    /// Falls back to QR decomposition via faer if the normal equations are
    /// ill-conditioned.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in `x`
    /// and `y` differ.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer samples
    /// than features.
    /// Returns [`FerroError::NumericalInstability`] if the system is singular.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedLinearRegression<F>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        // Validate input shapes.
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
                context: "LinearRegression requires at least one sample".into(),
            });
        }

        if self.fit_intercept {
            // Centering trick: center X and y, solve without intercept column,
            // then recover intercept as y_mean - x_mean . w.
            // This avoids the expensive matrix augmentation + QR path.
            let n = F::from(n_samples).unwrap();
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.sum() / n;

            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;

            // Try fast Cholesky normal equations first, fall back to QR.
            let w = linalg::solve_normal_equations(&x_centered, &y_centered)
                .or_else(|_| linalg::solve_lstsq(&x_centered, &y_centered))?;

            let intercept = y_mean - x_mean.dot(&w);

            Ok(FittedLinearRegression {
                coefficients: w,
                intercept,
            })
        } else {
            // Try fast Cholesky normal equations first, fall back to QR.
            let w = linalg::solve_normal_equations(x, y).or_else(|_| linalg::solve_lstsq(x, y))?;

            Ok(FittedLinearRegression {
                coefficients: w,
                intercept: F::zero(),
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLinearRegression<F>
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
    for FittedLinearRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for LinearRegression<F>
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

impl<F> FittedPipelineEstimator<F> for FittedLinearRegression<F>
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
    fn test_simple_linear_regression() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 1.0, epsilon = 1e-10);

        let preds = fitted.predict(&x).unwrap();
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiple_linear_regression() {
        // y = 1*x1 + 2*x2 + 3
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0]).unwrap();
        let y = array![6.0, 7.0, 10.0, 11.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coefficients()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_intercept() {
        // y = 2*x (through origin)
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = LinearRegression::<f64>::new().with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();

        assert_relative_eq!(fitted.coefficients()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length

        let model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_predict() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Wrong number of features
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![2.0, 4.0, 6.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 1);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = LinearRegression::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = LinearRegression::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
