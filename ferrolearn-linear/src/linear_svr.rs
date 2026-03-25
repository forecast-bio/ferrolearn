//! Linear Support Vector Regressor.
//!
//! This module provides [`LinearSVR`], an optimized linear SVR that operates
//! directly in the primal space without kernel overhead. It uses coordinate
//! descent on the L2-regularized epsilon-insensitive or squared
//! epsilon-insensitive loss.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::linear_svr::LinearSVR;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! let model = LinearSVR::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Loss function for [`LinearSVR`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSVRLoss {
    /// Epsilon-insensitive loss: `max(0, |y - f(x)| - epsilon)`.
    EpsilonInsensitive,
    /// Squared epsilon-insensitive loss: `max(0, |y - f(x)| - epsilon)^2`.
    SquaredEpsilonInsensitive,
}

/// Linear Support Vector Regressor (primal formulation).
///
/// Solves the L2-regularized epsilon-insensitive or squared
/// epsilon-insensitive loss via coordinate descent in the primal.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearSVR<F> {
    /// Inverse regularization strength.
    pub c: F,
    /// Width of the epsilon-insensitive tube.
    pub epsilon: F,
    /// Maximum number of coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the change in weight vector.
    pub tol: F,
    /// Loss function to use.
    pub loss: LinearSVRLoss,
}

impl<F: Float> LinearSVR<F> {
    /// Create a new `LinearSVR` with default settings.
    ///
    /// Defaults: `C = 1.0`, `epsilon = 0.1`, `max_iter = 1000`,
    /// `tol = 1e-4`, `loss = EpsilonInsensitive`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: F::one(),
            epsilon: F::from(0.1).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            loss: LinearSVRLoss::EpsilonInsensitive,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon tube width.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
        self
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

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVRLoss) -> Self {
        self.loss = loss;
        self
    }
}

impl<F: Float> Default for LinearSVR<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear Support Vector Regressor.
///
/// Stores the learned coefficients and intercept.
#[derive(Debug, Clone)]
pub struct FittedLinearSVR<F> {
    /// Learned coefficient vector (one per feature).
    coefficients: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<F>>
    for LinearSVR<F>
{
    type Fitted = FittedLinearSVR<F>;
    type Error = FerroError;

    /// Fit the linear SVR model using coordinate descent.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `C` not positive or epsilon negative.
    /// - [`FerroError::InsufficientSamples`] — no samples provided.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedLinearSVR<F>, FerroError> {
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
                context: "LinearSVR requires at least one sample".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        if self.epsilon < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "epsilon".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        let step = F::from(0.01).unwrap();

        for _iter in 0..self.max_iter {
            let mut max_change = F::zero();

            // Update each weight coordinate.
            for j in 0..n_features {
                let mut grad = w[j]; // regularization gradient
                for i in 0..n_samples {
                    let pred = x.row(i).dot(&w) + b;
                    let residual = y[i] - pred;
                    let abs_residual = residual.abs();

                    if abs_residual > self.epsilon {
                        match self.loss {
                            LinearSVRLoss::EpsilonInsensitive => {
                                let sign = if residual > F::zero() {
                                    F::one()
                                } else {
                                    -F::one()
                                };
                                grad = grad - self.c / n_f * sign * x[[i, j]];
                            }
                            LinearSVRLoss::SquaredEpsilonInsensitive => {
                                let two = F::from(2.0).unwrap();
                                let sign = if residual > F::zero() {
                                    F::one()
                                } else {
                                    -F::one()
                                };
                                grad = grad
                                    - two * self.c / n_f
                                        * (abs_residual - self.epsilon)
                                        * sign
                                        * x[[i, j]];
                            }
                        }
                    }
                }

                let new_w = w[j] - step * grad;
                let change = (new_w - w[j]).abs();
                if change > max_change {
                    max_change = change;
                }
                w[j] = new_w;
            }

            // Update intercept.
            {
                let mut grad_b = F::zero();
                for i in 0..n_samples {
                    let pred = x.row(i).dot(&w) + b;
                    let residual = y[i] - pred;
                    let abs_residual = residual.abs();

                    if abs_residual > self.epsilon {
                        match self.loss {
                            LinearSVRLoss::EpsilonInsensitive => {
                                let sign = if residual > F::zero() {
                                    F::one()
                                } else {
                                    -F::one()
                                };
                                grad_b = grad_b - self.c / n_f * sign;
                            }
                            LinearSVRLoss::SquaredEpsilonInsensitive => {
                                let two = F::from(2.0).unwrap();
                                let sign = if residual > F::zero() {
                                    F::one()
                                } else {
                                    -F::one()
                                };
                                grad_b = grad_b
                                    - two * self.c / n_f * (abs_residual - self.epsilon) * sign;
                            }
                        }
                    }
                }
                let new_b = b - step * grad_b;
                let change = (new_b - b).abs();
                if change > max_change {
                    max_change = change;
                }
                b = new_b;
            }

            if max_change < self.tol {
                break;
            }
        }

        Ok(FittedLinearSVR {
            coefficients: w,
            intercept: b,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLinearSVR<F>
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
    for FittedLinearSVR<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for LinearSVR<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
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

impl<F> FittedPipelineEstimator<F> for FittedLinearSVR<F>
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
        let m = LinearSVR::<f64>::new();
        assert_eq!(m.max_iter, 1000);
        assert!(m.c == 1.0);
        assert_relative_eq!(m.epsilon, 0.1);
        assert_eq!(m.loss, LinearSVRLoss::EpsilonInsensitive);
    }

    #[test]
    fn test_builder_setters() {
        let m = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.5)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive);
        assert!(m.c == 10.0);
        assert_relative_eq!(m.epsilon, 0.5);
        assert_eq!(m.max_iter, 500);
        assert_eq!(m.loss, LinearSVRLoss::SquaredEpsilonInsensitive);
    }

    #[test]
    fn test_fits_linear_data() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.0)
            .with_max_iter(10000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Should roughly recover y = 2x.
        for (p, &t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 3.0, "prediction {p} too far from target {t}");
        }
    }

    #[test]
    fn test_squared_epsilon_insensitive() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = LinearSVR::<f64>::new()
            .with_c(10.0)
            .with_epsilon(0.0)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
            .with_max_iter(10000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = LinearSVR::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearSVR::<f64>::new().with_c(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_negative_epsilon() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = LinearSVR::<f64>::new().with_epsilon(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let fitted = LinearSVR::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
            .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let fitted = LinearSVR::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0];

        let model = LinearSVR::<f64>::new().with_max_iter(5000);
        let fitted_pipe = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted_pipe.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
