//! Regressor that transforms the target before fitting and inverts it at
//! prediction time.
//!
//! [`TransformedTargetRegressor`] wraps a base pipeline regressor with a pair
//! of functions: a forward transform applied to `y` before training, and an
//! inverse transform applied to predictions.
//!
//! This is useful when the target distribution benefits from a transformation
//! (e.g., log, sqrt, Box-Cox) that the model itself does not handle
//! internally.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::transformed_target::TransformedTargetRegressor;
//! use ferrolearn_core::pipeline::Pipeline;
//!
//! // Log-transform the target, then exponentiate predictions.
//! let ttr = TransformedTargetRegressor::<f64>::new(
//!     Pipeline::new(), // would normally include an estimator step
//!     |y| y.ln(),
//!     |y| y.exp(),
//! );
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// TransformedTargetRegressor (unfitted)
// ---------------------------------------------------------------------------

/// A meta-regressor that applies a transformation to the target variable.
///
/// **Fit**: transforms `y` via `func`, then fits the inner pipeline on
/// `(X, func(y))`.
///
/// **Predict**: runs the pipeline's predict, then applies `inverse_func`
/// element-wise to the raw predictions.
pub struct TransformedTargetRegressor<F: Float + Send + Sync + 'static> {
    /// The inner regression pipeline.
    regressor: Pipeline<F>,
    /// Forward transform applied to `y` before fitting.
    func: fn(F) -> F,
    /// Inverse transform applied to predictions.
    inverse_func: fn(F) -> F,
}

impl<F: Float + Send + Sync + 'static> TransformedTargetRegressor<F> {
    /// Create a new `TransformedTargetRegressor`.
    ///
    /// # Parameters
    ///
    /// - `regressor` — the base regression pipeline.
    /// - `func` — forward transform applied to each element of `y` before fitting.
    /// - `inverse_func` — inverse transform applied to each prediction.
    pub fn new(regressor: Pipeline<F>, func: fn(F) -> F, inverse_func: fn(F) -> F) -> Self {
        Self {
            regressor,
            func,
            inverse_func,
        }
    }

    /// Return a reference to the inner pipeline.
    #[must_use]
    pub fn regressor(&self) -> &Pipeline<F> {
        &self.regressor
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>>
    for TransformedTargetRegressor<F>
{
    type Fitted = FittedTransformedTargetRegressor<F>;
    type Error = FerroError;

    /// Fit the regressor on transformed targets.
    ///
    /// 1. Apply `func` element-wise to `y`, producing `y_transformed`.
    /// 2. Fit the inner pipeline on `(x, y_transformed)`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::NumericalInstability`] if `func` produces NaN values.
    /// - Propagates errors from the inner pipeline's `fit`.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedTransformedTargetRegressor<F>, FerroError> {
        let y_transformed = y.mapv(self.func);

        // Check for NaN in transformed target
        if y_transformed.iter().any(|&v| v.is_nan()) {
            return Err(FerroError::NumericalInstability {
                message: "TransformedTargetRegressor: func produced NaN values in y".into(),
            });
        }

        let fitted_pipeline = self.regressor.fit(x, &y_transformed)?;

        Ok(FittedTransformedTargetRegressor {
            pipeline: fitted_pipeline,
            inverse_func: self.inverse_func,
        })
    }
}

// ---------------------------------------------------------------------------
// FittedTransformedTargetRegressor
// ---------------------------------------------------------------------------

/// A fitted transformed-target regressor.
///
/// Created by [`TransformedTargetRegressor::fit`]. Predictions are produced
/// by the inner fitted pipeline, then `inverse_func` is applied element-wise.
pub struct FittedTransformedTargetRegressor<F: Float + Send + Sync + 'static> {
    /// The fitted inner pipeline.
    pipeline:
        <Pipeline<F> as Fit<Array2<F>, Array1<F>>>::Fitted,
    /// Inverse transform to apply to predictions.
    inverse_func: fn(F) -> F,
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>>
    for FittedTransformedTargetRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict using the fitted pipeline, then apply the inverse transform.
    ///
    /// # Errors
    ///
    /// Propagates errors from the inner pipeline's `predict`.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let raw_preds = self.pipeline.predict(x)?;
        Ok(raw_preds.mapv(self.inverse_func))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ferrolearn_core::pipeline::{
        FittedPipelineEstimator, FittedPipelineTransformer, PipelineEstimator, PipelineTransformer,
    };
    use ndarray::array;

    // -- Test fixtures -------------------------------------------------------

    /// A trivial estimator that predicts the mean of the training y.
    struct MeanEstimator;

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            let mean = y.sum() / y.len() as f64;
            Ok(Box::new(FittedMeanEstimator { mean }))
        }
    }

    struct FittedMeanEstimator {
        mean: f64,
    }

    impl FittedPipelineEstimator<f64> for FittedMeanEstimator {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    /// A trivial transformer that doubles all values.
    struct Doubler;

    impl PipelineTransformer<f64> for Doubler {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedDoubler))
        }
    }

    struct FittedDoubler;

    impl FittedPipelineTransformer<f64> for FittedDoubler {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    fn make_pipeline() -> Pipeline<f64> {
        Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
    }

    fn make_pipeline_with_transformer() -> Pipeline<f64> {
        Pipeline::new()
            .transform_step("doubler", Box::new(Doubler))
            .estimator_step("mean", Box::new(MeanEstimator))
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_identity_transform() {
        // func = identity, inverse = identity → should behave exactly
        // like the raw pipeline.
        let ttr = TransformedTargetRegressor::new(make_pipeline(), |y| y, |y| y);

        let x = array![[1.0], [2.0], [3.0]];
        let y = array![10.0, 20.0, 30.0];
        let fitted = ttr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Mean of [10, 20, 30] = 20
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 20.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_doubling_transform() {
        // func doubles y → mean estimator sees [20, 40, 60] → mean=40
        // inverse_func halves predictions → 40/2 = 20
        let ttr = TransformedTargetRegressor::new(
            make_pipeline(),
            |y: f64| y * 2.0,
            |y: f64| y / 2.0,
        );

        let x = array![[1.0], [2.0], [3.0]];
        let y = array![10.0, 20.0, 30.0];
        let fitted = ttr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 20.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_log_exp_transform() {
        // func = ln, inverse = exp
        let ttr = TransformedTargetRegressor::new(
            make_pipeline(),
            |y: f64| y.ln(),
            |y: f64| y.exp(),
        );

        let x = array![[1.0], [2.0]];
        let y = array![1.0_f64.exp(), (2.0_f64).exp()]; // e^1, e^2
        let fitted = ttr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Transformed y = [1, 2], mean = 1.5
        // Prediction = exp(1.5)
        let expected = 1.5_f64.exp();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_with_pipeline_transformer() {
        let ttr = TransformedTargetRegressor::new(
            make_pipeline_with_transformer(),
            |y: f64| y,
            |y: f64| y,
        );

        let x = array![[1.0], [2.0], [3.0]];
        let y = array![10.0, 20.0, 30.0];
        let fitted = ttr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Doubler transforms X, but mean estimator only uses y → mean=20
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 20.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nan_func_error() {
        // func produces NaN
        let ttr = TransformedTargetRegressor::new(
            make_pipeline(),
            |_y: f64| f64::NAN,
            |y: f64| y,
        );

        let x = array![[1.0]];
        let y = array![1.0];
        assert!(ttr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_accessor() {
        let pipe = make_pipeline();
        let ttr = TransformedTargetRegressor::new(pipe, |y: f64| y, |y: f64| y);
        // Just verify we can access it without panicking
        let _r = ttr.regressor();
    }

    #[test]
    fn test_square_sqrt_transform() {
        // func = square, inverse = sqrt
        let ttr = TransformedTargetRegressor::new(
            make_pipeline(),
            |y: f64| y * y,
            |y: f64| y.sqrt(),
        );

        let x = array![[1.0], [2.0]];
        let y = array![3.0, 5.0];
        let fitted = ttr.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Transformed y = [9, 25], mean = 17
        // Prediction = sqrt(17)
        let expected = 17.0_f64.sqrt();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, expected, epsilon = 1e-10);
        }
    }
}
