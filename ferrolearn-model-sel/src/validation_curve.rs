//! Validation curve computation.
//!
//! [`validation_curve`] evaluates how a model's training and test scores
//! change as a single hyperparameter is varied. This is useful for
//! diagnosing overfitting and underfitting for specific parameters.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{KFold, validation_curve};
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y: &Array1<f64>, p: &Array1<f64>) -> Result<f64, FerroError> {
//!     let d = y - p; Ok(-d.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // let result = validation_curve(&x, &y, &KFold::new(5),
//! //     &[0.01, 0.1, 1.0, 10.0],
//! //     |alpha| make_pipeline(alpha),
//! //     neg_mse).unwrap();
//! ```

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};

use crate::cross_validation::CrossValidator;

// ---------------------------------------------------------------------------
// ValidationCurveResult
// ---------------------------------------------------------------------------

/// Results from a [`validation_curve`] evaluation.
///
/// Contains the parameter values tested and per-fold train/test scores at
/// each value.
#[derive(Debug, Clone)]
pub struct ValidationCurveResult {
    /// The parameter values that were evaluated.
    pub param_values: Vec<f64>,
    /// Training scores with shape `(n_params, n_folds)`.
    pub train_scores: Array2<f64>,
    /// Test scores with shape `(n_params, n_folds)`.
    pub test_scores: Array2<f64>,
}

// ---------------------------------------------------------------------------
// validation_curve
// ---------------------------------------------------------------------------

/// Compute train and test scores for varying hyperparameter values.
///
/// For each value in `param_values`, a fresh [`Pipeline`] is constructed via
/// `make_pipeline`, and cross-validation is run to measure both training and
/// test performance.
///
/// # Parameters
///
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] that produces fold indices.
/// - `param_values` — The hyperparameter values to evaluate.
/// - `make_pipeline` — A closure that creates an unfitted [`Pipeline`] for a
///   given parameter value.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`.
///
/// # Returns
///
/// A [`ValidationCurveResult`] with param_values, train_scores, and test_scores.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `param_values` is empty.
/// - Propagates any error from fold splitting, pipeline construction, model
///   fitting, predicting, or scoring.
pub fn validation_curve(
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    param_values: &[f64],
    make_pipeline: impl Fn(f64) -> Pipeline,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
) -> Result<ValidationCurveResult, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "validation_curve: y length must equal x number of rows".into(),
        });
    }

    if param_values.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "param_values".into(),
            reason: "must not be empty".into(),
        });
    }

    let folds = cv.fold_indices(n_samples)?;
    let n_folds = folds.len();
    let n_features = x.ncols();
    let n_params = param_values.len();

    let mut train_scores_data = Vec::with_capacity(n_params * n_folds);
    let mut test_scores_data = Vec::with_capacity(n_params * n_folds);

    for &param in param_values {
        let pipeline = make_pipeline(param);

        for (train_idx, test_idx) in &folds {
            let n_train = train_idx.len();
            let n_test = test_idx.len();

            // Build training subset.
            let mut x_train_data = Vec::with_capacity(n_train * n_features);
            for &i in train_idx {
                x_train_data.extend(x.row(i).iter().copied());
            }
            let x_train =
                Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "x_train".into(),
                        reason: e.to_string(),
                    }
                })?;
            let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();

            // Build test subset.
            let mut x_test_data = Vec::with_capacity(n_test * n_features);
            for &i in test_idx {
                x_test_data.extend(x.row(i).iter().copied());
            }
            let x_test =
                Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "x_test".into(),
                        reason: e.to_string(),
                    }
                })?;
            let y_test: Array1<f64> = test_idx.iter().map(|&i| y[i]).collect();

            // Fit and score.
            let fitted = pipeline.fit(&x_train, &y_train)?;

            let y_train_pred = fitted.predict(&x_train)?;
            let train_score = scoring(&y_train, &y_train_pred)?;
            train_scores_data.push(train_score);

            let y_test_pred = fitted.predict(&x_test)?;
            let test_score = scoring(&y_test, &y_test_pred)?;
            test_scores_data.push(test_score);
        }
    }

    let train_scores =
        Array2::from_shape_vec((n_params, n_folds), train_scores_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "train_scores".into(),
                reason: e.to_string(),
            }
        })?;
    let test_scores =
        Array2::from_shape_vec((n_params, n_folds), test_scores_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "test_scores".into(),
                reason: e.to_string(),
            }
        })?;

    Ok(ValidationCurveResult {
        param_values: param_values.to_vec(),
        train_scores,
        test_scores,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
    use ndarray::{Array1, Array2};

    use crate::KFold;

    // -- Test fixtures -------------------------------------------------------

    /// An estimator that predicts a constant value (the parameter).
    struct ConstantEstimator {
        value: f64,
    }

    struct FittedConstant {
        value: f64,
    }

    impl PipelineEstimator<f64> for ConstantEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedConstant { value: self.value }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedConstant {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.value))
        }
    }

    struct MeanEstimator;
    struct FittedMean {
        mean: f64,
    }

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedMean {
                mean: y.mean().unwrap_or(0.0),
            }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedMean {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_validation_curve_basic() {
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let kf = KFold::new(3);

        let result = validation_curve(
            &x,
            &y,
            &kf,
            &[0.0, 1.0, 2.0],
            |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
            neg_mse,
        )
        .unwrap();

        assert_eq!(result.param_values.len(), 3);
        assert_eq!(result.train_scores.shape(), &[3, 3]);
        assert_eq!(result.test_scores.shape(), &[3, 3]);
    }

    #[test]
    fn test_validation_curve_best_at_target() {
        // y = 1.0; constant estimator predicting 1.0 should have the best score.
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let kf = KFold::new(3);

        let result = validation_curve(
            &x,
            &y,
            &kf,
            &[0.0, 1.0, 5.0],
            |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
            neg_mse,
        )
        .unwrap();

        // The parameter value 1.0 (index 1) should have the best (closest to 0)
        // test scores.
        let mean_test_at_1 =
            result.test_scores.row(1).iter().sum::<f64>() / result.test_scores.ncols() as f64;
        assert!(
            mean_test_at_1.abs() < 1e-10,
            "expected ~0 for param=1.0, got {mean_test_at_1}"
        );
    }

    #[test]
    fn test_validation_curve_mean_estimator_ignores_param() {
        // MeanEstimator ignores the parameter, so all param_values produce the
        // same scores.
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 3.0);
        let kf = KFold::new(3);

        let result = validation_curve(
            &x,
            &y,
            &kf,
            &[0.1, 1.0, 10.0],
            |_| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator)),
            neg_mse,
        )
        .unwrap();

        // All rows should have approximately the same scores.
        for row_idx in 0..result.test_scores.nrows() {
            for &s in result.test_scores.row(row_idx).iter() {
                assert!(s.abs() < 1e-10, "expected ~0, got {s}");
            }
        }
    }

    #[test]
    fn test_validation_curve_empty_params_error() {
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        let kf = KFold::new(3);
        assert!(
            validation_curve(
                &x,
                &y,
                &kf,
                &[],
                |_| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator)),
                neg_mse,
            )
            .is_err()
        );
    }

    #[test]
    fn test_validation_curve_shape_mismatch() {
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(25);
        let kf = KFold::new(3);
        assert!(
            validation_curve(
                &x,
                &y,
                &kf,
                &[1.0],
                |_| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator)),
                neg_mse,
            )
            .is_err()
        );
    }

    #[test]
    fn test_validation_curve_scores_finite() {
        let y_data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::from_vec(y_data);
        let kf = KFold::new(3);

        let result = validation_curve(
            &x,
            &y,
            &kf,
            &[0.0, 5.0, 15.0],
            |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
            neg_mse,
        )
        .unwrap();

        for &s in result.train_scores.iter() {
            assert!(s.is_finite(), "train score should be finite, got {s}");
        }
        for &s in result.test_scores.iter() {
            assert!(s.is_finite(), "test score should be finite, got {s}");
        }
    }

    #[test]
    fn test_validation_curve_single_param() {
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 2.0);
        let kf = KFold::new(3);

        let result = validation_curve(
            &x,
            &y,
            &kf,
            &[2.0],
            |val| Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val })),
            neg_mse,
        )
        .unwrap();

        assert_eq!(result.param_values.len(), 1);
        assert_eq!(result.train_scores.shape(), &[1, 3]);
        for &s in result.test_scores.iter() {
            assert!(s.abs() < 1e-10, "expected ~0, got {s}");
        }
    }
}
