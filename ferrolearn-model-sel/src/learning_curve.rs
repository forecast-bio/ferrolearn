//! Learning curve computation.
//!
//! [`learning_curve`] evaluates a [`Pipeline`] for increasing training set
//! sizes, recording both training and test scores at each size for every
//! cross-validation fold. This is useful for diagnosing bias/variance
//! trade-offs and deciding whether more data would improve the model.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{KFold, learning_curve};
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y: &Array1<f64>, p: &Array1<f64>) -> Result<f64, FerroError> {
//!     let d = y - p; Ok(-d.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // let result = learning_curve(&pipeline, &x, &y, &KFold::new(5),
//! //     &[0.2, 0.4, 0.6, 0.8, 1.0], neg_mse).unwrap();
//! ```

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};

use crate::cross_validation::CrossValidator;

// ---------------------------------------------------------------------------
// LearningCurveResult
// ---------------------------------------------------------------------------

/// Results from a [`learning_curve`] evaluation.
///
/// Contains the training sizes used and the per-fold train/test scores at
/// each size.
#[derive(Debug, Clone)]
pub struct LearningCurveResult {
    /// The absolute number of training samples used at each size.
    pub train_sizes: Vec<usize>,
    /// Training scores with shape `(n_sizes, n_folds)`.
    pub train_scores: Array2<f64>,
    /// Test scores with shape `(n_sizes, n_folds)`.
    pub test_scores: Array2<f64>,
}

// ---------------------------------------------------------------------------
// learning_curve
// ---------------------------------------------------------------------------

/// Compute train and test scores for varying training set sizes.
///
/// For each value in `train_sizes`:
///
/// 1. Determine the absolute number of training samples. Values in `(0, 1]`
///    are treated as fractions of the full training fold; values `> 1` are
///    treated as absolute counts (truncated to `usize`).
/// 2. For each cross-validation fold, take the first `size` samples from the
///    training fold, fit the pipeline, then score on both the training
///    subset and the full test fold.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] that produces fold indices.
/// - `train_sizes` — Fractions (`(0, 1]`) or absolute sample counts for the
///   training subset at each point on the curve.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`.
///
/// # Returns
///
/// A [`LearningCurveResult`] with train_sizes, train_scores, and test_scores.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `train_sizes` is empty or any value
///   is non-positive.
/// - Propagates any error from fold splitting, model fitting, predicting, or
///   scoring.
pub fn learning_curve(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    train_sizes: &[f64],
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
) -> Result<LearningCurveResult, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "learning_curve: y length must equal x number of rows".into(),
        });
    }

    if train_sizes.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "train_sizes".into(),
            reason: "must not be empty".into(),
        });
    }

    for (i, &s) in train_sizes.iter().enumerate() {
        if s <= 0.0 || !s.is_finite() {
            return Err(FerroError::InvalidParameter {
                name: "train_sizes".into(),
                reason: format!("entry {i} is {s}; must be a positive, finite number"),
            });
        }
    }

    let folds = cv.fold_indices(n_samples)?;
    let n_folds = folds.len();
    let n_features = x.ncols();
    let n_sizes = train_sizes.len();

    // Pre-compute the absolute sizes based on the first fold's training set
    // size as the reference (all folds should have approximately the same
    // training size, so the first one is representative).
    let reference_train_len = folds[0].0.len();

    let abs_sizes: Vec<usize> = train_sizes
        .iter()
        .map(|&s| {
            if s <= 1.0 {
                // Fraction of training fold.
                ((s * reference_train_len as f64).ceil() as usize)
                    .max(1)
                    .min(reference_train_len)
            } else {
                // Absolute count.
                (s as usize).max(1).min(reference_train_len)
            }
        })
        .collect();

    let mut train_scores_data = Vec::with_capacity(n_sizes * n_folds);
    let mut test_scores_data = Vec::with_capacity(n_sizes * n_folds);

    for &size in &abs_sizes {
        for (train_idx, test_idx) in &folds {
            let effective_size = size.min(train_idx.len());

            // Build training subset (first `effective_size` samples of the fold).
            let sub_train_idx = &train_idx[..effective_size];

            let mut x_train_data = Vec::with_capacity(effective_size * n_features);
            for &i in sub_train_idx {
                x_train_data.extend(x.row(i).iter().copied());
            }
            let x_train = Array2::from_shape_vec((effective_size, n_features), x_train_data)
                .map_err(|e| FerroError::InvalidParameter {
                    name: "x_train".into(),
                    reason: e.to_string(),
                })?;
            let y_train: Array1<f64> = sub_train_idx.iter().map(|&i| y[i]).collect();

            // Build test subset.
            let n_test = test_idx.len();
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
        Array2::from_shape_vec((n_sizes, n_folds), train_scores_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "train_scores".into(),
                reason: e.to_string(),
            }
        })?;
    let test_scores =
        Array2::from_shape_vec((n_sizes, n_folds), test_scores_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "test_scores".into(),
                reason: e.to_string(),
            }
        })?;

    Ok(LearningCurveResult {
        train_sizes: abs_sizes,
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
    use ferrolearn_core::pipeline::{
        FittedPipelineEstimator, FittedPipelineTransformer, Pipeline, PipelineEstimator,
        PipelineTransformer,
    };
    use ndarray::{Array1, Array2};

    use crate::KFold;

    // -- Test fixtures -------------------------------------------------------

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

    struct IdentityTransformer;

    impl PipelineTransformer<f64> for IdentityTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedIdentity))
        }
    }

    struct FittedIdentity;

    impl FittedPipelineTransformer<f64> for FittedIdentity {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.clone())
        }
    }

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_learning_curve_basic() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 3));
        let y = Array1::<f64>::from_elem(30, 5.0);
        let kf = KFold::new(3);

        let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 1.0], neg_mse).unwrap();

        assert_eq!(result.train_sizes.len(), 2);
        assert_eq!(result.train_scores.shape(), &[2, 3]);
        assert_eq!(result.test_scores.shape(), &[2, 3]);
    }

    #[test]
    fn test_learning_curve_constant_target_scores_near_zero() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 3.0);
        let kf = KFold::new(3);

        let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 1.0], neg_mse).unwrap();

        // neg_mse of perfect predictions is 0.
        for &s in result.train_scores.iter() {
            assert!(s.abs() < 1e-10, "expected ~0 train score, got {s}");
        }
        for &s in result.test_scores.iter() {
            assert!(s.abs() < 1e-10, "expected ~0 test score, got {s}");
        }
    }

    #[test]
    fn test_learning_curve_with_transformer() {
        let pipeline = Pipeline::new()
            .transform_step("id", Box::new(IdentityTransformer))
            .estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let kf = KFold::new(3);

        let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 1.0], neg_mse).unwrap();
        assert_eq!(result.train_sizes.len(), 2);
    }

    #[test]
    fn test_learning_curve_absolute_sizes() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let kf = KFold::new(3);

        // Absolute sizes > 1 are treated as sample counts.
        let result = learning_curve(&pipeline, &x, &y, &kf, &[5.0, 10.0, 20.0], neg_mse).unwrap();

        assert_eq!(result.train_sizes.len(), 3);
        assert_eq!(result.train_sizes[0], 5);
        assert_eq!(result.train_sizes[1], 10);
        assert_eq!(result.train_sizes[2], 20);
    }

    #[test]
    fn test_learning_curve_fraction_sizes() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let kf = KFold::new(3);

        // With 30 samples and 3 folds, each fold has ~20 training samples.
        let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 1.0], neg_mse).unwrap();
        // 0.5 * 20 = 10, 1.0 * 20 = 20
        assert_eq!(result.train_sizes[0], 10);
        assert_eq!(result.train_sizes[1], 20);
    }

    #[test]
    fn test_learning_curve_empty_sizes_error() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        let kf = KFold::new(3);
        assert!(learning_curve(&pipeline, &x, &y, &kf, &[], neg_mse).is_err());
    }

    #[test]
    fn test_learning_curve_negative_size_error() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        let kf = KFold::new(3);
        assert!(learning_curve(&pipeline, &x, &y, &kf, &[-0.5], neg_mse).is_err());
    }

    #[test]
    fn test_learning_curve_zero_size_error() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        let kf = KFold::new(3);
        assert!(learning_curve(&pipeline, &x, &y, &kf, &[0.0], neg_mse).is_err());
    }

    #[test]
    fn test_learning_curve_shape_mismatch() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(25);
        let kf = KFold::new(3);
        assert!(learning_curve(&pipeline, &x, &y, &kf, &[0.5], neg_mse).is_err());
    }

    #[test]
    fn test_learning_curve_scores_are_finite() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let y_data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::from_vec(y_data);
        let kf = KFold::new(3);

        let result = learning_curve(&pipeline, &x, &y, &kf, &[0.3, 0.6, 1.0], neg_mse).unwrap();

        for &s in result.train_scores.iter() {
            assert!(s.is_finite(), "train score should be finite, got {s}");
        }
        for &s in result.test_scores.iter() {
            assert!(s.is_finite(), "test score should be finite, got {s}");
        }
    }
}
