//! Multi-output (multi-target) meta-estimators.
//!
//! This module provides meta-estimators that handle multi-target problems
//! by fitting one estimator per target column:
//!
//! - [`MultiOutputClassifier`] — wraps a classifier to support multi-target
//!   classification. Fits one classifier per target column.
//! - [`MultiOutputRegressor`] — wraps a regressor to support multi-target
//!   regression. Fits one regressor per target column.
//!
//! Both estimators accept a factory closure that produces fresh pipelines
//! for each target, following the same pattern used by
//! [`GridSearchCV`](crate::GridSearchCV) and
//! [`CalibratedClassifierCV`](crate::CalibratedClassifierCV).
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::multioutput::MultiOutputRegressor;
//! use ferrolearn_core::pipeline::Pipeline;
//! use ndarray::{Array1, Array2};
//!
//! let mor = MultiOutputRegressor::new(Box::new(|| Pipeline::<f64>::new()));
//! ```

use ferrolearn_core::pipeline::{FittedPipeline, Pipeline};
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Factory type alias
// ---------------------------------------------------------------------------

/// A boxed closure that creates a fresh [`Pipeline`] for a single-target
/// sub-problem.
type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ===========================================================================
// MultiOutputClassifier
// ===========================================================================

/// Multi-output classification meta-estimator.
///
/// For a multi-target classification problem with T target columns, fits
/// one classifier per target. Each classifier treats its target column as
/// the label array.
///
/// The target matrix `Y` has shape `(n_samples, n_targets)` and contains
/// integer class labels encoded as `f64` (e.g., `0.0`, `1.0`, `2.0`).
///
/// # Factory Closure
///
/// The `make_pipeline` closure is called T times during fitting. Each
/// invocation must return a fresh, unfitted [`Pipeline`].
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::multioutput::MultiOutputClassifier;
/// use ferrolearn_core::pipeline::Pipeline;
///
/// let moc = MultiOutputClassifier::new(Box::new(|| Pipeline::<f64>::new()));
/// ```
pub struct MultiOutputClassifier {
    /// Factory that creates a fresh pipeline for each target column.
    make_pipeline: PipelineFactory,
}

/// A fitted [`MultiOutputClassifier`] containing one classifier per target.
///
/// Implements [`Predict`] to produce multi-target predictions as an
/// `Array2<f64>` of shape `(n_samples, n_targets)`.
pub struct FittedMultiOutputClassifier {
    /// One fitted pipeline per target column.
    estimators: Vec<FittedPipeline<f64>>,
    /// Number of target columns.
    n_targets: usize,
}

impl MultiOutputClassifier {
    /// Create a new [`MultiOutputClassifier`].
    ///
    /// # Parameters
    ///
    /// - `make_pipeline` — a closure that returns a fresh [`Pipeline`]
    ///   suitable for single-target classification. Called once per target.
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self { make_pipeline }
    }

    /// Fit the multi-output classifier.
    ///
    /// For each target column in `y`, creates and fits a separate pipeline.
    ///
    /// # Parameters
    ///
    /// - `x` — feature matrix of shape `(n_samples, n_features)`.
    /// - `y` — target matrix of shape `(n_samples, n_targets)` with class
    ///   labels encoded as `f64`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if:
    /// - `x` and `y` have mismatched row counts.
    /// - `y` has zero columns.
    /// - Any per-target classifier fails to fit.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<FittedMultiOutputClassifier, FerroError> {
        let n_samples = x.nrows();
        let n_targets = y.ncols();

        if y.nrows() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, n_targets],
                actual: vec![y.nrows(), n_targets],
                context: "MultiOutputClassifier::fit: y rows must equal x rows".into(),
            });
        }

        if n_targets == 0 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "target matrix must have at least one column".into(),
            });
        }

        let mut estimators = Vec::with_capacity(n_targets);

        for t in 0..n_targets {
            let y_col = y.column(t).to_owned();
            let pipeline = (self.make_pipeline)();
            let fitted = pipeline.fit(x, &y_col)?;
            estimators.push(fitted);
        }

        Ok(FittedMultiOutputClassifier {
            estimators,
            n_targets,
        })
    }
}

impl FittedMultiOutputClassifier {
    /// Return the number of target columns.
    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Return the number of fitted estimators (equal to `n_targets`).
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

impl Predict<Array2<f64>> for FittedMultiOutputClassifier {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Predict multi-target class labels.
    ///
    /// Returns an `Array2<f64>` of shape `(n_samples, n_targets)` where
    /// each column contains the predictions from the corresponding
    /// per-target classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if any per-target classifier fails to predict.
    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, self.n_targets));

        for (t, est) in self.estimators.iter().enumerate() {
            let preds = est.predict(x)?;
            for i in 0..n_samples {
                result[[i, t]] = preds[i];
            }
        }

        Ok(result)
    }
}

// ===========================================================================
// MultiOutputRegressor
// ===========================================================================

/// Multi-output regression meta-estimator.
///
/// For a multi-target regression problem with T target columns, fits
/// one regressor per target. Each regressor treats its target column as
/// the label array.
///
/// # Factory Closure
///
/// The `make_pipeline` closure is called T times during fitting. Each
/// invocation must return a fresh, unfitted [`Pipeline`].
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::multioutput::MultiOutputRegressor;
/// use ferrolearn_core::pipeline::Pipeline;
///
/// let mor = MultiOutputRegressor::new(Box::new(|| Pipeline::<f64>::new()));
/// ```
pub struct MultiOutputRegressor {
    /// Factory that creates a fresh pipeline for each target column.
    make_pipeline: PipelineFactory,
}

/// A fitted [`MultiOutputRegressor`] containing one regressor per target.
///
/// Implements [`Predict`] to produce multi-target predictions as an
/// `Array2<f64>` of shape `(n_samples, n_targets)`.
pub struct FittedMultiOutputRegressor {
    /// One fitted pipeline per target column.
    estimators: Vec<FittedPipeline<f64>>,
    /// Number of target columns.
    n_targets: usize,
}

impl MultiOutputRegressor {
    /// Create a new [`MultiOutputRegressor`].
    ///
    /// # Parameters
    ///
    /// - `make_pipeline` — a closure that returns a fresh [`Pipeline`]
    ///   suitable for single-target regression. Called once per target.
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self { make_pipeline }
    }

    /// Fit the multi-output regressor.
    ///
    /// For each target column in `y`, creates and fits a separate pipeline.
    ///
    /// # Parameters
    ///
    /// - `x` — feature matrix of shape `(n_samples, n_features)`.
    /// - `y` — target matrix of shape `(n_samples, n_targets)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if:
    /// - `x` and `y` have mismatched row counts.
    /// - `y` has zero columns.
    /// - Any per-target regressor fails to fit.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<FittedMultiOutputRegressor, FerroError> {
        let n_samples = x.nrows();
        let n_targets = y.ncols();

        if y.nrows() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, n_targets],
                actual: vec![y.nrows(), n_targets],
                context: "MultiOutputRegressor::fit: y rows must equal x rows".into(),
            });
        }

        if n_targets == 0 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "target matrix must have at least one column".into(),
            });
        }

        let mut estimators = Vec::with_capacity(n_targets);

        for t in 0..n_targets {
            let y_col = y.column(t).to_owned();
            let pipeline = (self.make_pipeline)();
            let fitted = pipeline.fit(x, &y_col)?;
            estimators.push(fitted);
        }

        Ok(FittedMultiOutputRegressor {
            estimators,
            n_targets,
        })
    }
}

impl FittedMultiOutputRegressor {
    /// Return the number of target columns.
    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Return the number of fitted estimators (equal to `n_targets`).
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

impl Predict<Array2<f64>> for FittedMultiOutputRegressor {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Predict multi-target regression values.
    ///
    /// Returns an `Array2<f64>` of shape `(n_samples, n_targets)` where
    /// each column contains the predictions from the corresponding
    /// per-target regressor.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if any per-target regressor fails to predict.
    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_samples = x.nrows();
        let mut result = Array2::<f64>::zeros((n_samples, self.n_targets));

        for (t, est) in self.estimators.iter().enumerate() {
            let preds = est.predict(x)?;
            for i in 0..n_samples {
                result[[i, t]] = preds[i];
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
    use ndarray::Array1;

    // -----------------------------------------------------------------------
    // Test fixture: estimator that learns the training mean
    // -----------------------------------------------------------------------

    /// An estimator that predicts the mean of the training targets.
    struct MeanEstimator;

    struct FittedMeanEst {
        mean: f64,
    }

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedMeanEst {
                mean: y.mean().unwrap_or(0.0),
            }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedMeanEst {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    // -----------------------------------------------------------------------
    // Test fixture: estimator that uses the first feature as the prediction
    // -----------------------------------------------------------------------

    /// An estimator that predicts the sum of each row.
    struct SumEstimator;

    struct FittedSumEst;

    impl PipelineEstimator<f64> for SumEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedSumEst))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedSumEst {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let sums: Vec<f64> = x.rows().into_iter().map(|row| row.sum()).collect();
            Ok(Array1::from_vec(sums))
        }
    }

    /// Create a pipeline factory that produces mean estimators.
    fn make_mean_factory() -> PipelineFactory {
        Box::new(|| Pipeline::new().estimator_step("est", Box::new(MeanEstimator)))
    }

    /// Create a pipeline factory that produces sum estimators.
    fn make_sum_factory() -> PipelineFactory {
        Box::new(|| Pipeline::new().estimator_step("est", Box::new(SumEstimator)))
    }

    // -----------------------------------------------------------------------
    // MultiOutputClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_moc_fit_predict_two_targets() {
        // Target 1: all 0.0, Target 2: all 1.0
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y =
            Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();

        let moc = MultiOutputClassifier::new(make_mean_factory());
        let fitted = moc.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);
        assert_eq!(fitted.n_estimators(), 2);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.nrows(), 4);
        assert_eq!(preds.ncols(), 2);

        // MeanEstimator predicts the mean of training labels.
        // Target 0 mean = 0.0, Target 1 mean = 1.0.
        for i in 0..4 {
            assert!((preds[[i, 0]] - 0.0).abs() < 1e-10);
            assert!((preds[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_moc_shape_mismatch() {
        let x = Array2::<f64>::zeros((10, 2));
        let y = Array2::<f64>::zeros((8, 2)); // wrong row count

        let moc = MultiOutputClassifier::new(make_mean_factory());
        assert!(moc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_moc_zero_targets() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array2::<f64>::zeros((5, 0)); // no targets

        let moc = MultiOutputClassifier::new(make_mean_factory());
        assert!(moc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_moc_single_target() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array2::from_shape_vec((4, 1), vec![10.0, 10.0, 10.0, 10.0]).unwrap();

        let moc = MultiOutputClassifier::new(make_mean_factory());
        let fitted = moc.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_targets(), 1);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.ncols(), 1);
        for i in 0..4 {
            assert!((preds[[i, 0]] - 10.0).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // MultiOutputRegressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mor_fit_predict_two_targets() {
        // Each target column should get its own mean predictor.
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y =
            Array2::from_shape_vec((4, 2), vec![2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0]).unwrap();

        let mor = MultiOutputRegressor::new(make_mean_factory());
        let fitted = mor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_targets(), 2);
        assert_eq!(fitted.n_estimators(), 2);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.nrows(), 4);
        assert_eq!(preds.ncols(), 2);

        // Target 0 mean = 2.0, Target 1 mean = 4.0.
        for i in 0..4 {
            assert!((preds[[i, 0]] - 2.0).abs() < 1e-10);
            assert!((preds[[i, 1]] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mor_with_sum_estimator() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        let mor = MultiOutputRegressor::new(make_sum_factory());
        let fitted = mor.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        // SumEstimator sums each row: [3, 7, 11] for both targets.
        for t in 0..2 {
            assert!((preds[[0, t]] - 3.0).abs() < 1e-10);
            assert!((preds[[1, t]] - 7.0).abs() < 1e-10);
            assert!((preds[[2, t]] - 11.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mor_shape_mismatch() {
        let x = Array2::<f64>::zeros((10, 2));
        let y = Array2::<f64>::zeros((8, 2));

        let mor = MultiOutputRegressor::new(make_mean_factory());
        assert!(mor.fit(&x, &y).is_err());
    }

    #[test]
    fn test_mor_zero_targets() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array2::<f64>::zeros((5, 0));

        let mor = MultiOutputRegressor::new(make_mean_factory());
        assert!(mor.fit(&x, &y).is_err());
    }

    #[test]
    fn test_mor_single_target() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();

        let mor = MultiOutputRegressor::new(make_mean_factory());
        let fitted = mor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_targets(), 1);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.ncols(), 1);
        for i in 0..3 {
            assert!((preds[[i, 0]] - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mor_three_targets() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

        let mor = MultiOutputRegressor::new(make_mean_factory());
        let fitted = mor.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_targets(), 3);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.ncols(), 3);

        // Each target column has constant values: 1.0, 2.0, 3.0 respectively.
        for i in 0..2 {
            assert!((preds[[i, 0]] - 1.0).abs() < 1e-10);
            assert!((preds[[i, 1]] - 2.0).abs() < 1e-10);
            assert!((preds[[i, 2]] - 3.0).abs() < 1e-10);
        }
    }
}
