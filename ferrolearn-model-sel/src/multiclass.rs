//! Multi-class classification strategies.
//!
//! This module provides meta-estimators that decompose multi-class
//! classification problems into collections of binary classifiers:
//!
//! - [`OneVsRestClassifier`] — trains one binary classifier per class
//!   (class k vs. all other classes). Also known as One-vs-All (OvA).
//! - [`OneVsOneClassifier`] — trains one binary classifier per class pair,
//!   resulting in `K*(K-1)/2` classifiers for K classes.
//!
//! Both estimators accept a factory closure that produces fresh pipelines
//! for each binary sub-problem, following the same pattern used by
//! [`GridSearchCV`](crate::GridSearchCV) and
//! [`CalibratedClassifierCV`](crate::CalibratedClassifierCV).
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::multiclass::OneVsRestClassifier;
//! use ferrolearn_core::pipeline::Pipeline;
//! use ndarray::{Array1, Array2};
//!
//! // Factory that returns a fresh binary pipeline each time.
//! let factory = Box::new(|| {
//!     // In practice, build a real pipeline here.
//!     Pipeline::<f64>::new()
//! });
//!
//! let ovr = OneVsRestClassifier::new(factory);
//! ```

use ferrolearn_core::pipeline::{FittedPipeline, Pipeline};
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Factory type alias
// ---------------------------------------------------------------------------

/// A boxed closure that creates a fresh [`Pipeline`] for a binary sub-problem.
///
/// The factory is called once per binary classifier that needs to be trained.
type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Helper: select rows by index
// ---------------------------------------------------------------------------

/// Select rows from a 2D array by index.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let n_rows = indices.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &i in indices {
        data.extend(x.row(i).iter().copied());
    }
    Array2::from_shape_vec((n_rows, n_cols), data)
        .expect("select_rows: shape should always be valid")
}

/// Collect the sorted unique classes from a label array.
fn unique_classes(y: &Array1<usize>) -> Vec<usize> {
    let mut classes: Vec<usize> = y.iter().copied().collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

// ===========================================================================
// OneVsRestClassifier
// ===========================================================================

/// One-vs-Rest (OvR) multi-class classification strategy.
///
/// For a K-class problem, trains K binary classifiers. The k-th classifier
/// is trained on a binary problem where class k is the positive class and
/// all other classes form the negative class.
///
/// At prediction time, each classifier produces a score and the class
/// with the highest score is chosen.
///
/// # Factory Closure
///
/// The `make_pipeline` closure is called K times during fitting. Each
/// invocation must return a fresh, unfitted [`Pipeline`] configured
/// for binary classification.
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::multiclass::OneVsRestClassifier;
/// use ferrolearn_core::pipeline::Pipeline;
/// use ndarray::{Array1, Array2};
///
/// let ovr = OneVsRestClassifier::new(Box::new(|| Pipeline::<f64>::new()));
/// ```
pub struct OneVsRestClassifier {
    /// Factory that creates a fresh pipeline for each binary sub-problem.
    make_pipeline: PipelineFactory,
}

/// A fitted [`OneVsRestClassifier`] containing K binary classifiers.
///
/// Implements [`Predict`] to produce multi-class predictions by selecting
/// the class whose binary classifier produces the highest score.
pub struct FittedOneVsRestClassifier {
    /// One fitted pipeline per class, in the same order as `classes`.
    estimators: Vec<FittedPipeline<f64>>,
    /// The sorted unique class labels.
    classes: Vec<usize>,
}

impl OneVsRestClassifier {
    /// Create a new [`OneVsRestClassifier`].
    ///
    /// # Parameters
    ///
    /// - `make_pipeline` — a closure that returns a fresh [`Pipeline`]
    ///   suitable for binary classification. Called once per class.
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self { make_pipeline }
    }

    /// Fit the one-vs-rest classifier.
    ///
    /// For each unique class in `y`, creates a binary labeling where
    /// that class is positive (`1.0`) and all others are negative (`0.0`),
    /// then fits a fresh pipeline on the binary problem.
    ///
    /// # Parameters
    ///
    /// - `x` — feature matrix of shape `(n_samples, n_features)`.
    /// - `y` — class labels of length `n_samples`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if:
    /// - `x` and `y` have mismatched lengths.
    /// - Fewer than 2 classes are present.
    /// - The pipeline factory produces a pipeline that lacks an estimator.
    /// - Any binary classifier fails to fit.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedOneVsRestClassifier, FerroError> {
        let n_samples = x.nrows();

        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "OneVsRestClassifier::fit: y length must equal x rows".into(),
            });
        }

        let classes = unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "need at least 2 classes for multi-class classification, got {}",
                    classes.len()
                ),
            });
        }

        let mut estimators = Vec::with_capacity(classes.len());

        for &cls in &classes {
            // Create binary labels: 1.0 for class `cls`, 0.0 otherwise.
            let y_binary = Array1::from_vec(
                y.iter()
                    .map(|&l| if l == cls { 1.0 } else { 0.0 })
                    .collect(),
            );

            let pipeline = (self.make_pipeline)();
            let fitted = pipeline.fit(x, &y_binary)?;
            estimators.push(fitted);
        }

        Ok(FittedOneVsRestClassifier {
            estimators,
            classes,
        })
    }
}

impl FittedOneVsRestClassifier {
    /// Return the unique class labels, sorted in ascending order.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Return the number of binary classifiers (equal to the number of classes).
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Predict decision scores for each class.
    ///
    /// Returns an `Array2<f64>` of shape `(n_samples, n_classes)` where each
    /// column contains the scores from the corresponding binary classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if any binary classifier fails to predict.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut scores = Array2::<f64>::zeros((n_samples, n_classes));

        for (k, est) in self.estimators.iter().enumerate() {
            let preds = est.predict(x)?;
            for i in 0..n_samples {
                scores[[i, k]] = preds[i];
            }
        }

        Ok(scores)
    }
}

impl Predict<Array2<f64>> for FittedOneVsRestClassifier {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels by selecting the class whose binary classifier
    /// produces the highest score.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if any binary classifier fails to predict.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let scores = self.decision_function(x)?;
        let n_samples = x.nrows();

        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row = scores.row(i);
            let best_k = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);
            predictions.push(self.classes[best_k]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

// ===========================================================================
// OneVsOneClassifier
// ===========================================================================

/// One-vs-One (OvO) multi-class classification strategy.
///
/// For a K-class problem, trains `K*(K-1)/2` binary classifiers — one for
/// each pair of classes. At prediction time, each classifier votes for one
/// of its two classes, and the class with the most votes wins.
///
/// # Factory Closure
///
/// The `make_pipeline` closure is called `K*(K-1)/2` times during fitting.
/// Each invocation must return a fresh, unfitted [`Pipeline`].
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::multiclass::OneVsOneClassifier;
/// use ferrolearn_core::pipeline::Pipeline;
/// use ndarray::{Array1, Array2};
///
/// let ovo = OneVsOneClassifier::new(Box::new(|| Pipeline::<f64>::new()));
/// ```
pub struct OneVsOneClassifier {
    /// Factory that creates a fresh pipeline for each binary sub-problem.
    make_pipeline: PipelineFactory,
}

/// A fitted [`OneVsOneClassifier`] containing `K*(K-1)/2` pairwise classifiers.
///
/// Implements [`Predict`] using majority voting across all pairwise classifiers.
pub struct FittedOneVsOneClassifier {
    /// Fitted pairwise classifiers, stored alongside their (class_i, class_j) pair.
    estimators: Vec<(usize, usize, FittedPipeline<f64>)>,
    /// The sorted unique class labels.
    classes: Vec<usize>,
}

impl OneVsOneClassifier {
    /// Create a new [`OneVsOneClassifier`].
    ///
    /// # Parameters
    ///
    /// - `make_pipeline` — a closure that returns a fresh [`Pipeline`]
    ///   suitable for binary classification. Called once per class pair.
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self { make_pipeline }
    }

    /// Fit the one-vs-one classifier.
    ///
    /// For each unique pair of classes `(i, j)`, filters the dataset to
    /// only samples belonging to either class, labels class `i` as `1.0`
    /// and class `j` as `0.0`, then fits a fresh pipeline.
    ///
    /// # Parameters
    ///
    /// - `x` — feature matrix of shape `(n_samples, n_features)`.
    /// - `y` — class labels of length `n_samples`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if:
    /// - `x` and `y` have mismatched lengths.
    /// - Fewer than 2 classes are present.
    /// - Any binary classifier fails to fit.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedOneVsOneClassifier, FerroError> {
        let n_samples = x.nrows();

        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "OneVsOneClassifier::fit: y length must equal x rows".into(),
            });
        }

        let classes = unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "need at least 2 classes for multi-class classification, got {}",
                    classes.len()
                ),
            });
        }

        let n_pairs = classes.len() * (classes.len() - 1) / 2;
        let mut estimators = Vec::with_capacity(n_pairs);

        for (idx_i, &cls_i) in classes.iter().enumerate() {
            for &cls_j in &classes[idx_i + 1..] {
                // Filter to samples belonging to cls_i or cls_j.
                let indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &l)| {
                        if l == cls_i || l == cls_j {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                let x_pair = select_rows(x, &indices);
                let y_pair = Array1::from_vec(
                    indices
                        .iter()
                        .map(|&i| if y[i] == cls_i { 1.0 } else { 0.0 })
                        .collect(),
                );

                let pipeline = (self.make_pipeline)();
                let fitted = pipeline.fit(&x_pair, &y_pair)?;
                estimators.push((cls_i, cls_j, fitted));
            }
        }

        Ok(FittedOneVsOneClassifier {
            estimators,
            classes,
        })
    }
}

impl FittedOneVsOneClassifier {
    /// Return the unique class labels, sorted in ascending order.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Return the number of pairwise classifiers.
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

impl Predict<Array2<f64>> for FittedOneVsOneClassifier {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels using majority voting.
    ///
    /// For each sample, every pairwise classifier votes for one of its two
    /// classes. The score determines which class receives the vote: if the
    /// score exceeds 0.5, the vote goes to class i; otherwise to class j.
    /// The class with the most votes is predicted.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if any pairwise classifier fails to predict.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        // Vote matrix: votes[sample][class_index]
        let mut votes = vec![vec![0u32; n_classes]; n_samples];

        // Map class label -> index for vote accumulation.
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        for (cls_i, cls_j, est) in &self.estimators {
            let preds = est.predict(x)?;
            let idx_i = class_to_idx[cls_i];
            let idx_j = class_to_idx[cls_j];

            for s in 0..n_samples {
                if preds[s] > 0.5 {
                    votes[s][idx_i] += 1;
                } else {
                    votes[s][idx_j] += 1;
                }
            }
        }

        // Select the class with the most votes for each sample.
        let mut predictions = Vec::with_capacity(n_samples);
        for sample_votes in &votes {
            let best_idx = sample_votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, v)| *v)
                .map(|(k, _)| k)
                .unwrap_or(0);
            predictions.push(self.classes[best_idx]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};

    // -----------------------------------------------------------------------
    // Test fixture: a threshold-based binary classifier
    // -----------------------------------------------------------------------

    /// A binary classifier that learns the mean of positive-class features
    /// and predicts based on distance to that mean vs. the negative-class mean.
    struct ThresholdEstimator;

    struct FittedThreshold {
        /// Mean of positive-class samples (label = 1.0).
        pos_mean: f64,
        /// Mean of negative-class samples (label = 0.0).
        neg_mean: f64,
    }

    impl PipelineEstimator<f64> for ThresholdEstimator {
        fn fit_pipeline(
            &self,
            x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            let (mut pos_sum, mut pos_count) = (0.0, 0usize);
            let (mut neg_sum, mut neg_count) = (0.0, 0usize);

            for (i, &label) in y.iter().enumerate() {
                let row_mean = x.row(i).mean().unwrap_or(0.0);
                if label > 0.5 {
                    pos_sum += row_mean;
                    pos_count += 1;
                } else {
                    neg_sum += row_mean;
                    neg_count += 1;
                }
            }

            let pos_mean = if pos_count > 0 {
                pos_sum / pos_count as f64
            } else {
                0.0
            };
            let neg_mean = if neg_count > 0 {
                neg_sum / neg_count as f64
            } else {
                0.0
            };

            Ok(Box::new(FittedThreshold { pos_mean, neg_mean }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedThreshold {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let preds: Vec<f64> = x
                .rows()
                .into_iter()
                .map(|row| {
                    let val = row.mean().unwrap_or(0.0);
                    let d_pos = (val - self.pos_mean).abs();
                    let d_neg = (val - self.neg_mean).abs();
                    let total = d_pos + d_neg;
                    if total < 1e-15 {
                        0.5
                    } else {
                        // Continuous score: closer to pos_mean -> higher score
                        d_neg / total
                    }
                })
                .collect();
            Ok(Array1::from_vec(preds))
        }
    }

    /// Create a pipeline factory that produces threshold-based classifiers.
    fn make_threshold_factory() -> PipelineFactory {
        Box::new(|| Pipeline::new().estimator_step("clf", Box::new(ThresholdEstimator)))
    }

    // -----------------------------------------------------------------------
    // OneVsRestClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ovr_fit_predict_three_classes() {
        // Class 0: features near 0, Class 1: features near 5, Class 2: features near 10
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.1, 0.1, 0.0, 0.2, 0.1, // class 0
                5.0, 5.1, 5.1, 5.0, 5.2, 5.1, // class 1
                10.0, 10.1, 10.1, 10.0, 10.2, 10.1, // class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

        let ovr = OneVsRestClassifier::new(make_threshold_factory());
        let fitted = ovr.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_estimators(), 3);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 9);

        // Each sample should be classified to its own class (well-separated).
        for i in 0..3 {
            assert_eq!(preds[i], 0, "sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1, "sample {} should be class 1", i);
        }
        for i in 6..9 {
            assert_eq!(preds[i], 2, "sample {} should be class 2", i);
        }
    }

    #[test]
    fn test_ovr_decision_function_shape() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 5.0, 5.1, 10.0, 10.1]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let ovr = OneVsRestClassifier::new(make_threshold_factory());
        let fitted = ovr.fit(&x, &y).unwrap();

        let scores = fitted.decision_function(&x).unwrap();
        assert_eq!(scores.nrows(), 6);
        assert_eq!(scores.ncols(), 3);
    }

    #[test]
    fn test_ovr_shape_mismatch() {
        let x = Array2::<f64>::zeros((10, 2));
        let y = Array1::from_vec(vec![0, 1, 2]); // wrong length

        let ovr = OneVsRestClassifier::new(make_threshold_factory());
        assert!(ovr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ovr_single_class_fails() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0]);

        let ovr = OneVsRestClassifier::new(make_threshold_factory());
        assert!(ovr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ovr_two_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let ovr = OneVsRestClassifier::new(make_threshold_factory());
        let fitted = ovr.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_estimators(), 2);
        let preds = fitted.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(preds[i], 0);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1);
        }
    }

    // -----------------------------------------------------------------------
    // OneVsOneClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ovo_fit_predict_three_classes() {
        // 3 classes -> 3 pairwise classifiers
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.1, 0.1, 0.0, 0.2, 0.1, // class 0
                5.0, 5.1, 5.1, 5.0, 5.2, 5.1, // class 1
                10.0, 10.1, 10.1, 10.0, 10.2, 10.1, // class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);

        let ovo = OneVsOneClassifier::new(make_threshold_factory());
        let fitted = ovo.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_estimators(), 3); // C(3,2) = 3

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 9);

        for i in 0..3 {
            assert_eq!(preds[i], 0, "sample {} should be class 0", i);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1, "sample {} should be class 1", i);
        }
        for i in 6..9 {
            assert_eq!(preds[i], 2, "sample {} should be class 2", i);
        }
    }

    #[test]
    fn test_ovo_n_estimators_four_classes() {
        // 4 classes -> C(4,2) = 6 pairwise classifiers
        let x =
            Array2::from_shape_vec((8, 1), vec![0.0, 0.1, 3.0, 3.1, 6.0, 6.1, 9.0, 9.1]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2, 3, 3]);

        let ovo = OneVsOneClassifier::new(make_threshold_factory());
        let fitted = ovo.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_estimators(), 6);
        assert_eq!(fitted.classes(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_ovo_shape_mismatch() {
        let x = Array2::<f64>::zeros((10, 2));
        let y = Array1::from_vec(vec![0, 1, 2]);

        let ovo = OneVsOneClassifier::new(make_threshold_factory());
        assert!(ovo.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ovo_single_class_fails() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0]);

        let ovo = OneVsOneClassifier::new(make_threshold_factory());
        assert!(ovo.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ovo_two_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let ovo = OneVsOneClassifier::new(make_threshold_factory());
        let fitted = ovo.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_estimators(), 1); // C(2,1) = 1
        let preds = fitted.predict(&x).unwrap();
        for i in 0..3 {
            assert_eq!(preds[i], 0);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1);
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_unique_classes() {
        let y = Array1::from_vec(vec![2, 0, 1, 0, 2, 1, 0]);
        let classes = unique_classes(&y);
        assert_eq!(classes, vec![0, 1, 2]);
    }

    #[test]
    fn test_unique_classes_single() {
        let y = Array1::from_vec(vec![3, 3, 3]);
        let classes = unique_classes(&y);
        assert_eq!(classes, vec![3]);
    }
}
