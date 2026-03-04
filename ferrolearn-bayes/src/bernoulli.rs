//! Bernoulli Naive Bayes classifier.
//!
//! This module provides [`BernoulliNB`], suitable for binary/boolean feature
//! data. An optional binarization threshold can be used to convert continuous
//! features to binary values before fitting and prediction.
//!
//! The log-likelihood for feature `j` in class `c` is:
//!
//! ```text
//! log P(x_j | c) = x_j * log(p_cj) + (1 - x_j) * log(1 - p_cj)
//! ```
//!
//! where `p_cj = (N_cj + alpha) / (N_c + 2 * alpha)` is the smoothed
//! probability that feature `j` is present in class `c`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::BernoulliNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         1.0, 1.0, 0.0,
//!         1.0, 0.0, 0.0,
//!         1.0, 1.0, 0.0,
//!         0.0, 0.0, 1.0,
//!         0.0, 1.0, 1.0,
//!         0.0, 0.0, 1.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = BernoulliNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Bernoulli Naive Bayes classifier.
///
/// Suitable for binary feature data. Features can be binarized automatically
/// by setting `binarize` to a threshold value.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BernoulliNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
    /// Optional threshold for binarizing features. Values strictly above this
    /// threshold are set to 1; others to 0. If `None`, features are used as-is.
    pub binarize: Option<F>,
}

impl<F: Float> BernoulliNB<F> {
    /// Create a new `BernoulliNB` with default settings.
    ///
    /// Default: `alpha = 1.0`, `binarize = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            binarize: None,
        }
    }

    /// Set the Laplace smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the binarization threshold.
    ///
    /// Values strictly above this threshold become 1; all others become 0.
    #[must_use]
    pub fn with_binarize(mut self, threshold: F) -> Self {
        self.binarize = Some(threshold);
        self
    }
}

impl<F: Float> Default for BernoulliNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Binarize an array using the given threshold.
///
/// Values strictly above `threshold` become `F::one()`; others become `F::zero()`.
fn binarize_array<F: Float>(x: &Array2<F>, threshold: F) -> Array2<F> {
    x.mapv(|v| if v > threshold { F::one() } else { F::zero() })
}

/// Fitted Bernoulli Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedBernoulliNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Log feature-present probability per class, shape `(n_classes, n_features)`.
    log_prob: Array2<F>,
    /// Log complement (1 - p) per class, shape `(n_classes, n_features)`.
    log_neg_prob: Array2<F>,
    /// Binarization threshold (carried forward for prediction).
    binarize: Option<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for BernoulliNB<F> {
    type Fitted = FittedBernoulliNB<F>;
    type Error = FerroError;

    /// Fit the Bernoulli NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedBernoulliNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "BernoulliNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Optionally binarize.
        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_f = F::from(n_samples).unwrap();
        let two = F::from(2.0).unwrap();

        let mut log_prior = Array1::<F>::zeros(n_classes);
        let mut log_prob = Array2::<F>::zeros((n_classes, n_features));
        let mut log_neg_prob = Array2::<F>::zeros((n_classes, n_features));

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();
            let n_c_f = F::from(n_c).unwrap();
            log_prior[ci] = (n_c_f / n_f).ln();

            // Count occurrences of each feature in this class.
            for j in 0..n_features {
                let feature_count = class_indices
                    .iter()
                    .fold(F::zero(), |acc, &i| acc + x_bin[[i, j]]);

                // Smoothed probability: (N_cj + alpha) / (N_c + 2*alpha).
                let p = (feature_count + self.alpha) / (n_c_f + two * self.alpha);
                log_prob[[ci, j]] = p.ln();
                log_neg_prob[[ci, j]] = (F::one() - p).ln();
            }
        }

        Ok(FittedBernoulliNB {
            classes,
            log_prior,
            log_prob,
            log_neg_prob,
            binarize: self.binarize,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedBernoulliNB<F> {
    /// Compute joint log-likelihood for each class.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.log_prior[ci];
                for j in 0..n_features {
                    let xij = x[[i, j]];
                    // log P(x_j | c) = x_j * log(p_cj) + (1-x_j) * log(1-p_cj)
                    score = score
                        + xij * self.log_prob[[ci, j]]
                        + (F::one() - xij) * self.log_neg_prob[[ci, j]];
                }
                scores[[i, ci]] = score;
            }
        }

        scores
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    /// If binarize was set during fitting, features are binarized before
    /// prediction.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.log_prob.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted BernoulliNB".into(),
            });
        }

        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

        let log_scores = self.joint_log_likelihood(&x_bin);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let max_score = log_scores
                .row(i)
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b));

            let mut row_sum = F::zero();
            for ci in 0..n_classes {
                let p = (log_scores[[i, ci]] - max_score).exp();
                proba[[i, ci]] = p;
                row_sum = row_sum + p;
            }
            for ci in 0..n_classes {
                proba[[i, ci]] = proba[[i, ci]] / row_sum;
            }
        }

        Ok(proba)
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBernoulliNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// If binarize was set during fitting, features are binarized before
    /// prediction.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features_fitted = self.log_prob.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted BernoulliNB".into(),
            });
        }

        let x_bin = if let Some(threshold) = self.binarize {
            binarize_array(x, threshold)
        } else {
            x.clone()
        };

        let scores = self.joint_log_likelihood(&x_bin);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_score = scores[[i, 0]];
            for ci in 1..n_classes {
                if scores[[i, ci]] > best_score {
                    best_score = scores[[i, ci]];
                    best_class = ci;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedBernoulliNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for f64.
impl PipelineEstimator for BernoulliNB<f64> {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedBernoulliNBPipeline(fitted)))
    }
}

struct FittedBernoulliNBPipeline(FittedBernoulliNB<f64>);

unsafe impl Send for FittedBernoulliNBPipeline {}
unsafe impl Sync for FittedBernoulliNBPipeline {}

impl FittedPipelineEstimator for FittedBernoulliNBPipeline {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| v as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_binary_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
                0.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_bernoulli_nb_fit_predict() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_bernoulli_nb_predict_proba_sums_to_one() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bernoulli_nb_has_classes() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_bernoulli_nb_binarize_threshold() {
        // Continuous data binarized at 0.5.
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.8, 0.9, 0.1, 0.2, 0.1, 0.9, 0.1, 0.8, 0.7, 0.3,
                0.2, 0.8,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = BernoulliNB::<f64>::new().with_binarize(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_bernoulli_nb_binarize_zero_threshold() {
        // With threshold=0.0, all positive values become 1.
        let x =
            Array2::from_shape_vec((4, 2), vec![2.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 3.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = BernoulliNB::<f64>::new().with_binarize(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 0);
        assert_eq!(preds[3], 1);
    }

    #[test]
    fn test_bernoulli_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = BernoulliNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bernoulli_nb_shape_mismatch_predict() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_bernoulli_nb_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![5usize, 5, 5];
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 5));
    }

    #[test]
    fn test_bernoulli_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = BernoulliNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bernoulli_nb_default() {
        let model = BernoulliNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
        assert!(model.binarize.is_none());
    }

    #[test]
    fn test_bernoulli_nb_predict_proba_ordering() {
        let (x, y) = make_binary_data();
        let model = BernoulliNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        // First 3 samples should prefer class 0.
        for i in 0..3 {
            assert!(proba[[i, 0]] > proba[[i, 1]]);
        }
        // Last 3 samples should prefer class 1.
        for i in 3..6 {
            assert!(proba[[i, 1]] > proba[[i, 0]]);
        }
    }
}
