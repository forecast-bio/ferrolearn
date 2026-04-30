//! Complement Naive Bayes classifier.
//!
//! This module provides [`ComplementNB`], a variant of Multinomial Naive Bayes
//! that is particularly well-suited for imbalanced datasets. Instead of estimating
//! the likelihood of a feature given a class, it estimates the likelihood of the
//! feature given all *other* (complement) classes and inverts the weights.
//!
//! The weight for feature `j` in class `c` is:
//!
//! ```text
//! w_cj = log( (N_~cj + alpha) / (N_~c + alpha * n_features) )
//! ```
//!
//! where `N_~cj` is the total count of feature `j` in all classes except `c`,
//! and `N_~c` is the total count of all features in all classes except `c`.
//!
//! Prediction uses `argmin_c sum_j x_j * w_cj` (i.e., the class with the
//! *smallest* complement score is chosen).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::ComplementNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         5.0, 1.0, 0.0,
//!         4.0, 2.0, 0.0,
//!         6.0, 0.0, 1.0,
//!         0.0, 1.0, 5.0,
//!         1.0, 0.0, 4.0,
//!         0.0, 2.0, 6.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = ComplementNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Complement Naive Bayes classifier.
///
/// A variant of Multinomial NB that uses complement-class statistics.
/// More robust for imbalanced datasets.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct ComplementNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
    /// Optional user-supplied class priors. Note: ComplementNB does not
    /// use priors in the standard way (it uses complement weights), but
    /// this field is provided for API consistency with other NB variants.
    pub class_prior: Option<Vec<F>>,
    /// Whether to learn class priors from the data. Stored for API
    /// consistency; ComplementNB's predict does not consult priors in the
    /// multi-class case. Default: `true`.
    pub fit_prior: bool,
    /// When `false`, `alpha` values below `1e-10` are silently raised to
    /// `1e-10` (legacy behavior). Default: `true`.
    pub force_alpha: bool,
    /// When `true`, performs a second L1 normalization of the weights
    /// (Rennie et al. 2003 §4.4 "normalized weights" variant). Default:
    /// `false`.
    pub norm: bool,
}

impl<F: Float> ComplementNB<F> {
    /// Create a new `ComplementNB` with Laplace smoothing (`alpha = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            class_prior: None,
            fit_prior: true,
            force_alpha: true,
            norm: false,
        }
    }

    /// Set the Laplace smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set user-supplied class priors.
    ///
    /// The priors must have length equal to the number of classes discovered
    /// during fitting. Note: ComplementNB uses complement-class weights rather
    /// than direct class priors, but the priors are stored for API consistency.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Vec<F>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Toggle `fit_prior`. Stored for API consistency with other discrete NBs.
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Toggle the `force_alpha` policy. See struct field doc.
    #[must_use]
    pub fn with_force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }

    /// Toggle the second L1 normalization on weights (sklearn's `norm`
    /// parameter; Rennie et al. 2003 §4.4).
    #[must_use]
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }
}

impl<F: Float> Default for ComplementNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Complement Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedComplementNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Complement weights per class, shape `(n_classes, n_features)`.
    /// Each entry is `log( (N_~cj + alpha) / (N_~c + alpha * n_features) )`.
    weights: Array2<F>,
    /// Raw per-class feature count sums, shape `(n_classes, n_features)`.
    feature_counts: Array2<F>,
    /// Per-class sample counts.
    class_counts: Vec<usize>,
    /// Smoothing parameter carried forward for partial_fit (post-clamp
    /// when `force_alpha=false`).
    alpha: F,
    /// Whether to apply the second L1 normalization on weights (carried
    /// forward for partial_fit).
    norm: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for ComplementNB<F> {
    type Fitted = FittedComplementNB<F>;
    type Error = FerroError;

    /// Fit the Complement NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedComplementNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "ComplementNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Validate non-negative features.
        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "ComplementNB requires non-negative feature values".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_feat_f = F::from(n_features).unwrap();
        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        // Compute per-class feature count sums, shape (n_classes, n_features).
        let mut class_feature_counts = Array2::<F>::zeros((n_classes, n_features));
        let mut class_counts = vec![0usize; n_classes];

        for (sample_idx, &label) in y.iter().enumerate() {
            let ci = classes.iter().position(|&c| c == label).unwrap();
            class_counts[ci] += 1;
            for j in 0..n_features {
                class_feature_counts[[ci, j]] = class_feature_counts[[ci, j]] + x[[sample_idx, j]];
            }
        }

        // Total feature counts across all classes.
        let total_feature_counts: Array1<F> = class_feature_counts.rows().into_iter().fold(
            Array1::<F>::zeros(n_features),
            |acc, row| {
                let mut result = acc;
                for j in 0..n_features {
                    result[j] = result[j] + row[j];
                }
                result
            },
        );

        let total_all: F = total_feature_counts.sum();

        // Compute complement weights for each class.
        let mut weights = Array2::<F>::zeros((n_classes, n_features));

        for ci in 0..n_classes {
            // Complement counts: sum over all other classes.
            let complement_total = total_all - class_feature_counts.row(ci).sum();

            let denom = complement_total + alpha * n_feat_f;

            for j in 0..n_features {
                let complement_count_j = total_feature_counts[j] - class_feature_counts[[ci, j]];
                weights[[ci, j]] = ((complement_count_j + alpha) / denom).ln();
            }
        }

        if self.norm {
            apply_norm_inplace(&mut weights);
        }

        // Validate class_prior length if provided.
        if let Some(ref priors) = self.class_prior {
            if priors.len() != n_classes {
                return Err(FerroError::InvalidParameter {
                    name: "class_prior".into(),
                    reason: format!(
                        "length {} does not match number of classes {}",
                        priors.len(),
                        n_classes
                    ),
                });
            }
        }

        Ok(FittedComplementNB {
            classes,
            weights,
            feature_counts: class_feature_counts,
            class_counts,
            alpha,
            norm: self.norm,
        })
    }
}

/// Apply sklearn's `norm=True` second L1 normalization to complement
/// weights, then negate so ferrolearn's `argmin` predict semantics keep
/// matching sklearn's `argmax(X @ feature_log_prob.T)`.
///
/// Walks each row of `weights` (= sklearn's pre-negation `logged`):
/// - row_sum = Σ_j logged[c, j]    (≤ 0 since logs of probabilities)
/// - normalized = logged / row_sum  (≥ 0, rows sum to 1)
/// - weights[c, j] = -normalized    (re-negated to keep argmin convention)
fn apply_norm_inplace<F: Float>(weights: &mut Array2<F>) {
    let n_classes = weights.nrows();
    let n_features = weights.ncols();
    for ci in 0..n_classes {
        let row_sum = (0..n_features).fold(F::zero(), |acc, j| acc + weights[[ci, j]]);
        if row_sum == F::zero() {
            continue;
        }
        for j in 0..n_features {
            weights[[ci, j]] = -(weights[[ci, j]] / row_sum);
        }
    }
}

impl<F: Float + Send + Sync + 'static> FittedComplementNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Accumulates feature counts and class counts, then recomputes
    /// the complement weights.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row counts
    ///   or the number of features does not match the fitted model.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
    pub fn partial_fit(&mut self, x: &Array2<F>, y: &Array1<usize>) -> Result<(), FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Ok(());
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_features != self.weights.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.weights.ncols()],
                actual: vec![n_features],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }

        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "ComplementNB requires non-negative feature values".into(),
            });
        }

        // Accumulate counts for each existing class.
        for (ci, &class_label) in self.classes.clone().iter().enumerate() {
            let new_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if new_indices.is_empty() {
                continue;
            }

            self.class_counts[ci] += new_indices.len();

            for &i in &new_indices {
                for j in 0..n_features {
                    self.feature_counts[[ci, j]] = self.feature_counts[[ci, j]] + x[[i, j]];
                }
            }
        }

        // Recompute complement weights from accumulated feature_counts.
        let n_classes = self.classes.len();
        let n_feat_f = F::from(n_features).unwrap();

        let total_feature_counts: Array1<F> = self.feature_counts.rows().into_iter().fold(
            Array1::<F>::zeros(n_features),
            |acc, row| {
                let mut result = acc;
                for j in 0..n_features {
                    result[j] = result[j] + row[j];
                }
                result
            },
        );

        let total_all: F = total_feature_counts.sum();

        for ci in 0..n_classes {
            let complement_total = total_all - self.feature_counts.row(ci).sum();
            let denom = complement_total + self.alpha * n_feat_f;
            for j in 0..n_features {
                let complement_count_j = total_feature_counts[j] - self.feature_counts[[ci, j]];
                self.weights[[ci, j]] = ((complement_count_j + self.alpha) / denom).ln();
            }
        }

        if self.norm {
            apply_norm_inplace(&mut self.weights);
        }

        Ok(())
    }

    /// Compute complement scores for each class.
    ///
    /// Returns shape `(n_samples, n_classes)`. Lower is better.
    fn complement_scores(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = F::zero();
                for j in 0..n_features {
                    score = score + x[[i, j]] * self.weights[[ci, j]];
                }
                scores[[i, ci]] = score;
            }
        }

        scores
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Converts complement scores (lower=better) to probabilities by negating
    /// and applying softmax.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.weights.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }

        // Negate complement scores so that lower complement score → higher probability.
        let neg_scores = self.complement_scores(x).mapv(|v| -v);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let max_score = neg_scores
                .row(i)
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b));

            let mut row_sum = F::zero();
            for ci in 0..n_classes {
                let p = (neg_scores[[i, ci]] - max_score).exp();
                proba[[i, ci]] = p;
                row_sum = row_sum + p;
            }
            for ci in 0..n_classes {
                proba[[i, ci]] = proba[[i, ci]] / row_sum;
            }
        }

        Ok(proba)
    }

    /// Compute the joint log-likelihood scores using sklearn's sign
    /// convention: argmax(jll) gives the predicted class.
    ///
    /// Returns shape `(n_samples, n_classes)`. ComplementNB's complement
    /// scoring is "lower=better", so this method returns
    /// `-complement_scores` to match sklearn's convention where higher is
    /// better. Matches sklearn `ComplementNB._joint_log_likelihood`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_joint_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.weights.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }
        Ok(self.complement_scores(x).mapv(|v| -v))
    }

    /// Compute log of class probabilities (numerically stable).
    ///
    /// Returns shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let jll = self.predict_joint_log_proba(x)?;
        Ok(crate::log_softmax_rows(&jll))
    }

    /// Mean accuracy on the given test data and labels.
    ///
    /// Equivalent to sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the fitted model.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let n = y.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(F::from(correct).unwrap() / F::from(n).unwrap())
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedComplementNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Predicts the class with the *lowest* complement score.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features_fitted = self.weights.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted ComplementNB".into(),
            });
        }

        let scores = self.complement_scores(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            // Argmin: class with the smallest complement score.
            let mut best_class = 0;
            let mut best_score = scores[[i, 0]];
            for ci in 1..n_classes {
                if scores[[i, ci]] < best_score {
                    best_score = scores[[i, ci]];
                    best_class = ci;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedComplementNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for ComplementNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedComplementNBPipeline(fitted)))
    }
}

struct FittedComplementNBPipeline<F: Float + Send + Sync + 'static>(FittedComplementNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedComplementNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedComplementNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedComplementNBPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_count_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 0.0,
                2.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_complement_nb_fit_predict() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_complement_nb_predict_proba_sums_to_one() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complement_nb_has_classes() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_complement_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_shape_mismatch_predict() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_complement_nb_negative_features_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -0.5, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_single_class() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0usize, 0, 0];
        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_complement_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = ComplementNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_default() {
        let model = ComplementNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_complement_nb_imbalanced_data() {
        // ComplementNB is designed for imbalanced data.
        // 10 samples of class 0, 2 samples of class 1.
        let x = Array2::from_shape_vec(
            (12, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0,
                0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0,
                5.0, // class 1
                0.0, 2.0, 6.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];

        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Class 1 samples should be predicted as class 1.
        assert_eq!(preds[10], 1);
        assert_eq!(preds[11], 1);
    }

    #[test]
    fn test_complement_nb_partial_fit() {
        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0],
        )
        .unwrap();
        let y1 = array![0usize, 0, 1, 1];

        let model = ComplementNB::<f64>::new();
        let mut fitted = model.fit(&x1, &y1).unwrap();

        let x2 = Array2::from_shape_vec((2, 3), vec![6.0, 0.0, 1.0, 0.0, 2.0, 6.0]).unwrap();
        let y2 = array![0usize, 1];

        fitted.partial_fit(&x2, &y2).unwrap();

        let preds = fitted.predict(&x1).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_complement_nb_partial_fit_shape_mismatch() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new();
        let mut fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let y_bad = array![0usize, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_complement_nb_class_prior() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new().with_class_prior(vec![0.5, 0.5]);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_complement_nb_class_prior_wrong_length() {
        let (x, y) = make_count_data();
        let model = ComplementNB::<f64>::new().with_class_prior(vec![1.0]);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_complement_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 3),
            vec![
                5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 1.0,
                4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 1.0, 4.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = ComplementNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7);
    }
}
