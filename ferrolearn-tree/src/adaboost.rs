//! AdaBoost classifier.
//!
//! This module provides [`AdaBoostClassifier`], which implements the Adaptive
//! Boosting algorithm using decision tree stumps (depth-1 trees) as base
//! estimators. Two algorithm variants are supported:
//!
//! - **SAMME**: uses discrete class predictions and works with multiclass
//!   problems directly.
//! - **SAMME.R** (default): uses class probability estimates, typically giving
//!   better performance than SAMME.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::AdaBoostClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = AdaBoostClassifier::<f64>::new()
//!     .with_n_estimators(50)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use crate::decision_tree::{
    self, ClassificationCriterion, Node, build_classification_tree_with_feature_subset,
};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

// ---------------------------------------------------------------------------
// Algorithm enum
// ---------------------------------------------------------------------------

/// AdaBoost algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// SAMME: Stagewise Additive Modeling using a Multi-class Exponential loss.
    ///
    /// Uses discrete class predictions from each base estimator.
    Samme,
    /// SAMME.R: the "real" variant that uses class probability estimates.
    ///
    /// Generally outperforms SAMME.
    SammeR,
}

// ---------------------------------------------------------------------------
// AdaBoostClassifier
// ---------------------------------------------------------------------------

/// AdaBoost classifier using decision tree stumps as base estimators.
///
/// At each boosting round a decision tree stump (max depth = 1) is fitted
/// to the weighted training data. Misclassified samples receive higher
/// weight in subsequent rounds, allowing the ensemble to focus on hard
/// examples.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AdaBoostClassifier<F> {
    /// Number of boosting stages (stumps).
    pub n_estimators: usize,
    /// Learning rate (shrinkage). Lower values require more estimators.
    pub learning_rate: f64,
    /// Algorithm variant (`SAMME` or `SAMME.R`).
    pub algorithm: AdaBoostAlgorithm,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> AdaBoostClassifier<F> {
    /// Create a new `AdaBoostClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 50`, `learning_rate = 1.0`,
    /// `algorithm = SAMME.R`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            algorithm: AdaBoostAlgorithm::SammeR,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the algorithm variant.
    #[must_use]
    pub fn with_algorithm(mut self, algo: AdaBoostAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for AdaBoostClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedAdaBoostClassifier
// ---------------------------------------------------------------------------

/// A fitted AdaBoost classifier.
///
/// Stores the sequence of stumps and their weights. Predictions are made
/// by weighted majority vote (SAMME) or weighted probability averaging
/// (SAMME.R).
#[derive(Debug, Clone)]
pub struct FittedAdaBoostClassifier<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Sequence of fitted tree stumps.
    estimators: Vec<Vec<Node<F>>>,
    /// Weight of each estimator (SAMME) or kept for SAMME.R bookkeeping.
    estimator_weights: Vec<F>,
    /// Number of features.
    n_features: usize,
    /// Number of classes.
    n_classes: usize,
    /// Algorithm used.
    algorithm: AdaBoostAlgorithm,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for AdaBoostClassifier<F> {
    type Fitted = FittedAdaBoostClassifier<F>;
    type Error = FerroError;

    /// Fit the AdaBoost classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
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
                context: "AdaBoostClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "need at least 2 distinct classes".into(),
            });
        }

        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        match self.algorithm {
            AdaBoostAlgorithm::Samme => {
                self.fit_samme(x, &y_mapped, n_samples, n_features, n_classes, &classes)
            }
            AdaBoostAlgorithm::SammeR => {
                self.fit_samme_r(x, &y_mapped, n_samples, n_features, n_classes, &classes)
            }
        }
    }
}

impl<F: Float + Send + Sync + 'static> AdaBoostClassifier<F> {
    /// Fit using the SAMME algorithm (discrete predictions).
    fn fit_samme(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
        let lr = F::from(self.learning_rate).unwrap();
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-10).unwrap();

        // Initialize sample weights uniformly.
        let mut weights = vec![F::one() / n_f; n_samples];

        let all_features: Vec<usize> = (0..n_features).collect();
        let stump_params = decision_tree::TreeParams {
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Build weighted sample indices: replicate indices proportional to weight.
            let indices = resample_weighted(&weights, n_samples);

            let tree = build_classification_tree_with_feature_subset(
                x,
                y_mapped,
                n_classes,
                &indices,
                &all_features,
                &stump_params,
                ClassificationCriterion::Gini,
            );

            // Compute predictions and weighted error.
            let mut weighted_error = F::zero();
            let mut preds = vec![0usize; n_samples];
            for i in 0..n_samples {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf { value, .. } = tree[leaf_idx] {
                    preds[i] = value.to_f64().map(|f| f.round() as usize).unwrap_or(0);
                }
                if preds[i] != y_mapped[i] {
                    weighted_error = weighted_error + weights[i];
                }
            }

            // Normalise error.
            let weight_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            let err = if weight_sum > F::zero() {
                weighted_error / weight_sum
            } else {
                F::from(0.5).unwrap()
            };

            // If error is too high or zero, stop or skip.
            if err >= F::one() - F::one() / F::from(n_classes).unwrap() {
                // Error too high; stop boosting.
                if estimators.is_empty() {
                    // Keep at least one estimator.
                    estimators.push(tree);
                    estimator_weights.push(F::one());
                }
                break;
            }

            // Estimator weight: SAMME formula.
            let alpha = lr * ((F::one() - err).max(eps) / err.max(eps)).ln()
                + lr * (F::from(n_classes - 1).unwrap()).ln();

            // Update sample weights.
            for i in 0..n_samples {
                if preds[i] != y_mapped[i] {
                    weights[i] = weights[i] * alpha.exp();
                }
            }

            // Normalise weights.
            let new_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if new_sum > F::zero() {
                for w in &mut weights {
                    *w = *w / new_sum;
                }
            }

            estimators.push(tree);
            estimator_weights.push(alpha);
        }

        Ok(FittedAdaBoostClassifier {
            classes: classes.to_vec(),
            estimators,
            estimator_weights,
            n_features,
            n_classes,
            algorithm: AdaBoostAlgorithm::Samme,
        })
    }

    /// Fit using the SAMME.R algorithm (real-valued / probability-based).
    fn fit_samme_r(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
    ) -> Result<FittedAdaBoostClassifier<F>, FerroError> {
        let lr = F::from(self.learning_rate).unwrap();
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-10).unwrap();
        let k_f = F::from(n_classes).unwrap();

        // Initialize sample weights uniformly.
        let mut weights = vec![F::one() / n_f; n_samples];

        let all_features: Vec<usize> = (0..n_features).collect();
        let stump_params = decision_tree::TreeParams {
            max_depth: Some(1),
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            let indices = resample_weighted(&weights, n_samples);

            let tree = build_classification_tree_with_feature_subset(
                x,
                y_mapped,
                n_classes,
                &indices,
                &all_features,
                &stump_params,
                ClassificationCriterion::Gini,
            );

            // Get class probability estimates for each sample.
            let mut proba = vec![vec![F::zero(); n_classes]; n_samples];
            for (i, proba_row) in proba.iter_mut().enumerate() {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf {
                    class_distribution: Some(ref dist),
                    ..
                } = tree[leaf_idx]
                {
                    for (k, &p) in dist.iter().enumerate() {
                        proba_row[k] = p.max(eps);
                    }
                } else {
                    // Fallback: uniform.
                    for val in proba_row.iter_mut() {
                        *val = F::one() / k_f;
                    }
                }
                // Normalise.
                let row_sum: F = proba_row.iter().copied().fold(F::zero(), |a, b| a + b);
                if row_sum > F::zero() {
                    for val in proba_row.iter_mut() {
                        *val = *val / row_sum;
                    }
                }
            }

            // SAMME.R weight update: based on log-probability.
            // h_k(x) = (K-1) * (log(p_k(x)) - (1/K) * sum_j log(p_j(x)))
            // Then update: w_i *= exp(-(K-1)/K * lr * sum_k y_{ik} * log(p_k(x)))
            // Simplified: w_i *= exp(-lr * (K-1)/K * log(p_{y_i}(x)))
            let factor = lr * (k_f - F::one()) / k_f;
            let mut any_update = false;

            for i in 0..n_samples {
                let p_correct = proba[i][y_mapped[i]].max(eps);
                let exponent = -factor * p_correct.ln();
                weights[i] = weights[i] * exponent.exp();
                if exponent.abs() > eps {
                    any_update = true;
                }
            }

            // Normalise weights.
            let new_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if new_sum > F::zero() {
                for w in &mut weights {
                    *w = *w / new_sum;
                }
            }

            estimators.push(tree);
            estimator_weights.push(F::one()); // SAMME.R uses equal weight; prediction uses probabilities.

            if !any_update {
                break;
            }
        }

        Ok(FittedAdaBoostClassifier {
            classes: classes.to_vec(),
            estimators,
            estimator_weights,
            n_features,
            n_classes,
            algorithm: AdaBoostAlgorithm::SammeR,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedAdaBoostClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels.
    ///
    /// - **SAMME**: weighted majority vote using estimator weights.
    /// - **SAMME.R**: weighted average of log-probabilities.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();

        match self.algorithm {
            AdaBoostAlgorithm::Samme => self.predict_samme(x, n_samples),
            AdaBoostAlgorithm::SammeR => self.predict_samme_r(x, n_samples),
        }
    }
}

impl<F: Float + Send + Sync + 'static> FittedAdaBoostClassifier<F> {
    /// Predict using SAMME (weighted majority vote).
    fn predict_samme(&self, x: &Array2<F>, n_samples: usize) -> Result<Array1<usize>, FerroError> {
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut class_scores = vec![F::zero(); self.n_classes];

            for (t, tree_nodes) in self.estimators.iter().enumerate() {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    let class_idx = value.to_f64().map(|f| f.round() as usize).unwrap_or(0);
                    if class_idx < self.n_classes {
                        class_scores[class_idx] =
                            class_scores[class_idx] + self.estimator_weights[t];
                    }
                }
            }

            let best = class_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k)
                .unwrap_or(0);
            predictions[i] = self.classes[best];
        }

        Ok(predictions)
    }

    /// Predict using SAMME.R (weighted probability averaging).
    fn predict_samme_r(
        &self,
        x: &Array2<F>,
        n_samples: usize,
    ) -> Result<Array1<usize>, FerroError> {
        let eps = F::from(1e-10).unwrap();
        let k_f = F::from(self.n_classes).unwrap();
        let k_minus_1 = k_f - F::one();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut accumulated = vec![F::zero(); self.n_classes];

            for tree_nodes in &self.estimators {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf {
                    class_distribution: Some(ref dist),
                    ..
                } = tree_nodes[leaf_idx]
                {
                    // h_k(x) = (K-1) * (log(p_k) - mean(log(p_j)))
                    let log_probs: Vec<F> = dist.iter().map(|&p| p.max(eps).ln()).collect();
                    let mean_log: F = log_probs.iter().copied().fold(F::zero(), |a, b| a + b) / k_f;

                    for k in 0..self.n_classes {
                        accumulated[k] = accumulated[k] + k_minus_1 * (log_probs[k] - mean_log);
                    }
                } else {
                    // Leaf without distribution: predict from value.
                    if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                        let class_idx = value.to_f64().map(|f| f.round() as usize).unwrap_or(0);
                        if class_idx < self.n_classes {
                            accumulated[class_idx] = accumulated[class_idx] + F::one();
                        }
                    }
                }
            }

            let best = accumulated
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k)
                .unwrap_or(0);
            predictions[i] = self.classes[best];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedAdaBoostClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for AdaBoostClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedAdaBoostPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedAdaBoostClassifier<F>`.
struct FittedAdaBoostPipelineAdapter<F: Float + Send + Sync + 'static>(FittedAdaBoostClassifier<F>);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedAdaBoostPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Resample indices proportional to weights (weighted bootstrap).
///
/// Uses a systematic resampling approach: the cumulative weight distribution
/// determines which original indices appear in the resampled set.
fn resample_weighted<F: Float>(weights: &[F], n: usize) -> Vec<usize> {
    if weights.is_empty() {
        return Vec::new();
    }

    // Build cumulative distribution.
    let mut cumsum = Vec::with_capacity(weights.len());
    let mut running = F::zero();
    for &w in weights {
        running = running + w;
        cumsum.push(running);
    }

    // Normalise (in case weights don't sum to 1).
    let total = running;
    if total <= F::zero() {
        return (0..n).collect();
    }

    let mut indices = Vec::with_capacity(n);
    let step = total / F::from(n).unwrap();
    let mut threshold = step / F::from(2.0).unwrap(); // Start in the middle of the first bin.
    let mut j = 0;

    for _ in 0..n {
        while j < cumsum.len() - 1 && cumsum[j] < threshold {
            j += 1;
        }
        indices.push(j);
        threshold = threshold + step;
    }

    indices
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -- SAMME.R tests --

    #[test]
    fn test_adaboost_sammer_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_adaboost_sammer_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 5,
            "Expected at least 5/9 correct, got {}/9",
            correct
        );
    }

    // -- SAMME tests --

    #[test]
    fn test_adaboost_samme_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_adaboost_samme_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 5,
            "Expected at least 5/9 correct for SAMME multiclass, got {}/9",
            correct
        );
    }

    // -- Common tests --

    #[test]
    fn test_adaboost_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_adaboost_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_adaboost_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_adaboost_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_invalid_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f32>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_default_trait() {
        let model = AdaBoostClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 50);
        assert!((model.learning_rate - 1.0).abs() < 1e-10);
        assert_eq!(model.algorithm, AdaBoostAlgorithm::SammeR);
    }

    #[test]
    fn test_adaboost_non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![10, 10, 10, 20, 20, 20];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for &p in preds.iter() {
            assert!(p == 10 || p == 20);
        }
    }

    #[test]
    fn test_adaboost_sammer_learning_rate_effect() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // Low learning rate should still work (just slower convergence).
        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_samme_learning_rate_effect() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_learning_rate(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_many_features() {
        // 4 features, only first one is informative.
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_4_classes() {
        let x = Array2::from_shape_vec(
            (12, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 12);
        assert_eq!(fitted.n_classes(), 4);
    }

    // -- Resample helper tests --

    #[test]
    fn test_resample_weighted_uniform() {
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let indices = resample_weighted(&weights, 4);
        assert_eq!(indices.len(), 4);
        // With uniform weights, each index should appear once.
        for i in 0..4 {
            assert_eq!(indices[i], i);
        }
    }

    #[test]
    fn test_resample_weighted_skewed() {
        let weights = vec![0.0, 0.0, 0.0, 1.0];
        let indices = resample_weighted(&weights, 4);
        assert_eq!(indices.len(), 4);
        // All weight on last index.
        for &idx in &indices {
            assert_eq!(idx, 3);
        }
    }

    #[test]
    fn test_resample_weighted_empty() {
        let weights: Vec<f64> = Vec::new();
        let indices = resample_weighted(&weights, 0);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_adaboost_sammer_single_estimator() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_samme_single_estimator() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(1)
            .with_algorithm(AdaBoostAlgorithm::Samme)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_negative_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = AdaBoostClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }
}
