//! Random forest classifiers and regressors.
//!
//! This module provides [`RandomForestClassifier`] and [`RandomForestRegressor`],
//! which build ensembles of decision trees using bootstrap sampling and random
//! feature subsets (bagging). Trees are built in parallel via `rayon`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::RandomForestClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = RandomForestClassifier::<f64>::new()
//!     .with_n_estimators(10)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample as rand_sample_indices;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::decision_tree::{
    self, ClassificationCriterion, Node, build_classification_tree_with_feature_subset,
    build_regression_tree_with_feature_subset, compute_feature_importances,
};

// ---------------------------------------------------------------------------
// MaxFeatures
// ---------------------------------------------------------------------------

/// Strategy for selecting the number of features considered at each split.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaxFeatures {
    /// Use the square root of the total number of features (default for classifiers).
    Sqrt,
    /// Use the log2 of the total number of features.
    Log2,
    /// Use all features (default for regressors).
    All,
    /// Use a specific number of features.
    Fixed(usize),
    /// Use a fraction of the total number of features.
    Fraction(f64),
}

/// Resolve the `MaxFeatures` strategy to a concrete number.
fn resolve_max_features(strategy: MaxFeatures, n_features: usize) -> usize {
    let result = match strategy {
        MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
        MaxFeatures::Log2 => (n_features as f64).log2().ceil().max(1.0) as usize,
        MaxFeatures::All => n_features,
        MaxFeatures::Fixed(n) => n.min(n_features),
        MaxFeatures::Fraction(f) => ((n_features as f64) * f).ceil() as usize,
    };
    result.max(1).min(n_features)
}

/// Internal tree parameter struct reused from decision_tree.
///
/// Re-created here to avoid leaking internal details; the crate-internal
/// struct is the same shape.
fn make_tree_params(
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> decision_tree::TreeParams {
    decision_tree::TreeParams {
        max_depth,
        min_samples_split,
        min_samples_leaf,
    }
}

// ---------------------------------------------------------------------------
// RandomForestClassifier
// ---------------------------------------------------------------------------

/// Random forest classifier.
///
/// Builds an ensemble of decision tree classifiers, each trained on a
/// bootstrap sample with a random subset of features considered at each split.
/// Final predictions are made by majority vote.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier<F> {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Maximum depth of each tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    /// Splitting criterion.
    pub criterion: ClassificationCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> RandomForestClassifier<F> {
    /// Create a new `RandomForestClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_depth = None`,
    /// `max_features = Sqrt`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `random_state = None`,
    /// `criterion = Gini`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
            criterion: ClassificationCriterion::Gini,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of trees.
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the maximum features strategy.
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the minimum number of samples to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for RandomForestClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedRandomForestClassifier
// ---------------------------------------------------------------------------

/// A fitted random forest classifier.
///
/// Stores the ensemble of fitted decision trees and aggregates their
/// predictions by majority vote.
#[derive(Debug, Clone)]
pub struct FittedRandomForestClassifier<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (mean decrease in impurity, normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for RandomForestClassifier<F> {
    type Fitted = FittedRandomForestClassifier<F>;
    type Error = FerroError;

    /// Fit the random forest by building `n_estimators` decision trees in parallel.
    ///
    /// Each tree is trained on a bootstrap sample of the data, considering only
    /// a random subset of features at each split.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if `n_estimators` is 0.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedRandomForestClassifier<F>, FerroError> {
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
                context: "RandomForestClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        let max_features_n = resolve_max_features(self.max_features, n_features);
        let params = make_tree_params(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        );
        let criterion = self.criterion;

        // Generate per-tree seeds sequentially for determinism, then dispatch in parallel.
        let tree_seeds: Vec<u64> = if let Some(seed) = self.random_state {
            let mut master_rng = StdRng::seed_from_u64(seed);
            (0..self.n_estimators)
                .map(|_| {
                    use rand::RngCore;
                    master_rng.next_u64()
                })
                .collect()
        } else {
            (0..self.n_estimators)
                .map(|_| {
                    use rand::RngCore;
                    rand::rng().next_u64()
                })
                .collect()
        };

        // Build trees in parallel.
        let trees: Vec<Vec<Node<F>>> = tree_seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);

                // Bootstrap sample (with replacement).
                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| {
                        use rand::RngCore;
                        (rng.next_u64() as usize) % n_samples
                    })
                    .collect();

                // Random feature subset.
                let feature_indices: Vec<usize> =
                    rand_sample_indices(&mut rng, n_features, max_features_n).into_vec();

                build_classification_tree_with_feature_subset(
                    x,
                    &y_mapped,
                    n_classes,
                    &bootstrap_indices,
                    &feature_indices,
                    &params,
                    criterion,
                )
            })
            .collect();

        // Aggregate feature importances across trees.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees {
            let tree_imp = compute_feature_importances(tree_nodes, n_features, n_samples);
            total_importances = total_importances + tree_imp;
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedRandomForestClassifier {
            trees,
            classes,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedRandomForestClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels by majority vote across all trees.
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
        let n_classes = self.classes.len();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut votes = vec![0usize; n_classes];

            for tree_nodes in &self.trees {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    let class_idx = value.to_f64().map(|f| f.round() as usize).unwrap_or(0);
                    if class_idx < n_classes {
                        votes[class_idx] += 1;
                    }
                }
            }

            let winner = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.classes[winner];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedRandomForestClassifier<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedRandomForestClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for f64.
impl PipelineEstimator for RandomForestClassifier<f64> {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
        let y_usize = y.mapv(|v| v as usize);
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedForestClassifierPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedRandomForestClassifier<f64>`.
struct FittedForestClassifierPipelineAdapter(FittedRandomForestClassifier<f64>);

impl FittedPipelineEstimator for FittedForestClassifierPipelineAdapter {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| v as f64))
    }
}

// ---------------------------------------------------------------------------
// RandomForestRegressor
// ---------------------------------------------------------------------------

/// Random forest regressor.
///
/// Builds an ensemble of decision tree regressors, each trained on a
/// bootstrap sample with a random subset of features considered at each split.
/// Final predictions are the mean across all trees.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor<F> {
    /// Number of trees in the forest.
    pub n_estimators: usize,
    /// Maximum depth of each tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> RandomForestRegressor<F> {
    /// Create a new `RandomForestRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_depth = None`,
    /// `max_features = All`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            max_features: MaxFeatures::All,
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of trees.
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the maximum features strategy.
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the minimum number of samples to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for RandomForestRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedRandomForestRegressor
// ---------------------------------------------------------------------------

/// A fitted random forest regressor.
///
/// Stores the ensemble of fitted decision trees and aggregates their
/// predictions by averaging.
#[derive(Debug, Clone)]
pub struct FittedRandomForestRegressor<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (mean decrease in impurity, normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for RandomForestRegressor<F> {
    type Fitted = FittedRandomForestRegressor<F>;
    type Error = FerroError;

    /// Fit the random forest regressor.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if `n_estimators` is 0.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedRandomForestRegressor<F>, FerroError> {
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
                context: "RandomForestRegressor requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }

        let max_features_n = resolve_max_features(self.max_features, n_features);
        let params = make_tree_params(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        );

        // Generate per-tree seeds sequentially.
        let tree_seeds: Vec<u64> = if let Some(seed) = self.random_state {
            let mut master_rng = StdRng::seed_from_u64(seed);
            (0..self.n_estimators)
                .map(|_| {
                    use rand::RngCore;
                    master_rng.next_u64()
                })
                .collect()
        } else {
            (0..self.n_estimators)
                .map(|_| {
                    use rand::RngCore;
                    rand::rng().next_u64()
                })
                .collect()
        };

        // Build trees in parallel.
        let trees: Vec<Vec<Node<F>>> = tree_seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);

                let bootstrap_indices: Vec<usize> = (0..n_samples)
                    .map(|_| {
                        use rand::RngCore;
                        (rng.next_u64() as usize) % n_samples
                    })
                    .collect();

                let feature_indices: Vec<usize> =
                    rand_sample_indices(&mut rng, n_features, max_features_n).into_vec();

                build_regression_tree_with_feature_subset(
                    x,
                    y,
                    &bootstrap_indices,
                    &feature_indices,
                    &params,
                )
            })
            .collect();

        // Aggregate feature importances.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees {
            let tree_imp = compute_feature_importances(tree_nodes, n_features, n_samples);
            total_importances = total_importances + tree_imp;
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedRandomForestRegressor {
            trees,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedRandomForestRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values by averaging across all trees.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let n_trees_f = F::from(self.trees.len()).unwrap();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut sum = F::zero();

            for tree_nodes in &self.trees {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    sum = sum + value;
                }
            }

            predictions[i] = sum / n_trees_f;
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedRandomForestRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration for f64.
impl PipelineEstimator for RandomForestRegressor<f64> {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl FittedPipelineEstimator for FittedRandomForestRegressor<f64> {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- Classifier tests --

    #[test]
    fn test_forest_classifier_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(20)
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
    fn test_forest_classifier_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_forest_classifier_feature_importances() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 3);
        assert!(importances[0] > importances[1]);
        assert!(importances[0] > importances[2]);
    }

    #[test]
    fn test_forest_classifier_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_forest_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = RandomForestClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_forest_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = RandomForestClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_classifier_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = RandomForestClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_classifier_single_tree() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(1)
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_forest_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_forest_classifier_max_depth() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = RandomForestClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_max_depth(Some(1))
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    // -- Regressor tests --

    #[test]
    fn test_forest_regressor_simple() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = RandomForestRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert!(preds[i] < 3.0, "Expected ~1.0, got {}", preds[i]);
        }
        for i in 4..8 {
            assert!(preds[i] > 3.0, "Expected ~5.0, got {}", preds[i]);
        }
    }

    #[test]
    fn test_forest_regressor_reproducibility() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model = RandomForestRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(99);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (p1, p2) in preds1.iter().zip(preds2.iter()) {
            assert_relative_eq!(*p1, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_forest_regressor_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = RandomForestRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_forest_regressor_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = RandomForestRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_regressor_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = RandomForestRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_forest_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = RandomForestRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_regressor_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = RandomForestRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_forest_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = RandomForestRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_forest_regressor_max_features_strategies() {
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 7.0,
                5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 11.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        for strategy in &[
            MaxFeatures::Sqrt,
            MaxFeatures::Log2,
            MaxFeatures::All,
            MaxFeatures::Fixed(2),
            MaxFeatures::Fraction(0.5),
        ] {
            let model = RandomForestRegressor::<f64>::new()
                .with_n_estimators(5)
                .with_max_features(*strategy)
                .with_random_state(42);
            let fitted = model.fit(&x, &y).unwrap();
            let preds = fitted.predict(&x).unwrap();
            assert_eq!(preds.len(), 8);
        }
    }

    // -- MaxFeatures resolution tests --

    #[test]
    fn test_resolve_max_features_sqrt() {
        assert_eq!(resolve_max_features(MaxFeatures::Sqrt, 9), 3);
        assert_eq!(resolve_max_features(MaxFeatures::Sqrt, 10), 4);
        assert_eq!(resolve_max_features(MaxFeatures::Sqrt, 1), 1);
    }

    #[test]
    fn test_resolve_max_features_log2() {
        assert_eq!(resolve_max_features(MaxFeatures::Log2, 8), 3);
        assert_eq!(resolve_max_features(MaxFeatures::Log2, 1), 1);
    }

    #[test]
    fn test_resolve_max_features_all() {
        assert_eq!(resolve_max_features(MaxFeatures::All, 10), 10);
        assert_eq!(resolve_max_features(MaxFeatures::All, 1), 1);
    }

    #[test]
    fn test_resolve_max_features_fixed() {
        assert_eq!(resolve_max_features(MaxFeatures::Fixed(3), 10), 3);
        assert_eq!(resolve_max_features(MaxFeatures::Fixed(20), 10), 10);
    }

    #[test]
    fn test_resolve_max_features_fraction() {
        assert_eq!(resolve_max_features(MaxFeatures::Fraction(0.5), 10), 5);
        assert_eq!(resolve_max_features(MaxFeatures::Fraction(0.1), 10), 1);
    }

    #[test]
    fn test_forest_classifier_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = RandomForestClassifier::<f32>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_forest_regressor_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let model = RandomForestRegressor::<f32>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
