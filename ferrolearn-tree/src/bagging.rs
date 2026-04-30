//! Bagging classifiers and regressors.
//!
//! This module provides [`BaggingClassifier`] and [`BaggingRegressor`],
//! which build ensembles of decision trees using bootstrap sampling of
//! both samples and features. Trees are built in parallel via `rayon`.
//!
//! Unlike [`RandomForestClassifier`](crate::RandomForestClassifier) (which
//! randomises features at each split), Bagging selects a random subset of
//! features *per estimator* and trains each tree on the full selected feature
//! set. The interface is designed to be generalised to arbitrary base
//! estimators in the future.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::BaggingClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = BaggingClassifier::<f64>::new()
//!     .with_n_estimators(10)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use crate::decision_tree::{
    self, ClassificationCriterion, Node, build_classification_tree_with_feature_subset,
    build_regression_tree_with_feature_subset,
};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample as rand_sample_indices;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// BaggingClassifier
// ---------------------------------------------------------------------------

/// Bootstrap aggregation (bagging) classifier using decision trees.
///
/// Each estimator is trained on a bootstrap sample of the data, optionally
/// with a random subset of features. Final predictions are made by majority
/// vote across all estimators.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BaggingClassifier<F> {
    /// Number of estimators in the ensemble.
    pub n_estimators: usize,
    /// Fraction of samples to draw for each estimator (default 1.0).
    pub max_samples: f64,
    /// Fraction of features to draw for each estimator (default 1.0).
    pub max_features: f64,
    /// Whether to sample with replacement (default `true`).
    pub bootstrap: bool,
    /// Whether to sample features with replacement (default `false`).
    pub bootstrap_features: bool,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Maximum depth of each base decision tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> BaggingClassifier<F> {
    /// Create a new `BaggingClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 10`, `max_samples = 1.0`,
    /// `max_features = 1.0`, `bootstrap = true`,
    /// `bootstrap_features = false`, `random_state = None`,
    /// `max_depth = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 10,
            max_samples: 1.0,
            max_features: 1.0,
            bootstrap: true,
            bootstrap_features: false,
            random_state: None,
            max_depth: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of estimators.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the fraction of samples to draw per estimator.
    #[must_use]
    pub fn with_max_samples(mut self, frac: f64) -> Self {
        self.max_samples = frac;
        self
    }

    /// Set the fraction of features to draw per estimator.
    #[must_use]
    pub fn with_max_features(mut self, frac: f64) -> Self {
        self.max_features = frac;
        self
    }

    /// Set whether to sample with replacement.
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set whether to sample features with replacement.
    #[must_use]
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.bootstrap_features = bootstrap_features;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the maximum depth of each base decision tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
}

impl<F: Float> Default for BaggingClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedBaggingClassifier
// ---------------------------------------------------------------------------

/// A fitted bagging classifier.
///
/// Stores the ensemble of fitted decision trees and aggregates their
/// predictions by majority vote.
#[derive(Debug, Clone)]
pub struct FittedBaggingClassifier<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Feature indices used by each tree (maps tree features back to original).
    feature_indices: Vec<Vec<usize>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features in the original data.
    n_features: usize,
    /// Per-feature importance scores aggregated across the ensemble
    /// (normalized to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedBaggingClassifier<F> {
    /// Returns a reference to the individual tree node vectors.
    #[must_use]
    pub fn trees(&self) -> &[Vec<Node<F>>] {
        &self.trees
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Mean accuracy on the given test data and labels.
    /// Equivalent to sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::mean_accuracy(&preds, y))
    }

    /// Predict class probabilities by averaging per-tree class
    /// distributions across the bagged ensemble. Mirrors sklearn's
    /// `BaggingClassifier.predict_proba`.
    ///
    /// Returns an `(n_samples, n_classes)` array. Each tree contributes
    /// either its leaf's full class distribution or a one-hot vote based
    /// on the leaf's predicted class. Each tree gets a row sub-set
    /// according to the `feature_indices` it was trained on.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_trees_f = F::from(self.trees.len()).unwrap();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = x.row(i);
            for (t, tree_nodes) in self.trees.iter().enumerate() {
                let feat_idx = &self.feature_indices[t];
                let sub_row: Vec<F> = feat_idx.iter().map(|&fi| row[fi]).collect();
                let sub_view = ndarray::Array1::from(sub_row);
                let leaf_idx = decision_tree::traverse(tree_nodes, &sub_view.view());
                match &tree_nodes[leaf_idx] {
                    Node::Leaf {
                        class_distribution: Some(dist),
                        ..
                    } => {
                        for (j, &p) in dist.iter().enumerate().take(n_classes) {
                            proba[[i, j]] = proba[[i, j]] + p;
                        }
                    }
                    Node::Leaf { value, .. } => {
                        let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                        if class_idx < n_classes {
                            proba[[i, class_idx]] = proba[[i, class_idx]] + F::one();
                        }
                    }
                    _ => {}
                }
            }
            for j in 0..n_classes {
                proba[[i, j]] = proba[[i, j]] / n_trees_f;
            }
        }
        Ok(proba)
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba). Mirrors
    /// sklearn's `ClassifierMixin.predict_log_proba`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        Ok(crate::log_proba(&proba))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for BaggingClassifier<F> {
    type Fitted = FittedBaggingClassifier<F>;
    type Error = FerroError;

    /// Fit the bagging classifier by building `n_estimators` decision trees in parallel.
    ///
    /// Each tree is trained on a (bootstrap) sample of the data with a random
    /// subset of features.
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
    ) -> Result<FittedBaggingClassifier<F>, FerroError> {
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
                context: "BaggingClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_samples <= 0.0 || self.max_samples > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_samples".into(),
                reason: "must be in (0.0, 1.0]".into(),
            });
        }
        if self.max_features <= 0.0 || self.max_features > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_features".into(),
                reason: "must be in (0.0, 1.0]".into(),
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

        let n_sample_draw = ((n_samples as f64) * self.max_samples).ceil().max(1.0) as usize;
        let n_feature_draw = ((n_features as f64) * self.max_features).ceil().max(1.0) as usize;
        let n_feature_draw = n_feature_draw.min(n_features);

        let params = decision_tree::TreeParams {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        };
        let bootstrap = self.bootstrap;
        let bootstrap_features = self.bootstrap_features;

        // Generate per-tree seeds sequentially for determinism.
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
        let results: Vec<(Vec<Node<F>>, Vec<usize>)> = tree_seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);

                // Sample indices.
                let sample_indices: Vec<usize> = if bootstrap {
                    // With replacement.
                    (0..n_sample_draw)
                        .map(|_| {
                            use rand::RngCore;
                            (rng.next_u64() as usize) % n_samples
                        })
                        .collect()
                } else {
                    // Without replacement.
                    rand_sample_indices(&mut rng, n_samples, n_sample_draw).into_vec()
                };

                // Feature indices.
                let feat_indices: Vec<usize> = if bootstrap_features {
                    // With replacement.
                    (0..n_feature_draw)
                        .map(|_| {
                            use rand::RngCore;
                            (rng.next_u64() as usize) % n_features
                        })
                        .collect()
                } else if n_feature_draw == n_features {
                    (0..n_features).collect()
                } else {
                    rand_sample_indices(&mut rng, n_features, n_feature_draw).into_vec()
                };

                let tree = build_classification_tree_with_feature_subset(
                    x,
                    &y_mapped,
                    n_classes,
                    &sample_indices,
                    &feat_indices,
                    &params,
                    ClassificationCriterion::Gini,
                );

                (tree, feat_indices)
            })
            .collect();

        let (trees, feature_indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let feature_importances = decision_tree::aggregate_tree_importances(
            &trees,
            Some(&feature_indices),
            None,
            n_features,
        );

        Ok(FittedBaggingClassifier {
            trees,
            feature_indices,
            classes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedBaggingClassifier<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBaggingClassifier<F> {
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

            for (t, tree_nodes) in self.trees.iter().enumerate() {
                // Build a subsetted row using only the features this tree was trained on.
                let feat_idx = &self.feature_indices[t];
                let sub_row: Vec<F> = feat_idx.iter().map(|&fi| row[fi]).collect();
                let sub_view = ndarray::Array1::from(sub_row);

                let leaf_idx = decision_tree::traverse(tree_nodes, &sub_view.view());
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

impl<F: Float + Send + Sync + 'static> HasClasses for FittedBaggingClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for BaggingClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedBaggingClassifierPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedBaggingClassifier<F>`.
struct FittedBaggingClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedBaggingClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedBaggingClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// BaggingRegressor
// ---------------------------------------------------------------------------

/// Bootstrap aggregation (bagging) regressor using decision trees.
///
/// Each estimator is trained on a bootstrap sample of the data, optionally
/// with a random subset of features. Final predictions are the mean across
/// all estimators.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_tree::BaggingRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((6, 1), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
/// ]).unwrap();
/// let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let model = BaggingRegressor::<f64>::new()
///     .with_n_estimators(10)
///     .with_random_state(42);
/// let fitted = model.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BaggingRegressor<F> {
    /// Number of estimators in the ensemble.
    pub n_estimators: usize,
    /// Fraction of samples to draw for each estimator (default 1.0).
    pub max_samples: f64,
    /// Fraction of features to draw for each estimator (default 1.0).
    pub max_features: f64,
    /// Whether to sample with replacement (default `true`).
    pub bootstrap: bool,
    /// Whether to sample features with replacement (default `false`).
    pub bootstrap_features: bool,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Maximum depth of each base decision tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> BaggingRegressor<F> {
    /// Create a new `BaggingRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 10`, `max_samples = 1.0`,
    /// `max_features = 1.0`, `bootstrap = true`,
    /// `bootstrap_features = false`, `random_state = None`,
    /// `max_depth = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 10,
            max_samples: 1.0,
            max_features: 1.0,
            bootstrap: true,
            bootstrap_features: false,
            random_state: None,
            max_depth: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of estimators.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the fraction of samples to draw per estimator.
    #[must_use]
    pub fn with_max_samples(mut self, frac: f64) -> Self {
        self.max_samples = frac;
        self
    }

    /// Set the fraction of features to draw per estimator.
    #[must_use]
    pub fn with_max_features(mut self, frac: f64) -> Self {
        self.max_features = frac;
        self
    }

    /// Set whether to sample with replacement.
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set whether to sample features with replacement.
    #[must_use]
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.bootstrap_features = bootstrap_features;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the maximum depth of each base decision tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
}

impl<F: Float> Default for BaggingRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedBaggingRegressor
// ---------------------------------------------------------------------------

/// A fitted bagging regressor.
///
/// Stores the ensemble of fitted decision trees and aggregates their
/// predictions by averaging.
#[derive(Debug, Clone)]
pub struct FittedBaggingRegressor<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Feature indices used by each tree.
    feature_indices: Vec<Vec<usize>>,
    /// Number of features in the original data.
    n_features: usize,
    /// Per-feature importance scores aggregated across the ensemble
    /// (normalized to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedBaggingRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> FittedBaggingRegressor<F> {
    /// Returns a reference to the individual tree node vectors.
    #[must_use]
    pub fn trees(&self) -> &[Vec<Node<F>>] {
        &self.trees
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for BaggingRegressor<F> {
    type Fitted = FittedBaggingRegressor<F>;
    type Error = FerroError;

    /// Fit the bagging regressor by building `n_estimators` decision trees in parallel.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedBaggingRegressor<F>, FerroError> {
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
                context: "BaggingRegressor requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_samples <= 0.0 || self.max_samples > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_samples".into(),
                reason: "must be in (0.0, 1.0]".into(),
            });
        }
        if self.max_features <= 0.0 || self.max_features > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "max_features".into(),
                reason: "must be in (0.0, 1.0]".into(),
            });
        }

        let n_sample_draw = ((n_samples as f64) * self.max_samples).ceil().max(1.0) as usize;
        let n_feature_draw = ((n_features as f64) * self.max_features).ceil().max(1.0) as usize;
        let n_feature_draw = n_feature_draw.min(n_features);

        let params = decision_tree::TreeParams {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        };
        let bootstrap = self.bootstrap;
        let bootstrap_features = self.bootstrap_features;

        // Generate per-tree seeds sequentially for determinism.
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
        let results: Vec<(Vec<Node<F>>, Vec<usize>)> = tree_seeds
            .par_iter()
            .map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);

                // Sample indices.
                let sample_indices: Vec<usize> = if bootstrap {
                    (0..n_sample_draw)
                        .map(|_| {
                            use rand::RngCore;
                            (rng.next_u64() as usize) % n_samples
                        })
                        .collect()
                } else {
                    rand_sample_indices(&mut rng, n_samples, n_sample_draw).into_vec()
                };

                // Feature indices.
                let feat_indices: Vec<usize> = if bootstrap_features {
                    (0..n_feature_draw)
                        .map(|_| {
                            use rand::RngCore;
                            (rng.next_u64() as usize) % n_features
                        })
                        .collect()
                } else if n_feature_draw == n_features {
                    (0..n_features).collect()
                } else {
                    rand_sample_indices(&mut rng, n_features, n_feature_draw).into_vec()
                };

                let tree = build_regression_tree_with_feature_subset(
                    x,
                    y,
                    &sample_indices,
                    &feat_indices,
                    &params,
                );

                (tree, feat_indices)
            })
            .collect();

        let (trees, feature_indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let feature_importances = decision_tree::aggregate_tree_importances(
            &trees,
            Some(&feature_indices),
            None,
            n_features,
        );

        Ok(FittedBaggingRegressor {
            trees,
            feature_indices,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBaggingRegressor<F> {
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

            for (t, tree_nodes) in self.trees.iter().enumerate() {
                let feat_idx = &self.feature_indices[t];
                let sub_row: Vec<F> = feat_idx.iter().map(|&fi| row[fi]).collect();
                let sub_view = ndarray::Array1::from(sub_row);

                let leaf_idx = decision_tree::traverse(tree_nodes, &sub_view.view());
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    sum = sum + value;
                }
            }

            predictions[i] = sum / n_trees_f;
        }

        Ok(predictions)
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for BaggingRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedBaggingRegressor<F> {
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -- BaggingClassifier tests --

    #[test]
    fn test_bagging_classifier_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
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
    fn test_bagging_classifier_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_bagging_classifier_has_classes() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_bagging_classifier_feature_subsample() {
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_bagging_classifier_no_bootstrap() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_bootstrap(false)
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
    fn test_bagging_classifier_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1];

        let model = BaggingClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = BaggingClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_classifier_invalid_max_samples() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = BaggingClassifier::<f64>::new().with_max_samples(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model = BaggingClassifier::<f64>::new().with_max_samples(1.5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_classifier_predict_shape_mismatch() {
        let x_train = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y_train = array![0, 0, 0, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &y_train).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_bagging_classifier_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_bagging_classifier_with_max_depth() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_max_depth(Some(2))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    // -- BaggingRegressor tests --

    #[test]
    fn test_bagging_regressor_simple() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        // Predictions should be reasonable approximations.
        for i in 0..6 {
            assert!((preds[i] - y[i]).abs() < 2.0);
        }
    }

    #[test]
    fn test_bagging_regressor_reproducibility() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_bagging_regressor_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0]);

        let model = BaggingRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = BaggingRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_regressor_predict_shape_mismatch() {
        let x_train = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y_train = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &y_train).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_bagging_regressor_feature_subsample() {
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_bagging_regressor_with_max_depth() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_max_depth(Some(2))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_bagging_classifier_default() {
        let model = BaggingClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 10);
        assert!((model.max_samples - 1.0).abs() < f64::EPSILON);
        assert!((model.max_features - 1.0).abs() < f64::EPSILON);
        assert!(model.bootstrap);
        assert!(!model.bootstrap_features);
        assert!(model.random_state.is_none());
        assert!(model.max_depth.is_none());
    }

    #[test]
    fn test_bagging_regressor_default() {
        let model = BaggingRegressor::<f64>::default();
        assert_eq!(model.n_estimators, 10);
        assert!((model.max_samples - 1.0).abs() < f64::EPSILON);
        assert!((model.max_features - 1.0).abs() < f64::EPSILON);
        assert!(model.bootstrap);
        assert!(!model.bootstrap_features);
        assert!(model.random_state.is_none());
        assert!(model.max_depth.is_none());
    }

    #[test]
    fn test_bagging_classifier_zero_estimators() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = BaggingClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_regressor_zero_estimators() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        let model = BaggingRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_classifier_bootstrap_features() {
        let x = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_max_features(0.5)
            .with_bootstrap_features(true)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_bagging_regressor_no_bootstrap() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = BaggingRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_bootstrap(false)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_bagging_classifier_max_samples_subsample() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BaggingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_max_samples(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }
}
