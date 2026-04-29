//! Extremely randomized trees ensemble classifiers and regressors.
//!
//! This module provides [`ExtraTreesClassifier`] and [`ExtraTreesRegressor`],
//! which build ensembles of extremely randomized trees. Unlike
//! [`RandomForestClassifier`](crate::RandomForestClassifier), ExtraTrees
//! ensembles do **not** bootstrap by default: all trees see all samples, and
//! randomness comes solely from the random split thresholds and random feature
//! subsets at each node.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::ExtraTreesClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = ExtraTreesClassifier::<f64>::new()
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
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::decision_tree::{
    ClassificationCriterion, Node, TreeParams, compute_feature_importances, traverse,
};
use crate::extra_tree::{
    build_extra_classification_tree_for_ensemble, build_extra_regression_tree_for_ensemble,
};
use crate::random_forest::MaxFeatures;

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

/// Internal tree parameter struct helper.
fn make_tree_params(
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> TreeParams {
    TreeParams {
        max_depth,
        min_samples_split,
        min_samples_leaf,
    }
}

// ---------------------------------------------------------------------------
// ExtraTreesClassifier
// ---------------------------------------------------------------------------

/// Extremely randomized trees classifier (ensemble).
///
/// Builds an ensemble of [`ExtraTreeClassifier`](crate::ExtraTreeClassifier)
/// base estimators, each using random split thresholds and random feature
/// subsets at every node. Final predictions are made by majority vote.
///
/// Unlike [`RandomForestClassifier`](crate::RandomForestClassifier), bootstrap
/// sampling is **disabled** by default. Randomness comes from the random
/// thresholds and random feature subsets at each split.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreesClassifier<F> {
    /// Number of trees in the ensemble.
    pub n_estimators: usize,
    /// Maximum depth of each tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Whether to use bootstrap sampling. Default is `false`.
    pub bootstrap: bool,
    /// Splitting criterion.
    pub criterion: ClassificationCriterion,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    /// Number of parallel jobs. `None` means use all available cores.
    pub n_jobs: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> ExtraTreesClassifier<F> {
    /// Create a new `ExtraTreesClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_depth = None`,
    /// `max_features = Sqrt`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `bootstrap = false`,
    /// `criterion = Gini`, `random_state = None`, `n_jobs = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: false,
            criterion: ClassificationCriterion::Gini,
            random_state: None,
            n_jobs: None,
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

    /// Set the maximum features strategy.
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set whether to use bootstrap sampling.
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the number of parallel jobs.
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }
}

impl<F: Float> Default for ExtraTreesClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedExtraTreesClassifier
// ---------------------------------------------------------------------------

/// A fitted extremely randomized trees classifier (ensemble).
///
/// Stores the ensemble of fitted extra-trees and aggregates their
/// predictions by majority vote.
#[derive(Debug, Clone)]
pub struct FittedExtraTreesClassifier<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (mean decrease in impurity, normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedExtraTreesClassifier<F> {
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

    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Predict class probabilities for each sample by averaging tree predictions.
    ///
    /// Returns a 2-D array of shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
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
        let mut proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = x.row(i);
            for tree_nodes in &self.trees {
                let leaf_idx = traverse(tree_nodes, &row);
                if let Node::Leaf {
                    class_distribution: Some(ref dist),
                    ..
                } = tree_nodes[leaf_idx]
                {
                    for (j, &p) in dist.iter().enumerate() {
                        proba[[i, j]] = proba[[i, j]] + p;
                    }
                }
            }
            // Average across trees.
            for j in 0..n_classes {
                proba[[i, j]] = proba[[i, j]] / n_trees_f;
            }
        }
        Ok(proba)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for ExtraTreesClassifier<F> {
    type Fitted = FittedExtraTreesClassifier<F>;
    type Error = FerroError;

    /// Fit the ensemble by building `n_estimators` extra-trees in parallel.
    ///
    /// Each tree uses random split thresholds and random feature subsets at
    /// every node. If `bootstrap` is `true`, each tree is trained on a
    /// bootstrap sample; otherwise all samples are used.
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
    ) -> Result<FittedExtraTreesClassifier<F>, FerroError> {
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
                context: "ExtraTreesClassifier requires at least one sample".into(),
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
        let bootstrap = self.bootstrap;

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

        // Optionally configure thread pool.
        let trees: Vec<Vec<Node<F>>> = if let Some(n_jobs) = self.n_jobs {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
            pool.install(|| {
                tree_seeds
                    .par_iter()
                    .map(|&seed| {
                        build_single_classification_tree(
                            x,
                            &y_mapped,
                            n_classes,
                            n_samples,
                            n_features,
                            max_features_n,
                            &params,
                            criterion,
                            bootstrap,
                            seed,
                        )
                    })
                    .collect()
            })
        } else {
            tree_seeds
                .par_iter()
                .map(|&seed| {
                    build_single_classification_tree(
                        x,
                        &y_mapped,
                        n_classes,
                        n_samples,
                        n_features,
                        max_features_n,
                        &params,
                        criterion,
                        bootstrap,
                        seed,
                    )
                })
                .collect()
        };

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

        Ok(FittedExtraTreesClassifier {
            trees,
            classes,
            n_features,
            feature_importances: total_importances,
        })
    }
}

/// Build a single classification extra-tree (used by parallel dispatch).
#[allow(clippy::too_many_arguments)]
fn build_single_classification_tree<F: Float>(
    x: &Array2<F>,
    y_mapped: &[usize],
    n_classes: usize,
    n_samples: usize,
    n_features: usize,
    max_features_n: usize,
    params: &TreeParams,
    criterion: ClassificationCriterion,
    bootstrap: bool,
    seed: u64,
) -> Vec<Node<F>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let indices: Vec<usize> = if bootstrap {
        use rand::RngCore;
        (0..n_samples)
            .map(|_| (rng.next_u64() as usize) % n_samples)
            .collect()
    } else {
        (0..n_samples).collect()
    };

    build_extra_classification_tree_for_ensemble(
        x,
        y_mapped,
        n_classes,
        &indices,
        None, // feature selection happens inside the tree builder
        params,
        criterion,
        n_features,
        max_features_n,
        &mut rng,
    )
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedExtraTreesClassifier<F> {
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
                let leaf_idx = traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    let class_idx = value.to_f64().map_or(0, |f| f.round() as usize);
                    if class_idx < n_classes {
                        votes[class_idx] += 1;
                    }
                }
            }

            let winner = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &count)| count)
                .map_or(0, |(idx, _)| idx);
            predictions[i] = self.classes[winner];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedExtraTreesClassifier<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedExtraTreesClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for ExtraTreesClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedExtraTreesClassifierPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedExtraTreesClassifier<F>`.
struct FittedExtraTreesClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedExtraTreesClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedExtraTreesClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// ExtraTreesRegressor
// ---------------------------------------------------------------------------

/// Extremely randomized trees regressor (ensemble).
///
/// Builds an ensemble of [`ExtraTreeRegressor`](crate::ExtraTreeRegressor)
/// base estimators, each using random split thresholds and random feature
/// subsets at every node. Final predictions are the mean across all trees.
///
/// Unlike [`RandomForestRegressor`](crate::RandomForestRegressor), bootstrap
/// sampling is **disabled** by default.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreesRegressor<F> {
    /// Number of trees in the ensemble.
    pub n_estimators: usize,
    /// Maximum depth of each tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Whether to use bootstrap sampling. Default is `false`.
    pub bootstrap: bool,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    /// Number of parallel jobs. `None` means use all available cores.
    pub n_jobs: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> ExtraTreesRegressor<F> {
    /// Create a new `ExtraTreesRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_depth = None`,
    /// `max_features = All`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `bootstrap = false`,
    /// `random_state = None`, `n_jobs = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            bootstrap: false,
            random_state: None,
            n_jobs: None,
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

    /// Set the maximum features strategy.
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set whether to use bootstrap sampling.
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the number of parallel jobs.
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }
}

impl<F: Float> Default for ExtraTreesRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedExtraTreesRegressor
// ---------------------------------------------------------------------------

/// A fitted extremely randomized trees regressor (ensemble).
///
/// Stores the ensemble of fitted extra-trees and aggregates their
/// predictions by averaging.
#[derive(Debug, Clone)]
pub struct FittedExtraTreesRegressor<F> {
    /// Individual tree node vectors.
    trees: Vec<Vec<Node<F>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (mean decrease in impurity, normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedExtraTreesRegressor<F> {
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

    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for ExtraTreesRegressor<F> {
    type Fitted = FittedExtraTreesRegressor<F>;
    type Error = FerroError;

    /// Fit the ensemble by building `n_estimators` extra-trees in parallel.
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
    ) -> Result<FittedExtraTreesRegressor<F>, FerroError> {
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
                context: "ExtraTreesRegressor requires at least one sample".into(),
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
        let bootstrap = self.bootstrap;

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
        let trees: Vec<Vec<Node<F>>> = if let Some(n_jobs) = self.n_jobs {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
            pool.install(|| {
                tree_seeds
                    .par_iter()
                    .map(|&seed| {
                        build_single_regression_tree(
                            x,
                            y,
                            n_samples,
                            n_features,
                            max_features_n,
                            &params,
                            bootstrap,
                            seed,
                        )
                    })
                    .collect()
            })
        } else {
            tree_seeds
                .par_iter()
                .map(|&seed| {
                    build_single_regression_tree(
                        x,
                        y,
                        n_samples,
                        n_features,
                        max_features_n,
                        &params,
                        bootstrap,
                        seed,
                    )
                })
                .collect()
        };

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

        Ok(FittedExtraTreesRegressor {
            trees,
            n_features,
            feature_importances: total_importances,
        })
    }
}

/// Build a single regression extra-tree (used by parallel dispatch).
#[allow(clippy::too_many_arguments)]
fn build_single_regression_tree<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    n_samples: usize,
    n_features: usize,
    max_features_n: usize,
    params: &TreeParams,
    bootstrap: bool,
    seed: u64,
) -> Vec<Node<F>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let indices: Vec<usize> = if bootstrap {
        use rand::RngCore;
        (0..n_samples)
            .map(|_| (rng.next_u64() as usize) % n_samples)
            .collect()
    } else {
        (0..n_samples).collect()
    };

    build_extra_regression_tree_for_ensemble(
        x,
        y,
        &indices,
        None, // feature selection happens inside the tree builder
        params,
        n_features,
        max_features_n,
        &mut rng,
    )
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedExtraTreesRegressor<F> {
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
                let leaf_idx = traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    sum = sum + value;
                }
            }

            predictions[i] = sum / n_trees_f;
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedExtraTreesRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for ExtraTreesRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedExtraTreesRegressor<F> {
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
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- ExtraTreesClassifier tests --

    #[test]
    fn test_ensemble_classifier_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Should classify all training points correctly.
        assert_eq!(preds, y);
    }

    #[test]
    fn test_ensemble_classifier_no_bootstrap() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // Default: no bootstrap.
        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        assert!(!model.bootstrap);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    #[test]
    fn test_ensemble_classifier_with_bootstrap() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_bootstrap(true)
            .with_random_state(42);
        assert!(model.bootstrap);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_ensemble_classifier_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (6, 2));
        for i in 0..6 {
            let row_sum = proba.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ensemble_classifier_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        let total: f64 = importances.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
        // Feature 0 should dominate (feature 1 is constant).
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_ensemble_classifier_n_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(15)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 15);
    }

    #[test]
    fn test_ensemble_classifier_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 3, 3, 3]; // non-contiguous

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 3]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_ensemble_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0];
        let model = ExtraTreesClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = ExtraTreesClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_classifier_zero_estimators() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![0, 1];
        let model = ExtraTreesClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_classifier_deterministic() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model1 = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(123);
        let model2 = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(123);

        let preds1 = model1.fit(&x, &y).unwrap().predict(&x).unwrap();
        let preds2 = model2.fit(&x, &y).unwrap().predict(&x).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_ensemble_classifier_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    // -- ExtraTreesRegressor tests --

    #[test]
    fn test_ensemble_regressor_simple() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        // Ensemble should approximate the training data well.
        for i in 0..6 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_ensemble_regressor_constant_target() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![5.0, 5.0, 5.0, 5.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in &preds {
            assert_relative_eq!(p, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ensemble_regressor_no_bootstrap() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        assert!(!model.bootstrap);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_ensemble_regressor_with_bootstrap() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_bootstrap(true)
            .with_random_state(42);
        assert!(model.bootstrap);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_ensemble_regressor_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        let total: f64 = importances.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_ensemble_regressor_n_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(7)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 7);
    }

    #[test]
    fn test_ensemble_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0];
        let model = ExtraTreesRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let model = ExtraTreesRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_regressor_zero_estimators() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0];
        let model = ExtraTreesRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ensemble_regressor_deterministic() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model1 = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(99);
        let model2 = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(99);

        let preds1 = model1.fit(&x, &y).unwrap().predict(&x).unwrap();
        let preds2 = model2.fit(&x, &y).unwrap().predict(&x).unwrap();

        for i in 0..6 {
            assert_relative_eq!(preds1[i], preds2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_ensemble_regressor_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    // -- Builder tests --

    #[test]
    fn test_ensemble_classifier_builder() {
        let model = ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_max_depth(Some(5))
            .with_min_samples_split(10)
            .with_min_samples_leaf(3)
            .with_max_features(MaxFeatures::Log2)
            .with_bootstrap(true)
            .with_criterion(ClassificationCriterion::Entropy)
            .with_random_state(42)
            .with_n_jobs(4);

        assert_eq!(model.n_estimators, 50);
        assert_eq!(model.max_depth, Some(5));
        assert_eq!(model.min_samples_split, 10);
        assert_eq!(model.min_samples_leaf, 3);
        assert_eq!(model.max_features, MaxFeatures::Log2);
        assert!(model.bootstrap);
        assert_eq!(model.criterion, ClassificationCriterion::Entropy);
        assert_eq!(model.random_state, Some(42));
        assert_eq!(model.n_jobs, Some(4));
    }

    #[test]
    fn test_ensemble_regressor_builder() {
        let model = ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(25)
            .with_max_depth(Some(8))
            .with_min_samples_split(5)
            .with_min_samples_leaf(2)
            .with_max_features(MaxFeatures::Fraction(0.5))
            .with_bootstrap(true)
            .with_random_state(99)
            .with_n_jobs(2);

        assert_eq!(model.n_estimators, 25);
        assert_eq!(model.max_depth, Some(8));
        assert_eq!(model.min_samples_split, 5);
        assert_eq!(model.min_samples_leaf, 2);
        assert_eq!(model.max_features, MaxFeatures::Fraction(0.5));
        assert!(model.bootstrap);
        assert_eq!(model.random_state, Some(99));
        assert_eq!(model.n_jobs, Some(2));
    }

    #[test]
    fn test_ensemble_classifier_default() {
        let model = ExtraTreesClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert_eq!(model.max_depth, None);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
        assert_eq!(model.max_features, MaxFeatures::Sqrt);
        assert!(!model.bootstrap);
        assert_eq!(model.criterion, ClassificationCriterion::Gini);
        assert_eq!(model.random_state, None);
        assert_eq!(model.n_jobs, None);
    }

    #[test]
    fn test_ensemble_regressor_default() {
        let model = ExtraTreesRegressor::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert_eq!(model.max_depth, None);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
        assert_eq!(model.max_features, MaxFeatures::All);
        assert!(!model.bootstrap);
        assert_eq!(model.random_state, None);
        assert_eq!(model.n_jobs, None);
    }
}
