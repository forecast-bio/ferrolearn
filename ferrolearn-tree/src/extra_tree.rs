//! Extremely randomized tree classifiers and regressors.
//!
//! This module provides [`ExtraTreeClassifier`] and [`ExtraTreeRegressor`],
//! which are variants of decision trees where split thresholds are chosen
//! randomly rather than via exhaustive search. For each candidate feature,
//! a random threshold is drawn uniformly between the feature's minimum and
//! maximum values in the current node.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::ExtraTreeClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = ExtraTreeClassifier::<f64>::new()
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
use rand::seq::index::sample as rand_sample_indices;
use serde::{Deserialize, Serialize};

use crate::decision_tree::{
    ClassificationCriterion, Node, RegressionCriterion, TreeParams, compute_feature_importances,
    traverse,
};
use crate::random_forest::MaxFeatures;

// ---------------------------------------------------------------------------
// Internal data structs for extra-tree building
// ---------------------------------------------------------------------------

/// Data references for classification extra-tree building.
struct ClassificationData<'a, F> {
    x: &'a Array2<F>,
    y: &'a [usize],
    n_classes: usize,
    feature_indices: Option<&'a [usize]>,
    criterion: ClassificationCriterion,
}

/// Data references for regression extra-tree building.
struct RegressionData<'a, F> {
    x: &'a Array2<F>,
    y: &'a Array1<F>,
    feature_indices: Option<&'a [usize]>,
}

// ---------------------------------------------------------------------------
// ExtraTreeClassifier
// ---------------------------------------------------------------------------

/// Extremely randomized tree classifier.
///
/// Like a [`DecisionTreeClassifier`](crate::DecisionTreeClassifier), but split
/// thresholds are chosen randomly rather than via exhaustive search. For each
/// candidate feature, a random threshold is drawn uniformly between the
/// feature's minimum and maximum values in the current node.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreeClassifier<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Splitting criterion.
    pub criterion: ClassificationCriterion,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> ExtraTreeClassifier<F> {
    /// Create a new `ExtraTreeClassifier` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `max_features = Sqrt`,
    /// `criterion = Gini`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            criterion: ClassificationCriterion::Gini,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
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
}

impl<F: Float> Default for ExtraTreeClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedExtraTreeClassifier
// ---------------------------------------------------------------------------

/// A fitted extremely randomized tree classifier.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
/// Implements [`Predict`] for generating class predictions and
/// [`HasFeatureImportances`] for inspecting per-feature importance scores.
#[derive(Debug, Clone)]
pub struct FittedExtraTreeClassifier<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Sorted unique class labels observed during training.
    classes: Vec<usize>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedExtraTreeClassifier<F> {
    /// Returns a reference to the flat node storage of the tree.
    #[must_use]
    pub fn nodes(&self) -> &[Node<F>] {
        &self.nodes
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Predict class probabilities for each sample.
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
        let mut proba = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse(&self.nodes, &row);
            if let Node::Leaf {
                class_distribution: Some(ref dist),
                ..
            } = self.nodes[leaf]
            {
                for (j, &p) in dist.iter().enumerate() {
                    proba[[i, j]] = p;
                }
            }
        }
        Ok(proba)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for ExtraTreeClassifier<F> {
    type Fitted = FittedExtraTreeClassifier<F>;
    type Error = FerroError;

    /// Fit the extra-tree classifier on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedExtraTreeClassifier<F>, FerroError> {
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
                context: "ExtraTreeClassifier requires at least one sample".into(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_split".into(),
                reason: "must be at least 2".into(),
            });
        }
        if self.min_samples_leaf < 1 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_leaf".into(),
                reason: "must be at least 1".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Map class labels to indices 0..n_classes.
        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        let indices: Vec<usize> = (0..n_samples).collect();

        let max_features_n = resolve_max_features(self.max_features, n_features);

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let data = ClassificationData {
            x,
            y: &y_mapped,
            n_classes,
            feature_indices: None,
            criterion: self.criterion,
        };
        let params = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let mut nodes: Vec<Node<F>> = Vec::new();
        build_extra_classification_tree(
            &data,
            &indices,
            &mut nodes,
            0,
            &params,
            n_features,
            max_features_n,
            &mut rng,
        );

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedExtraTreeClassifier {
            nodes,
            classes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedExtraTreeClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
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
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = float_to_usize(value);
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedExtraTreeClassifier<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedExtraTreeClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for ExtraTreeClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedExtraTreeClassifierPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedExtraTreeClassifier<F>`.
struct FittedExtraTreeClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedExtraTreeClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedExtraTreeClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// ExtraTreeRegressor
// ---------------------------------------------------------------------------

/// Extremely randomized tree regressor.
///
/// Like a [`DecisionTreeRegressor`](crate::DecisionTreeRegressor), but split
/// thresholds are chosen randomly rather than via exhaustive search. For each
/// candidate feature, a random threshold is drawn uniformly between the
/// feature's minimum and maximum values in the current node.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTreeRegressor<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Strategy for the number of features considered at each split.
    pub max_features: MaxFeatures,
    /// Splitting criterion.
    pub criterion: RegressionCriterion,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> ExtraTreeRegressor<F> {
    /// Create a new `ExtraTreeRegressor` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `max_features = All`,
    /// `criterion = MSE`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::All,
            criterion: RegressionCriterion::Mse,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
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

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: RegressionCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for ExtraTreeRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedExtraTreeRegressor
// ---------------------------------------------------------------------------

/// A fitted extremely randomized tree regressor.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
#[derive(Debug, Clone)]
pub struct FittedExtraTreeRegressor<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedExtraTreeRegressor<F> {
    /// Returns a reference to the flat node storage of the tree.
    #[must_use]
    pub fn nodes(&self) -> &[Node<F>] {
        &self.nodes
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for ExtraTreeRegressor<F> {
    type Fitted = FittedExtraTreeRegressor<F>;
    type Error = FerroError;

    /// Fit the extra-tree regressor on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedExtraTreeRegressor<F>, FerroError> {
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
                context: "ExtraTreeRegressor requires at least one sample".into(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_split".into(),
                reason: "must be at least 2".into(),
            });
        }
        if self.min_samples_leaf < 1 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_leaf".into(),
                reason: "must be at least 1".into(),
            });
        }

        let indices: Vec<usize> = (0..n_samples).collect();
        let max_features_n = resolve_max_features(self.max_features, n_features);

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let data = RegressionData {
            x,
            y,
            feature_indices: None,
        };
        let params = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let mut nodes: Vec<Node<F>> = Vec::new();
        build_extra_regression_tree(
            &data,
            &indices,
            &mut nodes,
            0,
            &params,
            n_features,
            max_features_n,
            &mut rng,
        );

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedExtraTreeRegressor {
            nodes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedExtraTreeRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
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
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let leaf = traverse(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = value;
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedExtraTreeRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for ExtraTreeRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedExtraTreeRegressor<F> {
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Internal: helpers
// ---------------------------------------------------------------------------

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

/// Convert a `Float` value to `usize` (for class labels stored as floats).
fn float_to_usize<F: Float>(v: F) -> usize {
    v.to_f64().map(|f| f.round() as usize).unwrap_or(0)
}

/// Generate a uniform random float in `[min_val, max_val]`.
fn random_threshold<F: Float>(rng: &mut StdRng, min_val: F, max_val: F) -> F {
    use rand::RngCore;
    // Generate a random f64 in [0, 1) and scale to [min_val, max_val].
    let u = (rng.next_u64() as f64) / (u64::MAX as f64);
    let range = max_val - min_val;
    min_val + F::from(u).unwrap() * range
}

/// Compute the Gini impurity for a set of class counts.
fn gini_impurity<F: Float>(class_counts: &[usize], total: usize) -> F {
    if total == 0 {
        return F::zero();
    }
    let total_f = F::from(total).unwrap();
    let mut impurity = F::one();
    for &count in class_counts {
        let p = F::from(count).unwrap() / total_f;
        impurity = impurity - p * p;
    }
    impurity
}

/// Compute the Shannon entropy for a set of class counts.
fn entropy_impurity<F: Float>(class_counts: &[usize], total: usize) -> F {
    if total == 0 {
        return F::zero();
    }
    let total_f = F::from(total).unwrap();
    let mut ent = F::zero();
    for &count in class_counts {
        if count > 0 {
            let p = F::from(count).unwrap() / total_f;
            ent = ent - p * p.ln();
        }
    }
    ent
}

/// Compute impurity for a given classification criterion.
fn compute_impurity<F: Float>(
    class_counts: &[usize],
    total: usize,
    criterion: ClassificationCriterion,
) -> F {
    match criterion {
        ClassificationCriterion::Gini => gini_impurity(class_counts, total),
        ClassificationCriterion::Entropy => entropy_impurity(class_counts, total),
    }
}

/// Create a classification leaf node and return its index.
fn make_classification_leaf<F: Float>(
    nodes: &mut Vec<Node<F>>,
    class_counts: &[usize],
    n_classes: usize,
    n_samples: usize,
) -> usize {
    let majority_class = class_counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let total_f = if n_samples > 0 {
        F::from(n_samples).unwrap()
    } else {
        F::one()
    };
    let distribution: Vec<F> = (0..n_classes)
        .map(|c| F::from(class_counts[c]).unwrap() / total_f)
        .collect();

    let idx = nodes.len();
    nodes.push(Node::Leaf {
        value: F::from(majority_class).unwrap(),
        class_distribution: Some(distribution),
        n_samples,
    });
    idx
}

/// Compute the mean of target values for the given indices.
fn mean_value<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    if indices.is_empty() {
        return F::zero();
    }
    let sum: F = indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b);
    sum / F::from(indices.len()).unwrap()
}

// ---------------------------------------------------------------------------
// Extra-tree classification building
// ---------------------------------------------------------------------------

/// Build an extra-tree classification tree recursively with random thresholds.
///
/// At each node, a random subset of features is considered, and for each feature
/// a random threshold is drawn uniformly between the feature's min and max in the
/// current node.
#[allow(clippy::too_many_arguments)]
fn build_extra_classification_tree<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> usize {
    let n = indices.len();

    let mut class_counts = vec![0usize; data.n_classes];
    for &i in indices {
        class_counts[data.y[i]] += 1;
    }

    let should_stop = n < params.min_samples_split
        || params.max_depth.is_some_and(|d| depth >= d)
        || class_counts.iter().filter(|&&c| c > 0).count() <= 1;

    if should_stop {
        return make_classification_leaf(nodes, &class_counts, data.n_classes, n);
    }

    let best = find_random_classification_split(
        data,
        indices,
        params.min_samples_leaf,
        n_features,
        max_features_n,
        rng,
    );

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        // Ensure both children have at least min_samples_leaf.
        if left_indices.len() < params.min_samples_leaf
            || right_indices.len() < params.min_samples_leaf
        {
            return make_classification_leaf(nodes, &class_counts, data.n_classes, n);
        }

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_extra_classification_tree(
            data,
            &left_indices,
            nodes,
            depth + 1,
            params,
            n_features,
            max_features_n,
            rng,
        );
        let right_idx = build_extra_classification_tree(
            data,
            &right_indices,
            nodes,
            depth + 1,
            params,
            n_features,
            max_features_n,
            rng,
        );

        nodes[node_idx] = Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: left_idx,
            right: right_idx,
            impurity_decrease: best_impurity_decrease,
            n_samples: n,
        };

        node_idx
    } else {
        make_classification_leaf(nodes, &class_counts, data.n_classes, n)
    }
}

/// Find the best random split for a classification node.
///
/// For each candidate feature (from a random subset), pick a random threshold
/// between min and max of that feature in the current node. Return the split
/// with the largest impurity decrease, or `None` if no valid split exists.
#[allow(clippy::too_many_arguments)]
fn find_random_classification_split<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();

    let mut parent_counts = vec![0usize; data.n_classes];
    for &i in indices {
        parent_counts[data.y[i]] += 1;
    }
    let parent_impurity = compute_impurity::<F>(&parent_counts, n, data.criterion);

    let mut best_score = F::neg_infinity();
    let mut best_feature = 0;
    let mut best_threshold = F::zero();

    // Select random feature subset.
    let feature_subset: Vec<usize> = if let Some(feat_indices) = data.feature_indices {
        // If a feature subset is provided externally, sample from it.
        let k = max_features_n.min(feat_indices.len());
        rand_sample_indices(rng, feat_indices.len(), k)
            .into_vec()
            .into_iter()
            .map(|i| feat_indices[i])
            .collect()
    } else {
        let k = max_features_n.min(n_features);
        rand_sample_indices(rng, n_features, k).into_vec()
    };

    for feat in feature_subset {
        // Find min and max of this feature in the current node.
        let mut feat_min = F::infinity();
        let mut feat_max = F::neg_infinity();
        for &i in indices {
            let val = data.x[[i, feat]];
            if val < feat_min {
                feat_min = val;
            }
            if val > feat_max {
                feat_max = val;
            }
        }

        // Skip constant features.
        if feat_min >= feat_max {
            continue;
        }

        // Draw a random threshold uniformly in (min, max).
        let threshold = random_threshold(rng, feat_min, feat_max);

        // Count left and right.
        let mut left_counts = vec![0usize; data.n_classes];
        let mut right_counts = vec![0usize; data.n_classes];
        let mut left_n = 0usize;

        for &i in indices {
            let cls = data.y[i];
            if data.x[[i, feat]] <= threshold {
                left_counts[cls] += 1;
                left_n += 1;
            } else {
                right_counts[cls] += 1;
            }
        }

        let right_n = n - left_n;
        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }

        let left_impurity = compute_impurity::<F>(&left_counts, left_n, data.criterion);
        let right_impurity = compute_impurity::<F>(&right_counts, right_n, data.criterion);
        let left_weight = F::from(left_n).unwrap() / n_f;
        let right_weight = F::from(right_n).unwrap() / n_f;
        let weighted_child_impurity = left_weight * left_impurity + right_weight * right_impurity;
        let impurity_decrease = parent_impurity - weighted_child_impurity;

        if impurity_decrease > best_score {
            best_score = impurity_decrease;
            best_feature = feat;
            best_threshold = threshold;
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Extra-tree regression building
// ---------------------------------------------------------------------------

/// Build an extra-tree regression tree recursively with random thresholds.
#[allow(clippy::too_many_arguments)]
fn build_extra_regression_tree<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> usize {
    let n = indices.len();
    let mean = mean_value(data.y, indices);

    let should_stop = n < params.min_samples_split || params.max_depth.is_some_and(|d| depth >= d);

    if should_stop {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: mean,
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    // Check if variance is essentially zero.
    let parent_sum_sq: F = indices
        .iter()
        .map(|&i| {
            let diff = data.y[i] - mean;
            diff * diff
        })
        .fold(F::zero(), |a, b| a + b);
    let parent_mse = parent_sum_sq / F::from(n).unwrap();

    if parent_mse <= F::epsilon() {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: mean,
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    let best = find_random_regression_split(
        data,
        indices,
        params.min_samples_leaf,
        n_features,
        max_features_n,
        rng,
    );

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        // Ensure both children have at least min_samples_leaf.
        if left_indices.len() < params.min_samples_leaf
            || right_indices.len() < params.min_samples_leaf
        {
            let idx = nodes.len();
            nodes.push(Node::Leaf {
                value: mean,
                class_distribution: None,
                n_samples: n,
            });
            return idx;
        }

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_extra_regression_tree(
            data,
            &left_indices,
            nodes,
            depth + 1,
            params,
            n_features,
            max_features_n,
            rng,
        );
        let right_idx = build_extra_regression_tree(
            data,
            &right_indices,
            nodes,
            depth + 1,
            params,
            n_features,
            max_features_n,
            rng,
        );

        nodes[node_idx] = Node::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: left_idx,
            right: right_idx,
            impurity_decrease: best_impurity_decrease,
            n_samples: n,
        };

        node_idx
    } else {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: mean,
            class_distribution: None,
            n_samples: n,
        });
        idx
    }
}

/// Find the best random split for a regression node.
///
/// For each candidate feature (from a random subset), pick a random threshold
/// between min and max of that feature in the current node. Return the split
/// with the largest MSE decrease, or `None` if no valid split exists.
#[allow(clippy::too_many_arguments)]
fn find_random_regression_split<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();

    let parent_sum: F = indices
        .iter()
        .map(|&i| data.y[i])
        .fold(F::zero(), |a, b| a + b);
    let parent_sum_sq: F = indices
        .iter()
        .map(|&i| data.y[i] * data.y[i])
        .fold(F::zero(), |a, b| a + b);
    let parent_mse = parent_sum_sq / n_f - (parent_sum / n_f) * (parent_sum / n_f);

    let mut best_score = F::neg_infinity();
    let mut best_feature = 0;
    let mut best_threshold = F::zero();

    // Select random feature subset.
    let feature_subset: Vec<usize> = if let Some(feat_indices) = data.feature_indices {
        let k = max_features_n.min(feat_indices.len());
        rand_sample_indices(rng, feat_indices.len(), k)
            .into_vec()
            .into_iter()
            .map(|i| feat_indices[i])
            .collect()
    } else {
        let k = max_features_n.min(n_features);
        rand_sample_indices(rng, n_features, k).into_vec()
    };

    for feat in feature_subset {
        // Find min and max of this feature in the current node.
        let mut feat_min = F::infinity();
        let mut feat_max = F::neg_infinity();
        for &i in indices {
            let val = data.x[[i, feat]];
            if val < feat_min {
                feat_min = val;
            }
            if val > feat_max {
                feat_max = val;
            }
        }

        // Skip constant features.
        if feat_min >= feat_max {
            continue;
        }

        // Draw a random threshold uniformly in (min, max).
        let threshold = random_threshold(rng, feat_min, feat_max);

        // Compute left/right statistics.
        let mut left_sum = F::zero();
        let mut left_sum_sq = F::zero();
        let mut left_n: usize = 0;

        for &i in indices {
            if data.x[[i, feat]] <= threshold {
                let val = data.y[i];
                left_sum = left_sum + val;
                left_sum_sq = left_sum_sq + val * val;
                left_n += 1;
            }
        }

        let right_n = n - left_n;
        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }

        let left_n_f = F::from(left_n).unwrap();
        let right_n_f = F::from(right_n).unwrap();

        let left_mean = left_sum / left_n_f;
        let left_mse = left_sum_sq / left_n_f - left_mean * left_mean;

        let right_sum = parent_sum - left_sum;
        let right_sum_sq = parent_sum_sq - left_sum_sq;
        let right_mean = right_sum / right_n_f;
        let right_mse = right_sum_sq / right_n_f - right_mean * right_mean;

        let weighted_child_mse = (left_n_f * left_mse + right_n_f * right_mse) / n_f;
        let mse_decrease = parent_mse - weighted_child_mse;

        if mse_decrease > best_score {
            best_score = mse_decrease;
            best_feature = feat;
            best_threshold = threshold;
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Crate-internal functions for ensemble usage
// ---------------------------------------------------------------------------

/// Build a classification extra-tree with a subset of features for ensemble use.
///
/// Used internally by `ExtraTreesClassifier` to build individual trees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_extra_classification_tree_for_ensemble<F: Float>(
    x: &Array2<F>,
    y: &[usize],
    n_classes: usize,
    indices: &[usize],
    feature_indices: Option<&[usize]>,
    params: &TreeParams,
    criterion: ClassificationCriterion,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> Vec<Node<F>> {
    let data = ClassificationData {
        x,
        y,
        n_classes,
        feature_indices,
        criterion,
    };
    let mut nodes = Vec::new();
    build_extra_classification_tree(
        &data,
        indices,
        &mut nodes,
        0,
        params,
        n_features,
        max_features_n,
        rng,
    );
    nodes
}

/// Build a regression extra-tree with a subset of features for ensemble use.
///
/// Used internally by `ExtraTreesRegressor` to build individual trees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_extra_regression_tree_for_ensemble<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    feature_indices: Option<&[usize]>,
    params: &TreeParams,
    n_features: usize,
    max_features_n: usize,
    rng: &mut StdRng,
) -> Vec<Node<F>> {
    let data = RegressionData {
        x,
        y,
        feature_indices,
    };
    let mut nodes = Vec::new();
    build_extra_regression_tree(
        &data,
        indices,
        &mut nodes,
        0,
        params,
        n_features,
        max_features_n,
        rng,
    );
    nodes
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
    fn test_extra_classifier_simple_binary() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ExtraTreeClassifier::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        // ExtraTrees should separate linearly separable data.
        for i in 0..3 {
            assert_eq!(preds[i], 0, "sample {i} should be class 0");
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1, "sample {i} should be class 1");
        }
    }

    #[test]
    fn test_extra_classifier_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = ExtraTreeClassifier::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds, array![0, 0, 0]);
    }

    #[test]
    fn test_extra_classifier_max_depth_1() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExtraTreeClassifier::<f64>::new()
            .with_max_depth(Some(1))
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // With depth 1 and a single feature, it should still separate the classes.
        // The tree has exactly one split node and two leaves.
        assert_eq!(fitted.nodes().len(), 3);
    }

    #[test]
    fn test_extra_classifier_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ExtraTreeClassifier::<f64>::new()
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (6, 2));
        // Each row sums to 1.
        for i in 0..6 {
            let row_sum = proba.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extra_classifier_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExtraTreeClassifier::<f64>::new()
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        // The sum of importances should be 1 (normalised).
        let total: f64 = importances.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
        // Feature 0 should have higher importance (feature 1 is constant).
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_extra_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0]; // wrong length

        let model = ExtraTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_extra_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = ExtraTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_extra_classifier_invalid_min_samples_split() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 1];

        let model = ExtraTreeClassifier::<f64>::new().with_min_samples_split(1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_extra_classifier_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 2, 2, 2]; // non-contiguous classes

        let model = ExtraTreeClassifier::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 2]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_extra_classifier_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = ExtraTreeClassifier::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn test_extra_classifier_f32() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0f32, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ExtraTreeClassifier::<f32>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_extra_classifier_deterministic() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model1 = ExtraTreeClassifier::<f64>::new().with_random_state(123);
        let model2 = ExtraTreeClassifier::<f64>::new().with_random_state(123);

        let fitted1 = model1.fit(&x, &y).unwrap();
        let fitted2 = model2.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    // -- Regressor tests --

    #[test]
    fn test_extra_regressor_simple() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model = ExtraTreeRegressor::<f64>::new()
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // A deep extra-tree should roughly memorize the training data.
        assert_eq!(preds.len(), 6);
        for i in 0..6 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_extra_regressor_constant_target() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![5.0, 5.0, 5.0, 5.0];

        let model = ExtraTreeRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in preds.iter() {
            assert_relative_eq!(p, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extra_regressor_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let model = ExtraTreeRegressor::<f64>::new()
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        let total: f64 = importances.sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
        // Feature 0 drives the target; feature 1 is constant.
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_extra_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0]; // wrong length

        let model = ExtraTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_extra_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = ExtraTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_extra_regressor_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = ExtraTreeRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn test_extra_regressor_max_depth() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let model = ExtraTreeRegressor::<f64>::new()
            .with_max_depth(Some(1))
            .with_max_features(MaxFeatures::All)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        // With depth 1, the tree should have exactly 3 nodes: one split + two leaves.
        assert_eq!(fitted.nodes().len(), 3);
    }

    #[test]
    fn test_extra_regressor_deterministic() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let model1 = ExtraTreeRegressor::<f64>::new().with_random_state(99);
        let model2 = ExtraTreeRegressor::<f64>::new().with_random_state(99);

        let fitted1 = model1.fit(&x, &y).unwrap();
        let fitted2 = model2.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for i in 0..6 {
            assert_relative_eq!(preds1[i], preds2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_extra_regressor_f32() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0f32, 2.0, 3.0, 4.0];

        let model = ExtraTreeRegressor::<f32>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // -- Builder tests --

    #[test]
    fn test_classifier_builder_methods() {
        let model = ExtraTreeClassifier::<f64>::new()
            .with_max_depth(Some(5))
            .with_min_samples_split(10)
            .with_min_samples_leaf(3)
            .with_max_features(MaxFeatures::Log2)
            .with_criterion(ClassificationCriterion::Entropy)
            .with_random_state(42);

        assert_eq!(model.max_depth, Some(5));
        assert_eq!(model.min_samples_split, 10);
        assert_eq!(model.min_samples_leaf, 3);
        assert_eq!(model.max_features, MaxFeatures::Log2);
        assert_eq!(model.criterion, ClassificationCriterion::Entropy);
        assert_eq!(model.random_state, Some(42));
    }

    #[test]
    fn test_regressor_builder_methods() {
        let model = ExtraTreeRegressor::<f64>::new()
            .with_max_depth(Some(10))
            .with_min_samples_split(5)
            .with_min_samples_leaf(2)
            .with_max_features(MaxFeatures::Fixed(3))
            .with_criterion(RegressionCriterion::Mse)
            .with_random_state(99);

        assert_eq!(model.max_depth, Some(10));
        assert_eq!(model.min_samples_split, 5);
        assert_eq!(model.min_samples_leaf, 2);
        assert_eq!(model.max_features, MaxFeatures::Fixed(3));
        assert_eq!(model.criterion, RegressionCriterion::Mse);
        assert_eq!(model.random_state, Some(99));
    }

    #[test]
    fn test_classifier_default() {
        let model = ExtraTreeClassifier::<f64>::default();
        assert_eq!(model.max_depth, None);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
        assert_eq!(model.max_features, MaxFeatures::Sqrt);
        assert_eq!(model.criterion, ClassificationCriterion::Gini);
        assert_eq!(model.random_state, None);
    }

    #[test]
    fn test_regressor_default() {
        let model = ExtraTreeRegressor::<f64>::default();
        assert_eq!(model.max_depth, None);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
        assert_eq!(model.max_features, MaxFeatures::All);
        assert_eq!(model.criterion, RegressionCriterion::Mse);
        assert_eq!(model.random_state, None);
    }
}
