//! CART decision tree classifiers and regressors.
//!
//! This module provides [`DecisionTreeClassifier`] and [`DecisionTreeRegressor`],
//! implementing the Classification and Regression Trees (CART) algorithm with
//! configurable splitting criteria, depth limits, and minimum sample constraints.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::DecisionTreeClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = DecisionTreeClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Splitting criterion enums
// ---------------------------------------------------------------------------

/// Splitting criterion for classification trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationCriterion {
    /// Gini impurity.
    Gini,
    /// Shannon entropy.
    Entropy,
}

/// Splitting criterion for regression trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionCriterion {
    /// Mean squared error.
    Mse,
}

// ---------------------------------------------------------------------------
// Node representation (flat vec for cache efficiency)
// ---------------------------------------------------------------------------

/// A single node in the decision tree, stored in a flat `Vec` for cache efficiency.
///
/// Internal nodes hold a split (feature index + threshold), while leaf nodes
/// store a prediction value and optional class distribution (for classifiers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node<F> {
    /// An internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Threshold value; samples with `x[feature] <= threshold` go left.
        threshold: F,
        /// Index of the left child node in the flat vec.
        left: usize,
        /// Index of the right child node in the flat vec.
        right: usize,
        /// Weighted impurity decrease from this split (for feature importance).
        impurity_decrease: F,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
    /// A leaf node that stores a prediction.
    Leaf {
        /// Predicted value: class label (as F) for classifiers, mean for regressors.
        value: F,
        /// Class distribution (proportion of each class). Only used by classifiers.
        class_distribution: Option<Vec<F>>,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
}

// ---------------------------------------------------------------------------
// Internal config structs (to reduce argument counts)
// ---------------------------------------------------------------------------

/// Configuration parameters for tree building, bundled to reduce argument counts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeParams {
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
}

/// Data references for classification tree building.
struct ClassificationData<'a, F> {
    x: &'a Array2<F>,
    y: &'a [usize],
    n_classes: usize,
    feature_indices: Option<&'a [usize]>,
    criterion: ClassificationCriterion,
}

/// Data references for regression tree building.
struct RegressionData<'a, F> {
    x: &'a Array2<F>,
    y: &'a Array1<F>,
    feature_indices: Option<&'a [usize]>,
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier
// ---------------------------------------------------------------------------

/// CART decision tree classifier.
///
/// Builds a binary tree by recursively finding the feature and threshold that
/// maximises the reduction in the chosen impurity criterion (Gini or Entropy).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Splitting criterion.
    pub criterion: ClassificationCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> DecisionTreeClassifier<F> {
    /// Create a new `DecisionTreeClassifier` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `criterion = Gini`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: ClassificationCriterion::Gini,
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

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for DecisionTreeClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedDecisionTreeClassifier
// ---------------------------------------------------------------------------

/// A fitted CART decision tree classifier.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
/// Implements [`Predict`] for generating class predictions and
/// [`HasFeatureImportances`] for inspecting per-feature importance scores.
#[derive(Debug, Clone)]
pub struct FittedDecisionTreeClassifier<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Sorted unique class labels observed during training.
    classes: Vec<usize>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedDecisionTreeClassifier<F> {
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
            let leaf = traverse_tree(&self.nodes, &row);
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

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for DecisionTreeClassifier<F> {
    type Fitted = FittedDecisionTreeClassifier<F>;
    type Error = FerroError;

    /// Fit the decision tree classifier on the training data.
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
    ) -> Result<FittedDecisionTreeClassifier<F>, FerroError> {
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
                context: "DecisionTreeClassifier requires at least one sample".into(),
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
        build_classification_tree(&data, &indices, &mut nodes, 0, &params);

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedDecisionTreeClassifier {
            nodes,
            classes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDecisionTreeClassifier<F> {
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
            let leaf = traverse_tree(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = float_to_usize(value);
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedDecisionTreeClassifier<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedDecisionTreeClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for DecisionTreeClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedClassifierPipelineAdapter(fitted)))
    }
}

/// Adapter to make `FittedDecisionTreeClassifier<F>` work as a pipeline estimator.
struct FittedClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedDecisionTreeClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeRegressor
// ---------------------------------------------------------------------------

/// CART decision tree regressor.
///
/// Builds a binary tree by recursively finding the feature and threshold that
/// minimises the mean squared error of the split.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor<F> {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Splitting criterion.
    pub criterion: RegressionCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> DecisionTreeRegressor<F> {
    /// Create a new `DecisionTreeRegressor` with default settings.
    ///
    /// Defaults: `max_depth = None`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `criterion = MSE`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: RegressionCriterion::Mse,
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

    /// Set the splitting criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: RegressionCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for DecisionTreeRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedDecisionTreeRegressor
// ---------------------------------------------------------------------------

/// A fitted CART decision tree regressor.
///
/// Stores the learned tree as a flat `Vec<Node<F>>` for cache-friendly traversal.
#[derive(Debug, Clone)]
pub struct FittedDecisionTreeRegressor<F> {
    /// Flat node storage; index 0 is the root.
    nodes: Vec<Node<F>>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Per-feature importance scores (normalised to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for DecisionTreeRegressor<F> {
    type Fitted = FittedDecisionTreeRegressor<F>;
    type Error = FerroError;

    /// Fit the decision tree regressor on the training data.
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
        y: &Array1<F>,
    ) -> Result<FittedDecisionTreeRegressor<F>, FerroError> {
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
                context: "DecisionTreeRegressor requires at least one sample".into(),
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
        build_regression_tree(&data, &indices, &mut nodes, 0, &params);

        let feature_importances = compute_feature_importances(&nodes, n_features, n_samples);

        Ok(FittedDecisionTreeRegressor {
            nodes,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedDecisionTreeRegressor<F> {
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDecisionTreeRegressor<F> {
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
            let leaf = traverse_tree(&self.nodes, &row);
            if let Node::Leaf { value, .. } = self.nodes[leaf] {
                predictions[i] = value;
            }
        }
        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedDecisionTreeRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for DecisionTreeRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedDecisionTreeRegressor<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Internal: tree building helpers
// ---------------------------------------------------------------------------

/// Traverse the tree from root to leaf for a single sample, returning the leaf node index.
fn traverse_tree<F: Float>(nodes: &[Node<F>], sample: &ndarray::ArrayView1<F>) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx] {
            Node::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                if sample[*feature] <= *threshold {
                    idx = *left;
                } else {
                    idx = *right;
                }
            }
            Node::Leaf { .. } => return idx,
        }
    }
}

/// Traverse a tree from root to leaf for a single sample (crate-public wrapper).
///
/// Returns the index of the leaf node in the flat node vector.
pub(crate) fn traverse<F: Float>(nodes: &[Node<F>], sample: &ndarray::ArrayView1<F>) -> usize {
    traverse_tree(nodes, sample)
}

/// Convert a `Float` value to `usize` (for class labels stored as floats).
fn float_to_usize<F: Float>(v: F) -> usize {
    v.to_f64().map(|f| f.round() as usize).unwrap_or(0)
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

/// Compute the mean of target values for the given indices.
fn mean_value<F: Float>(y: &Array1<F>, indices: &[usize]) -> F {
    if indices.is_empty() {
        return F::zero();
    }
    let sum: F = indices.iter().map(|&i| y[i]).fold(F::zero(), |a, b| a + b);
    sum / F::from(indices.len()).unwrap()
}

/// Compute the MSE for the given indices relative to a given mean.
fn mse_for_indices<F: Float>(y: &Array1<F>, indices: &[usize], mean: F) -> F {
    if indices.is_empty() {
        return F::zero();
    }
    let sum_sq: F = indices
        .iter()
        .map(|&i| {
            let diff = y[i] - mean;
            diff * diff
        })
        .fold(F::zero(), |a, b| a + b);
    sum_sq / F::from(indices.len()).unwrap()
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

/// Build a classification tree recursively.
///
/// Returns the index of the node that was created at the root of this subtree.
fn build_classification_tree<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
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

    let best = find_best_classification_split(data, indices, params.min_samples_leaf);

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_classification_tree(data, &left_indices, nodes, depth + 1, params);
        let right_idx = build_classification_tree(data, &right_indices, nodes, depth + 1, params);

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

/// Find the best split for a classification node.
///
/// Returns `(feature_index, threshold, weighted_impurity_decrease)` or `None`.
fn find_best_classification_split<F: Float>(
    data: &ClassificationData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();
    let n_features = data.x.ncols();

    let mut parent_counts = vec![0usize; data.n_classes];
    for &i in indices {
        parent_counts[data.y[i]] += 1;
    }
    let parent_impurity = compute_impurity::<F>(&parent_counts, n, data.criterion);

    let mut best_score = F::neg_infinity();
    let mut best_feature = 0;
    let mut best_threshold = F::zero();

    // Iterate over either specified feature subset or all features.
    let feature_iter: Box<dyn Iterator<Item = usize>> =
        if let Some(feat_indices) = data.feature_indices {
            Box::new(feat_indices.iter().copied())
        } else {
            Box::new(0..n_features)
        };

    for feat in feature_iter {
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| data.x[[a, feat]].partial_cmp(&data.x[[b, feat]]).unwrap());

        let mut left_counts = vec![0usize; data.n_classes];
        let mut right_counts = parent_counts.clone();
        let mut left_n = 0usize;

        for split_pos in 0..n - 1 {
            let idx = sorted_indices[split_pos];
            let cls = data.y[idx];
            left_counts[cls] += 1;
            right_counts[cls] -= 1;
            left_n += 1;
            let right_n = n - left_n;

            let next_idx = sorted_indices[split_pos + 1];
            if data.x[[idx, feat]] == data.x[[next_idx, feat]] {
                continue;
            }

            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let left_impurity = compute_impurity::<F>(&left_counts, left_n, data.criterion);
            let right_impurity = compute_impurity::<F>(&right_counts, right_n, data.criterion);
            let left_weight = F::from(left_n).unwrap() / n_f;
            let right_weight = F::from(right_n).unwrap() / n_f;
            let weighted_child_impurity =
                left_weight * left_impurity + right_weight * right_impurity;
            let impurity_decrease = parent_impurity - weighted_child_impurity;

            if impurity_decrease > best_score {
                best_score = impurity_decrease;
                best_feature = feat;
                best_threshold =
                    (data.x[[idx, feat]] + data.x[[next_idx, feat]]) / F::from(2.0).unwrap();
            }
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

/// Build a regression tree recursively.
fn build_regression_tree<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    params: &TreeParams,
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

    let parent_mse = mse_for_indices(data.y, indices, mean);
    if parent_mse <= F::epsilon() {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: mean,
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    let best = find_best_regression_split(data, indices, params.min_samples_leaf);

    if let Some((best_feature, best_threshold, best_impurity_decrease)) = best {
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| data.x[[i, best_feature]] <= best_threshold);

        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_idx = build_regression_tree(data, &left_indices, nodes, depth + 1, params);
        let right_idx = build_regression_tree(data, &right_indices, nodes, depth + 1, params);

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

/// Find the best split for a regression node using MSE reduction.
///
/// Returns `(feature_index, threshold, weighted_mse_decrease)` or `None`.
fn find_best_regression_split<F: Float>(
    data: &RegressionData<'_, F>,
    indices: &[usize],
    min_samples_leaf: usize,
) -> Option<(usize, F, F)> {
    let n = indices.len();
    let n_f = F::from(n).unwrap();
    let n_features = data.x.ncols();

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

    let feature_iter: Box<dyn Iterator<Item = usize>> =
        if let Some(feat_indices) = data.feature_indices {
            Box::new(feat_indices.iter().copied())
        } else {
            Box::new(0..n_features)
        };

    for feat in feature_iter {
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| data.x[[a, feat]].partial_cmp(&data.x[[b, feat]]).unwrap());

        let mut left_sum = F::zero();
        let mut left_sum_sq = F::zero();
        let mut left_n: usize = 0;

        for split_pos in 0..n - 1 {
            let idx = sorted_indices[split_pos];
            let val = data.y[idx];
            left_sum = left_sum + val;
            left_sum_sq = left_sum_sq + val * val;
            left_n += 1;
            let right_n = n - left_n;

            let next_idx = sorted_indices[split_pos + 1];
            if data.x[[idx, feat]] == data.x[[next_idx, feat]] {
                continue;
            }

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
                best_threshold =
                    (data.x[[idx, feat]] + data.x[[next_idx, feat]]) / F::from(2.0).unwrap();
            }
        }
    }

    if best_score > F::zero() {
        Some((best_feature, best_threshold, best_score * n_f))
    } else {
        None
    }
}

/// Compute normalised feature importances from impurity decreases in the tree.
pub(crate) fn compute_feature_importances<F: Float>(
    nodes: &[Node<F>],
    n_features: usize,
    _total_samples: usize,
) -> Array1<F> {
    let mut importances = Array1::zeros(n_features);
    for node in nodes {
        if let Node::Split {
            feature,
            impurity_decrease,
            ..
        } = node
        {
            importances[*feature] = importances[*feature] + *impurity_decrease;
        }
    }
    let total: F = importances.iter().copied().fold(F::zero(), |a, b| a + b);
    if total > F::zero() {
        importances.mapv_inplace(|v| v / total);
    }
    importances
}

// ---------------------------------------------------------------------------
// Public builders for forest usage
// ---------------------------------------------------------------------------

/// Build a classification tree with a subset of features considered per split.
///
/// Used internally by `RandomForestClassifier` to build individual trees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_classification_tree_with_feature_subset<F: Float>(
    x: &Array2<F>,
    y: &[usize],
    n_classes: usize,
    indices: &[usize],
    feature_indices: &[usize],
    params: &TreeParams,
    criterion: ClassificationCriterion,
) -> Vec<Node<F>> {
    let data = ClassificationData {
        x,
        y,
        n_classes,
        feature_indices: Some(feature_indices),
        criterion,
    };
    let mut nodes = Vec::new();
    build_classification_tree(&data, indices, &mut nodes, 0, params);
    nodes
}

/// Build a regression tree with a subset of features considered per split.
pub(crate) fn build_regression_tree_with_feature_subset<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    indices: &[usize],
    feature_indices: &[usize],
    params: &TreeParams,
) -> Vec<Node<F>> {
    let data = RegressionData {
        x,
        y,
        feature_indices: Some(feature_indices),
    };
    let mut nodes = Vec::new();
    build_regression_tree(&data, indices, &mut nodes, 0, params);
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
    fn test_classifier_simple_binary() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for i in 0..3 {
            assert_eq!(preds[i], 0);
        }
        for i in 3..6 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_classifier_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds, array![0, 0, 0]);
    }

    #[test]
    fn test_classifier_max_depth_1() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_max_depth(Some(1));
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], 0);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1);
        }
    }

    #[test]
    fn test_classifier_min_samples_split() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_split(7);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let majority = preds[0];
        for &p in preds.iter() {
            assert_eq!(p, majority);
        }
    }

    #[test]
    fn test_classifier_min_samples_leaf() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_leaf(4);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let majority = preds[0];
        for &p in preds.iter() {
            assert_eq!(p, majority);
        }
    }

    #[test]
    fn test_classifier_gini_vs_entropy() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let gini_model =
            DecisionTreeClassifier::<f64>::new().with_criterion(ClassificationCriterion::Gini);
        let entropy_model =
            DecisionTreeClassifier::<f64>::new().with_criterion(ClassificationCriterion::Entropy);

        let fitted_gini = gini_model.fit(&x, &y).unwrap();
        let fitted_entropy = entropy_model.fit(&x, &y).unwrap();

        let preds_gini = fitted_gini.predict(&x).unwrap();
        let preds_entropy = fitted_entropy.predict(&x).unwrap();

        assert_eq!(preds_gini, y);
        assert_eq!(preds_entropy, y);
    }

    #[test]
    fn test_classifier_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (6, 2));
        for i in 0..6 {
            let row_sum: f64 = proba.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = DecisionTreeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        assert!(importances[0] > 0.0);
        let sum: f64 = importances.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_classifier_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_classifier_invalid_min_samples_split() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_split(1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_invalid_min_samples_leaf() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = DecisionTreeClassifier::<f64>::new().with_min_samples_leaf(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds, y);
    }

    #[test]
    fn test_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    // -- Regressor tests --

    #[test]
    fn test_regressor_simple() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_max_depth() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new().with_max_depth(Some(1));
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_relative_eq!(preds[i], 1.0, epsilon = 1e-10);
        }
        for i in 4..8 {
            assert_relative_eq!(preds[i], 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_constant_target() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 3.0, 3.0, 3.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &p in preds.iter() {
            assert_relative_eq!(p, 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = DecisionTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = DecisionTreeRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_feature_importances() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 2);
        assert!(importances[0] > 0.0);
        let sum: f64 = importances.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_min_samples_split() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new().with_min_samples_split(5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let mean = 2.5;
        for &p in preds.iter() {
            assert_relative_eq!(p, mean, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = DecisionTreeRegressor::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_regressor_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let model = DecisionTreeRegressor::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_classifier_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = DecisionTreeClassifier::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    // -- Internal helper tests --

    #[test]
    fn test_gini_impurity_pure() {
        let counts = vec![5, 0];
        let imp: f64 = gini_impurity(&counts, 5);
        assert_relative_eq!(imp, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gini_impurity_balanced() {
        let counts = vec![5, 5];
        let imp: f64 = gini_impurity(&counts, 10);
        assert_relative_eq!(imp, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_pure() {
        let counts = vec![5, 0];
        let ent: f64 = entropy_impurity(&counts, 5);
        assert_relative_eq!(ent, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_balanced() {
        let counts = vec![5, 5];
        let ent: f64 = entropy_impurity(&counts, 10);
        assert_relative_eq!(ent, 2.0f64.ln(), epsilon = 1e-10);
    }
}
