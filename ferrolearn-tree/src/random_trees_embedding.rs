//! Random trees embedding for unsupervised feature transformation.
//!
//! This module provides [`RandomTreesEmbedding`], which transforms input
//! features into a high-dimensional sparse binary representation by encoding
//! each sample as the concatenation of one-hot encoded leaf indices across
//! an ensemble of randomly built trees.
//!
//! Trees are built with purely random splits (random feature, random threshold
//! between min and max), ignoring any target variable. This makes the
//! embedding entirely unsupervised.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::RandomTreesEmbedding;
//! use ferrolearn_core::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
//! ]).unwrap();
//!
//! let model = RandomTreesEmbedding::<f64>::new()
//!     .with_n_estimators(5)
//!     .with_max_depth(Some(3))
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! let embedded = fitted.transform(&x).unwrap();
//! // Output has n_samples rows and (total_leaves_across_trees) columns.
//! assert_eq!(embedded.nrows(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array2, ArrayView1};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::decision_tree::Node;

// ---------------------------------------------------------------------------
// RandomTreesEmbedding
// ---------------------------------------------------------------------------

/// Random trees embedding for unsupervised feature transformation.
///
/// Builds an ensemble of randomly split trees (ignoring targets) and
/// represents each sample as a one-hot encoding of its leaf index in
/// each tree, concatenated across all trees.
///
/// This is useful for creating a nonlinear feature representation that
/// can be fed into linear models.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomTreesEmbedding<F> {
    /// Number of random trees to build.
    pub n_estimators: usize,
    /// Maximum depth of each tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split a node.
    pub min_samples_split: usize,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> RandomTreesEmbedding<F> {
    /// Create a new `RandomTreesEmbedding` with default settings.
    ///
    /// Defaults: `n_estimators = 10`, `max_depth = Some(5)`,
    /// `min_samples_split = 2`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 10,
            max_depth: Some(5),
            min_samples_split: 2,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of random trees.
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

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for RandomTreesEmbedding<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedRandomTreesEmbedding
// ---------------------------------------------------------------------------

/// A fitted random trees embedding.
///
/// Stores the ensemble of randomly built trees. Each tree's leaves are
/// enumerated, and [`Transform`] produces a one-hot encoded matrix
/// of shape `(n_samples, total_leaves)`.
#[derive(Debug, Clone)]
pub struct FittedRandomTreesEmbedding<F> {
    /// Individual trees, each stored as a flat node vector.
    trees: Vec<Vec<Node<F>>>,
    /// Per-tree leaf count (number of leaves in each tree).
    leaf_counts: Vec<usize>,
    /// Per-tree mapping from leaf node index to enumerated leaf position.
    /// `leaf_maps[t][node_idx]` gives the leaf's position index within tree `t`.
    leaf_maps: Vec<Vec<Option<usize>>>,
    /// Total number of leaves across all trees (output dimensionality).
    total_leaves: usize,
    /// Number of features the model was trained on.
    n_features: usize,
}

impl<F: Float + Send + Sync + 'static> FittedRandomTreesEmbedding<F> {
    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the total number of output features (total leaves across all trees).
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.total_leaves
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for RandomTreesEmbedding<F> {
    type Fitted = FittedRandomTreesEmbedding<F>;
    type Error = FerroError;

    /// Fit the random trees embedding on the training data.
    ///
    /// Builds an ensemble of trees with purely random splits (random feature,
    /// random threshold between feature min and max in the current node),
    /// ignoring any target variable.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedRandomTreesEmbedding<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RandomTreesEmbedding requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.min_samples_split < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples_split".into(),
                reason: "must be at least 2".into(),
            });
        }

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let indices: Vec<usize> = (0..n_samples).collect();

        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut leaf_counts = Vec::with_capacity(self.n_estimators);
        let mut leaf_maps = Vec::with_capacity(self.n_estimators);
        let mut total_leaves = 0;

        for _ in 0..self.n_estimators {
            let mut nodes = Vec::new();
            build_random_tree(
                x,
                &indices,
                &mut nodes,
                0,
                self.max_depth,
                self.min_samples_split,
                n_features,
                &mut rng,
            );

            // Enumerate leaf nodes.
            let mut leaf_map = vec![None; nodes.len()];
            let mut count = 0;
            for (idx, node) in nodes.iter().enumerate() {
                if matches!(node, Node::Leaf { .. }) {
                    leaf_map[idx] = Some(count);
                    count += 1;
                }
            }

            trees.push(nodes);
            leaf_counts.push(count);
            leaf_maps.push(leaf_map);
            total_leaves += count;
        }

        Ok(FittedRandomTreesEmbedding {
            trees,
            leaf_counts,
            leaf_maps,
            total_leaves,
            n_features,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedRandomTreesEmbedding<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform the input data into a one-hot encoded leaf-index representation.
    ///
    /// For each sample, traverse each tree to a leaf node, then one-hot encode
    /// the leaf index within that tree. The encodings for all trees are
    /// concatenated horizontally to produce the output.
    ///
    /// Output shape: `(n_samples, total_leaves_across_all_trees)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut output = Array2::zeros((n_samples, self.total_leaves));

        let mut col_offset = 0;
        for (tree_idx, tree_nodes) in self.trees.iter().enumerate() {
            let leaf_map = &self.leaf_maps[tree_idx];
            let n_leaves = self.leaf_counts[tree_idx];

            for i in 0..n_samples {
                let row = x.row(i);
                let leaf_node_idx = traverse_tree(tree_nodes, &row);
                if let Some(leaf_pos) = leaf_map[leaf_node_idx] {
                    output[[i, col_offset + leaf_pos]] = F::one();
                }
            }

            col_offset += n_leaves;
        }

        Ok(output)
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for RandomTreesEmbedding<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &ndarray::Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedRandomTreesEmbedding<F>
{
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Internal: random tree building (unsupervised)
// ---------------------------------------------------------------------------

/// Traverse a tree from root to leaf for a single sample, returning the leaf node index.
fn traverse_tree<F: Float>(nodes: &[Node<F>], sample: &ArrayView1<F>) -> usize {
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

/// Generate a uniform random float in `[min_val, max_val]`.
fn random_threshold<F: Float>(rng: &mut StdRng, min_val: F, max_val: F) -> F {
    use rand::RngCore;
    let u = (rng.next_u64() as f64) / (u64::MAX as f64);
    let range = max_val - min_val;
    min_val + F::from(u).unwrap() * range
}

/// Build a random tree recursively with purely random splits (unsupervised).
///
/// At each node, a random feature is chosen and a random threshold is drawn
/// uniformly between the feature's min and max in the current node.
#[allow(clippy::too_many_arguments)]
fn build_random_tree<F: Float>(
    x: &Array2<F>,
    indices: &[usize],
    nodes: &mut Vec<Node<F>>,
    depth: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> usize {
    let n = indices.len();

    // Stop if: too few samples, or max depth reached.
    let should_stop = n < min_samples_split || max_depth.is_some_and(|d| depth >= d);

    if should_stop {
        let idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: n,
        });
        return idx;
    }

    // Try random features until we find one that can split.
    let max_attempts = n_features * 2;
    for _ in 0..max_attempts {
        use rand::RngCore;
        let feature = (rng.next_u64() as usize) % n_features;

        // Find min and max of this feature for the current indices.
        let mut min_val = x[[indices[0], feature]];
        let mut max_val = min_val;
        for &i in &indices[1..] {
            let v = x[[i, feature]];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        // If all values are the same, try another feature.
        if min_val >= max_val {
            continue;
        }

        let threshold = random_threshold(rng, min_val, max_val);

        // Partition indices.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            if x[[i, feature]] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // If the split is degenerate, try again.
        if left_indices.is_empty() || right_indices.is_empty() {
            continue;
        }

        // Reserve a slot for this node.
        let node_idx = nodes.len();
        nodes.push(Node::Leaf {
            value: F::zero(),
            class_distribution: None,
            n_samples: 0,
        }); // placeholder

        let left_child = build_random_tree(
            x,
            &left_indices,
            nodes,
            depth + 1,
            max_depth,
            min_samples_split,
            n_features,
            rng,
        );
        let right_child = build_random_tree(
            x,
            &right_indices,
            nodes,
            depth + 1,
            max_depth,
            min_samples_split,
            n_features,
            rng,
        );

        nodes[node_idx] = Node::Split {
            feature,
            threshold,
            left: left_child,
            right: right_child,
            impurity_decrease: F::zero(),
            n_samples: n,
        };

        return node_idx;
    }

    // Could not find a splittable feature — make this a leaf.
    let idx = nodes.len();
    nodes.push(Node::Leaf {
        value: F::zero(),
        class_distribution: None,
        n_samples: n,
    });
    idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0,
                7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_default() {
        let model = RandomTreesEmbedding::<f64>::new();
        assert_eq!(model.n_estimators, 10);
        assert_eq!(model.max_depth, Some(5));
        assert_eq!(model.min_samples_split, 2);
        assert!(model.random_state.is_none());
    }

    #[test]
    fn test_builder() {
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(20)
            .with_max_depth(Some(3))
            .with_min_samples_split(5)
            .with_random_state(42);
        assert_eq!(model.n_estimators, 20);
        assert_eq!(model.max_depth, Some(3));
        assert_eq!(model.min_samples_split, 5);
        assert_eq!(model.random_state, Some(42));
    }

    #[test]
    fn test_fit_transform_basic() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();

        assert_eq!(embedded.nrows(), 8);
        // Each row should have exactly n_estimators ones (one per tree).
        for i in 0..8 {
            let row_sum: f64 = embedded.row(i).iter().copied().sum();
            assert!(
                (row_sum - 5.0).abs() < 1e-10,
                "row {i} should have exactly 5 ones, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_output_is_binary() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(3)
            .with_max_depth(Some(2))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();

        // All values should be 0.0 or 1.0
        for &val in embedded.iter() {
            assert!(
                (val - 0.0).abs() < 1e-10 || (val - 1.0).abs() < 1e-10,
                "values should be 0 or 1, got {val}"
            );
        }
    }

    #[test]
    fn test_total_leaves_matches_output_cols() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();

        assert_eq!(embedded.ncols(), fitted.n_output_features());
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 3));
        let model = RandomTreesEmbedding::<f64>::new();
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new().with_n_estimators(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_min_samples_split_error() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new().with_min_samples_split(1);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let x_train = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_test = Array2::<f64>::zeros((5, 10)); // wrong number of features
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_reproducibility() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);

        let fitted1 = model.fit(&x, &()).unwrap();
        let embedded1 = fitted1.transform(&x).unwrap();

        let fitted2 = model.fit(&x, &()).unwrap();
        let embedded2 = fitted2.transform(&x).unwrap();

        assert_eq!(embedded1, embedded2);
    }

    #[test]
    fn test_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let model = RandomTreesEmbedding::<f32>::new()
            .with_n_estimators(3)
            .with_max_depth(Some(2))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();
        assert_eq!(embedded.nrows(), 6);
    }

    #[test]
    fn test_fitted_accessors() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_estimators(), 5);
        assert_eq!(fitted.n_features(), 3);
        assert!(fitted.n_output_features() > 0);
    }

    #[test]
    fn test_deeper_trees_more_leaves() {
        let x = make_data();

        let shallow = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(1)
            .with_max_depth(Some(1))
            .with_random_state(42);
        let fitted_shallow = shallow.fit(&x, &()).unwrap();

        let deep = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(1)
            .with_max_depth(Some(5))
            .with_random_state(42);
        let fitted_deep = deep.fit(&x, &()).unwrap();

        assert!(
            fitted_deep.n_output_features() >= fitted_shallow.n_output_features(),
            "deeper trees should have at least as many leaves"
        );
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::<f64>::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(3)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();
        assert_eq!(embedded.nrows(), 1);
        // Single sample can't be split, so each tree has exactly 1 leaf.
        assert_eq!(embedded.ncols(), 3);
    }

    #[test]
    fn test_unlimited_depth() {
        let x = make_data();
        let model = RandomTreesEmbedding::<f64>::new()
            .with_n_estimators(3)
            .with_max_depth(None)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let embedded = fitted.transform(&x).unwrap();
        assert_eq!(embedded.nrows(), 8);
        assert!(embedded.ncols() > 0);
    }
}
