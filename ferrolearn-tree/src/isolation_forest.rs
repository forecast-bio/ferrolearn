//! Isolation forest anomaly detection.
//!
//! This module provides [`IsolationForest`], an unsupervised anomaly detection
//! algorithm that isolates observations by randomly selecting features and split
//! points. Anomalies are the points that require fewer random splits to isolate,
//! resulting in shorter average path lengths across the ensemble of isolation trees.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::IsolationForest;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  100.0, 100.0,
//! ]).unwrap();
//!
//! let model = IsolationForest::<f64>::new()
//!     .with_n_estimators(100)
//!     .with_contamination(0.1)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! // Normal points get 1, anomalies get -1
//! assert!(preds.iter().all(|&v| v == 1 || v == -1));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// IsolationTree node representation
// ---------------------------------------------------------------------------

/// A single node in an isolation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum IsoNode<F> {
    /// An internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Split threshold; samples with `x[feature] <= threshold` go left.
        threshold: F,
        /// Index of the left child in the flat node vector.
        left: usize,
        /// Index of the right child in the flat node vector.
        right: usize,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
    /// An external (leaf) node.
    Leaf {
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
}

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------

/// Isolation forest anomaly detector.
///
/// Builds an ensemble of isolation trees, each trained on a random subsample.
/// Anomaly scores are derived from the average path length: points that are
/// isolated in fewer splits are more anomalous.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest<F> {
    /// Number of isolation trees in the forest.
    pub n_estimators: usize,
    /// Number of samples to draw for each tree.
    pub max_samples: usize,
    /// Proportion of anomalies in the dataset, used to set the decision threshold.
    pub contamination: f64,
    /// Random seed for reproducibility. `None` means non-deterministic.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> IsolationForest<F> {
    /// Create a new `IsolationForest` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `max_samples = 256`,
    /// `contamination = 0.1`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of isolation trees.
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the number of samples to draw for each tree.
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set the contamination fraction (proportion of anomalies).
    #[must_use]
    pub fn with_contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for IsolationForest<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedIsolationForest
// ---------------------------------------------------------------------------

/// A fitted isolation forest anomaly detector.
///
/// Stores the ensemble of isolation trees and the anomaly score threshold
/// derived from the contamination parameter.
#[derive(Debug, Clone)]
pub struct FittedIsolationForest<F> {
    /// Individual isolation trees, each stored as a flat node vector.
    trees: Vec<Vec<IsoNode<F>>>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Decision threshold: anomaly_score > threshold => anomaly (-1).
    threshold: f64,
    /// Effective number of samples used per tree.
    max_samples: usize,
}

impl<F: Float + Send + Sync + 'static> FittedIsolationForest<F> {
    /// Returns the number of isolation trees.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the anomaly score threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Compute anomaly scores for each sample.
    ///
    /// Score = 2^(-mean_path_length / c(n)), where c(n) is the average path
    /// length of an unsuccessful search in a binary search tree. Scores
    /// close to 1 indicate anomalies; scores close to 0.5 indicate normal points.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<f64>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let c_n = average_path_length(self.max_samples);
        let n_trees = self.trees.len() as f64;
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let mut total_path = 0.0;
            for tree_nodes in &self.trees {
                total_path += path_length(tree_nodes, &row);
            }
            let mean_path = total_path / n_trees;
            scores[i] = f64::powf(2.0, -mean_path / c_n);
        }

        Ok(scores)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for IsolationForest<F> {
    type Fitted = FittedIsolationForest<F>;
    type Error = FerroError;

    /// Fit the isolation forest on the training data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if hyperparameters are invalid.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedIsolationForest<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "IsolationForest requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "max_samples".into(),
                reason: "must be at least 1".into(),
            });
        }
        if !(0.0..=0.5).contains(&self.contamination) {
            return Err(FerroError::InvalidParameter {
                name: "contamination".into(),
                reason: "must be in [0.0, 0.5]".into(),
            });
        }

        let effective_max_samples = self.max_samples.min(n_samples);
        let max_depth = (effective_max_samples as f64).log2().ceil() as usize;

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_os_rng()
        };

        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            // Sample indices with replacement.
            let sample_indices: Vec<usize> = (0..effective_max_samples)
                .map(|_| {
                    use rand::RngCore;
                    (rng.next_u64() as usize) % n_samples
                })
                .collect();

            let mut nodes = Vec::new();
            let indices: Vec<usize> = (0..sample_indices.len()).collect();
            // Build a view of the subsampled data.
            build_isolation_tree(
                x,
                &sample_indices,
                &indices,
                &mut nodes,
                0,
                max_depth,
                n_features,
                &mut rng,
            );
            trees.push(nodes);
        }

        // Compute anomaly scores on the training data to find the threshold.
        let fitted_no_threshold = FittedIsolationForest {
            trees,
            n_features,
            threshold: 0.0,
            max_samples: effective_max_samples,
        };

        let train_scores = fitted_no_threshold.score_samples(x)?;

        // Sort scores descending to find the contamination quantile.
        let mut sorted_scores: Vec<f64> = train_scores.iter().copied().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let contamination_idx =
            ((self.contamination * n_samples as f64).ceil() as usize).max(1).min(n_samples);
        let threshold = if contamination_idx < sorted_scores.len() {
            sorted_scores[contamination_idx - 1]
        } else {
            sorted_scores[sorted_scores.len() - 1]
        };

        Ok(FittedIsolationForest {
            trees: fitted_no_threshold.trees,
            n_features,
            threshold,
            max_samples: effective_max_samples,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedIsolationForest<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict anomaly labels for the given feature matrix.
    ///
    /// Returns 1 for normal points and -1 for anomalies.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let scores = self.score_samples(x)?;
        let predictions = scores.mapv(|s| if s >= self.threshold { -1 } else { 1 });
        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Internal: isolation tree building
// ---------------------------------------------------------------------------

/// Average path length of an unsuccessful search in a BST with `n` elements.
///
/// c(n) = 2 * (ln(n-1) + EULER_MASCHERONI) - 2*(n-1)/n  for n >= 2
/// c(1) = 0
fn average_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    // Euler-Mascheroni constant
    2.0 * ((n_f - 1.0).ln() + 0.5772156649015329) - 2.0 * (n_f - 1.0) / n_f
}

/// Compute the path length for a single sample through an isolation tree.
fn path_length<F: Float>(nodes: &[IsoNode<F>], sample: &ndarray::ArrayView1<F>) -> f64 {
    let mut idx = 0;
    let mut depth: f64 = 0.0;
    loop {
        match &nodes[idx] {
            IsoNode::Split {
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
                depth += 1.0;
            }
            IsoNode::Leaf { n_samples } => {
                // Add the expected path length for the remaining samples.
                return depth + average_path_length(*n_samples);
            }
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

/// Build an isolation tree recursively.
///
/// `sample_indices` maps local indices to rows in `x`.
/// `indices` are the local indices of points currently in this node.
#[allow(clippy::too_many_arguments)]
fn build_isolation_tree<F: Float>(
    x: &Array2<F>,
    sample_indices: &[usize],
    indices: &[usize],
    nodes: &mut Vec<IsoNode<F>>,
    depth: usize,
    max_depth: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> usize {
    let n = indices.len();

    // Stop if: only one sample, or max depth reached.
    if n <= 1 || depth >= max_depth {
        let idx = nodes.len();
        nodes.push(IsoNode::Leaf { n_samples: n });
        return idx;
    }

    // Try random features until we find one that can split, or exhaust attempts.
    let max_attempts = n_features * 2;
    for _ in 0..max_attempts {
        use rand::RngCore;
        let feature = (rng.next_u64() as usize) % n_features;

        // Find min and max of this feature for the current indices.
        let mut min_val = x[[sample_indices[indices[0]], feature]];
        let mut max_val = min_val;
        for &i in &indices[1..] {
            let v = x[[sample_indices[i], feature]];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        // If all values are the same for this feature, try another feature.
        if min_val >= max_val {
            continue;
        }

        let threshold = random_threshold(rng, min_val, max_val);

        // Partition indices.
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            if x[[sample_indices[i], feature]] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // If the split is degenerate (all on one side), try again.
        if left_indices.is_empty() || right_indices.is_empty() {
            continue;
        }

        // Reserve a slot for this node.
        let node_idx = nodes.len();
        nodes.push(IsoNode::Leaf { n_samples: 0 }); // placeholder

        let left_child = build_isolation_tree(
            x,
            sample_indices,
            &left_indices,
            nodes,
            depth + 1,
            max_depth,
            n_features,
            rng,
        );
        let right_child = build_isolation_tree(
            x,
            sample_indices,
            &right_indices,
            nodes,
            depth + 1,
            max_depth,
            n_features,
            rng,
        );

        nodes[node_idx] = IsoNode::Split {
            feature,
            threshold,
            left: left_child,
            right: right_child,
            n_samples: n,
        };

        return node_idx;
    }

    // Could not find a splittable feature — make this a leaf.
    let idx = nodes.len();
    nodes.push(IsoNode::Leaf { n_samples: n });
    idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_normal_data() -> Array2<f64> {
        // 10 normal points clustered around (5, 5)
        Array2::from_shape_vec(
            (10, 2),
            vec![
                4.5, 4.8, 5.1, 5.2, 4.9, 5.0, 5.3, 4.7, 4.8, 5.1, 5.0, 5.3, 5.2, 4.9, 4.7, 5.0,
                5.1, 4.8, 4.9, 5.2,
            ],
        )
        .unwrap()
    }

    fn make_data_with_anomaly() -> Array2<f64> {
        // 9 normal points + 1 clear anomaly at (100, 100)
        Array2::from_shape_vec(
            (10, 2),
            vec![
                4.5, 4.8, 5.1, 5.2, 4.9, 5.0, 5.3, 4.7, 4.8, 5.1, 5.0, 5.3, 5.2, 4.9, 4.7, 5.0,
                5.1, 4.8, 100.0, 100.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_isolation_forest_default() {
        let model = IsolationForest::<f64>::new();
        assert_eq!(model.n_estimators, 100);
        assert_eq!(model.max_samples, 256);
        assert!((model.contamination - 0.1).abs() < 1e-10);
        assert!(model.random_state.is_none());
    }

    #[test]
    fn test_isolation_forest_builder() {
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_max_samples(128)
            .with_contamination(0.05)
            .with_random_state(123);
        assert_eq!(model.n_estimators, 50);
        assert_eq!(model.max_samples, 128);
        assert!((model.contamination - 0.05).abs() < 1e-10);
        assert_eq!(model.random_state, Some(123));
    }

    #[test]
    fn test_fit_predict_basic() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        // All predictions should be either 1 or -1
        assert!(preds.iter().all(|&v| v == 1 || v == -1));
    }

    #[test]
    fn test_anomaly_detected() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(200)
            .with_contamination(0.15)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // The last point (100, 100) should be flagged as anomaly (-1)
        assert_eq!(preds[9], -1, "outlier should be detected as anomaly");
    }

    #[test]
    fn test_anomaly_scores() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(200)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let scores = fitted.score_samples(&x).unwrap();

        assert_eq!(scores.len(), 10);
        // The anomaly (last point) should have a higher score than normal points
        let anomaly_score = scores[9];
        let max_normal_score = scores.iter().take(9).copied().fold(0.0_f64, f64::max);
        assert!(
            anomaly_score > max_normal_score,
            "anomaly score ({anomaly_score}) should be greater than max normal score ({max_normal_score})"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = IsolationForest::<f64>::new();
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_estimators_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_n_estimators(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_max_samples_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_max_samples(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_contamination_error() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new().with_contamination(0.6);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x_train = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_test = Array2::<f64>::zeros((3, 5)); // wrong number of features
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_shape_mismatch() {
        let x_train = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_test = Array2::<f64>::zeros((3, 5));
        let result = fitted.score_samples(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_average_path_length_values() {
        assert!((average_path_length(1) - 0.0).abs() < 1e-10);
        // c(2) = 2*(ln(1) + 0.5772...) - 2*1/2 = 2*0.5772... - 1 = 0.1544...
        let c2 = average_path_length(2);
        assert!(c2 > 0.0 && c2 < 1.0, "c(2) = {c2}");
        // c(256) should be a reasonable number
        let c256 = average_path_length(256);
        assert!(c256 > 5.0 && c256 < 15.0, "c(256) = {c256}");
    }

    #[test]
    fn test_reproducibility() {
        let x = make_data_with_anomaly();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(999);

        let fitted1 = model.fit(&x, &()).unwrap();
        let preds1 = fitted1.predict(&x).unwrap();

        let fitted2 = model.fit(&x, &()).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_max_samples_larger_than_data() {
        // max_samples > n_samples should still work (clamped to n_samples).
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_max_samples(10000)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 100.0, 100.0],
        )
        .unwrap();
        let model = IsolationForest::<f32>::new()
            .with_n_estimators(50)
            .with_contamination(0.2)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
        assert!(preds.iter().all(|&v| v == 1 || v == -1));
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_contamination(0.0)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn test_fitted_accessors() {
        let x = make_normal_data();
        let model = IsolationForest::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_estimators(), 10);
        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.threshold() >= 0.0);
    }
}
