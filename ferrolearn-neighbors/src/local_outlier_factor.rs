//! Local Outlier Factor (LOF) for outlier detection.
//!
//! This module provides [`LocalOutlierFactor`], which measures the local
//! density deviation of a sample with respect to its neighbors. Samples
//! with substantially lower density than their neighbors are considered
//! outliers.
//!
//! # Algorithm
//!
//! 1. For each point, find the k nearest neighbors and their distances.
//! 2. `k_distance(p)` = distance to the k-th nearest neighbor.
//! 3. `reach_dist(p, o)` = max(`k_distance(o)`, `dist(p, o)`).
//! 4. Local Reachability Density: `lrd(p) = k / sum(reach_dist(p, o_i))`.
//! 5. `LOF(p) = mean(lrd(o_i)) / lrd(p)` for all neighbors `o_i`.
//! 6. `LOF >> 1` means the point is an outlier.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::local_outlier_factor::LocalOutlierFactor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
//!     -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05, -0.05,
//! ]).unwrap();
//!
//! let model = LocalOutlierFactor::<f64>::new();
//! let fitted = model.fit(&x, &()).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 8);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::knn::Algorithm;

// ---------------------------------------------------------------------------
// LocalOutlierFactor
// ---------------------------------------------------------------------------

/// Local Outlier Factor for outlier detection.
///
/// Measures the local density deviation of each sample with respect to its
/// k nearest neighbors. Points with substantially lower density than their
/// neighbors are classified as outliers.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor<F> {
    /// Number of neighbors to use. Default: `20`.
    pub n_neighbors: usize,
    /// The fraction of data expected to be outliers. Default: `0.1`.
    pub contamination: f64,
    /// The algorithm to use for neighbor search. Default: `Auto`.
    pub algorithm: Algorithm,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> LocalOutlierFactor<F> {
    /// Create a new `LocalOutlierFactor` with default settings.
    ///
    /// Defaults: `n_neighbors = 20`, `contamination = 0.1`, `algorithm = Auto`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 20,
            contamination: 0.1,
            algorithm: Algorithm::Auto,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the contamination fraction.
    #[must_use]
    pub fn with_contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl<F: Float> Default for LocalOutlierFactor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Local Outlier Factor model.
///
/// Stores the computed LOF scores and a threshold for classification.
/// Points with LOF above the threshold are outliers.
#[derive(Debug, Clone)]
pub struct FittedLocalOutlierFactor<F> {
    /// The training data (needed for prediction on new points).
    x_train: Array2<F>,
    /// LOF scores for each training sample.
    lof_scores: Vec<F>,
    /// Number of neighbors used.
    n_neighbors: usize,
    /// Threshold LOF score: points above this are outliers.
    threshold: F,
}

/// Compute Euclidean distance between two slices.
fn euclidean_dist<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| {
            let d = ai - bi;
            acc + d * d
        })
        .sqrt()
}

/// Find k nearest neighbors (brute force), excluding self.
/// Returns (neighbor_index, distance) sorted by distance.
fn knn_brute_force<F: Float>(data: &[Vec<F>], query_idx: usize, k: usize) -> Vec<(usize, F)> {
    let mut dists: Vec<(usize, F)> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != query_idx)
        .map(|(i, row)| (i, euclidean_dist(&data[query_idx], row)))
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Compute LOF scores for all points in the dataset.
fn compute_lof_scores<F: Float>(data: &[Vec<F>], k: usize) -> Vec<F> {
    let n = data.len();
    let effective_k = k.min(n - 1);

    if effective_k == 0 {
        return vec![F::one(); n];
    }

    // Step 1: Find k nearest neighbors for each point.
    let neighbors: Vec<Vec<(usize, F)>> = (0..n)
        .map(|i| knn_brute_force(data, i, effective_k))
        .collect();

    // Step 2: Compute k-distance for each point (distance to k-th neighbor).
    let k_dist: Vec<F> = neighbors
        .iter()
        .map(|nn| {
            if nn.is_empty() {
                F::zero()
            } else {
                nn[nn.len() - 1].1
            }
        })
        .collect();

    // Step 3: Compute local reachability density for each point.
    let eps = F::from(1e-15).unwrap();
    let lrd: Vec<F> = neighbors
        .iter()
        .map(|nn| {
            if nn.is_empty() {
                return F::one();
            }
            let sum_reach: F = nn
                .iter()
                .map(|&(neighbor_idx, dist)| {
                    // reach_dist(p, o) = max(k_distance(o), dist(p, o))
                    k_dist[neighbor_idx].max(dist)
                })
                .fold(F::zero(), |a, b| a + b);

            if sum_reach < eps {
                F::from(1e10).unwrap() // Very high density if all neighbors coincide.
            } else {
                F::from(nn.len()).unwrap() / sum_reach
            }
        })
        .collect();

    // Step 4: Compute LOF for each point.
    neighbors
        .iter()
        .enumerate()
        .map(|(i, nn)| {
            if nn.is_empty() || lrd[i] < eps {
                return F::one();
            }
            let mean_neighbor_lrd: F = nn
                .iter()
                .map(|&(neighbor_idx, _)| lrd[neighbor_idx])
                .fold(F::zero(), |a, b| a + b)
                / F::from(nn.len()).unwrap();

            mean_neighbor_lrd / lrd[i]
        })
        .collect()
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for LocalOutlierFactor<F> {
    type Fitted = FittedLocalOutlierFactor<F>;
    type Error = FerroError;

    /// Fit the LOF model by computing LOF scores for all training samples.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_neighbors` is zero or
    ///   `contamination` is not in `(0, 0.5]`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedLocalOutlierFactor<F>, FerroError> {
        let n_samples = x.nrows();

        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        if self.contamination <= 0.0 || self.contamination > 0.5 {
            return Err(FerroError::InvalidParameter {
                name: "contamination".into(),
                reason: "must be in (0, 0.5]".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LocalOutlierFactor requires at least 2 samples".into(),
            });
        }

        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();
        let lof_scores = compute_lof_scores(&data, self.n_neighbors);

        // Determine threshold from contamination: the (1 - contamination)-th
        // percentile of LOF scores. Points with LOF > threshold are outliers.
        let mut sorted_scores: Vec<F> = lof_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // The number of expected outliers.
        let n_outliers = (self.contamination * n_samples as f64).ceil().max(1.0) as usize;
        let n_outliers = n_outliers.min(n_samples - 1);
        // Threshold = the LOF score at position (n - n_outliers - 1), i.e., the
        // highest inlier score. Points strictly above this are outliers.
        let threshold_idx = n_samples - n_outliers - 1;
        let threshold = sorted_scores[threshold_idx];

        Ok(FittedLocalOutlierFactor {
            x_train: x.clone(),
            lof_scores,
            n_neighbors: self.n_neighbors,
            threshold,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLocalOutlierFactor<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict inlier (+1) or outlier (-1) labels.
    ///
    /// For training data, uses the pre-computed LOF scores.
    /// For new data, recomputes LOF scores against the training set.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        let n_samples = x.nrows();
        let n_train = self.x_train.nrows();

        // Check if x is exactly the training data (by pointer comparison of shape).
        // This is a heuristic; for general case we recompute.
        let is_training = n_samples == n_train
            && (0..n_samples).all(|i| (0..n_features).all(|j| x[[i, j]] == self.x_train[[i, j]]));

        let scores = if is_training {
            self.lof_scores.clone()
        } else {
            // Compute LOF for new data against training set.
            let train_data: Vec<Vec<F>> =
                (0..n_train).map(|i| self.x_train.row(i).to_vec()).collect();

            let effective_k = self.n_neighbors.min(n_train);
            let eps = F::from(1e-15).unwrap();

            // Compute k-distances for training points.
            let train_neighbors: Vec<Vec<(usize, F)>> = (0..n_train)
                .map(|i| knn_brute_force(&train_data, i, effective_k))
                .collect();

            let k_dist_train: Vec<F> = train_neighbors
                .iter()
                .map(|nn| {
                    if nn.is_empty() {
                        F::zero()
                    } else {
                        nn[nn.len() - 1].1
                    }
                })
                .collect();

            let lrd_train: Vec<F> = train_neighbors
                .iter()
                .map(|nn| {
                    if nn.is_empty() {
                        return F::one();
                    }
                    let sum_reach: F = nn
                        .iter()
                        .map(|&(neighbor_idx, dist)| k_dist_train[neighbor_idx].max(dist))
                        .fold(F::zero(), |a, b| a + b);
                    if sum_reach < eps {
                        F::from(1e10).unwrap()
                    } else {
                        F::from(nn.len()).unwrap() / sum_reach
                    }
                })
                .collect();

            // For each new point, find k nearest training neighbors.
            (0..n_samples)
                .map(|i| {
                    let query: Vec<F> = x.row(i).to_vec();

                    // Find k nearest training points.
                    let mut dists: Vec<(usize, F)> = train_data
                        .iter()
                        .enumerate()
                        .map(|(j, row)| (j, euclidean_dist(&query, row)))
                        .collect();
                    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    dists.truncate(effective_k);

                    if dists.is_empty() {
                        return F::one();
                    }

                    // Reachability distance and LRD for the query.
                    let sum_reach: F = dists
                        .iter()
                        .map(|&(neighbor_idx, dist)| k_dist_train[neighbor_idx].max(dist))
                        .fold(F::zero(), |a, b| a + b);

                    let lrd_query = if sum_reach < eps {
                        F::from(1e10).unwrap()
                    } else {
                        F::from(dists.len()).unwrap() / sum_reach
                    };

                    if lrd_query < eps {
                        return F::one();
                    }

                    // LOF: mean neighbor LRD / query LRD.
                    let mean_neighbor_lrd: F = dists
                        .iter()
                        .map(|&(neighbor_idx, _)| lrd_train[neighbor_idx])
                        .fold(F::zero(), |a, b| a + b)
                        / F::from(dists.len()).unwrap();

                    mean_neighbor_lrd / lrd_query
                })
                .collect()
        };

        let mut predictions = Array1::<isize>::zeros(n_samples);
        for (i, &score) in scores.iter().enumerate() {
            predictions[i] = if score <= self.threshold { 1 } else { -1 };
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> FittedLocalOutlierFactor<F> {
    /// Get the LOF scores for the training samples.
    ///
    /// Higher values indicate more anomalous points.
    #[must_use]
    pub fn lof_scores(&self) -> &[F] {
        &self.lof_scores
    }

    /// Get the threshold used for inlier/outlier classification.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_cluster_with_outlier() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05,
                -0.05, 10.0, 10.0, // outlier
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_lof_fit() {
        let x = make_cluster_with_outlier();
        let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(5);
        let result = model.fit(&x, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_lof_outlier_has_high_score() {
        let x = make_cluster_with_outlier();
        let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(5);
        let fitted = model.fit(&x, &()).unwrap();
        let scores = fitted.lof_scores();

        // The last point (outlier) should have the highest LOF score.
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 8, "Outlier should have highest LOF score");
    }

    #[test]
    fn test_lof_predict_training_data() {
        let x = make_cluster_with_outlier();
        let model = LocalOutlierFactor::<f64>::new()
            .with_n_neighbors(5)
            .with_contamination(0.15);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);

        // Most points should be inliers.
        let inliers: usize = preds.iter().filter(|&&p| p == 1).count();
        assert!(inliers >= 7, "Expected at least 7 inliers, got {inliers}");

        // The outlier (last point) should be detected.
        assert_eq!(preds[8], -1, "Outlier should be classified as -1");
    }

    #[test]
    fn test_lof_predict_new_data() {
        let x_train = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05,
                -0.05,
            ],
        )
        .unwrap();

        let model = LocalOutlierFactor::<f64>::new()
            .with_n_neighbors(5)
            .with_contamination(0.2);
        let fitted = model.fit(&x_train, &()).unwrap();

        // Test with an outlier point.
        let x_test = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds[0], -1, "Far-away point should be an outlier");
    }

    #[test]
    fn test_lof_shape_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(2);
        let fitted = model.fit(&x_train, &()).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_lof_invalid_n_neighbors() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(0);
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lof_invalid_contamination() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();

        let model = LocalOutlierFactor::<f64>::new().with_contamination(0.0);
        assert!(model.fit(&x, &()).is_err());

        let model2 = LocalOutlierFactor::<f64>::new().with_contamination(0.6);
        assert!(model2.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lof_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let model = LocalOutlierFactor::<f64>::new();
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_lof_default() {
        let model = LocalOutlierFactor::<f64>::default();
        assert_eq!(model.n_neighbors, 20);
        assert!((model.contamination - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_lof_uniform_cluster() {
        // All points in a tight cluster should have LOF close to 1.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, -0.01, 0.0, 0.0, -0.01,
            ],
        )
        .unwrap();

        let model = LocalOutlierFactor::<f64>::new().with_n_neighbors(3);
        let fitted = model.fit(&x, &()).unwrap();
        let scores = fitted.lof_scores();

        // All LOF scores should be close to 1 for a uniform cluster.
        for &score in scores {
            assert!(
                (score - 1.0).abs() < 1.0,
                "LOF score {score} too far from 1.0 for uniform cluster"
            );
        }
    }

    #[test]
    fn test_lof_builder_pattern() {
        let model = LocalOutlierFactor::<f64>::new()
            .with_n_neighbors(10)
            .with_contamination(0.2)
            .with_algorithm(Algorithm::BruteForce);

        assert_eq!(model.n_neighbors, 10);
        assert!((model.contamination - 0.2).abs() < 1e-10);
        assert_eq!(model.algorithm, Algorithm::BruteForce);
    }
}
