//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
//!
//! This module provides [`DBSCAN`], a density-based clustering algorithm
//! that groups together points that are closely packed together, marking
//! as outliers points that lie alone in low-density regions.
//!
//! # Algorithm
//!
//! 1. For each point, find all neighbors within distance `eps`.
//! 2. Points with at least `min_samples` neighbors (including themselves)
//!    are **core points**.
//! 3. Clusters are formed by connecting core points that are within `eps`
//!    of each other, and assigning border points to the cluster of their
//!    nearest core point.
//! 4. Points that are not reachable from any core point are labeled as
//!    **noise** (label = -1).
//!
//! # Notes
//!
//! DBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! only labels the training data. Use the fitted labels directly.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::DBSCAN;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//!
//! let model = DBSCAN::<f64>::new(1.0);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.labels();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::VecDeque;

/// DBSCAN clustering configuration (unfitted).
///
/// Holds hyperparameters for the DBSCAN algorithm. Call [`Fit::fit`]
/// to run the algorithm and produce a [`FittedDBSCAN`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct DBSCAN<F> {
    /// The maximum distance between two samples for them to be considered
    /// as in the same neighborhood.
    pub eps: F,
    /// The minimum number of samples in a neighborhood for a point to be
    /// considered a core point (including the point itself).
    pub min_samples: usize,
}

impl<F: Float> DBSCAN<F> {
    /// Create a new `DBSCAN` with the given `eps` radius.
    ///
    /// Uses default `min_samples = 5`.
    #[must_use]
    pub fn new(eps: F) -> Self {
        Self {
            eps,
            min_samples: 5,
        }
    }

    /// Set the minimum number of samples for core points.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }
}

/// Fitted DBSCAN model.
///
/// Stores the cluster labels and core sample indices from the training run.
/// Noise points are labeled with -1.
///
/// DBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedDBSCAN<F> {
    /// Cluster label for each training sample. Noise points have label -1.
    labels_: Array1<isize>,
    /// Indices of core samples in the training data.
    core_sample_indices_: Vec<usize>,
    /// Phantom data to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedDBSCAN<F> {
    /// Return the cluster labels for the training data.
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the indices of core samples in the training data.
    #[must_use]
    pub fn core_sample_indices(&self) -> &[usize] {
        &self.core_sample_indices_
    }

    /// Return the number of clusters found (excluding noise).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels_.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Find all neighbors within `eps` distance of point `idx`.
fn region_query<F: Float>(x: &Array2<F>, idx: usize, eps_sq: F) -> Vec<usize> {
    let n_samples = x.nrows();
    let row = x.row(idx);
    let row_slice = row.as_slice().unwrap_or(&[]);

    let mut neighbors = Vec::new();
    for j in 0..n_samples {
        let other = x.row(j);
        let other_slice = other.as_slice().unwrap_or(&[]);
        if squared_euclidean(row_slice, other_slice) <= eps_sq {
            neighbors.push(j);
        }
    }
    neighbors
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for DBSCAN<F> {
    type Fitted = FittedDBSCAN<F>;
    type Error = FerroError;

    /// Fit the DBSCAN model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `eps` is not positive
    /// or `min_samples` is zero.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedDBSCAN<F>, FerroError> {
        // Validate parameters.
        if self.eps <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "eps".into(),
                reason: "must be positive".into(),
            });
        }

        if self.min_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedDBSCAN {
                labels_: Array1::zeros(0),
                core_sample_indices_: Vec::new(),
                _marker: std::marker::PhantomData,
            });
        }

        let eps_sq = self.eps * self.eps;

        // Step 1: Find neighborhoods for all points.
        let neighborhoods: Vec<Vec<usize>> =
            (0..n_samples).map(|i| region_query(x, i, eps_sq)).collect();

        // Step 2: Identify core points.
        let is_core: Vec<bool> = neighborhoods
            .iter()
            .map(|n| n.len() >= self.min_samples)
            .collect();

        let core_sample_indices: Vec<usize> = (0..n_samples).filter(|&i| is_core[i]).collect();

        // Step 3: Expand clusters from core points via BFS.
        let mut labels = Array1::from_elem(n_samples, -1isize);
        let mut current_cluster: isize = -1;

        for i in 0..n_samples {
            // Skip non-core or already-labeled points.
            if !is_core[i] || labels[i] != -1 {
                continue;
            }

            // Start a new cluster.
            current_cluster += 1;
            labels[i] = current_cluster;

            // BFS expansion.
            let mut queue: VecDeque<usize> = VecDeque::new();
            for &neighbor in &neighborhoods[i] {
                if neighbor != i {
                    queue.push_back(neighbor);
                }
            }

            while let Some(q) = queue.pop_front() {
                if labels[q] == -1 {
                    // Assign to current cluster (was noise or unvisited).
                    labels[q] = current_cluster;
                }

                if !is_core[q] {
                    // Border point — don't expand further.
                    continue;
                }

                // If this core point was already assigned to this cluster
                // by a prior BFS step, skip expanding again.
                if labels[q] == current_cluster {
                    // Expand: add unvisited neighbors to queue.
                    for &neighbor in &neighborhoods[q] {
                        if labels[neighbor] == -1 {
                            labels[neighbor] = current_cluster;
                            if is_core[neighbor] {
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }
        }

        Ok(FittedDBSCAN {
            labels_: labels,
            core_sample_indices_: core_sample_indices,
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated clusters.
    fn make_two_clusters() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                // Cluster A near (0, 0)
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, // Cluster B near (10, 10)
                10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5, 10.5,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_simple_clusters() {
        let x = make_two_clusters();
        let model = DBSCAN::<f64>::new(1.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // First 4 should be in one cluster, last 4 in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);

        // Two clusters found.
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_noise_detection() {
        // Two tight clusters + one outlier.
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, // Outlier
                100.0, 100.0,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // Outlier should be noise.
        assert_eq!(labels[6], -1);
        // Others should not be noise.
        assert!(labels[0] >= 0);
        assert!(labels[3] >= 0);
    }

    #[test]
    fn test_core_border_noise_identification() {
        // Ring of points: 3 core, 1 border, 1 noise.
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // core (neighbors: 0,1,2)
                0.3, 0.0, // core (neighbors: 0,1,2)
                0.0, 0.3, // core (neighbors: 0,1,2)
                0.6, 0.0, // border (neighbor: 1 at least)
                10.0, 10.0, // noise
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        let core_indices = fitted.core_sample_indices();

        // Points 0, 1, 2 should be core points.
        assert!(core_indices.contains(&0));
        assert!(core_indices.contains(&1));
        assert!(core_indices.contains(&2));

        // Point 3 is a border point (reachable from core point 1).
        assert!(labels[3] >= 0);
        assert!(!core_indices.contains(&3));

        // Point 4 is noise.
        assert_eq!(labels[4], -1);
    }

    #[test]
    fn test_all_noise_with_high_min_samples() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // min_samples too high: no point has enough neighbors.
        let model = DBSCAN::<f64>::new(0.5).with_min_samples(100);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        for &label in labels.iter() {
            assert_eq!(label, -1);
        }
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_all_noise_with_tiny_eps() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // eps too small: no points are neighbors of each other (except self).
        let model = DBSCAN::<f64>::new(0.001).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        for &label in fitted.labels().iter() {
            assert_eq!(label, -1);
        }
    }

    #[test]
    fn test_single_point_cluster() {
        // One cluster + one isolated point.
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 100.0, 100.0])
            .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // Isolated point should be noise.
        assert_eq!(fitted.labels()[3], -1);
        // Others should be in a cluster.
        assert!(fitted.labels()[0] >= 0);
    }

    #[test]
    fn test_all_in_one_cluster() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1]).unwrap();

        let model = DBSCAN::<f64>::new(1.0).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // All should be in the same cluster.
        let labels = fitted.labels();
        assert!(labels[0] >= 0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = DBSCAN::<f64>::new(1.0);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 0);
        assert!(fitted.core_sample_indices().is_empty());
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(1);
        let fitted = model.fit(&x, &()).unwrap();

        // With min_samples=1, a single point is a core point and its own cluster.
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.core_sample_indices(), &[0]);
    }

    #[test]
    fn test_single_sample_noise() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // With min_samples=2, a single point cannot be a core point.
        assert_eq!(fitted.labels()[0], -1);
    }

    #[test]
    fn test_invalid_eps() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(-1.0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_eps() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(0.0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_min_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = DBSCAN::<f64>::new(1.0).with_min_samples(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_core_sample_indices_correct() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 100.0, 100.0],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        // All core samples should have enough neighbors.
        for &idx in fitted.core_sample_indices() {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f32>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 6);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_three_clusters() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
                10.0, 0.1,
            ],
        )
        .unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_clusters(), 3);

        let labels = fitted.labels();
        // Same cluster groupings.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        // Different clusters.
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_identical_points() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let model = DBSCAN::<f64>::new(0.5).with_min_samples(2);
        let fitted = model.fit(&x, &()).unwrap();

        // All identical points should be in the same cluster.
        let labels = fitted.labels();
        assert!(labels[0] >= 0);
        for &label in labels.iter() {
            assert_eq!(label, labels[0]);
        }
    }
}
