//! Bisecting K-Means clustering.
//!
//! This module provides [`BisectingKMeans`], a divisive hierarchical clustering
//! algorithm that recursively bisects clusters until the desired number of
//! clusters is reached.
//!
//! # Algorithm
//!
//! 1. Start with all points in a single cluster.
//! 2. Select the cluster to bisect based on the [`BisectingStrategy`]:
//!    - [`LargestCluster`](BisectingStrategy::LargestCluster): pick the cluster
//!      with the most samples.
//!    - [`LargestSSE`](BisectingStrategy::LargestSSE): pick the cluster with
//!      the largest sum of squared errors (inertia).
//! 3. Run 2-means (K-Means with `k=2`) on the selected cluster's points,
//!    repeating `n_init` times and keeping the best split.
//! 4. Replace the parent cluster with the two child clusters.
//! 5. Repeat steps 2-4 until the target number of clusters is reached.
//!
//! # Advantages
//!
//! Compared to standard K-Means:
//! - More deterministic: the hierarchical splitting is less sensitive to
//!   initialization.
//! - Can produce a dendrogram-like hierarchy of clusters.
//! - Tends to produce more balanced clusters.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::BisectingKMeans;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//!
//! let model = BisectingKMeans::<f64>::new(2);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.predict(&x).unwrap();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::marker::PhantomData;

/// Strategy for selecting which cluster to bisect next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BisectingStrategy {
    /// Bisect the cluster with the most samples.
    LargestCluster,
    /// Bisect the cluster with the largest sum of squared errors.
    LargestSSE,
}

/// Bisecting K-Means clustering configuration (unfitted).
///
/// Holds hyperparameters for the Bisecting K-Means algorithm. Call
/// [`Fit::fit`] to run the algorithm and produce a [`FittedBisectingKMeans`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BisectingKMeans<F> {
    /// Target number of clusters.
    n_clusters: usize,
    /// Maximum number of iterations per 2-means bisection.
    max_iter: usize,
    /// Number of K-Means restarts per bisection (best-of-n_init).
    n_init: usize,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    /// Strategy for selecting which cluster to bisect.
    bisecting_strategy: BisectingStrategy,
    /// Phantom data to retain the float type parameter.
    _marker: PhantomData<F>,
}

impl<F: Float> BisectingKMeans<F> {
    /// Create a new `BisectingKMeans` with the given target number of clusters.
    ///
    /// Defaults: `max_iter = 300`, `n_init = 10`, `random_state = None`,
    /// `bisecting_strategy = LargestCluster`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            n_init: 10,
            random_state: None,
            bisecting_strategy: BisectingStrategy::LargestCluster,
            _marker: PhantomData,
        }
    }

    /// Set the maximum number of iterations per bisection.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of K-Means restarts per bisection.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the bisecting strategy.
    #[must_use]
    pub fn with_bisecting_strategy(mut self, strategy: BisectingStrategy) -> Self {
        self.bisecting_strategy = strategy;
        self
    }
}

/// Fitted Bisecting K-Means model.
///
/// Stores the cluster centers, labels, and inertia. Implements [`Predict`]
/// to assign new data points to the nearest cluster center.
#[derive(Debug, Clone)]
pub struct FittedBisectingKMeans<F> {
    /// Cluster center coordinates, shape `(n_clusters, n_features)`.
    cluster_centers_: Array2<F>,
    /// Cluster label for each training sample.
    labels_: Array1<isize>,
    /// Total inertia (sum of squared distances to nearest center).
    inertia_: F,
}

impl<F: Float> FittedBisectingKMeans<F> {
    /// Return the cluster centers, shape `(n_clusters, n_features)`.
    #[must_use]
    pub fn cluster_centers(&self) -> &Array2<F> {
        &self.cluster_centers_
    }

    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the total inertia.
    #[must_use]
    pub fn inertia(&self) -> F {
        self.inertia_
    }

    /// Return the number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers_.nrows()
    }
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// A cluster being tracked during bisection.
#[derive(Debug, Clone)]
struct ClusterInfo<F> {
    /// Indices of samples in this cluster (into the original data matrix).
    indices: Vec<usize>,
    /// Centroid of this cluster.
    center: Vec<F>,
    /// Sum of squared errors for this cluster.
    sse: F,
}

/// Compute the centroid and SSE of a set of points.
fn compute_cluster_stats<F: Float>(x: &Array2<F>, indices: &[usize]) -> (Vec<F>, F) {
    let n_features = x.ncols();
    let n = indices.len();

    if n == 0 {
        return (vec![F::zero(); n_features], F::zero());
    }

    let n_f = F::from(n).unwrap();

    // Compute centroid.
    let mut center = vec![F::zero(); n_features];
    for &idx in indices {
        for j in 0..n_features {
            center[j] = center[j] + x[[idx, j]];
        }
    }
    for val in &mut center {
        *val = *val / n_f;
    }

    // Compute SSE.
    let mut sse = F::zero();
    for &idx in indices {
        let row = x.row(idx);
        let row_slice = row.as_slice().unwrap_or(&[]);
        sse = sse + squared_euclidean(row_slice, &center);
    }

    (center, sse)
}

/// Run 2-means on a subset of data points. Returns (labels_0_or_1, center0, center1, total_sse).
fn run_2means<F: Float>(
    x: &Array2<F>,
    indices: &[usize],
    max_iter: usize,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<F>, Vec<F>, F) {
    let n = indices.len();
    let n_features = x.ncols();

    // Pick 2 initial centers from the subset.
    let idx0 = rng.random_range(0..n);
    let mut idx1 = rng.random_range(0..n);
    // Ensure different seeds if possible.
    if n > 1 {
        while idx1 == idx0 {
            idx1 = rng.random_range(0..n);
        }
    }

    let mut center0: Vec<F> = (0..n_features).map(|j| x[[indices[idx0], j]]).collect();
    let mut center1: Vec<F> = (0..n_features).map(|j| x[[indices[idx1], j]]).collect();

    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assignment step.
        let mut changed = false;
        for (li, &sample_idx) in indices.iter().enumerate() {
            let row = x.row(sample_idx);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let d0 = squared_euclidean(row_slice, &center0);
            let d1 = squared_euclidean(row_slice, &center1);
            let new_label = if d0 <= d1 { 0 } else { 1 };
            if new_label != labels[li] {
                labels[li] = new_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centers.
        let mut new_c0 = vec![F::zero(); n_features];
        let mut new_c1 = vec![F::zero(); n_features];
        let mut count0 = F::zero();
        let mut count1 = F::zero();

        for (li, &sample_idx) in indices.iter().enumerate() {
            if labels[li] == 0 {
                count0 = count0 + F::one();
                for j in 0..n_features {
                    new_c0[j] = new_c0[j] + x[[sample_idx, j]];
                }
            } else {
                count1 = count1 + F::one();
                for j in 0..n_features {
                    new_c1[j] = new_c1[j] + x[[sample_idx, j]];
                }
            }
        }

        if count0 > F::zero() {
            for val in &mut new_c0 {
                *val = *val / count0;
            }
            center0 = new_c0;
        }
        if count1 > F::zero() {
            for val in &mut new_c1 {
                *val = *val / count1;
            }
            center1 = new_c1;
        }
    }

    // Compute total SSE.
    let mut total_sse = F::zero();
    for (li, &sample_idx) in indices.iter().enumerate() {
        let row = x.row(sample_idx);
        let row_slice = row.as_slice().unwrap_or(&[]);
        let center = if labels[li] == 0 { &center0 } else { &center1 };
        total_sse = total_sse + squared_euclidean(row_slice, center);
    }

    (labels, center0, center1, total_sse)
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for BisectingKMeans<F> {
    type Fitted = FittedBisectingKMeans<F>;
    type Error = FerroError;

    /// Fit the Bisecting K-Means model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_clusters` is zero or
    /// `n_init` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if the number of samples
    /// is less than `n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedBisectingKMeans<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate parameters.
        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }

        if self.n_init == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_init".into(),
                reason: "must be at least 1".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: 0,
                context: "BisectingKMeans requires at least n_clusters samples".into(),
            });
        }

        if n_samples < self.n_clusters {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: n_samples,
                context: "BisectingKMeans requires at least n_clusters samples".into(),
            });
        }

        // Initialize with all samples in one cluster.
        let all_indices: Vec<usize> = (0..n_samples).collect();
        let (center, sse) = compute_cluster_stats(x, &all_indices);

        let mut clusters: Vec<ClusterInfo<F>> = vec![ClusterInfo {
            indices: all_indices,
            center,
            sse,
        }];

        let base_seed = self.random_state.unwrap_or(0);
        let mut split_count: u64 = 0;

        // Bisect until we reach the target number of clusters.
        while clusters.len() < self.n_clusters {
            // Select cluster to bisect.
            let target_idx = match self.bisecting_strategy {
                BisectingStrategy::LargestCluster => clusters
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.indices.len() >= 2)
                    .max_by_key(|(_, c)| c.indices.len())
                    .map(|(i, _)| i),
                BisectingStrategy::LargestSSE => clusters
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.indices.len() >= 2)
                    .max_by(|(_, a), (_, b)| {
                        a.sse
                            .partial_cmp(&b.sse)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i),
            };

            let target_idx = match target_idx {
                Some(i) => i,
                None => break, // No cluster with >= 2 points to split.
            };

            let target = &clusters[target_idx];
            let target_indices = &target.indices;

            // Run 2-means n_init times, keep best.
            let mut best_labels: Option<Vec<usize>> = None;
            let mut best_c0: Vec<F> = Vec::new();
            let mut best_c1: Vec<F> = Vec::new();
            let mut best_sse = F::max_value();

            for run in 0..self.n_init {
                let seed = base_seed
                    .wrapping_add(split_count.wrapping_mul(1000))
                    .wrapping_add(run as u64);
                let mut rng = StdRng::seed_from_u64(seed);

                let (labels, c0, c1, sse) = run_2means(x, target_indices, self.max_iter, &mut rng);

                if sse < best_sse {
                    best_sse = sse;
                    best_labels = Some(labels);
                    best_c0 = c0;
                    best_c1 = c1;
                }
            }

            split_count += 1;

            let best_labels = best_labels.unwrap();

            // Split the target cluster into two.
            let mut indices0 = Vec::new();
            let mut indices1 = Vec::new();
            for (li, &sample_idx) in target_indices.iter().enumerate() {
                if best_labels[li] == 0 {
                    indices0.push(sample_idx);
                } else {
                    indices1.push(sample_idx);
                }
            }

            // Compute SSE for each child cluster.
            let sse0 = if indices0.is_empty() {
                F::zero()
            } else {
                let mut sse = F::zero();
                for &idx in &indices0 {
                    let row = x.row(idx);
                    let row_slice = row.as_slice().unwrap_or(&[]);
                    sse = sse + squared_euclidean(row_slice, &best_c0);
                }
                sse
            };

            let sse1 = if indices1.is_empty() {
                F::zero()
            } else {
                let mut sse = F::zero();
                for &idx in &indices1 {
                    let row = x.row(idx);
                    let row_slice = row.as_slice().unwrap_or(&[]);
                    sse = sse + squared_euclidean(row_slice, &best_c1);
                }
                sse
            };

            // Remove the target cluster and add two children.
            clusters.remove(target_idx);

            if !indices0.is_empty() {
                clusters.push(ClusterInfo {
                    indices: indices0,
                    center: best_c0,
                    sse: sse0,
                });
            }
            if !indices1.is_empty() {
                clusters.push(ClusterInfo {
                    indices: indices1,
                    center: best_c1,
                    sse: sse1,
                });
            }
        }

        // Build final results.
        let n_final_clusters = clusters.len();
        let mut cluster_centers = Array2::zeros((n_final_clusters, n_features));
        let mut labels = Array1::from_elem(n_samples, 0isize);
        let mut total_inertia = F::zero();

        for (ci, cluster) in clusters.iter().enumerate() {
            for j in 0..n_features {
                cluster_centers[[ci, j]] = cluster.center[j];
            }
            for &idx in &cluster.indices {
                labels[idx] = ci as isize;
            }
            total_inertia = total_inertia + cluster.sse;
        }

        Ok(FittedBisectingKMeans {
            cluster_centers_: cluster_centers,
            labels_: labels,
            inertia_: total_inertia,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBisectingKMeans<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Assign each sample to the nearest cluster center.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let n_features = x.ncols();
        let expected_features = self.cluster_centers_.ncols();

        if n_features != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![n_features],
                context: "number of features must match fitted BisectingKMeans model".into(),
            });
        }

        let n_samples = x.nrows();
        let k = self.cluster_centers_.nrows();
        let mut labels = Array1::from_elem(n_samples, 0isize);

        for i in 0..n_samples {
            let row = x.row(i);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let mut best_label = 0isize;
            let mut best_dist = F::max_value();
            for c in 0..k {
                let center = self.cluster_centers_.row(c);
                let center_slice = center.as_slice().unwrap_or(&[]);
                let d = squared_euclidean(row_slice, center_slice);
                if d < best_dist {
                    best_dist = d;
                    best_label = c as isize;
                }
            }
            labels[i] = best_label;
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create well-separated 2D blobs for testing.
    fn make_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                // Cluster 0 near (0, 0)
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // Cluster 1 near (10, 10)
                10.0, 10.0, 10.1, 10.1, 9.9, 10.1, // Cluster 2 near (0, 10)
                0.0, 10.0, 0.1, 10.1, -0.1, 9.9,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_well_separated_blobs() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // Points in the same blob should have the same label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        // Different blobs should have different labels.
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_two_clusters() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 10.0, 10.0, 10.5, 10.0, 10.0, 10.5, 10.5,
                10.5,
            ],
        )
        .unwrap();

        let model = BisectingKMeans::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_single_cluster() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let model = BisectingKMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // All in one cluster.
        for &label in fitted.labels() {
            assert_eq!(label, 0);
        }
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_predict_assigns_correctly() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // Predict on training data should match fit labels.
        let predicted = fitted.predict(&x).unwrap();
        assert_eq!(predicted, *fitted.labels());
    }

    #[test]
    fn test_predict_new_data() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // New points near each cluster center.
        let new_x =
            Array2::from_shape_vec((3, 2), vec![0.05, 0.05, 10.05, 10.05, 0.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        // Each new point should be in the same cluster as the nearby training points.
        assert_eq!(new_labels[0], fitted.labels()[0]);
        assert_eq!(new_labels[1], fitted.labels()[3]);
        assert_eq!(new_labels[2], fitted.labels()[6]);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_inertia_non_negative() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert!(fitted.inertia() >= 0.0);
    }

    #[test]
    fn test_cluster_centers_shape() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.cluster_centers().dim(), (3, 2));
    }

    #[test]
    fn test_largest_sse_strategy() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3)
            .with_random_state(42)
            .with_bisecting_strategy(BisectingStrategy::LargestSSE);
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
    }

    #[test]
    fn test_reproducibility() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(123);

        let fitted1 = model.fit(&x, &()).unwrap();
        let fitted2 = model.fit(&x, &()).unwrap();

        assert_eq!(fitted1.labels(), fitted2.labels());
    }

    #[test]
    fn test_zero_clusters() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_k_greater_than_n_samples() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let model = BisectingKMeans::<f64>::new(5);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = BisectingKMeans::<f64>::new(3);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_n_init() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_n_init(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = BisectingKMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 1);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1,
            ],
        )
        .unwrap();

        let model = BisectingKMeans::<f32>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_identical_points() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        // With identical points and k=1, should work fine.
        let model = BisectingKMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_labels_in_range() {
        let x = make_blobs();
        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let n_clusters = fitted.n_clusters() as isize;
        for &label in fitted.labels() {
            assert!(label >= 0);
            assert!(label < n_clusters);
        }
    }

    #[test]
    fn test_n_init_picks_best() {
        let x = make_blobs();

        let model_1 = BisectingKMeans::<f64>::new(3)
            .with_random_state(42)
            .with_n_init(1);
        let fitted_1 = model_1.fit(&x, &()).unwrap();

        let model_10 = BisectingKMeans::<f64>::new(3)
            .with_random_state(42)
            .with_n_init(10);
        let fitted_10 = model_10.fit(&x, &()).unwrap();

        // n_init=10 should produce inertia <= n_init=1 (or very close).
        assert!(fitted_10.inertia() <= fitted_1.inertia() + 1e-6);
    }

    #[test]
    fn test_k_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();

        let model = BisectingKMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // Each point should be its own cluster.
        assert_eq!(fitted.n_clusters(), 3);
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
    }
}
