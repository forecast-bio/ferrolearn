//! K-Means clustering with k-Means++ initialization.
//!
//! This module provides [`KMeans`], an unsupervised clustering algorithm
//! that partitions data into `k` clusters by minimizing within-cluster
//! sum of squared distances (inertia). The implementation uses Lloyd's
//! algorithm with k-Means++ initialization for smart centroid seeding.
//!
//! # Algorithm
//!
//! 1. **k-Means++ initialization**: pick the first center uniformly at random,
//!    then pick each subsequent center with probability proportional to D(x)²
//!    (squared distance to the nearest existing center).
//! 2. **Lloyd's algorithm**: alternate between assigning samples to the nearest
//!    centroid and recomputing centroids as the mean of their assigned samples.
//! 3. **Multi-start**: repeat `n_init` times and keep the result with the
//!    lowest inertia.
//!
//! The assignment step is parallelized with Rayon.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::KMeans;
//! use ferrolearn_core::{Fit, Predict, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//!
//! let model = KMeans::<f64>::new(2);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.predict(&x).unwrap();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// K-Means clustering configuration (unfitted).
///
/// Holds hyperparameters for the k-Means algorithm. Call [`Fit::fit`]
/// to run the algorithm and produce a [`FittedKMeans`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct KMeans<F> {
    /// Number of clusters to form.
    pub n_clusters: usize,
    /// Maximum number of Lloyd iterations per run.
    pub max_iter: usize,
    /// Convergence tolerance: the algorithm stops when the maximum
    /// centroid movement is less than this value.
    pub tol: F,
    /// Number of independent runs with different initializations.
    /// The result with the lowest inertia is kept.
    pub n_init: usize,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
}

impl<F: Float> KMeans<F> {
    /// Create a new `KMeans` with the given number of clusters.
    ///
    /// Uses default values: `max_iter = 300`, `tol = 1e-4`,
    /// `n_init = 10`, `random_state = None`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            n_init: 10,
            random_state: None,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of independent runs.
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
}

/// Fitted K-Means model.
///
/// Stores the learned cluster centers, labels, inertia, and iteration count.
/// Implements [`Predict`] to assign new samples to clusters and [`Transform`]
/// to compute distances to each centroid.
#[derive(Debug, Clone)]
pub struct FittedKMeans<F> {
    /// Cluster center coordinates, shape `(n_clusters, n_features)`.
    cluster_centers_: Array2<F>,
    /// Cluster label for each training sample.
    labels_: Array1<usize>,
    /// Sum of squared distances of samples to their closest cluster center.
    inertia_: F,
    /// Number of iterations run in the best run.
    n_iter_: usize,
}

impl<F: Float> FittedKMeans<F> {
    /// Return the cluster centers, shape `(n_clusters, n_features)`.
    #[must_use]
    pub fn cluster_centers(&self) -> &Array2<F> {
        &self.cluster_centers_
    }

    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }

    /// Return the inertia (sum of squared distances to nearest centroid).
    #[must_use]
    pub fn inertia(&self) -> F {
        self.inertia_
    }

    /// Return the number of iterations in the best run.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// k-Means++ initialization: pick `k` initial centroids.
fn kmeans_plus_plus<F: Float>(x: &Array2<F>, k: usize, rng: &mut StdRng) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut centers = Array2::zeros((k, n_features));

    // Pick first center uniformly at random.
    let first_idx = rng.random_range(0..n_samples);
    centers.row_mut(0).assign(&x.row(first_idx));

    // For each subsequent center, pick proportional to D(x)^2.
    let mut min_dists = Array1::from_elem(n_samples, F::max_value());

    for c in 1..k {
        // Update min distances with the most recently added center.
        let prev_center = centers.row(c - 1);
        for i in 0..n_samples {
            let d = squared_euclidean(
                x.row(i).as_slice().unwrap_or(&[]),
                prev_center.as_slice().unwrap_or(&[]),
            );
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }

        // Compute cumulative distribution.
        let total: F = min_dists.iter().fold(F::zero(), |acc, &d| acc + d);
        if total == F::zero() {
            // All points are identical to existing centers; just pick randomly.
            let idx = rng.random_range(0..n_samples);
            centers.row_mut(c).assign(&x.row(idx));
            continue;
        }

        let threshold: F = F::from(rng.random::<f64>()).unwrap_or(F::zero()) * total;
        let mut cumsum = F::zero();
        let mut chosen = n_samples - 1;
        for i in 0..n_samples {
            cumsum = cumsum + min_dists[i];
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centers.row_mut(c).assign(&x.row(chosen));
    }

    centers
}

/// Assign each sample to its nearest centroid (parallelized with Rayon).
///
/// Returns `(labels, inertia)`.
fn assign_clusters<F: Float + Send + Sync>(
    x: &Array2<F>,
    centers: &Array2<F>,
) -> (Array1<usize>, F) {
    let n_samples = x.nrows();
    let k = centers.nrows();

    let results: Vec<(usize, F)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = x.row(i);
            let row_slice = row.as_slice().unwrap_or(&[]);
            let mut best_label = 0;
            let mut best_dist = F::max_value();
            for c in 0..k {
                let center_row = centers.row(c);
                let center_slice = center_row.as_slice().unwrap_or(&[]);
                let d = squared_euclidean(row_slice, center_slice);
                if d < best_dist {
                    best_dist = d;
                    best_label = c;
                }
            }
            (best_label, best_dist)
        })
        .collect();

    let mut labels = Array1::zeros(n_samples);
    let mut inertia = F::zero();
    for (i, (label, dist)) in results.into_iter().enumerate() {
        labels[i] = label;
        inertia = inertia + dist;
    }

    (labels, inertia)
}

/// Recompute centroids as the mean of assigned samples.
///
/// Returns the new centroids and the maximum centroid movement.
fn recompute_centroids<F: Float>(
    x: &Array2<F>,
    labels: &Array1<usize>,
    k: usize,
    n_features: usize,
    old_centers: &Array2<F>,
) -> (Array2<F>, F) {
    let mut new_centers = Array2::zeros((k, n_features));
    let mut counts = vec![F::zero(); k];

    for (i, &label) in labels.iter().enumerate() {
        counts[label] = counts[label] + F::one();
        for j in 0..n_features {
            new_centers[[label, j]] = new_centers[[label, j]] + x[[i, j]];
        }
    }

    // Divide by count; if a cluster is empty, keep the old center.
    for c in 0..k {
        if counts[c] > F::zero() {
            for j in 0..n_features {
                new_centers[[c, j]] = new_centers[[c, j]] / counts[c];
            }
        } else {
            new_centers.row_mut(c).assign(&old_centers.row(c));
        }
    }

    // Compute maximum centroid movement.
    let mut max_shift = F::zero();
    for c in 0..k {
        let shift = squared_euclidean(
            new_centers.row(c).as_slice().unwrap_or(&[]),
            old_centers.row(c).as_slice().unwrap_or(&[]),
        );
        if shift > max_shift {
            max_shift = shift;
        }
    }

    (new_centers, max_shift.sqrt())
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KMeans<F> {
    type Fitted = FittedKMeans<F>;
    type Error = FerroError;

    /// Fit the k-Means model to the data.
    ///
    /// Runs Lloyd's algorithm `n_init` times with k-Means++ initialization,
    /// keeping the result with the lowest inertia.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_clusters` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if the number of samples
    /// is less than `n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedKMeans<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate parameters.
        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: 0,
                context: "KMeans requires at least n_clusters samples".into(),
            });
        }

        if n_samples < self.n_clusters {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: n_samples,
                context: "KMeans requires at least n_clusters samples".into(),
            });
        }

        if self.n_init == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_init".into(),
                reason: "must be at least 1".into(),
            });
        }

        let base_seed = self.random_state.unwrap_or(0);
        let mut best_result: Option<FittedKMeans<F>> = None;

        for run in 0..self.n_init {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(run as u64));

            // k-Means++ initialization.
            let mut centers = kmeans_plus_plus(x, self.n_clusters, &mut rng);

            let mut n_iter = 0;
            let mut labels = Array1::zeros(n_samples);
            let mut inertia = F::max_value();

            for iter in 0..self.max_iter {
                // Assign step (parallelized).
                let (new_labels, new_inertia) = assign_clusters(x, &centers);
                labels = new_labels;
                inertia = new_inertia;

                // Recompute centroids.
                let (new_centers, max_shift) =
                    recompute_centroids(x, &labels, self.n_clusters, n_features, &centers);
                centers = new_centers;
                n_iter = iter + 1;

                // Check convergence.
                if max_shift < self.tol {
                    break;
                }
            }

            let candidate = FittedKMeans {
                cluster_centers_: centers,
                labels_: labels,
                inertia_: inertia,
                n_iter_: n_iter,
            };

            match &best_result {
                None => best_result = Some(candidate),
                Some(best) => {
                    if candidate.inertia_ < best.inertia_ {
                        best_result = Some(candidate);
                    }
                }
            }
        }

        // SAFETY: n_init >= 1 is validated above, so best_result is always Some.
        best_result.ok_or_else(|| FerroError::InvalidParameter {
            name: "n_init".into(),
            reason: "internal error: no runs completed".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKMeans<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Assign each sample to the nearest cluster centroid.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let expected_features = self.cluster_centers_.ncols();

        if n_features != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![n_features],
                context: "number of features must match fitted KMeans model".into(),
            });
        }

        let (labels, _inertia) = assign_clusters(x, &self.cluster_centers_);
        Ok(labels)
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedKMeans<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Compute the distance from each sample to each cluster centroid.
    ///
    /// Returns a matrix of shape `(n_samples, n_clusters)` where element
    /// `[i, j]` is the Euclidean distance from sample `i` to centroid `j`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        let expected_features = self.cluster_centers_.ncols();

        if n_features != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![n_features],
                context: "number of features must match fitted KMeans model".into(),
            });
        }

        let n_samples = x.nrows();
        let k = self.cluster_centers_.nrows();

        let distances: Vec<F> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let row = x.row(i);
                let row_slice = row.as_slice().unwrap_or(&[]);
                (0..k)
                    .map(|c| {
                        let center_slice = self.cluster_centers_.row(c);
                        let cs = center_slice.as_slice().unwrap_or(&[]);
                        squared_euclidean(row_slice, cs).sqrt()
                    })
                    .collect::<Vec<F>>()
            })
            .collect();

        Array2::from_shape_vec((n_samples, k), distances).map_err(|_| {
            FerroError::NumericalInstability {
                message: "failed to construct distance matrix".into(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
        let model = KMeans::<f64>::new(3).with_random_state(42).with_n_init(5);
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
    fn test_convergence() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3)
            .with_random_state(42)
            .with_max_iter(1000)
            .with_n_init(1);
        let fitted = model.fit(&x, &()).unwrap();

        // Well-separated blobs should converge well before max_iter.
        assert!(fitted.n_iter() < 100);
    }

    #[test]
    fn test_n_init_picks_best() {
        let x = make_blobs();

        // With n_init=1, we might get a suboptimal result.
        let model_1 = KMeans::<f64>::new(3).with_random_state(42).with_n_init(1);
        let fitted_1 = model_1.fit(&x, &()).unwrap();

        // With n_init=10, we should get at least as good (usually better).
        let model_10 = KMeans::<f64>::new(3).with_random_state(42).with_n_init(10);
        let fitted_10 = model_10.fit(&x, &()).unwrap();

        // The n_init=10 run should have inertia <= n_init=1 run.
        assert!(fitted_10.inertia() <= fitted_1.inertia() + 1e-10);
    }

    #[test]
    fn test_kmeans_pp_initialization_deterministic() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(123).with_n_init(1);

        let fitted1 = model.fit(&x, &()).unwrap();
        let fitted2 = model.fit(&x, &()).unwrap();

        // Same seed should produce same result.
        assert_eq!(fitted1.labels(), fitted2.labels());
        assert_relative_eq!(fitted1.inertia(), fitted2.inertia(), epsilon = 1e-12);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(99);

        let fitted1 = model.fit(&x, &()).unwrap();
        let fitted2 = model.fit(&x, &()).unwrap();

        assert_eq!(fitted1.labels(), fitted2.labels());
    }

    #[test]
    fn test_predict_on_new_data() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // Predict on new points near each cluster.
        let new_x =
            Array2::from_shape_vec((3, 2), vec![0.05, 0.05, 10.05, 10.05, 0.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        // New points near cluster 0 center.
        let label_near_origin = new_labels[0];
        // Should match the training label of the origin cluster.
        assert_eq!(label_near_origin, fitted.labels()[0]);

        let label_near_10_10 = new_labels[1];
        assert_eq!(label_near_10_10, fitted.labels()[3]);

        let label_near_0_10 = new_labels[2];
        assert_eq!(label_near_0_10, fitted.labels()[6]);
    }

    #[test]
    fn test_transform_distances() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0])
            .unwrap();

        let model = KMeans::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let dists = fitted.transform(&x).unwrap();

        // Shape should be (n_samples, n_clusters).
        assert_eq!(dists.dim(), (4, 2));

        // Distance to own centroid should be small, distance to other should be large.
        for i in 0..4 {
            let own_cluster = fitted.labels()[i];
            let other_cluster = 1 - own_cluster;
            assert!(dists[[i, own_cluster]] < dists[[i, other_cluster]]);
        }
    }

    #[test]
    fn test_transform_shape() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let dists = fitted.transform(&x).unwrap();

        assert_eq!(dists.dim(), (9, 3));
    }

    #[test]
    fn test_cluster_centers_shape() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.cluster_centers().dim(), (3, 2));
    }

    #[test]
    fn test_inertia_non_negative() {
        let x = make_blobs();
        let model = KMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        assert!(fitted.inertia() >= 0.0);
    }

    #[test]
    fn test_k_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();

        let model = KMeans::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // Each point should be its own cluster, inertia should be ~0.
        assert_relative_eq!(fitted.inertia(), 0.0, epsilon = 1e-10);

        // All labels should be distinct.
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
    }

    #[test]
    fn test_single_cluster() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let model = KMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // All points should be in cluster 0.
        for &label in fitted.labels().iter() {
            assert_eq!(label, 0);
        }

        // Center should be the mean.
        let center = fitted.cluster_centers().row(0);
        assert_relative_eq!(center[0], 2.5, epsilon = 1e-10);
        assert_relative_eq!(center[1], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = KMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 1);
        assert_eq!(fitted.labels()[0], 0);
        assert_relative_eq!(fitted.inertia(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_k_greater_than_n_samples() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let model = KMeans::<f64>::new(5);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_clusters() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = KMeans::<f64>::new(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = KMeans::<f64>::new(3);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let model = KMeans::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&x_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        let model = KMeans::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.transform(&x_bad);
        assert!(result.is_err());
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

        let model = KMeans::<f32>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_two_clusters_on_line() {
        // Points on a line: cluster at x=0 and x=100.
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, -0.1, 100.0, 100.1, 99.9]).unwrap();

        let model = KMeans::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // First three should be same cluster.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        // Last three should be same cluster.
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        // Different clusters.
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_identical_points() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let model = KMeans::<f64>::new(1).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_relative_eq!(fitted.inertia(), 0.0, epsilon = 1e-10);
    }
}
