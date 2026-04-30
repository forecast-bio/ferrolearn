//! Affinity Propagation clustering.
//!
//! This module provides [`AffinityPropagation`], a clustering algorithm that
//! identifies exemplars (representative data points) by passing "responsibility"
//! and "availability" messages between all pairs of data points.
//!
//! # Key Advantage
//!
//! Unlike K-Means, Affinity Propagation does **not** require the number of
//! clusters to be specified in advance. The algorithm automatically determines
//! the number of clusters based on the data and the preference parameter.
//!
//! # Algorithm
//!
//! 1. Compute a similarity matrix `S` using negative squared Euclidean distance.
//! 2. Set diagonal preferences (self-similarity). If `preference` is `None`,
//!    the median of the off-diagonal similarities is used.
//! 3. Initialize responsibility (`R`) and availability (`A`) matrices to zero.
//! 4. Iteratively update `R` and `A` with damping until convergence:
//!    - **Responsibility** `R[i,k]`: how well-suited point `k` is to serve as
//!      an exemplar for point `i`, relative to other candidates.
//!    - **Availability** `A[i,k]`: how appropriate it would be for point `i`
//!      to choose point `k` as its exemplar, given other points' preferences.
//! 5. Exemplars are points where `R[k,k] + A[k,k] > 0`.
//! 6. Each non-exemplar point is assigned to its nearest exemplar.
//!
//! # Notes
//!
//! Affinity Propagation does **not** implement [`Predict`](ferrolearn_core::Predict)
//! — it only labels the training data. Use the fitted labels directly.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::AffinityPropagation;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//!
//! let model = AffinityPropagation::<f64>::new();
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.labels();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Affinity Propagation clustering configuration (unfitted).
///
/// Holds hyperparameters for the Affinity Propagation algorithm. Call
/// [`Fit::fit`] to run the algorithm and produce a [`FittedAffinityPropagation`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AffinityPropagation<F> {
    /// Damping factor in `[0.5, 1.0)`. Controls the rate at which messages
    /// are updated. Higher values provide more stability but slower convergence.
    damping: F,
    /// Maximum number of message-passing iterations.
    max_iter: usize,
    /// Number of iterations with no change in exemplars before declaring
    /// convergence.
    convergence_iter: usize,
    /// Self-similarity preference. If `None`, the median of the off-diagonal
    /// similarity matrix is used, which tends to produce a moderate number of
    /// clusters.
    preference: Option<F>,
}

impl<F: Float> AffinityPropagation<F> {
    /// Create a new `AffinityPropagation` with default parameters.
    ///
    /// Defaults: `damping = 0.5`, `max_iter = 200`, `convergence_iter = 15`,
    /// `preference = None` (median of similarity matrix).
    #[must_use]
    pub fn new() -> Self {
        Self {
            damping: F::from(0.5).unwrap(),
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
        }
    }

    /// Set the damping factor. Must be in `[0.5, 1.0)`.
    #[must_use]
    pub fn with_damping(mut self, damping: F) -> Self {
        self.damping = damping;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of stable iterations required for convergence.
    #[must_use]
    pub fn with_convergence_iter(mut self, convergence_iter: usize) -> Self {
        self.convergence_iter = convergence_iter;
        self
    }

    /// Set the preference (self-similarity) value.
    ///
    /// Lower values produce fewer clusters; higher values produce more.
    /// If not set, the median of the off-diagonal similarity matrix is used.
    #[must_use]
    pub fn with_preference(mut self, preference: F) -> Self {
        self.preference = Some(preference);
        self
    }
}

impl<F: Float> Default for AffinityPropagation<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Affinity Propagation model.
///
/// Stores the identified exemplars, cluster centers, and labels from the
/// training run.
///
/// Affinity Propagation does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedAffinityPropagation<F> {
    /// Cluster center coordinates (exemplar points), shape `(n_clusters, n_features)`.
    cluster_centers_: Array2<F>,
    /// Cluster label for each training sample.
    labels_: Array1<isize>,
    /// Indices of exemplar points in the training data.
    exemplar_indices_: Vec<usize>,
    /// Number of iterations run.
    n_iter_: usize,
}

impl<F: Float> FittedAffinityPropagation<F> {
    /// Return the cluster centers (exemplar points), shape `(n_clusters, n_features)`.
    #[must_use]
    pub fn cluster_centers(&self) -> &Array2<F> {
        &self.cluster_centers_
    }

    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the indices of exemplar points in the training data.
    #[must_use]
    pub fn exemplar_indices(&self) -> &[usize] {
        &self.exemplar_indices_
    }

    /// Return the number of iterations run.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Return the number of clusters found.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.exemplar_indices_.len()
    }
}

/// Compute the squared Euclidean distance between two slices.
fn squared_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Compute the median of a sorted slice of floats.
fn median_of_sorted<F: Float>(sorted: &[F]) -> F {
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / F::from(2.0).unwrap()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for AffinityPropagation<F> {
    type Fitted = FittedAffinityPropagation<F>;
    type Error = FerroError;

    /// Fit the Affinity Propagation model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `damping` is outside `[0.5, 1.0)`,
    /// `max_iter` is zero, or `convergence_iter` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if the data has no samples.
    /// Returns [`FerroError::ConvergenceFailure`] if no exemplars are found.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedAffinityPropagation<F>, FerroError> {
        let half = F::from(0.5).unwrap();
        let one = F::one();

        // Validate parameters.
        if self.damping < half || self.damping >= one {
            return Err(FerroError::InvalidParameter {
                name: "damping".into(),
                reason: "must be in [0.5, 1.0)".into(),
            });
        }

        if self.max_iter == 0 {
            return Err(FerroError::InvalidParameter {
                name: "max_iter".into(),
                reason: "must be at least 1".into(),
            });
        }

        if self.convergence_iter == 0 {
            return Err(FerroError::InvalidParameter {
                name: "convergence_iter".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n = x.nrows();
        let n_features = x.ncols();

        if n == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "AffinityPropagation requires at least 1 sample".into(),
            });
        }

        // Handle single-sample case directly.
        if n == 1 {
            return Ok(FittedAffinityPropagation {
                cluster_centers_: x.clone(),
                labels_: Array1::from_elem(1, 0isize),
                exemplar_indices_: vec![0],
                n_iter_: 0,
            });
        }

        // Step 1: Compute similarity matrix S = -||x_i - x_k||^2.
        let mut s = Array2::zeros((n, n));
        for i in 0..n {
            let row_i = x.row(i);
            let slice_i = row_i.as_slice().unwrap_or(&[]);
            for k in (i + 1)..n {
                let row_k = x.row(k);
                let slice_k = row_k.as_slice().unwrap_or(&[]);
                let neg_dist = -squared_euclidean(slice_i, slice_k);
                s[[i, k]] = neg_dist;
                s[[k, i]] = neg_dist;
            }
        }

        // Step 2: Set diagonal preferences.
        let pref = if let Some(p) = self.preference {
            p
        } else {
            // Compute median of off-diagonal similarities.
            let mut off_diag: Vec<F> = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for k in (i + 1)..n {
                    off_diag.push(s[[i, k]]);
                }
            }
            off_diag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            median_of_sorted(&off_diag)
        };

        for i in 0..n {
            s[[i, i]] = pref;
        }

        // Step 3: Initialize R and A to zero.
        let mut r = Array2::zeros((n, n));
        let mut a = Array2::zeros((n, n));

        let damping = self.damping;
        let one_minus_damping = one - damping;

        // Track convergence: count consecutive iterations with stable exemplars.
        let mut prev_exemplars: Vec<usize> = Vec::new();
        let mut stable_count: usize = 0;
        let mut n_iter = 0;

        // Step 4: Iterate.
        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update responsibilities.
            for i in 0..n {
                for k in 0..n {
                    // R_new[i,k] = S[i,k] - max_{k' != k}(A[i,k'] + S[i,k'])
                    let mut max_val = F::neg_infinity();
                    for kp in 0..n {
                        if kp != k {
                            let val = a[[i, kp]] + s[[i, kp]];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    let r_new = s[[i, k]] - max_val;
                    // Apply damping.
                    r[[i, k]] = damping * r[[i, k]] + one_minus_damping * r_new;
                }
            }

            // Update availabilities.
            for i in 0..n {
                for k in 0..n {
                    if i == k {
                        // A[k,k] = sum_{i' != k} max(0, R[i',k])
                        let mut sum = F::zero();
                        for ip in 0..n {
                            if ip != k {
                                let rval = r[[ip, k]];
                                if rval > F::zero() {
                                    sum = sum + rval;
                                }
                            }
                        }
                        let a_new = sum;
                        a[[k, k]] = damping * a[[k, k]] + one_minus_damping * a_new;
                    } else {
                        // A[i,k] = min(0, R[k,k] + sum_{i' != i,k} max(0, R[i',k]))
                        let mut sum = F::zero();
                        for ip in 0..n {
                            if ip != i && ip != k {
                                let rval = r[[ip, k]];
                                if rval > F::zero() {
                                    sum = sum + rval;
                                }
                            }
                        }
                        let a_new = r[[k, k]] + sum;
                        let a_clamped = if a_new < F::zero() { a_new } else { F::zero() };
                        a[[i, k]] = damping * a[[i, k]] + one_minus_damping * a_clamped;
                    }
                }
            }

            // Check convergence: extract current exemplars (points where
            // R[k,k] + A[k,k] > 0).
            let mut current_exemplars: Vec<usize> = Vec::new();
            for k in 0..n {
                if r[[k, k]] + a[[k, k]] > F::zero() {
                    current_exemplars.push(k);
                }
            }

            // Only count as stable if we have at least one exemplar and the
            // set hasn't changed.
            if !current_exemplars.is_empty() && current_exemplars == prev_exemplars {
                stable_count += 1;
                if stable_count >= self.convergence_iter {
                    break;
                }
            } else {
                stable_count = 0;
                prev_exemplars = current_exemplars;
            }
        }

        // Step 5: Identify exemplars. First try strict positive criterion,
        // then fall back to non-negative, then to argmax-based selection.
        let mut exemplar_indices: Vec<usize> = Vec::new();
        for k in 0..n {
            if r[[k, k]] + a[[k, k]] > F::zero() {
                exemplar_indices.push(k);
            }
        }

        // Fall back: if no strictly positive exemplars, use the argmax of
        // (R[i,:] + A[i,:]) for each row to determine exemplars.
        if exemplar_indices.is_empty() {
            let mut exemplar_set: Vec<usize> = Vec::new();
            for i in 0..n {
                let mut best_k = 0;
                let mut best_val = F::neg_infinity();
                for k in 0..n {
                    let val = r[[i, k]] + a[[i, k]];
                    if val > best_val {
                        best_val = val;
                        best_k = k;
                    }
                }
                if !exemplar_set.contains(&best_k) {
                    exemplar_set.push(best_k);
                }
            }
            exemplar_set.sort_unstable();
            exemplar_indices = exemplar_set;
        }

        // If still no exemplars, return a convergence failure.
        if exemplar_indices.is_empty() {
            return Err(FerroError::ConvergenceFailure {
                iterations: n_iter,
                message: "no exemplars identified; try adjusting the preference or damping".into(),
            });
        }

        // Step 6: Assign each point to the nearest exemplar using the
        // original negative squared Euclidean distance (not the modified
        // similarity matrix which has preference on the diagonal).
        let mut labels = Array1::from_elem(n, 0isize);
        for i in 0..n {
            // If this point is itself an exemplar, assign it to its own cluster.
            if let Some(pos) = exemplar_indices.iter().position(|&ex| ex == i) {
                labels[i] = pos as isize;
                continue;
            }
            let row_i = x.row(i);
            let slice_i = row_i.as_slice().unwrap_or(&[]);
            let mut best_cluster = 0isize;
            let mut best_dist = F::max_value();
            for (cluster_idx, &ex) in exemplar_indices.iter().enumerate() {
                let row_ex = x.row(ex);
                let slice_ex = row_ex.as_slice().unwrap_or(&[]);
                let d = squared_euclidean(slice_i, slice_ex);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = cluster_idx as isize;
                }
            }
            labels[i] = best_cluster;
        }

        // Build cluster centers from exemplar rows.
        let n_clusters = exemplar_indices.len();
        let mut cluster_centers = Array2::zeros((n_clusters, n_features));
        for (ci, &ex) in exemplar_indices.iter().enumerate() {
            cluster_centers.row_mut(ci).assign(&x.row(ex));
        }

        Ok(FittedAffinityPropagation {
            cluster_centers_: cluster_centers,
            labels_: labels,
            exemplar_indices_: exemplar_indices,
            n_iter_: n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> AffinityPropagation<F> {
    /// Fit on `x` and return the cluster labels for those samples in one
    /// call. Equivalent to sklearn `ClusterMixin.fit_predict`. Samples
    /// with no exemplar (degenerate runs) are labeled as `-1`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Fit::fit`].
    pub fn fit_predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(fitted.labels().clone())
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
    fn test_two_clusters() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new();
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

        // Should find 2 clusters.
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_exemplar_indices_are_valid() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        for &idx in fitted.exemplar_indices() {
            assert!(idx < 8);
        }
        // Exemplar indices should match number of clusters.
        assert_eq!(fitted.exemplar_indices().len(), fitted.n_clusters());
    }

    #[test]
    fn test_cluster_centers_match_exemplars() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        let centers = fitted.cluster_centers();
        assert_eq!(centers.nrows(), fitted.n_clusters());
        assert_eq!(centers.ncols(), 2);

        // Each center should match the corresponding exemplar row.
        for (ci, &ex) in fitted.exemplar_indices().iter().enumerate() {
            for j in 0..2 {
                assert!(
                    (centers[[ci, j]] - x[[ex, j]]).abs() < 1e-10,
                    "center mismatch at cluster {ci}, feature {j}"
                );
            }
        }
    }

    #[test]
    fn test_damping_effect() {
        let x = make_two_clusters();

        // Higher damping should still converge for well-separated data.
        let model = AffinityPropagation::<f64>::new().with_damping(0.9);
        let fitted = model.fit(&x, &()).unwrap();

        assert!(fitted.n_clusters() >= 1);
        assert_eq!(fitted.labels().len(), 8);
    }

    #[test]
    fn test_preference_controls_n_clusters() {
        let x = make_two_clusters();

        // Very low preference: should produce fewer clusters.
        let model_low = AffinityPropagation::<f64>::new().with_preference(-200.0);
        let fitted_low = model_low.fit(&x, &()).unwrap();

        // Very high preference: should produce more clusters.
        let model_high = AffinityPropagation::<f64>::new().with_preference(-0.1);
        let fitted_high = model_high.fit(&x, &()).unwrap();

        // Higher preference => more clusters (or at least as many).
        assert!(fitted_high.n_clusters() >= fitted_low.n_clusters());
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 1);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.n_clusters(), 1);
        assert_eq!(fitted.exemplar_indices(), &[0]);
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = AffinityPropagation::<f64>::new();
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_damping_too_low() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new().with_damping(0.3);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_damping_too_high() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new().with_damping(1.0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_iter() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new().with_max_iter(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_convergence_iter() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new().with_convergence_iter(0);
        let result = model.fit(&x, &());
        assert!(result.is_err());
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

        let model = AffinityPropagation::<f32>::new();
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);
        assert!(fitted.n_clusters() >= 1);
    }

    #[test]
    fn test_default_impl() {
        let model = AffinityPropagation::<f64>::default();
        let x = make_two_clusters();
        let fitted = model.fit(&x, &()).unwrap();
        assert!(fitted.n_clusters() >= 1);
    }

    #[test]
    fn test_three_clusters() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 20.0, 0.0, 20.1,
                0.0, 20.0, 0.1,
            ],
        )
        .unwrap();

        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // Points in same cluster should have same label.
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
    fn test_n_iter_positive() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_labels_in_range() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        let n_clusters = fitted.n_clusters() as isize;
        for &label in fitted.labels() {
            assert!(label >= 0);
            assert!(label < n_clusters);
        }
    }

    #[test]
    fn test_identical_points() {
        // All identical points: should form 1 cluster with a single exemplar.
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        // All should be in the same cluster.
        let labels = fitted.labels();
        for &label in labels {
            assert_eq!(label, labels[0]);
        }
    }

    #[test]
    fn test_two_samples() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        let model = AffinityPropagation::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 2);
        assert!(fitted.n_clusters() >= 1);
    }

    #[test]
    fn test_with_explicit_preference() {
        let x = make_two_clusters();
        let model = AffinityPropagation::<f64>::new().with_preference(-50.0);
        let fitted = model.fit(&x, &()).unwrap();

        assert!(fitted.n_clusters() >= 1);
        assert_eq!(fitted.labels().len(), 8);
    }
}
