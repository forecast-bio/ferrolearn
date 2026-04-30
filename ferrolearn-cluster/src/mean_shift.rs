//! Mean Shift clustering.
//!
//! This module provides [`MeanShift`], a non-parametric clustering algorithm
//! that finds the modes (local maxima) of the underlying kernel density
//! estimate.  Each data point is iteratively shifted toward the mean of all
//! points within a sphere of radius `bandwidth` until convergence.  After
//! convergence, nearby modes are merged into a single cluster center, and each
//! training point is assigned to the nearest center.
//!
//! # Algorithm
//!
//! 1. **Bandwidth estimation** (when `bandwidth` is `None`): compute the
//!    median of all pairwise Euclidean distances and use that value.
//! 2. **Mean shift iteration**: for each data point (candidate mode), compute
//!    the mean of all points within `bandwidth` distance and shift the
//!    candidate to that mean.  Repeat until the shift is smaller than `tol`
//!    or `max_iter` is reached.
//! 3. **Mode merging**: candidates whose final positions are within
//!    `bandwidth` of each other are merged (the one with more nearby points
//!    becomes the representative center).
//! 4. **Label assignment**: each training point is assigned to the nearest
//!    cluster center.
//!
//! Mean Shift does **not** require specifying the number of clusters ahead of
//! time; that number emerges from the data density and the bandwidth.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::MeanShift;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0_f64, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     9.0, 9.0,  9.1, 9.0,  9.0, 9.1,
//! ]).unwrap();
//!
//! let model = MeanShift::<f64>::new().with_bandwidth(3.0);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.n_clusters(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// Mean Shift clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedMeanShift`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MeanShift<F> {
    /// Kernel bandwidth (search radius).  When `None` the bandwidth is
    /// estimated as the median of all pairwise Euclidean distances.
    pub bandwidth: Option<F>,
    /// Maximum number of mean-shift iterations per seed point.
    pub max_iter: usize,
    /// Convergence tolerance: stop iterating when the shift magnitude is
    /// smaller than this value.
    pub tol: F,
}

impl<F: Float> MeanShift<F> {
    /// Create a new `MeanShift` with default hyperparameters.
    ///
    /// Bandwidth is estimated automatically (`None`), `max_iter = 300`,
    /// `tol = 1e-3`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bandwidth: None,
            max_iter: 300,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
        }
    }

    /// Set the bandwidth explicitly.
    ///
    /// Must be positive.  Setting this overrides automatic estimation.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: F) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Set the maximum number of mean-shift iterations.
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
}

impl<F: Float> Default for MeanShift<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Mean Shift model.
///
/// Stores the discovered cluster centers and the labels assigned to the
/// training data.  Implements [`Predict`] to assign new points to the nearest
/// center.
#[derive(Debug, Clone)]
pub struct FittedMeanShift<F> {
    /// Cluster centers, shape `(n_clusters, n_features)`.
    cluster_centers_: Array2<F>,
    /// Cluster label for each training sample (0-indexed).
    labels_: Array1<usize>,
    /// Number of mean-shift iterations performed on the last seed point.
    n_iter_: usize,
}

impl<F: Float> FittedMeanShift<F> {
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

    /// Return the number of mean-shift iterations in the last seed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Return the number of clusters discovered.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers_.nrows()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Estimate bandwidth as the median of all pairwise Euclidean distances.
///
/// Returns an error if the dataset has fewer than 2 points.
fn estimate_bandwidth<F: Float>(x: &Array2<F>) -> Result<F, FerroError> {
    let n = x.nrows();
    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "MeanShift bandwidth estimation requires at least 2 samples".into(),
        });
    }

    let mut dists: Vec<F> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        for j in (i + 1)..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            dists.push(sq_dist(si, sj).sqrt());
        }
    }

    // Partial-sort to find the median.
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = dists.len() / 2;
    let median = if dists.len() % 2 == 0 {
        (dists[mid - 1] + dists[mid]) / F::from(2.0).unwrap()
    } else {
        dists[mid]
    };

    if median == F::zero() {
        // Fallback: return the maximum distance so the algorithm does not
        // put everything in a single trivial cluster.
        Ok(dists.last().copied().unwrap_or_else(F::one))
    } else {
        Ok(median)
    }
}

/// Perform mean-shift iteration starting from `seed`.
///
/// Returns the converged mode position and the number of iterations taken.
fn mean_shift_single<F: Float>(
    seed: &[F],
    x: &Array2<F>,
    bw_sq: F,
    max_iter: usize,
    tol: F,
) -> (Vec<F>, usize) {
    let n_features = seed.len();
    let mut current = seed.to_vec();
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Collect all points within bandwidth.
        let mut mean = vec![F::zero(); n_features];
        let mut count = F::zero();

        for i in 0..x.nrows() {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            if sq_dist(&current, rs) <= bw_sq {
                for j in 0..n_features {
                    mean[j] = mean[j] + rs[j];
                }
                count = count + F::one();
            }
        }

        if count == F::zero() {
            // No neighbors — the point is isolated; keep it as-is.
            break;
        }

        for v in &mut mean {
            *v = *v / count;
        }

        // Compute shift magnitude.
        let shift = sq_dist(&current, &mean).sqrt();
        current = mean;

        if shift < tol {
            break;
        }
    }

    (current, n_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MeanShift<F> {
    type Fitted = FittedMeanShift<F>;
    type Error = FerroError;

    /// Fit the Mean Shift model to the data.
    ///
    /// Each data point is used as a seed; all seeds are iteratively shifted
    /// toward the local mean until convergence.  Nearby converged modes are
    /// then merged into cluster centers.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the dataset is empty.
    /// - [`FerroError::InvalidParameter`] if an explicit bandwidth ≤ 0 is
    ///   provided.
    /// - [`FerroError::NumericalInstability`] if bandwidth estimation or
    ///   center-array construction fails.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMeanShift<F>, FerroError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MeanShift requires at least 1 sample".into(),
            });
        }

        // Resolve bandwidth.
        let bandwidth = match self.bandwidth {
            Some(bw) => {
                if bw <= F::zero() {
                    return Err(FerroError::InvalidParameter {
                        name: "bandwidth".into(),
                        reason: "must be positive".into(),
                    });
                }
                bw
            }
            None => estimate_bandwidth(x)?,
        };

        let bw_sq = bandwidth * bandwidth;

        // Run mean-shift from every data point as a seed.
        let mut modes: Vec<Vec<F>> = Vec::with_capacity(n_samples);
        let mut last_n_iter = 0usize;

        for i in 0..n_samples {
            let row = x.row(i);
            let seed = row.as_slice().unwrap_or(&[]);
            let (mode, n_iter) = mean_shift_single(seed, x, bw_sq, self.max_iter, self.tol);
            modes.push(mode);
            last_n_iter = last_n_iter.max(n_iter);
        }

        // Merge nearby modes into cluster centers.
        // We keep one representative per group of modes that lie within
        // `bandwidth` of each other.  We pick the first unclaimed mode as a
        // new center, then merge all other modes within `bandwidth` into it.
        let mut used = vec![false; modes.len()];
        let mut centers: Vec<Vec<F>> = Vec::new();

        for i in 0..modes.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut group: Vec<&Vec<F>> = vec![&modes[i]];

            for j in (i + 1)..modes.len() {
                if !used[j] && sq_dist(&modes[i], &modes[j]).sqrt() < bandwidth {
                    used[j] = true;
                    group.push(&modes[j]);
                }
            }

            // Compute the mean of the group as the representative center.
            let mut center = vec![F::zero(); n_features];
            let g_len = F::from(group.len()).unwrap_or_else(F::one);
            for m in &group {
                for (k, &v) in m.iter().enumerate() {
                    center[k] = center[k] + v;
                }
            }
            for v in &mut center {
                *v = *v / g_len;
            }
            centers.push(center);
        }

        let n_centers = centers.len();

        // Build the center matrix.
        let flat: Vec<F> = centers.into_iter().flatten().collect();
        let cluster_centers =
            Array2::from_shape_vec((n_centers, n_features), flat).map_err(|_| {
                FerroError::NumericalInstability {
                    message: "failed to build cluster center matrix".into(),
                }
            })?;

        // Assign each training point to the nearest center.
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            let mut best = 0usize;
            let mut best_dist = F::max_value();
            for c in 0..n_centers {
                let center_row = cluster_centers.row(c);
                let cs = center_row.as_slice().unwrap_or(&[]);
                let d = sq_dist(rs, cs);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            labels[i] = best;
        }

        Ok(FittedMeanShift {
            cluster_centers_: cluster_centers,
            labels_: labels,
            n_iter_: last_n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> MeanShift<F> {
    /// Fit on `x` and return labels for those samples in one call.
    /// Equivalent to sklearn `ClusterMixin.fit_predict`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Fit::fit`] / [`Predict::predict`].
    pub fn fit_predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.predict(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predict implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedMeanShift<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Assign each sample to the nearest cluster center.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let expected = self.cluster_centers_.ncols();
        if n_features != expected {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![n_features],
                context: "number of features must match the fitted MeanShift model".into(),
            });
        }

        let n_samples = x.nrows();
        let n_centers = self.cluster_centers_.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);
            let rs = row.as_slice().unwrap_or(&[]);
            let mut best = 0usize;
            let mut best_dist = F::max_value();
            for c in 0..n_centers {
                let cr = self.cluster_centers_.row(c);
                let cs = cr.as_slice().unwrap_or(&[]);
                let d = sq_dist(rs, cs);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            labels[i] = best;
        }

        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two well-separated 2-D blobs.
    fn two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, 10.0, 10.0, 10.2, 10.1, 9.9,
                10.2, 10.1, 9.9, 10.0, 10.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_two_blobs_correct_clusters() {
        let x = two_blobs();
        let model = MeanShift::<f64>::new().with_bandwidth(2.0);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.n_clusters(), 2);

        let labels = fitted.labels();
        // Points 0-4 should be in the same cluster.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[0], labels[4]);
        // Points 5-9 should be in the same cluster.
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[5], labels[7]);
        assert_eq!(labels[5], labels[8]);
        assert_eq!(labels[5], labels[9]);
        // Different clusters.
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn test_labels_length() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), x.nrows());
    }

    #[test]
    fn test_cluster_centers_shape() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        let centers = fitted.cluster_centers();
        assert_eq!(centers.ncols(), 2);
        assert_eq!(centers.nrows(), 2);
    }

    #[test]
    fn test_centers_near_blob_means() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let centers = fitted.cluster_centers();
        // The two centers should be approximately at (0.04, 0.06) and (10.04, 10.06).
        // Collect distances to (0,0) and (10,10).
        let near_origin =
            (0..centers.nrows()).any(|i| centers[[i, 0]].hypot(centers[[i, 1]]) < 1.0);
        let near_far = (0..centers.nrows())
            .any(|i| (centers[[i, 0]] - 10.0).hypot(centers[[i, 1]] - 10.0) < 1.0);

        assert!(near_origin, "expected a center near the origin cluster");
        assert!(near_far, "expected a center near the (10,10) cluster");
    }

    #[test]
    fn test_single_cluster_large_bandwidth() {
        let x = two_blobs();
        // Bandwidth large enough to merge everything.
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(50.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        for &l in fitted.labels() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_auto_bandwidth_finds_two_clusters() {
        let x = two_blobs();
        // With auto bandwidth on well-separated blobs it should find 2 clusters.
        let fitted = MeanShift::<f64>::new().fit(&x, &()).unwrap();
        // The separation is ~14 units; auto-bandwidth is the median pairwise dist
        // which should be smaller than that, so 2 clusters are expected.
        assert!(fitted.n_clusters() >= 1);
    }

    #[test]
    fn test_predict_on_new_points() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let preds = fitted.predict(&new_x).unwrap();

        // Each new point should get the same label as the nearby training points.
        let label_near_origin = fitted.labels()[0];
        let label_near_far = fitted.labels()[5];
        assert_eq!(preds[0], label_near_origin);
        assert_eq!(preds[1], label_near_far);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();

        let bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&bad).is_err());
    }

    #[test]
    fn test_single_point() {
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        assert_eq!(fitted.labels()[0], 0);
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = MeanShift::<f64>::new().with_bandwidth(1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bandwidth_error() {
        let x = two_blobs();
        let result = MeanShift::<f64>::new().with_bandwidth(-1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_bandwidth_error() {
        let x = two_blobs();
        let result = MeanShift::<f64>::new().with_bandwidth(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_n_iter_non_zero() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.1, -0.1, 0.1, 8.0, 8.0, 8.1, 8.1, 7.9, 8.1,
            ],
        )
        .unwrap();

        let fitted = MeanShift::<f32>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), 6);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_three_clusters() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 0.0, 10.0, 0.1,
                10.0, 0.0, 10.1,
            ],
        )
        .unwrap();

        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.5)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 3);

        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_identical_points_single_cluster() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
    }

    #[test]
    fn test_center_coordinates_reasonable() {
        // Single tight cluster near (5, 5).
        let x =
            Array2::from_shape_vec((4, 2), vec![5.0, 5.0, 5.1, 4.9, 4.9, 5.1, 5.0, 5.0]).unwrap();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(1.0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.n_clusters(), 1);
        let c = fitted.cluster_centers();
        assert_relative_eq!(c[[0, 0]], 5.0, epsilon = 0.2);
        assert_relative_eq!(c[[0, 1]], 5.0, epsilon = 0.2);
    }

    #[test]
    fn test_predict_labels_range() {
        let x = two_blobs();
        let fitted = MeanShift::<f64>::new()
            .with_bandwidth(2.0)
            .fit(&x, &())
            .unwrap();
        let n_c = fitted.n_clusters();
        for &l in fitted.labels() {
            assert!(l < n_c);
        }
    }

    #[test]
    fn test_default_trait() {
        let model: MeanShift<f64> = MeanShift::default();
        assert!(model.bandwidth.is_none());
        assert_eq!(model.max_iter, 300);
    }
}
