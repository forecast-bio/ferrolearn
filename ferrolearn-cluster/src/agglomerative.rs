//! Agglomerative (bottom-up) hierarchical clustering.
//!
//! This module provides [`AgglomerativeClustering`], a hierarchical clustering
//! algorithm that builds a dendrogram by successively merging the two closest
//! clusters.  The merge criterion is determined by the [`Linkage`] strategy.
//!
//! # Algorithm
//!
//! 1. Initialise each data point as its own singleton cluster.
//! 2. Build an `n × n` pairwise distance matrix.
//! 3. Repeat until `n_clusters` clusters remain:
//!    a. Find the pair of clusters `(i, j)` with the smallest inter-cluster
//!    distance according to the chosen linkage.
//!    b. Merge them into a new cluster; record the merge in `children_`.
//!    c. Update distances using the Lance–Williams recurrence.
//!
//! The overall complexity is **O(n³)** in time and **O(n²)** in space, which
//! is practical for datasets up to a few thousand samples.
//!
//! # Linkage strategies
//!
//! | [`Linkage`]  | Distance formula | Properties |
//! |--------------|------------------|------------|
//! | `Single`     | `min d(a, b)`    | Chaining effect; handles non-convex shapes |
//! | `Complete`   | `max d(a, b)`    | Compact clusters |
//! | `Average`    | mean of pairwise | Compromise |
//! | `Ward`       | increase in SSE  | Minimises within-cluster variance |
//!
//! # Note
//!
//! [`AgglomerativeClustering`] implements [`Fit`] only.  There is no
//! `predict` method (mirroring scikit-learn's design).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::AgglomerativeClustering;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     8.0, 8.0,  8.1, 8.0,  8.0, 8.1,
//! ]).unwrap();
//!
//! let model = AgglomerativeClustering::<f64>::new(2);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ─────────────────────────────────────────────────────────────────────────────
// Public enums & configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// The linkage criterion used to measure distances between clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Ward linkage: merge the pair that minimises the increase in
    /// within-cluster sum of squared errors.
    Ward,
    /// Complete linkage: the distance between two clusters is the
    /// *maximum* distance between any pair of their members.
    Complete,
    /// Average linkage (UPGMA): the distance is the mean of all pairwise
    /// distances between the two clusters.
    Average,
    /// Single linkage: the distance between two clusters is the *minimum*
    /// pairwise distance between their members.
    Single,
}

/// Agglomerative clustering configuration (unfitted).
///
/// Call [`Fit::fit`] to run the algorithm and obtain a
/// [`FittedAgglomerativeClustering`].
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering<F> {
    /// Target number of clusters.
    pub n_clusters: usize,
    /// Linkage strategy for computing inter-cluster distances.
    pub linkage: Linkage,
    /// Phantom to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> AgglomerativeClustering<F> {
    /// Create a new `AgglomerativeClustering` with the given number of clusters.
    ///
    /// Uses default `linkage = Ward`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the linkage criterion.
    #[must_use]
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Agglomerative Clustering model.
///
/// Stores per-sample cluster labels, the actual cluster count, and the
/// merge history (dendrogram).
///
/// There is intentionally **no** `predict` method: new data cannot be
/// assigned to clusters without re-running the full algorithm.
#[derive(Debug, Clone)]
pub struct FittedAgglomerativeClustering<F> {
    /// Cluster label for each training sample, shape `(n_samples,)`.
    /// Labels are in the range `0 .. n_clusters_`.
    pub labels_: Array1<usize>,
    /// The actual number of clusters formed.
    pub n_clusters_: usize,
    /// Merge history: each element `(i, j)` records that the clusters
    /// with internal IDs `i` and `j` were merged.  Length =
    /// `n_samples - n_clusters`.
    pub children_: Vec<(usize, usize)>,
    /// Phantom to retain the float type parameter.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedAgglomerativeClustering<F> {
    /// Return the cluster label for each training sample.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }

    /// Return the number of clusters formed.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters_
    }

    /// Return the merge tree: pairs of cluster IDs that were merged.
    ///
    /// The entries are in merge order (earliest merge first).
    #[must_use]
    pub fn children(&self) -> &[(usize, usize)] {
        &self.children_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared Euclidean distance between two row slices.
#[inline]
fn sq_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Compute the full `n × n` pairwise squared-distance matrix.
fn pairwise_sq_dists<F: Float>(x: &Array2<F>) -> Vec<F> {
    let n = x.nrows();
    let mut d = vec![F::zero(); n * n];
    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        for j in (i + 1)..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            let dist = sq_euclidean(si, sj);
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Find the (i, j) pair with the smallest value in `dist_mat` among the
/// currently active clusters.
fn find_min_pair(dist_mat: &[f64], active: &[usize]) -> (usize, usize) {
    let mut best_i = active[0];
    let mut best_j = active[1];
    let n = (dist_mat.len() as f64).sqrt() as usize;
    let mut best_val = f64::INFINITY;

    for (ai, &i) in active.iter().enumerate() {
        for &j in active.iter().skip(ai + 1) {
            let v = dist_mat[i * n + j];
            if v < best_val {
                best_val = v;
                best_i = i;
                best_j = j;
            }
        }
    }
    (best_i, best_j)
}

/// Return type of the internal `agglomerate` helper.
type AgglomerateResult = Result<(Array1<usize>, Vec<(usize, usize)>), FerroError>;

/// Generic helper: run agglomerative clustering returning `(labels, children)`.
///
/// We work entirely with `f64` internally and accept the input as a trait
/// object of `Float` by converting upfront.
fn agglomerate<F: Float>(
    x: &Array2<F>,
    n_clusters_target: usize,
    linkage: Linkage,
) -> AgglomerateResult {
    let n_samples = x.nrows();

    // Convert data to f64 for internal computation.
    let x_f64: Array2<f64> = x.mapv(|v| v.to_f64().unwrap_or(0.0));

    // Build pairwise squared-distance matrix (n × n, flat, row-major).
    let mut sq_dists = pairwise_sq_dists(&x_f64);
    let n = n_samples;

    // For Ward linkage we also need cluster sizes and sum-of-squares.
    // For others we just track sizes to apply Lance–Williams updates.
    let mut sizes: Vec<f64> = vec![1.0; n];

    // active[i] = current internal cluster ID of the i-th active position.
    let mut active: Vec<usize> = (0..n).collect();

    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n - n_clusters_target);

    // cluster_id[i] = which leaf cluster i belongs to at the current merge step.
    // Initially each sample is its own cluster.
    let mut assignment: Vec<usize> = (0..n).collect();

    // Counter for new cluster IDs after merges (reuse the merged-into slot).
    // We track the merge history as pairs of original-or-merged IDs.

    while active.len() > n_clusters_target {
        // ── Find the two closest active clusters ────────────────────────────
        let (ci, cj) = find_min_pair(&sq_dists, &active);

        // Remove cj from active; ci absorbs cj.
        active.retain(|&id| id != cj);
        children.push((ci, cj));

        let ni = sizes[ci];
        let nj = sizes[cj];
        let new_size = ni + nj;

        // ── Update the distance matrix using Lance–Williams recurrence ───────
        // For the merged cluster (stored in slot ci), update dist to all
        // remaining active clusters.
        for &ck in &active {
            if ck == ci {
                continue;
            }
            let nk = sizes[ck];
            let d_ik = sq_dists[ci * n + ck];
            let d_jk = sq_dists[cj * n + ck];

            let new_dist = match linkage {
                Linkage::Single => {
                    if d_ik < d_jk {
                        d_ik
                    } else {
                        d_jk
                    }
                }
                Linkage::Complete => {
                    if d_ik > d_jk {
                        d_ik
                    } else {
                        d_jk
                    }
                }
                Linkage::Average => (ni * d_ik + nj * d_jk) / (ni + nj),
                Linkage::Ward => {
                    // Ward: squared Euclidean distance between new centroid
                    // and existing centroid, weighted by sizes.
                    // Lance–Williams for Ward:
                    // d(ij, k) = ((n_i + n_k)/(n_i+n_j+n_k)) * d(i,k)
                    //          + ((n_j + n_k)/(n_i+n_j+n_k)) * d(j,k)
                    //          - (n_k      /(n_i+n_j+n_k)) * d(i,j)
                    let d_ij = sq_dists[ci * n + cj];
                    let denom = ni + nj + nk;
                    ((ni + nk) / denom) * d_ik + ((nj + nk) / denom) * d_jk - (nk / denom) * d_ij
                }
            };

            sq_dists[ci * n + ck] = new_dist;
            sq_dists[ck * n + ci] = new_dist;
        }

        sizes[ci] = new_size;

        // Redirect all samples assigned to cj → ci.
        for s in assignment.iter_mut() {
            if *s == cj {
                *s = ci;
            }
        }
    }

    // ── Re-label active cluster IDs as 0 .. n_clusters_target ───────────────
    let mut id_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (new_id, &cluster_id) in active.iter().enumerate() {
        id_map.insert(cluster_id, new_id);
    }
    let labels: Array1<usize> = assignment
        .iter()
        .map(|id| *id_map.get(id).unwrap_or(&0))
        .collect();

    Ok((labels, children))
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impl: Fit
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for AgglomerativeClustering<F> {
    type Fitted = FittedAgglomerativeClustering<F>;
    type Error = FerroError;

    /// Run agglomerative clustering on `x`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_clusters == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < n_clusters`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedAgglomerativeClustering<F>, FerroError> {
        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: 0,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        if n_samples < self.n_clusters {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: n_samples,
                context: "AgglomerativeClustering requires at least n_clusters samples".into(),
            });
        }

        let (labels, children) = agglomerate(x, self.n_clusters, self.linkage)?;

        Ok(FittedAgglomerativeClustering {
            labels_: labels,
            n_clusters_: self.n_clusters,
            children_: children,
            _marker: std::marker::PhantomData,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated blobs.
    fn make_two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
                10.05, 10.05,
            ],
        )
        .unwrap()
    }

    /// Three well-separated blobs.
    fn make_three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 0.0, 10.0, 0.1,
                10.1, -0.1, 9.9,
            ],
        )
        .unwrap()
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_defaults() {
        let model = AgglomerativeClustering::<f64>::new(3);
        assert_eq!(model.n_clusters, 3);
        assert_eq!(model.linkage, Linkage::Ward);
    }

    #[test]
    fn test_with_linkage() {
        let model = AgglomerativeClustering::<f64>::new(2).with_linkage(Linkage::Complete);
        assert_eq!(model.linkage, Linkage::Complete);
    }

    // ── Error conditions ────────────────────────────────────────────────────

    #[test]
    fn test_zero_clusters_error() {
        let x = make_two_blobs();
        let result = AgglomerativeClustering::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = AgglomerativeClustering::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_more_clusters_than_samples_error() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = AgglomerativeClustering::<f64>::new(5).fit(&x, &());
        assert!(result.is_err());
    }

    // ── Ward linkage ────────────────────────────────────────────────────────

    #[test]
    fn test_ward_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Ward)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        // First 4 should be in the same cluster; last 4 in another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_ward_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Ward)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    // ── Complete linkage ────────────────────────────────────────────────────

    #[test]
    fn test_complete_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Complete)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_complete_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Complete)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Average linkage ─────────────────────────────────────────────────────

    #[test]
    fn test_average_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Average)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_average_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Average)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Single linkage ──────────────────────────────────────────────────────

    #[test]
    fn test_single_two_blobs() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2)
            .with_linkage(Linkage::Single)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_single_three_blobs() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3)
            .with_linkage(Linkage::Single)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Label properties ─────────────────────────────────────────────────────

    #[test]
    fn test_label_count_equals_n_samples() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), x.nrows());
    }

    #[test]
    fn test_labels_in_valid_range() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        for &l in fitted.labels().iter() {
            assert!(l < 3, "label {l} out of range");
        }
    }

    #[test]
    fn test_n_clusters_matches_config() {
        let x = make_three_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        assert_eq!(fitted.n_clusters(), 3);
    }

    // ── Children (merge tree) ────────────────────────────────────────────────

    #[test]
    fn test_children_length() {
        let x = make_two_blobs(); // 8 samples, 2 clusters → 6 merges
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.children().len(), x.nrows() - 2);
    }

    #[test]
    fn test_children_empty_when_n_clusters_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        assert!(fitted.children().is_empty());
    }

    // ── Special cases ─────────────────────────────────────────────────────────

    #[test]
    fn test_single_cluster() {
        let x = make_two_blobs();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        // All samples should be in cluster 0.
        for &l in fitted.labels().iter() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_n_clusters_equals_n_samples() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(3).fit(&x, &()).unwrap();
        // Each sample is its own cluster; labels should all be distinct.
        let labels = fitted.labels();
        assert_ne!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[1], labels[2]);
    }

    #[test]
    fn test_single_sample_single_cluster() {
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.n_clusters(), 1);
        assert!(fitted.children().is_empty());
    }

    #[test]
    fn test_1d_data() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, -0.1, 100.0, 100.1, 99.9]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(2).fit(&x, &()).unwrap();
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let fitted = AgglomerativeClustering::<f32>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 6);
        let labels = fitted.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_identical_points() {
        // All points identical → all should be in the same cluster.
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let fitted = AgglomerativeClustering::<f64>::new(1).fit(&x, &()).unwrap();
        for &l in fitted.labels().iter() {
            assert_eq!(l, 0);
        }
    }
}
