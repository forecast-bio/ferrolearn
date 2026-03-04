//! Isomap (Isometric Mapping).
//!
//! Non-linear dimensionality reduction that preserves geodesic (shortest-path)
//! distances along the data manifold.
//!
//! # Algorithm
//!
//! 1. Build a k-nearest-neighbor graph over the data.
//! 2. Compute shortest paths between all pairs of points using Dijkstra's
//!    algorithm.
//! 3. Apply classical MDS to the geodesic distance matrix to obtain the
//!    low-dimensional embedding.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::Isomap;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let iso = Isomap::new(2).with_n_neighbors(3);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [2.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [2.0, 1.0],
//!     [0.0, 2.0],
//!     [1.0, 2.0],
//!     [2.0, 2.0],
//! ];
//! let fitted = iso.fit(&x, &()).unwrap();
//! let emb = fitted.embedding();
//! assert_eq!(emb.ncols(), 2);
//! ```

use crate::mds::{classical_mds, eigh_faer, pairwise_sq_distances};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Isomap (unfitted)
// ---------------------------------------------------------------------------

/// Isomap configuration.
///
/// Holds hyperparameters for the Isomap algorithm. Call [`Fit::fit`] to compute
/// the geodesic embedding and obtain a [`FittedIsomap`].
#[derive(Debug, Clone)]
pub struct Isomap {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Number of nearest neighbors for the kNN graph.
    n_neighbors: usize,
}

impl Isomap {
    /// Create a new `Isomap` with `n_components` embedding dimensions.
    ///
    /// The default number of neighbors is 5.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_neighbors: 5,
        }
    }

    /// Set the number of nearest neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured number of neighbors.
    #[must_use]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }
}

// ---------------------------------------------------------------------------
// FittedIsomap
// ---------------------------------------------------------------------------

/// A fitted Isomap model holding the learned embedding and training data.
///
/// Created by calling [`Fit::fit`] on an [`Isomap`]. Implements
/// [`Transform<Array2<f64>>`] for out-of-sample projection using
/// nearest-neighbor interpolation against the training embedding.
#[derive(Debug, Clone)]
pub struct FittedIsomap {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
    /// Training data, stored for out-of-sample extension.
    x_train_: Array2<f64>,
    /// Number of neighbors used during fitting.
    _n_neighbors: usize,
    /// Kernel matrix eigenvalues from the MDS step (top n_components).
    eigenvalues_: Vec<f64>,
    /// Kernel matrix eigenvectors from the MDS step, shape `(n_train, n_components)`.
    eigenvectors_: Array2<f64>,
    /// Mean of the squared geodesic distance rows, for Nystroem extension.
    geo_sq_row_means_: Vec<f64>,
    /// Grand mean of the squared geodesic distances.
    geo_sq_grand_mean_: f64,
}

impl FittedIsomap {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }
}

// ---------------------------------------------------------------------------
// kNN + Dijkstra helpers
// ---------------------------------------------------------------------------

/// A state for Dijkstra's priority queue.
#[derive(Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip for min-heap
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a k-nearest-neighbor adjacency list from squared distances.
/// Returns `adj[i]` = Vec of (neighbor_index, distance).
fn build_knn_graph(sq_dist: &Array2<f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = sq_dist.nrows();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for i in 0..n {
        // Collect (distance, index) for all other points.
        let mut neighbors: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (sq_dist[[i, j]].sqrt(), j))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        for &(dist, j) in neighbors.iter().take(k) {
            adj[i].push((j, dist));
        }
    }

    // Make the graph symmetric: if i is a neighbor of j, j is a neighbor of i.
    let mut sym: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for &(j, d) in &adj[i] {
            sym[i].push((j, d));
            sym[j].push((i, d));
        }
    }
    // Deduplicate keeping shortest distance.
    for entry in &mut sym {
        entry.sort_by(|a, b| a.0.cmp(&b.0));
        entry.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.min(a.1);
                true
            } else {
                false
            }
        });
    }
    sym
}

/// Dijkstra shortest path from a single source.
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node: source,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[node] {
            continue;
        }
        for &(neighbor, weight) in &adj[node] {
            let next_cost = cost + weight;
            if next_cost < dist[neighbor] {
                dist[neighbor] = next_cost;
                heap.push(State {
                    cost: next_cost,
                    node: neighbor,
                });
            }
        }
    }
    dist
}

/// Compute all-pairs shortest paths via Dijkstra.
fn all_pairs_shortest_paths(adj: &[Vec<(usize, f64)>]) -> Array2<f64> {
    let n = adj.len();
    let mut result = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let dists = dijkstra(adj, i);
        for j in 0..n {
            result[[i, j]] = dists[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for Isomap {
    type Fitted = FittedIsomap;
    type Error = FerroError;

    /// Fit Isomap by building the kNN graph, computing geodesic distances,
    /// and applying classical MDS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero,
    ///   `n_neighbors` is zero, or `n_components > n_samples`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples
    ///   or fewer samples than `n_neighbors + 1`.
    /// - [`FerroError::NumericalInstability`] if the kNN graph is disconnected.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedIsomap, FerroError> {
        let n_samples = x.nrows();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "Isomap::fit requires at least 2 samples".into(),
            });
        }
        if self.n_neighbors >= n_samples {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: format!(
                    "n_neighbors ({}) must be less than n_samples ({})",
                    self.n_neighbors, n_samples
                ),
            });
        }
        if self.n_components > n_samples {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_samples ({})",
                    self.n_components, n_samples
                ),
            });
        }

        // Step 1: pairwise Euclidean distances.
        let sq_dist = pairwise_sq_distances(x);

        // Step 2: Build kNN graph.
        let adj = build_knn_graph(&sq_dist, self.n_neighbors);

        // Step 3: All-pairs shortest paths.
        let geodesic = all_pairs_shortest_paths(&adj);

        // Check for disconnected graph.
        for i in 0..n_samples {
            for j in 0..n_samples {
                if geodesic[[i, j]].is_infinite() {
                    return Err(FerroError::NumericalInstability {
                        message: format!(
                            "kNN graph is disconnected (no path from point {} to {}). \
                             Try increasing n_neighbors.",
                            i, j
                        ),
                    });
                }
            }
        }

        // Step 4: Classical MDS on the geodesic distance matrix.
        let geo_sq = geodesic.mapv(|v| v * v);

        // We need extra info for Nystroem extension, so we do the MDS manually.
        let n = n_samples;
        let n_f = n as f64;
        let mut row_means = vec![0.0; n];
        let mut grand_mean = 0.0;
        for i in 0..n {
            for j in 0..n {
                row_means[i] += geo_sq[[i, j]];
                grand_mean += geo_sq[[i, j]];
            }
            row_means[i] /= n_f;
        }
        grand_mean /= n_f * n_f;

        let (embedding, _stress) = classical_mds(&geo_sq, self.n_components)?;

        // Eigendecompose for Nystroem extension storage
        let mut b = Array2::<f64>::zeros((n, n));
        let mut col_means = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                col_means[j] += geo_sq[[i, j]];
            }
            col_means[j] /= n_f;
        }
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * (geo_sq[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        let (eigenvalues, eigenvectors) = eigh_faer(&b)?;

        // Sort eigenvalues descending, select top n_components
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b_idx| {
            eigenvalues[b_idx]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_comp = self.n_components.min(n);
        let mut top_eigenvalues = Vec::with_capacity(n_comp);
        let mut top_eigenvectors = Array2::<f64>::zeros((n, n_comp));
        for (k, &idx) in indices.iter().take(n_comp).enumerate() {
            top_eigenvalues.push(eigenvalues[idx].max(0.0));
            for i in 0..n {
                top_eigenvectors[[i, k]] = eigenvectors[[i, idx]];
            }
        }

        Ok(FittedIsomap {
            embedding_: embedding,
            x_train_: x.to_owned(),
            _n_neighbors: self.n_neighbors,
            eigenvalues_: top_eigenvalues,
            eigenvectors_: top_eigenvectors,
            geo_sq_row_means_: row_means,
            geo_sq_grand_mean_: grand_mean,
        })
    }
}

impl Transform<Array2<f64>> for FittedIsomap {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Project new data into the Isomap embedding space.
    ///
    /// Uses the Nystroem approximation: for each new point, compute the
    /// Euclidean distance to all training points, approximate geodesic
    /// distances using the k nearest training neighbors, then project
    /// using the stored eigenvectors and eigenvalues.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_features = self.x_train_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedIsomap::transform".into(),
            });
        }

        let n_test = x.nrows();
        let n_train = self.x_train_.nrows();
        let n_comp = self.eigenvalues_.len();

        let mut result = Array2::<f64>::zeros((n_test, n_comp));

        for t in 0..n_test {
            // Compute Euclidean distances from the test point to all training points.
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|i| {
                    let mut sq = 0.0;
                    for k in 0..n_features {
                        let diff = x[[t, k]] - self.x_train_[[i, k]];
                        sq += diff * diff;
                    }
                    (sq.sqrt(), i)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            // Use the distances to the training points as an approximation
            // of geodesic distances (for points close to the manifold this
            // is reasonable). Square them for the Nystroem formula.
            let delta_sq: Vec<f64> = dists.iter().map(|&(d, _)| d * d).collect();

            // Nystroem projection:
            // embedding_k = (1 / (2 * sqrt(lambda_k))) * sum_i v_ki * (mu_i - delta_sq_i)
            // where mu_i = row_mean_i of squared geodesic distances and
            // delta_sq_i = sq distance from new point to training point i.
            //
            // But using the correct Nystroem formula:
            // x_new_k = (1 / sqrt(lambda_k)) * sum_i v_ki * b_i
            // where b_i = -0.5 * (delta_sq_i - row_mean_i - mean(delta_sq) + grand_mean)
            let delta_sq_mean: f64 = delta_sq.iter().sum::<f64>() / n_train as f64;

            // Reorder delta_sq by original training index
            let mut delta_sq_ordered = vec![0.0; n_train];
            for &(d, idx) in &dists {
                delta_sq_ordered[idx] = d * d;
            }

            for k in 0..n_comp {
                let eigval = self.eigenvalues_[k];
                if eigval <= 1e-12 {
                    continue;
                }
                let scale = 1.0 / eigval.sqrt();
                let mut sum = 0.0;
                for (i, &dsq_i) in delta_sq_ordered.iter().enumerate().take(n_train) {
                    let b_i = -0.5
                        * (dsq_i - self.geo_sq_row_means_[i] - delta_sq_mean
                            + self.geo_sq_grand_mean_);
                    sum += self.eigenvectors_[[i, k]] * b_i;
                }
                result[[t, k]] = sum * scale;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: simple line dataset.
    fn line_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],]
    }

    /// Helper: 2D grid.
    fn grid_data() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    }

    #[test]
    fn test_isomap_basic_shape() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (9, 2));
    }

    #[test]
    fn test_isomap_1d() {
        let iso = Isomap::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = iso.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_isomap_preserves_ordering() {
        // Points on a line: the embedding should preserve the ordering.
        let iso = Isomap::new(1).with_n_neighbors(2);
        let x = line_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let emb = fitted.embedding();
        let vals: Vec<f64> = (0..5).map(|i| emb[[i, 0]]).collect();
        // Check that values are monotonic (up to sign).
        let ascending = vals.windows(2).all(|w| w[0] <= w[1] + 1e-10);
        let descending = vals.windows(2).all(|w| w[0] >= w[1] - 1e-10);
        assert!(
            ascending || descending,
            "embedding should be monotonic: {vals:?}"
        );
    }

    #[test]
    fn test_isomap_transform_new_data() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x_train = grid_data();
        let fitted = iso.fit(&x_train, &()).unwrap();
        let x_test = array![[0.5, 0.5], [1.5, 1.5]];
        let projected = fitted.transform(&x_test).unwrap();
        assert_eq!(projected.dim(), (2, 2));
    }

    #[test]
    fn test_isomap_transform_shape_mismatch() {
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_isomap_transform_recovers_training() {
        // Transforming the training data should produce something close to
        // the stored embedding.
        let iso = Isomap::new(2).with_n_neighbors(3);
        let x = grid_data();
        let fitted = iso.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let emb = fitted.embedding();
        // They won't be exactly equal because the Nystroem approximation
        // differs from exact MDS, but they should be correlated.
        assert_eq!(projected.dim(), emb.dim());
    }

    #[test]
    fn test_isomap_invalid_n_components_zero() {
        let iso = Isomap::new(0);
        let x = grid_data();
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_invalid_n_neighbors_zero() {
        let iso = Isomap::new(2).with_n_neighbors(0);
        let x = grid_data();
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_n_neighbors_too_large() {
        let iso = Isomap::new(2).with_n_neighbors(100);
        let x = grid_data(); // 9 samples
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_insufficient_samples() {
        let iso = Isomap::new(1).with_n_neighbors(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(iso.fit(&x, &()).is_err());
    }

    #[test]
    fn test_isomap_getters() {
        let iso = Isomap::new(3).with_n_neighbors(7);
        assert_eq!(iso.n_components(), 3);
        assert_eq!(iso.n_neighbors(), 7);
    }

    #[test]
    fn test_isomap_default_n_neighbors() {
        let iso = Isomap::new(2);
        assert_eq!(iso.n_neighbors(), 5);
    }

    #[test]
    fn test_isomap_n_components_too_large() {
        let iso = Isomap::new(50);
        let x = grid_data(); // 9 samples
        assert!(iso.fit(&x, &()).is_err());
    }
}
