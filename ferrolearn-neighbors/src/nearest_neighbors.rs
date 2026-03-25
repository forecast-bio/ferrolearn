//! Unsupervised nearest neighbor search.
//!
//! This module provides [`NearestNeighbors`], an unsupervised learner that
//! stores training data and builds a spatial index for efficient neighbor
//! queries. Unlike [`KNeighborsClassifier`](crate::KNeighborsClassifier) and
//! [`KNeighborsRegressor`](crate::KNeighborsRegressor), no labels are required.
//!
//! # Use Cases
//!
//! - Finding similar items in a dataset
//! - Graph construction (k-NN graphs)
//! - Anomaly detection via local density estimation
//! - Feature engineering with neighbor-based features
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::NearestNeighbors;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
//!     5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
//! ]).unwrap();
//!
//! let model = NearestNeighbors::<f64>::new();
//! let fitted = model.fit(&x, &()).unwrap();
//!
//! let (distances, indices) = fitted.kneighbors(&x, None).unwrap();
//! assert_eq!(distances.nrows(), 6);
//! assert_eq!(indices.nrows(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;
use num_traits::Float;

use crate::balltree::BallTree;
use crate::kdtree::{self, KdTree};
use crate::knn::Algorithm;

/// Result type for radius neighbor queries: a vector of `(distances, indices)`
/// per query point.
pub type RadiusNeighborResult<F> = Vec<(Vec<F>, Vec<usize>)>;

// ---------------------------------------------------------------------------
// NearestNeighbors (unfitted)
// ---------------------------------------------------------------------------

/// Unsupervised nearest neighbor search.
///
/// Stores training data and builds a spatial index for efficient neighbor
/// queries. No labels are needed — this is purely geometric.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::NearestNeighbors;
/// use ferrolearn_core::Fit;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
/// ]).unwrap();
///
/// let nn = NearestNeighbors::<f64>::new().with_n_neighbors(2);
/// let fitted = nn.fit(&x, &()).unwrap();
/// let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
/// assert_eq!(dists.dim(), (4, 2));
/// ```
#[derive(Debug, Clone)]
pub struct NearestNeighbors<F> {
    /// Default number of neighbors for queries (default 5).
    pub n_neighbors: usize,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// Leaf size for tree-based algorithms (unused for brute force).
    pub leaf_size: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> NearestNeighbors<F> {
    /// Create a new `NearestNeighbors` with default settings.
    ///
    /// Defaults: `n_neighbors = 5`, `algorithm = Auto`, `leaf_size = 30`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            algorithm: Algorithm::Auto,
            leaf_size: 30,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the default number of neighbors for queries.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the leaf size for tree-based algorithms.
    #[must_use]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }
}

impl<F: Float> Default for NearestNeighbors<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Spatial index wrapper
// ---------------------------------------------------------------------------

/// Which spatial index was built during fit.
enum SpatialIndex {
    None,
    KdTree(KdTree),
    BallTree(BallTree),
}

impl std::fmt::Debug for SpatialIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialIndex::None => write!(f, "BruteForce"),
            SpatialIndex::KdTree(t) => write!(f, "KdTree({t:?})"),
            SpatialIndex::BallTree(t) => write!(f, "BallTree({t:?})"),
        }
    }
}

/// Build the appropriate spatial index based on algorithm setting and dimensionality.
fn build_spatial_index<F: Float + Send + Sync + 'static>(
    algorithm: Algorithm,
    data: &Array2<F>,
) -> SpatialIndex {
    let n_features = data.ncols();
    match algorithm {
        Algorithm::Auto => {
            if n_features <= 15 {
                SpatialIndex::KdTree(KdTree::build(data))
            } else {
                SpatialIndex::BallTree(BallTree::build(data))
            }
        }
        Algorithm::KdTree => SpatialIndex::KdTree(KdTree::build(data)),
        Algorithm::BallTree => SpatialIndex::BallTree(BallTree::build(data)),
        Algorithm::BruteForce => SpatialIndex::None,
    }
}

// ---------------------------------------------------------------------------
// FittedNearestNeighbors
// ---------------------------------------------------------------------------

/// Fitted unsupervised nearest neighbor search.
///
/// Stores the training data and spatial index. Provides methods for
/// k-nearest-neighbor queries and radius-based queries.
#[derive(Debug)]
pub struct FittedNearestNeighbors<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Default number of neighbors for queries.
    n_neighbors: usize,
    /// Spatial index.
    spatial_index: SpatialIndex,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for NearestNeighbors<F> {
    type Fitted = FittedNearestNeighbors<F>;
    type Error = FerroError;

    /// Fit the nearest neighbor search by storing the training data and
    /// building a spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer
    /// samples than `n_neighbors`.
    fn fit(
        &self,
        x: &Array2<F>,
        _y: &(),
    ) -> Result<FittedNearestNeighbors<F>, FerroError> {
        let n_samples = x.nrows();

        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        if n_samples < self.n_neighbors {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples,
                context: format!(
                    "NearestNeighbors requires at least n_neighbors={} samples",
                    self.n_neighbors
                ),
            });
        }

        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedNearestNeighbors {
            x_train: x.clone(),
            n_neighbors: self.n_neighbors,
            spatial_index,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedNearestNeighbors<F> {
    /// Find the k nearest neighbors for each row in `x`.
    ///
    /// # Arguments
    ///
    /// - `x`: An `(n_queries, n_features)` array of query points.
    /// - `n_neighbors`: Number of neighbors to find. If `None`, uses the
    ///   default from construction.
    ///
    /// # Returns
    ///
    /// A tuple `(distances, indices)` where:
    /// - `distances` is an `(n_queries, k)` array of Euclidean distances.
    /// - `indices` is an `(n_queries, k)` array of training sample indices.
    ///
    /// Both are sorted by distance ascending within each row.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension of `x`
    /// does not match the training data.
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero or
    /// exceeds the number of training samples.
    pub fn kneighbors(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
    ) -> Result<(Array2<F>, Array2<usize>), FerroError> {
        let k = n_neighbors.unwrap_or(self.n_neighbors);
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        if k == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_train = self.x_train.nrows();
        if k > n_train {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: format!(
                    "n_neighbors={k} exceeds number of training samples={n_train}"
                ),
            });
        }

        let n_queries = x.nrows();
        let mut distances = Array2::<F>::zeros((n_queries, k));
        let mut indices = Array2::<usize>::zeros((n_queries, k));

        for i in 0..n_queries {
            let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
            let neighbors = self.find_knn(&query, k);

            for (j, (idx, dist)) in neighbors.into_iter().enumerate() {
                distances[[i, j]] = dist;
                indices[[i, j]] = idx;
            }
        }

        Ok((distances, indices))
    }

    /// Find all neighbors within a given radius for each row in `x`.
    ///
    /// # Arguments
    ///
    /// - `x`: An `(n_queries, n_features)` array of query points.
    /// - `radius`: The search radius (Euclidean distance).
    ///
    /// # Returns
    ///
    /// A vector of `(distances, indices)` tuples, one per query point.
    /// Each inner vector contains the neighbors found within `radius`,
    /// sorted by distance ascending.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension of `x`
    /// does not match the training data.
    /// Returns [`FerroError::InvalidParameter`] if `radius` is negative.
    pub fn radius_neighbors(
        &self,
        x: &Array2<F>,
        radius: F,
    ) -> Result<RadiusNeighborResult<F>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        if radius < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "radius".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_queries = x.nrows();
        let mut results = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
            let mut neighbors = self.find_radius(&query, radius);

            // Sort by distance ascending.
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let indices: Vec<usize> = neighbors.iter().map(|&(idx, _)| idx).collect();
            let distances: Vec<F> = neighbors.iter().map(|&(_, d)| d).collect();

            results.push((distances, indices));
        }

        Ok(results)
    }

    /// Return the number of training samples.
    #[must_use]
    pub fn n_samples_fit(&self) -> usize {
        self.x_train.nrows()
    }

    /// Return the shape of the training data.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.x_train.dim()
    }

    /// Find k nearest neighbors of a single query point.
    fn find_knn(&self, query: &[F], k: usize) -> Vec<(usize, F)> {
        match &self.spatial_index {
            SpatialIndex::KdTree(tree) => {
                let query_f64: Vec<f64> = query.iter().map(|v| v.to_f64().unwrap()).collect();
                tree.query(&self.x_train, &query_f64, k)
                    .into_iter()
                    .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                    .collect()
            }
            SpatialIndex::BallTree(tree) => {
                let query_f64: Vec<f64> = query.iter().map(|v| v.to_f64().unwrap()).collect();
                tree.query(&self.x_train, &query_f64, k)
                    .into_iter()
                    .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                    .collect()
            }
            SpatialIndex::None => kdtree::brute_force_knn(&self.x_train, query, k),
        }
    }

    /// Find all neighbors within radius of a single query point.
    fn find_radius(&self, query: &[F], radius: F) -> Vec<(usize, F)> {
        match &self.spatial_index {
            SpatialIndex::BallTree(tree) => {
                let query_f64: Vec<f64> = query.iter().map(|v| v.to_f64().unwrap()).collect();
                let radius_f64 = radius.to_f64().unwrap();
                tree.within_radius(&query_f64, radius_f64)
                    .into_iter()
                    .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                    .collect()
            }
            _ => {
                // Brute force radius search for KdTree and BruteForce modes.
                brute_force_radius(&self.x_train, query, radius)
            }
        }
    }
}

/// Brute-force radius search: find all training points within `radius` of `query`.
///
/// Returns `(index, distance)` pairs (unsorted).
fn brute_force_radius<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query: &[F],
    radius: F,
) -> Vec<(usize, F)> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let mut results = Vec::new();

    for i in 0..n_samples {
        let point: Vec<F> = (0..n_features).map(|j| data[[i, j]]).collect();
        let dist = kdtree::euclidean_distance(&point, query);
        if dist <= radius {
            results.push((i, dist));
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn sample_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_fit_basic() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new();
        let fitted = nn.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_samples_fit(), 6);
        assert_eq!(fitted.shape(), (6, 2));
    }

    #[test]
    fn test_fit_invalid_k_zero() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(0);
        assert!(nn.fit(&x, &()).is_err());
    }

    #[test]
    fn test_fit_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(5);
        assert!(nn.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kneighbors_default_k() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(dists.dim(), (6, 3));
        assert_eq!(idxs.dim(), (6, 3));

        // Distances should be sorted ascending within each row.
        for i in 0..6 {
            for j in 1..3 {
                assert!(dists[[i, j]] >= dists[[i, j - 1]]);
            }
        }
    }

    #[test]
    fn test_kneighbors_override_k() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, Some(2)).unwrap();
        assert_eq!(dists.dim(), (6, 2));
        assert_eq!(idxs.dim(), (6, 2));
    }

    #[test]
    fn test_kneighbors_k1_self_match() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        // Each point's nearest neighbor should be itself (distance 0).
        for i in 0..6 {
            assert_eq!(idxs[[i, 0]], i);
            assert!(dists[[i, 0]] < 1e-10);
        }
    }

    #[test]
    fn test_kneighbors_shape_mismatch() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new();
        let fitted = nn.fit(&x, &()).unwrap();

        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.kneighbors(&x_bad, None).is_err());
    }

    #[test]
    fn test_kneighbors_k_too_large() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);
        let fitted = nn.fit(&x, &()).unwrap();

        assert!(fitted.kneighbors(&x, Some(100)).is_err());
    }

    #[test]
    fn test_kneighbors_k_zero() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(3);
        let fitted = nn.fit(&x, &()).unwrap();

        assert!(fitted.kneighbors(&x, Some(0)).is_err());
    }

    #[test]
    fn test_radius_neighbors_basic() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        let results = fitted.radius_neighbors(&x, 1.5).unwrap();
        assert_eq!(results.len(), 4);

        // Point (0,0): neighbors within 1.5 are (0,0), (1,0), (0,1).
        assert_eq!(results[0].1.len(), 3);
        let mut idxs = results[0].1.clone();
        idxs.sort();
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn test_radius_neighbors_empty() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.0, 10.0, 10.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        let query = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let results = fitted.radius_neighbors(&query, 0.1).unwrap();
        assert!(results[0].0.is_empty());
        assert!(results[0].1.is_empty());
    }

    #[test]
    fn test_radius_neighbors_negative_radius() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new();
        let fitted = nn.fit(&x, &()).unwrap();

        assert!(fitted.radius_neighbors(&x, -1.0).is_err());
    }

    #[test]
    fn test_radius_neighbors_shape_mismatch() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new();
        let fitted = nn.fit(&x, &()).unwrap();

        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.radius_neighbors(&x_bad, 1.0).is_err());
    }

    #[test]
    fn test_radius_neighbors_sorted_by_distance() {
        let x = Array2::from_shape_vec(
            (4, 1),
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
        let results = fitted.radius_neighbors(&query, 5.0).unwrap();

        // All 4 points within radius 5.0.
        let dists = &results[0].0;
        for i in 1..dists.len() {
            assert!(dists[i] >= dists[i - 1]);
        }
    }

    #[test]
    fn test_brute_force_algorithm() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new()
            .with_n_neighbors(2)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(dists.dim(), (6, 2));
        assert_eq!(idxs.dim(), (6, 2));
    }

    #[test]
    fn test_kdtree_algorithm() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new()
            .with_n_neighbors(2)
            .with_algorithm(Algorithm::KdTree);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(dists.dim(), (6, 2));
        assert_eq!(idxs.dim(), (6, 2));
    }

    #[test]
    fn test_balltree_algorithm() {
        let x = sample_data();
        let nn = NearestNeighbors::<f64>::new()
            .with_n_neighbors(2)
            .with_algorithm(Algorithm::BallTree);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(dists.dim(), (6, 2));
        assert_eq!(idxs.dim(), (6, 2));
    }

    #[test]
    fn test_all_algorithms_agree_kneighbors() {
        let x = sample_data();

        let algos = [
            Algorithm::BruteForce,
            Algorithm::KdTree,
            Algorithm::BallTree,
        ];

        let mut reference_dists = None;
        let mut reference_idxs = None;

        for algo in &algos {
            let nn = NearestNeighbors::<f64>::new()
                .with_n_neighbors(3)
                .with_algorithm(*algo);
            let fitted = nn.fit(&x, &()).unwrap();
            let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();

            if reference_dists.is_none() {
                reference_dists = Some(dists);
                reference_idxs = Some(idxs);
            } else {
                let ref_d = reference_dists.as_ref().unwrap();
                // Compare distances (indices may vary for equidistant points).
                for i in 0..6 {
                    for j in 0..3 {
                        assert!(
                            (dists[[i, j]] - ref_d[[i, j]]).abs() < 1e-10,
                            "algo={algo:?}, row={i}, col={j}: {} vs {}",
                            dists[[i, j]],
                            ref_d[[i, j]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_radius_neighbors_brute_force() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = nn.fit(&x, &()).unwrap();

        let results = fitted.radius_neighbors(&x, 1.5).unwrap();

        // Point (0,0): neighbors are (0,0), (1,0), (0,1).
        let mut idxs = results[0].1.clone();
        idxs.sort();
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn test_radius_neighbors_balltree() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::BallTree);
        let fitted = nn.fit(&x, &()).unwrap();

        let results = fitted.radius_neighbors(&x, 1.5).unwrap();

        let mut idxs = results[0].1.clone();
        idxs.sort();
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn test_default_matches_new() {
        let nn = NearestNeighbors::<f64>::default();
        assert_eq!(nn.n_neighbors, 5);
        assert_eq!(nn.algorithm, Algorithm::Auto);
        assert_eq!(nn.leaf_size, 30);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let nn = NearestNeighbors::<f32>::new().with_n_neighbors(2);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(dists.dim(), (4, 2));
        assert_eq!(idxs.dim(), (4, 2));
    }

    #[test]
    fn test_single_point() {
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        let (dists, idxs) = fitted.kneighbors(&x, None).unwrap();
        assert_eq!(idxs[[0, 0]], 0);
        assert!(dists[[0, 0]] < 1e-10);
    }

    #[test]
    fn test_radius_zero() {
        let x = Array2::from_shape_vec(
            (3, 1),
            vec![0.0, 1.0, 2.0],
        )
        .unwrap();

        let nn = NearestNeighbors::<f64>::new().with_n_neighbors(1);
        let fitted = nn.fit(&x, &()).unwrap();

        // Radius 0 should only find exact matches.
        let results = fitted.radius_neighbors(&x, 0.0).unwrap();
        for i in 0..3 {
            assert_eq!(results[i].1.len(), 1);
            assert_eq!(results[i].1[0], i);
        }
    }
}
