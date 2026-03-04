//! Spectral Embedding (Laplacian Eigenmaps).
//!
//! Non-linear dimensionality reduction via the graph Laplacian. Points that
//! are close in the original space (according to an affinity measure) are
//! mapped to nearby points in the embedding.
//!
//! # Algorithm
//!
//! 1. Build an affinity matrix (RBF kernel or k-nearest-neighbors).
//! 2. Compute the normalised graph Laplacian:
//!    `L_sym = I - D^{-1/2} W D^{-1/2}`.
//! 3. Find the bottom-k eigenvectors of `L_sym`, excluding the trivial
//!    constant eigenvector (eigenvalue 0).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::{SpectralEmbedding, Affinity};
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let se = SpectralEmbedding::new(2);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [5.0, 5.0],
//!     [6.0, 5.0],
//!     [5.0, 6.0],
//!     [6.0, 6.0],
//! ];
//! let fitted = se.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().ncols(), 2);
//! ```

use crate::mds::eigh_faer;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Affinity type
// ---------------------------------------------------------------------------

/// The affinity function for building the weight matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Affinity {
    /// RBF (Gaussian) kernel: `W_ij = exp(-gamma * ||x_i - x_j||^2)`.
    RBF {
        /// Kernel bandwidth parameter.
        gamma: f64,
    },
    /// k-nearest-neighbors: `W_ij = 1` if `j` is among the k nearest
    /// neighbors of `i` (or vice versa), `0` otherwise.
    NearestNeighbors {
        /// Number of neighbors.
        n_neighbors: usize,
    },
}

// ---------------------------------------------------------------------------
// SpectralEmbedding (unfitted)
// ---------------------------------------------------------------------------

/// Spectral Embedding (Laplacian Eigenmaps) configuration.
///
/// Holds hyperparameters for the spectral embedding algorithm. Call
/// [`Fit::fit`] to compute the Laplacian eigenmap and obtain a
/// [`FittedSpectralEmbedding`].
#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Affinity function.
    affinity: Affinity,
}

impl SpectralEmbedding {
    /// Create a new `SpectralEmbedding` with `n_components` dimensions.
    ///
    /// The default affinity is `RBF { gamma: 1.0 }`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            affinity: Affinity::RBF { gamma: 1.0 },
        }
    }

    /// Set the affinity function.
    #[must_use]
    pub fn with_affinity(mut self, affinity: Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured affinity.
    #[must_use]
    pub fn affinity(&self) -> Affinity {
        self.affinity
    }
}

// ---------------------------------------------------------------------------
// FittedSpectralEmbedding
// ---------------------------------------------------------------------------

/// A fitted Spectral Embedding model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on a [`SpectralEmbedding`].
#[derive(Debug, Clone)]
pub struct FittedSpectralEmbedding {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
}

impl FittedSpectralEmbedding {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the affinity matrix.
fn build_affinity_matrix(x: &Array2<f64>, affinity: &Affinity) -> Array2<f64> {
    let n = x.nrows();
    match affinity {
        Affinity::RBF { gamma } => {
            let mut w = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut sq = 0.0;
                    for k in 0..x.ncols() {
                        let diff = x[[i, k]] - x[[j, k]];
                        sq += diff * diff;
                    }
                    let val = (-gamma * sq).exp();
                    w[[i, j]] = val;
                    w[[j, i]] = val;
                }
                // Diagonal is 0 (no self-loops).
            }
            w
        }
        Affinity::NearestNeighbors { n_neighbors } => {
            let k = *n_neighbors;
            let mut w = Array2::<f64>::zeros((n, n));
            // Compute all pairwise distances.
            let mut sq_dist = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut sq = 0.0;
                    for f in 0..x.ncols() {
                        let diff = x[[i, f]] - x[[j, f]];
                        sq += diff * diff;
                    }
                    sq_dist[[i, j]] = sq;
                    sq_dist[[j, i]] = sq;
                }
            }
            // For each point, find k nearest neighbors.
            for i in 0..n {
                let mut neighbors: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (sq_dist[[i, j]], j))
                    .collect();
                neighbors
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                for &(_, j) in neighbors.iter().take(k) {
                    w[[i, j]] = 1.0;
                    w[[j, i]] = 1.0; // symmetric
                }
            }
            w
        }
    }
}

/// Compute the normalised graph Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`.
fn normalised_laplacian(w: &Array2<f64>) -> Array2<f64> {
    let n = w.nrows();
    // Degree vector.
    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        let deg: f64 = (0..n).map(|j| w[[i, j]]).sum();
        d_inv_sqrt[i] = if deg > 1e-15 { 1.0 / deg.sqrt() } else { 0.0 };
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                l[[i, j]] = 1.0 - d_inv_sqrt[i] * w[[i, j]] * d_inv_sqrt[j];
            } else {
                l[[i, j]] = -d_inv_sqrt[i] * w[[i, j]] * d_inv_sqrt[j];
            }
        }
    }
    l
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for SpectralEmbedding {
    type Fitted = FittedSpectralEmbedding;
    type Error = FerroError;

    /// Fit spectral embedding by building the affinity matrix, computing the
    /// normalized Laplacian, and extracting the bottom eigenvectors.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedSpectralEmbedding, FerroError> {
        let n = x.nrows();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "SpectralEmbedding::fit requires at least 2 samples".into(),
            });
        }
        // We need n_components + 1 eigenvectors (to skip the trivial one).
        if self.n_components >= n {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) must be less than n_samples ({})",
                    self.n_components, n
                ),
            });
        }

        if let Affinity::NearestNeighbors { n_neighbors } = self.affinity {
            if n_neighbors == 0 {
                return Err(FerroError::InvalidParameter {
                    name: "n_neighbors".into(),
                    reason: "must be at least 1".into(),
                });
            }
            if n_neighbors >= n {
                return Err(FerroError::InvalidParameter {
                    name: "n_neighbors".into(),
                    reason: format!(
                        "n_neighbors ({}) must be less than n_samples ({})",
                        n_neighbors, n
                    ),
                });
            }
        }

        if let Affinity::RBF { gamma } = self.affinity {
            if gamma <= 0.0 {
                return Err(FerroError::InvalidParameter {
                    name: "gamma".into(),
                    reason: "must be positive".into(),
                });
            }
        }

        // Step 1: Build affinity matrix.
        let w = build_affinity_matrix(x, &self.affinity);

        // Step 2: Normalised Laplacian.
        let l = normalised_laplacian(&w);

        // Step 3: Eigendecompose (faer returns sorted ascending eigenvalues).
        let (eigenvalues, eigenvectors) = eigh_faer(&l)?;

        // Sort eigenvalues ascending (they should already be, but be safe).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Skip the first (trivial, eigenvalue ~ 0) eigenvector, take next n_components.
        let n_comp = self.n_components;
        let mut embedding = Array2::<f64>::zeros((n, n_comp));
        for (k, &idx) in indices.iter().skip(1).take(n_comp).enumerate() {
            for i in 0..n {
                embedding[[i, k]] = eigenvectors[[i, idx]];
            }
        }

        Ok(FittedSpectralEmbedding {
            embedding_: embedding,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: two well-separated clusters.
    fn two_clusters() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1],
        ]
    }

    /// Helper: simple dataset.
    fn simple_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0],]
    }

    #[test]
    fn test_spectral_embedding_basic_shape() {
        let se = SpectralEmbedding::new(2);
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (8, 2));
    }

    #[test]
    fn test_spectral_embedding_1d() {
        let se = SpectralEmbedding::new(1);
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_spectral_embedding_rbf_separates_clusters() {
        let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: 1.0 });
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        let emb = fitted.embedding();

        // The first 4 points (cluster 1) should have similar values,
        // and the last 4 (cluster 2) should have similar values, with
        // a gap between them.
        let c1_mean: f64 = (0..4).map(|i| emb[[i, 0]]).sum::<f64>() / 4.0;
        let c2_mean: f64 = (4..8).map(|i| emb[[i, 0]]).sum::<f64>() / 4.0;
        assert!(
            (c1_mean - c2_mean).abs() > 0.01,
            "clusters should be separated: c1={c1_mean}, c2={c2_mean}"
        );
    }

    #[test]
    fn test_spectral_embedding_knn_affinity() {
        let se =
            SpectralEmbedding::new(2).with_affinity(Affinity::NearestNeighbors { n_neighbors: 3 });
        let x = two_clusters();
        let fitted = se.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (8, 2));
    }

    #[test]
    fn test_spectral_embedding_invalid_n_components_zero() {
        let se = SpectralEmbedding::new(0);
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_n_components_too_large() {
        let se = SpectralEmbedding::new(5); // n_samples = 5, need < 5
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_insufficient_samples() {
        let se = SpectralEmbedding::new(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_knn_n_neighbors_zero() {
        let se =
            SpectralEmbedding::new(1).with_affinity(Affinity::NearestNeighbors { n_neighbors: 0 });
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_getters() {
        let se = SpectralEmbedding::new(3).with_affinity(Affinity::RBF { gamma: 0.5 });
        assert_eq!(se.n_components(), 3);
        assert_eq!(se.affinity(), Affinity::RBF { gamma: 0.5 });
    }

    #[test]
    fn test_spectral_embedding_knn_too_many_neighbors() {
        let se = SpectralEmbedding::new(1)
            .with_affinity(Affinity::NearestNeighbors { n_neighbors: 100 });
        let x = simple_data(); // 5 samples
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_negative_gamma() {
        let se = SpectralEmbedding::new(1).with_affinity(Affinity::RBF { gamma: -1.0 });
        let x = simple_data();
        assert!(se.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spectral_embedding_larger_dataset() {
        let n = 20;
        let d = 3;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i * d + j) as f64 / (n * d) as f64;
            }
        }
        let se = SpectralEmbedding::new(2);
        let fitted = se.fit(&data, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
    }
}
