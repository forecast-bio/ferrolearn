//! Spectral Clustering via graph Laplacian eigenmaps.
//!
//! This module provides [`SpectralClustering`], a graph-based clustering
//! algorithm that embeds the data into a low-dimensional space defined by the
//! eigenvectors of the graph Laplacian and then clusters the embedded points
//! with K-Means.
//!
//! # Algorithm
//!
//! 1. **Affinity matrix**: compute the pairwise RBF (Gaussian) kernel
//!    `A[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
//! 2. **Normalized graph Laplacian**: `L_sym = D^{-1/2} A D^{-1/2}`, where
//!    `D[i,i] = sum_j A[i,j]` is the degree matrix.
//! 3. **Eigendecomposition**: compute the `n_clusters` eigenvectors of
//!    `L_sym` that correspond to the **largest** eigenvalues (these are the
//!    smoothest graph signals).
//! 4. **Row-normalize** the embedding matrix so each row has unit L2 norm.
//! 5. **K-Means** clustering on the embedded points with `n_init` restarts.
//!
//! Spectral Clustering does **not** implement [`Predict`](ferrolearn_core::Predict)
//! because there is no simple way to embed new points into the learned eigenspace
//! without refitting.
//!
//! # Notes
//!
//! The eigendecomposition is performed in `f64` via `ferrolearn_core::Backend`
//! (`NdarrayFaerBackend`) regardless of the input float type `F`, because
//! `faer`'s solver only supports `f64`.  The results are then cast back to `F`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::SpectralClustering;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0_f64, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     9.0, 9.0,  9.1, 9.0,  9.0, 9.1,
//! ]).unwrap();
//!
//! let model = SpectralClustering::<f64>::new(2).with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```

use crate::kmeans::KMeans;
use ferrolearn_core::NdarrayFaerBackend;
use ferrolearn_core::backend::Backend;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// Spectral Clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce a
/// [`FittedSpectralClustering`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct SpectralClustering<F> {
    /// Number of clusters.
    pub n_clusters: usize,
    /// RBF kernel parameter: `A[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
    /// Defaults to `1.0`.
    pub gamma: F,
    /// Number of K-Means restarts in the final clustering step.
    pub n_init: usize,
    /// Optional random seed for the K-Means restarts.
    pub random_state: Option<u64>,
}

impl<F: Float> SpectralClustering<F> {
    /// Create a new `SpectralClustering` with the given number of clusters.
    ///
    /// Defaults: `gamma = 1.0`, `n_init = 10`, `random_state = None`.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            gamma: F::one(),
            n_init: 10,
            random_state: None,
        }
    }

    /// Set the RBF kernel parameter `gamma`.
    ///
    /// Must be positive.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of K-Means restarts.
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

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Spectral Clustering model.
///
/// Stores the cluster labels for the training data.
///
/// Spectral Clustering does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedSpectralClustering<F> {
    /// Cluster label for each training sample (0-indexed).
    labels_: Array1<usize>,
    /// Phantom for the float type.
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> FittedSpectralClustering<F> {
    /// Return the cluster labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<usize> {
        &self.labels_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the RBF affinity matrix in `f64`.
fn affinity_matrix<F: Float>(x: &Array2<F>, gamma: f64) -> Array2<f64> {
    let n = x.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            1.0_f64
        } else {
            let sq: F = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .fold(F::zero(), |acc, (&a, &b)| acc + (a - b) * (a - b));
            let sq64 = sq.to_f64().unwrap_or(0.0);
            (-gamma * sq64).exp()
        }
    })
}

/// Compute the normalized symmetric graph Laplacian `D^{-1/2} A D^{-1/2}`.
fn normalized_laplacian(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    // Degree vector: d[i] = sum_j A[i,j]
    let d: Vec<f64> = (0..n).map(|i| a.row(i).iter().sum()).collect();

    // D^{-1/2}: avoid division by zero.
    let d_inv_sqrt: Vec<f64> = d
        .iter()
        .map(|&di| if di > 0.0 { 1.0 / di.sqrt() } else { 0.0 })
        .collect();

    // L_sym[i,j] = d_inv_sqrt[i] * A[i,j] * d_inv_sqrt[j]
    Array2::from_shape_fn((n, n), |(i, j)| d_inv_sqrt[i] * a[[i, j]] * d_inv_sqrt[j])
}

/// Extract the top-`k` eigenvectors (by eigenvalue magnitude) of a symmetric
/// matrix.  Returns an `(n, k)` matrix.
fn top_k_eigenvectors(sym: &Array2<f64>, k: usize) -> Result<Array2<f64>, FerroError> {
    let (eigenvalues, eigenvectors) = NdarrayFaerBackend::eigh(sym)?;

    // `eigh` returns eigenvalues in non-decreasing order; the top-k are at the end.
    let n = eigenvalues.len();
    let start = n.saturating_sub(k);

    let n_rows = eigenvectors.nrows();
    let mut result = Array2::<f64>::zeros((n_rows, k));
    for (new_col, old_col) in (start..n).enumerate() {
        for row in 0..n_rows {
            result[[row, new_col]] = eigenvectors[[row, old_col]];
        }
    }

    Ok(result)
}

/// Row-normalize a matrix so each row has unit L2 norm.
/// Rows with zero norm are left as-is.
fn row_normalize(m: &Array2<f64>) -> Array2<f64> {
    let (n, d) = m.dim();
    Array2::from_shape_fn((n, d), |(i, j)| {
        let norm: f64 = m.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            m[[i, j]] / norm
        } else {
            m[[i, j]]
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SpectralClustering<F> {
    type Fitted = FittedSpectralClustering<F>;
    type Error = FerroError;

    /// Fit the Spectral Clustering model to the data.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_clusters == 0`, `gamma <= 0`,
    ///   or `n_init == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < n_clusters`.
    /// - [`FerroError::NumericalInstability`] if the eigendecomposition fails.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSpectralClustering<F>, FerroError> {
        let n_samples = x.nrows();

        // Validate parameters.
        if self.n_clusters == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_clusters".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: 0,
                context: "SpectralClustering requires at least n_clusters samples".into(),
            });
        }
        if n_samples < self.n_clusters {
            return Err(FerroError::InsufficientSamples {
                required: self.n_clusters,
                actual: n_samples,
                context: "SpectralClustering requires at least n_clusters samples".into(),
            });
        }

        let gamma64 = self.gamma.to_f64().unwrap_or(1.0);

        // Step 1: affinity matrix.
        let aff = affinity_matrix(x, gamma64);

        // Step 2: normalized Laplacian.
        let lap = normalized_laplacian(&aff);

        // Step 3: top-k eigenvectors.
        let k = self.n_clusters;
        let embed = top_k_eigenvectors(&lap, k)?;

        // Step 4: row-normalize.
        let embed_norm = row_normalize(&embed);

        // Step 5: K-Means on the embedded points.
        // Convert embedding to F.
        let embed_f: Array2<F> = Array2::from_shape_fn(embed_norm.dim(), |(i, j)| {
            F::from(embed_norm[[i, j]]).unwrap_or_else(F::zero)
        });

        let mut km = KMeans::<F>::new(k).with_n_init(self.n_init);
        if let Some(seed) = self.random_state {
            km = km.with_random_state(seed);
        }

        let fitted_km = km.fit(&embed_f, &())?;
        let labels = fitted_km.predict(&embed_f)?;

        Ok(FittedSpectralClustering {
            labels_: labels,
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
    fn test_two_blobs_two_clusters() {
        let x = two_blobs();
        let model = SpectralClustering::<f64>::new(2)
            .with_gamma(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 10);

        // Points 0-4 should share a label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[0], labels[4]);

        // Points 5-9 should share a different label.
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[5], labels[7]);
        assert_eq!(labels[5], labels[8]);
        assert_eq!(labels[5], labels[9]);

        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn test_labels_length_matches_n_samples() {
        let x = two_blobs();
        let fitted = SpectralClustering::<f64>::new(2)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), x.nrows());
    }

    #[test]
    fn test_labels_in_valid_range() {
        let x = two_blobs();
        let k = 2usize;
        let fitted = SpectralClustering::<f64>::new(k)
            .with_random_state(1)
            .fit(&x, &())
            .unwrap();
        for &l in fitted.labels() {
            assert!(l < k, "label {l} >= n_clusters {k}");
        }
    }

    #[test]
    fn test_single_cluster() {
        let x = two_blobs();
        let fitted = SpectralClustering::<f64>::new(1)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        for &l in fitted.labels() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_invalid_n_clusters_zero() {
        let x = two_blobs();
        let result = SpectralClustering::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_gamma_zero() {
        let x = two_blobs();
        let result = SpectralClustering::<f64>::new(2)
            .with_gamma(0.0)
            .fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_gamma_negative() {
        let x = two_blobs();
        let result = SpectralClustering::<f64>::new(2)
            .with_gamma(-1.0)
            .fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = SpectralClustering::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let result = SpectralClustering::<f64>::new(3).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_n_clusters_equals_n_samples() {
        // k == n should be valid (each point its own cluster).
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0]).unwrap();
        let fitted = SpectralClustering::<f64>::new(3)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), 3);
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

        let fitted = SpectralClustering::<f32>::new(2)
            .with_gamma(0.1)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.labels().len(), 6);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let x = two_blobs();
        let model = SpectralClustering::<f64>::new(2)
            .with_gamma(0.1)
            .with_random_state(7);

        let fitted1 = model.fit(&x, &()).unwrap();
        let fitted2 = model.fit(&x, &()).unwrap();
        assert_eq!(fitted1.labels(), fitted2.labels());
    }
}
