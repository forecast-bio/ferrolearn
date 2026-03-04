//! Multidimensional Scaling (MDS).
//!
//! Classical (metric) MDS embeds data into a low-dimensional space such that
//! pairwise distances are preserved as well as possible.
//!
//! # Algorithm
//!
//! 1. Compute the pairwise squared-distance matrix `D^2` (or accept a
//!    precomputed dissimilarity matrix).
//! 2. Double-centre `D^2`:  `B = -0.5 * J D^2 J`  where `J = I - (1/n) 11^T`.
//! 3. Eigendecompose `B` and retain the top `n_components` eigenvectors
//!    scaled by the square root of their eigenvalues.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::{MDS, Dissimilarity};
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let mds = MDS::new(2);
//! let x = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//! ];
//! let fitted = mds.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Dissimilarity type
// ---------------------------------------------------------------------------

/// How the input matrix should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dissimilarity {
    /// The input is a feature matrix; pairwise Euclidean distances will be
    /// computed internally.
    Euclidean,
    /// The input is already a square pairwise-distance matrix.
    Precomputed,
}

// ---------------------------------------------------------------------------
// MDS (unfitted)
// ---------------------------------------------------------------------------

/// Classical Multidimensional Scaling configuration.
///
/// Holds hyperparameters for the MDS algorithm. Call [`Fit::fit`] to compute
/// the embedding and obtain a [`FittedMDS`].
#[derive(Debug, Clone)]
pub struct MDS {
    /// Number of embedding dimensions.
    n_components: usize,
    /// Whether input is a feature matrix or a precomputed distance matrix.
    dissimilarity: Dissimilarity,
}

impl MDS {
    /// Create a new `MDS` that embeds into `n_components` dimensions.
    ///
    /// By default the input is treated as a feature matrix
    /// ([`Dissimilarity::Euclidean`]).
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            dissimilarity: Dissimilarity::Euclidean,
        }
    }

    /// Set the dissimilarity mode.
    #[must_use]
    pub fn with_dissimilarity(mut self, d: Dissimilarity) -> Self {
        self.dissimilarity = d;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured dissimilarity mode.
    #[must_use]
    pub fn dissimilarity(&self) -> Dissimilarity {
        self.dissimilarity
    }
}

// ---------------------------------------------------------------------------
// FittedMDS
// ---------------------------------------------------------------------------

/// A fitted MDS model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on an [`MDS`].
#[derive(Debug, Clone)]
pub struct FittedMDS {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
    /// Kruskal's stress-1 measure of fit quality.
    stress_: f64,
}

impl FittedMDS {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }

    /// Kruskal's stress-1 (lower is better).
    #[must_use]
    pub fn stress(&self) -> f64 {
        self.stress_
    }
}

// ---------------------------------------------------------------------------
// Helper: pairwise squared-Euclidean distance matrix
// ---------------------------------------------------------------------------

/// Compute the pairwise squared-Euclidean distance matrix.
pub(crate) fn pairwise_sq_distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..x.ncols() {
                let diff = x[[i, k]] - x[[j, k]];
                sq += diff * diff;
            }
            d[[i, j]] = sq;
            d[[j, i]] = sq;
        }
    }
    d
}

/// Compute Kruskal's stress-1.
fn kruskal_stress(dist_orig: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
    let n = embedding.nrows();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_orig = dist_orig[[i, j]].sqrt();
            let mut sq = 0.0;
            for k in 0..embedding.ncols() {
                let diff = embedding[[i, k]] - embedding[[j, k]];
                sq += diff * diff;
            }
            let d_embed = sq.sqrt();
            let diff = d_orig - d_embed;
            numerator += diff * diff;
            denominator += d_orig * d_orig;
        }
    }
    if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    }
}

/// Eigendecompose a symmetric matrix using faer's self-adjoint eigen.
pub(crate) fn eigh_faer(a: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("Symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues: Vec<f64> = decomp.S().column_vector().iter().copied().collect();
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Core classical MDS on a squared-distance matrix.
///
/// Returns `(embedding, stress)`.
pub(crate) fn classical_mds(
    sq_dist: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, f64), FerroError> {
    let n = sq_dist.nrows();

    // Double-centre: B = -0.5 * J * D^2 * J, where J = I - (1/n) * 11^T
    let n_f = n as f64;
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut grand_mean = 0.0;

    for i in 0..n {
        for j in 0..n {
            row_means[i] += sq_dist[[i, j]];
            col_means[j] += sq_dist[[i, j]];
            grand_mean += sq_dist[[i, j]];
        }
    }
    for i in 0..n {
        row_means[i] /= n_f;
        col_means[i] /= n_f;
    }
    grand_mean /= n_f * n_f;

    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (sq_dist[[i, j]] - row_means[i] - col_means[j] + grand_mean);
        }
    }

    // Eigendecompose B
    let (eigenvalues, eigenvectors) = eigh_faer(&b)?;

    // Sort eigenvalues descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b_idx| {
        eigenvalues[b_idx]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build embedding: X_k = v_k * sqrt(lambda_k)
    let n_comp = n_components.min(n);
    let mut embedding = Array2::<f64>::zeros((n, n_comp));
    for (k, &idx) in indices.iter().take(n_comp).enumerate() {
        let eigval = eigenvalues[idx].max(0.0);
        let scale = eigval.sqrt();
        for i in 0..n {
            embedding[[i, k]] = eigenvectors[[i, idx]] * scale;
        }
    }

    // Compute stress
    let stress = kruskal_stress(sq_dist, &embedding);

    Ok((embedding, stress))
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for MDS {
    type Fitted = FittedMDS;
    type Error = FerroError;

    /// Fit classical MDS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if `Precomputed` is set but the matrix
    ///   is not square.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedMDS, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        let sq_dist = match self.dissimilarity {
            Dissimilarity::Euclidean => {
                let n_samples = x.nrows();
                if n_samples < 2 {
                    return Err(FerroError::InsufficientSamples {
                        required: 2,
                        actual: n_samples,
                        context: "MDS::fit requires at least 2 samples".into(),
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
                pairwise_sq_distances(x)
            }
            Dissimilarity::Precomputed => {
                if x.nrows() != x.ncols() {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![x.nrows(), x.nrows()],
                        actual: vec![x.nrows(), x.ncols()],
                        context: "MDS with Precomputed dissimilarity requires a square matrix"
                            .into(),
                    });
                }
                let n = x.nrows();
                if n < 2 {
                    return Err(FerroError::InsufficientSamples {
                        required: 2,
                        actual: n,
                        context: "MDS::fit requires at least 2 samples".into(),
                    });
                }
                if self.n_components > n {
                    return Err(FerroError::InvalidParameter {
                        name: "n_components".into(),
                        reason: format!(
                            "n_components ({}) exceeds n_samples ({})",
                            self.n_components, n
                        ),
                    });
                }
                // Input is already distances; square them for classical MDS
                x.mapv(|v| v * v)
            }
        };

        let (embedding, stress) = classical_mds(&sq_dist, self.n_components)?;

        Ok(FittedMDS {
            embedding_: embedding,
            stress_: stress,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Helper: simple 2D dataset.
    fn square_data() -> Array2<f64> {
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],]
    }

    #[test]
    fn test_mds_basic_embedding_shape() {
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (4, 2));
    }

    #[test]
    fn test_mds_1d_embedding() {
        let mds = MDS::new(1);
        let x = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],];
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
    }

    #[test]
    fn test_mds_stress_non_negative() {
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        assert!(fitted.stress() >= 0.0);
    }

    #[test]
    fn test_mds_perfect_embedding_low_stress() {
        // 2D points embedded into 2D should have near-zero stress.
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        assert!(fitted.stress() < 0.1, "stress = {}", fitted.stress());
    }

    #[test]
    fn test_mds_preserves_distances() {
        let mds = MDS::new(2);
        let x = square_data();
        let fitted = mds.fit(&x, &()).unwrap();
        let emb = fitted.embedding();

        // Check that pairwise distances in the embedding approximately
        // match the original pairwise distances.
        let orig = pairwise_sq_distances(&x);
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d_orig = orig[[i, j]].sqrt();
                let mut sq = 0.0;
                for k in 0..emb.ncols() {
                    let diff = emb[[i, k]] - emb[[j, k]];
                    sq += diff * diff;
                }
                let d_emb = sq.sqrt();
                assert_abs_diff_eq!(d_orig, d_emb, epsilon = 0.3);
            }
        }
    }

    #[test]
    fn test_mds_precomputed() {
        // Build a precomputed distance matrix.
        let x = square_data();
        let sq = pairwise_sq_distances(&x);
        let dist = sq.mapv(f64::sqrt);

        let mds = MDS::new(2).with_dissimilarity(Dissimilarity::Precomputed);
        let fitted = mds.fit(&dist, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (4, 2));
    }

    #[test]
    fn test_mds_invalid_n_components_zero() {
        let mds = MDS::new(0);
        let x = square_data();
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_invalid_n_components_too_large() {
        let mds = MDS::new(10);
        let x = square_data(); // 4 samples
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_insufficient_samples() {
        let mds = MDS::new(1);
        let x = array![[1.0, 2.0]]; // 1 sample
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_precomputed_not_square() {
        let mds = MDS::new(1).with_dissimilarity(Dissimilarity::Precomputed);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(mds.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mds_collinear_data() {
        // Points on a line should embed well into 1D.
        let mds = MDS::new(1);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];
        let fitted = mds.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 1);
        // Differences between consecutive embeddings should be roughly equal.
        let emb = fitted.embedding();
        let mut vals: Vec<f64> = (0..5).map(|i| emb[[i, 0]]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let diffs: Vec<f64> = vals.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        for d in &diffs {
            assert_abs_diff_eq!(d, &diffs[0], epsilon = 0.1);
        }
    }

    #[test]
    fn test_mds_getters() {
        let mds = MDS::new(3).with_dissimilarity(Dissimilarity::Precomputed);
        assert_eq!(mds.n_components(), 3);
        assert_eq!(mds.dissimilarity(), Dissimilarity::Precomputed);
    }

    #[test]
    fn test_mds_larger_dataset() {
        let n = 20;
        let d = 5;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i * d + j) as f64 / (n * d) as f64;
            }
        }
        let mds = MDS::new(2);
        let fitted = mds.fit(&data, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
        assert!(fitted.stress() >= 0.0);
    }
}
