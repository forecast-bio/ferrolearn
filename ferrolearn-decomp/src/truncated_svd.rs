//! Truncated Singular Value Decomposition.
//!
//! [`TruncatedSVD`] performs dimensionality reduction by computing an
//! approximate rank-`n_components` SVD of the input matrix using the
//! randomized SVD algorithm (Halko, Martinsson, Tropp, 2011).
//!
//! Unlike [`PCA`](crate::PCA), `TruncatedSVD` does **not** centre the
//! data before decomposition, making it suitable for sparse matrices
//! where centring would destroy sparsity.
//!
//! # Algorithm (Randomized SVD)
//!
//! 1. Generate a random Gaussian matrix `Omega` of shape
//!    `(n_features, n_components + oversampling)`.
//! 2. Form the sample matrix `Y = X @ Omega`.
//! 3. Compute the QR factorisation `Y = Q R`.
//! 4. Form the small matrix `B = Q^T @ X`.
//! 5. Compute the full SVD of `B` (which is small) using the Jacobi method.
//! 6. Recover the top `n_components` left singular vectors, singular values,
//!    and right singular vectors.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::TruncatedSVD;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let svd = TruncatedSVD::<f64>::new(1);
//! let x = array![
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//!     [10.0, 11.0, 12.0],
//! ];
//! let fitted = svd.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// TruncatedSVD (unfitted)
// ---------------------------------------------------------------------------

/// Truncated SVD configuration.
///
/// Holds `n_components` and an optional `random_state` for reproducibility.
/// Calling [`Fit::fit`] computes the randomized SVD and returns a
/// [`FittedTruncatedSVD`].
#[derive(Debug, Clone)]
pub struct TruncatedSVD<F> {
    /// Number of components to retain.
    n_components: usize,
    /// Optional seed for the random number generator.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> TruncatedSVD<F> {
    /// Create a new `TruncatedSVD` that retains `n_components` components.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the random seed for reproducible results.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the number of components this SVD is configured to retain.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

// ---------------------------------------------------------------------------
// FittedTruncatedSVD
// ---------------------------------------------------------------------------

/// A fitted truncated SVD model.
///
/// Created by calling [`Fit::fit`] on a [`TruncatedSVD`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the top components.
#[derive(Debug, Clone)]
pub struct FittedTruncatedSVD<F> {
    /// Right singular vectors (V^T), shape `(n_components, n_features)`.
    /// Each row is a component direction.
    components_: Array2<F>,

    /// Singular values, length `n_components`, sorted descending.
    singular_values_: Array1<F>,

    /// Explained variance per component.
    explained_variance_: Array1<F>,

    /// Explained variance ratio per component.
    explained_variance_ratio_: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedTruncatedSVD<F> {
    /// Components (right singular vectors), shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Singular values, sorted descending.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values_
    }

    /// Explained variance per component.
    #[must_use]
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance_
    }

    /// Explained variance ratio per component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio_
    }
}

// ---------------------------------------------------------------------------
// QR decomposition (Householder)
// ---------------------------------------------------------------------------

/// Compute a thin QR decomposition: `A = Q @ R` where `Q` is `(m, k)`
/// and `R` is `(k, n)` with `k = min(m, n)`.
fn qr_decomposition<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> (Array2<F>, Array2<F>) {
    let (m, n) = a.dim();
    let k = m.min(n);
    let mut q = Array2::<F>::zeros((m, k));
    let mut r = Array2::<F>::zeros((k, n));
    let mut a_work = a.to_owned();

    for j in 0..k {
        // Extract the j-th column from row j onwards.
        let mut col = Array1::<F>::zeros(m - j);
        for i in j..m {
            col[i - j] = a_work[[i, j]];
        }

        // Compute the Householder vector.
        let norm = col
            .iter()
            .map(|&v| v * v)
            .fold(F::zero(), |a, b| a + b)
            .sqrt();
        if norm < F::from(1e-30).unwrap_or(F::epsilon()) {
            // Column is essentially zero.
            for i in j..m {
                q[[i, j]] = F::zero();
            }
            continue;
        }

        let sign = if col[0] >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        let mut v = col.clone();
        v[0] = v[0] + sign * norm;
        let v_norm = v
            .iter()
            .map(|&x| x * x)
            .fold(F::zero(), |a, b| a + b)
            .sqrt();
        if v_norm > F::from(1e-30).unwrap_or(F::epsilon()) {
            v.mapv_inplace(|x| x / v_norm);
        }

        // Apply Householder: A = A - 2 v (v^T A)
        let two = F::from(2.0).unwrap();
        for col_idx in j..n {
            let mut dot = F::zero();
            for row_idx in j..m {
                dot = dot + v[row_idx - j] * a_work[[row_idx, col_idx]];
            }
            for row_idx in j..m {
                a_work[[row_idx, col_idx]] =
                    a_work[[row_idx, col_idx]] - two * v[row_idx - j] * dot;
            }
        }

        // Build Q column by applying reflections.
        // For now, we build Q implicitly by accumulating reflections.
        // Store the reflection vector for later Q construction.
        r[[j, j]] = -sign * norm;
        for col_idx in (j + 1)..n {
            r[[j, col_idx]] = a_work[[j, col_idx]];
        }

        // We will reconstruct Q below.
        let _ = &q; // placeholder
    }

    // Reconstruct Q from A = QR. We have R upper-triangular in a_work.
    // Instead, let's use the modified Gram-Schmidt approach directly.
    // This is simpler and numerically adequate for our use case.
    let mut q2 = Array2::<F>::zeros((m, k));
    let mut r2 = Array2::<F>::zeros((k, n));

    // Modified Gram-Schmidt on columns of A.
    let mut basis = a.to_owned();
    for j in 0..k {
        // Orthogonalise column j against previous columns.
        for i in 0..j {
            let mut dot = F::zero();
            for row in 0..m {
                dot = dot + q2[[row, i]] * basis[[row, j]];
            }
            r2[[i, j]] = dot;
            for row in 0..m {
                basis[[row, j]] = basis[[row, j]] - dot * q2[[row, i]];
            }
        }

        let col_norm = (0..m)
            .map(|row| basis[[row, j]] * basis[[row, j]])
            .fold(F::zero(), |a, b| a + b)
            .sqrt();
        r2[[j, j]] = col_norm;

        if col_norm > F::from(1e-30).unwrap_or(F::epsilon()) {
            for row in 0..m {
                q2[[row, j]] = basis[[row, j]] / col_norm;
            }
        }
    }

    // Fill in the remaining R entries above column k.
    for j in k..n {
        for i in 0..k {
            let mut dot = F::zero();
            for row in 0..m {
                dot = dot + q2[[row, i]] * a[[row, j]];
            }
            r2[[i, j]] = dot;
        }
    }

    (q2, r2)
}

// ---------------------------------------------------------------------------
// SVD of small matrix via eigendecomposition
// ---------------------------------------------------------------------------

/// The result of an SVD: `(U, sigma, V^T)`.
type SvdResult<F> = (Array2<F>, Array1<F>, Array2<F>);

/// Compute a thin SVD of matrix `B` (k x n) via eigendecomposition of `B^T B`.
///
/// Returns `(U, sigma, V^T)` where:
/// - `U` is `(k, min(k, n))` — left singular vectors
/// - `sigma` is `min(k, n)` — singular values
/// - `Vt` is `(min(k, n), n)` — right singular vectors
fn svd_via_eigendecomp<F: Float + Send + Sync + 'static>(
    b: &Array2<F>,
) -> Result<SvdResult<F>, FerroError> {
    let (k, n) = b.dim();
    let rank = k.min(n);

    // Compute B^T B (n x n) or B B^T (k x k), whichever is smaller.
    if k <= n {
        // Eigendecompose B B^T (k x k).
        let bbt = b.dot(&b.t());
        let max_iter = k * k * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_internal(&bbt, max_iter)?;

        // Sort descending.
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b_idx| {
            eigenvalues[b_idx]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = Array1::<F>::zeros(rank);
        let mut u = Array2::<F>::zeros((k, rank));
        let mut vt = Array2::<F>::zeros((rank, n));

        for (out_idx, &eigen_idx) in indices.iter().take(rank).enumerate() {
            let eigval = eigenvalues[eigen_idx];
            let sv = if eigval > F::zero() {
                eigval.sqrt()
            } else {
                F::zero()
            };
            sigma[out_idx] = sv;

            // U column = eigenvector.
            for i in 0..k {
                u[[i, out_idx]] = eigenvectors[[i, eigen_idx]];
            }

            // V = B^T U / sigma.
            if sv > F::from(1e-30).unwrap_or(F::epsilon()) {
                for j in 0..n {
                    let mut val = F::zero();
                    for i in 0..k {
                        val = val + b[[i, j]] * u[[i, out_idx]];
                    }
                    vt[[out_idx, j]] = val / sv;
                }
            }
        }

        Ok((u, sigma, vt))
    } else {
        // Eigendecompose B^T B (n x n).
        let btb = b.t().dot(b);
        let max_iter = n * n * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_internal(&btb, max_iter)?;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b_idx| {
            eigenvalues[b_idx]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = Array1::<F>::zeros(rank);
        let mut u = Array2::<F>::zeros((k, rank));
        let mut vt = Array2::<F>::zeros((rank, n));

        for (out_idx, &eigen_idx) in indices.iter().take(rank).enumerate() {
            let eigval = eigenvalues[eigen_idx];
            let sv = if eigval > F::zero() {
                eigval.sqrt()
            } else {
                F::zero()
            };
            sigma[out_idx] = sv;

            // V column = eigenvector (stored as row in Vt).
            for j in 0..n {
                vt[[out_idx, j]] = eigenvectors[[j, eigen_idx]];
            }

            // U = B V / sigma.
            if sv > F::from(1e-30).unwrap_or(F::epsilon()) {
                for i in 0..k {
                    let mut val = F::zero();
                    for j in 0..n {
                        val = val + b[[i, j]] * vt[[out_idx, j]];
                    }
                    u[[i, out_idx]] = val / sv;
                }
            }
        }

        Ok((u, sigma, vt))
    }
}

/// Jacobi eigendecomposition (identical to the one in pca.rs, but local
/// to avoid circular module dependencies).
fn jacobi_eigen_internal<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        let eigenvalues = Array1::from_vec(vec![a[[0, 0]]]);
        let eigenvectors = Array2::from_shape_vec((1, 1), vec![F::one()]).unwrap();
        return Ok((eigenvalues, eigenvectors));
    }

    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or(F::epsilon());

    for _iteration in 0..max_iter {
        let mut max_off = F::zero();
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = mat[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }

        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or(F::one())
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let mip = mat[[i, p]];
                let miq = mat[[i, q]];
                new_mat[[i, p]] = c * mip - s * miq;
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = s * mip + c * miq;
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }

        new_mat[[p, p]] = c * c * app - F::from(2.0).unwrap() * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + F::from(2.0).unwrap() * s * c * apq + c * c * aqq;
        new_mat[[p, q]] = F::zero();
        new_mat[[q, p]] = F::zero();

        mat = new_mat;

        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }

    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge in TruncatedSVD".into(),
    })
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for TruncatedSVD<F> {
    type Fitted = FittedTruncatedSVD<F>;
    type Error = FerroError;

    /// Fit the truncated SVD using the randomized algorithm.
    ///
    /// The data is **not** centred (unlike PCA).
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features.
    /// - [`FerroError::InsufficientSamples`] if there are zero rows.
    /// - [`FerroError::ConvergenceFailure`] if the internal eigendecomposition
    ///   does not converge.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedTruncatedSVD<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features.min(n_samples) {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_samples, n_features) = {}",
                    self.n_components,
                    n_features.min(n_samples)
                ),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "TruncatedSVD::fit".into(),
            });
        }

        let oversampling = 10usize.min(n_features.saturating_sub(self.n_components));
        let n_random = self.n_components + oversampling;
        let n_random = n_random.min(n_features);

        // Step 1: Generate random Gaussian matrix Omega (n_features x n_random).
        let mut rng: rand::rngs::StdRng = match self.random_state {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::seed_from_u64(0), // deterministic default
        };

        let normal = StandardNormal;
        let mut omega = Array2::<F>::zeros((n_features, n_random));
        for elem in omega.iter_mut() {
            let val: f64 = normal.sample(&mut rng);
            *elem = F::from(val).unwrap_or(F::zero());
        }

        // Step 2: Form Y = X @ Omega (n_samples x n_random).
        let y_mat = x.dot(&omega);

        // Step 3: QR factorisation of Y.
        let (q, _r) = qr_decomposition(&y_mat);

        // Step 4: Form B = Q^T @ X (k x n_features), where k = ncols of Q.
        let b_mat = q.t().dot(x);

        // Step 5: SVD of B.
        let (_u_b, sigma_full, vt_full) = svd_via_eigendecomp(&b_mat)?;

        // Step 6: Take top n_components.
        let n_comp = self.n_components;
        let mut components = Array2::<F>::zeros((n_comp, n_features));
        let mut singular_values = Array1::<F>::zeros(n_comp);

        for i in 0..n_comp {
            if i < sigma_full.len() {
                singular_values[i] = sigma_full[i];
            }
            if i < vt_full.nrows() {
                for j in 0..n_features {
                    components[[i, j]] = vt_full[[i, j]];
                }
            }
        }

        // Compute explained variance.
        // explained_variance = sigma^2 / (n_samples - 1)
        let n_minus_1 = F::from(if n_samples > 1 { n_samples - 1 } else { 1 }).unwrap();
        let explained_variance = singular_values.mapv(|s| s * s / n_minus_1);

        // Compute total variance from X directly: sum of squared Frobenius / (n-1).
        let total_var = {
            let mut ss = F::zero();
            for &v in x.iter() {
                ss = ss + v * v;
            }
            ss / n_minus_1
        };

        let explained_variance_ratio = if total_var > F::zero() {
            explained_variance.mapv(|v| v / total_var)
        } else {
            Array1::zeros(n_comp)
        };

        Ok(FittedTruncatedSVD {
            components_: components,
            singular_values_: singular_values,
            explained_variance_: explained_variance,
            explained_variance_ratio_: explained_variance_ratio,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedTruncatedSVD<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the truncated SVD components: `X @ components^T`.
    ///
    /// Note: the data is **not** centred (unlike PCA).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedTruncatedSVD::transform".into(),
            });
        }
        Ok(x.dot(&self.components_.t()))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for TruncatedSVD<F> {
    /// Fit TruncatedSVD using the pipeline interface.
    ///
    /// The `y` argument is ignored; TruncatedSVD is unsupervised.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedTruncatedSVD<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
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

    #[test]
    fn test_truncated_svd_dimensionality_reduction() {
        let svd = TruncatedSVD::<f64>::new(1);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 1));
    }

    #[test]
    fn test_truncated_svd_correct_dimensions() {
        let svd = TruncatedSVD::<f64>::new(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
        assert_eq!(fitted.singular_values().len(), 2);
        assert_eq!(fitted.explained_variance().len(), 2);
        assert_eq!(fitted.explained_variance_ratio().len(), 2);
    }

    #[test]
    fn test_truncated_svd_singular_values_positive() {
        let svd = TruncatedSVD::<f64>::new(2);
        let x = array![
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 2.0],
            [2.2, 2.9, 3.0],
            [1.9, 2.2, 0.5],
            [3.1, 3.0, 1.5],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        for &s in fitted.singular_values().iter() {
            assert!(s >= 0.0, "singular value should be non-negative, got {s}");
        }
    }

    #[test]
    fn test_truncated_svd_singular_values_sorted_descending() {
        let svd = TruncatedSVD::<f64>::new(2);
        let x = array![
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 2.0],
            [2.2, 2.9, 3.0],
            [1.9, 2.2, 0.5],
            [3.1, 3.0, 1.5],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        let sv = fitted.singular_values();
        for i in 1..sv.len() {
            assert!(
                sv[i - 1] >= sv[i] - 1e-10,
                "singular values not sorted: sv[{}]={} < sv[{}]={}",
                i - 1,
                sv[i - 1],
                i,
                sv[i]
            );
        }
    }

    #[test]
    fn test_truncated_svd_explained_variance_ratio_le_1() {
        let svd = TruncatedSVD::<f64>::new(2);
        let x = array![
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 2.0],
            [2.2, 2.9, 3.0],
            [1.9, 2.2, 0.5],
            [3.1, 3.0, 1.5],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        assert!(ratio_sum <= 1.0 + 1e-6, "ratio sum exceeds 1: {ratio_sum}");
    }

    #[test]
    fn test_truncated_svd_no_centering() {
        // TruncatedSVD should not centre data. If we pass a matrix
        // with a large mean, the result should be different from PCA.
        let svd = TruncatedSVD::<f64>::new(1);
        let x = array![[100.0, 200.0], [101.0, 201.0], [102.0, 202.0],];
        let fitted = svd.fit(&x, &()).unwrap();
        // The singular values should be large (reflecting the mean).
        assert!(
            fitted.singular_values()[0] > 10.0,
            "expected large singular value for uncentred data"
        );
    }

    #[test]
    fn test_truncated_svd_random_state_reproducibility() {
        let svd1 = TruncatedSVD::<f64>::new(1).with_random_state(42);
        let svd2 = TruncatedSVD::<f64>::new(1).with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted1 = svd1.fit(&x, &()).unwrap();
        let fitted2 = svd2.fit(&x, &()).unwrap();

        for (a, b) in fitted1
            .singular_values()
            .iter()
            .zip(fitted2.singular_values().iter())
        {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_truncated_svd_single_component() {
        let svd = TruncatedSVD::<f64>::new(1);
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];
        let fitted = svd.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        assert_eq!(fitted.singular_values().len(), 1);
    }

    #[test]
    fn test_truncated_svd_shape_mismatch() {
        let svd = TruncatedSVD::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = svd.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_truncated_svd_invalid_n_components_zero() {
        let svd = TruncatedSVD::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(svd.fit(&x, &()).is_err());
    }

    #[test]
    fn test_truncated_svd_invalid_n_components_too_large() {
        let svd = TruncatedSVD::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(svd.fit(&x, &()).is_err());
    }

    #[test]
    fn test_truncated_svd_f32() {
        let svd = TruncatedSVD::<f32>::new(1);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = svd.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_truncated_svd_n_components_getter() {
        let svd = TruncatedSVD::<f64>::new(3);
        assert_eq!(svd.n_components(), 3);
        assert!(svd.random_state().is_none());
    }

    #[test]
    fn test_truncated_svd_random_state_getter() {
        let svd = TruncatedSVD::<f64>::new(2).with_random_state(123);
        assert_eq!(svd.random_state(), Some(123));
    }

    #[test]
    fn test_truncated_svd_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        struct SumEstimator;

        impl PipelineEstimator<f64> for SumEstimator {
            fn fit_pipeline(
                &self,
                _x: &Array2<f64>,
                _y: &Array1<f64>,
            ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
                Ok(Box::new(FittedSumEstimator))
            }
        }

        struct FittedSumEstimator;

        impl FittedPipelineEstimator<f64> for FittedSumEstimator {
            fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
                let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
                Ok(Array1::from_vec(sums))
            }
        }

        let pipeline = Pipeline::new()
            .transform_step("svd", Box::new(TruncatedSVD::<f64>::new(2)))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_truncated_svd_components_unit_length() {
        let svd = TruncatedSVD::<f64>::new(2).with_random_state(42);
        let x = array![
            [2.5, 2.4, 1.0],
            [0.5, 0.7, 2.0],
            [2.2, 2.9, 3.0],
            [1.9, 2.2, 0.5],
            [3.1, 3.0, 1.5],
        ];
        let fitted = svd.fit(&x, &()).unwrap();
        let c = fitted.components();
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
    }
}
