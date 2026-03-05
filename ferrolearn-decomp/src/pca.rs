//! Principal Component Analysis (PCA).
//!
//! PCA performs linear dimensionality reduction by projecting data onto
//! the directions of maximum variance (principal components). The input
//! data is first centred (mean-subtracted), then the covariance matrix
//! is eigendecomposed to find the top `n_components` directions.
//!
//! # Algorithm
//!
//! 1. Compute the per-feature mean and centre the data.
//! 2. Compute the covariance matrix `C = X_centered^T X_centered / (n - 1)`.
//! 3. Eigendecompose `C` using faer's optimised self-adjoint eigensolver
//!    (for f64/f32), with a Jacobi fallback for other float types.
//! 4. Sort eigenvalues in descending order and retain the top `n_components`.
//! 5. Store the corresponding eigenvectors as rows of `components_`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::PCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let pca = PCA::<f64>::new(1);
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let fitted = pca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::any::TypeId;

// ---------------------------------------------------------------------------
// PCA (unfitted)
// ---------------------------------------------------------------------------

/// Principal Component Analysis configuration.
///
/// Holds the `n_components` hyperparameter. Calling [`Fit::fit`] centres
/// the data, computes the eigendecomposition of the covariance matrix,
/// and returns a [`FittedPCA`] that can project new data.
#[derive(Debug, Clone)]
pub struct PCA<F> {
    /// The number of principal components to retain.
    n_components: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PCA<F> {
    /// Create a new `PCA` that retains `n_components` principal components.
    ///
    /// # Panics
    ///
    /// Does not panic. Validation of `n_components` against the data
    /// dimensions happens during [`Fit::fit`].
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of components this PCA is configured to retain.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

// ---------------------------------------------------------------------------
// FittedPCA
// ---------------------------------------------------------------------------

/// A fitted PCA model holding the learned principal components and statistics.
///
/// Created by calling [`Fit::fit`] on a [`PCA`]. Implements
/// [`Transform<Array2<F>>`] to project new data, and provides
/// [`inverse_transform`](FittedPCA::inverse_transform) to reconstruct
/// approximate original data.
#[derive(Debug, Clone)]
pub struct FittedPCA<F> {
    /// Principal component directions, shape `(n_components, n_features)`.
    /// Each row is a unit eigenvector of the covariance matrix.
    components_: Array2<F>,

    /// Variance explained by each component (eigenvalues of the covariance
    /// matrix, sorted descending).
    explained_variance_: Array1<F>,

    /// Ratio of variance explained by each component to total variance.
    explained_variance_ratio_: Array1<F>,

    /// Per-feature mean computed during fitting, used for centring.
    mean_: Array1<F>,

    /// Singular values corresponding to each component.
    singular_values_: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedPCA<F> {
    /// Principal components, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Explained variance per component (eigenvalues).
    #[must_use]
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance_
    }

    /// Explained variance ratio per component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio_
    }

    /// Per-feature mean learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean_
    }

    /// Singular values corresponding to each component.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values_
    }

    /// Reconstruct approximate original data from the reduced representation.
    ///
    /// Computes `X_approx = X_reduced @ components + mean`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x_reduced` does not equal `n_components`.
    pub fn inverse_transform(&self, x_reduced: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_components = self.components_.nrows();
        if x_reduced.ncols() != n_components {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x_reduced.nrows(), n_components],
                actual: vec![x_reduced.nrows(), x_reduced.ncols()],
                context: "FittedPCA::inverse_transform".into(),
            });
        }
        // X_approx = X_reduced @ components + mean
        let mut result = x_reduced.dot(&self.components_);
        for mut row in result.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v + m;
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition for symmetric matrices
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using the Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major
/// (column `i` is the eigenvector for `eigenvalues[i]`).
///
/// The eigenvalues are NOT sorted; the caller is responsible for sorting.
fn jacobi_eigen<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    // Initialise V to identity.
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or(F::epsilon());

    for iteration in 0..max_iter {
        // Find the largest off-diagonal element.
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
            // Converged.
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }

        // Compute the Jacobi rotation.
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or(F::one())
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
            // t = sign(tau) / (|tau| + sqrt(1 + tau^2))
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to mat: mat' = G^T mat G
        // Update rows/columns p and q.
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

        // Update eigenvector matrix.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }

        let _ = iteration; // suppress unused warning
    }

    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge".into(),
    })
}

// ---------------------------------------------------------------------------
// faer-accelerated eigendecomposition for f64 and f32
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a symmetric matrix using faer's optimised
/// self-adjoint eigensolver. Returns `(eigenvalues, eigenvectors)` where
/// `eigenvectors` is column-major (column `i` is the eigenvector for
/// `eigenvalues[i]`). Eigenvalues are returned in ascending order.
fn faer_eigen_f64(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Perform eigendecomposition of a symmetric f32 matrix using faer's
/// optimised self-adjoint eigensolver.
fn faer_eigen_f32(a: &Array2<f32>) -> Result<(Array1<f32>, Array2<f32>), FerroError> {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| a[[i, j]]);
    let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        FerroError::NumericalInstability {
            message: format!("faer symmetric eigendecomposition failed: {e:?}"),
        }
    })?;

    let eigenvalues = Array1::from_shape_fn(n, |i| decomp.S().column_vector()[i]);
    let eigenvectors = Array2::from_shape_fn((n, n), |(i, j)| decomp.U()[(i, j)]);

    Ok((eigenvalues, eigenvectors))
}

/// Dispatch eigendecomposition to faer for f64/f32, falling back to
/// Jacobi for other float types.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` is column-major.
/// Eigenvalues are NOT guaranteed to be sorted in any particular order.
fn eigen_dispatch<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_jacobi_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    // SAFETY: We check TypeId at runtime and only reinterpret when the types
    // match. The transmutes are between identical types (Array<f64> -> Array<F>
    // when F == f64, etc.).
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // F is f64 — cast through raw pointers to call the f64 specialisation.
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f64(a_f64)?;
        // Cast back from f64 to F (which is f64).
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&eigenvectors) };
        // Prevent double-free of the originals.
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &Array2<f32> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f32>>()) };
        let (eigenvalues, eigenvectors) = faer_eigen_f32(a_f32)?;
        let eigenvalues_f: Array1<F> =
            unsafe { std::mem::transmute_copy::<Array1<f32>, Array1<F>>(&eigenvalues) };
        let eigenvectors_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f32>, Array2<F>>(&eigenvectors) };
        std::mem::forget(eigenvalues);
        std::mem::forget(eigenvectors);
        Ok((eigenvalues_f, eigenvectors_f))
    } else {
        // Fallback to Jacobi for exotic float types.
        jacobi_eigen(a, max_jacobi_iter)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for PCA<F> {
    type Fitted = FittedPCA<F>;
    type Error = FerroError;

    /// Fit PCA by centring the data and eigendecomposing the covariance matrix.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ConvergenceFailure`] if the Jacobi eigendecomposition
    ///   does not converge.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedPCA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "PCA::fit requires at least 2 samples".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();

        // Step 1: compute mean and centre data.
        let mut mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let sum = col.iter().copied().fold(F::zero(), |a, b| a + b);
            mean[j] = sum / n_f;
        }

        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }

        // Step 2: compute covariance matrix C = X_centered^T @ X_centered / (n-1)
        let n_minus_1 = F::from(n_samples - 1).unwrap();
        let xt = x_centered.t();
        let mut cov = xt.dot(&x_centered);
        cov.mapv_inplace(|v| v / n_minus_1);

        // Step 3: eigendecompose (faer fast-path for f64/f32, Jacobi fallback)
        let max_jacobi_iter = n_features * n_features * 100 + 1000;
        let (eigenvalues, eigenvectors) = eigen_dispatch(&cov, max_jacobi_iter)?;

        // Step 4: sort eigenvalues descending and select top n_components
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_variance = eigenvalues.iter().copied().fold(F::zero(), |a, b| a + b);

        let n_comp = self.n_components;
        let mut components = Array2::<F>::zeros((n_comp, n_features));
        let mut explained_variance = Array1::<F>::zeros(n_comp);
        let mut explained_variance_ratio = Array1::<F>::zeros(n_comp);
        let mut singular_values = Array1::<F>::zeros(n_comp);

        for (k, &idx) in indices.iter().take(n_comp).enumerate() {
            let eigval = eigenvalues[idx];
            // Clamp small negative eigenvalues to zero (numerical noise).
            let eigval_clamped = if eigval < F::zero() {
                F::zero()
            } else {
                eigval
            };
            explained_variance[k] = eigval_clamped;
            explained_variance_ratio[k] = if total_variance > F::zero() {
                eigval_clamped / total_variance
            } else {
                F::zero()
            };
            // singular_value = sqrt(eigenvalue * (n_samples - 1))
            singular_values[k] = (eigval_clamped * n_minus_1).sqrt();

            // The eigenvector is a column of `eigenvectors`; store it as a row
            // of `components_`.
            for j in 0..n_features {
                components[[k, j]] = eigenvectors[[j, idx]];
            }
        }

        Ok(FittedPCA {
            components_: components,
            explained_variance_: explained_variance,
            explained_variance_ratio_: explained_variance_ratio,
            mean_: mean,
            singular_values_: singular_values,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the principal components: `(X - mean) @ components^T`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedPCA::transform".into(),
            });
        }

        // Centre the data.
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v - m;
            }
        }

        // Project: X_centered @ components^T
        Ok(x_centered.dot(&self.components_.t()))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PCA<F> {
    /// Fit PCA using the pipeline interface.
    ///
    /// The `y` argument is ignored; PCA is unsupervised.
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedPCA<F> {
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
    fn test_pca_dimensionality_reduction() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 1));
    }

    #[test]
    fn test_pca_explained_variance_ratio_sums_le_1() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // When n_components == n_features, ratio should sum to ~1.0.
        assert!(ratio_sum <= 1.0 + 1e-10, "ratio sum = {ratio_sum}");
    }

    #[test]
    fn test_pca_explained_variance_ratio_partial() {
        let pca = PCA::<f64>::new(1);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        // With 1 component out of 2, ratio should be strictly less than 1.
        assert!(ratio_sum <= 1.0 + 1e-10);
        assert!(ratio_sum > 0.0);
    }

    #[test]
    fn test_pca_components_orthonormal() {
        let pca = PCA::<f64>::new(2);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let c = fitted.components();

        // Check that each component is unit length.
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
        }

        // Check mutual orthogonality.
        for i in 0..c.nrows() {
            for j in (i + 1)..c.nrows() {
                let dot: f64 = c
                    .row(i)
                    .iter()
                    .zip(c.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_pca_inverse_transform_roundtrip() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // With n_components == n_features, reconstruction should be exact.
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pca_inverse_transform_approx() {
        // With fewer components, reconstruction is lossy but the error
        // should be bounded.
        let pca = PCA::<f64>::new(1);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        // Reconstruction should not be wildly off.
        let total_error: f64 = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let total_var: f64 = {
            let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|&v| (v - mean_x).powi(2)).sum()
        };
        // Relative reconstruction error should be reasonable.
        assert!(
            total_error < total_var,
            "error={total_error}, var={total_var}"
        );
    }

    #[test]
    fn test_pca_n_components_equals_n_features() {
        let pca = PCA::<f64>::new(3);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = pca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        assert_abs_diff_eq!(ratio_sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_pca_single_component() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        assert_eq!(fitted.explained_variance().len(), 1);
    }

    #[test]
    fn test_pca_shape_mismatch_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_shape_mismatch_inverse_transform() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = pca.fit(&x, &()).unwrap();
        // inverse_transform expects 1 column (n_components), not 3
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.inverse_transform(&x_bad).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_zero() {
        let pca = PCA::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_invalid_n_components_too_large() {
        let pca = PCA::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_insufficient_samples() {
        let pca = PCA::<f64>::new(1);
        let x = array![[1.0, 2.0]]; // only 1 sample
        assert!(pca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_pca_explained_variance_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &v in fitted.explained_variance().iter() {
            assert!(v >= 0.0, "negative variance: {v}");
        }
    }

    #[test]
    fn test_pca_singular_values_positive() {
        let pca = PCA::<f64>::new(2);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        for &s in fitted.singular_values().iter() {
            assert!(s >= 0.0, "negative singular value: {s}");
        }
    }

    #[test]
    fn test_pca_f32() {
        let pca = PCA::<f32>::new(1);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = pca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_pca_n_components_getter() {
        let pca = PCA::<f64>::new(3);
        assert_eq!(pca.n_components(), 3);
    }

    #[test]
    fn test_pca_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        // Trivial estimator that sums each row.
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
            .transform_step("pca", Box::new(PCA::<f64>::new(1)))
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
}
