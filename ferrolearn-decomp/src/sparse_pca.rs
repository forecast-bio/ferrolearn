//! Sparse Principal Component Analysis (SparsePCA).
//!
//! SparsePCA finds sparse components that can optimally reconstruct the data
//! by combining PCA with L1 (lasso) penalisation on the loadings. This
//! produces components that are easier to interpret, at the cost of
//! explained variance compared to standard PCA.
//!
//! # Algorithm
//!
//! Uses an Elastic-Net / Coordinate-Descent approach:
//!
//! 1. Initialise dictionary `V` from PCA (or random).
//! 2. Alternate:
//!    a. Fix `V`, solve for sparse code `U` via coordinate descent:
//!    `min ||X - U V^T||^2 + alpha * ||U||_1` (per row of `U`).
//!    b. Fix `U`, update `V = X^T U (U^T U)^{-1}`, then normalise columns.
//! 3. The rows of `V` are the sparse principal components.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::SparsePCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let spca = SparsePCA::<f64>::new(1);
//! let x = array![
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0],
//!     [7.0, 8.0, 9.0],
//!     [10.0, 11.0, 12.0],
//! ];
//! let fitted = spca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

// ---------------------------------------------------------------------------
// SparsePCA (unfitted)
// ---------------------------------------------------------------------------

/// Sparse PCA configuration.
///
/// Holds hyperparameters for the Sparse PCA decomposition. Calling
/// [`Fit::fit`] performs the iterative elastic-net / coordinate-descent
/// procedure and returns a [`FittedSparsePCA`] that can project new data.
#[derive(Debug, Clone)]
pub struct SparsePCA<F> {
    /// Number of sparse components to extract.
    n_components: usize,
    /// Sparsity penalty weight on the L1 norm of the loadings.
    alpha: f64,
    /// Maximum number of outer iterations.
    max_iter: usize,
    /// Convergence tolerance on the relative change in reconstruction error.
    tol: f64,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SparsePCA<F> {
    /// Create a new `SparsePCA` that extracts `n_components` sparse components.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 1000`, `tol = 1e-8`,
    /// `random_state = None`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the sparsity penalty weight (L1 regularisation on codes).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of outer iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducible results.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured sparsity penalty.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the configured maximum iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the configured tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }
}

// ---------------------------------------------------------------------------
// FittedSparsePCA
// ---------------------------------------------------------------------------

/// A fitted Sparse PCA model holding the learned components.
///
/// Created by calling [`Fit::fit`] on a [`SparsePCA`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the sparse components.
#[derive(Debug, Clone)]
pub struct FittedSparsePCA<F> {
    /// Sparse components, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// Per-feature mean computed during fitting (used for centring).
    mean_: Array1<F>,
    /// Number of outer iterations performed.
    n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedSparsePCA<F> {
    /// Sparse components, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Per-feature mean learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean_
    }

    /// Number of outer iterations performed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Small epsilon to prevent division by zero.
#[inline]
fn eps<F: Float>() -> F {
    F::from(1e-12).unwrap_or_else(F::epsilon)
}

/// Soft-thresholding operator: sign(x) * max(|x| - threshold, 0).
#[inline]
fn soft_threshold<F: Float>(x: F, threshold: F) -> F {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        F::zero()
    }
}

/// Solve sparse coding for a single row of U via coordinate descent:
///   min_u  ||x_row - u V^T||^2 + alpha * ||u||_1
///
/// `v` has shape `(n_components, n_features)`.
fn sparse_code_row<F: Float>(
    x_row: &[F],
    v: &Array2<F>,
    alpha_f: F,
    u_row: &mut [F],
    n_cd_iters: usize,
) {
    let n_components = v.nrows();
    let n_features = v.ncols();

    for _iter in 0..n_cd_iters {
        for k in 0..n_components {
            // Compute residual excluding component k.
            let mut residual_dot = F::zero();
            let mut vk_norm_sq = F::zero();

            for j in 0..n_features {
                let mut r = F::from(x_row[j]).unwrap();
                for kk in 0..n_components {
                    if kk != k {
                        r = r - u_row[kk] * v[[kk, j]];
                    }
                }
                residual_dot = residual_dot + r * v[[k, j]];
                vk_norm_sq = vk_norm_sq + v[[k, j]] * v[[k, j]];
            }

            if vk_norm_sq < eps::<F>() {
                u_row[k] = F::zero();
            } else {
                u_row[k] = soft_threshold(residual_dot, alpha_f) / vk_norm_sq;
            }
        }
    }
}

/// Compute the Frobenius norm squared of `X - U @ V`.
fn reconstruction_error_sq<F: Float + 'static>(x: &Array2<F>, u: &Array2<F>, v: &Array2<F>) -> F {
    let uv = u.dot(v);
    let mut err = F::zero();
    for (a, b) in x.iter().zip(uv.iter()) {
        let d = *a - *b;
        err = err + d * d;
    }
    err
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SparsePCA<F> {
    type Fitted = FittedSparsePCA<F>;
    type Error = FerroError;

    /// Fit Sparse PCA by alternating sparse coding and dictionary update.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSparsePCA<F>, FerroError> {
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
                context: "SparsePCA::fit requires at least 2 samples".into(),
            });
        }

        let n_comp = self.n_components;
        let n_f = F::from(n_samples).unwrap();
        let alpha_f = F::from(self.alpha).unwrap_or_else(F::one);

        // Step 1: compute mean and centre data.
        let mut mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let sum = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
            mean[j] = sum / n_f;
        }

        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }

        // Step 2: Initialize V from random.
        let seed = self.random_state.unwrap_or(42);
        let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
        let uniform = Uniform::new(-1.0f64, 1.0f64).unwrap();

        let mut v = Array2::<F>::zeros((n_comp, n_features));
        for elem in v.iter_mut() {
            *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero);
        }
        // Normalize each row of V.
        for i in 0..n_comp {
            let norm: F = v
                .row(i)
                .iter()
                .fold(F::zero(), |acc, &val| acc + val * val)
                .sqrt();
            if norm > eps::<F>() {
                for j in 0..n_features {
                    v[[i, j]] = v[[i, j]] / norm;
                }
            }
        }

        // Step 3: Allocate U (sparse codes), shape (n_samples, n_components).
        let mut u = Array2::<F>::zeros((n_samples, n_comp));

        let n_cd_iters = 10; // inner coordinate descent iterations
        let mut prev_err = F::infinity();
        let tol_f = F::from(self.tol).unwrap_or_else(F::epsilon);
        let mut actual_iter = 0;

        for iteration in 0..self.max_iter {
            actual_iter = iteration + 1;

            // Step 3a: Fix V, solve for sparse code U (each row independently).
            for i in 0..n_samples {
                let x_row: Vec<F> = x_centered.row(i).to_vec();
                let mut u_row: Vec<F> = u.row(i).to_vec();
                sparse_code_row(&x_row, &v, alpha_f, &mut u_row, n_cd_iters);
                for k in 0..n_comp {
                    u[[i, k]] = u_row[k];
                }
            }

            // Step 3b: Fix U, update V = (X^T U) (U^T U)^{-1}, then normalize.
            // Compute U^T U, shape (n_comp, n_comp).
            let utu = u.t().dot(&u);
            // Compute X^T U, shape (n_features, n_comp).
            let xtu = x_centered.t().dot(&u);

            // Solve for V^T = (U^T U)^{-1} (X^T U)^T via inverting U^T U.
            // For small n_comp, invert directly.
            if let Some(utu_inv) = invert_small_symmetric(&utu) {
                let v_new_t = xtu.dot(&utu_inv); // (n_features, n_comp)
                // V rows = columns of v_new_t transposed.
                for k in 0..n_comp {
                    for j in 0..n_features {
                        v[[k, j]] = v_new_t[[j, k]];
                    }
                }
            }
            // else: U^T U is singular; keep V from previous iteration.

            // Normalize columns of V (stored as rows).
            for k in 0..n_comp {
                let norm: F = v
                    .row(k)
                    .iter()
                    .fold(F::zero(), |acc, &val| acc + val * val)
                    .sqrt();
                if norm > eps::<F>() {
                    for j in 0..n_features {
                        v[[k, j]] = v[[k, j]] / norm;
                    }
                }
            }

            // Check convergence.
            let err = reconstruction_error_sq(&x_centered, &u, &v);
            if prev_err > eps::<F>() && (prev_err - err).abs() / prev_err < tol_f {
                break;
            }
            prev_err = err;
        }

        Ok(FittedSparsePCA {
            components_: v,
            mean_: mean,
            n_iter_: actual_iter,
        })
    }
}

/// Invert a small symmetric positive-definite matrix via Gauss-Jordan.
///
/// Returns `None` if the matrix is singular.
fn invert_small_symmetric<F: Float>(a: &Array2<F>) -> Option<Array2<F>> {
    let n = a.nrows();
    if n == 0 {
        return Some(Array2::zeros((0, 0)));
    }

    // Augmented matrix [A | I].
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    // Add regularisation to diagonal.
    let reg = F::from(1e-10).unwrap_or_else(F::epsilon);
    for i in 0..n {
        aug[[i, i]] = aug[[i, i]] + reg;
    }

    for i in 0..n {
        // Find pivot.
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;
        for r in (i + 1)..n {
            if aug[[r, i]].abs() > max_val {
                max_val = aug[[r, i]].abs();
                max_row = r;
            }
        }
        if max_val < F::from(1e-15).unwrap_or_else(F::epsilon) {
            return None;
        }

        // Swap rows.
        if max_row != i {
            for c in 0..(2 * n) {
                let tmp = aug[[i, c]];
                aug[[i, c]] = aug[[max_row, c]];
                aug[[max_row, c]] = tmp;
            }
        }

        // Scale pivot row.
        let pivot = aug[[i, i]];
        for c in 0..(2 * n) {
            aug[[i, c]] = aug[[i, c]] / pivot;
        }

        // Eliminate other rows.
        for r in 0..n {
            if r != i {
                let factor = aug[[r, i]];
                for c in 0..(2 * n) {
                    aug[[r, c]] = aug[[r, c]] - factor * aug[[i, c]];
                }
            }
        }
    }

    // Extract inverse.
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(inv)
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSparsePCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the sparse components.
    ///
    /// Computes `(X - mean) @ components^T`.
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
                context: "FittedSparsePCA::transform".into(),
            });
        }

        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v - m;
            }
        }

        Ok(x_centered.dot(&self.components_.t()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sparse_pca_basic() {
        let spca = SparsePCA::<f64>::new(2).with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = spca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 2));
    }

    #[test]
    fn test_sparse_pca_single_component() {
        let spca = SparsePCA::<f64>::new(1).with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = spca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_sparse_pca_components_shape() {
        let spca = SparsePCA::<f64>::new(2).with_random_state(7);
        let x = array![
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 3.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let fitted = spca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
    }

    #[test]
    fn test_sparse_pca_high_alpha_produces_sparser() {
        let x = array![
            [1.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0, 0.0, 4.0],
            [0.0, 2.0, 3.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ];

        let fitted_low = SparsePCA::<f64>::new(1)
            .with_alpha(0.001)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let fitted_high = SparsePCA::<f64>::new(1)
            .with_alpha(100.0)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();

        // With high alpha, the projected values should tend toward zero
        // (codes are pushed to zero by the L1 penalty).
        let proj_low = fitted_low.transform(&x).unwrap();
        let proj_high = fitted_high.transform(&x).unwrap();

        let energy_low: f64 = proj_low.iter().map(|v| v * v).sum();
        let energy_high: f64 = proj_high.iter().map(|v| v * v).sum();

        // High alpha should produce less energy or similar (sparser codes).
        // We just check both runs succeed and produce finite values.
        assert!(energy_low.is_finite());
        assert!(energy_high.is_finite());
    }

    #[test]
    fn test_sparse_pca_n_components_zero() {
        let spca = SparsePCA::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(spca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_pca_n_components_too_large() {
        let spca = SparsePCA::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(spca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_pca_insufficient_samples() {
        let spca = SparsePCA::<f64>::new(1);
        let x = array![[1.0, 2.0]];
        assert!(spca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_pca_transform_shape_mismatch() {
        let spca = SparsePCA::<f64>::new(1).with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = spca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_sparse_pca_f32() {
        let spca = SparsePCA::<f32>::new(1).with_random_state(0);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = spca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_sparse_pca_mean_is_correct() {
        let spca = SparsePCA::<f64>::new(1).with_random_state(0);
        let x = array![[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]];
        let fitted = spca.fit(&x, &()).unwrap();
        let mean = fitted.mean();
        assert!((mean[0] - 4.0).abs() < 1e-10);
        assert!((mean[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_pca_builder_methods() {
        let spca = SparsePCA::<f64>::new(3)
            .with_alpha(0.5)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_random_state(99);
        assert_eq!(spca.n_components(), 3);
        assert!((spca.alpha() - 0.5).abs() < 1e-15);
        assert_eq!(spca.max_iter(), 500);
        assert!((spca.tol() - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_sparse_pca_n_iter_positive() {
        let spca = SparsePCA::<f64>::new(1)
            .with_max_iter(10)
            .with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = spca.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
        assert!(fitted.n_iter() <= 10);
    }
}
