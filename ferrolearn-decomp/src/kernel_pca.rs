//! Kernel Principal Component Analysis (Kernel PCA).
//!
//! [`KernelPCA`] performs non-linear dimensionality reduction by first mapping
//! data into a higher-dimensional (possibly infinite-dimensional) feature space
//! via a kernel function, then performing standard PCA in that space.
//!
//! # Kernels
//!
//! - **Linear**: `K(x, y) = x . y` (equivalent to standard PCA)
//! - **RBF** (Gaussian): `K(x, y) = exp(-gamma * ||x - y||^2)`
//! - **Polynomial**: `K(x, y) = (gamma * x . y + coef0)^degree`
//! - **Sigmoid**: `K(x, y) = tanh(gamma * x . y + coef0)`
//!
//! # Algorithm
//!
//! 1. Compute the kernel matrix `K` of shape `(n_samples, n_samples)`.
//! 2. Centre `K` in feature space: `K_c = K - 1_n K - K 1_n + 1_n K 1_n`
//!    where `1_n` is the `(n, n)` matrix with all entries `1/n`.
//! 3. Eigendecompose `K_c` using the Jacobi iterative method.
//! 4. Sort eigenvalues descending and retain the top `n_components`.
//! 5. Scale eigenvectors by `1 / sqrt(eigenvalue)`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::{KernelPCA, Kernel};
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let kpca = KernelPCA::<f64>::new(2).with_kernel(Kernel::RBF);
//! let x = array![
//!     [1.0, 2.0],
//!     [3.0, 4.0],
//!     [5.0, 6.0],
//!     [7.0, 8.0],
//!     [9.0, 10.0],
//! ];
//! let fitted = kpca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Kernel type
// ---------------------------------------------------------------------------

/// The kernel function for Kernel PCA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Linear kernel: `K(x, y) = x . y`.
    Linear,
    /// RBF (Gaussian) kernel: `K(x, y) = exp(-gamma * ||x-y||^2)`.
    RBF,
    /// Polynomial kernel: `K(x, y) = (gamma * x . y + coef0)^degree`.
    Polynomial,
    /// Sigmoid kernel: `K(x, y) = tanh(gamma * x . y + coef0)`.
    Sigmoid,
}

// ---------------------------------------------------------------------------
// KernelPCA (unfitted)
// ---------------------------------------------------------------------------

/// Kernel PCA configuration.
///
/// Holds hyperparameters for the kernel PCA decomposition. Calling
/// [`Fit::fit`] computes the kernel eigendecomposition and returns a
/// [`FittedKernelPCA`] that can project new data via [`Transform::transform`].
#[derive(Debug, Clone)]
pub struct KernelPCA<F> {
    /// Number of components to retain.
    n_components: usize,
    /// The kernel function.
    kernel: Kernel,
    /// Kernel coefficient. Defaults to `1.0 / n_features` for RBF.
    gamma: Option<f64>,
    /// Degree for polynomial kernel.
    degree: usize,
    /// Independent term for polynomial and sigmoid kernels.
    coef0: f64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> KernelPCA<F> {
    /// Create a new `KernelPCA` that retains `n_components` components.
    ///
    /// Defaults: kernel=`Linear`, gamma=`None` (auto: `1/n_features`),
    /// degree=3, coef0=0.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            kernel: Kernel::Linear,
            gamma: None,
            degree: 3,
            coef0: 0.0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the kernel function.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the gamma parameter for RBF, polynomial, and sigmoid kernels.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the degree for the polynomial kernel.
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the independent term for polynomial and sigmoid kernels.
    #[must_use]
    pub fn with_coef0(mut self, coef0: f64) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured kernel.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Return the configured gamma, if any.
    #[must_use]
    pub fn gamma(&self) -> Option<f64> {
        self.gamma
    }

    /// Return the configured polynomial degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the configured coef0.
    #[must_use]
    pub fn coef0(&self) -> f64 {
        self.coef0
    }
}

// ---------------------------------------------------------------------------
// FittedKernelPCA
// ---------------------------------------------------------------------------

/// A fitted Kernel PCA model holding learned eigendecomposition.
///
/// Created by calling [`Fit::fit`] on a [`KernelPCA`]. Implements
/// [`Transform<Array2<F>>`] to project new data.
#[derive(Debug, Clone)]
pub struct FittedKernelPCA<F> {
    /// Scaled eigenvectors (alphas), shape `(n_samples, n_components)`.
    /// Each column is an eigenvector scaled by `1 / sqrt(eigenvalue)`.
    alphas_: Array2<F>,

    /// Eigenvalues corresponding to each component (sorted descending).
    eigenvalues_: Array1<F>,

    /// Training data, kept for computing kernel with new data.
    x_fit_: Array2<F>,

    /// Column means of the training kernel matrix, shape `(n_samples,)`.
    /// Used for centring the kernel of new data.
    k_fit_col_means_: Array1<F>,

    /// Grand mean of the training kernel matrix.
    k_fit_grand_mean_: F,

    /// Kernel configuration.
    kernel: Kernel,
    /// Effective gamma used.
    gamma: f64,
    /// Polynomial degree.
    degree: usize,
    /// Independent term.
    coef0: f64,
}

impl<F: Float + Send + Sync + 'static> FittedKernelPCA<F> {
    /// Scaled eigenvectors (alphas), shape `(n_samples, n_components)`.
    #[must_use]
    pub fn alphas(&self) -> &Array2<F> {
        &self.alphas_
    }

    /// Eigenvalues corresponding to each component.
    #[must_use]
    pub fn eigenvalues(&self) -> &Array1<F> {
        &self.eigenvalues_
    }
}

// ---------------------------------------------------------------------------
// Kernel computation helpers
// ---------------------------------------------------------------------------

/// Compute the kernel value between two vectors.
fn kernel_value<F: Float>(
    x: &[F],
    y: &[F],
    kernel: Kernel,
    gamma: f64,
    degree: usize,
    coef0: f64,
) -> F {
    let gamma_f = F::from(gamma).unwrap();
    let coef0_f = F::from(coef0).unwrap();

    match kernel {
        Kernel::Linear => {
            let mut dot = F::zero();
            for (&a, &b) in x.iter().zip(y.iter()) {
                dot = dot + a * b;
            }
            dot
        }
        Kernel::RBF => {
            let mut sq_dist = F::zero();
            for (&a, &b) in x.iter().zip(y.iter()) {
                let diff = a - b;
                sq_dist = sq_dist + diff * diff;
            }
            (-gamma_f * sq_dist).exp()
        }
        Kernel::Polynomial => {
            let mut dot = F::zero();
            for (&a, &b) in x.iter().zip(y.iter()) {
                dot = dot + a * b;
            }
            let base = gamma_f * dot + coef0_f;
            let mut result = F::one();
            for _ in 0..degree {
                result = result * base;
            }
            result
        }
        Kernel::Sigmoid => {
            let mut dot = F::zero();
            for (&a, &b) in x.iter().zip(y.iter()) {
                dot = dot + a * b;
            }
            (gamma_f * dot + coef0_f).tanh()
        }
    }
}

/// Compute the kernel matrix between rows of X1 and rows of X2.
fn compute_kernel_matrix<F: Float>(
    x1: &Array2<F>,
    x2: &Array2<F>,
    kernel: Kernel,
    gamma: f64,
    degree: usize,
    coef0: f64,
) -> Array2<F> {
    let n1 = x1.nrows();
    let n2 = x2.nrows();
    let mut k = Array2::<F>::zeros((n1, n2));

    for i in 0..n1 {
        let row_i: Vec<F> = x1.row(i).to_vec();
        for j in 0..n2 {
            let row_j: Vec<F> = x2.row(j).to_vec();
            k[[i, j]] = kernel_value(&row_i, &row_j, kernel, gamma, degree, coef0);
        }
    }

    k
}

/// Centre a kernel matrix in feature space.
///
/// `K_c = K - 1_n K - K 1_n + 1_n K 1_n`
/// where `1_n` is `(n, n)` with all entries `1/n`.
fn centre_kernel_matrix<F: Float>(k: &mut Array2<F>) {
    let n = k.nrows();
    let n_f = F::from(n).unwrap();

    // Compute column means.
    let mut col_means = Array1::<F>::zeros(n);
    for j in 0..n {
        let mut sum = F::zero();
        for i in 0..n {
            sum = sum + k[[i, j]];
        }
        col_means[j] = sum / n_f;
    }

    // Compute grand mean.
    let grand_mean = col_means.iter().copied().fold(F::zero(), |a, b| a + b) / n_f;

    // Centre: K[i,j] = K[i,j] - col_mean[j] - row_mean[i] + grand_mean
    // For a symmetric matrix, col_mean == row_mean.
    for i in 0..n {
        for j in 0..n {
            k[[i, j]] = k[[i, j]] - col_means[i] - col_means[j] + grand_mean;
        }
    }
}

/// Jacobi eigendecomposition for symmetric matrices (local copy).
fn jacobi_eigen_symmetric<F: Float + Send + Sync + 'static>(
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
        message: "Jacobi eigendecomposition did not converge in KernelPCA".into(),
    })
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KernelPCA<F> {
    type Fitted = FittedKernelPCA<F>;
    type Error = FerroError;

    /// Fit Kernel PCA by computing the kernel matrix, centring it in
    /// feature space, and eigendecomposing.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of samples.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ConvergenceFailure`] if the Jacobi eigendecomposition
    ///   does not converge.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedKernelPCA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "KernelPCA::fit requires at least 2 samples".into(),
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

        // Determine effective gamma.
        let effective_gamma = self.gamma.unwrap_or(1.0 / n_features as f64);

        // Step 1: Compute kernel matrix.
        let mut k =
            compute_kernel_matrix(x, x, self.kernel, effective_gamma, self.degree, self.coef0);

        // Save column means and grand mean before centring (needed for transform).
        let n_f = F::from(n_samples).unwrap();
        let mut k_col_means = Array1::<F>::zeros(n_samples);
        for j in 0..n_samples {
            let mut sum = F::zero();
            for i in 0..n_samples {
                sum = sum + k[[i, j]];
            }
            k_col_means[j] = sum / n_f;
        }
        let k_grand_mean = k_col_means.iter().copied().fold(F::zero(), |a, b| a + b) / n_f;

        // Step 2: Centre kernel matrix.
        centre_kernel_matrix(&mut k);

        // Step 3: Eigendecompose.
        let max_iter = n_samples * n_samples * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&k, max_iter)?;

        // Step 4: Sort descending, pick top n_components.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_comp = self.n_components;
        let mut alphas = Array2::<F>::zeros((n_samples, n_comp));
        let mut top_eigenvalues = Array1::<F>::zeros(n_comp);

        for (k_idx, &eigen_idx) in indices.iter().take(n_comp).enumerate() {
            let eigval = eigenvalues[eigen_idx];
            let eigval_clamped = if eigval > F::zero() {
                eigval
            } else {
                F::zero()
            };
            top_eigenvalues[k_idx] = eigval_clamped;

            // Scale eigenvector by 1/sqrt(eigenvalue).
            let scale = if eigval_clamped > F::from(1e-12).unwrap_or(F::epsilon()) {
                F::one() / eigval_clamped.sqrt()
            } else {
                F::zero()
            };

            for i in 0..n_samples {
                alphas[[i, k_idx]] = eigenvectors[[i, eigen_idx]] * scale;
            }
        }

        Ok(FittedKernelPCA {
            alphas_: alphas,
            eigenvalues_: top_eigenvalues,
            x_fit_: x.to_owned(),
            k_fit_col_means_: k_col_means,
            k_fit_grand_mean_: k_grand_mean,
            kernel: self.kernel,
            gamma: effective_gamma,
            degree: self.degree,
            coef0: self.coef0,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedKernelPCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project new data onto the learned kernel principal components.
    ///
    /// Computes the kernel between the new data and the training data,
    /// centres it appropriately, then projects using the learned eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the number seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.x_fit_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedKernelPCA::transform".into(),
            });
        }

        let n_test = x.nrows();
        let n_train = self.x_fit_.nrows();
        let n_f = F::from(n_train).unwrap();

        // Compute kernel matrix between test and training data.
        let k_test = compute_kernel_matrix(
            x,
            &self.x_fit_,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        );

        // Centre the test kernel matrix.
        // K_test_centered[i,j] = K_test[i,j] - mean_train_col[j]
        //                        - mean_test_row[i] + grand_mean_train
        // where mean_test_row[i] = (1/n_train) * sum_j K_test[i,j]

        let mut k_centered = Array2::<F>::zeros((n_test, n_train));
        for i in 0..n_test {
            // Row mean of the test kernel row.
            let mut row_mean = F::zero();
            for j in 0..n_train {
                row_mean = row_mean + k_test[[i, j]];
            }
            row_mean = row_mean / n_f;

            for j in 0..n_train {
                k_centered[[i, j]] =
                    k_test[[i, j]] - self.k_fit_col_means_[j] - row_mean + self.k_fit_grand_mean_;
            }
        }

        // Project: X_new = K_centered @ alphas
        Ok(k_centered.dot(&self.alphas_))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for KernelPCA<F> {
    /// Fit KernelPCA using the pipeline interface.
    ///
    /// The `y` argument is ignored; Kernel PCA is unsupervised.
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedKernelPCA<F> {
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

    /// Helper: create a dataset with some non-linear structure.
    fn circle_dataset() -> Array2<f64> {
        // Points roughly on two concentric circles.
        array![
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0],
        ]
    }

    /// Helper: create a simple linear dataset.
    fn linear_dataset() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    }

    #[test]
    fn test_kernel_pca_linear_basic() {
        let kpca = KernelPCA::<f64>::new(2).with_kernel(Kernel::Linear);
        let x = linear_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (5, 2));
    }

    #[test]
    fn test_kernel_pca_rbf_basic() {
        let kpca = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.5);
        let x = circle_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (8, 2));
    }

    #[test]
    fn test_kernel_pca_polynomial_basic() {
        let kpca = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::Polynomial)
            .with_degree(2)
            .with_gamma(1.0)
            .with_coef0(1.0);
        let x = circle_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (8, 2));
    }

    #[test]
    fn test_kernel_pca_sigmoid_basic() {
        let kpca = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::Sigmoid)
            .with_gamma(0.01)
            .with_coef0(0.0);
        let x = linear_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (5, 2));
    }

    #[test]
    fn test_kernel_pca_eigenvalues_non_negative() {
        let kpca = KernelPCA::<f64>::new(3)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.1);
        let x = circle_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        for &ev in fitted.eigenvalues().iter() {
            assert!(ev >= 0.0, "eigenvalue should be non-negative, got {ev}");
        }
    }

    #[test]
    fn test_kernel_pca_eigenvalues_sorted_descending() {
        let kpca = KernelPCA::<f64>::new(3)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.1);
        let x = circle_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let ev = fitted.eigenvalues();
        for i in 1..ev.len() {
            assert!(
                ev[i - 1] >= ev[i] - 1e-10,
                "eigenvalues not sorted: ev[{}]={} < ev[{}]={}",
                i - 1,
                ev[i - 1],
                i,
                ev[i]
            );
        }
    }

    #[test]
    fn test_kernel_pca_single_component() {
        let kpca = KernelPCA::<f64>::new(1)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.5);
        let x = circle_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        assert_eq!(fitted.alphas().ncols(), 1);
        assert_eq!(fitted.eigenvalues().len(), 1);
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_kernel_pca_invalid_n_components_zero() {
        let kpca = KernelPCA::<f64>::new(0);
        let x = linear_dataset();
        assert!(kpca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kernel_pca_invalid_n_components_too_large() {
        let kpca = KernelPCA::<f64>::new(20);
        let x = linear_dataset(); // 5 samples
        assert!(kpca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kernel_pca_insufficient_samples() {
        let kpca = KernelPCA::<f64>::new(1);
        let x = array![[1.0, 2.0]]; // only 1 sample
        assert!(kpca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_kernel_pca_shape_mismatch_transform() {
        let kpca = KernelPCA::<f64>::new(1).with_kernel(Kernel::Linear);
        let x = linear_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0]]; // 2 features instead of 3
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_kernel_pca_transform_new_data() {
        let kpca = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.5);
        let x_train = circle_dataset();
        let fitted = kpca.fit(&x_train, &()).unwrap();
        let x_test = array![[1.5, 0.5], [-0.5, 1.5]];
        let projected = fitted.transform(&x_test).unwrap();
        assert_eq!(projected.dim(), (2, 2));
    }

    #[test]
    fn test_kernel_pca_auto_gamma() {
        // When gamma is not set, it should default to 1/n_features.
        let kpca = KernelPCA::<f64>::new(2).with_kernel(Kernel::RBF);
        let x = linear_dataset(); // 3 features, so gamma = 1/3
        let fitted = kpca.fit(&x, &()).unwrap();
        // Just verify it ran without error.
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (5, 2));
    }

    #[test]
    fn test_kernel_pca_getters() {
        let kpca = KernelPCA::<f64>::new(3)
            .with_kernel(Kernel::Polynomial)
            .with_gamma(0.5)
            .with_degree(4)
            .with_coef0(2.0);
        assert_eq!(kpca.n_components(), 3);
        assert_eq!(kpca.kernel(), Kernel::Polynomial);
        assert_eq!(kpca.gamma(), Some(0.5));
        assert_eq!(kpca.degree(), 4);
        assert_abs_diff_eq!(kpca.coef0(), 2.0);
    }

    #[test]
    fn test_kernel_pca_f32() {
        let kpca = KernelPCA::<f32>::new(1).with_kernel(Kernel::Linear);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_kernel_pca_linear_resembles_pca() {
        // Linear kernel PCA should produce results similar to standard PCA
        // (up to sign and scale).
        let kpca = KernelPCA::<f64>::new(1).with_kernel(Kernel::Linear);
        let x = linear_dataset();
        let fitted = kpca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        // The projection should be 1-dimensional and the values should
        // be linearly spaced (since the data lies on a line).
        assert_eq!(projected.ncols(), 1);
        // Check that differences between consecutive projections are roughly equal.
        let diffs: Vec<f64> = (1..projected.nrows())
            .map(|i| (projected[[i, 0]] - projected[[i - 1, 0]]).abs())
            .collect();
        for d in &diffs {
            assert_abs_diff_eq!(d, &diffs[0], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kernel_pca_pipeline_integration() {
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
            .transform_step(
                "kpca",
                Box::new(
                    KernelPCA::<f64>::new(2)
                        .with_kernel(Kernel::RBF)
                        .with_gamma(0.5),
                ),
            )
            .estimator_step("sum", Box::new(SumEstimator));

        let x = circle_dataset();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_kernel_pca_max_components_equals_samples() {
        let kpca = KernelPCA::<f64>::new(5).with_kernel(Kernel::Linear);
        let x = linear_dataset(); // 5 samples
        let fitted = kpca.fit(&x, &()).unwrap();
        assert_eq!(fitted.eigenvalues().len(), 5);
    }

    #[test]
    fn test_kernel_pca_rbf_sensitivity_to_gamma() {
        // Different gamma values should produce different projections.
        let kpca_small = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.01);
        let kpca_large = KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::RBF)
            .with_gamma(10.0);
        let x = circle_dataset();
        let fitted_small = kpca_small.fit(&x, &()).unwrap();
        let fitted_large = kpca_large.fit(&x, &()).unwrap();
        let proj_small = fitted_small.transform(&x).unwrap();
        let proj_large = fitted_large.transform(&x).unwrap();
        // The projections should differ.
        let mut diff_sum = 0.0;
        for (a, b) in proj_small.iter().zip(proj_large.iter()) {
            diff_sum += (a - b).abs();
        }
        assert!(
            diff_sum > 1e-6,
            "different gamma should produce different projections"
        );
    }
}
