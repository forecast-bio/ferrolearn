//! Low-rank kernel approximation using the Nystroem method.
//!
//! [`Nystroem`] approximates a kernel map by sampling a subset of the training
//! data and using the kernel matrix between sampled and all points to construct
//! a low-rank feature embedding. The method works with any kernel function
//! (RBF, polynomial, linear, sigmoid).
//!
//! The approximation is:
//!
//! ```text
//! K ~ K_nq @ K_qq^{-1/2} @ (K_nq @ K_qq^{-1/2})^T
//! ```
//!
//! where `K_qq` is the kernel matrix of the sampled basis points and `K_nq`
//! is the kernel between all points and the basis.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_kernel::{Nystroem, KernelType};
//! use ferrolearn_core::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 * 0.1).collect()).unwrap();
//! let nystroem = Nystroem::<f64>::new()
//!     .with_kernel(KernelType::Rbf)
//!     .with_n_components(10);
//! let fitted = nystroem.fit(&x, &()).unwrap();
//! let z = fitted.transform(&x).unwrap();
//! assert_eq!(z.ncols(), 10);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::seq::SliceRandom;

/// Kernel type for the Nystroem approximation and Kernel Ridge regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Radial basis function: `exp(-gamma * ||x - y||^2)`.
    Rbf,
    /// Polynomial: `(gamma * <x, y> + coef0)^degree`.
    Polynomial,
    /// Linear: `<x, y>`.
    Linear,
    /// Sigmoid (hyperbolic tangent): `tanh(gamma * <x, y> + coef0)`.
    Sigmoid,
}

/// Low-rank kernel approximation via the Nystroem method.
///
/// Samples a subset of training points and uses the resulting kernel sub-matrix
/// to construct an approximate feature embedding.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Nystroem<F> {
    /// Kernel function to use.
    kernel: KernelType,
    /// Kernel parameter for RBF/Sigmoid/Polynomial. Default: `1.0 / n_features` (set at fit time).
    gamma: Option<F>,
    /// Polynomial degree (default 3).
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid (default 0.0).
    coef0: F,
    /// Number of basis functions / components (default 100).
    n_components: usize,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> Nystroem<F> {
    /// Create a new `Nystroem` with default settings (RBF kernel, 100 components).
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernel: KernelType::Rbf,
            gamma: None,
            degree: 3,
            coef0: F::zero(),
            n_components: 100,
            random_state: None,
        }
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the kernel parameter `gamma`.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the polynomial degree.
    #[must_use]
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the coefficient for Polynomial/Sigmoid kernels.
    #[must_use]
    pub fn with_coef0(mut self, coef0: F) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set the number of basis components.
    #[must_use]
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for Nystroem<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nystroem approximation holding basis points and normalization matrix.
///
/// Created by calling [`Fit::fit`] on a [`Nystroem`].
#[derive(Debug, Clone)]
pub struct FittedNystroem<F> {
    /// Sampled basis points of shape `(n_components, n_features)`.
    basis: Array2<F>,
    /// Normalization matrix: `V @ D^{-1/2}` of shape `(n_components, n_components)`.
    normalization: Array2<F>,
    /// Kernel type.
    kernel: KernelType,
    /// Effective gamma.
    gamma: F,
    /// Polynomial degree.
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid.
    coef0: F,
}

impl<F: Float + Send + Sync + 'static> FittedNystroem<F> {
    /// Return the basis points used for the approximation.
    #[must_use]
    pub fn basis(&self) -> &Array2<F> {
        &self.basis
    }

    /// Return the number of basis components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.basis.nrows()
    }
}

/// Compute a single kernel value between two vectors.
pub(crate) fn kernel_value<F: Float>(
    x: &ndarray::ArrayView1<F>,
    y: &ndarray::ArrayView1<F>,
    kernel: KernelType,
    gamma: F,
    degree: usize,
    coef0: F,
) -> F {
    match kernel {
        KernelType::Rbf => {
            let diff = x.to_owned() - y;
            let sq_dist: F = diff.iter().map(|&d| d * d).fold(F::zero(), |a, b| a + b);
            (-gamma * sq_dist).exp()
        }
        KernelType::Polynomial => {
            let dot: F = x
                .iter()
                .zip(y.iter())
                .map(|(&a, &b)| a * b)
                .fold(F::zero(), |acc, v| acc + v);
            let base = gamma * dot + coef0;
            let mut result = F::one();
            for _ in 0..degree {
                result = result * base;
            }
            result
        }
        KernelType::Linear => x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| a * b)
            .fold(F::zero(), |acc, v| acc + v),
        KernelType::Sigmoid => {
            let dot: F = x
                .iter()
                .zip(y.iter())
                .map(|(&a, &b)| a * b)
                .fold(F::zero(), |acc, v| acc + v);
            (gamma * dot + coef0).tanh()
        }
    }
}

/// Compute the kernel matrix between two sets of points.
pub(crate) fn compute_kernel_matrix<F: Float>(
    a: &Array2<F>,
    b: &Array2<F>,
    kernel: KernelType,
    gamma: F,
    degree: usize,
    coef0: F,
) -> Array2<F> {
    let n_a = a.nrows();
    let n_b = b.nrows();
    let mut k = Array2::zeros((n_a, n_b));
    for i in 0..n_a {
        for j in 0..n_b {
            k[[i, j]] = kernel_value(&a.row(i), &b.row(j), kernel, gamma, degree, coef0);
        }
    }
    k
}

/// Eigendecompose a symmetric matrix using faer.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by descending eigenvalue.
/// The computation is done in f64 for numerical stability; results are
/// converted back to `F`.
///
/// Returns `None` if the eigendecomposition fails.
fn symmetric_eigen<F: Float>(mat: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = mat.nrows();

    // Convert ndarray to faer Mat<f64>
    let faer_mat = faer::Mat::from_fn(n, n, |i, j| mat[[i, j]].to_f64().unwrap());

    // Compute eigendecomposition of the symmetric matrix
    let eigen = faer_mat
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("eigendecomposition failed: {e:?}"),
        })?;
    let s_col = eigen.S().column_vector();
    let u_mat = eigen.U();

    // Extract eigenvalues and sort by descending magnitude
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        s_col[j]
            .partial_cmp(&s_col[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues = Array1::from_shape_fn(n, |i| F::from(s_col[indices[i]]).unwrap());
    let sorted_eigenvectors =
        Array2::from_shape_fn((n, n), |(i, j)| F::from(u_mat[(i, indices[j])]).unwrap());

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for Nystroem<F> {
    type Fitted = FittedNystroem<F>;
    type Error = FerroError;

    /// Fit the Nystroem approximation.
    ///
    /// Samples basis points from the training data, computes their kernel matrix,
    /// and eigendecomposes it to create the normalization matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_components` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    /// Returns [`FerroError::NumericalInstability`] if the basis kernel matrix
    /// is degenerate.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedNystroem<F>, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "Nystroem::fit".into(),
            });
        }

        let n_features = x.ncols();
        let effective_gamma = self.gamma.unwrap_or_else(|| {
            if n_features > 0 {
                F::one() / F::from(n_features).unwrap()
            } else {
                F::one()
            }
        });

        // Select basis points
        let n_basis = self.n_components.min(n_samples);
        let basis = if n_basis >= n_samples {
            // Use all points
            x.clone()
        } else {
            // Random subset
            let mut rng = match self.random_state {
                Some(seed) => rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed),
                None => rand_xoshiro::Xoshiro256PlusPlus::from_os_rng(),
            };
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(n_basis);

            let mut basis_data = Vec::with_capacity(n_basis * n_features);
            for &idx in &indices {
                for j in 0..n_features {
                    basis_data.push(x[[idx, j]]);
                }
            }
            Array2::from_shape_vec((n_basis, n_features), basis_data).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!("failed to create basis matrix: {e}"),
                }
            })?
        };

        // Compute kernel matrix of basis points
        let k_basis = compute_kernel_matrix(
            &basis,
            &basis,
            self.kernel,
            effective_gamma,
            self.degree,
            self.coef0,
        );

        // Eigendecompose K_basis = V @ D @ V^T
        let (eigenvalues, eigenvectors) = symmetric_eigen(&k_basis)?;

        // Compute normalization = V @ D^{-1/2}
        // Only use eigenvalues > epsilon to avoid numerical issues
        let eps = F::from(1e-12).unwrap();
        let mut normalization = Array2::<F>::zeros((n_basis, n_basis));
        for j in 0..n_basis {
            let ev = eigenvalues[j];
            if ev > eps {
                let inv_sqrt = F::one() / ev.sqrt();
                for i in 0..n_basis {
                    normalization[[i, j]] = eigenvectors[[i, j]] * inv_sqrt;
                }
            }
            // Columns corresponding to near-zero eigenvalues remain zero
        }

        Ok(FittedNystroem {
            basis,
            normalization,
            kernel: self.kernel,
            gamma: effective_gamma,
            degree: self.degree,
            coef0: self.coef0,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedNystroem<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data using the Nystroem approximation.
    ///
    /// Computes the kernel between new points and basis points, then applies
    /// the normalization: `Z = K_new @ normalization`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the basis points.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.basis.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.basis.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedNystroem::transform feature count must match fit data".into(),
            });
        }

        // Compute kernel between new points and basis
        let k_new = compute_kernel_matrix(
            x,
            &self.basis,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        );

        // Apply normalization
        let z = k_new.dot(&self.normalization);

        Ok(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    fn make_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        Array2::from_shape_vec((n, d), data).unwrap()
    }

    #[test]
    fn basic_fit_transform_rbf() {
        let x = make_data(30, 4, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_kernel(KernelType::Rbf)
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (30, 10));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn output_shape() {
        let x = make_data(50, 5, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_n_components(20)
            .with_random_state(0);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.nrows(), 50);
        assert_eq!(z.ncols(), 20);
    }

    #[test]
    fn n_components_exceeds_n_samples() {
        // When n_components > n_samples, use all samples
        let x = make_data(10, 3, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_n_components(100)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_components(), 10);
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.ncols(), 10);
    }

    #[test]
    fn kernel_approximation_quality_rbf() {
        let gamma = 0.5;
        // Small dataset so all points are used as basis (exact Nystroem)
        let x = make_data(10, 2, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_kernel(KernelType::Rbf)
            .with_gamma(gamma)
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();

        // Verify the Gram matrix of the embedding approximates the kernel matrix
        // When all points are used as basis, this should be very close.
        // Check that diagonal (self-kernel) is close to 1.0
        for i in 0..x.nrows() {
            let zi = z.row(i);
            let self_k: f64 = zi.dot(&zi);
            // RBF self-kernel is exp(0)=1.0
            assert!(
                (self_k - 1.0).abs() < 0.5,
                "Self-kernel for row {i}: {self_k:.4} (expected ~1.0)"
            );
        }
    }

    #[test]
    fn polynomial_kernel() {
        let x = make_data(20, 3, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_kernel(KernelType::Polynomial)
            .with_gamma(1.0)
            .with_degree(2)
            .with_coef0(1.0)
            .with_n_components(15)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (20, 15));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn linear_kernel() {
        let x = make_data(20, 3, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_kernel(KernelType::Linear)
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (20, 10));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn sigmoid_kernel() {
        let x = make_data(20, 3, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_kernel(KernelType::Sigmoid)
            .with_gamma(0.1)
            .with_coef0(0.0)
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (20, 10));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn reproducible_with_seed() {
        let x = make_data(30, 4, 42);
        let n1 = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(99);
        let n2 = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(99);
        let z1 = n1.fit(&x, &()).unwrap().transform(&x).unwrap();
        let z2 = n2.fit(&x, &()).unwrap().transform(&x).unwrap();
        for (a, b) in z1.iter().zip(z2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let x = make_data(30, 4, 42);
        let z1 = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(1)
            .fit(&x, &())
            .unwrap()
            .transform(&x)
            .unwrap();
        let z2 = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(2)
            .fit(&x, &())
            .unwrap()
            .transform(&x)
            .unwrap();
        let max_diff = (&z1 - &z2)
            .mapv(f64::abs)
            .into_iter()
            .fold(0.0f64, f64::max);
        assert!(max_diff > 0.01);
    }

    #[test]
    fn rejects_zero_components() {
        let x = make_data(10, 3, 42);
        let nystroem = Nystroem::<f64>::new().with_n_components(0);
        assert!(nystroem.fit(&x, &()).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let nystroem = Nystroem::<f64>::new().with_random_state(42);
        assert!(nystroem.fit(&x, &()).is_err());
    }

    #[test]
    fn transform_rejects_wrong_features() {
        let x = make_data(20, 3, 42);
        let fitted = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let x_wrong = make_data(5, 4, 42);
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn transform_new_data() {
        let x_train = make_data(30, 3, 42);
        let x_test = make_data(10, 3, 99);
        let fitted = Nystroem::<f64>::new()
            .with_n_components(15)
            .with_random_state(42)
            .fit(&x_train, &())
            .unwrap();
        let z = fitted.transform(&x_test).unwrap();
        assert_eq!(z.dim(), (10, 15));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn f32_support() {
        let data: Vec<f32> = (0..60).map(|i| i as f32 * 0.1).collect();
        let x = Array2::from_shape_vec((20, 3), data).unwrap();
        let nystroem = Nystroem::<f32>::new()
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (20, 10));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn builder_chain() {
        let n = Nystroem::<f64>::new()
            .with_kernel(KernelType::Polynomial)
            .with_gamma(0.5)
            .with_degree(4)
            .with_coef0(1.0)
            .with_n_components(50)
            .with_random_state(42);
        assert_eq!(n.n_components, 50);
        assert_eq!(n.degree, 4);
        assert_eq!(n.kernel, KernelType::Polynomial);
    }

    #[test]
    fn single_sample() {
        let x = make_data(1, 3, 42);
        let nystroem = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(42);
        let fitted = nystroem.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (1, 1)); // n_components clamped to n_samples=1
    }

    #[test]
    fn default_gamma_scales_with_features() {
        // Default gamma = 1/n_features
        let x = make_data(20, 5, 42);
        let fitted = Nystroem::<f64>::new()
            .with_n_components(10)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        assert_abs_diff_eq!(fitted.gamma, 0.2, epsilon = 1e-15);
    }
}
