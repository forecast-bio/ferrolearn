//! Kernel functions for Gaussian Process regression and classification.
//!
//! This module provides the [`GPKernel`] trait and standard covariance kernels
//! used in Gaussian Process models. Unlike the NW/local-polynomial kernels in
//! [`crate::kernels`], GP kernels compute *pairwise covariance matrices*
//! between sets of input points.
//!
//! # Available kernels
//!
//! | Kernel | Formula |
//! |--------|---------|
//! | [`RBFKernel`] | `k(x, x') = exp(-||x - x'||^2 / (2 l^2))` |
//! | [`MaternKernel`] | Matern family (nu = 0.5, 1.5, 2.5) |
//! | [`ConstantKernel`] | `k(x, x') = c` |
//! | [`WhiteKernel`] | `k(x, x') = sigma^2 * delta(x, x')` |
//! | [`DotProductKernel`] | `k(x, x') = sigma_0^2 + x . x'` |
//! | [`SumKernel`] | `k = k1 + k2` |
//! | [`ProductKernel`] | `k = k1 * k2` |
//!
//! Kernels can be composed via `+` and `*` operators on `Box<dyn GPKernel<F>>`.

use ndarray::{Array1, Array2};
use num_traits::Float;

/// Trait for covariance kernels used in Gaussian Process models.
///
/// A GP kernel computes the covariance between pairs of input points.
/// Implementations must be thread-safe (`Send + Sync`) and expose their
/// hyperparameters for optimization.
pub trait GPKernel<F: Float>: Send + Sync {
    /// Compute the full kernel matrix `K(X1, X2)` where `K[i,j] = k(x1_i, x2_j)`.
    ///
    /// # Arguments
    ///
    /// * `x1` - First set of points, shape `(n1, d)`.
    /// * `x2` - Second set of points, shape `(n2, d)`.
    ///
    /// # Returns
    ///
    /// Kernel matrix of shape `(n1, n2)`.
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F>;

    /// Compute only the diagonal of `K(X, X)`.
    ///
    /// This is more efficient than computing the full matrix when only
    /// variances are needed (e.g., predictive variance).
    fn diagonal(&self, x: &Array2<F>) -> Array1<F>;

    /// Number of tunable hyperparameters.
    fn n_params(&self) -> usize;

    /// Get the current hyperparameter values (in log space for positive params).
    fn get_params(&self) -> Vec<F>;

    /// Set hyperparameters from a slice (in log space for positive params).
    ///
    /// # Panics
    ///
    /// May panic if `params.len() != self.n_params()`.
    fn set_params(&mut self, params: &[F]);

    /// Clone this kernel into a boxed trait object.
    fn clone_box(&self) -> Box<dyn GPKernel<F>>;
}

impl<F: Float + Send + Sync + 'static> Clone for Box<dyn GPKernel<F>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Squared-distance helper
// ---------------------------------------------------------------------------

/// Compute the squared Euclidean distance matrix between rows of `x1` and `x2`.
fn squared_distances<F: Float>(x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
    let n1 = x1.nrows();
    let n2 = x2.nrows();
    let mut dists = Array2::<F>::zeros((n1, n2));
    for i in 0..n1 {
        for j in 0..n2 {
            let mut sum = F::zero();
            for d in 0..x1.ncols() {
                let diff = x1[[i, d]] - x2[[j, d]];
                sum = sum + diff * diff;
            }
            dists[[i, j]] = sum;
        }
    }
    dists
}

/// Compute the Euclidean distance matrix between rows of `x1` and `x2`.
fn euclidean_distances<F: Float>(x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
    squared_distances(x1, x2).mapv(num_traits::Float::sqrt)
}

// ---------------------------------------------------------------------------
// RBF (Squared Exponential) Kernel
// ---------------------------------------------------------------------------

/// Radial Basis Function (squared exponential) kernel.
///
/// `k(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))`
///
/// This is the most commonly used GP kernel. It produces infinitely
/// differentiable (very smooth) functions.
#[derive(Debug, Clone)]
pub struct RBFKernel<F> {
    /// Length scale parameter. Controls the smoothness of the function.
    pub length_scale: F,
}

impl<F: Float> RBFKernel<F> {
    /// Create a new RBF kernel with the given length scale.
    #[must_use]
    pub fn new(length_scale: F) -> Self {
        Self { length_scale }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for RBFKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let two = F::from(2.0).unwrap();
        let ls2 = self.length_scale * self.length_scale;
        let sq = squared_distances(x1, x2);
        sq.mapv(|d| (-d / (two * ls2)).exp())
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        // k(x, x) = exp(0) = 1 for all x
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.length_scale.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.length_scale = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Matern Kernel
// ---------------------------------------------------------------------------

/// Matern kernel with parameter `nu` controlling smoothness.
///
/// Supported values of `nu`:
/// - `0.5`: Exponential kernel (Ornstein-Uhlenbeck). Produces rough, non-differentiable paths.
/// - `1.5`: Once-differentiable functions. Good default for many applications.
/// - `2.5`: Twice-differentiable functions. Smoother than 1.5, less smooth than RBF.
///
/// For nu = 0.5: `k(x,x') = exp(-r/l)` where `r = ||x - x'||`
/// For nu = 1.5: `k(x,x') = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)`
/// For nu = 2.5: `k(x,x') = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)`
#[derive(Debug, Clone)]
pub struct MaternKernel<F> {
    /// Length scale parameter.
    pub length_scale: F,
    /// Smoothness parameter. Must be one of `0.5`, `1.5`, or `2.5`.
    pub nu: F,
}

impl<F: Float> MaternKernel<F> {
    /// Create a new Matern kernel.
    ///
    /// # Arguments
    ///
    /// * `length_scale` - Length scale parameter (positive).
    /// * `nu` - Smoothness parameter. Must be `0.5`, `1.5`, or `2.5`.
    #[must_use]
    pub fn new(length_scale: F, nu: F) -> Self {
        Self { length_scale, nu }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for MaternKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let dists = euclidean_distances(x1, x2);
        let ls = self.length_scale;
        let half = F::from(0.5).unwrap();
        let one_point_five = F::from(1.5).unwrap();
        let two_point_five = F::from(2.5).unwrap();

        if (self.nu - half).abs() < F::from(1e-8).unwrap() {
            // nu = 0.5: exponential
            dists.mapv(|r| (-r / ls).exp())
        } else if (self.nu - one_point_five).abs() < F::from(1e-8).unwrap() {
            // nu = 1.5
            let sqrt3 = F::from(3.0f64.sqrt()).unwrap();
            dists.mapv(|r| {
                let z = sqrt3 * r / ls;
                (F::one() + z) * (-z).exp()
            })
        } else if (self.nu - two_point_five).abs() < F::from(1e-8).unwrap() {
            // nu = 2.5
            let sqrt5 = F::from(5.0f64.sqrt()).unwrap();
            let five_thirds = F::from(5.0 / 3.0).unwrap();
            dists.mapv(|r| {
                let z = sqrt5 * r / ls;
                let r_over_l = r / ls;
                (F::one() + z + five_thirds * r_over_l * r_over_l) * (-z).exp()
            })
        } else {
            // Fallback: treat as RBF-like for unsupported nu
            let two = F::from(2.0).unwrap();
            let ls2 = ls * ls;
            let sq = squared_distances(x1, x2);
            sq.mapv(|d| (-d / (two * ls2)).exp())
        }
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        // k(x, x) = 1 for all Matern kernels (distance 0)
        Array1::from_elem(x.nrows(), F::one())
    }

    fn n_params(&self) -> usize {
        1 // only length_scale is optimizable; nu is fixed
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.length_scale.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.length_scale = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Constant Kernel
// ---------------------------------------------------------------------------

/// Constant kernel: `k(x, x') = constant_value`.
///
/// Useful as a component in composite kernels. When used as a product with
/// another kernel, it scales the signal variance.
#[derive(Debug, Clone)]
pub struct ConstantKernel<F> {
    /// The constant covariance value.
    pub constant_value: F,
}

impl<F: Float> ConstantKernel<F> {
    /// Create a new constant kernel.
    #[must_use]
    pub fn new(constant_value: F) -> Self {
        Self { constant_value }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for ConstantKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        Array2::from_elem((x1.nrows(), x2.nrows()), self.constant_value)
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), self.constant_value)
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.constant_value.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.constant_value = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// White Kernel
// ---------------------------------------------------------------------------

/// White (noise) kernel: `k(x, x') = noise_level * delta(x, x')`.
///
/// Adds independent identically distributed noise to the diagonal.
/// The covariance is `noise_level` when `x == x'` and zero otherwise.
///
/// In practice, during training the kernel matrix already includes the
/// noise on the diagonal, so this kernel only contributes to the diagonal
/// of `K(X_train, X_train)`.
#[derive(Debug, Clone)]
pub struct WhiteKernel<F> {
    /// Noise variance.
    pub noise_level: F,
}

impl<F: Float> WhiteKernel<F> {
    /// Create a new white noise kernel.
    #[must_use]
    pub fn new(noise_level: F) -> Self {
        Self { noise_level }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for WhiteKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::<F>::zeros((n1, n2));
        // Only add noise where points coincide (same index in training set).
        // For K(X_train, X_train), this means the diagonal.
        if n1 == n2 {
            let eps = F::from(1e-12).unwrap();
            for i in 0..n1 {
                // Check if the rows are identical
                let mut same = true;
                for d in 0..x1.ncols() {
                    if (x1[[i, d]] - x2[[i, d]]).abs() > eps {
                        same = false;
                        break;
                    }
                }
                if same {
                    k[[i, i]] = self.noise_level;
                }
            }
        }
        k
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        Array1::from_elem(x.nrows(), self.noise_level)
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.noise_level.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.noise_level = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Dot Product Kernel
// ---------------------------------------------------------------------------

/// Dot product (linear) kernel: `k(x, x') = sigma_0^2 + x . x'`.
///
/// This kernel is non-stationary (translation-variant). It corresponds
/// to Bayesian linear regression when `sigma_0 = 0`.
#[derive(Debug, Clone)]
pub struct DotProductKernel<F> {
    /// Inhomogeneity parameter. Controls the bias term.
    pub sigma_0: F,
}

impl<F: Float> DotProductKernel<F> {
    /// Create a new dot product kernel.
    #[must_use]
    pub fn new(sigma_0: F) -> Self {
        Self { sigma_0 }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for DotProductKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let s0_sq = self.sigma_0 * self.sigma_0;
        let dot = x1.dot(&x2.t());
        dot.mapv(|v| v + s0_sq)
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let s0_sq = self.sigma_0 * self.sigma_0;
        let n = x.nrows();
        let mut diag = Array1::<F>::zeros(n);
        for i in 0..n {
            let row = x.row(i);
            diag[i] = row.dot(&row) + s0_sq;
        }
        diag
    }

    fn n_params(&self) -> usize {
        1
    }

    fn get_params(&self) -> Vec<F> {
        vec![self.sigma_0.ln()]
    }

    fn set_params(&mut self, params: &[F]) {
        self.sigma_0 = params[0].exp();
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Sum Kernel
// ---------------------------------------------------------------------------

/// Sum of two kernels: `k(x, x') = k1(x, x') + k2(x, x')`.
///
/// Used to combine independent signal components. For example,
/// `ConstantKernel(1.0) * RBFKernel(1.0) + WhiteKernel(0.1)` models
/// a smooth signal plus independent noise.
pub struct SumKernel<F: Float + Send + Sync + 'static> {
    /// First kernel.
    pub k1: Box<dyn GPKernel<F>>,
    /// Second kernel.
    pub k2: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for SumKernel<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumKernel").finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> SumKernel<F> {
    /// Create a sum of two kernels.
    pub fn new(k1: Box<dyn GPKernel<F>>, k2: Box<dyn GPKernel<F>>) -> Self {
        Self { k1, k2 }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for SumKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 + m2
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 + d2
    }

    fn n_params(&self) -> usize {
        self.k1.n_params() + self.k2.n_params()
    }

    fn get_params(&self) -> Vec<F> {
        let mut params = self.k1.get_params();
        params.extend(self.k2.get_params());
        params
    }

    fn set_params(&mut self, params: &[F]) {
        let n1 = self.k1.n_params();
        self.k1.set_params(&params[..n1]);
        self.k2.set_params(&params[n1..]);
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(SumKernel {
            k1: self.k1.clone_box(),
            k2: self.k2.clone_box(),
        })
    }
}

// ---------------------------------------------------------------------------
// Product Kernel
// ---------------------------------------------------------------------------

/// Product of two kernels: `k(x, x') = k1(x, x') * k2(x, x')`.
///
/// Used to scale kernels. For example, `ConstantKernel(c) * RBFKernel(l)`
/// produces an RBF kernel with signal variance `c`.
pub struct ProductKernel<F: Float + Send + Sync + 'static> {
    /// First kernel.
    pub k1: Box<dyn GPKernel<F>>,
    /// Second kernel.
    pub k2: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for ProductKernel<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductKernel").finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> ProductKernel<F> {
    /// Create a product of two kernels.
    pub fn new(k1: Box<dyn GPKernel<F>>, k2: Box<dyn GPKernel<F>>) -> Self {
        Self { k1, k2 }
    }
}

impl<F: Float + Send + Sync + 'static> GPKernel<F> for ProductKernel<F> {
    fn compute(&self, x1: &Array2<F>, x2: &Array2<F>) -> Array2<F> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 * m2
    }

    fn diagonal(&self, x: &Array2<F>) -> Array1<F> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 * d2
    }

    fn n_params(&self) -> usize {
        self.k1.n_params() + self.k2.n_params()
    }

    fn get_params(&self) -> Vec<F> {
        let mut params = self.k1.get_params();
        params.extend(self.k2.get_params());
        params
    }

    fn set_params(&mut self, params: &[F]) {
        let n1 = self.k1.n_params();
        self.k1.set_params(&params[..n1]);
        self.k2.set_params(&params[n1..]);
    }

    fn clone_box(&self) -> Box<dyn GPKernel<F>> {
        Box::new(ProductKernel {
            k1: self.k1.clone_box(),
            k2: self.k2.clone_box(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn make_x1() -> Array2<f64> {
        Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap()
    }

    fn make_x2() -> Array2<f64> {
        Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap()
    }

    // --- RBF ---

    #[test]
    fn rbf_self_covariance_is_one() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            assert_abs_diff_eq!(km[[i, i]], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rbf_symmetry() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km[[i, j]], km[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn rbf_diagonal() {
        let k = RBFKernel::new(1.0);
        let x = make_x1();
        let diag = k.diagonal(&x);
        assert_eq!(diag.len(), 3);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn rbf_length_scale_effect() {
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let k_short = RBFKernel::new(0.5);
        let k_long = RBFKernel::new(5.0);

        let v_short = k_short.compute(&x1, &x2)[[0, 0]];
        let v_long = k_long.compute(&x1, &x2)[[0, 0]];

        // Longer length scale => higher correlation at same distance
        assert!(v_long > v_short);
    }

    #[test]
    fn rbf_params_roundtrip() {
        let mut k = RBFKernel::new(2.0);
        let params = k.get_params();
        assert_eq!(params.len(), 1);
        assert_abs_diff_eq!(params[0], 2.0f64.ln(), epsilon = 1e-12);

        k.set_params(&[1.0f64.ln()]);
        assert_abs_diff_eq!(k.length_scale, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn rbf_cross_covariance() {
        let k = RBFKernel::new(1.0);
        let x1 = make_x1();
        let x2 = make_x2();
        let km = k.compute(&x1, &x2);
        assert_eq!(km.dim(), (3, 2));
        // k(origin, origin) = 1
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    // --- Matern ---

    #[test]
    fn matern_05_is_exponential() {
        let k = MaternKernel::new(1.0, 0.5);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let v = k.compute(&x1, &x2)[[0, 0]];
        assert_abs_diff_eq!(v, (-1.0f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn matern_15_at_zero() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn matern_25_at_zero() {
        let k = MaternKernel::new(1.0, 2.5);
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn matern_symmetry() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km[[i, j]], km[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn matern_diagonal() {
        let k = MaternKernel::new(1.0, 2.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn matern_params_roundtrip() {
        let mut k = MaternKernel::new(3.0, 1.5);
        let params = k.get_params();
        assert_eq!(params.len(), 1);
        assert_abs_diff_eq!(params[0], 3.0f64.ln(), epsilon = 1e-12);

        k.set_params(&[0.5f64.ln()]);
        assert_abs_diff_eq!(k.length_scale, 0.5, epsilon = 1e-12);
    }

    // --- Constant ---

    #[test]
    fn constant_kernel() {
        let k = ConstantKernel::new(3.0);
        let x1 = make_x1();
        let x2 = make_x2();
        let km = k.compute(&x1, &x2);
        for &v in &km {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn constant_diagonal() {
        let k = ConstantKernel::new(2.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 2.5, epsilon = 1e-12);
        }
    }

    // --- White ---

    #[test]
    fn white_kernel_diagonal_only() {
        let k = WhiteKernel::new(0.1);
        let x = make_x1();
        let km = k.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                if i == j {
                    assert_abs_diff_eq!(km[[i, j]], 0.1, epsilon = 1e-12);
                } else {
                    assert_abs_diff_eq!(km[[i, j]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn white_kernel_cross_different_sizes() {
        let k = WhiteKernel::new(0.1);
        let x1 = make_x1(); // 3 rows
        let x2 = make_x2(); // 2 rows
        // Cross-covariance: different sizes, so all zeros
        let km = k.compute(&x1, &x2);
        assert_eq!(km.dim(), (3, 2));
        for &v in &km {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn white_kernel_cross_same_size() {
        // When x1 == x2 (same points), diagonal gets noise
        let k = WhiteKernel::new(0.1);
        let x = make_x1(); // 3 rows
        let km = k.compute(&x, &x);
        for i in 0..3 {
            assert_abs_diff_eq!(km[[i, i]], 0.1, epsilon = 1e-12);
        }
    }

    #[test]
    fn white_diagonal() {
        let k = WhiteKernel::new(0.5);
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 0.5, epsilon = 1e-12);
        }
    }

    // --- DotProduct ---

    #[test]
    fn dot_product_at_origin() {
        let k = DotProductKernel::new(1.0);
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let km = k.compute(&x, &x);
        // sigma_0^2 + 0 = 1
        assert_abs_diff_eq!(km[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn dot_product_linear() {
        let k = DotProductKernel::new(0.0);
        let x1 = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![4.0]).unwrap();
        let km = k.compute(&x1, &x2);
        assert_abs_diff_eq!(km[[0, 0]], 12.0, epsilon = 1e-12);
    }

    #[test]
    fn dot_product_diagonal() {
        let k = DotProductKernel::new(1.0);
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let diag = k.diagonal(&x);
        // [1^2 + 2^2 + 1, 3^2 + 4^2 + 1] = [6, 26]
        assert_abs_diff_eq!(diag[0], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 26.0, epsilon = 1e-12);
    }

    // --- Sum ---

    #[test]
    fn sum_kernel() {
        let k = SumKernel::new(
            Box::new(ConstantKernel::new(1.0)),
            Box::new(ConstantKernel::new(2.0)),
        );
        let x = make_x1();
        let km = k.compute(&x, &x);
        for &v in &km {
            assert_abs_diff_eq!(v, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn sum_kernel_params() {
        let k = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.1)),
        );
        assert_eq!(k.n_params(), 2);
        let params = k.get_params();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn sum_kernel_diagonal() {
        let k = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.5)),
        );
        let x = make_x1();
        let diag = k.diagonal(&x);
        for &d in &diag {
            assert_abs_diff_eq!(d, 1.5, epsilon = 1e-12);
        }
    }

    // --- Product ---

    #[test]
    fn product_kernel() {
        let k = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(ConstantKernel::new(3.0)),
        );
        let x = make_x1();
        let km = k.compute(&x, &x);
        for &v in &km {
            assert_abs_diff_eq!(v, 6.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn product_kernel_params() {
        let k = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        assert_eq!(k.n_params(), 2);
    }

    #[test]
    fn product_kernel_scaling() {
        // ConstantKernel(c) * RBFKernel(l) should scale RBF output by c
        let c = 5.0;
        let l = 1.0;
        let k_rbf = RBFKernel::new(l);
        let k_scaled = ProductKernel::new(
            Box::new(ConstantKernel::new(c)),
            Box::new(RBFKernel::new(l)),
        );
        let x1 = make_x1();
        let x2 = make_x2();
        let km_rbf = k_rbf.compute(&x1, &x2);
        let km_scaled = k_scaled.compute(&x1, &x2);
        for i in 0..x1.nrows() {
            for j in 0..x2.nrows() {
                assert_abs_diff_eq!(km_scaled[[i, j]], c * km_rbf[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // --- Clone ---

    #[test]
    fn clone_box_preserves_params() {
        let k: Box<dyn GPKernel<f64>> = Box::new(RBFKernel::new(2.5));
        let k2 = k.clone_box();
        let x = make_x1();
        let km1 = k.compute(&x, &x);
        let km2 = k2.compute(&x, &x);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert_abs_diff_eq!(km1[[i, j]], km2[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // --- f32 support ---

    #[test]
    fn rbf_f32() {
        let k = RBFKernel::new(1.0f32);
        let x = Array2::from_shape_vec((2, 1), vec![0.0f32, 1.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0f32, epsilon = 1e-6);
        assert!(km[[0, 1]] > 0.0f32);
        assert!(km[[0, 1]] < 1.0f32);
    }

    #[test]
    fn matern_f32() {
        let k = MaternKernel::new(1.0f32, 1.5);
        let x = Array2::from_shape_vec((2, 1), vec![0.0f32, 1.0]).unwrap();
        let km = k.compute(&x, &x);
        assert_abs_diff_eq!(km[[0, 0]], 1.0f32, epsilon = 1e-5);
    }

    // --- Positive semi-definiteness ---

    #[test]
    fn rbf_positive_semidefinite() {
        // For a valid kernel, x^T K x >= 0 for all x
        let k = RBFKernel::new(1.0);
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let km = k.compute(&x, &x);
        let v = array![1.0, -1.0, 0.5, 0.3, -0.2];
        let vtk = km.dot(&v);
        let quad_form = v.dot(&vtk);
        assert!(
            quad_form >= -1e-10,
            "Quadratic form should be non-negative, got {quad_form}"
        );
    }

    #[test]
    fn matern_15_positive_semidefinite() {
        let k = MaternKernel::new(1.0, 1.5);
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let km = k.compute(&x, &x);
        let v = array![1.0, -1.0, 0.5, 0.3];
        let vtk = km.dot(&v);
        let quad_form = v.dot(&vtk);
        assert!(
            quad_form >= -1e-10,
            "Quadratic form should be non-negative, got {quad_form}"
        );
    }
}
