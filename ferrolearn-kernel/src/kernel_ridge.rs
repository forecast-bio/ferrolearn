//! Kernel Ridge Regression.
//!
//! [`KernelRidge`] combines Ridge regression with the kernel trick, allowing
//! nonlinear regression via the dual formulation:
//!
//! ```text
//! (K + alpha * I) @ dual_coef = y
//! ```
//!
//! where `K` is the kernel matrix of the training data. Prediction for new
//! data uses:
//!
//! ```text
//! y_pred = K_new @ dual_coef
//! ```
//!
//! This is equivalent to Ridge regression in the kernel-induced feature space
//! but operates directly in the dual (kernel) space, avoiding explicit
//! feature computation.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_kernel::{KernelRidge, KernelType};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![0.0, 1.0, 4.0, 9.0, 16.0f64]; // y = x^2
//!
//! let model = KernelRidge::<f64>::new()
//!     .with_alpha(1.0)
//!     .with_kernel(KernelType::Rbf)
//!     .with_gamma(0.5);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 5);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::nystroem::{KernelType, compute_kernel_matrix};

/// Kernel Ridge Regression.
///
/// Combines Ridge regression (L2 regularization) with the kernel trick to
/// perform nonlinear regression. The kernel function implicitly maps input
/// features to a high-dimensional space without computing the mapping explicitly.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct KernelRidge<F> {
    /// Regularization strength (default 1.0).
    alpha: F,
    /// Kernel function to use (default RBF).
    kernel: KernelType,
    /// Kernel parameter for RBF/Sigmoid/Polynomial.
    /// Default: `1.0 / n_features` (set at fit time).
    gamma: Option<F>,
    /// Polynomial degree (default 3).
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid (default 0.0).
    coef0: F,
}

impl<F: Float + Send + Sync + 'static> KernelRidge<F> {
    /// Create a new `KernelRidge` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `kernel = Rbf`, `gamma = None` (auto),
    /// `degree = 3`, `coef0 = 0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            kernel: KernelType::Rbf,
            gamma: None,
            degree: 3,
            coef0: F::zero(),
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
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
}

impl<F: Float + Send + Sync + 'static> Default for KernelRidge<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Kernel Ridge Regression model.
///
/// Stores the training data and dual coefficients. Implements [`Predict`]
/// to generate predictions for new data.
#[derive(Debug, Clone)]
pub struct FittedKernelRidge<F> {
    /// Training feature matrix used during fitting.
    x_fit: Array2<F>,
    /// Dual coefficients: solution to `(K + alpha*I) @ dual_coef = y`.
    dual_coef: Array1<F>,
    /// Kernel type.
    kernel: KernelType,
    /// Effective gamma.
    gamma: F,
    /// Polynomial degree.
    degree: usize,
    /// Coefficient for Polynomial/Sigmoid.
    coef0: F,
}

impl<F: Float + Send + Sync + 'static> FittedKernelRidge<F> {
    /// Return the dual coefficients.
    #[must_use]
    pub fn dual_coef(&self) -> &Array1<F> {
        &self.dual_coef
    }

    /// Return a reference to the stored training data.
    #[must_use]
    pub fn x_fit(&self) -> &Array2<F> {
        &self.x_fit
    }
}

/// Solve symmetric positive-definite system `A @ x = b` via Cholesky decomposition.
///
/// Returns `Err` if the matrix is not positive definite.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();

    // Compute lower triangular L such that A = L @ L^T
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "regularized kernel matrix is not positive definite".into(),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L @ z = b
    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * z[j];
        }
        z[i] = sum / l[[i, i]];
    }

    // Backward substitution: L^T @ x = z
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Solve `A @ x = b` via Gaussian elimination with partial pivoting (fallback).
fn gaussian_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap() {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in KernelRidge solve".into(),
            });
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap() {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in KernelRidge back substitution".into(),
            });
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for KernelRidge<F> {
    type Fitted = FittedKernelRidge<F>;
    type Error = FerroError;

    /// Fit the Kernel Ridge Regression model.
    ///
    /// Computes the kernel matrix of the training data, adds regularization,
    /// and solves for the dual coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is negative.
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    /// Returns [`FerroError::NumericalInstability`] if the regularized kernel
    /// matrix is singular.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedKernelRidge<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "KernelRidge::fit".into(),
            });
        }
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
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

        // Compute kernel matrix K
        let mut k =
            compute_kernel_matrix(x, x, self.kernel, effective_gamma, self.degree, self.coef0);

        // Add regularization: K + alpha * I
        for i in 0..n_samples {
            k[[i, i]] = k[[i, i]] + self.alpha;
        }

        // Solve (K + alpha*I) @ dual_coef = y
        let dual_coef = cholesky_solve(&k, y).or_else(|_| gaussian_solve(&k, y))?;

        Ok(FittedKernelRidge {
            x_fit: x.clone(),
            dual_coef,
            kernel: self.kernel,
            gamma: effective_gamma,
            degree: self.degree,
            coef0: self.coef0,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKernelRidge<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for new data.
    ///
    /// Computes the kernel between new data and training data, then
    /// multiplies by the dual coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.x_fit.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_fit.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "KernelRidge::predict feature count must match training data".into(),
            });
        }

        // Compute kernel between new points and training data
        let k_new = compute_kernel_matrix(
            x,
            &self.x_fit,
            self.kernel,
            self.gamma,
            self.degree,
            self.coef0,
        );

        // y_pred = K_new @ dual_coef
        Ok(k_new.dot(&self.dual_coef))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};

    fn make_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        Array2::from_shape_vec((n, d), data).unwrap()
    }

    #[test]
    fn basic_fit_predict() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn constant_target() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_elem(10, 5.0);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.1);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for &p in preds.iter() {
            assert_abs_diff_eq!(p, 5.0, epsilon = 0.5);
        }
    }

    #[test]
    fn linear_kernel_approximates_linear() {
        // With linear kernel, KRR should approximate linear regression
        // Offset x so linear kernel matrix is non-singular
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi + 1.0);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Linear);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..10 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn polynomial_kernel_fits_quadratic() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64 * 0.5).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| xi * xi);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Polynomial)
            .with_gamma(1.0)
            .with_degree(2)
            .with_coef0(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..10 {
            assert!(
                (preds[i] - y[i]).abs() < 2.0,
                "Poly kernel pred {:.2} vs true {:.2}",
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn small_alpha_interpolates() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(1e-6)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 0.1);
        }
    }

    #[test]
    fn large_alpha_shrinks_predictions() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        // With very large alpha, dual_coef ≈ y/alpha → predictions near zero
        let model_large = KernelRidge::<f64>::new()
            .with_alpha(1e6)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted_large = model_large.fit(&x, &y).unwrap();
        let preds_large = fitted_large.predict(&x).unwrap();

        let model_small = KernelRidge::<f64>::new()
            .with_alpha(0.01)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted_small = model_small.fit(&x, &y).unwrap();
        let preds_small = fitted_small.predict(&x).unwrap();

        // Large alpha should produce much smaller magnitude predictions
        let mag_large: f64 = preds_large.iter().map(|&p| p.abs()).sum::<f64>() / 5.0;
        let mag_small: f64 = preds_small.iter().map(|&p| p.abs()).sum::<f64>() / 5.0;
        assert!(
            mag_large < mag_small,
            "Large alpha should shrink predictions: large_mag={mag_large:.4}, small_mag={mag_small:.4}"
        );
    }

    #[test]
    fn rejects_mismatched_y_length() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0f64]; // Wrong length
        let model = KernelRidge::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_negative_alpha() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let model = KernelRidge::<f64>::new().with_alpha(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<f64>::zeros(0);
        let model = KernelRidge::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let fitted = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0)
            .fit(&x, &y)
            .unwrap();
        let x_wrong = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn predict_new_data() {
        let x_train = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y_train: Array1<f64> = x_train.column(0).mapv(|xi| xi.sin());
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((3, 1), vec![0.5, 1.5, 2.5]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds.len(), 3);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn multivariate() {
        let x = make_data(30, 3, 42);
        let y = Array1::from_shape_fn(30, |i| x[[i, 0]] + x[[i, 1]] * x[[i, 2]]);
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.1)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 30);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn f32_support() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let x = Array2::from_shape_vec((10, 1), data).unwrap();
        let y: Array1<f32> = x.column(0).mapv(|xi| xi * xi);
        let model = KernelRidge::<f32>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn dual_coef_accessible() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let fitted = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.dual_coef().len(), 5);
        assert_eq!(fitted.x_fit().nrows(), 5);
    }

    #[test]
    fn builder_chain() {
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.5)
            .with_kernel(KernelType::Polynomial)
            .with_gamma(2.0)
            .with_degree(4)
            .with_coef0(1.0);
        assert_eq!(model.degree, 4);
        assert_eq!(model.kernel, KernelType::Polynomial);
    }

    #[test]
    fn zero_alpha_exact_interpolation() {
        // With alpha=0, should exactly interpolate (modulo numerical issues)
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(0.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(preds[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = array![5.0f64];
        let model = KernelRidge::<f64>::new()
            .with_alpha(1.0)
            .with_kernel(KernelType::Rbf)
            .with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 1);
        assert!(preds[0].is_finite());
    }
}
