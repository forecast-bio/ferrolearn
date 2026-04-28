//! Nu-parameterized Support Vector Machines.
//!
//! This module provides [`NuSVC`] (classification) and [`NuSVR`] (regression),
//! which are nu-parameterized variants of [`SVC`](super::svm::SVC) and
//! [`SVR`](super::svm::SVR). Instead of setting the penalty parameter `C`
//! directly, the user specifies `nu` in `(0, 1]`, which is an upper bound on
//! the fraction of training errors and a lower bound on the fraction of
//! support vectors.
//!
//! # Internals
//!
//! `NuSVC` converts `nu` to an equivalent `C = 1 / (nu * n_samples)` and
//! delegates to [`SVC`](super::svm::SVC). `NuSVR` converts `nu` to
//! `epsilon = 0` and `C = 1 / (nu * n_samples)`, delegating to
//! [`SVR`](super::svm::SVR).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
//! use ferrolearn_linear::svm::{LinearKernel, RbfKernel};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.5, 1.0,  1.0, 1.5,
//!     5.0, 5.0,  5.5, 5.0,  5.0, 5.5,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = NuSVC::<f64, LinearKernel>::new(LinearKernel);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use crate::svm::{FittedSVC, FittedSVR, Kernel, SVC, SVR};

// ---------------------------------------------------------------------------
// NuSVC
// ---------------------------------------------------------------------------

/// Nu-parameterized Support Vector Classifier.
///
/// Instead of specifying `C` directly, the user sets `nu` in `(0, 1]`.
/// Internally, `C` is derived as `1 / (nu * n_samples)`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`LinearKernel`](super::svm::LinearKernel)).
#[derive(Debug, Clone)]
pub struct NuSVC<F, K> {
    /// The nu parameter, in `(0, 1]`. Default: `0.5`.
    pub nu: F,
    /// The kernel function.
    pub kernel: K,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> NuSVC<F, K> {
    /// Create a new `NuSVC` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `nu = 0.5`, `tol = 1e-3`, `max_iter = 10000`, `cache_size = 1024`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            nu: F::from(0.5).unwrap(),
            kernel,
            tol: F::from(1e-3).unwrap(),
            max_iter: 10000,
            cache_size: 1024,
        }
    }

    /// Set the nu parameter.
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the kernel cache size.
    #[must_use]
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }
}

/// Fitted Nu-SVC. Wraps a [`FittedSVC`].
#[derive(Debug, Clone)]
pub struct FittedNuSVC<F, K>(FittedSVC<F, K>);

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<usize>> for NuSVC<F, K>
{
    type Fitted = FittedNuSVC<F, K>;
    type Error = FerroError;

    /// Fit the NuSVC model by converting nu to C and delegating to SVC.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`.
    /// - All errors from [`SVC::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedNuSVC<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NuSVC requires at least one sample".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let c = F::one() / (self.nu * n_f);

        let svc = SVC::new(self.kernel.clone())
            .with_c(c)
            .with_tol(self.tol)
            .with_max_iter(self.max_iter)
            .with_cache_size(self.cache_size);

        let fitted = svc.fit(x, y)?;
        Ok(FittedNuSVC(fitted))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedNuSVC<F, K>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        self.0.predict(x)
    }
}

impl<F: Float, K: Kernel<F>> FittedNuSVC<F, K> {
    /// Compute the raw decision function values for each sample.
    ///
    /// Delegates to the inner [`FittedSVC::decision_function`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the input has no columns.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.0.decision_function(x)
    }
}

// ---------------------------------------------------------------------------
// NuSVR
// ---------------------------------------------------------------------------

/// Nu-parameterized Support Vector Regressor.
///
/// Instead of specifying `C` and `epsilon` directly, the user sets `nu`
/// in `(0, 1]`. Internally, `C = 1 / (nu * n_samples)` and `epsilon = 0`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type.
#[derive(Debug, Clone)]
pub struct NuSVR<F, K> {
    /// The nu parameter, in `(0, 1]`. Default: `0.5`.
    pub nu: F,
    /// The kernel function.
    pub kernel: K,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> NuSVR<F, K> {
    /// Create a new `NuSVR` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `nu = 0.5`, `tol = 1e-3`, `max_iter = 10000`, `cache_size = 1024`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            nu: F::from(0.5).unwrap(),
            kernel,
            tol: F::from(1e-3).unwrap(),
            max_iter: 10000,
            cache_size: 1024,
        }
    }

    /// Set the nu parameter.
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the kernel cache size.
    #[must_use]
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }
}

/// Fitted Nu-SVR. Wraps a [`FittedSVR`].
#[derive(Debug, Clone)]
pub struct FittedNuSVR<F, K>(FittedSVR<F, K>);

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<F>> for NuSVR<F, K>
{
    type Fitted = FittedNuSVR<F, K>;
    type Error = FerroError;

    /// Fit the NuSVR model by converting nu to C and delegating to SVR.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`.
    /// - All errors from [`SVR::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedNuSVR<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NuSVR requires at least one sample".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let c = F::one() / (self.nu * n_f);

        let svr = SVR::new(self.kernel.clone())
            .with_c(c)
            .with_epsilon(F::zero())
            .with_tol(self.tol)
            .with_max_iter(self.max_iter)
            .with_cache_size(self.cache_size);

        let fitted = svr.fit(x, y)?;
        Ok(FittedNuSVR(fitted))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedNuSVR<F, K>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.0.predict(x)
    }
}

impl<F: Float, K: Kernel<F>> FittedNuSVR<F, K> {
    /// Compute the raw decision function values for each sample.
    ///
    /// Delegates to the inner [`FittedSVR::decision_function`].
    ///
    /// # Errors
    ///
    /// Returns `Ok` always.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.0.decision_function(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::{LinearKernel, RbfKernel};
    use ndarray::array;

    #[test]
    fn test_nusvc_linear_separable() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_nusvc_rbf() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let model = NuSVC::new(RbfKernel::with_gamma(0.5)).with_nu(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_nusvc_decision_function() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let df = fitted.decision_function(&x).unwrap();
        assert_eq!(df.nrows(), 6);
    }

    #[test]
    fn test_nusvc_invalid_nu_zero() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvc_invalid_nu_above_one() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(1.5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvc_nu_one() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(1.0);
        let result = model.fit(&x, &y);
        // Should succeed (nu=1 is valid)
        assert!(result.is_ok());
    }

    #[test]
    fn test_nusvr_simple() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let model = NuSVR::new(LinearKernel).with_nu(0.5).with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (*p - actual).abs() < 3.0,
                "NuSVR prediction {p} too far from actual {actual}"
            );
        }
    }

    #[test]
    fn test_nusvr_decision_function() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = NuSVR::new(LinearKernel).with_nu(0.5).with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert!((df[i] - preds[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nusvr_invalid_nu() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = NuSVR::new(LinearKernel).with_nu(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model2 = NuSVR::new(LinearKernel).with_nu(-0.5);
        assert!(model2.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvc_builder_pattern() {
        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.3)
            .with_tol(1e-4)
            .with_max_iter(5000)
            .with_cache_size(2048);

        assert!((model.nu - 0.3).abs() < 1e-10);
        assert!((model.tol - 1e-4).abs() < 1e-10);
        assert_eq!(model.max_iter, 5000);
        assert_eq!(model.cache_size, 2048);
    }

    #[test]
    fn test_nusvr_builder_pattern() {
        let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.8)
            .with_tol(1e-5)
            .with_max_iter(20000)
            .with_cache_size(512);

        assert!((model.nu - 0.8).abs() < 1e-10);
        assert!((model.tol - 1e-5).abs() < 1e-10);
        assert_eq!(model.max_iter, 20000);
        assert_eq!(model.cache_size, 512);
    }
}
