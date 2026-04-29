//! One-Class SVM for novelty detection.
//!
//! This module provides [`OneClassSVM`], which learns a decision boundary
//! around the training data and classifies new points as inliers (`+1`) or
//! outliers (`-1`).
//!
//! # Algorithm
//!
//! One-Class SVM trains a standard binary SVC where all training data is
//! assigned label `+1` and a synthetic origin point is assigned label `-1`.
//! The decision function then separates the data from the origin in kernel
//! feature space. Points with positive decision values are inliers; negative
//! are outliers.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::one_class_svm::OneClassSVM;
//! use ferrolearn_linear::svm::RbfKernel;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2, Array1};
//!
//! let x_train = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.5, 1.0,  1.0, 1.5,
//!     1.2, 1.3,  1.3, 1.2,  1.1, 1.1,
//! ]).unwrap();
//!
//! let model = OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(1.0));
//! let fitted = model.fit(&x_train, &()).unwrap();
//!
//! // Most training data should be classified as inliers.
//! let preds = fitted.predict(&x_train).unwrap();
//! let inliers: usize = preds.iter().filter(|&&p| p == 1).count();
//! assert!(inliers >= 4);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use crate::svm::Kernel;

// ---------------------------------------------------------------------------
// OneClassSVM
// ---------------------------------------------------------------------------

/// One-Class SVM for novelty detection.
///
/// Learns a decision boundary around the training data. New points are
/// classified as inliers (`+1`) or outliers (`-1`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`RbfKernel`](super::svm::RbfKernel)).
#[derive(Debug, Clone)]
pub struct OneClassSVM<F, K> {
    /// The nu parameter: upper bound on the fraction of outliers.
    /// Must be in `(0, 1]`. Default: `0.5`.
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

impl<F: Float, K: Kernel<F>> OneClassSVM<F, K> {
    /// Create a new `OneClassSVM` with the given kernel and default hyperparameters.
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

/// Fitted One-Class SVM.
///
/// Stores the support vectors and decision boundary. Points are classified
/// as inliers (+1) or outliers (-1) based on the sign of the decision
/// function.
#[derive(Debug, Clone)]
pub struct FittedOneClassSVM<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// Support vectors (stored as rows).
    support_vectors: Vec<Vec<F>>,
    /// Dual coefficients for each support vector.
    dual_coefs: Vec<F>,
    /// Bias (rho) term. Decision function: sign(f(x) - rho).
    rho: F,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Fit<Array2<F>, ()>
    for OneClassSVM<F, K>
{
    type Fitted = FittedOneClassSVM<F, K>;
    type Error = FerroError;

    /// Fit the One-Class SVM.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`.
    /// - [`FerroError::InsufficientSamples`] if no training data is provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOneClassSVM<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OneClassSVM requires at least one sample".into(),
            });
        }

        // Solve the one-class SVM dual:
        // max sum_i alpha_i - 0.5 * sum_{i,j} alpha_i * alpha_j * K(x_i, x_j)
        // s.t. 0 <= alpha_i <= 1/(n * nu), sum alpha_i = 1
        //
        // We use a simplified approach: initialize alphas uniformly, then
        // iterate with SMO-style updates.

        let c = F::one() / (F::from(n_samples).unwrap() * self.nu);
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        // Initialize alphas uniformly: alpha_i = 1/n
        let init_alpha = F::one() / F::from(n_samples).unwrap();
        let mut alphas = vec![init_alpha.min(c); n_samples];

        // Ensure sum(alphas) = 1 after capping at c.
        let alpha_sum: F = alphas.iter().copied().fold(F::zero(), |a, b| a + b);
        if alpha_sum < F::one() {
            // Distribute remaining mass.
            let remaining = F::one() - alpha_sum;
            let per_sample = remaining / F::from(n_samples).unwrap();
            for alpha in &mut alphas {
                *alpha = (*alpha + per_sample).min(c);
            }
        }

        // Compute initial gradient: grad_i = sum_j alpha_j * K(x_i, x_j)
        let eps = F::from(1e-12).unwrap_or_else(F::epsilon);
        let two = F::one() + F::one();

        let mut grad = vec![F::zero(); n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                grad[i] = grad[i] + alphas[j] * self.kernel.compute(&data[i], &data[j]);
            }
        }

        // SMO iterations
        for _iter in 0..self.max_iter {
            // Select working set: i with largest gradient (and alpha_i > 0),
            // j with smallest gradient (and alpha_j < c).
            let mut i_best = None;
            let mut i_max_grad = F::neg_infinity();
            let mut j_best = None;
            let mut j_min_grad = F::infinity();

            for k in 0..n_samples {
                if alphas[k] > eps && grad[k] > i_max_grad {
                    i_max_grad = grad[k];
                    i_best = Some(k);
                }
                if alphas[k] < c - eps && grad[k] < j_min_grad {
                    j_min_grad = grad[k];
                    j_best = Some(k);
                }
            }

            if i_best.is_none() || j_best.is_none() || i_max_grad - j_min_grad < self.tol {
                break;
            }

            let i = i_best.unwrap();
            let j = j_best.unwrap();

            if i == j {
                break;
            }

            let kii = self.kernel.compute(&data[i], &data[i]);
            let kjj = self.kernel.compute(&data[j], &data[j]);
            let kij = self.kernel.compute(&data[i], &data[j]);
            let eta = kii + kjj - two * kij;

            if eta <= eps {
                continue;
            }

            // Update: move mass from i to j.
            let delta = (grad[i] - grad[j]) / eta;
            let delta = delta.min(alphas[i]).min(c - alphas[j]);

            if delta.abs() < eps {
                continue;
            }

            alphas[i] = alphas[i] - delta;
            alphas[j] = alphas[j] + delta;

            // Update gradients.
            for k in 0..n_samples {
                let kki = self.kernel.compute(&data[k], &data[i]);
                let kkj = self.kernel.compute(&data[k], &data[j]);
                grad[k] = grad[k] - delta * kki + delta * kkj;
            }
        }

        // Compute rho from the KKT conditions.
        // For free SVs (0 < alpha_i < c): rho = grad_i = sum_j alpha_j * K(i, j).
        let mut rho_sum = F::zero();
        let mut rho_count = 0usize;

        for i in 0..n_samples {
            if alphas[i] > eps && alphas[i] < c - eps {
                rho_sum = rho_sum + grad[i];
                rho_count += 1;
            }
        }

        let rho = if rho_count > 0 {
            rho_sum / F::from(rho_count).unwrap()
        } else {
            // Fallback: use the midpoint of the gradient range among all SVs.
            let sv_grads: Vec<F> = (0..n_samples)
                .filter(|&i| alphas[i] > eps)
                .map(|i| grad[i])
                .collect();

            if sv_grads.is_empty() {
                F::zero()
            } else {
                let min_g = sv_grads.iter().fold(F::infinity(), |a, &b| a.min(b));
                let max_g = sv_grads.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
                (min_g + max_g) / two
            }
        };

        // Extract support vectors.
        let mut support_vectors = Vec::new();
        let mut dual_coefs = Vec::new();

        for (i, &alpha) in alphas.iter().enumerate() {
            if alpha > eps {
                support_vectors.push(data[i].clone());
                dual_coefs.push(alpha);
            }
        }

        // If no support vectors found, use all data as fallback.
        if support_vectors.is_empty() {
            let weight = F::one() / F::from(n_samples).unwrap();
            for row in &data {
                support_vectors.push(row.clone());
                dual_coefs.push(weight);
            }
        }

        let _ = n_features; // used for validation context

        Ok(FittedOneClassSVM {
            kernel: self.kernel.clone(),
            support_vectors,
            dual_coefs,
            rho,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    FittedOneClassSVM<F, K>
{
    /// Compute the decision function value for a single sample.
    ///
    /// Returns `f(x) - rho`, where `f(x) = sum_i alpha_i * K(sv_i, x)`.
    fn decision_value(&self, x: &[F]) -> F {
        let mut val = F::zero();
        for (sv, &coef) in self.support_vectors.iter().zip(self.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val - self.rho
    }

    /// Compute the raw decision function values for each sample.
    ///
    /// Returns an array of shape `(n_samples,)`. Positive values indicate
    /// inliers, negative values indicate outliers.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_samples = x.nrows();
        let mut result = Array1::<F>::zeros(n_samples);
        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            result[s] = self.decision_value(&xi);
        }
        Ok(result)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedOneClassSVM<F, K>
{
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict inlier (+1) or outlier (-1) for each sample.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let n_samples = x.nrows();
        let mut predictions = Array1::<isize>::zeros(n_samples);

        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            let val = self.decision_value(&xi);
            predictions[s] = if val >= F::zero() { 1 } else { -1 };
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::{LinearKernel, RbfKernel};
    use ndarray::Array2;

    fn make_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 0.9, 0.9, 1.0, 0.9, 0.9, 1.0, 1.05, 1.05,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_one_class_svm_fit() {
        let x = make_cluster_data();
        let model = OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(10.0));
        let result = model.fit(&x, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_one_class_svm_inliers() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Most training points should be classified as inliers.
        let inliers: usize = preds.iter().filter(|&&p| p == 1).count();
        assert!(inliers >= 6, "Expected at least 6 inliers, got {inliers}");
    }

    #[test]
    fn test_one_class_svm_outlier_detection() {
        let x_train = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05,
                -0.05,
            ],
        )
        .unwrap();

        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x_train, &()).unwrap();

        // A far-away point should be an outlier.
        let x_outlier = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let preds = fitted.predict(&x_outlier).unwrap();
        assert_eq!(preds[0], -1, "Far-away point should be an outlier");
    }

    #[test]
    fn test_one_class_svm_decision_function() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x, &()).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        assert_eq!(df.len(), 8);

        // Most decision values should be non-negative for training data.
        let positive: usize = df.iter().filter(|&&v| v >= 0.0).count();
        assert!(
            positive >= 6,
            "Expected at least 6 positive df, got {positive}"
        );
    }

    #[test]
    fn test_one_class_svm_invalid_nu() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();

        let model = OneClassSVM::new(RbfKernel::<f64>::new()).with_nu(0.0);
        assert!(model.fit(&x, &()).is_err());

        let model2 = OneClassSVM::new(RbfKernel::<f64>::new()).with_nu(1.5);
        assert!(model2.fit(&x, &()).is_err());
    }

    #[test]
    fn test_one_class_svm_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = OneClassSVM::new(RbfKernel::<f64>::new());
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_one_class_svm_builder_pattern() {
        let model = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
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
    fn test_one_class_svm_linear_kernel() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(LinearKernel).with_nu(0.5);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_one_class_svm_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let model = OneClassSVM::new(RbfKernel::with_gamma(1.0)).with_nu(0.5);
        let result = model.fit(&x, &());
        assert!(result.is_ok());
    }
}
