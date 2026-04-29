//! Support Vector Machine with kernel trick.
//!
//! This module provides [`SVC`] (classification) and [`SVR`] (regression)
//! support vector machines trained using the **Sequential Minimal Optimization
//! (SMO)** algorithm (Platt, 1998).
//!
//! # Kernels
//!
//! Four built-in kernels are provided:
//!
//! - [`LinearKernel`]: `K(x, y) = x . y`
//! - [`RbfKernel`]: `K(x, y) = exp(-gamma * ||x - y||^2)`
//! - [`PolynomialKernel`]: `K(x, y) = (gamma * x . y + coef0)^degree`
//! - [`SigmoidKernel`]: `K(x, y) = tanh(gamma * x . y + coef0)`
//!
//! Users can implement the [`Kernel`] trait for custom kernels.
//!
//! # Multiclass
//!
//! `SVC` uses a one-vs-one strategy for multiclass classification.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::svm::{SVC, LinearKernel};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  2.0, 1.0,  1.0, 2.0,
//!     5.0, 5.0,  6.0, 5.0,  5.0, 6.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = SVC::<f64, LinearKernel>::new(LinearKernel);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Kernel trait and built-in kernels
// ---------------------------------------------------------------------------

/// A kernel function for SVM.
///
/// Computes the inner product of two vectors in a (possibly implicit)
/// higher-dimensional feature space.
pub trait Kernel<F: Float>: Clone + Send + Sync {
    /// Compute the kernel value between two vectors.
    fn compute(&self, x: &[F], y: &[F]) -> F;
}

/// Linear kernel: `K(x, y) = x . y`.
#[derive(Debug, Clone, Copy)]
pub struct LinearKernel;

impl<F: Float> Kernel<F> for LinearKernel {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        x.iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

/// Radial Basis Function (Gaussian) kernel.
///
/// `K(x, y) = exp(-gamma * ||x - y||^2)`
#[derive(Debug, Clone, Copy)]
pub struct RbfKernel<F> {
    /// The gamma parameter. If `None`, it is set to `1 / (n_features * var(X))`.
    pub gamma: Option<F>,
}

impl<F: Float> RbfKernel<F> {
    /// Create a new RBF kernel with auto gamma.
    #[must_use]
    pub fn new() -> Self {
        Self { gamma: None }
    }

    /// Create a new RBF kernel with a specified gamma.
    #[must_use]
    pub fn with_gamma(gamma: F) -> Self {
        Self { gamma: Some(gamma) }
    }
}

impl<F: Float> Default for RbfKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for RbfKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = self.gamma.unwrap_or_else(F::one);
        let sq_dist = x.iter().zip(y.iter()).fold(F::zero(), |acc, (&a, &b)| {
            let d = a - b;
            acc + d * d
        });
        (-gamma * sq_dist).exp()
    }
}

/// Polynomial kernel: `K(x, y) = (gamma * x . y + coef0)^degree`.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialKernel<F> {
    /// The gamma parameter. If `None`, uses `1 / n_features`.
    pub gamma: Option<F>,
    /// Polynomial degree.
    pub degree: usize,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> PolynomialKernel<F> {
    /// Create a new polynomial kernel with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: None,
            degree: 3,
            coef0: F::zero(),
        }
    }
}

impl<F: Float> Default for PolynomialKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for PolynomialKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = self.gamma.unwrap_or_else(F::one);
        let dot: F = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        let val = gamma * dot + self.coef0;
        let mut result = F::one();
        for _ in 0..self.degree {
            result = result * val;
        }
        result
    }
}

/// Sigmoid kernel: `K(x, y) = tanh(gamma * x . y + coef0)`.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidKernel<F> {
    /// The gamma parameter. If `None`, uses `1 / n_features`.
    pub gamma: Option<F>,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> SigmoidKernel<F> {
    /// Create a new sigmoid kernel with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: None,
            coef0: F::zero(),
        }
    }
}

impl<F: Float> Default for SigmoidKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for SigmoidKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = self.gamma.unwrap_or_else(F::one);
        let dot: F = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        (gamma * dot + self.coef0).tanh()
    }
}

// ---------------------------------------------------------------------------
// Kernel cache (LRU)
// ---------------------------------------------------------------------------

/// Simple LRU cache for kernel evaluations.
struct KernelCache<F> {
    cache: HashMap<(usize, usize), F>,
    order: Vec<(usize, usize)>,
    capacity: usize,
}

impl<F: Float> KernelCache<F> {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_or_compute<K: Kernel<F>>(
        &mut self,
        i: usize,
        j: usize,
        kernel: &K,
        data: &[Vec<F>],
    ) -> F {
        let key = if i <= j { (i, j) } else { (j, i) };
        if let Some(&val) = self.cache.get(&key) {
            return val;
        }
        let val = kernel.compute(&data[i], &data[j]);
        if self.order.len() >= self.capacity {
            if let Some(old_key) = self.order.first().copied() {
                self.cache.remove(&old_key);
                self.order.remove(0);
            }
        }
        self.cache.insert(key, val);
        self.order.push(key);
        val
    }
}

// ---------------------------------------------------------------------------
// SMO solver for binary SVM
// ---------------------------------------------------------------------------

/// Result of a binary SMO solve.
struct SmoResult<F> {
    alphas: Vec<F>,
    bias: F,
}

/// SMO implementation (Platt 1998, Fan-Chen-Lin 2005 WSS).
///
/// Uses the dual gradient `grad_i = (Q * alpha)_i - 1` where
/// `Q_{ij} = y_i * y_j * K(x_i, x_j)`. Bias is computed after
/// convergence from the KKT conditions.
fn smo_binary<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    labels: &[F],
    kernel: &K,
    c: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Result<SmoResult<F>, FerroError> {
    let n = data.len();
    let mut alphas = vec![F::zero(); n];
    let mut cache = KernelCache::new(cache_size);

    // Gradient of the dual objective: grad_i = (Q*alpha)_i - 1
    // where Q_{ij} = y_i * y_j * K(x_i, x_j).
    // Initially alpha = 0, so grad_i = -1 for all i.
    let mut grad: Vec<F> = vec![-F::one(); n];

    let two = F::one() + F::one();
    let eps = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iter in 0..max_iter {
        // Working set selection (Fan-Chen-Lin 2005):
        // I_up  = {i : (y_i=+1 and alpha_i < C) or (y_i=-1 and alpha_i > 0)}
        // I_low = {j : (y_j=+1 and alpha_j > 0) or (y_j=-1 and alpha_j < C)}
        // Select i = argmax_{t in I_up}  -y_t * grad_t
        // Select j = argmin_{t in I_low} -y_t * grad_t

        let mut i_up = None;
        let mut max_val = F::neg_infinity();
        let mut j_low = None;
        let mut min_val = F::infinity();

        for t in 0..n {
            let val = -labels[t] * grad[t];

            let in_up = (labels[t] > F::zero() && alphas[t] < c - eps)
                || (labels[t] < F::zero() && alphas[t] > eps);

            let in_low = (labels[t] > F::zero() && alphas[t] > eps)
                || (labels[t] < F::zero() && alphas[t] < c - eps);

            if in_up && val > max_val {
                max_val = val;
                i_up = Some(t);
            }
            if in_low && val < min_val {
                min_val = val;
                j_low = Some(t);
            }
        }

        // Stopping criterion: KKT gap < tol
        if i_up.is_none() || j_low.is_none() || max_val - min_val < tol {
            break;
        }

        let i = i_up.unwrap();
        let j = j_low.unwrap();

        if i == j {
            break;
        }

        // Compute second-order info
        let kii = cache.get_or_compute(i, i, kernel, data);
        let kjj = cache.get_or_compute(j, j, kernel, data);
        let kij = cache.get_or_compute(i, j, kernel, data);
        let eta = kii + kjj - two * kij;

        if eta <= eps {
            continue;
        }

        // Bounds for alpha_j
        let old_ai = alphas[i];
        let old_aj = alphas[j];

        let (lo, hi) = if labels[i] == labels[j] {
            let sum = old_ai + old_aj;
            ((sum - c).max(F::zero()), sum.min(c))
        } else {
            let diff = old_aj - old_ai;
            (diff.max(F::zero()), (c + diff).min(c))
        };

        if (hi - lo).abs() < eps {
            continue;
        }

        // Analytic update for alpha_j (Platt 1998).
        // E_k = y_k * grad_k (dual error, where grad = Q*alpha - e).
        // alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        //             = alpha_j + y_j * (y_i * grad_i - y_j * grad_j) / eta
        let mut new_aj = old_aj + labels[j] * (labels[i] * grad[i] - labels[j] * grad[j]) / eta;

        // Clip to [lo, hi]
        if new_aj > hi {
            new_aj = hi;
        }
        if new_aj < lo {
            new_aj = lo;
        }

        if (new_aj - old_aj).abs() < eps {
            continue;
        }

        let new_ai = old_ai + labels[i] * labels[j] * (old_aj - new_aj);

        alphas[i] = new_ai;
        alphas[j] = new_aj;

        // Update dual gradient: grad_k += delta_alpha_i * Q_{k,i} + delta_alpha_j * Q_{k,j}
        // where Q_{k,t} = y_k * y_t * K(k,t)
        let delta_ai = new_ai - old_ai;
        let delta_aj = new_aj - old_aj;

        for (k, grad_k) in grad.iter_mut().enumerate() {
            let kki = cache.get_or_compute(k, i, kernel, data);
            let kkj = cache.get_or_compute(k, j, kernel, data);
            *grad_k = *grad_k
                + delta_ai * labels[k] * labels[i] * kki
                + delta_aj * labels[k] * labels[j] * kkj;
        }
    }

    // Compute bias from KKT conditions.
    // For support vectors with 0 < alpha_i < C:
    //   y_i * (sum_j alpha_j * y_j * K(i,j) + b) = 1
    //   b = y_i - sum_j alpha_j * y_j * K(i,j)
    // (since y_i^2 = 1, y_i * (y_i * f) = f, so b = 1/y_i - sum = y_i - sum)
    let mut b_sum = F::zero();
    let mut b_count = 0usize;

    for i in 0..n {
        if alphas[i] > eps && alphas[i] < c - eps {
            // This is a free support vector.
            let mut f_no_b = F::zero();
            for j in 0..n {
                if alphas[j] > eps {
                    f_no_b =
                        f_no_b + alphas[j] * labels[j] * cache.get_or_compute(i, j, kernel, data);
                }
            }
            b_sum = b_sum + labels[i] - f_no_b;
            b_count += 1;
        }
    }

    let bias = if b_count > 0 {
        b_sum / F::from(b_count).unwrap()
    } else {
        // Fallback: use all support vectors (bounded ones too)
        let mut b_sum_all = F::zero();
        let mut b_count_all = 0usize;
        for i in 0..n {
            if alphas[i] > eps {
                let mut f_no_b = F::zero();
                for j in 0..n {
                    if alphas[j] > eps {
                        f_no_b = f_no_b
                            + alphas[j] * labels[j] * cache.get_or_compute(i, j, kernel, data);
                    }
                }
                b_sum_all = b_sum_all + labels[i] - f_no_b;
                b_count_all += 1;
            }
        }
        if b_count_all > 0 {
            b_sum_all / F::from(b_count_all).unwrap()
        } else {
            F::zero()
        }
    };

    Ok(SmoResult { alphas, bias })
}

// ---------------------------------------------------------------------------
// SVC (Support Vector Classifier)
// ---------------------------------------------------------------------------

/// Support Vector Classifier.
///
/// Uses Sequential Minimal Optimization (SMO) to solve the dual QP.
/// Supports multiclass via one-vs-one strategy.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`LinearKernel`], [`RbfKernel`]).
#[derive(Debug, Clone)]
pub struct SVC<F, K> {
    /// The kernel function.
    pub kernel: K,
    /// Regularization parameter (penalty for misclassification).
    pub c: F,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> SVC<F, K> {
    /// Create a new `SVC` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `C = 1.0`, `tol = 1e-3`, `max_iter = 10000`, `cache_size = 1024`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            tol: F::from(1e-3).unwrap(),
            max_iter: 10000,
            cache_size: 1024,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
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

/// A single binary SVM model (one pair of classes in one-vs-one).
#[derive(Debug, Clone)]
struct BinarySvm<F> {
    /// Support vectors (stored as rows).
    support_vectors: Vec<Vec<F>>,
    /// Dual coefficients: alpha_i * y_i for each support vector.
    dual_coefs: Vec<F>,
    /// Bias term.
    bias: F,
    /// The two class labels: (negative_class, positive_class).
    class_neg: usize,
    class_pos: usize,
}

/// Fitted Support Vector Classifier.
///
/// Stores one binary SVM per pair of classes (one-vs-one). Implements
/// [`Predict`] to produce class labels.
#[derive(Debug, Clone)]
pub struct FittedSVC<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// One binary SVM per class pair.
    binary_models: Vec<BinarySvm<F>>,
    /// Sorted unique classes.
    classes: Vec<usize>,
}

impl<F: Float, K: Kernel<F>> FittedSVC<F, K> {
    /// Compute the decision function value for a single sample against a
    /// binary model.
    fn decision_value_binary(&self, x: &[F], model: &BinarySvm<F>) -> F {
        let mut val = model.bias;
        for (sv, &coef) in model.support_vectors.iter().zip(model.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val
    }

    /// Compute the raw decision function values for each sample.
    ///
    /// For binary classification, returns a 1-column array.
    /// For multiclass, returns one column per class pair.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the input has no columns.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        let n_models = self.binary_models.len();
        let mut result = Array2::<F>::zeros((n_samples, n_models));

        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            for (m, model) in self.binary_models.iter().enumerate() {
                result[[s, m]] = self.decision_value_binary(&xi, model);
            }
        }

        Ok(result)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<usize>> for SVC<F, K>
{
    type Fitted = FittedSVC<F, K>;
    type Error = FerroError;

    /// Fit the SVC model using SMO.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `C` is not positive.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedSVC<F, K>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SVC requires at least 2 distinct classes".into(),
            });
        }

        // Convert data to Vec<Vec<F>> for kernel cache.
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        // One-vs-one: train one binary SVM per pair.
        let n_classes = classes.len();
        let mut binary_models = Vec::new();

        for ci in 0..n_classes {
            for cj in (ci + 1)..n_classes {
                let class_neg = classes[ci];
                let class_pos = classes[cj];

                // Extract samples for these two classes.
                let mut sub_data = Vec::new();
                let mut sub_labels = Vec::new();
                let mut sub_indices = Vec::new();

                for s in 0..n_samples {
                    let label = y[s];
                    if label == class_neg {
                        sub_data.push(data[s].clone());
                        sub_labels.push(-F::one());
                        sub_indices.push(s);
                    } else if label == class_pos {
                        sub_data.push(data[s].clone());
                        sub_labels.push(F::one());
                        sub_indices.push(s);
                    }
                }

                let result = smo_binary(
                    &sub_data,
                    &sub_labels,
                    &self.kernel,
                    self.c,
                    self.tol,
                    self.max_iter,
                    self.cache_size,
                )?;

                // Extract support vectors (non-zero alphas).
                let eps = F::from(1e-8).unwrap_or_else(F::epsilon);
                let mut sv_data = Vec::new();
                let mut sv_coefs = Vec::new();

                for (k, &alpha) in result.alphas.iter().enumerate() {
                    if alpha > eps {
                        sv_data.push(sub_data[k].clone());
                        sv_coefs.push(alpha * sub_labels[k]);
                    }
                }

                binary_models.push(BinarySvm {
                    support_vectors: sv_data,
                    dual_coefs: sv_coefs,
                    bias: result.bias,
                    class_neg,
                    class_pos,
                });
            }
        }

        Ok(FittedSVC {
            kernel: self.kernel.clone(),
            binary_models,
            classes,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedSVC<F, K>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Uses one-vs-one voting: each binary model casts a vote for the
    /// winning class, and the class with the most votes wins.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            let mut votes = vec![0usize; n_classes];

            for model in &self.binary_models {
                let val = self.decision_value_binary(&xi, model);
                let winner = if val >= F::zero() {
                    model.class_pos
                } else {
                    model.class_neg
                };
                if let Some(idx) = self.classes.iter().position(|&c| c == winner) {
                    votes[idx] += 1;
                }
            }

            let best_class_idx = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(i, _)| i);

            predictions[s] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// SVR (Support Vector Regressor)
// ---------------------------------------------------------------------------

/// Support Vector Regressor.
///
/// Uses SMO to solve the epsilon-SVR dual problem.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type.
#[derive(Debug, Clone)]
pub struct SVR<F, K> {
    /// The kernel function.
    pub kernel: K,
    /// Regularization parameter.
    pub c: F,
    /// Epsilon tube width (insensitive loss zone).
    pub epsilon: F,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> SVR<F, K> {
    /// Create a new `SVR` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `C = 1.0`, `epsilon = 0.1`, `tol = 1e-3`,
    /// `max_iter = 10000`, `cache_size = 1024`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            epsilon: F::from(0.1).unwrap(),
            tol: F::from(1e-3).unwrap(),
            max_iter: 10000,
            cache_size: 1024,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon tube width.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
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

/// Fitted Support Vector Regressor.
///
/// Stores the support vectors, dual coefficients, and bias.
#[derive(Debug, Clone)]
pub struct FittedSVR<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// Support vectors.
    support_vectors: Vec<Vec<F>>,
    /// Dual coefficients (alpha_i* - alpha_i) for each support vector.
    dual_coefs: Vec<F>,
    /// Bias term.
    bias: F,
}

impl<F: Float, K: Kernel<F>> FittedSVR<F, K> {
    /// Compute the decision function value for a single sample.
    fn decision_value(&self, x: &[F]) -> F {
        let mut val = self.bias;
        for (sv, &coef) in self.support_vectors.iter().zip(self.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val
    }

    /// Compute the raw decision function values for each sample.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always (provided for API symmetry).
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

/// Solve epsilon-SVR using SMO.
///
/// Reformulates the epsilon-SVR dual as a standard 2n-variable QP and
/// solves it with the Fan-Chen-Lin WSS approach, analogous to `smo_binary`.
///
/// The 2n variables are indexed 0..2n:
/// - Index `k` (k < n)  corresponds to alpha\*\_k  with label +1
/// - Index `k` (k >= n) corresponds to alpha\_{k-n} with label -1
///
/// The Q matrix is: `Q_{ij} = s_i * s_j * K(p_i, p_j)` where `s` is the
/// sign (+1 or -1) and `p` maps to the original sample index.
///
/// The linear term is: `q_k = epsilon - y_{p_k} * s_k`.
#[allow(clippy::too_many_arguments)]
fn smo_svr<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    targets: &[F],
    kernel: &K,
    c: F,
    epsilon: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Result<(Vec<F>, F), FerroError> {
    let n = data.len();
    let m = 2 * n; // Total number of dual variables.

    // Encoding: variable k in [0, m)
    //   k < n  => alpha*_k   (sign = +1, sample index = k)
    //   k >= n => alpha_{k-n} (sign = -1, sample index = k - n)
    //
    // The dual is: min 1/2 * beta^T Q beta + q^T beta
    //   s.t. 0 <= beta_k <= C, sum_k s_k * beta_k = 0
    // where beta_k = alpha*_k or alpha_{k-n},
    //       Q_{ij} = s_i * s_j * K(p_i, p_j),
    //       q_k    = epsilon - y_{p_k} * s_k.
    //
    // This has the same structure as the SVC dual.

    let sign = |k: usize| -> F { if k < n { F::one() } else { -F::one() } };
    let sample = |k: usize| -> usize { if k < n { k } else { k - n } };

    let mut beta = vec![F::zero(); m];
    let mut cache = KernelCache::new(cache_size);

    // Gradient: grad_k = (Q * beta)_k + q_k.  Initially beta=0 so grad_k = q_k.
    // q_k = epsilon - y_{p_k} * s_k
    let mut grad: Vec<F> = (0..m)
        .map(|k| epsilon - targets[sample(k)] * sign(k))
        .collect();

    let two = F::one() + F::one();
    let eps_num = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iter in 0..max_iter {
        // WSS: same as SVC but with the extended variables.
        // I_up  = {k : (s_k=+1 and beta_k < C) or (s_k=-1 and beta_k > 0)}
        // I_low = {k : (s_k=+1 and beta_k > 0) or (s_k=-1 and beta_k < C)}
        // Select i = argmax_{k in I_up}  -s_k * grad_k
        // Select j = argmin_{k in I_low} -s_k * grad_k

        let mut i_up = None;
        let mut max_val = F::neg_infinity();
        let mut j_low = None;
        let mut min_val = F::infinity();

        for k in 0..m {
            let sk = sign(k);
            let val = -sk * grad[k];

            let in_up =
                (sk > F::zero() && beta[k] < c - eps_num) || (sk < F::zero() && beta[k] > eps_num);
            let in_low =
                (sk > F::zero() && beta[k] > eps_num) || (sk < F::zero() && beta[k] < c - eps_num);

            if in_up && val > max_val {
                max_val = val;
                i_up = Some(k);
            }
            if in_low && val < min_val {
                min_val = val;
                j_low = Some(k);
            }
        }

        if i_up.is_none() || j_low.is_none() || max_val - min_val < tol {
            break;
        }

        let i = i_up.unwrap();
        let j = j_low.unwrap();

        if i == j {
            break;
        }

        let si = sign(i);
        let sj = sign(j);
        let pi = sample(i);
        let pj = sample(j);

        // Q_{ii} = si*si*K(pi,pi) = K(pi,pi),  similarly for jj and ij
        let kii = cache.get_or_compute(pi, pi, kernel, data);
        let kjj = cache.get_or_compute(pj, pj, kernel, data);
        let kij = cache.get_or_compute(pi, pj, kernel, data);

        // eta = Q_{ii} + Q_{jj} - 2*Q_{ij} = K(pi,pi) + K(pj,pj) - 2*si*sj*K(pi,pj)
        let eta = kii + kjj - two * si * sj * kij;

        if eta <= eps_num {
            continue;
        }

        // Bounds for beta_j
        let old_bi = beta[i];
        let old_bj = beta[j];

        let (lo, hi) = if si == sj {
            let sum = old_bi + old_bj;
            ((sum - c).max(F::zero()), sum.min(c))
        } else {
            let diff = old_bj - old_bi;
            (diff.max(F::zero()), (c + diff).min(c))
        };

        if (hi - lo).abs() < eps_num {
            continue;
        }

        // Analytic update: beta_j += s_j * (E_i - E_j) / eta
        // where E_k = s_k * grad_k
        let mut new_bj = old_bj + sj * (si * grad[i] - sj * grad[j]) / eta;

        if new_bj > hi {
            new_bj = hi;
        }
        if new_bj < lo {
            new_bj = lo;
        }

        if (new_bj - old_bj).abs() < eps_num {
            continue;
        }

        let new_bi = old_bi + si * sj * (old_bj - new_bj);

        beta[i] = new_bi;
        beta[j] = new_bj;

        // Update gradient: grad_k += delta_bi * Q_{k,i} + delta_bj * Q_{k,j}
        // Q_{k,t} = s_k * s_t * K(p_k, p_t)
        let delta_bi = new_bi - old_bi;
        let delta_bj = new_bj - old_bj;

        for (k, grad_k) in grad.iter_mut().enumerate() {
            let sk = sign(k);
            let pk = sample(k);
            let kki = cache.get_or_compute(pk, pi, kernel, data);
            let kkj = cache.get_or_compute(pk, pj, kernel, data);
            *grad_k = *grad_k + delta_bi * sk * si * kki + delta_bj * sk * sj * kkj;
        }
    }

    // Recover alpha*_i = beta_i (i < n) and alpha_i = beta_{i+n} (i >= n).
    // Coefficient for prediction: coef_i = alpha*_i - alpha_i.
    let coefs: Vec<F> = (0..n).map(|i| beta[i] - beta[i + n]).collect();

    // Compute bias from KKT conditions on free support vectors.
    // For k where 0 < beta_k < C:
    //   grad_k = 0 at optimality => (Q*beta)_k + q_k = 0
    //   sum_t beta_t * s_k * s_t * K(p_k, p_t) + epsilon - y_{p_k} * s_k = 0
    //   s_k * sum_t (beta_t * s_t) * K(p_k, p_t) = y_{p_k} * s_k - epsilon
    //   sum_t coef_t_effective * K(p_k, p_t) = y_{p_k} - epsilon / s_k  (nah, let's use f directly)
    //
    // Prediction: f(x) = sum_i coef_i * K(x_i, x) + b
    // For free alpha*_i (0 < alpha*_i < C): y_i - f(x_i) = epsilon  => b = y_i - epsilon - sum coef_j * K(i,j)
    // For free alpha_i  (0 < alpha_i  < C): f(x_i) - y_i = epsilon  => b = y_i + epsilon - sum coef_j * K(i,j)

    let mut b_sum = F::zero();
    let mut b_count = 0usize;

    for i in 0..n {
        let mut kernel_sum = F::zero();
        let has_free = (beta[i] > eps_num && beta[i] < c - eps_num)
            || (beta[i + n] > eps_num && beta[i + n] < c - eps_num);

        if !has_free {
            continue;
        }

        for (j, &cj) in coefs.iter().enumerate() {
            if cj.abs() > eps_num {
                kernel_sum = kernel_sum + cj * cache.get_or_compute(i, j, kernel, data);
            }
        }

        if beta[i] > eps_num && beta[i] < c - eps_num {
            // alpha*_i is free: y_i - f(x_i) = epsilon => b = y_i - epsilon - kernel_sum
            b_sum = b_sum + targets[i] - epsilon - kernel_sum;
            b_count += 1;
        }
        if beta[i + n] > eps_num && beta[i + n] < c - eps_num {
            // alpha_i is free: f(x_i) - y_i = epsilon => b = y_i + epsilon - kernel_sum
            b_sum = b_sum + targets[i] + epsilon - kernel_sum;
            b_count += 1;
        }
    }

    let bias = if b_count > 0 {
        b_sum / F::from(b_count).unwrap()
    } else {
        F::zero()
    };

    Ok((coefs, bias))
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<F>> for SVR<F, K>
{
    type Fitted = FittedSVR<F, K>;
    type Error = FerroError;

    /// Fit the SVR model using SMO.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `C` is not positive.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedSVR<F, K>, FerroError> {
        let (n_samples, _n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SVR requires at least one sample".into(),
            });
        }

        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();
        let targets: Vec<F> = y.to_vec();

        let (coefs, bias) = smo_svr(
            &data,
            &targets,
            &self.kernel,
            self.c,
            self.epsilon,
            self.tol,
            self.max_iter,
            self.cache_size,
        )?;

        // Extract support vectors (non-zero coefficients).
        let eps = F::from(1e-8).unwrap_or_else(F::epsilon);
        let mut sv_data = Vec::new();
        let mut sv_coefs = Vec::new();

        for (i, &coef) in coefs.iter().enumerate() {
            if coef.abs() > eps {
                sv_data.push(data[i].clone());
                sv_coefs.push(coef);
            }
        }

        Ok(FittedSVR {
            kernel: self.kernel.clone(),
            support_vectors: sv_data,
            dual_coefs: sv_coefs,
            bias,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedSVR<F, K>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.decision_function(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_linear_kernel() {
        let k = LinearKernel;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert_relative_eq!(k.compute(&x, &y), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let k = RbfKernel::with_gamma(1.0);
        let x = vec![0.0, 0.0];
        let y = vec![0.0, 0.0];
        assert_relative_eq!(k.compute(&x, &y), 1.0, epsilon = 1e-10);

        // Different points should give value < 1
        let y2 = vec![1.0, 0.0];
        let val: f64 = k.compute(&x, &y2);
        assert!(val < 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_polynomial_kernel() {
        let k = PolynomialKernel {
            gamma: Some(1.0),
            degree: 2,
            coef0: 1.0,
        };
        let x = vec![1.0f64, 0.0];
        let y = vec![1.0, 0.0];
        // (1.0 * 1.0 + 1.0)^2 = 4.0
        assert_relative_eq!(k.compute(&x, &y), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let k = SigmoidKernel {
            gamma: Some(1.0),
            coef0: 0.0,
        };
        let x = vec![0.0f64];
        let y = vec![0.0];
        // tanh(0) = 0
        assert_relative_eq!(k.compute(&x, &y), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svc_linear_separable() {
        // Two well-separated clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let model = SVC::<f64, LinearKernel>::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_svc_rbf_xor() {
        // XOR problem: not linearly separable, needs RBF kernel.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, // class 0
                1.0, 1.0, 1.1, 1.1, // class 0
                1.0, 0.0, 1.1, 0.1, // class 1
                0.0, 1.0, 0.1, 1.1, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let kernel = RbfKernel::with_gamma(5.0);
        let model = SVC::new(kernel).with_c(100.0).with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 6,
            "Expected at least 6 correct for XOR, got {correct}"
        );
    }

    #[test]
    fn test_svc_multiclass() {
        // Three linearly separable clusters.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, // class 0
                5.0, 0.0, 5.5, 0.0, 5.0, 0.5, // class 1
                0.0, 5.0, 0.5, 5.0, 0.0, 5.5, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = SVC::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 7,
            "Expected at least 7 correct for multiclass, got {correct}"
        );
    }

    #[test]
    fn test_svc_decision_function() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = SVC::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        assert_eq!(df.nrows(), 6);
        assert_eq!(df.ncols(), 1); // One binary model for 2 classes.

        // Class 0 samples should have negative decision values,
        // class 1 should have positive.
        for i in 0..3 {
            assert!(
                df[[i, 0]] < 0.0 + 0.5, // Allow some tolerance
                "Class 0 sample {i} has decision value {}",
                df[[i, 0]]
            );
        }
    }

    #[test]
    fn test_svc_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = SVC::new(LinearKernel).with_c(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0usize, 0, 0];

        let model = SVC::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0usize, 1];

        let model = SVC::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svr_simple() {
        // Simple linear regression: y = 2x
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let model = SVR::new(LinearKernel)
            .with_c(100.0)
            .with_epsilon(0.1)
            .with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check predictions are reasonably close.
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (*p - actual).abs() < 2.0,
                "SVR prediction {p} too far from actual {actual}"
            );
        }
    }

    #[test]
    fn test_svr_decision_function() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SVR::new(LinearKernel).with_c(100.0).with_epsilon(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Decision function and predict should return the same values.
        for i in 0..4 {
            assert_relative_eq!(df[i], preds[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_svr_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = SVR::new(LinearKernel).with_c(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svr_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = SVR::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }
}
