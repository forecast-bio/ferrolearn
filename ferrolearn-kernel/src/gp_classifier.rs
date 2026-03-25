//! Gaussian Process classification via Laplace approximation.
//!
//! This module implements [`GaussianProcessClassifier`], a probabilistic
//! classifier that models the decision boundary as a Gaussian Process.
//!
//! # Algorithm (Binary Classification)
//!
//! Since the GP likelihood for classification (Bernoulli) is non-Gaussian,
//! exact inference is intractable. We use the **Laplace approximation**:
//!
//! 1. Find the MAP estimate `f*` of the latent function values by Newton's
//!    method on the un-normalized log posterior.
//! 2. Approximate the posterior as a Gaussian centered at `f*` with
//!    covariance `(K^{-1} + W)^{-1}`, where `W = diag(pi * (1 - pi))`.
//!
//! For multi-class problems, we use one-vs-rest binary GP classifiers.

use ndarray::{Array1, Array2};
use num_traits::Float;

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::gp_kernels::{GPKernel, RBFKernel};

/// Gaussian Process classifier using Laplace approximation.
///
/// Binary classification uses the logistic sigmoid link function.
/// Multi-class classification uses one-vs-rest decomposition.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_kernel::gp_classifier::GaussianProcessClassifier;
/// use ferrolearn_kernel::gp_kernels::RBFKernel;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 3.0, 3.5, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1]);
///
/// let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
/// let fitted = gpc.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
pub struct GaussianProcessClassifier<F: Float + Send + Sync + 'static> {
    /// Covariance kernel.
    kernel: Box<dyn GPKernel<F>>,
    /// Maximum iterations for the Laplace approximation Newton loop.
    /// Default: `100`.
    max_iter: usize,
    /// Convergence tolerance for the Newton loop.
    /// Default: `1e-6`.
    tol: F,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for GaussianProcessClassifier<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianProcessClassifier")
            .field("max_iter", &self.max_iter)
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> GaussianProcessClassifier<F> {
    /// Create a new GP classifier with the given kernel.
    pub fn new(kernel: Box<dyn GPKernel<F>>) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: F::from(1e-6).unwrap(),
        }
    }

    /// Create a GP classifier with an RBF kernel and default length scale.
    pub fn default_rbf() -> Self {
        Self::new(Box::new(RBFKernel::new(F::one())))
    }

    /// Set the maximum number of Newton iterations.
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
}

/// Fitted GP binary classifier (Laplace approximation).
///
/// Fields `pi_hat` and `l_factor` are stored for future predictive
/// variance computation (posterior integration).
#[allow(dead_code)]
struct FittedBinaryGPC<F: Float + Send + Sync + 'static> {
    /// Training features.
    x_train: Array2<F>,
    /// Kernel matrix K(X, X).
    k_mat: Array2<F>,
    /// Latent function values at convergence.
    f_hat: Array1<F>,
    /// Sigmoid(f_hat) — class probabilities at training points.
    pi_hat: Array1<F>,
    /// Cholesky factor of B = I + sqrt(W) K sqrt(W).
    l_factor: Array2<F>,
    /// Kernel used during fitting.
    kernel: Box<dyn GPKernel<F>>,
}

/// Fitted Gaussian Process classifier.
///
/// For binary classification, wraps a single Laplace-approximated GP.
/// For multi-class, uses one-vs-rest with per-class binary GPs.
pub struct FittedGaussianProcessClassifier<F: Float + Send + Sync + 'static> {
    /// The class labels (sorted, unique).
    classes: Vec<usize>,
    /// Binary classifiers: one per class for OvR (or single for binary).
    binary_models: Vec<FittedBinaryGPC<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FittedGaussianProcessClassifier<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedGaussianProcessClassifier")
            .field("n_classes", &self.classes.len())
            .field("classes", &self.classes)
            .finish()
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianProcessClassifier<F> {
    /// Predict class probabilities for new points.
    ///
    /// For binary classification, returns a 2-column array `[P(class=0), P(class=1)]`.
    /// For multi-class, returns `n_samples x n_classes` probabilities (normalized OvR).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension does not match.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        if n_classes == 2 {
            // Binary case: use single model
            let probs = predict_binary_proba(&self.binary_models[0], x)?;
            let mut result = Array2::<F>::zeros((n_samples, 2));
            for i in 0..n_samples {
                result[[i, 1]] = probs[i];
                result[[i, 0]] = F::one() - probs[i];
            }
            Ok(result)
        } else {
            // Multi-class OvR: get raw probabilities, then normalize
            let mut raw = Array2::<F>::zeros((n_samples, n_classes));
            for (c, model) in self.binary_models.iter().enumerate() {
                let probs = predict_binary_proba(model, x)?;
                for i in 0..n_samples {
                    raw[[i, c]] = probs[i];
                }
            }
            // Normalize rows to sum to 1
            for i in 0..n_samples {
                let row_sum: F = (0..n_classes)
                    .map(|c| raw[[i, c]])
                    .fold(F::zero(), |a, b| a + b);
                if row_sum > F::zero() {
                    for c in 0..n_classes {
                        raw[[i, c]] = raw[[i, c]] / row_sum;
                    }
                } else {
                    // Uniform if all zeros
                    let uniform = F::one() / F::from(n_classes).unwrap();
                    for c in 0..n_classes {
                        raw[[i, c]] = uniform;
                    }
                }
            }
            Ok(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// Sigmoid and helpers
// ---------------------------------------------------------------------------

/// Logistic sigmoid: sigma(f) = 1 / (1 + exp(-f)).
fn sigmoid<F: Float>(f: F) -> F {
    F::one() / (F::one() + (-f).exp())
}

/// Fit a binary GP classifier using Laplace approximation.
///
/// `y_binary` should contain `F::zero()` (class 0) or `F::one()` (class 1).
fn fit_binary_gpc<F: Float + Send + Sync + 'static>(
    kernel: &dyn GPKernel<F>,
    x: &Array2<F>,
    y_binary: &Array1<F>,
    max_iter: usize,
    tol: F,
) -> Result<FittedBinaryGPC<F>, FerroError> {
    let n = x.nrows();
    let k_mat = kernel.compute(x, x);

    // Initialize latent function values to zero
    let mut f = Array1::<F>::zeros(n);

    for _iter in 0..max_iter {
        // pi = sigmoid(f)
        let pi: Array1<F> = f.mapv(sigmoid);

        // W = diag(pi * (1 - pi))
        let w: Array1<F> = pi
            .iter()
            .zip(f.iter())
            .map(|(&p, _)| {
                let w_val = p * (F::one() - p);
                // Clamp to avoid zero/negative
                if w_val < F::from(1e-12).unwrap() {
                    F::from(1e-12).unwrap()
                } else {
                    w_val
                }
            })
            .collect();

        let sqrt_w: Array1<F> = w.mapv(|v| v.sqrt());

        // B = I + sqrt(W) K sqrt(W)
        let mut b = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = sqrt_w[i] * k_mat[[i, j]] * sqrt_w[j];
            }
            b[[i, i]] = b[[i, i]] + F::one();
        }

        // Cholesky of B
        let l = cholesky_gpc(&b)?;

        // b_vec = W f + (y - pi)
        let b_vec: Array1<F> = w
            .iter()
            .zip(f.iter())
            .zip(y_binary.iter().zip(pi.iter()))
            .map(|((&wi, &fi), (&yi, &pii))| wi * fi + (yi - pii))
            .collect();

        // a = b_vec - sqrt(W) @ solve(L^T, solve(L, sqrt(W) @ K @ b_vec))
        // Step 1: compute K @ b_vec
        let k_b = mat_vec_mul(&k_mat, &b_vec);
        // Step 2: sqrt(W) * (K @ b_vec)
        let sw_kb: Array1<F> = sqrt_w
            .iter()
            .zip(k_b.iter())
            .map(|(&s, &v)| s * v)
            .collect();
        // Step 3: solve L z = sw_kb
        let z = forward_solve_gpc(&l, &sw_kb);
        // Step 4: solve L^T z2 = z
        let z2 = backward_solve_gpc(&l, &z);
        // Step 5: sqrt(W) * z2
        let sw_z2: Array1<F> = sqrt_w.iter().zip(z2.iter()).map(|(&s, &v)| s * v).collect();
        // Step 6: a = b_vec - sw_z2
        let a: Array1<F> = b_vec
            .iter()
            .zip(sw_z2.iter())
            .map(|(&b, &s)| b - s)
            .collect();

        // f_new = K @ a
        let f_new = mat_vec_mul(&k_mat, &a);

        // Check convergence
        let max_change = f_new
            .iter()
            .zip(f.iter())
            .map(|(&fn_i, &f_i)| (fn_i - f_i).abs())
            .fold(F::zero(), |a, b| if a > b { a } else { b });

        f = f_new;

        if max_change < tol {
            break;
        }
    }

    // Final pi and L
    let pi_hat: Array1<F> = f.mapv(sigmoid);
    let w_final: Array1<F> = pi_hat
        .iter()
        .map(|&p| {
            let w_val = p * (F::one() - p);
            if w_val < F::from(1e-12).unwrap() {
                F::from(1e-12).unwrap()
            } else {
                w_val
            }
        })
        .collect();
    let sqrt_w_final: Array1<F> = w_final.mapv(|v| v.sqrt());
    let mut b_final = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b_final[[i, j]] = sqrt_w_final[i] * k_mat[[i, j]] * sqrt_w_final[j];
        }
        b_final[[i, i]] = b_final[[i, i]] + F::one();
    }
    let l_final = cholesky_gpc(&b_final)?;

    Ok(FittedBinaryGPC {
        x_train: x.clone(),
        k_mat,
        f_hat: f,
        pi_hat,
        l_factor: l_final,
        kernel: kernel.clone_box(),
    })
}

/// Predict binary class probabilities at new points.
fn predict_binary_proba<F: Float + Send + Sync + 'static>(
    model: &FittedBinaryGPC<F>,
    x: &Array2<F>,
) -> Result<Array1<F>, FerroError> {
    if x.ncols() != model.x_train.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows(), model.x_train.ncols()],
            actual: vec![x.nrows(), x.ncols()],
            context: "predict feature count must match training data".into(),
        });
    }

    let n = model.x_train.nrows();
    let _n_pred = x.nrows();

    // K* = k(X_new, X_train), shape (n_pred, n)
    let k_star = model.kernel.compute(x, &model.x_train);

    // Latent mean: f* = K* @ (y - pi_hat) where we stored f_hat and pi_hat
    // Actually: f* = K* @ grad_log_posterior
    // grad = y_binary - pi_hat, but we don't have y_binary here.
    // Instead, the standard approach is: f* = K* @ K^{-1} f_hat
    // which equals K* @ alpha where alpha = K^{-1} f_hat

    // Compute K^{-1} f_hat via the stored Cholesky of B
    // We can use: K^{-1} f_hat ~ (y - pi) from the converged Newton step
    // Since at convergence: f = K @ a where a = W f + (y - pi) - sqrt(W) L^{-T} L^{-1} sqrt(W) K (W f + (y - pi))
    // The simpler approach: K^{-1} f_hat can be solved directly.
    // Let's solve K alpha = f_hat using Cholesky of K + jitter

    // Add small jitter for numerical stability
    let jitter = F::from(1e-8).unwrap();
    let mut k_reg = model.k_mat.clone();
    for i in 0..n {
        k_reg[[i, i]] = k_reg[[i, i]] + jitter;
    }
    let l_k = cholesky_gpc(&k_reg)?;
    let alpha = cholesky_solve_gpc(&l_k, &model.f_hat);

    // Latent mean at test points
    let f_star = k_star.dot(&alpha);

    // Latent variance at test points (for more calibrated probabilities)
    // v = L_k^{-1} K*^T
    // Compute predictive probabilities using latent mean.
    // Using sigmoid of the latent mean (deterministic approximation).
    // A more accurate approach would integrate over the posterior variance,
    // but sigmoid(f_star) is the standard Laplace approximation output.
    let probs = f_star.mapv(sigmoid);

    Ok(probs)
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (duplicated from gaussian_process.rs to keep
// modules independent; could be extracted to a shared module later)
// ---------------------------------------------------------------------------

/// Matrix-vector multiplication.
fn mat_vec_mul<F: Float + 'static>(a: &Array2<F>, v: &Array1<F>) -> Array1<F> {
    a.dot(v)
}

/// Cholesky decomposition: A = L L^T.
fn cholesky_gpc<F: Float>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
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
                        message: format!(
                            "B matrix is not positive definite in Laplace approximation (pivot {i})"
                        ),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Forward substitution: solve L x = b.
fn forward_solve_gpc<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Backward substitution: solve L^T x = b.
fn backward_solve_gpc<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve (L L^T) x = b.
fn cholesky_solve_gpc<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let z = forward_solve_gpc(l, b);
    backward_solve_gpc(l, &z)
}

// ---------------------------------------------------------------------------
// Fit / Predict implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for GaussianProcessClassifier<F>
{
    type Fitted = FittedGaussianProcessClassifier<F>;
    type Error = FerroError;

    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedGaussianProcessClassifier<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "GaussianProcessClassifier::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        // Find unique classes
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "need at least 2 classes, got {} unique class(es)",
                    classes.len()
                ),
            });
        }

        let binary_models = if classes.len() == 2 {
            // Binary: single model, y = 0 or 1 (map to the second class)
            let y_binary: Array1<F> =
                y.mapv(|v| if v == classes[1] { F::one() } else { F::zero() });
            let model =
                fit_binary_gpc(self.kernel.as_ref(), x, &y_binary, self.max_iter, self.tol)?;
            vec![model]
        } else {
            // Multi-class: one-vs-rest
            let mut models = Vec::with_capacity(classes.len());
            for &cls in &classes {
                let y_binary: Array1<F> = y.mapv(|v| if v == cls { F::one() } else { F::zero() });
                let model =
                    fit_binary_gpc(self.kernel.as_ref(), x, &y_binary, self.max_iter, self.tol)?;
                models.push(model);
            }
            models
        };

        Ok(FittedGaussianProcessClassifier {
            classes,
            binary_models,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianProcessClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let proba = self.predict_proba(x)?;
        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_prob = proba[[i, 0]];
            for c in 1..self.classes.len() {
                if proba[[i, c]] > best_prob {
                    best_prob = proba[[i, c]];
                    best_class = c;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp_kernels::{ConstantKernel, MaternKernel, ProductKernel, SumKernel, WhiteKernel};
    use ndarray::{Array2, array};

    // Helper to create linearly separable binary data
    fn make_binary_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        (x, y)
    }

    fn make_binary_2d() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, // class 0
                3.0, 3.0, 3.5, 3.5, 3.0, 3.5, 3.5, 3.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn make_multiclass_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec((9, 1), vec![0.0, 0.5, 1.0, 4.0, 4.5, 5.0, 8.0, 8.5, 9.0])
            .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, y)
    }

    // --- Basic fit/predict ---

    #[test]
    fn fit_predict_binary() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
        // Should get most training points right
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 8,
            "Expected at least 8/10 correct, got {correct}/10"
        );
    }

    #[test]
    fn fit_predict_binary_2d() {
        let (x, y) = make_binary_2d();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/8 correct, got {correct}/8"
        );
    }

    // --- Predict proba ---

    #[test]
    fn predict_proba_binary() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (10, 2));

        // Probabilities should sum to 1
        for i in 0..10 {
            let row_sum = proba[[i, 0]] + proba[[i, 1]];
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {i} sums to {row_sum}, expected 1.0"
            );
        }

        // Probabilities should be in [0, 1]
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "Probability {p} out of range");
        }
    }

    #[test]
    fn predict_proba_class_0_near_0() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // Points clearly in class 0 region should have high P(class=0)
        assert!(
            proba[[0, 0]] > 0.5,
            "P(class=0) at x=0.0 should be > 0.5, got {}",
            proba[[0, 0]]
        );

        // Points clearly in class 1 region should have high P(class=1)
        assert!(
            proba[[9, 1]] > 0.5,
            "P(class=1) at x=7.0 should be > 0.5, got {}",
            proba[[9, 1]]
        );
    }

    // --- Multi-class ---

    #[test]
    fn fit_predict_multiclass() {
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 9);

        // Check predictions contain valid classes
        for &p in pred.iter() {
            assert!(p <= 2, "Prediction {p} not in valid classes [0, 1, 2]");
        }

        // Should get most right
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/9 correct, got {correct}/9"
        );
    }

    #[test]
    fn predict_proba_multiclass() {
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (9, 3));

        // Probabilities should sum to ~1 per row
        for i in 0..9 {
            let row_sum: f64 = (0..3).map(|c| proba[[i, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-8,
                "Row {i} sums to {row_sum}, expected 1.0"
            );
        }
    }

    // --- Error handling ---

    #[test]
    fn fit_rejects_single_class() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_elem(5, 0usize);
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn fit_rejects_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = array![0usize];
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn fit_rejects_mismatched_y() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 1, 0]; // Wrong length
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    // --- Different kernels ---

    #[test]
    fn fit_with_matern() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(MaternKernel::new(1.0, 1.5)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    #[test]
    fn fit_with_sum_kernel() {
        let (x, y) = make_binary_data();
        let kernel = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.1)),
        );
        let gpc = GaussianProcessClassifier::new(Box::new(kernel));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    #[test]
    fn fit_with_product_kernel() {
        let (x, y) = make_binary_data();
        let kernel = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        let gpc = GaussianProcessClassifier::new(Box::new(kernel));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- Builder pattern ---

    #[test]
    fn builder_pattern() {
        let gpc = GaussianProcessClassifier::default_rbf()
            .max_iter(50)
            .tol(1e-8);

        let (x, y) = make_binary_data();
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- f32 support ---

    #[test]
    fn f32_fit_predict() {
        let x = Array2::from_shape_vec((8, 1), vec![0.0f32, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5])
            .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let gpc = GaussianProcessClassifier::<f32>::new(Box::new(RBFKernel::new(1.0f32)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 8);
    }

    // --- Convergence ---

    #[test]
    fn converges_with_few_iterations() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0))).max_iter(5);
        // Should still produce reasonable results even with few iterations
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- Non-contiguous class labels ---

    #[test]
    fn non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0]).unwrap();
        let y = array![10usize, 10, 10, 20, 20, 20];
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Predictions should be from {10, 20}
        for &p in pred.iter() {
            assert!(p == 10 || p == 20, "Expected 10 or 20, got {p}");
        }
    }
}
