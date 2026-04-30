//! Gaussian Process regression.
//!
//! This module implements [`GaussianProcessRegressor`], a Bayesian nonparametric
//! regression model that provides both predictions and uncertainty estimates.
//!
//! # Algorithm
//!
//! Given training data `(X, y)` and a kernel function `k`:
//!
//! 1. Compute `K = k(X, X) + alpha * I` (kernel matrix + noise regularization).
//! 2. Cholesky decompose: `L = cholesky(K)`.
//! 3. Solve `L L^T alpha_vec = y` for `alpha_vec`.
//!
//! Prediction at new points `X*`:
//!
//! - Mean: `y* = K(X*, X) @ alpha_vec + y_mean`
//! - Variance: `var = diag(K(X*, X*)) - sum(v^2)` where `v = L^{-1} K(X, X*)`.

use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256Plus;

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::gp_kernels::{GPKernel, RBFKernel};

/// Gaussian Process regressor.
///
/// A Bayesian nonparametric model that infers a distribution over functions.
/// Provides both point predictions and uncertainty estimates (predictive variance).
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_kernel::gaussian_process::GaussianProcessRegressor;
/// use ferrolearn_kernel::gp_kernels::RBFKernel;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
///
/// let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
/// let fitted = gp.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
pub struct GaussianProcessRegressor<F: Float + Send + Sync + 'static> {
    /// Covariance kernel.
    kernel: Box<dyn GPKernel<F>>,
    /// Noise regularization added to the diagonal of the kernel matrix.
    /// Default: `1e-10`.
    alpha: F,
    /// Whether to normalize targets by subtracting the mean before fitting.
    /// The mean is added back during prediction. Default: `false`.
    normalize_y: bool,
    /// Number of random restarts for kernel hyperparameter optimization.
    /// Default: `0` (no optimization, use kernel parameters as-is).
    n_restarts_optimizer: usize,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for GaussianProcessRegressor<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianProcessRegressor")
            .field("normalize_y", &self.normalize_y)
            .field("n_restarts_optimizer", &self.n_restarts_optimizer)
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> GaussianProcessRegressor<F> {
    /// Create a new GP regressor with the given kernel and default settings.
    ///
    /// Defaults: `alpha = 1e-10`, `normalize_y = false`, `n_restarts_optimizer = 0`.
    pub fn new(kernel: Box<dyn GPKernel<F>>) -> Self {
        Self {
            kernel,
            alpha: F::from(1e-10).unwrap(),
            normalize_y: false,
            n_restarts_optimizer: 0,
        }
    }

    /// Create a GP regressor with an RBF kernel and default length scale.
    pub fn default_rbf() -> Self {
        Self::new(Box::new(RBFKernel::new(F::one())))
    }

    /// Set the noise regularization parameter.
    #[must_use]
    pub fn alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable or disable target normalization.
    #[must_use]
    pub fn normalize_y(mut self, normalize: bool) -> Self {
        self.normalize_y = normalize;
        self
    }

    /// Set the number of random restarts for optimizer.
    #[must_use]
    pub fn n_restarts_optimizer(mut self, n: usize) -> Self {
        self.n_restarts_optimizer = n;
        self
    }
}

/// Fitted Gaussian Process regressor.
///
/// Holds the Cholesky factor and weight vector needed for prediction.
/// Use [`predict`](Predict::predict) for point predictions, or
/// [`predict_with_std`](FittedGaussianProcessRegressor::predict_with_std)
/// for predictions with uncertainty.
pub struct FittedGaussianProcessRegressor<F: Float + Send + Sync + 'static> {
    /// Training features.
    x_train: Array2<F>,
    /// Cholesky factor L of the kernel matrix (lower triangular).
    l_factor: Array2<F>,
    /// Weight vector: alpha_vec = K^{-1} y (via Cholesky solve).
    alpha_vec: Array1<F>,
    /// Mean of y (subtracted during training if normalize_y is true).
    y_mean: F,
    /// Kernel used during fitting (stored for prediction).
    kernel: Box<dyn GPKernel<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FittedGaussianProcessRegressor<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedGaussianProcessRegressor")
            .field("n_train", &self.x_train.nrows())
            .field("n_features", &self.x_train.ncols())
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianProcessRegressor<F> {
    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }

    /// Predict mean and standard deviation at new points.
    ///
    /// Returns `(y_mean, y_std)` where `y_std` is the square root of the
    /// posterior predictive variance.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension does
    /// not match the training data.
    pub fn predict_with_std(&self, x: &Array2<F>) -> Result<(Array1<F>, Array1<F>), FerroError> {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "predict feature count must match training data".into(),
            });
        }

        let k_star = self.kernel.compute(x, &self.x_train);

        // Mean: K* @ alpha_vec + y_mean
        let y_pred = k_star.dot(&self.alpha_vec).mapv(|v| v + self.y_mean);

        // Variance: diag(K**) - sum(v^2, axis=0)
        // where v = L^{-1} K*^T
        let k_star_diag = self.kernel.diagonal(x);
        let n_train = self.x_train.nrows();
        let n_pred = x.nrows();

        // Solve L v = K*^T for v via forward substitution
        // K*^T has shape (n_train, n_pred)
        let k_star_t = k_star.t().to_owned();
        let mut v = Array2::<F>::zeros((n_train, n_pred));
        for col in 0..n_pred {
            // Forward substitution for this column
            for i in 0..n_train {
                let mut sum = k_star_t[[i, col]];
                for j in 0..i {
                    sum = sum - self.l_factor[[i, j]] * v[[j, col]];
                }
                v[[i, col]] = sum / self.l_factor[[i, i]];
            }
        }

        // var = k_star_diag - sum(v^2, axis=0)
        let mut var = k_star_diag;
        for col in 0..n_pred {
            let mut sum_sq = F::zero();
            for row in 0..n_train {
                sum_sq = sum_sq + v[[row, col]] * v[[row, col]];
            }
            var[col] = var[col] - sum_sq;
            // Clamp to avoid negative variance from numerical errors
            if var[col] < F::zero() {
                var[col] = F::zero();
            }
        }

        let std = var.mapv(num_traits::Float::sqrt);
        Ok((y_pred, std))
    }

    /// Draw `n_samples` realizations from the GP posterior at the query
    /// points `x`. Mirrors sklearn `GaussianProcessRegressor.sample_y`.
    ///
    /// Returns shape `(n_query, n_samples)`. Each column is one posterior
    /// draw `mean + L_post @ z` where `L_post` is the Cholesky factor of
    /// the posterior covariance `K** - K*ᵀ K⁻¹ K*` and `z ~ N(0, I)`.
    ///
    /// `random_state = Some(seed)` makes draws reproducible (uses
    /// `Xoshiro256Plus`); `None` reseeds from the OS RNG.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the feature dimension does not
    ///   match the training data.
    /// - [`FerroError::NumericalInstability`] if the posterior covariance
    ///   fails Cholesky (very rare; a small jitter `1e-10` is added to
    ///   the diagonal first).
    pub fn sample_y(
        &self,
        x: &Array2<F>,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> Result<Array2<F>, FerroError>
    where
        StandardNormal: Distribution<F>,
    {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "sample_y feature count must match training data".into(),
            });
        }

        let n_query = x.nrows();
        let n_train = self.x_train.nrows();

        let k_star = self.kernel.compute(x, &self.x_train);
        let k_star_star = self.kernel.compute(x, x);

        // Posterior mean: K* @ alpha_vec + y_mean.
        let mean = k_star.dot(&self.alpha_vec).mapv(|v| v + self.y_mean);

        // Solve L V = K*^T column-by-column for V (shape (n_train, n_query)).
        let k_star_t = k_star.t().to_owned();
        let mut v = Array2::<F>::zeros((n_train, n_query));
        for col in 0..n_query {
            for i in 0..n_train {
                let mut sum = k_star_t[[i, col]];
                for j in 0..i {
                    sum = sum - self.l_factor[[i, j]] * v[[j, col]];
                }
                v[[i, col]] = sum / self.l_factor[[i, i]];
            }
        }

        // Posterior covariance: K** - V^T V, with a small jitter on the
        // diagonal for Cholesky stability.
        let mut k_post = Array2::<F>::zeros((n_query, n_query));
        let jitter = F::from(1e-10).unwrap();
        for i in 0..n_query {
            for j in 0..n_query {
                let mut s = k_star_star[[i, j]];
                for k in 0..n_train {
                    s = s - v[[k, i]] * v[[k, j]];
                }
                k_post[[i, j]] = s;
                if i == j {
                    k_post[[i, j]] = k_post[[i, j]] + jitter;
                    if k_post[[i, j]] < jitter {
                        k_post[[i, j]] = jitter;
                    }
                }
            }
        }

        let l_post = cholesky(&k_post)?;

        let mut rng = match random_state {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_seed(rand::random()),
        };

        let mut out = Array2::<F>::zeros((n_query, n_samples));
        for s in 0..n_samples {
            // Draw z ~ N(0, I).
            let mut z = Array1::<F>::zeros(n_query);
            for i in 0..n_query {
                z[i] = StandardNormal.sample(&mut rng);
            }
            // Sample = mean + L_post @ z.
            for i in 0..n_query {
                let mut sum = F::zero();
                for j in 0..=i {
                    sum = sum + l_post[[i, j]] * z[j];
                }
                out[[i, s]] = mean[i] + sum;
            }
        }
        Ok(out)
    }

    /// Get the log marginal likelihood of the fitted model.
    ///
    /// This is useful for model selection and hyperparameter optimization.
    ///
    /// `log p(y|X) = -0.5 * y^T K^{-1} y - sum(log(diag(L))) - n/2 * log(2*pi)`
    #[must_use]
    pub fn log_marginal_likelihood(&self, y: &Array1<F>) -> F {
        let n = F::from(self.x_train.nrows()).unwrap();
        let y_centered = if self.y_mean == F::zero() {
            y.clone()
        } else {
            y.mapv(|v| v - self.y_mean)
        };

        // -0.5 * y^T alpha
        let data_fit = F::from(-0.5).unwrap() * y_centered.dot(&self.alpha_vec);

        // -sum(log(diag(L)))
        let mut log_det = F::zero();
        for i in 0..self.l_factor.nrows() {
            log_det = log_det + self.l_factor[[i, i]].ln();
        }
        let complexity = -log_det;

        // -n/2 * log(2*pi)
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        let norm_const = F::from(-0.5).unwrap() * n * two_pi.ln();

        data_fit + complexity + norm_const
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (pure Rust, generic over F)
// ---------------------------------------------------------------------------

/// Compute the lower Cholesky factor L such that A = L L^T.
///
/// Returns `Err` if A is not positive definite.
fn cholesky<F: Float>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
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
                        message: format!("kernel matrix is not positive definite at pivot {i}"),
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

/// Solve L x = b via forward substitution, where L is lower triangular.
fn forward_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
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

/// Solve L^T x = b via backward substitution, where L is lower triangular.
fn backward_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve (L L^T) x = b via Cholesky factorization.
fn cholesky_solve<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let z = forward_solve(l, b);
    backward_solve(l, &z)
}

// ---------------------------------------------------------------------------
// Fit / Predict implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for GaussianProcessRegressor<F> {
    type Fitted = FittedGaussianProcessRegressor<F>;
    type Error = FerroError;

    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedGaussianProcessRegressor<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples < 1 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n_samples,
                context: "GaussianProcessRegressor::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        // Optionally normalize y
        let y_mean = if self.normalize_y {
            y.sum() / F::from(n_samples).unwrap()
        } else {
            F::zero()
        };
        let y_centered = if self.normalize_y {
            y.mapv(|v| v - y_mean)
        } else {
            y.clone()
        };

        // Compute kernel matrix K + alpha * I
        let mut k_mat = self.kernel.compute(x, x);
        for i in 0..n_samples {
            k_mat[[i, i]] = k_mat[[i, i]] + self.alpha;
        }

        // Cholesky decomposition
        let l = cholesky(&k_mat)?;

        // Solve K alpha_vec = y_centered
        let alpha_vec = cholesky_solve(&l, &y_centered);

        Ok(FittedGaussianProcessRegressor {
            x_train: x.clone(),
            l_factor: l,
            alpha_vec,
            y_mean,
            kernel: self.kernel.clone_box(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianProcessRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "predict feature count must match training data".into(),
            });
        }

        let k_star = self.kernel.compute(x, &self.x_train);
        let y_pred = k_star.dot(&self.alpha_vec).mapv(|v| v + self.y_mean);
        Ok(y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp_kernels::{
        ConstantKernel, DotProductKernel, MaternKernel, ProductKernel, SumKernel, WhiteKernel,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    // Helper to create simple training data
    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];
        (x, y)
    }

    fn make_quadratic_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];
        (x, y)
    }

    // --- Basic fit/predict ---

    #[test]
    fn fit_predict_basic() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 5);
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_predict_interpolation() {
        // GP should near-interpolate training data (with small alpha)
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0))).alpha(1e-10);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn fit_predict_normalize_y() {
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0)))
            .normalize_y(true)
            .alpha(1e-10);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-3);
        }
    }

    // --- Predict with std ---

    #[test]
    fn predict_with_std_basic() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let (mean, std) = fitted.predict_with_std(&x).unwrap();
        assert_eq!(mean.len(), 5);
        assert_eq!(std.len(), 5);
        // Std at training points should be near zero
        for &s in &std {
            assert!(s < 1.0, "Training point std should be small, got {s}");
        }
    }

    #[test]
    fn predict_with_std_far_away() {
        // Points far from training data should have high uncertainty
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_far = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();
        let (_, std_far) = fitted.predict_with_std(&x_far).unwrap();
        let (_, std_near) = fitted.predict_with_std(&x).unwrap();

        let max_near_std = std_near.iter().copied().fold(0.0f64, f64::max);
        assert!(
            std_far[0] > max_near_std,
            "Far point std ({}) should exceed near std ({})",
            std_far[0],
            max_near_std
        );
    }

    #[test]
    fn predict_with_std_variance_nonnegative() {
        let (x, y) = make_quadratic_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((10, 1), (-5..5).map(f64::from).collect()).unwrap();
        let (_, std) = fitted.predict_with_std(&x_test).unwrap();
        for &s in &std {
            assert!(s >= 0.0, "std should be non-negative, got {s}");
        }
    }

    // --- Error handling ---

    #[test]
    fn fit_rejects_mismatched_y() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0]; // Wrong length
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        assert!(gp.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn predict_with_std_rejects_wrong_features() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict_with_std(&x_wrong).is_err());
    }

    // --- Different kernels ---

    #[test]
    fn fit_with_matern_15() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 1.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 0.5);
        }
    }

    #[test]
    fn fit_with_matern_25() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 2.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_dot_product() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(DotProductKernel::new(1.0))).alpha(1e-6);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_sum_kernel() {
        let (x, y) = make_linear_data();
        let kernel = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.01)),
        );
        let gp = GaussianProcessRegressor::new(Box::new(kernel));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn fit_with_product_kernel() {
        let (x, y) = make_linear_data();
        let kernel = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        let gp = GaussianProcessRegressor::new(Box::new(kernel));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    // --- Log marginal likelihood ---

    #[test]
    fn log_marginal_likelihood_is_finite() {
        let (x, y) = make_linear_data();
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let lml = fitted.log_marginal_likelihood(&y);
        assert!(lml.is_finite(), "LML should be finite, got {lml}");
    }

    #[test]
    fn log_marginal_likelihood_prefers_right_scale() {
        // LML should be higher when the kernel length scale matches the data
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| f64::from(i) * 0.5).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);

        let gp_good = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-6);
        let gp_bad = GaussianProcessRegressor::new(Box::new(RBFKernel::new(0.01))).alpha(1e-6);

        let fitted_good = gp_good.fit(&x, &y).unwrap();
        let fitted_bad = gp_bad.fit(&x, &y).unwrap();

        let lml_good = fitted_good.log_marginal_likelihood(&y);
        let lml_bad = fitted_bad.log_marginal_likelihood(&y);

        assert!(
            lml_good > lml_bad,
            "Good length scale LML ({lml_good}) should exceed bad ({lml_bad})"
        );
    }

    // --- Multivariate ---

    #[test]
    fn multivariate_2d() {
        let n = 20;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let t = i as f64 / n as f64;
                vec![t, t * t]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(0.5)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), n);
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    // --- f32 support ---

    #[test]
    fn f32_fit_predict() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0f32, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0f32, 1.0, 4.0, 9.0, 16.0]);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(2.0f32))).alpha(1e-6f32);
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert!(
                (pred[i] - y[i]).abs() < 1.0f32,
                "f32 pred {i} too far: {}",
                pred[i]
            );
        }
    }

    // --- Builder pattern ---

    #[test]
    fn builder_pattern() {
        let gp = GaussianProcessRegressor::default_rbf()
            .alpha(1e-6)
            .normalize_y(true)
            .n_restarts_optimizer(5);

        let (x, y) = make_linear_data();
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 5);
    }

    // --- Single sample ---

    #[test]
    fn single_sample() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = array![5.0f64];
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_abs_diff_eq!(pred[0], 5.0, epsilon = 1e-6);
    }

    // --- Constant target ---

    #[test]
    fn constant_target() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_elem(5, 7.0);
        let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert_abs_diff_eq!(p, 7.0, epsilon = 1e-4);
        }
    }
}
