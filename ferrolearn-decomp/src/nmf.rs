//! Non-negative Matrix Factorization (NMF).
//!
//! [`NMF`] decomposes a non-negative matrix `X` into two non-negative
//! factors `W` and `H` such that `X ~ W * H`, where:
//! - `X` has shape `(n_samples, n_features)`
//! - `W` has shape `(n_samples, n_components)`
//! - `H` has shape `(n_components, n_features)`
//!
//! # Algorithm
//!
//! Two solvers are supported:
//!
//! - **Multiplicative Update** (Lee & Seung, 2001): iteratively update `W` and
//!   `H` using multiplicative rules that guarantee non-negativity.
//! - **Coordinate Descent**: iteratively solve for each element of `W` and `H`
//!   using closed-form coordinate-wise updates.
//!
//! # Initialization
//!
//! - **Random**: initialize `W` and `H` with random non-negative values.
//! - **NNDSVD**: Non-Negative Double SVD, initializes `W` and `H` from a
//!   truncated SVD of `X`, setting negative entries to zero.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::NMF;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let nmf = NMF::<f64>::new(2);
//! let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let fitted = nmf.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// The solver algorithm for NMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NMFSolver {
    /// Multiplicative update rules (Lee & Seung, 2001).
    MultiplicativeUpdate,
    /// Coordinate descent.
    CoordinateDescent,
}

/// The initialization strategy for NMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NMFInit {
    /// Random non-negative initialization.
    Random,
    /// Non-Negative Double SVD initialization.
    Nndsvd,
}

// ---------------------------------------------------------------------------
// NMF (unfitted)
// ---------------------------------------------------------------------------

/// Non-negative Matrix Factorization configuration.
///
/// Holds hyperparameters for the NMF decomposition. Calling [`Fit::fit`]
/// computes the factorization and returns a [`FittedNMF`] that can
/// project new data via [`Transform::transform`].
#[derive(Debug, Clone)]
pub struct NMF<F> {
    /// Number of components to extract.
    n_components: usize,
    /// Maximum number of iterations for the solver.
    max_iter: usize,
    /// Convergence tolerance for the solver.
    tol: f64,
    /// The solver algorithm to use.
    solver: NMFSolver,
    /// The initialization strategy.
    init: NMFInit,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> NMF<F> {
    /// Create a new `NMF` that extracts `n_components` components.
    ///
    /// Defaults: `max_iter=200`, `tol=1e-4`, solver=`MultiplicativeUpdate`,
    /// init=`Random`, no random seed.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            solver: NMFSolver::MultiplicativeUpdate,
            init: NMFInit::Random,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of iterations.
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

    /// Set the solver algorithm.
    #[must_use]
    pub fn with_solver(mut self, solver: NMFSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set the initialization strategy.
    #[must_use]
    pub fn with_init(mut self, init: NMFInit) -> Self {
        self.init = init;
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

    /// Return the configured solver.
    #[must_use]
    pub fn solver(&self) -> NMFSolver {
        self.solver
    }

    /// Return the configured initialization strategy.
    #[must_use]
    pub fn init(&self) -> NMFInit {
        self.init
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

// ---------------------------------------------------------------------------
// FittedNMF
// ---------------------------------------------------------------------------

/// A fitted NMF model holding the learned components and reconstruction error.
///
/// Created by calling [`Fit::fit`] on an [`NMF`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the learned components.
#[derive(Debug, Clone)]
pub struct FittedNMF<F> {
    /// Learned component matrix H, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// The Frobenius norm of the reconstruction error at convergence.
    reconstruction_err_: F,
    /// Number of iterations performed.
    n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedNMF<F> {
    /// Learned components (H matrix), shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Frobenius norm of the reconstruction error `||X - W*H||_F`.
    #[must_use]
    pub fn reconstruction_err(&self) -> F {
        self.reconstruction_err_
    }

    /// Number of iterations performed during fitting.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of `X - W * H`.
fn reconstruction_error<F: Float + 'static>(x: &Array2<F>, w: &Array2<F>, h: &Array2<F>) -> F {
    let wh = w.dot(h);
    let mut err = F::zero();
    for (a, b) in x.iter().zip(wh.iter()) {
        let diff = *a - *b;
        err = err + diff * diff;
    }
    err.sqrt()
}

/// Small epsilon to prevent division by zero.
fn eps<F: Float>() -> F {
    F::from(1e-12).unwrap_or(F::epsilon())
}

/// Initialize W and H with random non-negative values.
fn init_random<F: Float>(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for elem in w.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or(F::zero()) + eps::<F>();
    }

    let mut h = Array2::<F>::zeros((n_components, n_features));
    for elem in h.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or(F::zero()) + eps::<F>();
    }

    (w, h)
}

/// NNDSVD initialization: compute a truncated SVD-like initialization.
///
/// Uses a simple approach: compute `X^T X`, eigendecompose, then use the
/// top eigenvectors to initialize H, and solve for W = X * H^+ (pseudoinverse).
fn init_nndsvd<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    seed: u64,
) -> Result<(Array2<F>, Array2<F>), FerroError> {
    let (n_samples, n_features) = x.dim();

    // Compute mean of X for scale.
    let mut total = F::zero();
    for &v in x.iter() {
        total = total + v;
    }
    let avg = (total / F::from(n_samples * n_features).unwrap())
        .abs()
        .sqrt();
    let avg = if avg < eps::<F>() { F::one() } else { avg };

    // Compute X^T X.
    let xtx = x.t().dot(x);

    // Eigendecompose with Jacobi.
    let max_iter = n_features * n_features * 100 + 1000;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&xtx, max_iter)?;

    // Sort eigenvalues descending.
    let mut indices: Vec<usize> = (0..n_features).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build H from top eigenvectors (as rows), clamp negatives to zero.
    let mut h = Array2::<F>::zeros((n_components, n_features));
    for (k, &idx) in indices.iter().take(n_components).enumerate() {
        for j in 0..n_features {
            let val = eigenvectors[[j, idx]];
            h[[k, j]] = if val > F::zero() { val } else { F::zero() };
        }
        // Ensure row is not all zeros.
        let row_sum: F = h.row(k).iter().copied().fold(F::zero(), |a, b| a + b);
        if row_sum < eps::<F>() {
            // Fall back to small random values.
            let mut rng: rand::rngs::StdRng =
                SeedableRng::seed_from_u64(seed.wrapping_add(k as u64));
            let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();
            for j in 0..n_features {
                h[[k, j]] = F::from(uniform.sample(&mut rng)).unwrap_or(F::zero()) * avg;
            }
        }
    }

    // Compute W = X * H^T * (H * H^T)^{-1}, but simpler: use multiplicative
    // update step starting from random W.
    let mut w = Array2::<F>::zeros((n_samples, n_components));
    // Solve W by least squares: W = X * H^T * pinv(H * H^T)
    // For simplicity, initialize W = X * H^T and normalize.
    let ht = h.t();
    let w_init = x.dot(&ht);
    for i in 0..n_samples {
        for k in 0..n_components {
            let val = w_init[[i, k]];
            w[[i, k]] = if val > F::zero() { val } else { eps::<F>() };
        }
    }

    Ok((w, h))
}

/// Jacobi eigendecomposition for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where column `i` of `eigenvectors`
/// is the eigenvector for `eigenvalues[i]`. Eigenvalues are NOT sorted.
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
        message: "Jacobi eigendecomposition did not converge in NMF NNDSVD init".into(),
    })
}

/// Multiplicative update solver (Lee & Seung, 2001).
///
/// Update rules:
///   W <- W * (X H^T) / (W H H^T + eps)
///   H <- H * (W^T X) / (W^T W H + eps)
fn solve_multiplicative_update<F: Float + 'static>(
    x: &Array2<F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    max_iter: usize,
    tol: f64,
) -> usize {
    let tol_f = F::from(tol).unwrap_or(F::epsilon());
    let epsilon = eps::<F>();
    let mut prev_err = reconstruction_error(x, w, h);

    for iteration in 0..max_iter {
        // Update H: H <- H * (W^T X) / (W^T W H + eps)
        let wt = w.t();
        let numerator_h = wt.dot(x);
        let denominator_h = wt.dot(&*w).dot(&*h);

        for (h_val, (num, den)) in h
            .iter_mut()
            .zip(numerator_h.iter().zip(denominator_h.iter()))
        {
            *h_val = *h_val * (*num / (*den + epsilon));
        }

        // Update W: W <- W * (X H^T) / (W H H^T + eps)
        let ht = h.t();
        let numerator_w = x.dot(&ht);
        let denominator_w = w.dot(&*h).dot(&ht);

        for (w_val, (num, den)) in w
            .iter_mut()
            .zip(numerator_w.iter().zip(denominator_w.iter()))
        {
            *w_val = *w_val * (*num / (*den + epsilon));
        }

        // Check convergence.
        let err = reconstruction_error(x, w, h);
        if (prev_err - err).abs() < tol_f {
            return iteration + 1;
        }
        prev_err = err;
    }

    max_iter
}

/// Coordinate descent solver.
///
/// Updates each element of H and W by solving a scalar minimization problem.
fn solve_coordinate_descent<F: Float + 'static>(
    x: &Array2<F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    max_iter: usize,
    tol: f64,
) -> usize {
    let (n_samples, n_features) = x.dim();
    let n_components = h.nrows();
    let tol_f = F::from(tol).unwrap_or(F::epsilon());
    let epsilon = eps::<F>();
    let mut prev_err = reconstruction_error(x, w, h);

    for iteration in 0..max_iter {
        // Update H: for each k, j, solve for H[k,j]
        // H[k,j] = max(0, (W[:,k]^T * (X[:,j] - W * H[:,j] + W[:,k]*H[k,j])) / (W[:,k]^T W[:,k]))
        for k in 0..n_components {
            let mut wk_norm_sq = F::zero();
            for i in 0..n_samples {
                wk_norm_sq = wk_norm_sq + w[[i, k]] * w[[i, k]];
            }

            if wk_norm_sq < epsilon {
                continue;
            }

            for j in 0..n_features {
                // Compute residual + current contribution.
                let mut numerator = F::zero();
                for i in 0..n_samples {
                    let mut wh_ij = F::zero();
                    for kk in 0..n_components {
                        if kk != k {
                            wh_ij = wh_ij + w[[i, kk]] * h[[kk, j]];
                        }
                    }
                    numerator = numerator + w[[i, k]] * (x[[i, j]] - wh_ij);
                }

                h[[k, j]] = if numerator > F::zero() {
                    numerator / wk_norm_sq
                } else {
                    F::zero()
                };
            }
        }

        // Update W: for each i, k, solve for W[i,k]
        for k in 0..n_components {
            let mut hk_norm_sq = F::zero();
            for j in 0..n_features {
                hk_norm_sq = hk_norm_sq + h[[k, j]] * h[[k, j]];
            }

            if hk_norm_sq < epsilon {
                continue;
            }

            for i in 0..n_samples {
                let mut numerator = F::zero();
                for j in 0..n_features {
                    let mut wh_ij = F::zero();
                    for kk in 0..n_components {
                        if kk != k {
                            wh_ij = wh_ij + w[[i, kk]] * h[[kk, j]];
                        }
                    }
                    numerator = numerator + h[[k, j]] * (x[[i, j]] - wh_ij);
                }

                w[[i, k]] = if numerator > F::zero() {
                    numerator / hk_norm_sq
                } else {
                    F::zero()
                };
            }
        }

        // Check convergence.
        let err = reconstruction_error(x, w, h);
        if (prev_err - err).abs() < tol_f {
            return iteration + 1;
        }
        prev_err = err;
    }

    max_iter
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for NMF<F> {
    type Fitted = FittedNMF<F>;
    type Error = FerroError;

    /// Fit the NMF model by decomposing `X ~ W * H`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the minimum of `n_samples` and `n_features`.
    /// - [`FerroError::InvalidParameter`] if any entry of `X` is negative.
    /// - [`FerroError::InsufficientSamples`] if there are zero samples.
    /// - [`FerroError::ConvergenceFailure`] if NNDSVD initialization fails.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedNMF<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NMF::fit".into(),
            });
        }
        if self.n_components > n_samples.min(n_features) {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_samples, n_features) = {}",
                    self.n_components,
                    n_samples.min(n_features)
                ),
            });
        }

        // Check non-negativity.
        for &val in x.iter() {
            if val < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "NMF requires all entries in X to be non-negative".into(),
                });
            }
        }

        let seed = self.random_state.unwrap_or(0);

        // Initialize W and H.
        let (mut w, mut h) = match self.init {
            NMFInit::Random => init_random(n_samples, n_features, self.n_components, seed),
            NMFInit::Nndsvd => init_nndsvd(x, self.n_components, seed)?,
        };

        // Solve.
        let n_iter = match self.solver {
            NMFSolver::MultiplicativeUpdate => {
                solve_multiplicative_update(x, &mut w, &mut h, self.max_iter, self.tol)
            }
            NMFSolver::CoordinateDescent => {
                solve_coordinate_descent(x, &mut w, &mut h, self.max_iter, self.tol)
            }
        };

        let reconstruction_err = reconstruction_error(x, &w, &h);

        Ok(FittedNMF {
            components_: h,
            reconstruction_err_: reconstruction_err,
            n_iter_: n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedNMF<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the learned NMF components.
    ///
    /// Solves for `W` in `X ~ W * H` using multiplicative updates with
    /// `H` fixed to the learned components.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the number of columns does not
    ///   match the number of features seen during fitting.
    /// - [`FerroError::InvalidParameter`] if any entry of `X` is negative.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedNMF::transform".into(),
            });
        }

        // Check non-negativity.
        for &val in x.iter() {
            if val < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "NMF requires all entries in X to be non-negative".into(),
                });
            }
        }

        let n_samples = x.nrows();
        let n_components = self.components_.nrows();
        let epsilon = eps::<F>();

        // Initialize W with uniform small values.
        let mut w = Array2::<F>::zeros((n_samples, n_components));
        let init_val = F::from(0.1).unwrap_or(F::one());
        for elem in w.iter_mut() {
            *elem = init_val;
        }

        // Run multiplicative updates with H fixed.
        let h = &self.components_;
        for _iter in 0..200 {
            let wt_num = x.dot(&h.t());
            let wt_den = w.dot(h).dot(&h.t());

            for (w_val, (num, den)) in w.iter_mut().zip(wt_num.iter().zip(wt_den.iter())) {
                *w_val = *w_val * (*num / (*den + epsilon));
            }
        }

        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (f64 specialisation)
// ---------------------------------------------------------------------------

impl PipelineTransformer for NMF<f64> {
    /// Fit NMF using the pipeline interface.
    ///
    /// The `y` argument is ignored; NMF is unsupervised.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl FittedPipelineTransformer for FittedNMF<f64> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
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

    /// Helper: create a small non-negative dataset.
    fn small_dataset() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    }

    /// Helper: create a larger non-negative dataset.
    fn medium_dataset() -> Array2<f64> {
        array![
            [5.0, 3.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 5.0, 4.0],
            [0.0, 0.0, 4.0, 3.0],
        ]
    }

    #[test]
    fn test_nmf_basic_fit() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
    }

    #[test]
    fn test_nmf_components_non_negative() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        for &val in fitted.components().iter() {
            assert!(
                val >= 0.0,
                "component value should be non-negative, got {val}"
            );
        }
    }

    #[test]
    fn test_nmf_transform_dimensions() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 2));
    }

    #[test]
    fn test_nmf_transform_non_negative() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        for &val in projected.iter() {
            assert!(val >= 0.0, "W value should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_nmf_reconstruction_error_decreases() {
        let nmf_few = NMF::<f64>::new(2).with_random_state(42).with_max_iter(10);
        let nmf_many = NMF::<f64>::new(2).with_random_state(42).with_max_iter(200);
        let x = small_dataset();
        let fitted_few = nmf_few.fit(&x, &()).unwrap();
        let fitted_many = nmf_many.fit(&x, &()).unwrap();
        assert!(
            fitted_many.reconstruction_err() <= fitted_few.reconstruction_err() + 1e-6,
            "more iterations should reduce error: few={}, many={}",
            fitted_few.reconstruction_err(),
            fitted_many.reconstruction_err()
        );
    }

    #[test]
    fn test_nmf_reconstruction_error_positive() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.reconstruction_err() >= 0.0);
    }

    #[test]
    fn test_nmf_coordinate_descent_solver() {
        let nmf = NMF::<f64>::new(2)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
        for &val in fitted.components().iter() {
            assert!(val >= 0.0, "CD component should be non-negative, got {val}");
        }
    }

    #[test]
    fn test_nmf_nndsvd_init() {
        let nmf = NMF::<f64>::new(2)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
        for &val in fitted.components().iter() {
            assert!(
                val >= 0.0,
                "NNDSVD component should be non-negative, got {val}"
            );
        }
    }

    #[test]
    fn test_nmf_cd_with_nndsvd() {
        let nmf = NMF::<f64>::new(2)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(42);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 4));
    }

    #[test]
    fn test_nmf_invalid_n_components_zero() {
        let nmf = NMF::<f64>::new(0);
        let x = small_dataset();
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_invalid_n_components_too_large() {
        let nmf = NMF::<f64>::new(10);
        let x = small_dataset(); // 4x3
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_negative_input_rejected() {
        let nmf = NMF::<f64>::new(1);
        let x = array![[1.0, -2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_transform_shape_mismatch() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0]]; // wrong number of features
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_nmf_transform_negative_rejected() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_neg = array![[1.0, -2.0, 3.0]];
        assert!(fitted.transform(&x_neg).is_err());
    }

    #[test]
    fn test_nmf_reproducibility() {
        let nmf1 = NMF::<f64>::new(2).with_random_state(42);
        let nmf2 = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted1 = nmf1.fit(&x, &()).unwrap();
        let fitted2 = nmf2.fit(&x, &()).unwrap();
        for (a, b) in fitted1.components().iter().zip(fitted2.components().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nmf_single_component() {
        let nmf = NMF::<f64>::new(1).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_nmf_n_iter_positive() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = small_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_nmf_getters() {
        let nmf = NMF::<f64>::new(3)
            .with_max_iter(100)
            .with_tol(1e-5)
            .with_solver(NMFSolver::CoordinateDescent)
            .with_init(NMFInit::Nndsvd)
            .with_random_state(99);
        assert_eq!(nmf.n_components(), 3);
        assert_eq!(nmf.max_iter(), 100);
        assert_abs_diff_eq!(nmf.tol(), 1e-5);
        assert_eq!(nmf.solver(), NMFSolver::CoordinateDescent);
        assert_eq!(nmf.init(), NMFInit::Nndsvd);
        assert_eq!(nmf.random_state(), Some(99));
    }

    #[test]
    fn test_nmf_f32() {
        let nmf = NMF::<f32>::new(1).with_random_state(42);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_nmf_zero_entries() {
        let nmf = NMF::<f64>::new(2).with_random_state(42);
        let x = array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));
    }

    #[test]
    fn test_nmf_pipeline_integration() {
        use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
        use ferrolearn_core::traits::Predict;

        struct SumEstimator;

        impl PipelineEstimator for SumEstimator {
            fn fit_pipeline(
                &self,
                _x: &Array2<f64>,
                _y: &Array1<f64>,
            ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
                Ok(Box::new(FittedSumEstimator))
            }
        }

        struct FittedSumEstimator;

        impl FittedPipelineEstimator for FittedSumEstimator {
            fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
                let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
                Ok(Array1::from_vec(sums))
            }
        }

        let pipeline = Pipeline::new()
            .transform_step("nmf", Box::new(NMF::<f64>::new(2).with_random_state(42)))
            .estimator_step("sum", Box::new(SumEstimator));

        let x = small_dataset();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let fitted = pipeline.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_nmf_medium_dataset_mu() {
        let nmf = NMF::<f64>::new(3)
            .with_solver(NMFSolver::MultiplicativeUpdate)
            .with_random_state(42)
            .with_max_iter(500);
        let x = medium_dataset();
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (3, 4));
        // Reconstruction error should be reasonable.
        assert!(
            fitted.reconstruction_err() < 10.0,
            "reconstruction error too large: {}",
            fitted.reconstruction_err()
        );
    }

    #[test]
    fn test_nmf_insufficient_samples() {
        let nmf = NMF::<f64>::new(1);
        let x = Array2::<f64>::zeros((0, 3));
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_nmf_more_components_lower_error() {
        let nmf1 = NMF::<f64>::new(1).with_random_state(42).with_max_iter(300);
        let nmf2 = NMF::<f64>::new(2).with_random_state(42).with_max_iter(300);
        let x = medium_dataset();
        let fitted1 = nmf1.fit(&x, &()).unwrap();
        let fitted2 = nmf2.fit(&x, &()).unwrap();
        assert!(
            fitted2.reconstruction_err() <= fitted1.reconstruction_err() + 1e-6,
            "more components should reduce error: 1comp={}, 2comp={}",
            fitted1.reconstruction_err(),
            fitted2.reconstruction_err()
        );
    }
}
