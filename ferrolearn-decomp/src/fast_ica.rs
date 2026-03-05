//! Fast Independent Component Analysis (FastICA).
//!
//! FastICA separates a multivariate signal into additive independent components
//! by maximising non-Gaussianity (negentropy approximation).
//!
//! # Algorithm
//!
//! 1. **Centre**: subtract the mean of each feature.
//! 2. **Whiten** (PCA whitening): decorrelate and scale the data so that each
//!    component has unit variance.
//! 3. **FastICA iteration**: for each unmixing direction `w`, iterate:
//!    ```text
//!    w' = E[X g(w^T X)] - E[g'(w^T X)] w
//!    w' = w' / ||w'||
//!    ```
//!    until convergence, using a chosen nonlinearity `g`.
//! 4. Two variants are supported: `Parallel` (update all directions
//!    simultaneously) and `Deflation` (extract one at a time via Gram-Schmidt).
//!
//! # Non-linearities
//!
//! - [`NonLinearity::LogCosh`]: `g(u) = tanh(u)`.
//! - [`NonLinearity::Exp`]: `g(u) = u exp(-u²/2)`.
//! - [`NonLinearity::Cube`]: `g(u) = u³`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::fast_ica::{FastICA, Algorithm, NonLinearity};
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let ica = FastICA::new(2)
//!     .with_algorithm(Algorithm::Deflation)
//!     .with_fun(NonLinearity::LogCosh)
//!     .with_random_state(0);
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
//! ).unwrap();
//! let fitted = ica.fit(&x, &()).unwrap();
//! let sources = fitted.transform(&x).unwrap();
//! assert_eq!(sources.ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// FastICA iteration strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Update all unmixing directions simultaneously.
    Parallel,
    /// Extract one unmixing direction at a time (Gram-Schmidt orthogonalisation).
    Deflation,
}

/// Non-linearity function used to approximate negentropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonLinearity {
    /// `g(u) = tanh(u)`, `g'(u) = 1 - tanh²(u)`.
    LogCosh,
    /// `g(u) = u exp(-u²/2)`, `g'(u) = (1 - u²) exp(-u²/2)`.
    Exp,
    /// `g(u) = u³`, `g'(u) = 3u²`.
    Cube,
}

// ---------------------------------------------------------------------------
// FastICA (unfitted)
// ---------------------------------------------------------------------------

/// FastICA configuration.
///
/// Calling [`Fit::fit`] whitens the data and runs the FastICA algorithm,
/// returning a [`FittedFastICA`].
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type.
#[derive(Debug, Clone)]
pub struct FastICA<F> {
    /// Number of independent components to extract.
    n_components: usize,
    /// Iteration strategy.
    algorithm: Algorithm,
    /// Non-linearity function.
    fun: NonLinearity,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Optional random seed.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> FastICA<F> {
    /// Create a new `FastICA` that extracts `n_components` independent components.
    ///
    /// Defaults: `algorithm = Parallel`, `fun = LogCosh`, `max_iter = 200`,
    /// `tol = 1e-4`, no fixed random seed.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            algorithm: Algorithm::Parallel,
            fun: NonLinearity::LogCosh,
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the iteration strategy.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the non-linearity function.
    #[must_use]
    pub fn with_fun(mut self, fun: NonLinearity) -> Self {
        self.fun = fun;
        self
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

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

impl<F: Float + Send + Sync + 'static> Default for FastICA<F> {
    fn default() -> Self {
        Self::new(1)
    }
}

// ---------------------------------------------------------------------------
// FittedFastICA
// ---------------------------------------------------------------------------

/// A fitted FastICA model.
///
/// Created by calling [`Fit::fit`] on a [`FastICA`].
/// Implements [`Transform<Array2<F>>`] to unmix new signals.
#[derive(Debug, Clone)]
pub struct FittedFastICA<F> {
    /// Unmixing matrix (applied after whitening), shape `(n_components, n_components_white)`.
    ///
    /// To recover sources from whitened data: `S = unmixing @ X_white`.
    components: Array2<F>,

    /// Mixing matrix (pseudo-inverse of the unmixing), shape `(n_features, n_components)`.
    mixing: Array2<F>,

    /// Per-feature mean, shape `(n_features,)`.
    mean: Array1<F>,

    /// Whitening matrix, shape `(n_components, n_features)`.
    whitening: Array2<F>,

    /// Number of iterations performed.
    n_iter: usize,

    /// Number of features seen during fitting.
    n_features: usize,
}

impl<F: Float + Send + Sync + 'static> FittedFastICA<F> {
    /// Unmixing matrix applied to whitened data, shape `(n_components, n_components)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components
    }

    /// Mixing matrix (approximate pseudo-inverse of unmixing + whitening).
    #[must_use]
    pub fn mixing(&self) -> &Array2<F> {
        &self.mixing
    }

    /// Per-feature mean learned during fitting.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean
    }

    /// Number of iterations performed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply the non-linearity `g` and its derivative `g'` element-wise.
///
/// Returns `(g_vals, g_prime_vals)`.
fn apply_nonlinearity<F: Float>(u: &Array1<F>, fun: NonLinearity) -> (Array1<F>, Array1<F>) {
    let n = u.len();
    let mut g_vals = Array1::<F>::zeros(n);
    let mut gp_vals = Array1::<F>::zeros(n);
    let half = F::from(0.5).unwrap();
    for i in 0..n {
        let ui = u[i];
        match fun {
            NonLinearity::LogCosh => {
                // g(u) = tanh(u)
                // Use the formula: tanh(x) = (e^2x - 1)/(e^2x + 1)
                let t = if ui > F::from(20.0).unwrap() {
                    F::one()
                } else if ui < F::from(-20.0).unwrap() {
                    -F::one()
                } else {
                    let e2 = (ui * F::from(2.0).unwrap()).exp();
                    (e2 - F::one()) / (e2 + F::one())
                };
                g_vals[i] = t;
                gp_vals[i] = F::one() - t * t;
            }
            NonLinearity::Exp => {
                // g(u) = u exp(-u²/2)
                let neg_u2_half = -(ui * ui) * half;
                let exp_v = neg_u2_half.exp();
                g_vals[i] = ui * exp_v;
                gp_vals[i] = (F::one() - ui * ui) * exp_v;
            }
            NonLinearity::Cube => {
                // g(u) = u³
                g_vals[i] = ui * ui * ui;
                gp_vals[i] = F::from(3.0).unwrap() * ui * ui;
            }
        }
    }
    (g_vals, gp_vals)
}

/// Compute `g` and mean of `g'` for all samples.
///
/// `x_white_w`: the projections `W_row @ X_white`, shape `(n_samples,)`.
/// Returns `(mean_g_prime, g_vals)` where `g_vals` has shape `(n_samples,)`.
fn ica_step_values<F: Float>(projections: &Array1<F>, fun: NonLinearity) -> (F, Array1<F>) {
    let (g_vals, gp_vals) = apply_nonlinearity(projections, fun);
    let n_f = F::from(projections.len()).unwrap();
    let mean_gp = gp_vals.iter().copied().fold(F::zero(), |a, b| a + b) / n_f;
    (mean_gp, g_vals)
}

/// Gram-Schmidt orthogonalisation of `W` (row vectors).
fn gs_orthogonalise<F: Float>(w: &mut Array2<F>, col: usize) {
    let k = col;
    // w[k] -= sum_{j<k} (w[k] . w[j]) w[j]
    for j in 0..k {
        let dot = (0..w.ncols())
            .map(|d| w[[k, d]] * w[[j, d]])
            .fold(F::zero(), |a, b| a + b);
        for d in 0..w.ncols() {
            let wd = w[[j, d]];
            w[[k, d]] = w[[k, d]] - dot * wd;
        }
    }
    // Normalise.
    let norm = (0..w.ncols())
        .map(|d| w[[k, d]] * w[[k, d]])
        .fold(F::zero(), |a, b| a + b)
        .sqrt();
    if norm > F::from(1e-15).unwrap() {
        for d in 0..w.ncols() {
            w[[k, d]] = w[[k, d]] / norm;
        }
    }
}

/// Symmetric orthogonalisation: W ← (W W^T)^{-1/2} W.
fn sym_orthogonalise<F: Float + Send + Sync + 'static>(
    w: &mut Array2<F>,
) -> Result<(), FerroError> {
    let k = w.nrows();
    // Compute S = W W^T (k × k).
    let mut s = Array2::<F>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let dot = (0..w.ncols())
                .map(|d| w[[i, d]] * w[[j, d]])
                .fold(F::zero(), |a, b| a + b);
            s[[i, j]] = dot;
        }
    }
    // Eigendecompose S = V D V^T.
    let max_iter = k * k * 100 + 1000;
    let (eigenvalues, eigenvectors) = jacobi_eigen_small(&s, max_iter)?;
    // W_new = V D^{-1/2} V^T W
    // = Σ_i (1/sqrt(d_i)) (v_i v_i^T) W
    let mut w_new = Array2::<F>::zeros((k, w.ncols()));
    let eps = F::from(1e-10).unwrap();
    for i in 0..k {
        let d = eigenvalues[i];
        let scale = if d > eps {
            F::one() / d.sqrt()
        } else {
            F::one()
        };
        // v_i is column i of eigenvectors.
        // outer product: v_i v_i^T W = v_i (v_i^T W)
        // (v_i^T W) is a row vector of shape (1, n_comp).
        let mut vi_t_w = Array1::<F>::zeros(w.ncols());
        for d_idx in 0..k {
            for col in 0..w.ncols() {
                vi_t_w[col] = vi_t_w[col] + eigenvectors[[d_idx, i]] * w[[d_idx, col]];
            }
        }
        for row in 0..k {
            for col in 0..w.ncols() {
                w_new[[row, col]] =
                    w_new[[row, col]] + scale * eigenvectors[[row, i]] * vi_t_w[col];
            }
        }
    }
    *w = w_new;
    Ok(())
}

/// Jacobi eigendecomposition for a small k×k symmetric matrix.
fn jacobi_eigen_small<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }
    let tol = F::from(1e-12).unwrap_or(F::epsilon());
    let two = F::from(2.0).unwrap();
    for _ in 0..max_iter {
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
            let tau = (aqq - app) / (two * apq);
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
        new_mat[[p, p]] = c * c * app - two * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + two * s * c * apq + c * c * aqq;
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
    // Didn't fully converge, but return best estimate.
    let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
    Ok((eigenvalues, v))
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for FastICA<F> {
    type Fitted = FittedFastICA<F>;
    type Error = FerroError;

    /// Fit FastICA to data.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or
    ///   exceeds `n_features`.
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples are provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedFastICA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "FastICA requires at least 2 samples".into(),
            });
        }

        let k = self.n_components;
        let n_f = F::from(n_samples).unwrap();

        // --- Step 1: Centre --------------------------------------------------
        let mut mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let s = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
            mean[j] = s / n_f;
        }
        let mut xc = x.to_owned();
        for mut row in xc.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }

        // --- Step 2: Whiten (PCA) -------------------------------------------
        // Covariance matrix C = X_c^T X_c / n  (n_features × n_features)
        let cov = xc.t().dot(&xc).mapv(|v| v / n_f);

        // Eigendecompose C.
        let max_jacobi = n_features * n_features * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_small(&cov, max_jacobi)?;

        // Sort descending.
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build whitening matrix K: k × n_features.
        // K[i, :] = eigenvectors[:, indices[i]] / sqrt(eigenvalues[indices[i]])
        let eps = F::from(1e-10).unwrap();
        let mut whitening = Array2::<F>::zeros((k, n_features));
        for i in 0..k {
            let idx = indices[i];
            let ev = eigenvalues[idx];
            let scale = if ev > eps {
                F::one() / ev.sqrt()
            } else {
                F::zero()
            };
            for j in 0..n_features {
                whitening[[i, j]] = eigenvectors[[j, idx]] * scale;
            }
        }

        // Whitened data X_w = K @ X_c^T  (k × n_samples), then transpose to n × k.
        let x_white_t = whitening.dot(&xc.t()); // k × n
        let x_white = x_white_t.t().to_owned(); // n × k

        // --- Step 3: FastICA -------------------------------------------------
        let seed = self.random_state.unwrap_or(42);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let std_normal = StandardNormal;

        // Initialise W as a k × k random matrix (rows are unmixing directions).
        let mut w = Array2::<F>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                let v: f64 = std_normal.sample(&mut rng);
                w[[i, j]] = F::from(v).unwrap();
            }
        }
        // Orthogonalise initial W.
        sym_orthogonalise(&mut w)?;

        let tol_f = F::from(self.tol).unwrap();
        let mut n_iter = 0usize;

        match self.algorithm {
            Algorithm::Parallel => {
                for iter in 0..self.max_iter {
                    let mut w_new = Array2::<F>::zeros((k, k));
                    // For each component i, update using all samples.
                    for i in 0..k {
                        // Projection: u = X_w @ w[i]  (n_samples,)
                        let w_row: Array1<F> = w.row(i).to_owned();
                        let u: Array1<F> = x_white.dot(&w_row);
                        let (mean_gp, g_vals) = ica_step_values(&u, self.fun);
                        // w_new[i] = (1/n) X_w^T g(u) - mean_g' * w[i]
                        // X_w^T g(u) = sum_t x_w[t] g(u[t])  (k-vector)
                        let mut xw_t_g = Array1::<F>::zeros(k);
                        for t in 0..n_samples {
                            for d in 0..k {
                                xw_t_g[d] = xw_t_g[d] + x_white[[t, d]] * g_vals[t];
                            }
                        }
                        for d in 0..k {
                            xw_t_g[d] = xw_t_g[d] / n_f;
                        }
                        for d in 0..k {
                            w_new[[i, d]] = xw_t_g[d] - mean_gp * w_row[d];
                        }
                    }
                    // Symmetric orthogonalisation.
                    sym_orthogonalise(&mut w_new)?;

                    // Convergence: max |1 - |w_new[i] . w[i]||
                    let mut max_change = F::zero();
                    for i in 0..k {
                        let dot: F = (0..k)
                            .map(|d| w_new[[i, d]] * w[[i, d]])
                            .fold(F::zero(), |a, b| a + b);
                        let change = (F::one() - dot.abs()).abs();
                        if change > max_change {
                            max_change = change;
                        }
                    }
                    w = w_new;
                    n_iter = iter + 1;
                    if max_change < tol_f {
                        break;
                    }
                }
            }
            Algorithm::Deflation => {
                for i in 0..k {
                    for iter in 0..self.max_iter {
                        // Projection: u = X_w @ w[i]  (n_samples,)
                        let w_row: Array1<F> = w.row(i).to_owned();
                        let u: Array1<F> = x_white.dot(&w_row);
                        let (mean_gp, g_vals) = ica_step_values(&u, self.fun);
                        // w_new = (1/n) X_w^T g(u) - mean_g' * w[i]
                        let mut w_new_row = Array1::<F>::zeros(k);
                        for t in 0..n_samples {
                            for d in 0..k {
                                w_new_row[d] = w_new_row[d] + x_white[[t, d]] * g_vals[t];
                            }
                        }
                        for d in 0..k {
                            w_new_row[d] = w_new_row[d] / n_f - mean_gp * w_row[d];
                        }
                        // Gram-Schmidt orthogonalisation.
                        for j in 0..i {
                            let dot: F = (0..k)
                                .map(|d| w_new_row[d] * w[[j, d]])
                                .fold(F::zero(), |a, b| a + b);
                            for d in 0..k {
                                let wd = w[[j, d]];
                                w_new_row[d] = w_new_row[d] - dot * wd;
                            }
                        }
                        // Normalise.
                        let norm = w_new_row
                            .iter()
                            .copied()
                            .map(|v| v * v)
                            .fold(F::zero(), |a, b| a + b)
                            .sqrt();
                        if norm > F::from(1e-15).unwrap() {
                            w_new_row.mapv_inplace(|v| v / norm);
                        }
                        // Convergence: |1 - |w_new . w_old||
                        let dot: F = (0..k)
                            .map(|d| w_new_row[d] * w_row[d])
                            .fold(F::zero(), |a, b| a + b);
                        let change = (F::one() - dot.abs()).abs();
                        for d in 0..k {
                            w[[i, d]] = w_new_row[d];
                        }
                        n_iter = iter + 1;
                        if change < tol_f {
                            break;
                        }
                    }
                    // Gram-Schmidt after finalising component i.
                    gs_orthogonalise(&mut w, i);
                }
            }
        }

        // --- Mixing matrix ---------------------------------------------------
        // The full unmixing pipeline is: s = W @ K @ (x - mean)
        // where K is the whitening matrix (k × n_features), W is k × k.
        // The mixing matrix M satisfies s ≈ W K x_c, so x_c ≈ K^T W^T s (Moore-Penrose pseudo-inverse).
        // mixing = K^T W^T  (n_features × k)
        let mixing = whitening.t().dot(&w.t()); // n_features × k

        Ok(FittedFastICA {
            components: w,
            mixing,
            mean,
            whitening,
            n_iter,
            n_features,
        })
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedFastICA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Unmix new signals: `S = (W @ K @ (X - mean)^T)^T`.
    ///
    /// Returns an array of shape `(n_samples, n_components)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the model.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedFastICA::transform".into(),
            });
        }
        // Centre.
        let mut xc = x.to_owned();
        for mut row in xc.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean.iter()) {
                *v = *v - m;
            }
        }
        // Whiten: X_w = K @ X_c^T  (k × n), transpose to n × k.
        let x_white = self.whitening.dot(&xc.t()).t().to_owned(); // n × k
        // Unmix: S = (W @ X_w^T)^T = X_w @ W^T  (n × k)
        let sources = x_white.dot(&self.components.t());
        Ok(sources)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for FastICA<F> {
    /// Fit using the pipeline interface (ignores `y`).
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedFastICA<F> {
    /// Transform via the pipeline interface.
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
    use ndarray::Array2;

    fn mixed_signals() -> Array2<f64> {
        // Two synthetic source signals, then mixed.
        let n = 50;
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.2;
            // source 1: sine wave, source 2: sawtooth
            let s1 = t.sin();
            let s2 = (t * 0.5).cos();
            // mixing matrix
            x[[i, 0]] = 0.5 * s1 + 0.5 * s2;
            x[[i, 1]] = 0.2 * s1 + 0.8 * s2;
        }
        x
    }

    #[test]
    fn test_ica_fit_returns_fitted() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 2));
    }

    #[test]
    fn test_ica_transform_shape() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let sources = fitted.transform(&x).unwrap();
        assert_eq!(sources.dim(), (50, 2));
    }

    #[test]
    fn test_ica_parallel_algorithm() {
        let ica = FastICA::<f64>::new(2)
            .with_algorithm(Algorithm::Parallel)
            .with_random_state(1);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 2);
    }

    #[test]
    fn test_ica_deflation_algorithm() {
        let ica = FastICA::<f64>::new(2)
            .with_algorithm(Algorithm::Deflation)
            .with_random_state(2);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 2);
    }

    #[test]
    fn test_ica_logcosh() {
        let ica = FastICA::<f64>::new(2)
            .with_fun(NonLinearity::LogCosh)
            .with_random_state(3);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let s = fitted.transform(&x).unwrap();
        assert_eq!(s.ncols(), 2);
    }

    #[test]
    fn test_ica_exp() {
        let ica = FastICA::<f64>::new(2)
            .with_fun(NonLinearity::Exp)
            .with_random_state(4);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let s = fitted.transform(&x).unwrap();
        assert_eq!(s.ncols(), 2);
    }

    #[test]
    fn test_ica_cube() {
        let ica = FastICA::<f64>::new(2)
            .with_fun(NonLinearity::Cube)
            .with_random_state(5);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let s = fitted.transform(&x).unwrap();
        assert_eq!(s.ncols(), 2);
    }

    #[test]
    fn test_ica_n_iter_positive() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() >= 1);
    }

    #[test]
    fn test_ica_mixing_shape() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert_eq!(fitted.mixing().dim(), (2, 2));
    }

    #[test]
    fn test_ica_mean_shape() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        assert_eq!(fitted.mean().len(), 2);
    }

    #[test]
    fn test_ica_transform_shape_mismatch() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let x_bad = Array2::<f64>::zeros((3, 5));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_ica_error_zero_components() {
        let ica = FastICA::<f64>::new(0);
        let x = mixed_signals();
        assert!(ica.fit(&x, &()).is_err());
    }

    #[test]
    fn test_ica_error_too_many_components() {
        let ica = FastICA::<f64>::new(10); // n_features = 2
        let x = mixed_signals();
        assert!(ica.fit(&x, &()).is_err());
    }

    #[test]
    fn test_ica_error_insufficient_samples() {
        let ica = FastICA::<f64>::new(1);
        let x = Array2::<f64>::zeros((1, 2));
        assert!(ica.fit(&x, &()).is_err());
    }

    #[test]
    fn test_ica_single_component() {
        let ica = FastICA::<f64>::new(1).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let s = fitted.transform(&x).unwrap();
        assert_eq!(s.dim(), (50, 1));
    }

    #[test]
    fn test_ica_sources_not_all_zero() {
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let fitted = ica.fit(&x, &()).unwrap();
        let s = fitted.transform(&x).unwrap();
        let total: f64 = s.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0);
    }

    #[test]
    fn test_ica_reproducible_with_seed() {
        let ica1 = FastICA::<f64>::new(2).with_random_state(7);
        let ica2 = FastICA::<f64>::new(2).with_random_state(7);
        let x = mixed_signals();
        let f1 = ica1.fit(&x, &()).unwrap();
        let f2 = ica2.fit(&x, &()).unwrap();
        for (a, b) in f1.components().iter().zip(f2.components().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_ica_pipeline_transformer() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let ica = FastICA::<f64>::new(2).with_random_state(0);
        let x = mixed_signals();
        let y = Array1::<f64>::zeros(50);
        let fitted = ica.fit_pipeline(&x, &y).unwrap();
        let out = fitted.transform_pipeline(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_ica_n_components_getter() {
        let ica = FastICA::<f64>::new(3);
        assert_eq!(ica.n_components(), 3);
    }

    #[test]
    fn test_ica_nonlinearity_values() {
        // Check g(0) = 0 for all non-linearities.
        let u = Array1::from_vec(vec![0.0f64]);
        let (g_lc, _) = apply_nonlinearity(&u, NonLinearity::LogCosh);
        let (g_exp, _) = apply_nonlinearity(&u, NonLinearity::Exp);
        let (g_cube, _) = apply_nonlinearity(&u, NonLinearity::Cube);
        assert_abs_diff_eq!(g_lc[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g_exp[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g_cube[0], 0.0, epsilon = 1e-10);
    }
}
