//! Gaussian Mixture Models (GMM) using the Expectation-Maximisation algorithm.
//!
//! This module provides [`GaussianMixture`], an unsupervised soft-clustering
//! algorithm that models the data as a weighted sum of `K` multivariate
//! Gaussian distributions.  The implementation supports four covariance
//! structures:
//!
//! | [`CovarianceType`] | Parameters per component | Notes |
//! |--------------------|--------------------------|-------|
//! | `Full`             | d × d matrix             | Most flexible, most expensive |
//! | `Tied`             | one shared d × d matrix  | All components share the same covariance |
//! | `Diag`             | d diagonal elements      | Diagonal covariance per component |
//! | `Spherical`        | 1 scalar σ²              | Isotropic covariance per component |
//!
//! # Algorithm
//!
//! The EM algorithm alternates between:
//!
//! 1. **E-step** – compute the *responsibility* `r[n, k]` (probability that
//!    sample `n` belongs to component `k`) using the current parameters.
//! 2. **M-step** – update the mixture weights, means and covariances by
//!    maximising the expected complete-data log-likelihood.
//!
//! Convergence is declared when the change in average log-likelihood between
//! two consecutive iterations falls below `tol`.  The algorithm is run
//! `n_init` times; the run with the highest final log-likelihood is kept.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::GaussianMixture;
//! use ferrolearn_core::{Fit, Predict, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     8.0, 8.0,  8.1, 8.0,  8.0, 8.1,
//! ]).unwrap();
//!
//! let model = GaussianMixture::<f64>::new(2).with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.predict(&x).unwrap();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Public enums & configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// The structure assumed for each component's covariance matrix.
///
/// Choosing a simpler structure reduces the number of free parameters and
/// can improve numerical stability on small datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceType {
    /// Each component has its own full `d × d` covariance matrix.
    Full,
    /// All components share one full `d × d` covariance matrix.
    Tied,
    /// Each component has its own diagonal covariance (variances only,
    /// no covariances), stored as a length-`d` vector.
    Diag,
    /// Each component has a single scalar variance; the covariance matrix
    /// is `σ² I`.
    Spherical,
}

/// Gaussian Mixture Model configuration (unfitted).
///
/// Call [`Fit::fit`] to run the EM algorithm and obtain a
/// [`FittedGaussianMixture`].
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GaussianMixture<F> {
    /// Number of mixture components.
    pub n_components: usize,
    /// Covariance parameterisation.
    pub covariance_type: CovarianceType,
    /// Maximum number of EM iterations per run.
    pub max_iter: usize,
    /// Convergence tolerance on the average log-likelihood change.
    pub tol: F,
    /// Number of independent EM runs.  The best result (highest
    /// log-likelihood) is returned.
    pub n_init: usize,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
}

impl<F: Float> GaussianMixture<F> {
    /// Create a new `GaussianMixture` with `n_components` components.
    ///
    /// Defaults: `covariance_type = Full`, `max_iter = 100`, `tol = 1e-3`,
    /// `n_init = 1`, `random_state = None`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: CovarianceType::Full,
            max_iter: 100,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            n_init: 1,
            random_state: None,
        }
    }

    /// Set the covariance type.
    #[must_use]
    pub fn with_covariance_type(mut self, cov: CovarianceType) -> Self {
        self.covariance_type = cov;
        self
    }

    /// Set the maximum number of EM iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of independent EM runs.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Gaussian Mixture Model.
///
/// Stores the learned mixture weights, means and covariance representations
/// produced by the EM algorithm.  Implements:
///
/// * [`Predict`] — returns the most-probable component index per sample.
/// * [`Transform`] — returns the full responsibility matrix
///   (shape `(n_samples, n_components)`).
#[derive(Debug, Clone)]
pub struct FittedGaussianMixture<F> {
    /// Mixture weights, shape `(n_components,)`.  Sum to 1.
    pub weights_: Array1<F>,
    /// Component means, shape `(n_components, n_features)`.
    pub means_: Array2<F>,
    /// Covariance parameters:
    ///
    /// * `Full`/`Tied` → shape `(n_components * n_features, n_features)`, i.e.
    ///   `n_components` stacked full matrices (for `Tied` all rows are equal).
    /// * `Diag`        → shape `(n_components, n_features)`, each row = variances.
    /// * `Spherical`   → shape `(n_components, 1)`, each row = single variance.
    pub covariances_: Array2<F>,
    /// `true` if the EM algorithm converged within `max_iter` iterations.
    pub converged_: bool,
    /// Final average log-likelihood of the training data.
    pub lower_bound_: F,
    /// The covariance type used during fitting.
    covariance_type_: CovarianceType,
    /// Number of features seen during fitting.
    n_features_: usize,
}

impl<F: Float> FittedGaussianMixture<F> {
    /// Return the mixture weights (shape `(n_components,)`).
    #[must_use]
    pub fn weights(&self) -> &Array1<F> {
        &self.weights_
    }

    /// Return the component means (shape `(n_components, n_features)`).
    #[must_use]
    pub fn means(&self) -> &Array2<F> {
        &self.means_
    }

    /// Return the covariance parameters (layout depends on
    /// [`CovarianceType`]).
    #[must_use]
    pub fn covariances(&self) -> &Array2<F> {
        &self.covariances_
    }

    /// Whether the EM algorithm converged.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged_
    }

    /// Final average log-likelihood of the training data.
    #[must_use]
    pub fn lower_bound(&self) -> F {
        self.lower_bound_
    }

    /// Compute the *Bayesian Information Criterion* for model selection.
    ///
    /// Lower BIC indicates a better model (balancing fit and complexity).
    ///
    /// `BIC = -2 · log_likelihood · n_samples + n_params · ln(n_samples)`
    #[must_use]
    pub fn bic(&self, n_samples: usize) -> F {
        let n = F::from(n_samples).unwrap_or(F::one());
        let log_n = n.ln();
        let params = F::from(self.n_free_params()).unwrap_or(F::one());
        -F::from(2.0).unwrap() * self.lower_bound_ * n + params * log_n
    }

    /// Compute the *Akaike Information Criterion* for model selection.
    ///
    /// Lower AIC indicates a better model.
    ///
    /// `AIC = -2 · log_likelihood · n_samples + 2 · n_params`
    #[must_use]
    pub fn aic(&self, n_samples: usize) -> F {
        let n = F::from(n_samples).unwrap_or(F::one());
        let two = F::from(2.0).unwrap();
        let params = F::from(self.n_free_params()).unwrap_or(F::one());
        -two * self.lower_bound_ * n + two * params
    }

    /// Number of free parameters in the model.
    fn n_free_params(&self) -> usize {
        let k = self.weights_.len();
        let d = self.n_features_;
        let cov_params = match self.covariance_type_ {
            CovarianceType::Full => k * d * (d + 1) / 2,
            CovarianceType::Tied => d * (d + 1) / 2,
            CovarianceType::Diag => k * d,
            CovarianceType::Spherical => k,
        };
        // means + covariance + (k-1) for mixture weights
        k * d + cov_params + (k - 1)
    }

    /// Compute log-responsibilities (log r[n, k]) for all samples.
    ///
    /// Returns a matrix of shape `(n_samples, n_components)`.
    fn log_responsibilities(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let k = self.weights_.len();

        if n_features != self.n_features_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features_],
                actual: vec![n_features],
                context: "number of features must match fitted GaussianMixture".into(),
            });
        }

        let mut log_resp = Array2::zeros((n_samples, k));
        let two_pi = F::from(std::f64::consts::TAU).unwrap();

        for ki in 0..k {
            let log_w = self.weights_[ki].ln();
            let mean = self.means_.row(ki);

            let log_det;
            let log_norm;

            match self.covariance_type_ {
                CovarianceType::Full | CovarianceType::Tied => {
                    let cov_offset = ki * n_features;
                    // Extract the covariance block for this component.
                    let cov_block = self
                        .covariances_
                        .slice(ndarray::s![cov_offset..cov_offset + n_features, ..]);
                    let (ld, ln) =
                        log_det_and_norm_full(&cov_block.to_owned(), n_features, two_pi)?;
                    log_det = ld;
                    log_norm = ln;

                    for ni in 0..n_samples {
                        let diff: Vec<F> = (0..n_features).map(|j| x[[ni, j]] - mean[j]).collect();
                        let maha = mahalanobis_full(&diff, &cov_block.to_owned(), n_features)?;
                        log_resp[[ni, ki]] =
                            log_w + log_norm - F::from(0.5).unwrap() * (log_det + maha);
                    }
                }
                CovarianceType::Diag => {
                    let variances = self.covariances_.row(ki);
                    // log|Σ| = sum(log σ²_j)
                    log_det = variances.iter().fold(F::zero(), |acc, &v| acc + v.ln());
                    // normalisation: -d/2 * log(2π) - 1/2 * log|Σ|
                    log_norm = -F::from(n_features as f64 / 2.0).unwrap() * two_pi.ln()
                        - F::from(0.5).unwrap() * log_det;

                    for ni in 0..n_samples {
                        let maha: F = (0..n_features).fold(F::zero(), |acc, j| {
                            let d = x[[ni, j]] - mean[j];
                            acc + d * d / variances[j]
                        });
                        log_resp[[ni, ki]] = log_w + log_norm - F::from(0.5).unwrap() * maha;
                    }
                }
                CovarianceType::Spherical => {
                    let var = self.covariances_[[ki, 0]];
                    // log|Σ| = d * log(σ²)
                    log_det = F::from(n_features as f64).unwrap() * var.ln();
                    log_norm = -F::from(n_features as f64 / 2.0).unwrap() * two_pi.ln()
                        - F::from(0.5).unwrap() * log_det;

                    for ni in 0..n_samples {
                        let sq: F = (0..n_features).fold(F::zero(), |acc, j| {
                            let d = x[[ni, j]] - mean[j];
                            acc + d * d
                        });
                        let maha = sq / var;
                        log_resp[[ni, ki]] = log_w + log_norm - F::from(0.5).unwrap() * maha;
                    }
                }
            }
        }

        Ok(log_resp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute log|Σ| and the log-normalisation constant for a full covariance matrix.
///
/// Uses a straightforward Cholesky-like decomposition for small `d`.
fn log_det_and_norm_full<F: Float>(
    cov: &Array2<F>,
    d: usize,
    two_pi: F,
) -> Result<(F, F), FerroError> {
    // Compute log-det via Cholesky factor diagonals.
    let chol = cholesky(cov, d)?;
    let mut log_det = F::zero();
    for i in 0..d {
        if chol[[i, i]] <= F::zero() {
            return Err(FerroError::NumericalInstability {
                message: "covariance matrix is not positive definite".into(),
            });
        }
        log_det = log_det + chol[[i, i]].ln();
    }
    // For real Cholesky: log|Σ| = 2 * sum(log(diag(L)))
    log_det = log_det + log_det;
    let log_norm =
        -F::from(d as f64 / 2.0).unwrap() * two_pi.ln() - F::from(0.5).unwrap() * log_det;
    Ok((log_det, log_norm))
}

/// Compute the Mahalanobis distance `diff^T Σ^{-1} diff` for a full matrix.
///
/// Solves via the Cholesky factor so we can reuse it.
fn mahalanobis_full<F: Float>(diff: &[F], cov: &Array2<F>, d: usize) -> Result<F, FerroError> {
    let chol = cholesky(cov, d)?;
    // Forward substitution: solve L y = diff.
    let mut y = vec![F::zero(); d];
    for i in 0..d {
        let mut s = diff[i];
        for j in 0..i {
            s = s - chol[[i, j]] * y[j];
        }
        if chol[[i, i]] == F::zero() {
            return Err(FerroError::NumericalInstability {
                message: "covariance matrix has zero diagonal in Cholesky".into(),
            });
        }
        y[i] = s / chol[[i, i]];
    }
    Ok(y.iter().fold(F::zero(), |acc, &v| acc + v * v))
}

/// Compute the lower-triangular Cholesky factor `L` such that `Σ = L L^T`.
///
/// Adds a small regularisation `reg = 1e-6` to the diagonal to ensure
/// positive definiteness in the presence of numerical noise.
fn cholesky<F: Float>(cov: &Array2<F>, d: usize) -> Result<Array2<F>, FerroError> {
    let reg = F::from(1e-6).unwrap_or(F::epsilon());
    let mut l = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = cov[[i, j]];
            if i == j {
                s = s + reg;
            }
            for p in 0..j {
                s = s - l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: format!("covariance not positive-definite at diagonal [{i},{i}]"),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                if l[[j, j]] == F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky: zero diagonal element".into(),
                    });
                }
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// log-sum-exp along axis 1 (for each row).
///
/// Returns `(log_resp_normalised, log_probs)` where `log_probs[n]` is
/// `log Σ_k exp(log_resp[n, k])`.
fn log_sum_exp_rows<F: Float>(log_resp: &Array2<F>) -> (Array2<F>, Array1<F>) {
    let n_samples = log_resp.nrows();
    let k = log_resp.ncols();
    let mut log_probs = Array1::zeros(n_samples);
    let mut normalised = Array2::zeros((n_samples, k));

    for n in 0..n_samples {
        // Find max for numerical stability.
        let max_val = (0..k)
            .map(|ki| log_resp[[n, ki]])
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });

        let sum_exp: F = (0..k).fold(F::zero(), |acc, ki| {
            acc + (log_resp[[n, ki]] - max_val).exp()
        });
        let lse = max_val + sum_exp.ln();
        log_probs[n] = lse;
        for ki in 0..k {
            normalised[[n, ki]] = log_resp[[n, ki]] - lse;
        }
    }
    (normalised, log_probs)
}

// ─────────────────────────────────────────────────────────────────────────────
// EM internals
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise component means by sampling `k` distinct rows from `x`.
fn init_means<F: Float>(x: &Array2<F>, k: usize, rng: &mut StdRng) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut means = Array2::zeros((k, n_features));

    // Pick k random indices (with possible repetition if k > n_samples).
    for ki in 0..k {
        let idx = rng.random_range(0..n_samples);
        means.row_mut(ki).assign(&x.row(idx));
        // Add a tiny jitter to avoid degenerate covariances.
        for j in 0..n_features {
            let jitter: f64 = rng.random_range(-1e-4..1e-4);
            means[[ki, j]] = means[[ki, j]] + F::from(jitter).unwrap_or(F::zero());
        }
    }
    means
}

/// Build a regularised initial full covariance (or tied, which reuses one block).
fn init_full_cov<F: Float>(n_features: usize) -> Array2<F> {
    let mut cov = Array2::zeros((n_features, n_features));
    let reg = F::from(1.0).unwrap_or(F::one());
    for j in 0..n_features {
        cov[[j, j]] = reg;
    }
    cov
}

/// Run one complete EM cycle.  Returns `(fitted, log_likelihood)`.
#[allow(clippy::too_many_lines)]
fn run_em<F: Float>(
    x: &Array2<F>,
    n_components: usize,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: F,
    rng: &mut StdRng,
) -> Result<FittedGaussianMixture<F>, FerroError> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let k = n_components;

    // ── Initialise parameters ────────────────────────────────────────────────
    let mut weights = Array1::from_elem(k, F::from(1.0 / k as f64).unwrap());
    let mut means = init_means(x, k, rng);

    // Initialise covariances.
    let mut covariances: Array2<F> = match covariance_type {
        CovarianceType::Full => {
            let mut c = Array2::zeros((k * n_features, n_features));
            for ki in 0..k {
                let block = init_full_cov(n_features);
                let offset = ki * n_features;
                c.slice_mut(ndarray::s![offset..offset + n_features, ..])
                    .assign(&block);
            }
            c
        }
        CovarianceType::Tied => {
            // One shared block; store k identical copies for indexing simplicity.
            let block = init_full_cov(n_features);
            let mut c = Array2::zeros((k * n_features, n_features));
            for ki in 0..k {
                let offset = ki * n_features;
                c.slice_mut(ndarray::s![offset..offset + n_features, ..])
                    .assign(&block);
            }
            c
        }
        CovarianceType::Diag => Array2::from_elem((k, n_features), F::one()),
        CovarianceType::Spherical => Array2::from_elem((k, 1), F::one()),
    };

    let mut prev_ll = F::neg_infinity();
    let mut converged = false;

    for _iter in 0..max_iter {
        // ── E-step ────────────────────────────────────────────────────────────
        // Build a temporary fitted model just to call log_responsibilities.
        let tmp = FittedGaussianMixture {
            weights_: weights.clone(),
            means_: means.clone(),
            covariances_: covariances.clone(),
            converged_: false,
            lower_bound_: prev_ll,
            covariance_type_: covariance_type,
            n_features_: n_features,
        };
        let log_resp_raw = tmp.log_responsibilities(x)?;
        let (log_resp, log_probs) = log_sum_exp_rows(&log_resp_raw);

        // Average log-likelihood.
        let ll: F =
            log_probs.iter().fold(F::zero(), |acc, &v| acc + v) / F::from(n_samples).unwrap();

        if (ll - prev_ll).abs() < tol {
            converged = true;
            prev_ll = ll;
            break;
        }
        prev_ll = ll;

        // Responsibilities in linear scale: r[n, k] = exp(log_resp[n, k]).
        let resp: Array2<F> = log_resp.mapv(|v| v.exp());

        // ── M-step ────────────────────────────────────────────────────────────
        // Effective counts N_k = Σ_n r[n, k].
        let nk: Array1<F> = (0..k)
            .map(|ki| resp.column(ki).iter().fold(F::zero(), |acc, &v| acc + v))
            .collect::<Array1<F>>();

        let reg_nk = F::from(10.0 * f64::EPSILON).unwrap();

        // Update weights.
        let total: F = nk.iter().fold(F::zero(), |acc, &v| acc + v);
        for ki in 0..k {
            weights[ki] = (nk[ki] + reg_nk) / (total + F::from(k).unwrap() * reg_nk);
        }

        // Update means.
        for ki in 0..k {
            let nki = nk[ki] + reg_nk;
            for j in 0..n_features {
                let s: F = (0..n_samples).fold(F::zero(), |acc, n| acc + resp[[n, ki]] * x[[n, j]]);
                means[[ki, j]] = s / nki;
            }
        }

        // Update covariances.
        match covariance_type {
            CovarianceType::Full => {
                for ki in 0..k {
                    let nki = nk[ki] + reg_nk;
                    let offset = ki * n_features;
                    let mut cov_k = Array2::<F>::zeros((n_features, n_features));
                    for n in 0..n_samples {
                        let r = resp[[n, ki]];
                        for i in 0..n_features {
                            let di = x[[n, i]] - means[[ki, i]];
                            for j in 0..=i {
                                let dj = x[[n, j]] - means[[ki, j]];
                                cov_k[[i, j]] = cov_k[[i, j]] + r * di * dj;
                            }
                        }
                    }
                    // Symmetrise and normalise.
                    for i in 0..n_features {
                        cov_k[[i, i]] = cov_k[[i, i]] / nki;
                        for j in 0..i {
                            cov_k[[i, j]] = cov_k[[i, j]] / nki;
                            cov_k[[j, i]] = cov_k[[i, j]];
                        }
                    }
                    covariances
                        .slice_mut(ndarray::s![offset..offset + n_features, ..])
                        .assign(&cov_k);
                }
            }
            CovarianceType::Tied => {
                // Weighted sum across all components.
                let mut cov_tied = Array2::<F>::zeros((n_features, n_features));
                let total_nk = nk.iter().fold(F::zero(), |acc, &v| acc + v) + reg_nk;
                for ki in 0..k {
                    let nki = nk[ki];
                    for n in 0..n_samples {
                        let r = resp[[n, ki]];
                        for i in 0..n_features {
                            let di = x[[n, i]] - means[[ki, i]];
                            for j in 0..=i {
                                let dj = x[[n, j]] - means[[ki, j]];
                                cov_tied[[i, j]] = cov_tied[[i, j]] + r * di * dj;
                                let _ = nki; // used in outer scope
                            }
                        }
                    }
                }
                for i in 0..n_features {
                    cov_tied[[i, i]] = cov_tied[[i, i]] / total_nk;
                    for j in 0..i {
                        cov_tied[[i, j]] = cov_tied[[i, j]] / total_nk;
                        cov_tied[[j, i]] = cov_tied[[i, j]];
                    }
                }
                // Copy the single tied covariance into all k blocks.
                for ki in 0..k {
                    let offset = ki * n_features;
                    covariances
                        .slice_mut(ndarray::s![offset..offset + n_features, ..])
                        .assign(&cov_tied);
                }
            }
            CovarianceType::Diag => {
                for ki in 0..k {
                    let nki = nk[ki] + reg_nk;
                    for j in 0..n_features {
                        let s: F = (0..n_samples).fold(F::zero(), |acc, n| {
                            let d = x[[n, j]] - means[[ki, j]];
                            acc + resp[[n, ki]] * d * d
                        });
                        let var = s / nki;
                        // Regularise diagonal.
                        covariances[[ki, j]] = if var < F::from(1e-6).unwrap() {
                            F::from(1e-6).unwrap()
                        } else {
                            var
                        };
                    }
                }
            }
            CovarianceType::Spherical => {
                for ki in 0..k {
                    let nki = nk[ki] + reg_nk;
                    let d_f = F::from(n_features as f64).unwrap();
                    let s: F = (0..n_samples).fold(F::zero(), |acc, n| {
                        let sq: F = (0..n_features).fold(F::zero(), |a, j| {
                            let d = x[[n, j]] - means[[ki, j]];
                            a + d * d
                        });
                        acc + resp[[n, ki]] * sq
                    });
                    let var = s / (nki * d_f);
                    covariances[[ki, 0]] = if var < F::from(1e-6).unwrap() {
                        F::from(1e-6).unwrap()
                    } else {
                        var
                    };
                }
            }
        }
    }

    Ok(FittedGaussianMixture {
        weights_: weights,
        means_: means,
        covariances_: covariances,
        converged_: converged,
        lower_bound_: prev_ll,
        covariance_type_: covariance_type,
        n_features_: n_features,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls: Fit
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GaussianMixture<F> {
    type Fitted = FittedGaussianMixture<F>;
    type Error = FerroError;

    /// Fit the Gaussian Mixture Model using the EM algorithm.
    ///
    /// Runs `n_init` independent EM runs and returns the model with the
    /// highest final log-likelihood.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components == 0` or
    ///   `n_init == 0`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < n_components`.
    /// - [`FerroError::NumericalInstability`] if a covariance matrix
    ///   becomes singular during EM.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGaussianMixture<F>, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_init == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_init".into(),
                reason: "must be at least 1".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: self.n_components,
                actual: 0,
                context: "GaussianMixture requires at least n_components samples".into(),
            });
        }
        if n_samples < self.n_components {
            return Err(FerroError::InsufficientSamples {
                required: self.n_components,
                actual: n_samples,
                context: "GaussianMixture requires at least n_components samples".into(),
            });
        }

        let base_seed = self.random_state.unwrap_or(0);
        let mut best: Option<FittedGaussianMixture<F>> = None;

        for run in 0..self.n_init {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(run as u64));
            let candidate = run_em(
                x,
                self.n_components,
                self.covariance_type,
                self.max_iter,
                self.tol,
                &mut rng,
            )?;
            match &best {
                None => best = Some(candidate),
                Some(b) => {
                    if candidate.lower_bound_ > b.lower_bound_ {
                        best = Some(candidate);
                    }
                }
            }
        }

        best.ok_or_else(|| FerroError::InvalidParameter {
            name: "n_init".into(),
            reason: "internal error: no EM runs completed".into(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls: Predict and Transform
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianMixture<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Assign each sample to its most-probable mixture component.
    ///
    /// Returns an integer label array of shape `(n_samples,)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count differs
    /// from the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let resp = self.transform(x)?;
        let labels: Array1<usize> = resp
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0usize, F::neg_infinity()), |(best_k, best_v), (ki, &v)| {
                        if v > best_v {
                            (ki, v)
                        } else {
                            (best_k, best_v)
                        }
                    })
                    .0
            })
            .collect();
        Ok(labels)
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedGaussianMixture<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Compute the posterior responsibility matrix (shape `(n_samples, n_components)`).
    ///
    /// Each row sums to 1.  This is the "soft" assignment of samples to
    /// components.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count differs
    /// from the fitted model.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let log_resp_raw = self.log_responsibilities(x)?;
        let (log_resp_norm, _) = log_sum_exp_rows(&log_resp_raw);
        Ok(log_resp_norm.mapv(|v| v.exp()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two well-separated Gaussian blobs.
    fn make_two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.1, 0.1, 10.0, 10.0, 10.1,
                10.0, 10.0, 10.1, 9.9, 10.0, 10.0, 9.9, 10.1, 10.1,
            ],
        )
        .unwrap()
    }

    /// Three well-separated blobs (for multi-component tests).
    fn make_three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 10.1, 0.0, 10.0, 0.1,
                10.1, -0.1, 9.9,
            ],
        )
        .unwrap()
    }

    // ── Construction & builder ──────────────────────────────────────────────

    #[test]
    fn test_new_defaults() {
        let gmm = GaussianMixture::<f64>::new(3);
        assert_eq!(gmm.n_components, 3);
        assert_eq!(gmm.covariance_type, CovarianceType::Full);
        assert_eq!(gmm.max_iter, 100);
        assert_eq!(gmm.n_init, 1);
        assert!(gmm.random_state.is_none());
    }

    #[test]
    fn test_builder_methods() {
        let gmm = GaussianMixture::<f64>::new(2)
            .with_covariance_type(CovarianceType::Diag)
            .with_max_iter(50)
            .with_tol(1e-6)
            .with_n_init(3)
            .with_random_state(7);
        assert_eq!(gmm.covariance_type, CovarianceType::Diag);
        assert_eq!(gmm.max_iter, 50);
        assert_eq!(gmm.n_init, 3);
        assert_eq!(gmm.random_state, Some(7));
    }

    // ── Error conditions ────────────────────────────────────────────────────

    #[test]
    fn test_zero_components_error() {
        let x = make_two_blobs();
        let result = GaussianMixture::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_n_init_error() {
        let x = make_two_blobs();
        let result = GaussianMixture::<f64>::new(2).with_n_init(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = GaussianMixture::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_more_components_than_samples_error() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = GaussianMixture::<f64>::new(5).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_feature_mismatch_error() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&bad).is_err());
    }

    #[test]
    fn test_transform_feature_mismatch_error() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.transform(&bad).is_err());
    }

    // ── Fitting behaviour ───────────────────────────────────────────────────

    #[test]
    fn test_fit_two_blobs_full_covariance() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.weights().len(), 2);
        assert_eq!(fitted.means().dim(), (2, 2));
        // Weights must sum to 1.
        let w_sum: f64 = fitted.weights().iter().sum();
        assert_relative_eq!(w_sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fit_diag_covariance() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_covariance_type(CovarianceType::Diag)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.covariances().dim(), (2, 2));
    }

    #[test]
    fn test_fit_spherical_covariance() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_covariance_type(CovarianceType::Spherical)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.covariances().dim(), (2, 1));
    }

    #[test]
    fn test_fit_tied_covariance() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_covariance_type(CovarianceType::Tied)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        // k * n_features rows, n_features columns.
        assert_eq!(fitted.covariances().dim(), (4, 2));
    }

    #[test]
    fn test_single_component() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(1)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        assert_eq!(fitted.weights().len(), 1);
        assert_relative_eq!(fitted.weights()[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reproducibility() {
        let x = make_two_blobs();
        let gmm = GaussianMixture::<f64>::new(2).with_random_state(123);
        let f1 = gmm.fit(&x, &()).unwrap();
        let f2 = gmm.fit(&x, &()).unwrap();
        assert_relative_eq!(f1.lower_bound(), f2.lower_bound(), epsilon = 1e-10);
    }

    #[test]
    fn test_n_init_picks_best() {
        let x = make_two_blobs();
        let f1 = GaussianMixture::<f64>::new(2)
            .with_random_state(0)
            .with_n_init(1)
            .fit(&x, &())
            .unwrap();
        let f5 = GaussianMixture::<f64>::new(2)
            .with_random_state(0)
            .with_n_init(5)
            .fit(&x, &())
            .unwrap();
        // n_init=5 should find a solution at least as good.
        assert!(f5.lower_bound() >= f1.lower_bound() - 1e-6);
    }

    // ── Predict ─────────────────────────────────────────────────────────────

    #[test]
    fn test_predict_shape() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), x.nrows());
    }

    #[test]
    fn test_predict_valid_range() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        for &l in labels.iter() {
            assert!(l < 2, "label {l} out of range");
        }
    }

    #[test]
    fn test_predict_well_separated_clusters() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(300)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        // First 6 points should have the same label; last 6 should have the same label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[6], labels[8]);
        // The two groups should differ.
        assert_ne!(labels[0], labels[6]);
    }

    #[test]
    fn test_predict_three_blobs() {
        let x = make_three_blobs();
        let fitted = GaussianMixture::<f64>::new(3)
            .with_random_state(7)
            .with_max_iter(300)
            .with_n_init(3)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
    }

    // ── Transform (predict_proba) ───────────────────────────────────────────

    #[test]
    fn test_transform_shape() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let resp = fitted.transform(&x).unwrap();
        assert_eq!(resp.dim(), (12, 2));
    }

    #[test]
    fn test_transform_rows_sum_to_one() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let resp = fitted.transform(&x).unwrap();
        for row in resp.rows() {
            let s: f64 = row.iter().sum();
            assert_relative_eq!(s, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_transform_values_in_0_1() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let resp = fitted.transform(&x).unwrap();
        for &v in resp.iter() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-10);
        }
    }

    // ── BIC / AIC ───────────────────────────────────────────────────────────

    #[test]
    fn test_bic_finite() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let bic = fitted.bic(x.nrows());
        assert!(bic.is_finite(), "BIC should be finite");
    }

    #[test]
    fn test_aic_finite() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let aic = fitted.aic(x.nrows());
        assert!(aic.is_finite(), "AIC should be finite");
    }

    #[test]
    fn test_bic_increases_with_more_components_on_two_blobs() {
        // Fitting k=2 to two-blob data should give lower BIC than k=5.
        let x = make_two_blobs();
        let bic2 = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap()
            .bic(x.nrows());
        let bic5 = GaussianMixture::<f64>::new(5)
            .with_random_state(42)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap()
            .bic(x.nrows());
        // This holds when the penalty dominates the likelihood gain.
        assert!(bic2 < bic5, "bic2={bic2} bic5={bic5}");
    }

    // ── f32 support ─────────────────────────────────────────────────────────

    #[test]
    fn test_f32_support() {
        let x = Array2::<f32>::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 9.9,
                10.1,
            ],
        )
        .unwrap();
        let fitted = GaussianMixture::<f32>::new(2)
            .with_random_state(0)
            .with_max_iter(200)
            .fit(&x, &())
            .unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    // ── Accessor methods ────────────────────────────────────────────────────

    #[test]
    fn test_accessor_methods() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(1)
            .fit(&x, &())
            .unwrap();
        // Just ensure they don't panic and return reasonable values.
        assert_eq!(fitted.weights().len(), 2);
        assert_eq!(fitted.means().nrows(), 2);
        assert!(fitted.lower_bound().is_finite());
        // converged is bool — just check it's accessible.
        let _ = fitted.converged();
    }

    #[test]
    fn test_lower_bound_finite() {
        let x = make_two_blobs();
        let fitted = GaussianMixture::<f64>::new(2)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        assert!(fitted.lower_bound().is_finite());
    }
}
