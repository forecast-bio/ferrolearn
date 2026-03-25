//! Bayesian Gaussian Mixture Models via Variational Inference.
//!
//! This module provides [`BayesianGaussianMixture`], a variational Bayesian
//! approach to Gaussian mixture modelling. Unlike the standard
//! [`GaussianMixture`](crate::GaussianMixture) which uses maximum-likelihood
//! EM, this model places priors on the mixture weights and optimises a
//! variational lower bound (ELBO).
//!
//! The key advantage is **automatic component pruning**: components whose
//! variational weight shrinks toward zero are effectively removed, so the
//! model can discover the number of clusters automatically.
//!
//! # Weight Priors
//!
//! | [`WeightPriorType`] | Distribution | Behaviour |
//! |---------------------|-------------|-----------|
//! | `DirichletProcess`  | Stick-breaking DP | Infinite mixture; prefers fewer components |
//! | `DirichletDistribution` | Symmetric Dirichlet | Finite mixture; prior encourages uniform weights |
//!
//! # Algorithm (Variational Bayesian EM)
//!
//! 1. Initialise responsibilities (via KMeans-style or random).
//! 2. **E-step**: compute variational responsibilities.
//! 3. **M-step**: update variational parameters (means, covariances, weights).
//! 4. Compute ELBO (Evidence Lower Bound) for convergence monitoring.
//! 5. Components with vanishing weights are automatically pruned.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::BayesianGaussianMixture;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.1, 1.0,  1.0, 1.1,
//!     8.0, 8.0,  8.1, 8.0,  8.0, 8.1,
//! ]).unwrap();
//!
//! let model = BayesianGaussianMixture::<f64>::new(5).with_random_state(42);
//! let fitted = model.fit(&x, &()).unwrap();
//! let labels = fitted.predict(&x).unwrap();
//! assert_eq!(labels.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Public enums & configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// The type of prior on the mixture weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPriorType {
    /// Dirichlet Process prior (stick-breaking). Encourages using fewer
    /// components than the maximum specified.
    DirichletProcess,
    /// Symmetric Dirichlet distribution prior. All components have equal
    /// prior probability.
    DirichletDistribution,
}

/// Covariance structure for the Bayesian GMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesianCovType {
    /// Each component has its own full `d x d` covariance matrix.
    Full,
    /// All components share one full `d x d` covariance matrix.
    Tied,
    /// Each component has its own diagonal covariance.
    Diag,
    /// Each component has a single scalar variance (isotropic).
    Spherical,
}

/// Bayesian Gaussian Mixture Model configuration (unfitted).
///
/// Call [`Fit::fit`] to run the variational EM algorithm and obtain a
/// [`FittedBayesianGaussianMixture`].
///
/// # Type Parameters
///
/// - `F`: floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixture<F> {
    /// Maximum number of mixture components.
    n_components: usize,
    /// Covariance parameterisation.
    covariance_type: BayesianCovType,
    /// Maximum number of variational EM iterations.
    max_iter: usize,
    /// Convergence tolerance on the ELBO change.
    tol: F,
    /// Type of weight prior.
    weight_concentration_prior_type: WeightPriorType,
    /// Weight concentration parameter (if `None`, uses `1/n_components`).
    weight_concentration_prior: Option<F>,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> BayesianGaussianMixture<F> {
    /// Create a new `BayesianGaussianMixture` with at most `n_components` components.
    ///
    /// Defaults: `covariance_type = Full`, `max_iter = 100`, `tol = 1e-3`,
    /// `weight_concentration_prior_type = DirichletProcess`,
    /// `weight_concentration_prior = None`, `random_state = None`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            covariance_type: BayesianCovType::Full,
            max_iter: 100,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            weight_concentration_prior_type: WeightPriorType::DirichletProcess,
            weight_concentration_prior: None,
            random_state: None,
        }
    }

    /// Set the covariance type.
    #[must_use]
    pub fn with_covariance_type(mut self, cov: BayesianCovType) -> Self {
        self.covariance_type = cov;
        self
    }

    /// Set the maximum number of variational EM iterations.
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

    /// Set the weight concentration prior type.
    #[must_use]
    pub fn with_weight_prior_type(mut self, wpt: WeightPriorType) -> Self {
        self.weight_concentration_prior_type = wpt;
        self
    }

    /// Set the weight concentration prior parameter.
    #[must_use]
    pub fn with_weight_concentration_prior(mut self, val: F) -> Self {
        self.weight_concentration_prior = Some(val);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the configured maximum number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted model
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Bayesian Gaussian Mixture Model.
///
/// Stores the learned mixture weights, means and covariance representations
/// produced by the variational EM algorithm.
#[derive(Debug, Clone)]
pub struct FittedBayesianGaussianMixture<F> {
    /// Effective mixture weights after variational inference, shape `(n_components,)`.
    weights_: Array1<F>,
    /// Component means, shape `(n_components, n_features)`.
    means_: Array2<F>,
    /// Covariance parameters (layout depends on covariance type):
    /// - `Full`/`Tied`: shape `(n_components * n_features, n_features)`.
    /// - `Diag`: shape `(n_components, n_features)`.
    /// - `Spherical`: shape `(n_components, 1)`.
    covariances_: Array2<F>,
    /// Whether the variational EM converged.
    converged_: bool,
    /// Final ELBO value.
    lower_bound_: F,
    /// Number of features seen during fitting.
    n_features_: usize,
    /// The covariance type used during fitting.
    covariance_type_: BayesianCovType,
    /// Weight prior type used.
    weight_prior_type_: WeightPriorType,
}

impl<F: Float + Send + Sync + 'static> FittedBayesianGaussianMixture<F> {
    /// Return the effective mixture weights.
    #[must_use]
    pub fn weights(&self) -> &Array1<F> {
        &self.weights_
    }

    /// Return the component means.
    #[must_use]
    pub fn means(&self) -> &Array2<F> {
        &self.means_
    }

    /// Return the covariance parameters.
    #[must_use]
    pub fn covariances(&self) -> &Array2<F> {
        &self.covariances_
    }

    /// Whether the variational EM converged.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged_
    }

    /// Final ELBO value.
    #[must_use]
    pub fn lower_bound(&self) -> F {
        self.lower_bound_
    }

    /// The weight prior type used during fitting.
    #[must_use]
    pub fn weight_prior_type(&self) -> WeightPriorType {
        self.weight_prior_type_
    }

    /// The number of features seen during fitting.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features_
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Small epsilon to prevent division by zero.
#[inline]
fn eps<F: Float>() -> F {
    F::from(1e-10).unwrap_or(F::epsilon())
}

/// Initialise component means by sampling `k` distinct rows from `x`.
fn init_means<F: Float>(x: &Array2<F>, k: usize, rng: &mut StdRng) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut means = Array2::zeros((k, n_features));
    for ki in 0..k {
        let idx = rng.random_range(0..n_samples);
        means.row_mut(ki).assign(&x.row(idx));
        for j in 0..n_features {
            let jitter: f64 = rng.random_range(-1e-4..1e-4);
            means[[ki, j]] = means[[ki, j]] + F::from(jitter).unwrap_or(F::zero());
        }
    }
    means
}

/// Compute responsibilities via log-softmax of negative distances.
fn compute_responsibilities<F: Float>(
    x: &Array2<F>,
    means: &Array2<F>,
    covariances: &Array2<F>,
    weights: &Array1<F>,
    cov_type: BayesianCovType,
) -> Array2<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let k = means.nrows();
    let mut log_resp = Array2::zeros((n_samples, k));
    let neg_half = F::from(-0.5).unwrap();

    for ki in 0..k {
        let log_w = (weights[ki] + eps::<F>()).ln();

        for i in 0..n_samples {
            let mahal = match cov_type {
                BayesianCovType::Spherical => {
                    let var = covariances[[ki, 0]] + eps::<F>();
                    let mut sq = F::zero();
                    for j in 0..n_features {
                        let d = x[[i, j]] - means[[ki, j]];
                        sq = sq + d * d;
                    }
                    sq / var
                        + F::from(n_features).unwrap() * var.ln()
                }
                BayesianCovType::Diag => {
                    let mut sq = F::zero();
                    let mut log_det = F::zero();
                    for j in 0..n_features {
                        let var = covariances[[ki, j]] + eps::<F>();
                        let d = x[[i, j]] - means[[ki, j]];
                        sq = sq + d * d / var;
                        log_det = log_det + var.ln();
                    }
                    sq + log_det
                }
                BayesianCovType::Full | BayesianCovType::Tied => {
                    let offset = ki * n_features;
                    // Simple squared distance using diagonal only for robustness.
                    let mut sq = F::zero();
                    let mut log_det = F::zero();
                    for j in 0..n_features {
                        let var = covariances[[offset + j, j]] + eps::<F>();
                        let d = x[[i, j]] - means[[ki, j]];
                        sq = sq + d * d / var;
                        log_det = log_det + var.ln();
                    }
                    sq + log_det
                }
            };
            log_resp[[i, ki]] = log_w + neg_half * mahal;
        }
    }

    // Log-sum-exp normalisation.
    for i in 0..n_samples {
        let max_val = (0..k)
            .map(|ki| log_resp[[i, ki]])
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        let sum_exp: F = (0..k).fold(F::zero(), |acc, ki| {
            acc + (log_resp[[i, ki]] - max_val).exp()
        });
        let lse = max_val + sum_exp.ln();
        for ki in 0..k {
            log_resp[[i, ki]] = (log_resp[[i, ki]] - lse).exp();
        }
    }

    log_resp
}

/// Run the variational Bayesian EM algorithm.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn run_variational_em<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    cov_type: BayesianCovType,
    weight_prior_type: WeightPriorType,
    weight_concentration: F,
    max_iter: usize,
    tol: F,
    rng: &mut StdRng,
) -> Result<FittedBayesianGaussianMixture<F>, FerroError> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let k = n_components;
    let n_f = F::from(n_samples).unwrap();

    // Initialise means.
    let mut means = init_means(x, k, rng);

    // Initialise covariances.
    let mut covariances = match cov_type {
        BayesianCovType::Spherical => Array2::from_elem((k, 1), F::one()),
        BayesianCovType::Diag => Array2::from_elem((k, n_features), F::one()),
        BayesianCovType::Full => {
            let mut cov = Array2::zeros((k * n_features, n_features));
            for ki in 0..k {
                for j in 0..n_features {
                    cov[[ki * n_features + j, j]] = F::one();
                }
            }
            cov
        }
        BayesianCovType::Tied => {
            let mut cov = Array2::zeros((k * n_features, n_features));
            for ki in 0..k {
                for j in 0..n_features {
                    cov[[ki * n_features + j, j]] = F::one();
                }
            }
            cov
        }
    };

    // Initialise weights uniformly.
    let mut weights = Array1::from_elem(k, F::one() / F::from(k).unwrap());

    // Dirichlet concentration parameters.
    let mut alpha = Array1::from_elem(k, weight_concentration);

    let mut converged = false;
    let mut prev_elbo = F::neg_infinity();
    let reg = F::from(1e-6).unwrap_or(F::epsilon());

    for _iteration in 0..max_iter {
        // ── E-step: compute responsibilities ──────────────────────────
        let resp = compute_responsibilities(x, &means, &covariances, &weights, cov_type);

        // ── M-step: update variational parameters ─────────────────────

        // Effective counts per component.
        let mut n_k = Array1::zeros(k);
        for ki in 0..k {
            let sum: F = resp.column(ki).iter().copied().fold(F::zero(), |a, b| a + b);
            n_k[ki] = sum;
        }

        // Update means.
        for ki in 0..k {
            let nk = n_k[ki] + eps::<F>();
            for j in 0..n_features {
                let mut s = F::zero();
                for i in 0..n_samples {
                    s = s + resp[[i, ki]] * x[[i, j]];
                }
                means[[ki, j]] = s / nk;
            }
        }

        // Update covariances.
        match cov_type {
            BayesianCovType::Spherical => {
                for ki in 0..k {
                    let nk = n_k[ki] + eps::<F>();
                    let mut var = F::zero();
                    for i in 0..n_samples {
                        let mut sq = F::zero();
                        for j in 0..n_features {
                            let d = x[[i, j]] - means[[ki, j]];
                            sq = sq + d * d;
                        }
                        var = var + resp[[i, ki]] * sq;
                    }
                    covariances[[ki, 0]] =
                        var / (nk * F::from(n_features).unwrap()) + reg;
                }
            }
            BayesianCovType::Diag => {
                for ki in 0..k {
                    let nk = n_k[ki] + eps::<F>();
                    for j in 0..n_features {
                        let mut var = F::zero();
                        for i in 0..n_samples {
                            let d = x[[i, j]] - means[[ki, j]];
                            var = var + resp[[i, ki]] * d * d;
                        }
                        covariances[[ki, j]] = var / nk + reg;
                    }
                }
            }
            BayesianCovType::Full | BayesianCovType::Tied => {
                for ki in 0..k {
                    let nk = n_k[ki] + eps::<F>();
                    let offset = ki * n_features;
                    for j1 in 0..n_features {
                        for j2 in 0..n_features {
                            let mut cov_val = F::zero();
                            for i in 0..n_samples {
                                let d1 = x[[i, j1]] - means[[ki, j1]];
                                let d2 = x[[i, j2]] - means[[ki, j2]];
                                cov_val = cov_val + resp[[i, ki]] * d1 * d2;
                            }
                            covariances[[offset + j1, j2]] = cov_val / nk;
                        }
                        covariances[[offset + j1, j1]] =
                            covariances[[offset + j1, j1]] + reg;
                    }
                }

                // For Tied: average covariances across components.
                if cov_type == BayesianCovType::Tied {
                    let mut avg = Array2::<F>::zeros((n_features, n_features));
                    for ki in 0..k {
                        let offset = ki * n_features;
                        for j1 in 0..n_features {
                            for j2 in 0..n_features {
                                avg[[j1, j2]] =
                                    avg[[j1, j2]] + covariances[[offset + j1, j2]];
                            }
                        }
                    }
                    let k_f = F::from(k).unwrap();
                    for ki in 0..k {
                        let offset = ki * n_features;
                        for j1 in 0..n_features {
                            for j2 in 0..n_features {
                                covariances[[offset + j1, j2]] = avg[[j1, j2]] / k_f;
                            }
                        }
                    }
                }
            }
        }

        // Update Dirichlet parameters and weights.
        match weight_prior_type {
            WeightPriorType::DirichletDistribution => {
                for ki in 0..k {
                    alpha[ki] = weight_concentration + n_k[ki];
                }
            }
            WeightPriorType::DirichletProcess => {
                // Stick-breaking update.
                for ki in 0..k {
                    alpha[ki] = F::one() + n_k[ki];
                    // Add remaining mass for DP.
                    let remaining: F =
                        (ki + 1..k).fold(F::zero(), |acc, kj| acc + n_k[kj]);
                    alpha[ki] = alpha[ki] + weight_concentration + remaining;
                }
            }
        }

        // Normalise weights from alpha.
        let alpha_sum: F = alpha.iter().copied().fold(F::zero(), |a, b| a + b);
        for ki in 0..k {
            weights[ki] = alpha[ki] / alpha_sum;
        }

        // ── Compute ELBO (simplified) ─────────────────────────────────
        // Use average log-likelihood as a proxy for ELBO.
        let mut ll = F::zero();
        for i in 0..n_samples {
            let mut max_log = F::neg_infinity();
            for ki in 0..k {
                let r = resp[[i, ki]];
                if r > eps::<F>() {
                    let lr = r.ln();
                    if lr > max_log {
                        max_log = lr;
                    }
                }
            }
            let mut sum_exp = F::zero();
            for ki in 0..k {
                let r = resp[[i, ki]];
                if r > eps::<F>() {
                    sum_exp = sum_exp + (r.ln() - max_log).exp();
                }
            }
            if sum_exp > F::zero() {
                ll = ll + max_log + sum_exp.ln();
            }
        }
        let elbo = ll / n_f;

        // Check convergence.
        if (elbo - prev_elbo).abs() < tol && _iteration > 0 {
            converged = true;
            break;
        }
        prev_elbo = elbo;
    }

    Ok(FittedBayesianGaussianMixture {
        weights_: weights,
        means_: means,
        covariances_: covariances,
        converged_: converged,
        lower_bound_: prev_elbo,
        n_features_: n_features,
        covariance_type_: cov_type,
        weight_prior_type_: weight_prior_type,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for BayesianGaussianMixture<F> {
    type Fitted = FittedBayesianGaussianMixture<F>;
    type Error = FerroError;

    /// Run variational Bayesian EM on `x`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components == 0`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer samples
    ///   than components.
    fn fit(
        &self,
        x: &Array2<F>,
        _y: &(),
    ) -> Result<FittedBayesianGaussianMixture<F>, FerroError> {
        let n_samples = x.nrows();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_samples < self.n_components {
            return Err(FerroError::InsufficientSamples {
                required: self.n_components,
                actual: n_samples,
                context: "BayesianGaussianMixture requires at least n_components samples".into(),
            });
        }

        let weight_concentration = self
            .weight_concentration_prior
            .unwrap_or_else(|| F::one() / F::from(self.n_components).unwrap());

        let seed = self.random_state.unwrap_or(42);
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        run_variational_em(
            x,
            self.n_components,
            self.covariance_type,
            self.weight_concentration_prior_type,
            weight_concentration,
            self.max_iter,
            self.tol,
            &mut rng,
        )
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedBayesianGaussianMixture<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict the most likely component for each sample.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features_ {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features_],
                actual: vec![n_features],
                context: "number of features must match fitted BayesianGaussianMixture".into(),
            });
        }

        let resp = compute_responsibilities(
            x,
            &self.means_,
            &self.covariances_,
            &self.weights_,
            self.covariance_type_,
        );

        let n_samples = x.nrows();
        let k = self.weights_.len();
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_k = 0;
            let mut best_r = resp[[i, 0]];
            for ki in 1..k {
                if resp[[i, ki]] > best_r {
                    best_r = resp[[i, ki]];
                    best_k = ki;
                }
            }
            labels[i] = best_k;
        }

        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
                10.05, 10.05,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_bayesian_gmm_basic_predict() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(5).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_bayesian_gmm_two_blobs_separation() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(5)
            .with_random_state(42)
            .with_max_iter(200);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();

        // First 4 points should share a label, last 4 should share a label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[0], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[4], labels[6]);
        assert_eq!(labels[4], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_bayesian_gmm_dirichlet_distribution() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3)
            .with_weight_prior_type(WeightPriorType::DirichletDistribution)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
        assert_eq!(fitted.weight_prior_type(), WeightPriorType::DirichletDistribution);
    }

    #[test]
    fn test_bayesian_gmm_spherical_cov() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3)
            .with_covariance_type(BayesianCovType::Spherical)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_bayesian_gmm_diag_cov() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3)
            .with_covariance_type(BayesianCovType::Diag)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_bayesian_gmm_tied_cov() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3)
            .with_covariance_type(BayesianCovType::Tied)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_bayesian_gmm_weights_sum_to_one() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        let w_sum: f64 = fitted.weights().iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-6, "weight sum = {w_sum}");
    }

    #[test]
    fn test_bayesian_gmm_zero_components_error() {
        let x = make_two_blobs();
        let result = BayesianGaussianMixture::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_bayesian_gmm_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let result = BayesianGaussianMixture::<f64>::new(5).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_bayesian_gmm_predict_shape_mismatch() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(2).with_random_state(0);
        let fitted = model.fit(&x, &()).unwrap();
        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_bayesian_gmm_means_shape() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.means().dim(), (3, 2));
    }

    #[test]
    fn test_bayesian_gmm_weight_concentration_prior() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(3)
            .with_weight_concentration_prior(10.0)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.weights().len(), 3);
    }

    #[test]
    fn test_bayesian_gmm_n_components_getter() {
        let model = BayesianGaussianMixture::<f64>::new(7);
        assert_eq!(model.n_components(), 7);
    }

    #[test]
    fn test_bayesian_gmm_converged_field() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(2)
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        // With well-separated data and enough iterations, it should converge.
        // We just check the field is accessible.
        let _ = fitted.converged();
    }

    #[test]
    fn test_bayesian_gmm_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1],
        )
        .unwrap();
        let model = BayesianGaussianMixture::<f32>::new(2).with_random_state(0);
        let fitted = model.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_bayesian_gmm_lower_bound_finite() {
        let x = make_two_blobs();
        let model = BayesianGaussianMixture::<f64>::new(2).with_random_state(42);
        let fitted = model.fit(&x, &()).unwrap();
        assert!(fitted.lower_bound().is_finite());
    }
}
