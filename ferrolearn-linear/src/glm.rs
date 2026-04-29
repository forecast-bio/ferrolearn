//! Generalized Linear Models (GLM).
//!
//! This module provides IRLS-based GLM regressors for count and positive
//! continuous data:
//!
//! - **[`GLMRegressor`]** — Generic GLM with selectable [`GLMFamily`]
//! - **[`PoissonRegressor`]** — Convenience wrapper with Poisson family
//! - **[`GammaRegressor`]** — Convenience wrapper with Gamma family
//! - **[`TweedieRegressor`]** — Convenience wrapper with Tweedie family
//!
//! All models use Iteratively Reweighted Least Squares (IRLS) with a log
//! link function and L2 regularization.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::PoissonRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 5.0, 10.0, 20.0];
//!
//! let model = PoissonRegressor::<f64>::new().with_alpha(0.0);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

// ---------------------------------------------------------------------------
// GLMFamily
// ---------------------------------------------------------------------------

/// The distributional family for a Generalized Linear Model.
///
/// Determines the variance function V(mu):
/// - **Poisson**: V(mu) = mu
/// - **Gamma**: V(mu) = mu^2
/// - **Tweedie(p)**: V(mu) = mu^p
#[derive(Debug, Clone, Copy)]
pub enum GLMFamily {
    /// Poisson family — variance proportional to the mean.
    Poisson,
    /// Gamma family — variance proportional to the squared mean.
    Gamma,
    /// Tweedie family with power parameter `p`.
    ///
    /// - `p = 0` gives Normal (constant variance)
    /// - `p = 1` gives Poisson
    /// - `p = 2` gives Gamma
    /// - `1 < p < 2` gives compound Poisson-Gamma
    Tweedie(f64),
}

impl GLMFamily {
    /// Compute the variance function V(mu) for a given mean `mu`.
    fn variance<F: Float + FromPrimitive>(&self, mu: F) -> F {
        match self {
            GLMFamily::Poisson => mu,
            GLMFamily::Gamma => mu * mu,
            GLMFamily::Tweedie(p) => {
                let power = F::from(*p).unwrap();
                mu.powf(power)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GLMRegressor
// ---------------------------------------------------------------------------

/// Generalized Linear Model regressor.
///
/// Fitted via IRLS with a log link function. The [`GLMFamily`] controls
/// the assumed variance-mean relationship.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GLMRegressor<F> {
    /// Distributional family (Poisson, Gamma, or Tweedie).
    pub family: GLMFamily,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum coefficient change.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> GLMRegressor<F> {
    /// Create a new `GLMRegressor` with the given family.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new(family: GLMFamily) -> Self {
        Self {
            family,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

/// Fitted GLM regressor.
///
/// Stores the learned coefficients and intercept on the log-link scale.
/// Predictions are computed as `exp(X @ coef + intercept)`.
#[derive(Debug, Clone)]
pub struct FittedGLMRegressor<F> {
    /// Learned coefficient vector on the log scale.
    coefficients: Array1<F>,
    /// Learned intercept on the log scale.
    intercept: F,
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Poisson regressor — GLM with Poisson family and log link.
///
/// Suitable for modelling count data (y >= 0, integer-valued).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct PoissonRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> PoissonRegressor<F> {
    /// Create a new `PoissonRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for PoissonRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gamma regressor — GLM with Gamma family and log link.
///
/// Suitable for modelling positive continuous data (y > 0).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GammaRegressor<F> {
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> GammaRegressor<F> {
    /// Create a new `GammaRegressor` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `max_iter = 100`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for GammaRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Tweedie regressor — GLM with Tweedie family and log link.
///
/// The `power` parameter controls the variance-mean relationship:
/// V(mu) = mu^power.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct TweedieRegressor<F> {
    /// Tweedie power parameter.
    pub power: f64,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of IRLS iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Whether to fit an intercept.
    pub fit_intercept: bool,
}

impl<F: Float + FromPrimitive> TweedieRegressor<F> {
    /// Create a new `TweedieRegressor` with default settings.
    ///
    /// Defaults: `power = 1.5`, `alpha = 1.0`, `max_iter = 100`,
    /// `tol = 1e-4`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            power: 1.5,
            alpha: F::one(),
            max_iter: 100,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the Tweedie power parameter.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of IRLS iterations.
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

    /// Set whether to fit an intercept.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float + FromPrimitive> Default for TweedieRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cholesky solve for `A x = b`.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "Cholesky: matrix not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s = s - l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s = s - l[[k, i]] * x_sol[k];
        }
        x_sol[i] = s / l[[i, i]];
    }

    Ok(x_sol)
}

/// Gaussian elimination with partial pivoting.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
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
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix in Gaussian elimination".into(),
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

    let mut x_sol = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s = s - aug[[i, j]] * x_sol[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot in back substitution".into(),
            });
        }
        x_sol[i] = s / aug[[i, i]];
    }

    Ok(x_sol)
}

/// Solve the weighted ridge system `(X^T W X + alpha I) w = X^T W z`.
fn weighted_ridge_solve<F: Float + FromPrimitive>(
    x: &Array2<F>,
    z: &Array1<F>,
    weights: &Array1<F>,
    alpha: F,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    let mut xtwx = Array2::<F>::zeros((n_features, n_features));
    let mut xtwz = Array1::<F>::zeros(n_features);

    for i in 0..n_samples {
        let wi = weights[i];
        let xi = x.row(i);
        for r in 0..n_features {
            xtwz[r] = xtwz[r] + wi * xi[r] * z[i];
            for c in 0..n_features {
                xtwx[[r, c]] = xtwx[[r, c]] + wi * xi[r] * xi[c];
            }
        }
    }

    // Add L2 regularization (do not penalise intercept column if present).
    for i in 0..n_features {
        xtwx[[i, i]] = xtwx[[i, i]] + alpha;
    }

    cholesky_solve(&xtwx, &xtwz).or_else(|_| gaussian_solve(n_features, &xtwx, &xtwz))
}

/// Core IRLS fitting logic shared by all GLM variants.
fn fit_glm_irls<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    family: &GLMFamily,
    alpha: F,
    max_iter: usize,
    tol: F,
    fit_intercept: bool,
) -> Result<FittedGLMRegressor<F>, FerroError> {
    let (n_samples, n_features_orig) = x.dim();

    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }

    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "GLM requires at least one sample".into(),
        });
    }

    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }

    // All y values must be positive for log link.
    let min_y = F::from(1e-10).unwrap();
    for &yi in y.iter() {
        if yi < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "target values must be non-negative for GLM with log link".into(),
            });
        }
    }

    // Build design matrix (optionally prepend intercept column).
    let n_cols = if fit_intercept {
        n_features_orig + 1
    } else {
        n_features_orig
    };

    let mut x_design = Array2::<F>::zeros((n_samples, n_cols));
    if fit_intercept {
        for i in 0..n_samples {
            x_design[[i, 0]] = F::one();
            for j in 0..n_features_orig {
                x_design[[i, j + 1]] = x[[i, j]];
            }
        }
    } else {
        x_design.assign(x);
    }

    // Clamp y for log.
    let y_safe: Array1<F> = y.mapv(|v| if v < min_y { min_y } else { v });

    // Initialise eta = log(y), mu = y.
    let mut eta: Array1<F> = y_safe.mapv(|v| v.ln());
    let mut mu: Array1<F> = y_safe.clone();
    let mut coef = Array1::<F>::zeros(n_cols);

    let min_mu = F::from(1e-10).unwrap();
    let max_mu = F::from(1e10).unwrap();

    for _iter in 0..max_iter {
        let coef_old = coef.clone();

        // Compute IRLS weights and working response.
        let mut weights = Array1::<F>::zeros(n_samples);
        let mut z = Array1::<F>::zeros(n_samples);

        for i in 0..n_samples {
            let mu_i = mu[i].max(min_mu).min(max_mu);
            let var_i = family.variance(mu_i).max(min_mu);
            // Log link: g'(mu) = 1/mu, so working response = eta + (y - mu)/mu
            //           weight = mu^2 / V(mu)  (from W = 1/(g'^2 * V))
            let g_prime = F::one() / mu_i; // derivative of log link
            z[i] = eta[i] + (y_safe[i] - mu_i) * g_prime;
            weights[i] = F::one() / (g_prime * g_prime * var_i);
            // Clamp weight.
            if weights[i] < min_mu {
                weights[i] = min_mu;
            }
        }

        // Solve weighted ridge.
        coef = weighted_ridge_solve(&x_design, &z, &weights, alpha)?;

        // Update eta and mu.
        eta = x_design.dot(&coef);
        for i in 0..n_samples {
            // Clamp eta to prevent overflow in exp.
            let eta_i = eta[i].max(F::from(-20.0).unwrap()).min(F::from(20.0).unwrap());
            eta[i] = eta_i;
            mu[i] = eta_i.exp().max(min_mu).min(max_mu);
        }

        // Check convergence.
        let max_change = coef
            .iter()
            .zip(coef_old.iter())
            .map(|(&c, &co)| (c - co).abs())
            .fold(F::zero(), |a, b| if b > a { b } else { a });

        if max_change < tol {
            break;
        }
    }

    // Extract intercept and feature coefficients.
    let (intercept, coefficients) = if fit_intercept {
        let intercept = coef[0];
        let coefficients = Array1::from_iter(coef.iter().skip(1).copied());
        (intercept, coefficients)
    } else {
        (F::zero(), coef)
    };

    Ok(FittedGLMRegressor {
        coefficients,
        intercept,
    })
}

// ---------------------------------------------------------------------------
// Fit — GLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GLMRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the GLM via IRLS.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — zero samples.
    /// - [`FerroError::InvalidParameter`] — negative alpha or negative y.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            &self.family,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — PoissonRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for PoissonRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Poisson GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Poisson,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — GammaRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for GammaRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Gamma GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Gamma,
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Fit — TweedieRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for TweedieRegressor<F>
{
    type Fitted = FittedGLMRegressor<F>;
    type Error = FerroError;

    /// Fit the Tweedie GLM via IRLS.
    ///
    /// # Errors
    ///
    /// See [`GLMRegressor::fit`].
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedGLMRegressor<F>, FerroError> {
        fit_glm_irls(
            x,
            y,
            &GLMFamily::Tweedie(self.power),
            self.alpha,
            self.max_iter,
            self.tol,
            self.fit_intercept,
        )
    }
}

// ---------------------------------------------------------------------------
// Predict / HasCoefficients / Pipeline — FittedGLMRegressor
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedGLMRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict using the fitted GLM.
    ///
    /// Computes `exp(X @ coefficients + intercept)` (inverse log link).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.coefficients.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.coefficients.len()],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }
        let eta = x.dot(&self.coefficients) + self.intercept;
        Ok(eta.mapv(|v| v.exp()))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedGLMRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration for GLMRegressor.
impl<F> PipelineEstimator<F> for GLMRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedGLMRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// Pipeline integration for PoissonRegressor.
impl<F> PipelineEstimator<F> for PoissonRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// Pipeline integration for GammaRegressor.
impl<F> PipelineEstimator<F> for GammaRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// Pipeline integration for TweedieRegressor.
impl<F> PipelineEstimator<F> for TweedieRegressor<F>
where
    F: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // ---- GLMRegressor ----

    #[test]
    fn test_glm_poisson_defaults() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Poisson);
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_glm_builder() {
        let m = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.5)
            .with_max_iter(200)
            .with_tol(1e-6)
            .with_fit_intercept(false);
        assert_relative_eq!(m.alpha, 0.5);
        assert_eq!(m.max_iter, 200);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_glm_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];
        assert!(GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_glm_negative_alpha() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        assert!(GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .with_alpha(-1.0)
            .fit(&x, &y)
            .is_err());
    }

    #[test]
    fn test_glm_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        // Predictions should be positive.
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Gamma)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GLMRegressor::<f64>::new(GLMFamily::Tweedie(1.5))
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_glm_predict_feature_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_glm_has_coefficients() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let fitted = GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_glm_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let model = GLMRegressor::<f64>::new(GLMFamily::Poisson).with_alpha(0.0);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- PoissonRegressor ----

    #[test]
    fn test_poisson_defaults() {
        let m = PoissonRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_poisson_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_poisson_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- GammaRegressor ----

    #[test]
    fn test_gamma_defaults() {
        let m = GammaRegressor::<f64>::new();
        assert_relative_eq!(m.alpha, 1.0);
        assert_eq!(m.max_iter, 100);
    }

    #[test]
    fn test_gamma_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_gamma_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = GammaRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- TweedieRegressor ----

    #[test]
    fn test_tweedie_defaults() {
        let m = TweedieRegressor::<f64>::new();
        assert_relative_eq!(m.power, 1.5);
        assert_relative_eq!(m.alpha, 1.0);
    }

    #[test]
    fn test_tweedie_fit_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];

        let fitted = TweedieRegressor::<f64>::new()
            .with_power(1.5)
            .with_alpha(0.0)
            .with_max_iter(200)
            .fit(&x, &y)
            .unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
        for &p in preds.iter() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_tweedie_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 5.0, 10.0, 20.0];
        let fitted = TweedieRegressor::<f64>::new()
            .with_alpha(0.0)
            .fit_pipeline(&x, &y)
            .unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // ---- Variance function ----

    #[test]
    fn test_variance_poisson() {
        let v = GLMFamily::Poisson.variance(3.0_f64);
        assert_relative_eq!(v, 3.0);
    }

    #[test]
    fn test_variance_gamma() {
        let v = GLMFamily::Gamma.variance(3.0_f64);
        assert_relative_eq!(v, 9.0);
    }

    #[test]
    fn test_variance_tweedie() {
        let v = GLMFamily::Tweedie(1.5).variance(4.0_f64);
        assert_relative_eq!(v, 4.0_f64.powf(1.5), epsilon = 1e-10);
    }

    #[test]
    fn test_glm_negative_y() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, -2.0, 3.0];
        assert!(GLMRegressor::<f64>::new(GLMFamily::Poisson)
            .fit(&x, &y)
            .is_err());
    }
}
