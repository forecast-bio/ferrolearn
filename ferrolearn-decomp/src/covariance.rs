//! Covariance estimation.
//!
//! This module provides several covariance matrix estimators following
//! the scikit-learn API: empirical, shrunk, Ledoit-Wolf, OAS,
//! Minimum Covariance Determinant (FAST-MCD), and Elliptic Envelope
//! (outlier detection on top of MCD).
//!
//! All estimators follow the ferrolearn `Fit` trait pattern. The unfitted
//! struct holds hyperparameters; calling [`Fit::fit`] returns a fitted
//! struct that stores the learned covariance, location (mean), and
//! precision (inverse covariance).
//!
//! # Algorithms
//!
//! - [`EmpiricalCovariance`] — Maximum-likelihood covariance estimator.
//! - [`ShrunkCovariance`] — Covariance with fixed shrinkage toward a
//!   scaled identity.
//! - [`LedoitWolf`] — Optimal shrinkage minimising Frobenius risk
//!   (Ledoit & Wolf, 2004).
//! - [`OAS`] — Oracle Approximating Shrinkage (Chen, Wiesel, Eldar & Hero,
//!   2010).
//! - [`MinCovDet`] — Minimum Covariance Determinant via the FAST-MCD
//!   algorithm.
//! - [`EllipticEnvelope`] — Outlier detection using Mahalanobis distances
//!   from a robust (MCD) covariance estimate.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::EmpiricalCovariance;
//! use ferrolearn_core::traits::Fit;
//! use ndarray::array;
//!
//! let est = EmpiricalCovariance::<f64>::new();
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let fitted = est.fit(&x, &()).unwrap();
//! assert_eq!(fitted.covariance().dim(), (2, 2));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

// ============================================================================
// Helpers
// ============================================================================

/// Compute the per-column mean of a matrix.
fn col_mean<F: Float>(x: &Array2<F>, n: usize) -> Array1<F> {
    let p = x.ncols();
    let n_f = F::from(n).unwrap();
    let mut mean = Array1::<F>::zeros(p);
    for j in 0..p {
        let s = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
        mean[j] = s / n_f;
    }
    mean
}

/// Compute the empirical covariance matrix `(X-mean)^T (X-mean) / n`.
///
/// If `assume_centered` is true, uses X directly without subtracting the
/// mean. Divides by `n` (MLE estimator), not `n-1`.
fn empirical_cov<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    mean: &Array1<F>,
    assume_centered: bool,
) -> Array2<F> {
    let (n, _p) = x.dim();
    let n_f = F::from(n).unwrap();

    if assume_centered {
        let xt = x.t();
        let mut cov = xt.dot(x);
        cov.mapv_inplace(|v| v / n_f);
        cov
    } else {
        let mut x_c = x.to_owned();
        for mut row in x_c.rows_mut() {
            for (v, &m) in row.iter_mut().zip(mean.iter()) {
                *v = *v - m;
            }
        }
        let xt = x_c.t();
        let mut cov = xt.dot(&x_c);
        cov.mapv_inplace(|v| v / n_f);
        cov
    }
}

/// Compute a lower-triangular Cholesky factor L such that A = L L^T.
///
/// Adds a small regularisation to the diagonal for numerical stability.
fn cholesky<F: Float>(a: &Array2<F>, d: usize) -> Result<Array2<F>, FerroError> {
    let reg = F::from(1e-8).unwrap_or(F::epsilon());
    let mut l = Array2::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[[i, j]];
            if i == j {
                s = s + reg;
            }
            for p in 0..j {
                s = s - l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: format!(
                            "covariance not positive-definite at diagonal [{i},{i}]"
                        ),
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

/// Compute the inverse of a symmetric positive-definite matrix via Cholesky.
///
/// Given `A = L L^T`, computes `A^{-1} = L^{-T} L^{-1}`.
fn spd_inverse<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
) -> Result<Array2<F>, FerroError> {
    let d = a.nrows();
    let l = cholesky(a, d)?;

    // Invert L (lower-triangular).
    let mut l_inv = Array2::<F>::zeros((d, d));
    for i in 0..d {
        l_inv[[i, i]] = F::one() / l[[i, i]];
        for j in (0..i).rev() {
            let mut s = F::zero();
            for k in (j + 1)..=i {
                s = s + l[[i, k]] * l_inv[[k, j]];
            }
            l_inv[[i, j]] = -s / l[[j, j]];
        }
    }

    // A^{-1} = L^{-T} L^{-1}
    let mut inv = Array2::<F>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = F::zero();
            for k in i.max(j)..d {
                s = s + l_inv[[k, i]] * l_inv[[k, j]];
            }
            inv[[i, j]] = s;
            inv[[j, i]] = s;
        }
    }
    Ok(inv)
}

/// Compute Mahalanobis distances for each row of `x` given location and precision.
///
/// `mahal_i = sqrt((x_i - loc)^T precision (x_i - loc))`
fn mahalanobis_distances<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    location: &Array1<F>,
    precision: &Array2<F>,
) -> Array1<F> {
    let n = x.nrows();
    let p = x.ncols();
    let mut dists = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut val = F::zero();
        for j in 0..p {
            let diff_j = x[[i, j]] - location[j];
            for k in 0..p {
                let diff_k = x[[i, k]] - location[k];
                val = val + diff_j * precision[[j, k]] * diff_k;
            }
        }
        dists[i] = val.abs().sqrt();
    }
    dists
}

/// Compute the trace of a square matrix.
fn trace<F: Float>(a: &Array2<F>) -> F {
    let n = a.nrows();
    let mut t = F::zero();
    for i in 0..n {
        t = t + a[[i, i]];
    }
    t
}

/// Compute trace(A^2) for a symmetric matrix (more efficiently via Frobenius norm squared).
fn trace_sq<F: Float>(a: &Array2<F>) -> F {
    let n = a.nrows();
    let mut s = F::zero();
    for i in 0..n {
        for j in 0..n {
            s = s + a[[i, j]] * a[[i, j]];
        }
    }
    s
}

/// Apply shrinkage: `(1-s)*cov + s*(trace(cov)/p)*I`.
fn shrink_covariance<F: Float + Send + Sync + 'static>(
    cov: &Array2<F>,
    shrinkage: F,
    p: usize,
) -> Array2<F> {
    let tr = trace(cov);
    let mu = tr / F::from(p).unwrap();
    let one_minus_s = F::one() - shrinkage;
    let mut result = cov.mapv(|v| one_minus_s * v);
    for i in 0..p {
        result[[i, i]] = result[[i, i]] + shrinkage * mu;
    }
    result
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky.
fn log_det_spd<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> Result<F, FerroError> {
    let d = a.nrows();
    let l = cholesky(a, d)?;
    let mut log_det = F::zero();
    for i in 0..d {
        log_det = log_det + l[[i, i]].ln();
    }
    Ok(log_det + log_det) // 2 * sum(log(diag(L)))
}

/// Chi-squared quantile approximation using the Wilson-Hilferty transformation.
///
/// Computes the approximate `q`-quantile of `chi^2(k)`.
fn chi2_quantile_approx(k: f64, q: f64) -> f64 {
    // Wilson-Hilferty: chi^2_k(q) ~ k * (1 - 2/(9k) + z_q * sqrt(2/(9k)))^3
    // where z_q is the standard normal quantile.
    let z = normal_quantile_approx(q);
    let ratio = 2.0 / (9.0 * k);
    let base = 1.0 - ratio + z * ratio.sqrt();
    let result = k * base * base * base;
    if result < 0.0 { 0.0 } else { result }
}

/// Approximate standard normal quantile (Beasley-Springer-Moro algorithm).
fn normal_quantile_approx(p: f64) -> f64 {
    // Rational approximation for the standard normal quantile.
    // Uses Abramowitz and Stegun 26.2.23 for the central region.
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < f64::EPSILON {
        return 0.0;
    }

    let sign;
    let pp;
    if p < 0.5 {
        pp = p;
        sign = -1.0;
    } else {
        pp = 1.0 - p;
        sign = 1.0;
    }

    let t = (-2.0 * pp.ln()).sqrt();
    // Coefficients from Abramowitz and Stegun.
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    sign * (t - num / den)
}

// ============================================================================
// Fitted covariance (shared return type)
// ============================================================================

/// A fitted covariance model holding the estimated covariance matrix,
/// location (mean), and precision (inverse covariance).
///
/// Created by calling [`Fit::fit`] on any covariance estimator.
#[derive(Debug, Clone)]
pub struct FittedCovariance<F> {
    /// Estimated covariance matrix, shape `(p, p)`.
    covariance_: Array2<F>,
    /// Location vector (mean), shape `(p,)`.
    location_: Array1<F>,
    /// Precision matrix (inverse of covariance), shape `(p, p)`.
    precision_: Array2<F>,
}

impl<F: Float + Send + Sync + 'static> FittedCovariance<F> {
    /// The estimated covariance matrix, shape `(p, p)`.
    #[must_use]
    pub fn covariance(&self) -> &Array2<F> {
        &self.covariance_
    }

    /// The location vector (mean), shape `(p,)`.
    #[must_use]
    pub fn location(&self) -> &Array1<F> {
        &self.location_
    }

    /// The precision matrix (inverse of the covariance), shape `(p, p)`.
    #[must_use]
    pub fn precision(&self) -> &Array2<F> {
        &self.precision_
    }

    /// Compute per-sample Mahalanobis distances.
    ///
    /// Returns an array of length `n` where each entry is
    /// `sqrt((x_i - location)^T precision (x_i - location))`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x` does not match the dimensionality of the fitted model.
    pub fn mahalanobis(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let p = self.location_.len();
        if x.ncols() != p {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), p],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedCovariance::mahalanobis".into(),
            });
        }
        Ok(mahalanobis_distances(x, &self.location_, &self.precision_))
    }
}

// ============================================================================
// 1. EmpiricalCovariance
// ============================================================================

/// Maximum-likelihood covariance estimator.
///
/// Computes the sample covariance matrix `(X - mean)^T (X - mean) / n`.
/// If `assume_centered` is true, the mean is assumed to be zero and not
/// subtracted.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::EmpiricalCovariance;
/// use ferrolearn_core::traits::Fit;
/// use ndarray::array;
///
/// let est = EmpiricalCovariance::<f64>::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = est.fit(&x, &()).unwrap();
/// assert_eq!(fitted.covariance().dim(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct EmpiricalCovariance<F> {
    /// Whether to assume the data is already centred (mean = 0).
    assume_centered: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> EmpiricalCovariance<F> {
    /// Create a new `EmpiricalCovariance` estimator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assume_centered: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to assume the data is already centred.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for EmpiricalCovariance<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for EmpiricalCovariance<F> {
    type Fitted = FittedCovariance<F>;
    type Error = FerroError;

    /// Fit the empirical covariance estimator.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 1 sample is provided.
    /// - [`FerroError::NumericalInstability`] if the covariance cannot be inverted.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedCovariance<F>, FerroError> {
        let (n, _p) = x.dim();
        if n < 1 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n,
                context: "EmpiricalCovariance::fit".into(),
            });
        }

        let location = if self.assume_centered {
            Array1::<F>::zeros(x.ncols())
        } else {
            col_mean(x, n)
        };

        let cov = empirical_cov(x, &location, self.assume_centered);
        let precision = spd_inverse(&cov)?;

        Ok(FittedCovariance {
            covariance_: cov,
            location_: location,
            precision_: precision,
        })
    }
}

// ============================================================================
// 2. ShrunkCovariance
// ============================================================================

/// Covariance estimator with fixed shrinkage toward a scaled identity.
///
/// The shrunk covariance is computed as:
///
/// ```text
/// cov_shrunk = (1 - s) * cov_emp + s * (trace(cov_emp) / p) * I
/// ```
///
/// where `s` is the shrinkage coefficient in `[0, 1]` and `p` is the
/// number of features.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::ShrunkCovariance;
/// use ferrolearn_core::traits::Fit;
/// use ndarray::array;
///
/// let est = ShrunkCovariance::<f64>::new(0.1);
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = est.fit(&x, &()).unwrap();
/// assert_eq!(fitted.covariance().dim(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct ShrunkCovariance<F> {
    /// The shrinkage coefficient in `[0, 1]`.
    shrinkage: F,
    /// Whether to assume the data is already centred.
    assume_centered: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> ShrunkCovariance<F> {
    /// Create a new `ShrunkCovariance` estimator with the given shrinkage.
    ///
    /// The shrinkage value must be in `[0, 1]`.
    #[must_use]
    pub fn new(shrinkage: F) -> Self {
        Self {
            shrinkage,
            assume_centered: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to assume the data is already centred.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for ShrunkCovariance<F> {
    type Fitted = FittedCovariance<F>;
    type Error = FerroError;

    /// Fit the shrunk covariance estimator.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 1 sample.
    /// - [`FerroError::InvalidParameter`] if shrinkage is outside `[0, 1]`.
    /// - [`FerroError::NumericalInstability`] if the covariance cannot be inverted.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedCovariance<F>, FerroError> {
        let (n, p) = x.dim();
        if n < 1 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n,
                context: "ShrunkCovariance::fit".into(),
            });
        }
        if self.shrinkage < F::zero() || self.shrinkage > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "shrinkage".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        let location = if self.assume_centered {
            Array1::<F>::zeros(p)
        } else {
            col_mean(x, n)
        };

        let cov_emp = empirical_cov(x, &location, self.assume_centered);
        let cov = shrink_covariance(&cov_emp, self.shrinkage, p);
        let precision = spd_inverse(&cov)?;

        Ok(FittedCovariance {
            covariance_: cov,
            location_: location,
            precision_: precision,
        })
    }
}

// ============================================================================
// 3. LedoitWolf
// ============================================================================

/// Ledoit-Wolf optimal shrinkage covariance estimator.
///
/// Automatically selects the shrinkage coefficient that minimises the
/// Frobenius risk (Ledoit & Wolf, 2004). The formula estimates the
/// optimal blend between the empirical covariance and a scaled identity
/// target.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::LedoitWolf;
/// use ferrolearn_core::traits::Fit;
/// use ndarray::array;
///
/// let est = LedoitWolf::<f64>::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let fitted = est.fit(&x, &()).unwrap();
/// assert_eq!(fitted.covariance().dim(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct LedoitWolf<F> {
    /// Whether to assume the data is already centred.
    assume_centered: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> LedoitWolf<F> {
    /// Create a new `LedoitWolf` estimator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assume_centered: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to assume the data is already centred.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for LedoitWolf<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Ledoit-Wolf model, extending the base covariance with the
/// computed shrinkage coefficient.
#[derive(Debug, Clone)]
pub struct FittedLedoitWolf<F> {
    /// The base fitted covariance.
    inner: FittedCovariance<F>,
    /// The optimal shrinkage coefficient that was computed.
    shrinkage_: F,
}

impl<F: Float + Send + Sync + 'static> FittedLedoitWolf<F> {
    /// The estimated covariance matrix, shape `(p, p)`.
    #[must_use]
    pub fn covariance(&self) -> &Array2<F> {
        &self.inner.covariance_
    }

    /// The location vector (mean), shape `(p,)`.
    #[must_use]
    pub fn location(&self) -> &Array1<F> {
        &self.inner.location_
    }

    /// The precision matrix (inverse of the covariance), shape `(p, p)`.
    #[must_use]
    pub fn precision(&self) -> &Array2<F> {
        &self.inner.precision_
    }

    /// The optimal shrinkage coefficient that was computed.
    #[must_use]
    pub fn shrinkage(&self) -> F {
        self.shrinkage_
    }

    /// Compute per-sample Mahalanobis distances.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x` does not match the dimensionality of the fitted model.
    pub fn mahalanobis(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.inner.mahalanobis(x)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for LedoitWolf<F> {
    type Fitted = FittedLedoitWolf<F>;
    type Error = FerroError;

    /// Fit the Ledoit-Wolf covariance estimator.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] if the covariance cannot be inverted.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedLedoitWolf<F>, FerroError> {
        let (n, p) = x.dim();
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "LedoitWolf::fit".into(),
            });
        }

        let location = if self.assume_centered {
            Array1::<F>::zeros(p)
        } else {
            col_mean(x, n)
        };

        // Centre the data.
        let mut x_c = x.to_owned();
        if !self.assume_centered {
            for mut row in x_c.rows_mut() {
                for (v, &m) in row.iter_mut().zip(location.iter()) {
                    *v = *v - m;
                }
            }
        }

        let n_f = F::from(n).unwrap();
        let p_f = F::from(p).unwrap();

        // S = X^T X / n  (empirical covariance, MLE)
        let xt = x_c.t();
        let mut s = xt.dot(&x_c);
        s.mapv_inplace(|v| v / n_f);

        // mu = trace(S) / p
        let mu = trace(&s) / p_f;

        // delta = sum((s_ij - mu*delta_ij)^2) / p^2
        // where delta_ij is Kronecker delta
        let mut delta_sum = F::zero();
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { mu } else { F::zero() };
                let diff = s[[i, j]] - target;
                delta_sum = delta_sum + diff * diff;
            }
        }
        let delta = delta_sum / (p_f * p_f);

        // beta: estimate from sample fourth moments
        // beta = (1/n^2) sum_k sum_{i,j} ((x_k^T e_i)(x_k^T e_j) - s_ij)^2 / p^2
        // i.e., average over samples of || x_k x_k^T - S ||_F^2 / (n*p^2)
        let mut beta_sum = F::zero();
        for k in 0..n {
            // Compute || x_k x_k^T / 1 - S ||_F^2
            // = sum_{i,j} (x_ki * x_kj - s_ij)^2
            let row_k = x_c.row(k);
            let mut row_err = F::zero();
            for i in 0..p {
                for j in 0..p {
                    let outer = row_k[i] * row_k[j];
                    let diff = outer - s[[i, j]];
                    row_err = row_err + diff * diff;
                }
            }
            beta_sum = beta_sum + row_err;
        }
        let beta = beta_sum / (n_f * n_f * p_f * p_f);

        // Optimal shrinkage
        let shrinkage = if delta > F::zero() {
            let ratio = beta / delta;
            if ratio < F::zero() {
                F::zero()
            } else if ratio > F::one() {
                F::one()
            } else {
                ratio
            }
        } else {
            F::one()
        };

        let cov = shrink_covariance(&s, shrinkage, p);
        let precision = spd_inverse(&cov)?;

        Ok(FittedLedoitWolf {
            inner: FittedCovariance {
                covariance_: cov,
                location_: location,
                precision_: precision,
            },
            shrinkage_: shrinkage,
        })
    }
}

// ============================================================================
// 4. OAS (Oracle Approximating Shrinkage)
// ============================================================================

/// Oracle Approximating Shrinkage covariance estimator.
///
/// Computes the optimal shrinkage coefficient using the OAS formula
/// (Chen, Wiesel, Eldar & Hero, 2010), which provides a good
/// approximation to the oracle shrinkage under Gaussian assumptions.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::OAS;
/// use ferrolearn_core::traits::Fit;
/// use ndarray::array;
///
/// let est = OAS::<f64>::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let fitted = est.fit(&x, &()).unwrap();
/// assert_eq!(fitted.covariance().dim(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct OAS<F> {
    /// Whether to assume the data is already centred.
    assume_centered: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> OAS<F> {
    /// Create a new `OAS` estimator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assume_centered: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to assume the data is already centred.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for OAS<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted OAS model, extending the base covariance with the computed
/// shrinkage coefficient.
#[derive(Debug, Clone)]
pub struct FittedOAS<F> {
    /// The base fitted covariance.
    inner: FittedCovariance<F>,
    /// The optimal shrinkage coefficient that was computed.
    shrinkage_: F,
}

impl<F: Float + Send + Sync + 'static> FittedOAS<F> {
    /// The estimated covariance matrix, shape `(p, p)`.
    #[must_use]
    pub fn covariance(&self) -> &Array2<F> {
        &self.inner.covariance_
    }

    /// The location vector (mean), shape `(p,)`.
    #[must_use]
    pub fn location(&self) -> &Array1<F> {
        &self.inner.location_
    }

    /// The precision matrix (inverse of the covariance), shape `(p, p)`.
    #[must_use]
    pub fn precision(&self) -> &Array2<F> {
        &self.inner.precision_
    }

    /// The optimal shrinkage coefficient that was computed.
    #[must_use]
    pub fn shrinkage(&self) -> F {
        self.shrinkage_
    }

    /// Compute per-sample Mahalanobis distances.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x` does not match the dimensionality of the fitted model.
    pub fn mahalanobis(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.inner.mahalanobis(x)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for OAS<F> {
    type Fitted = FittedOAS<F>;
    type Error = FerroError;

    /// Fit the OAS covariance estimator.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples.
    /// - [`FerroError::NumericalInstability`] if the covariance cannot be inverted.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOAS<F>, FerroError> {
        let (n, p) = x.dim();
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "OAS::fit".into(),
            });
        }

        let location = if self.assume_centered {
            Array1::<F>::zeros(p)
        } else {
            col_mean(x, n)
        };

        let s = empirical_cov(x, &location, self.assume_centered);

        let n_f = F::from(n).unwrap();
        let p_f = F::from(p).unwrap();
        let two = F::from(2.0).unwrap();

        let tr_s = trace(&s);
        let tr_s2 = trace_sq(&s);

        // OAS formula:
        // rho_num = (1 - 2/p) * trace(S^2) + trace(S)^2
        // rho_den = (n + 1 - 2/p) * (trace(S^2) - trace(S)^2 / p)
        let rho_num = (F::one() - two / p_f) * tr_s2 + tr_s * tr_s;
        let rho_den = (n_f + F::one() - two / p_f) * (tr_s2 - tr_s * tr_s / p_f);

        let shrinkage = if rho_den.abs() < F::from(1e-15).unwrap_or(F::epsilon()) {
            F::one()
        } else {
            let ratio = rho_num / rho_den;
            if ratio < F::zero() {
                F::zero()
            } else if ratio > F::one() {
                F::one()
            } else {
                ratio
            }
        };

        let cov = shrink_covariance(&s, shrinkage, p);
        let precision = spd_inverse(&cov)?;

        Ok(FittedOAS {
            inner: FittedCovariance {
                covariance_: cov,
                location_: location,
                precision_: precision,
            },
            shrinkage_: shrinkage,
        })
    }
}

// ============================================================================
// 5. MinCovDet
// ============================================================================

/// Minimum Covariance Determinant (FAST-MCD) estimator.
///
/// Robust covariance estimation that is resilient to outliers. The
/// algorithm draws random subsets of `h` samples, applies C-steps
/// (select the `h` samples with smallest Mahalanobis distances,
/// recompute covariance), and keeps the subset with the lowest
/// determinant.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::MinCovDet;
/// use ferrolearn_core::traits::Fit;
/// use ndarray::array;
///
/// let est = MinCovDet::<f64>::new();
/// let x = array![
///     [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
///     [2.0, 3.0], [4.0, 5.0], [6.0, 7.0],
/// ];
/// let fitted = est.fit(&x, &()).unwrap();
/// assert_eq!(fitted.covariance().dim(), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct MinCovDet<F> {
    /// Fraction of samples to use for the support set. If `None`,
    /// defaults to `(n + p + 1) / 2 / n`.
    support_fraction: Option<f64>,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> MinCovDet<F> {
    /// Create a new `MinCovDet` estimator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            support_fraction: None,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the support fraction (must be in `(0.5, 1]`).
    #[must_use]
    pub fn support_fraction(mut self, frac: f64) -> Self {
        self.support_fraction = Some(frac);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for MinCovDet<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted MinCovDet model.
#[derive(Debug, Clone)]
pub struct FittedMinCovDet<F> {
    /// The base fitted covariance.
    inner: FittedCovariance<F>,
    /// Boolean mask: `true` for support samples used in the final estimate.
    support_: Vec<bool>,
}

impl<F: Float + Send + Sync + 'static> FittedMinCovDet<F> {
    /// The estimated (robust) covariance matrix, shape `(p, p)`.
    #[must_use]
    pub fn covariance(&self) -> &Array2<F> {
        &self.inner.covariance_
    }

    /// The location vector (robust mean), shape `(p,)`.
    #[must_use]
    pub fn location(&self) -> &Array1<F> {
        &self.inner.location_
    }

    /// The precision matrix (inverse of the covariance), shape `(p, p)`.
    #[must_use]
    pub fn precision(&self) -> &Array2<F> {
        &self.inner.precision_
    }

    /// Boolean support mask. `true` for samples that were in the final
    /// support set (inliers).
    #[must_use]
    pub fn support(&self) -> &[bool] {
        &self.support_
    }

    /// Compute per-sample Mahalanobis distances.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns in
    /// `x` does not match the dimensionality of the fitted model.
    pub fn mahalanobis(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.inner.mahalanobis(x)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MinCovDet<F> {
    type Fitted = FittedMinCovDet<F>;
    type Error = FerroError;

    /// Fit the Minimum Covariance Determinant estimator.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than `p + 1` samples.
    /// - [`FerroError::InvalidParameter`] if `support_fraction` is invalid.
    /// - [`FerroError::NumericalInstability`] if the covariance cannot be inverted.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMinCovDet<F>, FerroError> {
        let (n, p) = x.dim();

        if n < p + 1 {
            return Err(FerroError::InsufficientSamples {
                required: p + 1,
                actual: n,
                context: "MinCovDet::fit requires n >= p+1".into(),
            });
        }

        // Compute h (support size).
        let h = if let Some(frac) = self.support_fraction {
            if frac <= 0.5 || frac > 1.0 {
                return Err(FerroError::InvalidParameter {
                    name: "support_fraction".into(),
                    reason: "must be in (0.5, 1.0]".into(),
                });
            }
            let h_raw = (frac * n as f64).ceil() as usize;
            h_raw.max(p + 1).min(n)
        } else {
            // Default: (n + p + 1) / 2
            (n + p + 1).div_ceil(2).max(p + 1).min(n)
        };

        let mut rng = if let Some(seed) = self.random_state {
            Xoshiro256PlusPlus::seed_from_u64(seed)
        } else {
            Xoshiro256PlusPlus::seed_from_u64(0)
        };

        let n_trials = 10usize.min(n);
        let c_steps = 30;

        let mut best_log_det = F::from(f64::INFINITY).unwrap();
        let mut best_location = col_mean(x, n);
        let mut best_cov = empirical_cov(x, &best_location, false);
        let mut best_support = vec![true; n];

        let index_dist = Uniform::new(0usize, n).unwrap();

        for _trial in 0..n_trials {
            // Draw a random initial subset of size p+1.
            let mut subset: Vec<usize> = Vec::with_capacity(p + 1);
            while subset.len() < p + 1 {
                let idx = index_dist.sample(&mut rng);
                if !subset.contains(&idx) {
                    subset.push(idx);
                }
            }

            // C-steps.
            for _step in 0..c_steps {
                // Compute mean and covariance of subset.
                let sub_n = subset.len();
                let sub_n_f = F::from(sub_n).unwrap();
                let mut loc = Array1::<F>::zeros(p);
                for &idx in &subset {
                    for j in 0..p {
                        loc[j] = loc[j] + x[[idx, j]];
                    }
                }
                loc.mapv_inplace(|v| v / sub_n_f);

                let mut cov = Array2::<F>::zeros((p, p));
                for &idx in &subset {
                    for i in 0..p {
                        let di = x[[idx, i]] - loc[i];
                        for j in 0..p {
                            let dj = x[[idx, j]] - loc[j];
                            cov[[i, j]] = cov[[i, j]] + di * dj;
                        }
                    }
                }
                cov.mapv_inplace(|v| v / sub_n_f);

                // Add small regularisation for numerical stability.
                let reg = F::from(1e-10).unwrap_or(F::epsilon());
                for i in 0..p {
                    cov[[i, i]] = cov[[i, i]] + reg;
                }

                // Compute precision for Mahalanobis distances.
                let prec = match spd_inverse(&cov) {
                    Ok(pr) => pr,
                    Err(_) => break,
                };

                // Compute distances for all samples.
                let dists = mahalanobis_distances(x, &loc, &prec);

                // Sort by distance and select the h closest.
                let mut indices_dists: Vec<(usize, F)> =
                    (0..n).map(|i| (i, dists[i])).collect();
                indices_dists.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let new_subset: Vec<usize> =
                    indices_dists.iter().take(h).map(|&(idx, _)| idx).collect();

                if new_subset == subset {
                    break; // converged
                }
                subset = new_subset;
            }

            // Evaluate final subset: compute covariance and log-determinant.
            let sub_n = subset.len();
            let sub_n_f = F::from(sub_n).unwrap();
            let mut loc = Array1::<F>::zeros(p);
            for &idx in &subset {
                for j in 0..p {
                    loc[j] = loc[j] + x[[idx, j]];
                }
            }
            loc.mapv_inplace(|v| v / sub_n_f);

            let mut cov = Array2::<F>::zeros((p, p));
            for &idx in &subset {
                for i in 0..p {
                    let di = x[[idx, i]] - loc[i];
                    for j in 0..p {
                        let dj = x[[idx, j]] - loc[j];
                        cov[[i, j]] = cov[[i, j]] + di * dj;
                    }
                }
            }
            cov.mapv_inplace(|v| v / sub_n_f);

            let reg = F::from(1e-10).unwrap_or(F::epsilon());
            for i in 0..p {
                cov[[i, i]] = cov[[i, i]] + reg;
            }

            if let Ok(ld) = log_det_spd(&cov) {
                if ld < best_log_det {
                    best_log_det = ld;
                    best_location = loc;
                    best_cov = cov;
                    best_support = vec![false; n];
                    for &idx in &subset {
                        best_support[idx] = true;
                    }
                }
            }
        }

        let precision = spd_inverse(&best_cov)?;

        Ok(FittedMinCovDet {
            inner: FittedCovariance {
                covariance_: best_cov,
                location_: best_location,
                precision_: precision,
            },
            support_: best_support,
        })
    }
}

// ============================================================================
// 6. EllipticEnvelope
// ============================================================================

/// Outlier detection using an elliptic envelope fitted via Minimum
/// Covariance Determinant.
///
/// Fits a robust covariance using [`MinCovDet`] and classifies samples
/// as inliers (+1) or outliers (-1) based on whether their Mahalanobis
/// distance exceeds a chi-squared threshold at the `(1 - contamination)`
/// quantile.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::EllipticEnvelope;
/// use ferrolearn_core::traits::{Fit, Predict};
/// use ndarray::array;
///
/// let est = EllipticEnvelope::<f64>::new();
/// let x = array![
///     [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
///     [2.0, 3.0], [4.0, 5.0], [6.0, 7.0],
/// ];
/// let fitted = est.fit(&x, &()).unwrap();
/// let labels = fitted.predict(&x).unwrap();
/// assert_eq!(labels.len(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct EllipticEnvelope<F> {
    /// Fraction of samples to use for the MCD support set.
    support_fraction: Option<f64>,
    /// The proportion of outliers expected in the data. Controls the
    /// chi-squared threshold (default 0.1).
    contamination: f64,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> EllipticEnvelope<F> {
    /// Create a new `EllipticEnvelope` with default contamination of 0.1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            support_fraction: None,
            contamination: 0.1,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the contamination fraction (must be in `(0, 0.5)`).
    #[must_use]
    pub fn contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }

    /// Set the support fraction for the underlying MCD estimator.
    #[must_use]
    pub fn support_fraction(mut self, frac: f64) -> Self {
        self.support_fraction = Some(frac);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for EllipticEnvelope<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Elliptic Envelope model.
#[derive(Debug, Clone)]
pub struct FittedEllipticEnvelope<F> {
    /// The underlying fitted MCD model.
    mcd: FittedMinCovDet<F>,
    /// The squared Mahalanobis distance threshold: samples with
    /// `dist^2 > threshold` are outliers.
    threshold_: F,
}

impl<F: Float + Send + Sync + 'static> FittedEllipticEnvelope<F> {
    /// The estimated (robust) covariance matrix.
    #[must_use]
    pub fn covariance(&self) -> &Array2<F> {
        self.mcd.covariance()
    }

    /// The location vector (robust mean).
    #[must_use]
    pub fn location(&self) -> &Array1<F> {
        self.mcd.location()
    }

    /// The precision matrix (inverse of the covariance).
    #[must_use]
    pub fn precision(&self) -> &Array2<F> {
        self.mcd.precision()
    }

    /// The squared Mahalanobis distance threshold for outlier detection.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold_
    }

    /// Compute per-sample Mahalanobis distances.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` has wrong number of columns.
    pub fn mahalanobis(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.mcd.mahalanobis(x)
    }

    /// Compute the decision function: negative Mahalanobis distance
    /// shifted by the threshold.
    ///
    /// Positive values indicate inliers, negative values indicate outliers.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` has wrong number of columns.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let dists = self.mcd.mahalanobis(x)?;
        // decision = threshold - dist^2 (positive = inlier)
        Ok(dists.mapv(|d| self.threshold_ - d * d))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for EllipticEnvelope<F> {
    type Fitted = FittedEllipticEnvelope<F>;
    type Error = FerroError;

    /// Fit the Elliptic Envelope.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `contamination` is outside `(0, 0.5)`.
    /// - Propagates errors from [`MinCovDet::fit`].
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedEllipticEnvelope<F>, FerroError> {
        if self.contamination <= 0.0 || self.contamination >= 0.5 {
            return Err(FerroError::InvalidParameter {
                name: "contamination".into(),
                reason: "must be in (0, 0.5)".into(),
            });
        }

        let p = x.ncols();

        let mut mcd = MinCovDet::<F>::new();
        if let Some(frac) = self.support_fraction {
            mcd = mcd.support_fraction(frac);
        }
        if let Some(seed) = self.random_state {
            mcd = mcd.random_state(seed);
        }
        let fitted_mcd = mcd.fit(x, &())?;

        // Threshold: chi2_quantile(p, 1 - contamination)
        // The Mahalanobis distance squared follows chi2(p) under Gaussian.
        let quantile = 1.0 - self.contamination;
        let threshold = chi2_quantile_approx(p as f64, quantile);
        let threshold_f = F::from(threshold).unwrap();

        Ok(FittedEllipticEnvelope {
            mcd: fitted_mcd,
            threshold_: threshold_f,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedEllipticEnvelope<F> {
    type Output = Array1<i32>;
    type Error = FerroError;

    /// Predict outlier labels: +1 for inliers, -1 for outliers.
    ///
    /// A sample is an outlier if its squared Mahalanobis distance exceeds
    /// the chi-squared threshold.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` has wrong number of columns.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<i32>, FerroError> {
        let dists = self.mcd.mahalanobis(x)?;
        Ok(dists.mapv(|d| {
            if d * d > self.threshold_ {
                -1
            } else {
                1
            }
        }))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---- EmpiricalCovariance ----

    #[test]
    fn test_empirical_covariance_basic() {
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert_eq!(fitted.location().len(), 2);
        assert_eq!(fitted.precision().dim(), (2, 2));
    }

    #[test]
    fn test_empirical_covariance_values() {
        // 2 features, 4 samples: known result.
        // X = [[0, 0], [2, 0], [0, 2], [2, 2]]
        // mean = [1, 1]
        // X - mean = [[-1,-1], [1,-1], [-1,1], [1,1]]
        // cov = (X-mu)^T(X-mu)/n = [[1,0],[0,1]]
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let fitted = est.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.covariance()[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.covariance()[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.covariance()[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.location()[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.location()[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empirical_covariance_assume_centered() {
        let est = EmpiricalCovariance::<f64>::new().assume_centered(true);
        let x = array![[1.0, 0.0], [-1.0, 0.0]];
        let fitted = est.fit(&x, &()).unwrap();
        // Mean should be zero.
        assert_abs_diff_eq!(fitted.location()[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.location()[1], 0.0, epsilon = 1e-10);
        // Cov = X^T X / n = [[1, 0], [0, 0]]
        assert_abs_diff_eq!(fitted.covariance()[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empirical_covariance_mahalanobis() {
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let fitted = est.fit(&x, &()).unwrap();
        let dists = fitted.mahalanobis(&x).unwrap();
        assert_eq!(dists.len(), 4);
        // All points are equidistant from the mean in this symmetric case.
        for i in 0..4 {
            assert!(dists[i] > 0.0, "distance should be positive");
        }
        // Symmetry: all distances should be equal.
        assert_abs_diff_eq!(dists[0], dists[1], epsilon = 1e-8);
        assert_abs_diff_eq!(dists[0], dists[2], epsilon = 1e-8);
        assert_abs_diff_eq!(dists[0], dists[3], epsilon = 1e-8);
    }

    #[test]
    fn test_empirical_covariance_shape_mismatch() {
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = est.fit(&x, &()).unwrap();
        let bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.mahalanobis(&bad).is_err());
    }

    #[test]
    fn test_empirical_covariance_precision_is_inverse() {
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let fitted = est.fit(&x, &()).unwrap();
        // cov * precision ~ I
        let prod = fitted.covariance().dot(fitted.precision());
        let p = fitted.covariance().nrows();
        for i in 0..p {
            for j in 0..p {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(prod[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_empirical_covariance_symmetric() {
        let est = EmpiricalCovariance::<f64>::new();
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 3.0, 5.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_empirical_covariance_default() {
        let est = EmpiricalCovariance::<f64>::default();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    // ---- ShrunkCovariance ----

    #[test]
    fn test_shrunk_covariance_zero_shrinkage() {
        // Zero shrinkage should equal empirical covariance.
        let emp = EmpiricalCovariance::<f64>::new();
        let shrunk = ShrunkCovariance::<f64>::new(0.0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let f_emp = emp.fit(&x, &()).unwrap();
        let f_shrunk = shrunk.fit(&x, &()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(
                    f_emp.covariance()[[i, j]],
                    f_shrunk.covariance()[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_shrunk_covariance_full_shrinkage() {
        // Full shrinkage should give a scaled identity.
        let shrunk = ShrunkCovariance::<f64>::new(1.0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = shrunk.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        // Off-diagonal should be zero.
        assert_abs_diff_eq!(cov[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cov[[1, 0]], 0.0, epsilon = 1e-10);
        // Diagonal should be equal (scaled identity).
        assert_abs_diff_eq!(cov[[0, 0]], cov[[1, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_shrunk_covariance_invalid_shrinkage() {
        let shrunk = ShrunkCovariance::<f64>::new(-0.1);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(shrunk.fit(&x, &()).is_err());

        let shrunk2 = ShrunkCovariance::<f64>::new(1.5);
        assert!(shrunk2.fit(&x, &()).is_err());
    }

    #[test]
    fn test_shrunk_covariance_intermediate() {
        let shrunk = ShrunkCovariance::<f64>::new(0.5);
        let x = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let fitted = shrunk.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        // (1-0.5)*1.0 + 0.5*1.0 = 1.0 for diagonal
        assert_abs_diff_eq!(cov[[0, 0]], 1.0, epsilon = 1e-10);
        // Off-diagonal: (1-0.5)*0 = 0
        assert_abs_diff_eq!(cov[[0, 1]], 0.0, epsilon = 1e-10);
    }

    // ---- LedoitWolf ----

    #[test]
    fn test_ledoit_wolf_basic() {
        let est = LedoitWolf::<f64>::new();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert!(fitted.shrinkage() >= 0.0);
        assert!(fitted.shrinkage() <= 1.0);
    }

    #[test]
    fn test_ledoit_wolf_shrinkage_in_range() {
        let est = LedoitWolf::<f64>::new();
        let x = array![
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3],
            [0.5, 0.3, 1.0],
            [1.2, 0.8, 0.4],
            [0.3, 1.1, 0.9],
            [0.7, 0.2, 1.3],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert!(
            fitted.shrinkage() >= 0.0 && fitted.shrinkage() <= 1.0,
            "shrinkage = {} out of range",
            fitted.shrinkage()
        );
    }

    #[test]
    fn test_ledoit_wolf_symmetric() {
        let est = LedoitWolf::<f64>::new();
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 3.0, 5.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_ledoit_wolf_mahalanobis() {
        let est = LedoitWolf::<f64>::new();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let dists = fitted.mahalanobis(&x).unwrap();
        assert_eq!(dists.len(), 4);
        for &d in dists.iter() {
            assert!(d >= 0.0, "Mahalanobis distance should be non-negative");
        }
    }

    #[test]
    fn test_ledoit_wolf_insufficient_samples() {
        let est = LedoitWolf::<f64>::new();
        let x = array![[1.0, 2.0]];
        assert!(est.fit(&x, &()).is_err());
    }

    #[test]
    fn test_ledoit_wolf_default() {
        let est = LedoitWolf::<f64>::default();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    // ---- OAS ----

    #[test]
    fn test_oas_basic() {
        let est = OAS::<f64>::new();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert!(fitted.shrinkage() >= 0.0);
        assert!(fitted.shrinkage() <= 1.0);
    }

    #[test]
    fn test_oas_shrinkage_in_range() {
        let est = OAS::<f64>::new();
        let x = array![
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3],
            [0.5, 0.3, 1.0],
            [1.2, 0.8, 0.4],
            [0.3, 1.1, 0.9],
            [0.7, 0.2, 1.3],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert!(
            fitted.shrinkage() >= 0.0 && fitted.shrinkage() <= 1.0,
            "shrinkage = {} out of range",
            fitted.shrinkage()
        );
    }

    #[test]
    fn test_oas_symmetric() {
        let est = OAS::<f64>::new();
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 3.0, 5.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_oas_mahalanobis() {
        let est = OAS::<f64>::new();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let dists = fitted.mahalanobis(&x).unwrap();
        assert_eq!(dists.len(), 4);
        for &d in dists.iter() {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_oas_insufficient_samples() {
        let est = OAS::<f64>::new();
        let x = array![[1.0, 2.0]];
        assert!(est.fit(&x, &()).is_err());
    }

    #[test]
    fn test_oas_default() {
        let est = OAS::<f64>::default();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    // ---- MinCovDet ----

    #[test]
    fn test_mincovdet_basic() {
        let est = MinCovDet::<f64>::new().random_state(42);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert_eq!(fitted.location().len(), 2);
        assert_eq!(fitted.support().len(), 7);
    }

    #[test]
    fn test_mincovdet_with_outlier() {
        let est = MinCovDet::<f64>::new().random_state(0);
        // Cluster at origin + one outlier.
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.1, -0.1],
            [0.05, 0.05],
            [10.0, 10.0], // outlier
        ];
        let fitted = est.fit(&x, &()).unwrap();
        // The outlier should generally not be in the support set.
        let dists = fitted.mahalanobis(&x).unwrap();
        // The outlier should have the largest distance.
        let outlier_dist = dists[6];
        for i in 0..6 {
            assert!(
                dists[i] < outlier_dist,
                "inlier dist {} should be < outlier dist {}",
                dists[i],
                outlier_dist
            );
        }
    }

    #[test]
    fn test_mincovdet_support_fraction() {
        let est = MinCovDet::<f64>::new()
            .support_fraction(0.7)
            .random_state(42);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let n_support: usize = fitted.support().iter().filter(|&&b| b).count();
        assert!(n_support >= 4, "expected at least 4 support samples");
    }

    #[test]
    fn test_mincovdet_invalid_support_fraction() {
        let est = MinCovDet::<f64>::new().support_fraction(0.3);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        assert!(est.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mincovdet_insufficient_samples() {
        let est = MinCovDet::<f64>::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(est.fit(&x, &()).is_err());
    }

    #[test]
    fn test_mincovdet_default() {
        let est = MinCovDet::<f64>::default();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    #[test]
    fn test_mincovdet_symmetric() {
        let est = MinCovDet::<f64>::new().random_state(0);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [2.0, 3.0, 5.0],
            [3.0, 4.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let cov = fitted.covariance();
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert_abs_diff_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-10);
            }
        }
    }

    // ---- EllipticEnvelope ----

    #[test]
    fn test_elliptic_envelope_basic() {
        let est = EllipticEnvelope::<f64>::new().random_state(42);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        assert_eq!(labels.len(), 7);
        for &l in labels.iter() {
            assert!(l == 1 || l == -1);
        }
    }

    #[test]
    fn test_elliptic_envelope_detects_outlier() {
        let est = EllipticEnvelope::<f64>::new()
            .contamination(0.15)
            .random_state(0);
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [-0.1, -0.1],
            [0.05, 0.05],
            [100.0, 100.0], // extreme outlier
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let labels = fitted.predict(&x).unwrap();
        // The extreme outlier should be labelled -1.
        assert_eq!(labels[6], -1, "extreme outlier should be detected");
    }

    #[test]
    fn test_elliptic_envelope_decision_function() {
        let est = EllipticEnvelope::<f64>::new().random_state(42);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        let scores = fitted.decision_function(&x).unwrap();
        let labels = fitted.predict(&x).unwrap();
        // Positive decision = inlier = +1, negative = outlier = -1.
        for i in 0..x.nrows() {
            if scores[i] > 0.0 {
                assert_eq!(labels[i], 1);
            } else {
                assert_eq!(labels[i], -1);
            }
        }
    }

    #[test]
    fn test_elliptic_envelope_invalid_contamination() {
        let est = EllipticEnvelope::<f64>::new().contamination(0.0);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ];
        assert!(est.fit(&x, &()).is_err());

        let est2 = EllipticEnvelope::<f64>::new().contamination(0.5);
        assert!(est2.fit(&x, &()).is_err());
    }

    #[test]
    fn test_elliptic_envelope_threshold_positive() {
        let est = EllipticEnvelope::<f64>::new().random_state(0);
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert!(fitted.threshold() > 0.0, "threshold should be positive");
    }

    #[test]
    fn test_elliptic_envelope_default() {
        let est = EllipticEnvelope::<f64>::default();
        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    // ---- Chi-squared approximation ----

    #[test]
    fn test_chi2_quantile_approx_sanity() {
        // chi2(2, 0.95) ~ 5.991
        let q = chi2_quantile_approx(2.0, 0.95);
        assert!(q > 4.0 && q < 8.0, "chi2(2,0.95) approx = {q}");

        // chi2(5, 0.99) ~ 15.086
        let q2 = chi2_quantile_approx(5.0, 0.99);
        assert!(q2 > 10.0 && q2 < 20.0, "chi2(5,0.99) approx = {q2}");
    }

    #[test]
    fn test_normal_quantile_approx_sanity() {
        // z(0.5) should be ~0
        let z_half = normal_quantile_approx(0.5);
        assert_abs_diff_eq!(z_half, 0.0, epsilon = 1e-10);

        // z(0.975) ~ 1.96
        let z_975 = normal_quantile_approx(0.975);
        assert!(
            (z_975 - 1.96).abs() < 0.05,
            "z(0.975) = {z_975}, expected ~1.96"
        );
    }

    // ---- f32 support ----

    #[test]
    fn test_empirical_covariance_f32() {
        let est = EmpiricalCovariance::<f32>::new();
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    #[test]
    fn test_shrunk_covariance_f32() {
        let est = ShrunkCovariance::<f32>::new(0.5);
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    #[test]
    fn test_ledoit_wolf_f32() {
        let est = LedoitWolf::<f32>::new();
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }

    #[test]
    fn test_oas_f32() {
        let est = OAS::<f32>::new();
        let x: Array2<f32> = array![
            [0.0f32, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let fitted = est.fit(&x, &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (2, 2));
    }
}
