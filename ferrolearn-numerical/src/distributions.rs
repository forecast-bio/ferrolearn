//! Unified interface to statistical distributions, providing scipy.stats equivalents.
//!
//! This module wraps [`statrs`] distributions behind a common
//! [`ContinuousDistribution`] trait that exposes PDF, CDF, survival function,
//! percent-point function (inverse CDF), mean and variance in a single
//! interface.
//!
//! # Univariate distributions
//!
//! | Wrapper struct  | Underlying `statrs` type               |
//! |-----------------|----------------------------------------|
//! | [`Normal`]      | `statrs::distribution::Normal`         |
//! | [`ChiSquared`]  | `statrs::distribution::ChiSquared`     |
//! | [`FDist`]       | `statrs::distribution::FisherSnedecor` |
//! | [`StudentsT`]   | `statrs::distribution::StudentsT`      |
//! | [`Beta`]        | `statrs::distribution::Beta`           |
//! | [`Gamma`]       | `statrs::distribution::Gamma`          |
//!
//! # Multivariate distributions
//!
//! [`Dirichlet`] is a multivariate distribution and does not implement
//! [`ContinuousDistribution`]. It provides sampling (via [`rand 0.9`](rand))
//! and a log-PDF method instead.
//!
//! # Convenience functions
//!
//! Several free functions compute common p-values used in ML hypothesis tests:
//! [`chi2_sf`], [`f_sf`], [`t_test_two_tailed`], and [`norm_sf`].

use ndarray::Array1;
use statrs::distribution::{self as sd, Continuous, ContinuousCDF};
use statrs::statistics::Distribution as StatrsDistribution;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Unified interface for univariate continuous distributions.
///
/// Every method has a sensible mathematical definition; the default
/// implementation of [`sf`](ContinuousDistribution::sf) computes `1 - cdf(x)`.
pub trait ContinuousDistribution {
    /// Probability density function evaluated at `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function evaluated at `x`.
    fn cdf(&self, x: f64) -> f64;

    /// Survival function (1 - CDF) evaluated at `x`.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Percent-point function (inverse CDF / quantile function).
    ///
    /// Returns the value `x` such that `cdf(x) = p`.
    fn ppf(&self, p: f64) -> f64;

    /// Mean of the distribution.
    fn mean(&self) -> f64;

    /// Variance of the distribution.
    fn variance(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Unwrap an `Option<f64>` that statrs returns for mean/variance.
///
/// All the distributions we wrap have well-defined mean and variance for valid
/// parameters, so `None` indicates a programming error in parameter
/// construction.
fn unwrap_stat(opt: Option<f64>, name: &str, stat: &str) -> f64 {
    opt.unwrap_or_else(|| {
        panic!(
            "BUG: {name}::{stat}() returned None -- this should not happen \
             for valid parameters"
        )
    })
}

// ---------------------------------------------------------------------------
// Normal
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::Normal`.
///
/// Represents a normal (Gaussian) distribution with given mean and standard
/// deviation.
#[derive(Debug, Clone)]
pub struct Normal {
    inner: sd::Normal,
}

impl Normal {
    /// Creates a new normal distribution with the given `mean` and `std_dev`.
    ///
    /// # Errors
    ///
    /// Returns an error if `std_dev` is not positive, or if either parameter
    /// is NaN.
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, String> {
        sd::Normal::new(mean, std_dev)
            .map(|inner| Self { inner })
            .map_err(|e| format!("Normal::new failed: {e}"))
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        unwrap_stat(StatrsDistribution::mean(&self.inner), "Normal", "mean")
    }

    fn variance(&self) -> f64 {
        unwrap_stat(
            StatrsDistribution::variance(&self.inner),
            "Normal",
            "variance",
        )
    }
}

// ---------------------------------------------------------------------------
// ChiSquared
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::ChiSquared`.
///
/// Represents a chi-squared distribution with `df` degrees of freedom.
#[derive(Debug, Clone)]
pub struct ChiSquared {
    inner: sd::ChiSquared,
}

impl ChiSquared {
    /// Creates a new chi-squared distribution with the given degrees of
    /// freedom.
    ///
    /// # Errors
    ///
    /// Returns an error if `df` is not positive or is NaN.
    pub fn new(df: f64) -> Result<Self, String> {
        sd::ChiSquared::new(df)
            .map(|inner| Self { inner })
            .map_err(|e| format!("ChiSquared::new failed: {e}"))
    }
}

impl ContinuousDistribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        unwrap_stat(StatrsDistribution::mean(&self.inner), "ChiSquared", "mean")
    }

    fn variance(&self) -> f64 {
        unwrap_stat(
            StatrsDistribution::variance(&self.inner),
            "ChiSquared",
            "variance",
        )
    }
}

// ---------------------------------------------------------------------------
// FDist (Fisher-Snedecor)
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::FisherSnedecor`.
///
/// Represents the F-distribution with numerator degrees of freedom `df1` and
/// denominator degrees of freedom `df2`.
#[derive(Debug, Clone)]
pub struct FDist {
    inner: sd::FisherSnedecor,
}

impl FDist {
    /// Creates a new F-distribution with the given degrees of freedom.
    ///
    /// # Errors
    ///
    /// Returns an error if either `df1` or `df2` is not positive or is
    /// non-finite.
    pub fn new(df1: f64, df2: f64) -> Result<Self, String> {
        sd::FisherSnedecor::new(df1, df2)
            .map(|inner| Self { inner })
            .map_err(|e| format!("FDist::new failed: {e}"))
    }
}

impl ContinuousDistribution for FDist {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        // F-distribution mean is only defined for df2 > 2.
        // For valid parameters where mean is undefined, return NaN rather
        // than panicking.
        StatrsDistribution::mean(&self.inner).unwrap_or(f64::NAN)
    }

    fn variance(&self) -> f64 {
        // F-distribution variance is only defined for df2 > 4.
        StatrsDistribution::variance(&self.inner).unwrap_or(f64::NAN)
    }
}

// ---------------------------------------------------------------------------
// StudentsT
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::StudentsT`.
///
/// Represents Student's t-distribution with `df` degrees of freedom,
/// centered at 0 with scale 1.
#[derive(Debug, Clone)]
pub struct StudentsT {
    inner: sd::StudentsT,
}

impl StudentsT {
    /// Creates a new Student's t-distribution with the given degrees of
    /// freedom.
    ///
    /// The distribution is centered at 0 with scale 1 (the standard
    /// parameterization used in hypothesis testing).
    ///
    /// # Errors
    ///
    /// Returns an error if `df` is not positive or is NaN.
    pub fn new(df: f64) -> Result<Self, String> {
        // statrs::StudentsT takes (location, scale, freedom).
        sd::StudentsT::new(0.0, 1.0, df)
            .map(|inner| Self { inner })
            .map_err(|e| format!("StudentsT::new failed: {e}"))
    }
}

impl ContinuousDistribution for StudentsT {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        // t-distribution mean is only defined for df > 1.
        StatrsDistribution::mean(&self.inner).unwrap_or(f64::NAN)
    }

    fn variance(&self) -> f64 {
        // t-distribution variance is only defined for df > 2.
        StatrsDistribution::variance(&self.inner).unwrap_or(f64::NAN)
    }
}

// ---------------------------------------------------------------------------
// Beta
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::Beta`.
///
/// Represents the Beta distribution with shape parameters `alpha` and `beta`.
#[derive(Debug, Clone)]
pub struct Beta {
    inner: sd::Beta,
}

impl Beta {
    /// Creates a new Beta distribution with the given shape parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if either `alpha` or `beta` is not positive, is
    /// infinite, or is NaN.
    pub fn new(alpha: f64, beta: f64) -> Result<Self, String> {
        sd::Beta::new(alpha, beta)
            .map(|inner| Self { inner })
            .map_err(|e| format!("Beta::new failed: {e}"))
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        unwrap_stat(StatrsDistribution::mean(&self.inner), "Beta", "mean")
    }

    fn variance(&self) -> f64 {
        unwrap_stat(
            StatrsDistribution::variance(&self.inner),
            "Beta",
            "variance",
        )
    }
}

// ---------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------

/// Wrapper around `statrs::distribution::Gamma`.
///
/// Represents the Gamma distribution parameterized by `shape` and `rate`
/// (inverse scale).
#[derive(Debug, Clone)]
pub struct Gamma {
    inner: sd::Gamma,
}

impl Gamma {
    /// Creates a new Gamma distribution with the given `shape` and `rate`.
    ///
    /// # Errors
    ///
    /// Returns an error if either parameter is not positive or is NaN.
    pub fn new(shape: f64, rate: f64) -> Result<Self, String> {
        sd::Gamma::new(shape, rate)
            .map(|inner| Self { inner })
            .map_err(|e| format!("Gamma::new failed: {e}"))
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        Continuous::pdf(&self.inner, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    fn sf(&self, x: f64) -> f64 {
        ContinuousCDF::sf(&self.inner, x)
    }

    fn ppf(&self, p: f64) -> f64 {
        ContinuousCDF::inverse_cdf(&self.inner, p)
    }

    fn mean(&self) -> f64 {
        unwrap_stat(StatrsDistribution::mean(&self.inner), "Gamma", "mean")
    }

    fn variance(&self) -> f64 {
        unwrap_stat(
            StatrsDistribution::variance(&self.inner),
            "Gamma",
            "variance",
        )
    }
}

// ---------------------------------------------------------------------------
// Dirichlet (multivariate -- sampling only, no scalar CDF)
// ---------------------------------------------------------------------------

/// Dirichlet distribution over the probability simplex.
///
/// The Dirichlet distribution is a multivariate distribution and does **not**
/// implement [`ContinuousDistribution`] because its PDF is multivariate and
/// it has no scalar CDF.
///
/// Sampling uses [`rand 0.9`](rand) via the standard Gamma-based algorithm
/// (sample independent `Gamma(alpha_i, 1)` variates and normalize).
///
/// The log-PDF is computed directly using the formula:
///
/// ```text
/// ln(pdf(x)) = ln(Gamma(sum(alpha))) - sum(ln(Gamma(alpha_i)))
///              + sum((alpha_i - 1) * ln(x_i))
/// ```
#[derive(Debug, Clone)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    /// Creates a new Dirichlet distribution with the given concentration
    /// parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` has fewer than 2 elements, or if any
    /// element is non-positive, infinite, or NaN.
    pub fn new(alpha: &[f64]) -> Result<Self, String> {
        if alpha.len() < 2 {
            return Err("Dirichlet::new failed: alpha must have at least 2 elements".into());
        }
        if alpha.iter().any(|&a| !a.is_finite() || a <= 0.0) {
            return Err(
                "Dirichlet::new failed: all alpha elements must be finite and positive".into(),
            );
        }
        Ok(Self {
            alpha: alpha.to_vec(),
        })
    }

    /// Draws a single sample from the Dirichlet distribution.
    ///
    /// The returned array sums to 1.0 (up to floating-point precision) and
    /// has the same length as the concentration parameter vector.
    ///
    /// Uses the standard algorithm: sample independent
    /// `Gamma(alpha_i, 1)` variates and normalize by their sum.
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Array1<f64> {
        use rand_distr::Distribution as _;

        let k = self.alpha.len();
        let mut samples = Array1::zeros(k);
        let mut sum = 0.0;

        for (i, &a) in self.alpha.iter().enumerate() {
            // rand_distr::Gamma uses (shape, scale) parameterization.
            let gamma = rand_distr::Gamma::new(a, 1.0)
                .expect("Gamma distribution parameters already validated");
            let val = gamma.sample(rng);
            samples[i] = val;
            sum += val;
        }

        // Normalize so the sample lies on the simplex.
        if sum > 0.0 {
            samples /= sum;
        }

        samples
    }

    /// Computes the natural logarithm of the multivariate PDF at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have the same length as the concentration
    /// parameter vector, if any element is outside `(0, 1)`, or if the
    /// elements do not sum to approximately 1.
    pub fn ln_pdf(&self, x: &[f64]) -> f64 {
        use statrs::function::gamma::ln_gamma;

        assert_eq!(
            self.alpha.len(),
            x.len(),
            "x must have the same length as alpha"
        );

        let mut sum_x = 0.0;
        let mut sum_alpha = 0.0;
        let mut term = 0.0;

        for (&x_i, &alpha_i) in x.iter().zip(self.alpha.iter()) {
            assert!(
                0.0 < x_i && x_i < 1.0,
                "all elements of x must be in (0, 1)"
            );
            term += (alpha_i - 1.0) * x_i.ln() - ln_gamma(alpha_i);
            sum_x += x_i;
            sum_alpha += alpha_i;
        }

        assert!(
            (sum_x - 1.0).abs() < 1e-4,
            "elements of x must sum to approximately 1"
        );

        term + ln_gamma(sum_alpha)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Survival function (1 - CDF) for the chi-squared distribution.
///
/// Returns the p-value for a chi-squared test statistic `x` with `df`
/// degrees of freedom.
///
/// # Panics
///
/// Panics if `df` is not positive.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    let d = ChiSquared::new(df).expect("chi2_sf: invalid df");
    d.sf(x)
}

/// Survival function (1 - CDF) for the F-distribution.
///
/// Returns the p-value for an F-test statistic `x` with `df1` and `df2`
/// degrees of freedom.
///
/// # Panics
///
/// Panics if either `df1` or `df2` is not positive.
pub fn f_sf(x: f64, df1: f64, df2: f64) -> f64 {
    let d = FDist::new(df1, df2).expect("f_sf: invalid degrees of freedom");
    d.sf(x)
}

/// Two-tailed p-value for a t-test statistic.
///
/// Returns `2 * P(T > |t_stat|)` where `T` follows a Student's
/// t-distribution with `df` degrees of freedom.
///
/// # Panics
///
/// Panics if `df` is not positive.
pub fn t_test_two_tailed(t_stat: f64, df: f64) -> f64 {
    let d = StudentsT::new(df).expect("t_test_two_tailed: invalid df");
    2.0 * d.sf(t_stat.abs())
}

/// Survival function (1 - CDF) for the standard normal distribution.
///
/// Returns the one-tailed p-value for a z-score.
///
/// # Panics
///
/// This function does not panic for any finite `z`.
pub fn norm_sf(z: f64) -> f64 {
    let d = Normal::new(0.0, 1.0).expect("standard normal is always valid");
    d.sf(z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normal_pdf_cdf_ppf() {
        let n = Normal::new(0.0, 1.0).unwrap();

        // pdf(0) for N(0,1) = 1/sqrt(2*pi) ~ 0.3989
        assert_abs_diff_eq!(n.pdf(0.0), 0.3989422804014327, epsilon = 1e-3);

        // cdf(0) = 0.5
        assert_abs_diff_eq!(n.cdf(0.0), 0.5, epsilon = 1e-6);

        // ppf(0.975) ~ 1.96
        assert_abs_diff_eq!(n.ppf(0.975), 1.959963984540054, epsilon = 1e-3);

        // mean and variance
        assert_abs_diff_eq!(n.mean(), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(n.variance(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn chi2_cdf() {
        let c = ChiSquared::new(2.0).unwrap();

        // chi2(2) cdf at 5.991 ~ 0.95
        assert_abs_diff_eq!(c.cdf(5.991), 0.95, epsilon = 1e-3);

        // mean = df = 2, variance = 2*df = 4
        assert_abs_diff_eq!(c.mean(), 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(c.variance(), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn f_dist_sf() {
        let f = FDist::new(3.0, 40.0).unwrap();

        // F(3,40) sf at 2.84 ~ 0.05
        assert_abs_diff_eq!(f.sf(2.84), 0.05, epsilon = 1e-2);
    }

    #[test]
    fn t_dist_symmetry() {
        let t = StudentsT::new(10.0).unwrap();

        // For a symmetric distribution: cdf(x) + cdf(-x) = 1
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            let sum = t.cdf(x) + t.cdf(-x);
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }

        // sf(x) = 1 - cdf(x)
        for &x in &[0.0, 1.0, -1.5] {
            assert_abs_diff_eq!(t.sf(x), 1.0 - t.cdf(x), epsilon = 1e-6);
        }
    }

    #[test]
    fn beta_mean_variance() {
        let b = Beta::new(2.0, 5.0).unwrap();

        // Beta(2,5): mean = alpha / (alpha + beta) = 2/7
        assert_abs_diff_eq!(b.mean(), 2.0 / 7.0, epsilon = 1e-6);

        // variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        //          = 10 / (49 * 8) = 10/392
        let expected_var = (2.0 * 5.0) / ((2.0 + 5.0_f64).powi(2) * (2.0 + 5.0 + 1.0));
        assert_abs_diff_eq!(b.variance(), expected_var, epsilon = 1e-6);
    }

    #[test]
    fn gamma_ppf_round_trip() {
        let g = Gamma::new(2.0, 1.0).unwrap();

        // ppf(cdf(x)) should round-trip
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            let p = g.cdf(x);
            let x_back = g.ppf(p);
            assert_abs_diff_eq!(x_back, x, epsilon = 1e-3);
        }
    }

    #[test]
    fn dirichlet_sample_sums_to_one() {
        use rand::SeedableRng;

        let dir = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..100 {
            let s = dir.sample(&mut rng);
            assert_eq!(s.len(), 3);
            assert_abs_diff_eq!(s.sum(), 1.0, epsilon = 1e-10);
            // All components should be positive.
            for &v in s.iter() {
                assert!(v > 0.0);
            }
        }
    }

    #[test]
    fn chi2_sf_matches_cdf() {
        let df = 5.0;
        let c = ChiSquared::new(df).unwrap();

        for &x in &[1.0, 3.0, 5.0, 10.0, 15.0] {
            let sf_val = chi2_sf(x, df);
            let cdf_val = c.cdf(x);
            assert_abs_diff_eq!(sf_val, 1.0 - cdf_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn convenience_p_values() {
        // chi2_sf: chi2(1) sf at 3.841 ~ 0.05
        assert_abs_diff_eq!(chi2_sf(3.841, 1.0), 0.05, epsilon = 1e-3);

        // f_sf: F(1,100) sf at 3.936 ~ 0.05
        assert_abs_diff_eq!(f_sf(3.936, 1.0, 100.0), 0.05, epsilon = 1e-2);

        // t_test_two_tailed: t(30) at 2.042 ~ 0.05
        assert_abs_diff_eq!(t_test_two_tailed(2.042, 30.0), 0.05, epsilon = 1e-2);

        // norm_sf: z=1.645 ~ 0.05
        assert_abs_diff_eq!(norm_sf(1.645), 0.05, epsilon = 1e-3);

        // norm_sf: z=0 = 0.5
        assert_abs_diff_eq!(norm_sf(0.0), 0.5, epsilon = 1e-6);
    }
}
