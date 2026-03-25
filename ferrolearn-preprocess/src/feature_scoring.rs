//! Feature scoring functions for feature selection.
//!
//! This module provides standalone univariate scoring functions that compute
//! per-feature statistics and p-values:
//!
//! - [`f_classif`] — ANOVA F-statistic for classification.
//! - [`f_regression`] — univariate F-statistic via Pearson correlation.
//! - [`chi2`] — chi-squared statistic for non-negative features.
//!
//! These functions return `(F-statistics, p-values)` tuples and can be used
//! directly or passed to [`SelectKBest`](crate::feature_selection::SelectKBest)
//! / [`SelectPercentile`](crate::select_percentile::SelectPercentile) via the
//! [`ScoreFunc`](crate::feature_selection::ScoreFunc) enum.

use ferrolearn_core::error::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ===========================================================================
// f_classif — ANOVA F-statistic
// ===========================================================================

/// Compute the ANOVA F-statistic and approximate p-values for each feature.
///
/// For each feature column the between-class and within-class sum of squares
/// are computed. The F-statistic is:
///
/// ```text
/// F = (SSB / (k - 1)) / (SSW / (n - k))
/// ```
///
/// where *k* is the number of distinct classes and *n* is the number of
/// samples.
///
/// P-values are approximated using the regularized incomplete beta function
/// from `ferrolearn-numerical`. If the numerical CDF is unavailable, `NaN`
/// is returned for the p-value.
///
/// # Returns
///
/// `(f_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has zero rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
/// - [`FerroError::InvalidParameter`] if fewer than 2 classes are present.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::f_classif;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 100.0], [2.0, 200.0], [10.0, 100.0], [11.0, 200.0]];
/// let y: Array1<usize> = array![0, 0, 1, 1];
/// let (f_stats, p_vals) = f_classif(&x, &y).unwrap();
/// assert_eq!(f_stats.len(), 2);
/// // Feature 0 separates classes well → high F
/// assert!(f_stats[0] > f_stats[1]);
/// ```
pub fn f_classif<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "f_classif".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "f_classif — y must have same length as x rows".into(),
        });
    }

    // Collect per-class row indices
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }
    let n_classes = class_indices.len();
    if n_classes < 2 {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: format!("f_classif requires at least 2 classes, got {n_classes}"),
        });
    }

    let n_features = x.ncols();
    let n_f = F::from(n_samples).unwrap();

    let df_between = n_classes - 1;
    let df_within = n_samples - n_classes;
    let df_b = F::from(df_between).unwrap();
    let df_w = F::from(df_within).unwrap();

    let mut f_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let grand_mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;

        let mut ss_between = F::zero();
        let mut ss_within = F::zero();

        for rows in class_indices.values() {
            let n_k = F::from(rows.len()).unwrap();
            let class_mean = rows
                .iter()
                .map(|&i| col[i])
                .fold(F::zero(), |acc, v| acc + v)
                / n_k;
            let diff = class_mean - grand_mean;
            ss_between = ss_between + n_k * diff * diff;
            for &i in rows {
                let d = col[i] - class_mean;
                ss_within = ss_within + d * d;
            }
        }

        let f = if df_w == F::zero() {
            F::zero()
        } else {
            let ms_between = ss_between / df_b;
            let ms_within = ss_within / df_w;
            if ms_within == F::zero() {
                F::infinity()
            } else {
                ms_between / ms_within
            }
        };

        f_stats[j] = f;
        p_vals[j] = f_distribution_sf(f, df_between, df_within);
    }

    Ok((f_stats, p_vals))
}

// ===========================================================================
// f_regression — Pearson correlation-based F-statistic
// ===========================================================================

/// Compute univariate F-statistics via Pearson correlation for regression.
///
/// For each feature the Pearson correlation coefficient *r* with the target
/// is computed, then:
///
/// ```text
/// F = r^2 * (n - 2) / (1 - r^2)
/// ```
///
/// # Returns
///
/// `(f_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has fewer than 3 rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::f_regression;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 100.0], [2.0, 200.0], [3.0, 100.0], [4.0, 200.0]];
/// let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
/// let (f_stats, _p_vals) = f_regression(&x, &y).unwrap();
/// assert_eq!(f_stats.len(), 2);
/// ```
pub fn f_regression<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples < 3 {
        return Err(FerroError::InsufficientSamples {
            required: 3,
            actual: n_samples,
            context: "f_regression requires at least 3 samples".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "f_regression — y must have same length as x rows".into(),
        });
    }

    let n_f = F::from(n_samples).unwrap();
    let n_features = x.ncols();

    // Precompute y stats
    let y_mean = y.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;
    let y_var = y
        .iter()
        .copied()
        .map(|v| (v - y_mean) * (v - y_mean))
        .fold(F::zero(), |acc, v| acc + v);

    let two = F::from(2.0).unwrap();

    let mut f_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let x_mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;
        let x_var = col
            .iter()
            .copied()
            .map(|v| (v - x_mean) * (v - x_mean))
            .fold(F::zero(), |acc, v| acc + v);

        let cov = col
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .fold(F::zero(), |acc, v| acc + v);

        let denom = x_var * y_var;
        let r = if denom == F::zero() {
            F::zero()
        } else {
            cov / denom.sqrt()
        };

        let r2 = r * r;
        let f = if r2 >= F::one() {
            F::infinity()
        } else {
            r2 * (n_f - two) / (F::one() - r2)
        };

        f_stats[j] = f;
        // F-distribution with df1=1, df2=n-2
        p_vals[j] = f_distribution_sf(f, 1, n_samples - 2);
    }

    Ok((f_stats, p_vals))
}

// ===========================================================================
// chi2 — Chi-squared statistic
// ===========================================================================

/// Compute chi-squared statistics between each non-negative feature and the
/// class labels.
///
/// For each feature the observed and expected frequencies per class are
/// computed, then:
///
/// ```text
/// chi2 = sum_class (observed - expected)^2 / expected
/// ```
///
/// where `observed` is the sum of feature values for samples of that class,
/// and `expected` is the expected sum under the null hypothesis (proportional
/// to the class frequency and the overall feature sum).
///
/// # Returns
///
/// `(chi2_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has zero rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
/// - [`FerroError::InvalidParameter`] if any feature value is negative.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::chi2;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
/// let y: Array1<usize> = array![0, 1, 0, 1];
/// let (chi2_stats, _p_vals) = chi2(&x, &y).unwrap();
/// assert_eq!(chi2_stats.len(), 2);
/// ```
pub fn chi2<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "chi2".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "chi2 — y must have same length as x rows".into(),
        });
    }

    // Validate non-negative
    for j in 0..x.ncols() {
        for i in 0..n_samples {
            if x[[i, j]] < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "x".into(),
                    reason: format!(
                        "chi2 requires non-negative features, found negative value at ({i}, {j})"
                    ),
                });
            }
        }
    }

    // Collect per-class row indices
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }

    let n_classes = class_indices.len();
    let n_features = x.ncols();
    let n_f = F::from(n_samples).unwrap();

    let mut chi2_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let total_sum = col.iter().copied().fold(F::zero(), |acc, v| acc + v);

        if total_sum == F::zero() {
            // All zero → chi2 = 0, p = 1
            chi2_stats[j] = F::zero();
            p_vals[j] = F::one();
            continue;
        }

        let mut chi2_val = F::zero();

        for rows in class_indices.values() {
            let n_k = F::from(rows.len()).unwrap();
            let observed = rows
                .iter()
                .map(|&i| col[i])
                .fold(F::zero(), |acc, v| acc + v);
            let expected = total_sum * n_k / n_f;

            if expected > F::zero() {
                let diff = observed - expected;
                chi2_val = chi2_val + diff * diff / expected;
            }
        }

        chi2_stats[j] = chi2_val;
        // Chi-squared distribution with df = n_classes - 1
        let df = n_classes.saturating_sub(1);
        p_vals[j] = chi2_distribution_sf(chi2_val, df);
    }

    Ok((chi2_stats, p_vals))
}

// ===========================================================================
// Distribution helper: F-distribution survival function (1 - CDF)
// ===========================================================================

/// Approximate the survival function (1 - CDF) of the F-distribution.
///
/// Uses the relationship between the F-distribution and the regularized
/// incomplete beta function:
///
/// ```text
/// P(F > x) = I_{d2/(d2 + d1*x)}(d2/2, d1/2)
/// ```
///
/// Returns `NaN` if the computation cannot be performed.
fn f_distribution_sf<F: Float>(x: F, df1: usize, df2: usize) -> F {
    if x <= F::zero() {
        return F::one();
    }
    if df1 == 0 || df2 == 0 {
        return F::nan();
    }

    let d1 = F::from(df1).unwrap();
    let d2 = F::from(df2).unwrap();

    // I_{d2/(d2 + d1*x)}(d2/2, d1/2)
    let z = d2 / (d2 + d1 * x);
    let a = d2 / F::from(2.0).unwrap();
    let b = d1 / F::from(2.0).unwrap();

    regularized_incomplete_beta(z, a, b)
}

/// Approximate the survival function (1 - CDF) of the chi-squared distribution.
///
/// Uses the relationship: chi2 with k df = Gamma(k/2, 2), and
/// P(X > x) = 1 - gamma_cdf = upper regularized gamma Q(k/2, x/2).
///
/// We use the relationship to the regularized incomplete beta function:
/// Q(a, x) = I_{x/(x+a)}(... ) — but more simply, chi2 with k df is
/// equivalent to F(k, inf) scaled. We use a direct series approximation.
fn chi2_distribution_sf<F: Float>(x: F, df: usize) -> F {
    if x <= F::zero() {
        return F::one();
    }
    if df == 0 {
        return F::nan();
    }

    // Use the upper regularized gamma function Q(k/2, x/2)
    let a = F::from(df).unwrap() / F::from(2.0).unwrap();
    let z = x / F::from(2.0).unwrap();

    upper_regularized_gamma(a, z)
}

/// Upper regularized gamma function Q(a, x) = 1 - P(a, x).
///
/// Uses a continued fraction expansion for x >= a + 1, and the series
/// expansion otherwise.
fn upper_regularized_gamma<F: Float>(a: F, x: F) -> F {
    if x <= F::zero() {
        return F::one();
    }

    let one = F::one();
    let two = F::from(2.0).unwrap();

    // Use series for P(a, x) when x < a + 1, then Q = 1 - P
    if x < a + one {
        let p = lower_regularized_gamma_series(a, x);
        return one - p;
    }

    // Continued fraction for Q(a, x) — Lentz's method
    let eps = F::from(1.0e-12).unwrap();
    let tiny = F::from(1.0e-30).unwrap();

    let mut c = tiny;
    let mut d = F::one() / (x + one - a);
    let mut f = d;

    for n_iter in 1..200 {
        let n = F::from(n_iter).unwrap();
        // Even term
        let an_even = n * (a - n);
        let bn_even = x + two * n + one - a;
        d = F::one() / (bn_even + an_even * d);
        c = bn_even + an_even / c;
        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < eps {
            break;
        }
    }

    // Q(a, x) = e^(-x) * x^a / Gamma(a) * f
    let log_prefix = a * x.ln() - x - ln_gamma(a);
    let prefix = log_prefix.exp();
    let result = prefix * f;

    // Clamp to [0, 1]
    if result < F::zero() {
        F::zero()
    } else if result > one {
        one
    } else {
        result
    }
}

/// Lower regularized gamma function P(a, x) via series expansion.
fn lower_regularized_gamma_series<F: Float>(a: F, x: F) -> F {
    let eps = F::from(1.0e-12).unwrap();
    let one = F::one();

    let mut sum = one / a;
    let mut term = one / a;

    for n in 1..200 {
        let n_f = F::from(n).unwrap();
        term = term * x / (a + n_f);
        sum = sum + term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - ln_gamma(a);
    let result = log_prefix.exp() * sum;

    // Clamp to [0, 1]
    if result < F::zero() {
        F::zero()
    } else if result > one {
        one
    } else {
        result
    }
}

/// Regularized incomplete beta function I_x(a, b) using a continued fraction
/// (Lentz's method).
fn regularized_incomplete_beta<F: Float>(x: F, a: F, b: F) -> F {
    let one = F::one();
    let two = F::from(2.0).unwrap();

    if x <= F::zero() {
        return F::zero();
    }
    if x >= one {
        return one;
    }

    // Use the symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + one) / (a + b + two) {
        return one - regularized_incomplete_beta(one - x, b, a);
    }

    // Prefix: x^a * (1-x)^b / (a * Beta(a,b))
    let log_prefix = a * x.ln() + b * (one - x).ln() - ln_beta(a, b) - a.ln();
    let prefix = log_prefix.exp();

    // Continued fraction (Lentz's algorithm)
    let eps = F::from(1.0e-12).unwrap();
    let tiny = F::from(1.0e-30).unwrap();

    let mut f = tiny;
    let mut c = tiny;
    let mut d = one;

    for m in 0..200 {
        let m_f = F::from(m).unwrap();
        let (a_m, b_m) = if m == 0 {
            (one, one)
        } else if m % 2 == 0 {
            // Even: d_{2m} term
            let k = m_f / two;
            let num = k * (b - k) * x / ((a + two * k - one) * (a + two * k));
            (num, one)
        } else {
            // Odd: d_{2m+1} term
            let k = (m_f - one) / two;
            let num =
                -((a + k) * (a + b + k) * x) / ((a + two * k) * (a + two * k + one));
            (num, one)
        };

        if m == 0 {
            f = b_m;
            c = b_m;
            d = one / b_m;
            continue;
        }

        d = b_m + a_m * d;
        if d.abs() < tiny {
            d = tiny;
        }
        d = one / d;

        c = b_m + a_m / c;
        if c.abs() < tiny {
            c = tiny;
        }

        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < eps {
            break;
        }
    }

    let result = prefix * f;
    if result < F::zero() {
        F::zero()
    } else if result > one {
        one
    } else {
        result
    }
}

/// Log of the beta function: ln(Beta(a, b)) = lnGamma(a) + lnGamma(b) - lnGamma(a+b).
fn ln_beta<F: Float>(a: F, b: F) -> F {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation of ln(Gamma(x)) for x > 0.
fn ln_gamma<F: Float>(x: F) -> F {
    // Lanczos coefficients (g=7, n=9)
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let one = F::one();
    let half = F::from(0.5).unwrap();
    let g = F::from(7.0).unwrap();

    if x < half {
        // Reflection formula
        let pi = F::from(std::f64::consts::PI).unwrap();
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(one - x);
    }

    let z = x - one;
    let mut sum = F::from(coefs[0]).unwrap();
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        sum = sum + F::from(c).unwrap() / (z + F::from(i).unwrap());
    }

    let t = z + g + half;
    let sqrt_2pi = F::from(2.506_628_274_631_000_5).unwrap();

    sqrt_2pi.ln() + (z + half) * t.ln() - t + sum.ln()
}

// ===========================================================================
// ScoreFunc integration
// ===========================================================================

/// Add `FRegression` and `Chi2` variants to `ScoreFunc`.
///
/// This cannot extend the existing enum directly, so we provide adapter
/// functions that compute scores in the format expected by `SelectKBest`.
///
/// Compute scores for the given score function name, returning F-scores only.
///
/// This is a convenience dispatcher for integration with feature selection.
pub fn compute_scores_classif<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    func: &str,
) -> Result<Vec<F>, FerroError> {
    match func {
        "f_classif" => {
            let (f_stats, _) = f_classif(x, y)?;
            Ok(f_stats.to_vec())
        }
        "chi2" => {
            let (chi2_stats, _) = chi2(x, y)?;
            Ok(chi2_stats.to_vec())
        }
        _ => Err(FerroError::InvalidParameter {
            name: "func".into(),
            reason: format!("unknown classification score function: {func}"),
        }),
    }
}

/// Compute regression scores, returning F-statistics only.
pub fn compute_scores_regression<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Vec<F>, FerroError> {
    let (f_stats, _) = f_regression(x, y)?;
    Ok(f_stats.to_vec())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // f_classif tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f_classif_basic() {
        // Feature 0 separates classes well, feature 1 does not
        let x = array![
            [1.0_f64, 5.0],
            [1.5, 5.5],
            [2.0, 4.5],
            [10.0, 5.0],
            [10.5, 4.5],
            [11.0, 5.5]
        ];
        let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
        let (f_stats, p_vals) = f_classif(&x, &y).unwrap();
        assert_eq!(f_stats.len(), 2);
        assert_eq!(p_vals.len(), 2);
        // Feature 0 should have much higher F than feature 1
        assert!(f_stats[0] > f_stats[1]);
        // p-value for feature 0 should be very small
        assert!(p_vals[0] < 0.05);
    }

    #[test]
    fn test_f_classif_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_shape_mismatch() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1, 2]; // wrong length
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_single_class_error() {
        let x = array![[1.0_f64], [2.0], [3.0]];
        let y: Array1<usize> = array![0, 0, 0];
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_perfect_separation() {
        // Feature perfectly separates classes → infinite F
        let x = array![[0.0_f64], [0.0], [1.0], [1.0]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let (f_stats, _) = f_classif(&x, &y).unwrap();
        assert!(f_stats[0].is_infinite());
    }

    #[test]
    fn test_f_classif_p_values_bounded() {
        let x = array![
            [1.0_f64, 10.0],
            [2.0, 20.0],
            [3.0, 10.0],
            [4.0, 20.0],
            [5.0, 10.0],
            [6.0, 20.0]
        ];
        let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
        let (_, p_vals) = f_classif(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!(p >= 0.0 && p <= 1.0, "p-value {p} out of bounds");
        }
    }

    // -----------------------------------------------------------------------
    // f_regression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f_regression_perfect_correlation() {
        // Feature 0 = target → r=1 → F=infinity
        let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].is_infinite() || f_stats[0] > 1.0e6);
    }

    #[test]
    fn test_f_regression_no_correlation() {
        // Orthogonal feature → r≈0 → F≈0
        let x = array![[1.0_f64], [-1.0], [1.0], [-1.0]];
        let y: Array1<f64> = array![1.0, 1.0, -1.0, -1.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].abs() < 1.0e-6);
    }

    #[test]
    fn test_f_regression_too_few_samples() {
        let x = array![[1.0_f64], [2.0]];
        let y: Array1<f64> = array![1.0, 2.0];
        assert!(f_regression(&x, &y).is_err());
    }

    #[test]
    fn test_f_regression_shape_mismatch() {
        let x = array![[1.0_f64], [2.0], [3.0]];
        let y: Array1<f64> = array![1.0, 2.0]; // wrong length
        assert!(f_regression(&x, &y).is_err());
    }

    #[test]
    fn test_f_regression_p_values_bounded() {
        let x = array![
            [1.0_f64, 10.0],
            [2.0, 20.0],
            [3.0, 15.0],
            [4.0, 25.0],
            [5.0, 10.0]
        ];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, p_vals) = f_regression(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!(p >= 0.0 && p <= 1.0, "p-value {p} out of bounds");
        }
    }

    #[test]
    fn test_f_regression_constant_feature() {
        // Constant feature → r=0 → F=0
        let x = array![[5.0_f64], [5.0], [5.0], [5.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].abs() < 1.0e-6);
    }

    // -----------------------------------------------------------------------
    // chi2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chi2_basic() {
        // Feature 0 correlates with class, feature 1 is random
        let x = array![
            [1.0_f64, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ];
        let y: Array1<usize> = array![1, 1, 0, 0, 1, 1, 0, 0];
        let (chi2_stats, p_vals) = chi2(&x, &y).unwrap();
        assert_eq!(chi2_stats.len(), 2);
        assert_eq!(p_vals.len(), 2);
        // Feature 0 perfectly correlates → higher chi2
        assert!(chi2_stats[0] > chi2_stats[1]);
    }

    #[test]
    fn test_chi2_negative_value_error() {
        let x = array![[1.0_f64, -1.0], [0.0, 1.0]];
        let y: Array1<usize> = array![0, 1];
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_shape_mismatch() {
        let x = array![[1.0_f64], [2.0]];
        let y: Array1<usize> = array![0]; // wrong length
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_all_zeros() {
        let x = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let y: Array1<usize> = array![0, 1];
        let (chi2_stats, p_vals) = chi2(&x, &y).unwrap();
        assert_eq!(chi2_stats[0], 0.0);
        assert_eq!(p_vals[0], 1.0);
    }

    #[test]
    fn test_chi2_p_values_bounded() {
        let x = array![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let y: Array1<usize> = array![0, 1, 0, 1, 0, 1];
        let (_, p_vals) = chi2(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!(p >= 0.0 && p <= 1.0, "p-value {p} out of bounds");
        }
    }

    // -----------------------------------------------------------------------
    // Distribution helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1 → ln = 0
        let val: f64 = ln_gamma(1.0);
        assert!((val).abs() < 1.0e-10);

        // Gamma(2) = 1 → ln = 0
        let val2: f64 = ln_gamma(2.0);
        assert!((val2).abs() < 1.0e-10);

        // Gamma(3) = 2 → ln = ln(2)
        let val3: f64 = ln_gamma(3.0);
        assert!((val3 - 2.0_f64.ln()).abs() < 1.0e-10);

        // Gamma(0.5) = sqrt(pi) → ln = 0.5 * ln(pi)
        let val4: f64 = ln_gamma(0.5);
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((val4 - expected).abs() < 1.0e-8);
    }

    #[test]
    fn test_regularized_incomplete_beta_boundaries() {
        // I_0(a, b) = 0
        let val: f64 = regularized_incomplete_beta(0.0, 1.0, 1.0);
        assert!((val).abs() < 1.0e-10);

        // I_1(a, b) = 1
        let val2: f64 = regularized_incomplete_beta(1.0, 1.0, 1.0);
        assert!((val2 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_f_distribution_sf_zero() {
        // P(F > 0) = 1
        let val: f64 = f_distribution_sf(0.0, 2, 10);
        assert!((val - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_f_distribution_sf_large_f() {
        // Very large F → p ≈ 0
        let val: f64 = f_distribution_sf(1000.0, 2, 100);
        assert!(val < 0.001);
    }

    // -----------------------------------------------------------------------
    // compute_scores_classif / compute_scores_regression
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_scores_classif_f_classif() {
        let x = array![
            [1.0_f64, 5.0],
            [1.5, 5.5],
            [10.0, 5.0],
            [10.5, 4.5]
        ];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let scores = compute_scores_classif(&x, &y, "f_classif").unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_compute_scores_classif_chi2() {
        let x = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let y: Array1<usize> = array![0, 1, 0, 1];
        let scores = compute_scores_classif(&x, &y, "chi2").unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_compute_scores_classif_unknown() {
        let x = array![[1.0_f64]];
        let y: Array1<usize> = array![0];
        assert!(compute_scores_classif(&x, &y, "unknown").is_err());
    }

    #[test]
    fn test_compute_scores_regression() {
        let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let scores = compute_scores_regression(&x, &y).unwrap();
        assert_eq!(scores.len(), 2);
    }
}
