//! scipy.special parity: high-value scalar special functions.
//!
//! All functions take and return `f64`. Implementations are concise series /
//! Lanczos-style approximations chosen for adequate accuracy across the
//! domain typically used by ML estimators (e.g. priors, log-likelihoods,
//! Gaussian CDFs).
//!
//! Provided:
//!
//! - [`gamma`] — Γ(x) via Lanczos.
//! - [`lgamma`] — log|Γ(x)| (signed for negative non-integer x).
//! - [`digamma`] — ψ(x) = d/dx ln Γ(x).
//! - [`beta`] — B(a, b).
//! - [`lbeta`] — log B(a, b).
//! - [`erf`] — error function via Abramowitz & Stegun 7.1.26 polynomial.
//! - [`erfc`] — 1 - erf(x), computed for accuracy in the tails.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Lanczos coefficients (g = 7, n = 9 — same set scipy uses)
// ---------------------------------------------------------------------------
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_2,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Compute Γ(x) for any real `x`.
///
/// Uses Lanczos's approximation; reflection formula handles negative `x`.
#[must_use]
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        // reflection: Γ(x) Γ(1-x) = π / sin(π x)
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let z = x - 1.0;
        let mut a = LANCZOS_COEFFS[0];
        for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
            a += c / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
    }
}

/// Compute ln|Γ(x)|.
///
/// Returns NaN for non-positive integer `x` (poles of Γ).
#[must_use]
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.5 {
        let s = (PI * x).sin().abs();
        if s == 0.0 {
            return f64::NAN;
        }
        // log Γ(x) = log π − log|sin πx| − log Γ(1 − x)
        return PI.ln() - s.ln() - lgamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut a = LANCZOS_COEFFS[0];
    for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Compute the digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence ψ(x) = ψ(x + 1) − 1/x to push x ≥ 6, then an
/// asymptotic expansion.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    let mut result = 0.0_f64;
    let mut z = x;
    if z < 0.5 {
        // reflection
        return digamma(1.0 - z) - PI / (PI * z).tan();
    }
    while z < 6.0 {
        result -= 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    // Asymptotic series: ψ(z) ≈ ln z − 1/(2z) − Σ B_{2n}/(2n z^{2n})
    result += z.ln() - 0.5 * inv;
    let mut acc = inv2;
    let coeffs = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        5.0 / 660.0,
    ];
    for &c in &coeffs {
        result -= c * acc;
        acc *= inv2;
    }
    result
}

/// Compute the Beta function B(a, b) = Γ(a) Γ(b) / Γ(a + b).
#[must_use]
pub fn beta(a: f64, b: f64) -> f64 {
    (lbeta(a, b)).exp()
}

/// Compute log B(a, b).
#[must_use]
pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Compute the error function erf(x) using Abramowitz & Stegun 7.1.26.
///
/// Maximum error is < 1.5e-7 over `(-∞, ∞)` — adequate for normal-CDF
/// downstream uses.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x_abs);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592;
    let y = 1.0 - poly * t * (-x_abs * x_abs).exp();
    sign * y
}

/// Compute the complementary error function erfc(x) = 1 - erf(x).
///
/// Computed via the same polynomial as [`erf`] without the `1 -` cancellation,
/// which preserves accuracy in the tails.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592;
    poly * t * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return false;
        }
        (a - b).abs() < tol || (b != 0.0 && ((a - b) / b).abs() < tol)
    }

    #[test]
    fn gamma_integers() {
        assert!(approx(gamma(1.0), 1.0, 1e-10));
        assert!(approx(gamma(2.0), 1.0, 1e-10));
        assert!(approx(gamma(3.0), 2.0, 1e-10));
        assert!(approx(gamma(4.0), 6.0, 1e-10));
        assert!(approx(gamma(5.0), 24.0, 1e-10));
        assert!(approx(gamma(10.0), 362_880.0, 1e-7));
    }

    #[test]
    fn gamma_half() {
        // Γ(1/2) = sqrt(π)
        assert!(approx(gamma(0.5), PI.sqrt(), 1e-10));
        // Γ(3/2) = sqrt(π) / 2
        assert!(approx(gamma(1.5), PI.sqrt() / 2.0, 1e-10));
    }

    #[test]
    fn gamma_reflection_negative() {
        // Γ(-0.5) = -2 sqrt(π)
        let expected = -2.0 * PI.sqrt();
        assert!(approx(gamma(-0.5), expected, 1e-10));
    }

    #[test]
    fn lgamma_matches_gamma() {
        for x in [0.5, 1.5, 2.5, 7.5, 12.0] {
            let lg = lgamma(x);
            let g = gamma(x);
            assert!(
                approx(lg, g.ln(), 1e-9),
                "lgamma({x})={lg} vs ln gamma={}",
                g.ln()
            );
        }
    }

    #[test]
    fn lgamma_pole_at_nonpositive_int() {
        assert!(lgamma(0.0).is_nan());
        assert!(lgamma(-1.0).is_nan());
        assert!(lgamma(-5.0).is_nan());
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ ≈ -0.577_215_664_901_532_86
        assert!(approx(digamma(1.0), -0.577_215_664_901_532_86, 1e-9));
        // ψ(2) = 1 - γ
        assert!(approx(digamma(2.0), 1.0 - 0.577_215_664_901_532_86, 1e-9));
        // ψ(0.5) = -γ - 2 ln 2
        let expected = -0.577_215_664_901_532_86 - 2.0 * 2.0_f64.ln();
        assert!(approx(digamma(0.5), expected, 1e-9));
    }

    #[test]
    fn beta_symmetry() {
        // B(a, b) = B(b, a)
        assert!(approx(beta(2.0, 5.0), beta(5.0, 2.0), 1e-12));
    }

    #[test]
    fn beta_known_value() {
        // B(2, 3) = 1/12
        assert!(approx(beta(2.0, 3.0), 1.0 / 12.0, 1e-9));
        // B(0.5, 0.5) = π
        assert!(approx(beta(0.5, 0.5), PI, 1e-9));
    }

    #[test]
    fn erf_known_values() {
        assert!(approx(erf(0.0), 0.0, 1e-9));
        assert!(approx(erf(1.0), 0.842_700_792_949_715, 2e-7));
        assert!(approx(erf(-1.0), -0.842_700_792_949_715, 2e-7));
        // erf(infinity) -> 1
        assert!(approx(erf(10.0), 1.0, 1e-9));
        assert!(approx(erf(-10.0), -1.0, 1e-9));
    }

    #[test]
    fn erfc_consistent_with_erf() {
        for &x in &[-2.0_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 5.0] {
            let lhs = erf(x) + erfc(x);
            assert!(
                (lhs - 1.0).abs() < 5e-7,
                "erf+erfc != 1 at x={x} (got {lhs})"
            );
        }
    }
}
