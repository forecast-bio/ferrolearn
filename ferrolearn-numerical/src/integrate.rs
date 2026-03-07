//! Adaptive numerical quadrature — `scipy.integrate` equivalents.
//!
//! This module provides two families of numerical integration routines:
//!
//! - **Adaptive Simpson's rule** ([`quad`], [`quad_with_limit`]) — recursive
//!   subdivision with Richardson-extrapolation error control.
//! - **Gauss-Legendre quadrature** ([`gauss_legendre`],
//!   [`gauss_legendre_composite`]) — fixed-order quadrature using classical
//!   Gauss-Legendre nodes and weights. Orders 1--10 use hardcoded tables;
//!   orders 11--20 are computed on-the-fly via the Golub-Welsch algorithm.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_numerical::integrate::{quad, gauss_legendre};
//!
//! // Adaptive Simpson: integrate x^2 from 0 to 1 = 1/3
//! let result = quad(|x| x * x, 0.0, 1.0, 1e-10);
//! assert!((result.value - 1.0 / 3.0).abs() < 1e-10);
//!
//! // 5-point Gauss-Legendre: exact for polynomials up to degree 9
//! let result = gauss_legendre(|x| x * x, 0.0, 1.0, 5).unwrap();
//! assert!((result.value - 1.0 / 3.0).abs() < 1e-14);
//! ```

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of numerical integration.
///
/// Contains the estimated integral value, an error estimate (when available),
/// and the total number of function evaluations performed.
#[derive(Debug, Clone)]
pub struct QuadratureResult {
    /// The estimated integral value.
    pub value: f64,
    /// Estimated absolute error (for adaptive methods; 0.0 for fixed-order
    /// Gauss-Legendre).
    pub error_estimate: f64,
    /// Number of function evaluations used.
    pub n_evals: usize,
}

// ===========================================================================
// Adaptive Simpson's rule
// ===========================================================================

/// Default maximum recursion depth for [`quad`].
const DEFAULT_MAX_DEPTH: usize = 50;

/// Adaptive Simpson quadrature.
///
/// Integrates `f(x)` from `a` to `b` using recursive adaptive Simpson's rule
/// with a default maximum recursion depth of 50.
///
/// The error is controlled by Richardson extrapolation: the recursion stops
/// on a sub-interval when the difference between the whole-interval Simpson
/// estimate and the sum of the two half-interval estimates satisfies
/// `|S_whole - S_left - S_right| < 15 * tol`.
///
/// # Arguments
///
/// * `f`   — The integrand.
/// * `a`   — Lower integration bound.
/// * `b`   — Upper integration bound.
/// * `tol` — Desired absolute tolerance.
///
/// # Returns
///
/// A [`QuadratureResult`] containing the estimated value, error estimate, and
/// total number of function evaluations.
#[must_use]
pub fn quad<F>(f: F, a: f64, b: f64, tol: f64) -> QuadratureResult
where
    F: Fn(f64) -> f64,
{
    quad_with_limit(f, a, b, tol, DEFAULT_MAX_DEPTH)
}

/// Adaptive Simpson quadrature with configurable maximum recursion depth.
///
/// Behaves identically to [`quad`] but allows the caller to control the
/// maximum recursion depth explicitly.
///
/// # Arguments
///
/// * `f`         — The integrand.
/// * `a`         — Lower integration bound.
/// * `b`         — Upper integration bound.
/// * `tol`       — Desired absolute tolerance.
/// * `max_depth` — Maximum number of recursive subdivisions.
///
/// # Returns
///
/// A [`QuadratureResult`] containing the estimated value, error estimate, and
/// total number of function evaluations.
#[must_use]
pub fn quad_with_limit<F>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> QuadratureResult
where
    F: Fn(f64) -> f64,
{
    let mut ctx = SimpsonContext::new(f);

    let fa = ctx.eval(a);
    let fb = ctx.eval(b);
    let m = (a + b) / 2.0;
    let fm = ctx.eval(m);

    let s_whole = simpson_from_values(a, b, fa, fm, fb);

    let iv = SimpsonInterval {
        a,
        b,
        fa,
        fm,
        fb,
        s: s_whole,
    };
    let (value, error_estimate) = ctx.adaptive_recurse(iv, tol, max_depth);

    QuadratureResult {
        value,
        error_estimate,
        n_evals: ctx.n_evals,
    }
}

/// Simpson's rule value given pre-evaluated function values.
///
/// `S = (b - a) / 6 * (f(a) + 4 * f(m) + f(b))`
fn simpson_from_values(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) / 6.0 * (fa + 4.0 * fm + fb)
}

/// A Simpson sub-interval with cached function values and estimate.
struct SimpsonInterval {
    /// Left endpoint.
    a: f64,
    /// Right endpoint.
    b: f64,
    /// f(a).
    fa: f64,
    /// f(midpoint).
    fm: f64,
    /// f(b).
    fb: f64,
    /// Simpson estimate on the whole interval.
    s: f64,
}

/// Internal state for adaptive Simpson recursion, bundling the integrand and
/// evaluation counter to keep the recursive signature compact.
struct SimpsonContext<F> {
    /// The integrand.
    f: F,
    /// Running count of function evaluations.
    n_evals: usize,
}

impl<F: Fn(f64) -> f64> SimpsonContext<F> {
    /// Create a new context wrapping the given integrand.
    fn new(f: F) -> Self {
        Self { f, n_evals: 0 }
    }

    /// Evaluate the integrand at `x`, incrementing the counter.
    fn eval(&mut self, x: f64) -> f64 {
        self.n_evals += 1;
        (self.f)(x)
    }

    /// Recursive core of adaptive Simpson's rule.
    ///
    /// Returns `(value, error_estimate)`.
    fn adaptive_recurse(&mut self, iv: SimpsonInterval, tol: f64, depth: usize) -> (f64, f64) {
        let m = (iv.a + iv.b) / 2.0;

        let f_m_left = self.eval((iv.a + m) / 2.0);
        let f_m_right = self.eval((m + iv.b) / 2.0);

        let s_left = simpson_from_values(iv.a, m, iv.fa, f_m_left, iv.fm);
        let s_right = simpson_from_values(m, iv.b, iv.fm, f_m_right, iv.fb);

        let error = (s_left + s_right - iv.s) / 15.0;

        if depth == 0 || error.abs() < tol {
            // Richardson extrapolation correction.
            let value = s_left + s_right + error;
            (value, error.abs())
        } else {
            let left = SimpsonInterval {
                a: iv.a,
                b: m,
                fa: iv.fa,
                fm: f_m_left,
                fb: iv.fm,
                s: s_left,
            };
            let right = SimpsonInterval {
                a: m,
                b: iv.b,
                fa: iv.fm,
                fm: f_m_right,
                fb: iv.fb,
                s: s_right,
            };
            // Recurse on each half with half the tolerance.
            let (v_left, e_left) = self.adaptive_recurse(left, tol / 2.0, depth - 1);
            let (v_right, e_right) = self.adaptive_recurse(right, tol / 2.0, depth - 1);
            (v_left + v_right, e_left + e_right)
        }
    }
}

// ===========================================================================
// Gauss-Legendre quadrature
// ===========================================================================

/// Fixed-order Gauss-Legendre quadrature.
///
/// Integrates `f(x)` from `a` to `b` using `n`-point Gauss-Legendre
/// quadrature on the whole interval.
///
/// Supported orders: 1 through 20. Orders 1--10 use hardcoded high-precision
/// tables; orders 11--20 are computed dynamically via the Golub-Welsch
/// algorithm.
///
/// An `n`-point rule is exact for polynomials of degree at most `2n - 1`.
///
/// # Arguments
///
/// * `f` — The integrand.
/// * `a` — Lower integration bound.
/// * `b` — Upper integration bound.
/// * `n` — Number of quadrature points (1..=20).
///
/// # Errors
///
/// Returns an error string if `n` is 0 or greater than 20.
pub fn gauss_legendre<F>(f: F, a: f64, b: f64, n: usize) -> Result<QuadratureResult, String>
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gl_nodes_weights(n)?;
    let result = gl_evaluate(&f, a, b, &nodes, &weights);
    Ok(result)
}

/// Composite Gauss-Legendre quadrature.
///
/// Divides the interval `[a, b]` into `n_panels` equal sub-intervals and
/// applies `n_points`-point Gauss-Legendre quadrature on each panel. This
/// greatly improves accuracy for functions that are not well-approximated by
/// a single high-order polynomial on the whole domain.
///
/// # Arguments
///
/// * `f`        — The integrand.
/// * `a`        — Lower integration bound.
/// * `b`        — Upper integration bound.
/// * `n_points` — Number of GL quadrature points per panel (1..=20).
/// * `n_panels` — Number of equal sub-intervals.
///
/// # Errors
///
/// Returns an error string if `n_points` is 0 or greater than 20, or if
/// `n_panels` is 0.
pub fn gauss_legendre_composite<F>(
    f: F,
    a: f64,
    b: f64,
    n_points: usize,
    n_panels: usize,
) -> Result<QuadratureResult, String>
where
    F: Fn(f64) -> f64,
{
    if n_panels == 0 {
        return Err("n_panels must be at least 1".to_string());
    }

    let (nodes, weights) = gl_nodes_weights(n_points)?;

    let panel_width = (b - a) / n_panels as f64;
    let mut total_value = 0.0;
    let mut total_evals = 0usize;

    for i in 0..n_panels {
        let panel_a = a + i as f64 * panel_width;
        let panel_b = panel_a + panel_width;
        let result = gl_evaluate(&f, panel_a, panel_b, &nodes, &weights);
        total_value += result.value;
        total_evals += result.n_evals;
    }

    Ok(QuadratureResult {
        value: total_value,
        error_estimate: 0.0,
        n_evals: total_evals,
    })
}

/// Evaluate an `n`-point GL rule on `[a, b]` given reference nodes/weights
/// on `[-1, 1]`.
fn gl_evaluate<F>(f: &F, a: f64, b: f64, nodes: &[f64], weights: &[f64]) -> QuadratureResult
where
    F: Fn(f64) -> f64,
{
    let half_len = (b - a) / 2.0;
    let mid = (a + b) / 2.0;

    let mut value = 0.0;
    for (t, w) in nodes.iter().zip(weights.iter()) {
        let x = half_len * t + mid;
        value += w * f(x);
    }
    value *= half_len;

    QuadratureResult {
        value,
        error_estimate: 0.0,
        n_evals: nodes.len(),
    }
}

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes and weights
// ---------------------------------------------------------------------------

/// Return the `n`-point Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Orders 1--10 use hardcoded tables; orders 11--20 are computed via the
/// Golub-Welsch algorithm.
fn gl_nodes_weights(n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n == 0 || n > 20 {
        return Err(format!("Gauss-Legendre order must be in 1..=20, got {n}"));
    }

    if n <= 10 {
        let (nodes, weights) = gl_table(n);
        Ok((nodes.to_vec(), weights.to_vec()))
    } else {
        Ok(golub_welsch(n))
    }
}

/// Hardcoded Gauss-Legendre tables for orders 1 through 10 on `[-1, 1]`.
///
/// Values are from Abramowitz & Stegun / DLMF, carried to full f64 precision.
#[allow(clippy::excessive_precision)]
fn gl_table(n: usize) -> (&'static [f64], &'static [f64]) {
    match n {
        1 => (&[0.0], &[2.0]),
        2 => (
            &[-0.5773502691896257645091736, 0.5773502691896257645091736],
            &[1.0, 1.0],
        ),
        3 => (
            &[
                -0.7745966692414833770358531,
                0.0,
                0.7745966692414833770358531,
            ],
            &[
                0.5555555555555555555555556,
                0.8888888888888888888888889,
                0.5555555555555555555555556,
            ],
        ),
        4 => (
            &[
                -0.8611363115940525752239465,
                -0.3399810435848562648026658,
                0.3399810435848562648026658,
                0.8611363115940525752239465,
            ],
            &[
                0.3478548451374538573730639,
                0.6521451548625461426269361,
                0.6521451548625461426269361,
                0.3478548451374538573730639,
            ],
        ),
        5 => (
            &[
                -0.9061798459386639927976269,
                -0.5384693101056830910363144,
                0.0,
                0.5384693101056830910363144,
                0.9061798459386639927976269,
            ],
            &[
                0.2369268850561890875142640,
                0.4786286704993664680412915,
                0.5688888888888888888888889,
                0.4786286704993664680412915,
                0.2369268850561890875142640,
            ],
        ),
        6 => (
            &[
                -0.9324695142031520278123016,
                -0.6612093864662645136613996,
                -0.2386191860831969086305017,
                0.2386191860831969086305017,
                0.6612093864662645136613996,
                0.9324695142031520278123016,
            ],
            &[
                0.1713244923791703450402961,
                0.3607615730481386075698335,
                0.4679139345726910473898703,
                0.4679139345726910473898703,
                0.3607615730481386075698335,
                0.1713244923791703450402961,
            ],
        ),
        7 => (
            &[
                -0.9491079123427585245261897,
                -0.7415311855993944398638648,
                -0.4058451513773971669066064,
                0.0,
                0.4058451513773971669066064,
                0.7415311855993944398638648,
                0.9491079123427585245261897,
            ],
            &[
                0.1294849661688696932706114,
                0.2797053914892766679014678,
                0.3818300505051189449503698,
                0.4179591836734693877551020,
                0.3818300505051189449503698,
                0.2797053914892766679014678,
                0.1294849661688696932706114,
            ],
        ),
        8 => (
            &[
                -0.9602898564975362316835609,
                -0.7966664774136267395915539,
                -0.5255324099163289858177390,
                -0.1834346424956498049394761,
                0.1834346424956498049394761,
                0.5255324099163289858177390,
                0.7966664774136267395915539,
                0.9602898564975362316835609,
            ],
            &[
                0.1012285362903762591525314,
                0.2223810344533744705443560,
                0.3137066458778872873379622,
                0.3626837833783619829651504,
                0.3626837833783619829651504,
                0.3137066458778872873379622,
                0.2223810344533744705443560,
                0.1012285362903762591525314,
            ],
        ),
        9 => (
            &[
                -0.9681602395076260898355762,
                -0.8360311073266357942994298,
                -0.6133714327005903973087020,
                -0.3242534234038089290385380,
                0.0,
                0.3242534234038089290385380,
                0.6133714327005903973087020,
                0.8360311073266357942994298,
                0.9681602395076260898355762,
            ],
            &[
                0.0812743883615744119718922,
                0.1806481606948574040584720,
                0.2606106964029354623187429,
                0.3123470770400028400686304,
                0.3302393550012597631645251,
                0.3123470770400028400686304,
                0.2606106964029354623187429,
                0.1806481606948574040584720,
                0.0812743883615744119718922,
            ],
        ),
        10 => (
            &[
                -0.9739065285171717200779640,
                -0.8650633666889845107320967,
                -0.6794095682990244062343274,
                -0.4333953941292471907992659,
                -0.1488743389816312108848260,
                0.1488743389816312108848260,
                0.4333953941292471907992659,
                0.6794095682990244062343274,
                0.8650633666889845107320967,
                0.9739065285171717200779640,
            ],
            &[
                0.0666713443086881375935688,
                0.1494513491505805931457763,
                0.2190863625159820439955349,
                0.2692667193099963550912269,
                0.2955242247147528701738930,
                0.2955242247147528701738930,
                0.2692667193099963550912269,
                0.2190863625159820439955349,
                0.1494513491505805931457763,
                0.0666713443086881375935688,
            ],
        ),
        _ => unreachable!("gl_table called with n={n}, expected 1..=10"),
    }
}

// ---------------------------------------------------------------------------
// Golub-Welsch algorithm
// ---------------------------------------------------------------------------

/// Compute Gauss-Legendre nodes and weights for order `n` using the
/// Golub-Welsch algorithm.
///
/// The `n`-point Gauss-Legendre nodes are eigenvalues of the `n x n`
/// symmetric tridiagonal Jacobi matrix for the Legendre polynomials:
///
///   - Diagonal entries: `alpha_i = 0` for all `i`.
///   - Off-diagonal entries: `beta_i = i / sqrt(4 * i^2 - 1)`.
///
/// The weights are `w_i = 2 * v_i[0]^2` where `v_i` is the normalised
/// eigenvector corresponding to the `i`-th eigenvalue.
///
/// We use the implicit-shift QR algorithm for the symmetric tridiagonal
/// eigenproblem.
fn golub_welsch(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Build the symmetric tridiagonal Jacobi matrix for Legendre polynomials.
    // Diagonal: all zeros.
    // Sub-diagonal: beta_i = i / sqrt(4 i^2 - 1) for i = 1, ..., n-1.
    let mut diag = vec![0.0_f64; n];
    let mut sub = vec![0.0_f64; n.saturating_sub(1)];

    for i in 1..n {
        let fi = i as f64;
        sub[i - 1] = fi / (4.0 * fi * fi - 1.0).sqrt();
    }

    // Eigenvector matrix Q (row-major n x n), initialised to identity.
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    // Compute eigenvalues and eigenvectors via implicit-shift QR.
    trid_qr_eigen(&mut diag, &mut sub, &mut q, n);

    // Eigenvalues are in `diag`; the first component of eigenvector i is
    // q[0 * n + i] = q[i].  Weights: w_i = 2 * q[i]^2.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap());

    let nodes: Vec<f64> = indices.iter().map(|&i| diag[i]).collect();
    let weights: Vec<f64> = indices.iter().map(|&i| 2.0 * q[i] * q[i]).collect();

    (nodes, weights)
}

/// Implicit-shift QR iteration for a symmetric tridiagonal matrix.
///
/// On entry, `diag[0..n]` holds the diagonal and `sub[0..n-1]` holds the
/// sub-diagonal. On exit, `diag` contains the eigenvalues and `q` (row-major
/// `n x n`) contains the eigenvectors as columns.
fn trid_qr_eigen(diag: &mut [f64], sub: &mut [f64], q: &mut [f64], n: usize) {
    if n <= 1 {
        return;
    }

    let max_iter = 30 * n;

    for _ in 0..max_iter {
        // Find the largest unreduced block by scanning from the bottom.
        // An off-diagonal element sub[i] is "zero" if
        // |sub[i]| <= eps * (|diag[i]| + |diag[i+1]|).
        let mut bot = n - 1;
        while bot > 0 {
            if sub[bot - 1].abs() <= 1e-14 * (diag[bot - 1].abs() + diag[bot].abs()).max(1e-300) {
                sub[bot - 1] = 0.0;
                bot -= 1;
            } else {
                break;
            }
        }

        if bot == 0 {
            // All off-diagonals are zero — fully deflated.
            break;
        }

        // Find the top of the unreduced block ending at `bot`.
        let mut top = bot - 1;
        while top > 0 {
            if sub[top - 1].abs() <= 1e-14 * (diag[top - 1].abs() + diag[top].abs()).max(1e-300) {
                sub[top - 1] = 0.0;
                break;
            }
            top -= 1;
        }

        // Apply one implicit QR step with Wilkinson shift on [top, bot].
        let shift = wilkinson_shift(diag[bot - 1], sub[bot - 1], diag[bot]);
        implicit_qr_step(diag, sub, q, n, top, bot, shift);
    }
}

/// Wilkinson shift: the eigenvalue of the trailing 2x2 block
///
///   | a  b |
///   | b  d |
///
/// that is closer to `d`.
fn wilkinson_shift(a: f64, b: f64, d: f64) -> f64 {
    let delta = (a - d) / 2.0;
    if delta.abs() < 1e-300 {
        d - b.abs()
    } else {
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        d - b * b / (delta + sign * (delta * delta + b * b).sqrt())
    }
}

/// One implicit QR step with shift on the block `[top, bot]` (inclusive) of
/// the tridiagonal matrix.
///
/// This implements the bulge-chase variant: a Givens rotation is applied at
/// position `top` to introduce a "bulge" which is then chased down the
/// diagonal via successive Givens rotations.
fn implicit_qr_step(
    diag: &mut [f64],
    sub: &mut [f64],
    q: &mut [f64],
    n: usize,
    top: usize,
    bot: usize,
    shift: f64,
) {
    // Initial bulge.
    let mut x = diag[top] - shift;
    let mut z = sub[top];

    for k in top..bot {
        // Compute Givens rotation to zero out z.
        let r = x.hypot(z);
        let c = if r.abs() < 1e-300 { 1.0 } else { x / r };
        let s = if r.abs() < 1e-300 { 0.0 } else { z / r };

        // Update sub-diagonal above the rotation (if not the first step).
        if k > top {
            sub[k - 1] = r;
        }

        // Apply the similarity transform to the 2x2 block at (k, k+1).
        //
        // The tridiagonal elements involved:
        //   diag[k], diag[k+1], sub[k]
        // and possibly sub[k+1] (which creates the next bulge).
        let d0 = diag[k];
        let d1 = diag[k + 1];
        let e = sub[k];

        // After applying G(c,s) from both sides:
        diag[k] = c * c * d0 + 2.0 * c * s * e + s * s * d1;
        diag[k + 1] = s * s * d0 - 2.0 * c * s * e + c * c * d1;
        sub[k] = c * s * (d1 - d0) + (c * c - s * s) * e;

        // Chase the bulge: the rotation may have created a nonzero element
        // at position sub[k+1] (coupling k+1 and k+2).
        if k + 1 < bot {
            let bulge = s * sub[k + 1];
            sub[k + 1] *= c;
            x = sub[k];
            z = bulge;
        }

        // Accumulate the Givens rotation into the eigenvector matrix Q.
        // For each row i: (Q[i,k], Q[i,k+1]) <- rotation.
        for i in 0..n {
            let qk = q[i * n + k];
            let qk1 = q[i * n + k + 1];
            q[i * n + k] = c * qk + s * qk1;
            q[i * n + k + 1] = -s * qk + c * qk1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn quad_constant() {
        let result = quad(|_| 5.0, 0.0, 1.0, 1e-10);
        assert_abs_diff_eq!(result.value, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn quad_linear() {
        let result = quad(|x| x, 0.0, 1.0, 1e-10);
        assert_abs_diff_eq!(result.value, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn quad_polynomial() {
        let result = quad(|x| x.powi(4), 0.0, 1.0, 1e-10);
        assert_abs_diff_eq!(result.value, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn quad_sin() {
        let result = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-10);
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn quad_gaussian() {
        // integral of exp(-x^2) from -5 to 5 should approximate sqrt(pi).
        let result = quad(|x| (-x * x).exp(), -5.0, 5.0, 1e-10);
        assert_abs_diff_eq!(result.value, std::f64::consts::PI.sqrt(), epsilon = 1e-8);
    }

    #[test]
    fn gauss_legendre_exact_polynomial() {
        // n-point GL is exact for polynomials of degree <= 2n-1.

        // n=5 should be exact for x^9 (odd, integral = 0).
        let result = gauss_legendre(|x| x.powi(9), -1.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result.value, 0.0, epsilon = 1e-13);

        // n=5 should be exact for x^8: integral from -1 to 1 = 2/9.
        let result = gauss_legendre(|x| x.powi(8), -1.0, 1.0, 5).unwrap();
        assert_abs_diff_eq!(result.value, 2.0 / 9.0, epsilon = 1e-13);

        // n=3 exact for degree 5: integral of x^5 from 0 to 1 = 1/6.
        let result = gauss_legendre(|x| x.powi(5), 0.0, 1.0, 3).unwrap();
        assert_abs_diff_eq!(result.value, 1.0 / 6.0, epsilon = 1e-13);
    }

    #[test]
    fn gauss_legendre_sin() {
        let result = gauss_legendre(|x| x.sin(), 0.0, std::f64::consts::PI, 10).unwrap();
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn composite_gl_accuracy() {
        // Composite GL with 10 panels of 5-point GL should be very accurate
        // for smooth functions. We integrate sin(x) from 0 to pi (exact = 2)
        // which avoids the truncation issue of infinite-domain integrands.
        let result =
            gauss_legendre_composite(|x| x.sin(), 0.0, std::f64::consts::PI, 5, 10).unwrap();
        assert_abs_diff_eq!(result.value, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn quad_tight_tolerance() {
        // Tighter tolerance should give a more accurate result.
        let loose = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-4);
        let tight = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-12);

        let exact = 2.0;
        let err_loose = (loose.value - exact).abs();
        let err_tight = (tight.value - exact).abs();

        assert!(
            err_tight <= err_loose,
            "tight tolerance error ({err_tight}) should be <= loose error ({err_loose})"
        );
        assert_abs_diff_eq!(tight.value, exact, epsilon = 1e-12);
    }

    // Additional tests for Golub-Welsch computed orders (n > 10).
    #[test]
    fn gauss_legendre_order_15() {
        // 15-point GL is exact for degree <= 29.
        // Test: integral of x^28 from -1 to 1 = 2/29.
        let result = gauss_legendre(|x| x.powi(28), -1.0, 1.0, 15).unwrap();
        assert_abs_diff_eq!(result.value, 2.0 / 29.0, epsilon = 1e-10);
    }

    #[test]
    fn gauss_legendre_order_20() {
        // 20-point GL is exact for degree <= 39.
        // Test: integral of x^38 from -1 to 1 = 2/39.
        let result = gauss_legendre(|x| x.powi(38), -1.0, 1.0, 20).unwrap();
        assert_abs_diff_eq!(result.value, 2.0 / 39.0, epsilon = 1e-8);
    }

    #[test]
    fn gauss_legendre_invalid_order() {
        let result = gauss_legendre(|x| x, 0.0, 1.0, 0);
        assert!(result.is_err());

        let result = gauss_legendre(|x| x, 0.0, 1.0, 21);
        assert!(result.is_err());
    }

    #[test]
    fn composite_gl_zero_panels_error() {
        let result = gauss_legendre_composite(|x| x, 0.0, 1.0, 5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn quad_counts_evals() {
        // Ensure n_evals is tracked and positive.
        let result = quad(|x| x * x, 0.0, 1.0, 1e-6);
        assert!(result.n_evals > 0);
    }

    #[test]
    fn golub_welsch_weights_sum_to_two() {
        // For any order n, GL weights on [-1, 1] should sum to 2
        // (since integral of 1 from -1 to 1 = 2).
        for n in 11..=20 {
            let (_, weights) = golub_welsch(n);
            let sum: f64 = weights.iter().sum();
            assert_abs_diff_eq!(sum, 2.0, epsilon = 1e-12);
        }
    }
}
