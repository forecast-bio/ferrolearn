//! Cubic spline interpolation.
//!
//! This module provides [`CubicSpline`], a piecewise cubic polynomial
//! interpolant equivalent to
//! [`scipy.interpolate.CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html).
//!
//! Two boundary conditions are supported via [`BoundaryCondition`]:
//!
//! - **Natural** — second derivatives are zero at the endpoints.
//! - **Not-a-knot** — third derivative is continuous at the second and
//!   second-to-last knots, giving exact reproduction of cubic polynomials.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_numerical::interpolate::{CubicSpline, BoundaryCondition};
//!
//! let x = [0.0, 1.0, 2.0, 3.0];
//! let y = [0.0, 1.0, 4.0, 9.0]; // roughly y = x^2
//!
//! let spline = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();
//! let val = spline.eval(1.5);
//! assert!((val - 2.25).abs() < 0.5); // close to 1.5^2
//! ```

use ndarray::Array1;

/// Boundary condition for the cubic spline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Natural spline: S''(x_0) = S''(x_n) = 0.
    Natural,
    /// Not-a-knot: third derivative continuous at x_1 and x_{n-1}.
    NotAKnot,
}

/// A cubic spline interpolant.
///
/// Given data points (x_i, y_i), constructs a piecewise cubic polynomial
/// S(x) such that S(x_i) = y_i and S is twice continuously differentiable.
///
/// On each interval \[x_i, x_{i+1}\] the spline is:
///
/// ```text
/// S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
/// ```
///
/// where a_i = y_i.
///
/// For evaluation outside the data range the polynomial from the first or
/// last interval is used (extrapolation).
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// Knot x-coordinates (sorted, length n+1).
    knots: Vec<f64>,
    /// Coefficients a_i = y_i for each interval (length n).
    a: Vec<f64>,
    /// First-derivative coefficients for each interval (length n).
    b: Vec<f64>,
    /// Second-derivative / 2 coefficients for each interval (length n).
    c: Vec<f64>,
    /// Third-derivative / 6 coefficients for each interval (length n).
    d: Vec<f64>,
}

impl CubicSpline {
    /// Construct a cubic spline through the given data points.
    ///
    /// # Arguments
    ///
    /// * `x` — Knot x-coordinates. Must be strictly increasing.
    /// * `y` — Knot y-coordinates. Must have the same length as `x`.
    /// * `bc` — The [`BoundaryCondition`] to apply at the endpoints.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `x` and `y` have different lengths.
    /// - Fewer than 2 data points are provided.
    /// - `x` is not strictly increasing.
    pub fn new(x: &[f64], y: &[f64], bc: BoundaryCondition) -> Result<Self, String> {
        let n_points = x.len();

        if n_points != y.len() {
            return Err(format!(
                "x and y must have the same length, got {} and {}",
                n_points,
                y.len()
            ));
        }
        if n_points < 2 {
            return Err(format!(
                "at least 2 data points are required, got {n_points}"
            ));
        }
        for i in 1..n_points {
            if x[i] <= x[i - 1] {
                return Err(format!(
                    "x must be strictly increasing, but x[{}] = {} <= x[{}] = {}",
                    i,
                    x[i],
                    i - 1,
                    x[i - 1]
                ));
            }
        }

        let n = n_points - 1; // number of intervals

        // Interval widths.
        let h: Vec<f64> = (0..n).map(|i| x[i + 1] - x[i]).collect();

        // For only 2 data points the spline degenerates to a line.
        if n == 1 {
            let a0 = y[0];
            let b0 = (y[1] - y[0]) / h[0];
            // Both natural and not-a-knot produce c=0, d=0 for a single interval.
            return Ok(Self {
                knots: x.to_vec(),
                a: vec![a0],
                b: vec![b0],
                c: vec![0.0],
                d: vec![0.0],
            });
        }

        // Solve for c_i (c values at each knot, length n+1).
        // The interior equations (for i = 1..n-1) are:
        //   h_{i-1} * c_{i-1} + 2*(h_{i-1}+h_i)*c_i + h_i*c_{i+1}
        //     = 3*((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})
        let c_knots = match bc {
            BoundaryCondition::Natural => solve_natural(n, &h, y),
            BoundaryCondition::NotAKnot => solve_not_a_knot(n, &h, y),
        };

        // Derive per-interval coefficients.
        let mut a_coef = Vec::with_capacity(n);
        let mut b_coef = Vec::with_capacity(n);
        let mut c_coef = Vec::with_capacity(n);
        let mut d_coef = Vec::with_capacity(n);

        for i in 0..n {
            let ai = y[i];
            let ci = c_knots[i];
            let ci1 = c_knots[i + 1];
            let di = (ci1 - ci) / (3.0 * h[i]);
            let bi = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * ci + ci1) / 3.0;

            a_coef.push(ai);
            b_coef.push(bi);
            c_coef.push(ci);
            d_coef.push(di);
        }

        Ok(Self {
            knots: x.to_vec(),
            a: a_coef,
            b: b_coef,
            c: c_coef,
            d: d_coef,
        })
    }

    /// Find the interval index for a given x value.
    ///
    /// Returns the index `i` such that `knots[i] <= x < knots[i+1]`,
    /// clamped to `[0, n-1]` for extrapolation.
    fn find_interval(&self, x: f64) -> usize {
        let n = self.knots.len() - 1;
        if x <= self.knots[0] {
            return 0;
        }
        if x >= self.knots[n] {
            return n - 1;
        }
        // Binary search: find the rightmost knot <= x.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.knots[mid] <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is now the first index where knots[lo] > x, so the interval is lo-1.
        lo - 1
    }

    /// Evaluate the spline at a single point.
    ///
    /// For `x` outside the data range, the polynomial of the first or last
    /// interval is used (extrapolation).
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let i = self.find_interval(x);
        let t = x - self.knots[i];
        self.a[i] + t * (self.b[i] + t * (self.c[i] + t * self.d[i]))
    }

    /// Evaluate the spline at multiple points.
    ///
    /// Returns an [`Array1<f64>`] of the same length as the input slice.
    #[must_use]
    pub fn eval_array(&self, x: &[f64]) -> Array1<f64> {
        Array1::from_iter(x.iter().map(|&xi| self.eval(xi)))
    }

    /// Evaluate the first derivative of the spline at a single point.
    ///
    /// ```text
    /// S'_i(x) = b_i + 2*c_i*(x - x_i) + 3*d_i*(x - x_i)^2
    /// ```
    #[must_use]
    pub fn derivative(&self, x: f64) -> f64 {
        let i = self.find_interval(x);
        let t = x - self.knots[i];
        self.b[i] + t * (2.0 * self.c[i] + t * 3.0 * self.d[i])
    }

    /// Evaluate the second derivative of the spline at a single point.
    ///
    /// ```text
    /// S''_i(x) = 2*c_i + 6*d_i*(x - x_i)
    /// ```
    #[must_use]
    pub fn second_derivative(&self, x: f64) -> f64 {
        let i = self.find_interval(x);
        let t = x - self.knots[i];
        2.0 * self.c[i] + 6.0 * self.d[i] * t
    }

    /// Evaluate the definite integral of the spline from `a` to `b`.
    ///
    /// The integral is computed analytically by summing the integrals of
    /// each cubic piece that falls within \[`a`, `b`\]. If `a > b` the
    /// result is negated.
    #[must_use]
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if (a - b).abs() < f64::EPSILON {
            return 0.0;
        }
        if a > b {
            return -self.integrate(b, a);
        }

        let i_start = self.find_interval(a);
        let i_end = self.find_interval(b);

        let mut total = 0.0;

        for i in i_start..=i_end {
            // Determine the local limits within interval i.
            let lo = if i == i_start { a } else { self.knots[i] };
            let hi = if i == i_end { b } else { self.knots[i + 1] };

            let t_lo = lo - self.knots[i];
            let t_hi = hi - self.knots[i];

            total += self.antiderivative_at(i, t_hi) - self.antiderivative_at(i, t_lo);
        }

        total
    }

    /// Evaluate the antiderivative of interval `i` at local coordinate `t`.
    ///
    /// ```text
    /// F_i(t) = a_i*t + b_i*t^2/2 + c_i*t^3/3 + d_i*t^4/4
    /// ```
    fn antiderivative_at(&self, i: usize, t: f64) -> f64 {
        let a = self.a[i];
        let b = self.b[i];
        let c = self.c[i];
        let d = self.d[i];
        t * (a + t * (b / 2.0 + t * (c / 3.0 + t * d / 4.0)))
    }
}

/// Solve the tridiagonal system for natural boundary conditions.
///
/// Natural BC: c_0 = 0, c_n = 0.
/// Interior equations (i = 1..n-1):
///   h_{i-1}*c_{i-1} + 2*(h_{i-1}+h_i)*c_i + h_i*c_{i+1} = rhs_i
///
/// This reduces to an (n-1) x (n-1) tridiagonal system for c_1..c_{n-1},
/// solved with the Thomas algorithm in O(n) time.
fn solve_natural(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n + 1];
    }

    let m = n - 1; // number of interior unknowns

    // Build the tridiagonal system.
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m.saturating_sub(1)];
    let mut lower = vec![0.0; m.saturating_sub(1)];
    let mut rhs = vec![0.0; m];

    for j in 0..m {
        let i = j + 1; // knot index
        diag[j] = 2.0 * (h[i - 1] + h[i]);
        rhs[j] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }
    for j in 0..m.saturating_sub(1) {
        let i = j + 1;
        upper[j] = h[i];
        lower[j] = h[i];
    }

    let solution = thomas_solve(&mut diag, &mut upper, &mut lower, &mut rhs);

    // Assemble full c vector: [0, c_1, ..., c_{n-1}, 0].
    let mut c = vec![0.0; n + 1];
    c[1..=m].copy_from_slice(&solution[..m]);
    c
}

/// Solve the system for not-a-knot boundary conditions.
///
/// Not-a-knot BC requires d_0 = d_1 and d_{n-2} = d_{n-1}, i.e. the third
/// derivative is continuous at the second and second-to-last knots.
///
/// Since d_i = (c_{i+1} - c_i) / (3*h_i), the conditions translate to:
///
/// ```text
/// (c_1 - c_0) / h_0 = (c_2 - c_1) / h_1
/// (c_{n-1} - c_{n-2}) / h_{n-2} = (c_n - c_{n-1}) / h_{n-1}
/// ```
///
/// Combined with the n-1 interior equations this yields an (n+1) x (n+1)
/// system solved by a modified Thomas algorithm.
fn solve_not_a_knot(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n + 1];
    }
    solve_not_a_knot_general(n, h, y)
}

/// General not-a-knot solver for n >= 2.
///
/// Builds an (n+1) x (n+1) system:
///   - Row 0: not-a-knot at the left end
///   - Rows 1..n-1: interior continuity equations
///   - Row n: not-a-knot at the right end
///
/// The system is almost tridiagonal — rows 0 and n each touch three
/// consecutive unknowns that extend one position beyond the standard
/// tridiagonal band. A modified Gaussian elimination handles the extra
/// entries in O(n) time.
fn solve_not_a_knot_general(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let size = n + 1;

    let mut sub = vec![0.0; size];
    let mut diag = vec![0.0; size];
    let mut sup = vec![0.0; size];
    let mut rhs = vec![0.0; size];

    // Row 0: not-a-knot at left end.
    // -h_1*c_0 + (h_0+h_1)*c_1 - h_0*c_2 = 0
    diag[0] = -h[1];
    sup[0] = h[0] + h[1];
    let extra_sup_0 = -h[0]; // coefficient of c_2 in row 0
    rhs[0] = 0.0;

    // Interior rows (i = 1..n-1).
    for i in 1..n {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Row n: not-a-knot at right end.
    // -h_{n-1}*c_{n-2} + (h_{n-2}+h_{n-1})*c_{n-1} - h_{n-2}*c_n = 0
    let extra_sub_n = -h[n - 1]; // coefficient of c_{n-2} in row n
    sub[n] = h[n - 2] + h[n - 1];
    diag[n] = -h[n - 2];
    rhs[n] = 0.0;

    // --- Forward elimination (modified Thomas algorithm) ---

    // Row 0 → Row 1: eliminate sub[1] using diag[0].
    // Row 0 has an extra entry (extra_sup_0) at column 2 that propagates
    // into row 1's super-diagonal entry.
    {
        let factor = sub[1] / diag[0];
        sub[1] = 0.0;
        diag[1] -= factor * sup[0];
        sup[1] -= factor * extra_sup_0;
        rhs[1] -= factor * rhs[0];
    }

    // Rows 2..n-1: standard Thomas forward sweep.
    for i in 2..n {
        let factor = sub[i] / diag[i - 1];
        sub[i] = 0.0;
        diag[i] -= factor * sup[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Row n: first eliminate the extra sub-diagonal entry at column n-2.
    if n >= 2 {
        let factor = extra_sub_n / diag[n - 2];
        // This elimination introduces a contribution at column n-1
        // (from sup[n-2]) into sub[n].
        sub[n] -= factor * sup[n - 2];
        rhs[n] -= factor * rhs[n - 2];
    }

    // Row n: now eliminate sub[n] using row n-1.
    {
        let factor = sub[n] / diag[n - 1];
        sub[n] = 0.0;
        diag[n] -= factor * sup[n - 1];
        rhs[n] -= factor * rhs[n - 1];
    }

    // --- Back-substitution ---
    let mut c = vec![0.0; size];
    c[n] = rhs[n] / diag[n];

    for i in (1..n).rev() {
        c[i] = (rhs[i] - sup[i] * c[i + 1]) / diag[i];
    }

    // Row 0 still has the original three-column structure (diag, sup, extra_sup_0),
    // so we substitute c[1] and c[2] back explicitly.
    c[0] = (rhs[0] - sup[0] * c[1] - extra_sup_0 * c[2]) / diag[0];

    c
}

/// Solve a tridiagonal system using the Thomas algorithm.
///
/// The system has `n` equations. `diag` and `rhs` have length `n`.
/// `upper` and `lower` have length `n-1`.
///
/// Modifies the input arrays in place and returns the solution vector.
fn thomas_solve(
    diag: &mut [f64],
    upper: &mut [f64],
    lower: &mut [f64],
    rhs: &mut [f64],
) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    // Forward elimination.
    for i in 1..n {
        let factor = lower[i - 1] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Back-substitution.
    let mut x = vec![0.0; n];
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn interpolates_data_points() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 0.0, 1.0, 0.0];

        for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
            let spline = CubicSpline::new(&x, &y, bc).unwrap();
            for (xi, yi) in x.iter().zip(y.iter()) {
                assert_abs_diff_eq!(spline.eval(*xi), *yi, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn linear_data_exact() {
        // y = 2x + 1 — any cubic spline should reproduce a linear function.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
            let spline = CubicSpline::new(&x, &y, bc).unwrap();
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
                assert_abs_diff_eq!(spline.eval(t), 2.0 * t + 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn quadratic_natural() {
        // y = x^2 — natural spline forces S''(endpoints)=0, but the true
        // second derivative is 2, so it won't be exact near the boundaries.
        // In the interior it should still be close.
        let x: Vec<f64> = (0..=10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let spline = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();
        for t in [2.5, 3.7, 5.0, 7.3] {
            assert_abs_diff_eq!(spline.eval(t), t * t, epsilon = 1e-2);
        }
    }

    #[test]
    fn not_a_knot_cubic_exact() {
        // y = x^3 — not-a-knot should reproduce a cubic polynomial exactly.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();

        let spline = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            assert_abs_diff_eq!(spline.eval(t), t * t * t, epsilon = 1e-10);
        }
    }

    #[test]
    fn derivative_of_linear() {
        // y = 3x — derivative should be 3 everywhere.
        let x = [0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi).collect();

        for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
            let spline = CubicSpline::new(&x, &y, bc).unwrap();
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
                assert_abs_diff_eq!(spline.derivative(t), 3.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn integration_linear() {
        // y = x, integral from 0 to 1 should be 0.5.
        let x = [0.0, 0.5, 1.0];
        let y = [0.0, 0.5, 1.0];

        for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
            let spline = CubicSpline::new(&x, &y, bc).unwrap();
            assert_abs_diff_eq!(spline.integrate(0.0, 1.0), 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn integration_quadratic() {
        // y = x^2, integral from 0 to 1 should be 1/3.
        let x: Vec<f64> = (0..=10).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let spline = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
        assert_abs_diff_eq!(spline.integrate(0.0, 1.0), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn monotone_data() {
        // Monotonically increasing data — spline should not oscillate wildly.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0];

        let spline = CubicSpline::new(&x, &y, BoundaryCondition::Natural).unwrap();

        // Check that intermediate values stay within a reasonable range.
        for i in 0..x.len() - 1 {
            let mid = f64::midpoint(x[i], x[i + 1]);
            let val = spline.eval(mid);
            let lo = y[i].min(y[i + 1]) - 1.0;
            let hi = y[i].max(y[i + 1]) + 1.0;
            assert!(
                val >= lo && val <= hi,
                "spline({mid}) = {val}, expected between {lo} and {hi}"
            );
        }
    }

    #[test]
    fn two_points_minimum() {
        // Exactly 2 data points — should produce a linear interpolant.
        let x = [1.0, 3.0];
        let y = [2.0, 6.0];

        for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
            let spline = CubicSpline::new(&x, &y, bc).unwrap();
            assert_abs_diff_eq!(spline.eval(1.0), 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(spline.eval(2.0), 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(spline.eval(3.0), 6.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn unsorted_x_error() {
        let x = [1.0, 3.0, 2.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];

        let result = CubicSpline::new(&x, &y, BoundaryCondition::Natural);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("strictly increasing"));
    }
}
