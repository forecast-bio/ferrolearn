//! Unconstrained optimization algorithms for smooth objectives.
//!
//! This module provides scipy.optimize equivalents for the ferrolearn ML
//! framework. Two optimizers are available:
//!
//! - **[`NewtonCG`]** — Truncated Newton with conjugate-gradient inner loop
//!   and backtracking line search. Good for large-scale smooth problems when
//!   Hessian-vector products are cheap.
//! - **[`TrustRegionNCG`]** — Trust-region Newton-CG using the Steihaug-Toint
//!   CG subproblem solver. More robust than line-search Newton-CG, especially
//!   near saddle points or in ill-conditioned regions.
//!
//! Both optimizers require a closure that returns the objective value and
//! gradient, and a second closure that computes Hessian-vector products.
//!
//! # Example
//!
//! ```
//! use ndarray::{array, Array1};
//! use ferrolearn_numerical::optimize::{NewtonCG, TrustRegionNCG};
//!
//! // Minimize f(x) = 0.5 * (x0^2 + 2*x1^2)
//! let fun_grad = |x: &Array1<f64>| {
//!     let f = 0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1]);
//!     let g = array![x[0], 2.0 * x[1]];
//!     (f, g)
//! };
//! let hessp = |_x: &Array1<f64>, p: &Array1<f64>| {
//!     array![p[0], 2.0 * p[1]]
//! };
//!
//! let result = NewtonCG::new()
//!     .minimize(fun_grad, hessp, array![5.0, 3.0])
//!     .unwrap();
//! assert!(result.converged);
//! ```

use ndarray::Array1;

/// Result of an optimization run.
///
/// Contains the solution vector, objective value, gradient, iteration count,
/// and a flag indicating whether the optimizer converged to the requested
/// tolerance.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The solution vector.
    pub x: Array1<f64>,
    /// The objective function value at the solution.
    pub fun: f64,
    /// The gradient at the solution.
    pub grad: Array1<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Newton-CG (Truncated Newton with Conjugate Gradient)
// ---------------------------------------------------------------------------

/// Newton-CG optimizer with backtracking line search.
///
/// Uses conjugate gradient to approximately solve the Newton system
/// `H d = -g` at each step, followed by a backtracking line search with the
/// Armijo sufficient-decrease condition. The CG inner loop terminates early
/// when negative curvature is encountered, ensuring a descent direction.
///
/// # Builder
///
/// ```
/// use ferrolearn_numerical::optimize::NewtonCG;
///
/// let opt = NewtonCG::new()
///     .with_max_iter(500)
///     .with_tol(1e-10);
/// ```
pub struct NewtonCG {
    /// Maximum number of outer (Newton) iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Maximum number of CG iterations per Newton step.
    pub max_cg_iter: usize,
}

impl Default for NewtonCG {
    fn default() -> Self {
        Self::new()
    }
}

impl NewtonCG {
    /// Create a new `NewtonCG` optimizer with default settings.
    ///
    /// Defaults: `max_iter = 200`, `tol = 1e-8`, `max_cg_iter = 200`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            max_cg_iter: 200,
        }
    }

    /// Set the maximum number of outer Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the gradient norm convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of CG iterations per Newton step.
    #[must_use]
    pub fn with_max_cg_iter(mut self, max_cg_iter: usize) -> Self {
        self.max_cg_iter = max_cg_iter;
        self
    }

    /// Minimize an unconstrained objective using Newton-CG.
    ///
    /// # Arguments
    ///
    /// - `fun_grad` — closure returning `(f(x), grad f(x))`.
    /// - `hessp` — closure returning the Hessian-vector product `H(x) p`.
    /// - `x0` — initial guess.
    ///
    /// # Returns
    ///
    /// An [`OptimizeResult`] on success, or an error message if the input is
    /// invalid (e.g., zero-length initial guess).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x0` has length zero.
    pub fn minimize<FG, HP>(
        &self,
        mut fun_grad: FG,
        mut hessp: HP,
        x0: Array1<f64>,
    ) -> Result<OptimizeResult, String>
    where
        FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err("initial guess x0 must have at least one element".into());
        }

        let mut x = x0;
        let (mut f, mut g) = fun_grad(&x);

        for iter in 0..self.max_iter {
            let g_norm = norm(&g);
            if g_norm < self.tol {
                return Ok(OptimizeResult {
                    x,
                    fun: f,
                    grad: g,
                    n_iter: iter,
                    converged: true,
                });
            }

            // CG tolerance: use Eisenstat-Walker forcing term.
            let cg_tol = f64::min(0.5, g_norm.sqrt()) * g_norm;

            // Approximately solve H d = -g using CG.
            let d = cg_solve(&mut hessp, &x, &g, cg_tol, self.max_cg_iter, n);

            // Backtracking line search (Armijo condition).
            let dg = dot(&g, &d);
            if dg >= 0.0 {
                // CG gave a non-descent direction (should be rare); fall back
                // to steepest descent.
                let d_sd = &g * (-1.0);
                let alpha = backtracking_line_search(&mut fun_grad, &x, f, &g, &d_sd);
                x = &x + &(&d_sd * alpha);
            } else {
                let alpha = backtracking_line_search(&mut fun_grad, &x, f, &g, &d);
                x = &x + &(&d * alpha);
            }

            let (f_new, g_new) = fun_grad(&x);
            f = f_new;
            g = g_new;
        }

        let g_norm = norm(&g);
        Ok(OptimizeResult {
            x,
            fun: f,
            grad: g,
            n_iter: self.max_iter,
            converged: g_norm < self.tol,
        })
    }
}

/// CG inner solver: approximately solve `H d = -g`.
///
/// Returns the step `d`. Terminates early on negative curvature or when the
/// residual is small enough.
fn cg_solve<HP>(
    hessp: &mut HP,
    x: &Array1<f64>,
    g: &Array1<f64>,
    tol: f64,
    max_iter: usize,
    _n: usize,
) -> Array1<f64>
where
    HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    let n = g.len();
    let mut d = Array1::zeros(n);
    let mut r = g.clone(); // residual = g + H d = g (since d=0)
    let mut p = &r * (-1.0); // search direction

    for _cg_iter in 0..max_iter {
        let hp = hessp(x, &p);
        let p_hp = dot(&p, &hp);

        // Negative curvature: return current d if nonzero, else -g direction.
        if p_hp <= 1e-30 {
            if dot(&d, &d) > 0.0 {
                return d;
            }
            return g * (-1.0);
        }

        let r_dot_r = dot(&r, &r);
        let alpha = r_dot_r / p_hp;

        d = &d + &(&p * alpha);
        r = &r + &(&hp * alpha);

        let r_norm = norm(&r);
        if r_norm < tol {
            return d;
        }

        let r_dot_r_new = dot(&r, &r);
        let beta = r_dot_r_new / r_dot_r;
        p = &r * (-1.0) + &(&p * beta);
    }

    d
}

/// Backtracking line search with Armijo sufficient-decrease condition.
///
/// Returns a step size `alpha` satisfying `f(x + alpha d) <= f(x) + c alpha g^T d`.
fn backtracking_line_search<FG>(
    fun_grad: &mut FG,
    x: &Array1<f64>,
    f0: f64,
    g: &Array1<f64>,
    d: &Array1<f64>,
) -> f64
where
    FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
{
    let c = 1e-4;
    let rho = 0.5;
    let dg = dot(g, d);
    let mut alpha = 1.0;

    for _ in 0..40 {
        let x_new = x + &(d * alpha);
        let (f_new, _) = fun_grad(&x_new);
        if f_new <= f0 + c * alpha * dg {
            return alpha;
        }
        alpha *= rho;
    }

    alpha
}

// ---------------------------------------------------------------------------
// Trust-Region Newton-CG (Steihaug-Toint)
// ---------------------------------------------------------------------------

/// Trust-region Newton-CG optimizer using the Steihaug-Toint CG subproblem
/// solver.
///
/// At each iteration, the CG inner loop approximately solves `H d = -g`
/// subject to the constraint `||d|| <= delta` (the trust-region radius).
/// The trust-region radius is adapted based on the ratio of actual to
/// predicted reduction.
///
/// This method is more robust than line-search Newton-CG, especially for
/// problems with near-singular Hessians or saddle points.
///
/// # Builder
///
/// ```
/// use ferrolearn_numerical::optimize::TrustRegionNCG;
///
/// let opt = TrustRegionNCG::new()
///     .with_max_iter(500)
///     .with_tol(1e-10);
/// ```
pub struct TrustRegionNCG {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient norm convergence tolerance.
    pub tol: f64,
    /// Initial trust-region radius.
    pub initial_radius: f64,
    /// Maximum trust-region radius.
    pub max_radius: f64,
}

impl Default for TrustRegionNCG {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustRegionNCG {
    /// Create a new `TrustRegionNCG` optimizer with default settings.
    ///
    /// Defaults: `max_iter = 200`, `tol = 1e-8`, `initial_radius = 1.0`,
    /// `max_radius = 1000.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
            initial_radius: 1.0,
            max_radius: 1000.0,
        }
    }

    /// Set the maximum number of outer iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the gradient norm convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the initial trust-region radius.
    #[must_use]
    pub fn with_initial_radius(mut self, radius: f64) -> Self {
        self.initial_radius = radius;
        self
    }

    /// Set the maximum trust-region radius.
    #[must_use]
    pub fn with_max_radius(mut self, radius: f64) -> Self {
        self.max_radius = radius;
        self
    }

    /// Minimize an unconstrained objective using trust-region Newton-CG
    /// (Steihaug-Toint).
    ///
    /// # Arguments
    ///
    /// - `fun_grad` — closure returning `(f(x), grad f(x))`.
    /// - `hessp` — closure returning the Hessian-vector product `H(x) p`.
    /// - `x0` — initial guess.
    ///
    /// # Returns
    ///
    /// An [`OptimizeResult`] on success, or an error message if the input is
    /// invalid (e.g., zero-length initial guess).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x0` has length zero.
    pub fn minimize<FG, HP>(
        &self,
        mut fun_grad: FG,
        mut hessp: HP,
        x0: Array1<f64>,
    ) -> Result<OptimizeResult, String>
    where
        FG: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err("initial guess x0 must have at least one element".into());
        }

        let eta = 1e-4; // step acceptance threshold
        let mut x = x0;
        let (mut f, mut g) = fun_grad(&x);
        let mut delta = self.initial_radius;

        for iter in 0..self.max_iter {
            let g_norm = norm(&g);
            if g_norm < self.tol {
                return Ok(OptimizeResult {
                    x,
                    fun: f,
                    grad: g,
                    n_iter: iter,
                    converged: true,
                });
            }

            // Solve the trust-region CG subproblem (Steihaug-Toint).
            let d = steihaug_cg(&mut hessp, &x, &g, delta, n);

            // Predicted reduction: -( g^T d + 0.5 d^T H d ).
            let hd = hessp(&x, &d);
            let pred = -(dot(&g, &d) + 0.5 * dot(&d, &hd));

            // Actual reduction.
            let x_new = &x + &d;
            let (f_new, g_new) = fun_grad(&x_new);
            let ared = f - f_new;

            let d_norm = norm(&d);

            // Compute ratio rho = actual / predicted.
            let rho = if pred.abs() < 1e-30 {
                // Predicted reduction essentially zero — treat as good step
                // if actual reduction is non-negative.
                if ared >= 0.0 { 1.0 } else { 0.0 }
            } else {
                ared / pred
            };

            // Update trust-region radius.
            if rho < 0.25 {
                delta *= 0.25;
            } else if rho > 0.75 && (d_norm - delta).abs() < 1e-12 * delta.max(1.0) {
                delta = (2.0 * delta).min(self.max_radius);
            }

            // Accept or reject step.
            if rho > eta {
                x = x_new;
                f = f_new;
                g = g_new;
            }
        }

        let g_norm = norm(&g);
        Ok(OptimizeResult {
            x,
            fun: f,
            grad: g,
            n_iter: self.max_iter,
            converged: g_norm < self.tol,
        })
    }
}

/// Steihaug-Toint CG subproblem solver.
///
/// Approximately solves `H d = -g` subject to `||d|| <= delta`.
/// On negative curvature or if the CG step would leave the trust region,
/// the solution is extended (or truncated) to the trust-region boundary.
fn steihaug_cg<HP>(
    hessp: &mut HP,
    x: &Array1<f64>,
    g: &Array1<f64>,
    delta: f64,
    n: usize,
) -> Array1<f64>
where
    HP: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    let max_cg = n.max(200);
    let tol = f64::min(0.5, norm(g).sqrt()) * norm(g);

    let mut d = Array1::zeros(n);
    let mut r = g.clone();
    let mut p = &r * (-1.0);

    // If gradient is already tiny, return zero step.
    if norm(&r) < tol {
        return d;
    }

    for _cg_iter in 0..max_cg {
        let hp = hessp(x, &p);
        let p_hp = dot(&p, &hp);

        // Negative curvature: go to trust-region boundary along p.
        if p_hp <= 1e-30 {
            let tau = boundary_step(&d, &p, delta);
            return &d + &(&p * tau);
        }

        let r_dot_r = dot(&r, &r);
        let alpha = r_dot_r / p_hp;

        let d_next = &d + &(&p * alpha);

        // If the CG step leaves the trust region, truncate to boundary.
        if norm(&d_next) >= delta {
            let tau = boundary_step(&d, &p, delta);
            return &d + &(&p * tau);
        }

        d = d_next;
        r = &r + &(&hp * alpha);

        if norm(&r) < tol {
            return d;
        }

        let r_dot_r_new = dot(&r, &r);
        let beta = r_dot_r_new / r_dot_r;
        p = &r * (-1.0) + &(&p * beta);
    }

    d
}

/// Find the positive scalar `tau` such that `||d + tau p|| = delta`.
///
/// Solves the quadratic `||d + tau p||^2 = delta^2` and returns the larger
/// (positive) root.
fn boundary_step(d: &Array1<f64>, p: &Array1<f64>, delta: f64) -> f64 {
    let dd = dot(d, d);
    let dp = dot(d, p);
    let pp = dot(p, p);

    // Quadratic: pp tau^2 + 2 dp tau + (dd - delta^2) = 0
    let discrim = dp * dp - pp * (dd - delta * delta);
    let sqrt_discrim = if discrim > 0.0 { discrim.sqrt() } else { 0.0 };

    // We want the positive root.
    (-dp + sqrt_discrim) / pp
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Euclidean (L2) norm of a vector.
#[inline]
fn norm(v: &Array1<f64>) -> f64 {
    dot(v, v).sqrt()
}

/// Dot product of two vectors.
#[inline]
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, array};

    // -----------------------------------------------------------------------
    // Helpers: Quadratic f(x) = 0.5 x^T A x - b^T x
    // -----------------------------------------------------------------------

    /// Returns (fun_grad, hessp) for a 3-dimensional quadratic with
    /// A = diag(2, 4, 6) and b = [1, 2, 3].
    ///
    /// The solution is x* = A^{-1} b = [0.5, 0.5, 0.5].
    fn quadratic_3d() -> (
        impl FnMut(&Array1<f64>) -> (f64, Array1<f64>),
        impl FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    ) {
        let a_diag = array![2.0, 4.0, 6.0];
        let b = array![1.0, 2.0, 3.0];

        let a_diag2 = a_diag.clone();
        let b2 = b.clone();

        let fun_grad = move |x: &Array1<f64>| {
            let ax = &a_diag * x;
            let f_val = 0.5 * x.dot(&ax) - x.dot(&b);
            let g = &ax - &b;
            (f_val, g)
        };

        let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &a_diag2 * p + &(&b2 * 0.0);

        (fun_grad, hessp)
    }

    // -----------------------------------------------------------------------
    // Helpers: Rosenbrock f(x,y) = (1-x)^2 + 100(y - x^2)^2
    // -----------------------------------------------------------------------

    fn rosenbrock_fun_grad(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        let g = array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b,];
        (f, g)
    }

    fn rosenbrock_hessp(x: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
        // H = [[2 - 400(y - x^2) + 800 x^2,  -400 x],
        //      [-400 x,                        200   ]]
        let h00 = 2.0 - 400.0 * (x[1] - x[0] * x[0]) + 800.0 * x[0] * x[0];
        let h01 = -400.0 * x[0];
        let h11 = 200.0;
        array![h00 * p[0] + h01 * p[1], h01 * p[0] + h11 * p[1],]
    }

    // -----------------------------------------------------------------------
    // Newton-CG tests
    // -----------------------------------------------------------------------

    #[test]
    fn newton_cg_quadratic() {
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result = NewtonCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on a quadratic");
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[2], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn newton_cg_rosenbrock() {
        let x0 = array![-1.0, 1.0];

        let result = NewtonCG::new()
            .with_max_iter(500)
            .with_tol(1e-10)
            .minimize(rosenbrock_fun_grad, rosenbrock_hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on Rosenbrock");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    // -----------------------------------------------------------------------
    // Trust-Region NCG tests
    // -----------------------------------------------------------------------

    #[test]
    fn trust_region_quadratic() {
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result = TrustRegionNCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on a quadratic");
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[1], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(result.x[2], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn trust_region_rosenbrock() {
        let x0 = array![-1.0, 1.0];

        let result = TrustRegionNCG::new()
            .with_max_iter(500)
            .with_tol(1e-10)
            .minimize(rosenbrock_fun_grad, rosenbrock_hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on Rosenbrock");
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn trust_region_high_dimensional() {
        // 10-dimensional quadratic: f(x) = 0.5 sum_i (i+1) x_i^2 - sum_i x_i
        // Solution: x_i = 1 / (i+1)
        let n = 10;
        let diag: Array1<f64> = (1..=n).map(|i| i as f64).collect();
        let b = Array1::ones(n);

        let diag2 = diag.clone();

        let fun_grad = move |x: &Array1<f64>| {
            let ax = &diag * x;
            let f_val = 0.5 * x.dot(&ax) - x.dot(&b);
            let g = &ax - &b;
            (f_val, g)
        };

        let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &diag2 * p;

        let x0 = Array1::from_elem(n, 5.0);

        let result = TrustRegionNCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");

        assert!(result.converged, "should converge on 10-d quadratic");
        for i in 0..n {
            let expected = 1.0 / (i + 1) as f64;
            assert_abs_diff_eq!(result.x[i], expected, epsilon = 1e-8);
        }
    }

    #[test]
    fn newton_cg_convergence_flag() {
        // Easy problem: should converge with default settings.
        let (fun_grad, hessp) = quadratic_3d();
        let x0 = array![1.0, 1.0, 1.0];

        let result = NewtonCG::new()
            .minimize(fun_grad, hessp, x0)
            .expect("optimization should succeed");
        assert!(result.converged, "should converge on easy problem");

        // Same problem but max_iter=1: should NOT converge.
        let (fun_grad2, hessp2) = quadratic_3d();
        let x0 = array![10.0, -5.0, 3.0];

        let result2 = NewtonCG::new()
            .with_max_iter(1)
            .minimize(fun_grad2, hessp2, x0)
            .expect("optimization should succeed (even if not converged)");
        assert!(
            !result2.converged,
            "should not converge with only 1 iteration from a distant start"
        );
    }
}
