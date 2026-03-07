//! Accuracy comparison tests that verify Rust results match scipy reference values.
//!
//! Each test section targets a specific module of `ferrolearn-numerical` and compares
//! against analytically known values or high-precision scipy reference outputs.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, array};
use sprs::TriMat;

use ferrolearn_numerical::distributions::{
    Beta, ChiSquared, ContinuousDistribution, FDist, Gamma, Normal, StudentsT,
};
use ferrolearn_numerical::integrate::{gauss_legendre, quad};
use ferrolearn_numerical::interpolate::{BoundaryCondition, CubicSpline};
use ferrolearn_numerical::optimize::{NewtonCG, TrustRegionNCG};
use ferrolearn_numerical::sparse_eig::{LanczosSolver, WhichEigenvalues};
use ferrolearn_numerical::sparse_graph;

// ===========================================================================
// Helper: build a CSR matrix from triplets
// ===========================================================================

fn csr_from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> sprs::CsMat<f64> {
    let mut tri = TriMat::new((n, n));
    for &(r, c, v) in triplets {
        tri.add_triplet(r, c, v);
    }
    tri.to_csr()
}

// ===========================================================================
// Sparse Eigensolver Accuracy
// ===========================================================================

/// 20x20 tridiagonal matrix with pattern (1, -2, 1) -- the discrete Laplacian.
/// Analytical eigenvalues: lambda_k = -4*sin^2(k*pi/(2*(n+1))) for k=1..n.
/// Compute top 5 eigenvalues (least negative) with LanczosSolver and verify
/// they match analytical values within 1e-8.
#[test]
fn eigsh_tridiagonal_matches_analytical() {
    let n = 20;

    // Build the tridiagonal matrix as a sparse CSR matrix via triplets.
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, -2.0);
        if i + 1 < n {
            tri.add_triplet(i, i + 1, 1.0);
            tri.add_triplet(i + 1, i, 1.0);
        }
    }
    let mat = tri.to_csr();

    // Compute top 5 eigenvalues (largest algebraic, i.e. least negative).
    let solver = LanczosSolver::new(5)
        .with_which(WhichEigenvalues::LargestAlgebraic)
        .with_tol(1e-12)
        .with_max_iter(500);
    let result = solver
        .solve_sparse(&mat)
        .expect("Lanczos solver should converge on the discrete Laplacian");

    // Analytical eigenvalues for the (1, -2, 1) tridiagonal matrix:
    //   lambda_k = -4*sin^2(k*pi/(2*(n+1))) for k = 1..n
    // All are negative. The largest (least negative) correspond to k = 1..5.
    let mut analytical: Vec<f64> = (1..=n)
        .map(|k| {
            let theta = k as f64 * std::f64::consts::PI / (n as f64 + 1.0);
            -4.0 * (theta / 2.0).sin().powi(2)
        })
        .collect();
    analytical.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending (least negative first)

    let mut computed: Vec<f64> = result.eigenvalues.to_vec();
    computed.sort_by(|a, b| b.partial_cmp(a).unwrap());

    for i in 0..5 {
        assert_abs_diff_eq!(computed[i], analytical[i], epsilon = 1e-8,);
    }
}

/// 20x20 diagonal sparse matrix with well-separated eigenvalues.
/// Uses entries 1, 4, 9, 16, ..., 400 (perfect squares) so eigenvalues
/// are well-separated and the Lanczos solver can resolve them cleanly.
/// Top 5 should be [400, 361, 324, 289, 256].
/// Bottom 5 should be [1, 4, 9, 16, 25].
/// Verify within 1e-8.
#[test]
fn eigsh_diagonal_exact() {
    let n = 20;
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        let val = ((i + 1) * (i + 1)) as f64;
        tri.add_triplet(i, i, val);
    }
    let mat = tri.to_csr();

    // Top 5 eigenvalues.
    let top_result = LanczosSolver::new(5)
        .with_which(WhichEigenvalues::LargestAlgebraic)
        .with_tol(1e-12)
        .with_max_iter(500)
        .solve_sparse(&mat)
        .expect("Lanczos solver should converge on diagonal matrix (top 5)");

    let mut top_evals: Vec<f64> = top_result.eigenvalues.to_vec();
    top_evals.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let expected_top = [400.0, 361.0, 324.0, 289.0, 256.0];
    for (i, &expected) in expected_top.iter().enumerate() {
        assert_abs_diff_eq!(top_evals[i], expected, epsilon = 1e-8,);
    }

    // Bottom 5 eigenvalues.
    let bottom_result = LanczosSolver::new(5)
        .with_which(WhichEigenvalues::SmallestAlgebraic)
        .with_tol(1e-12)
        .with_max_iter(500)
        .solve_sparse(&mat)
        .expect("Lanczos solver should converge on diagonal matrix (bottom 5)");

    let mut bottom_evals: Vec<f64> = bottom_result.eigenvalues.to_vec();
    bottom_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let expected_bottom = [1.0, 4.0, 9.0, 16.0, 25.0];
    for (i, &expected) in expected_bottom.iter().enumerate() {
        assert_abs_diff_eq!(bottom_evals[i], expected, epsilon = 1e-8,);
    }
}

// ===========================================================================
// Sparse Graph Accuracy
// ===========================================================================

/// Small 6-node weighted graph with known shortest paths.
/// Verify exact distances match hand-computed values.
#[test]
fn dijkstra_known_graph() {
    // Graph (directed):
    //
    //   0 --7--> 1 --1--> 2
    //   |        ^        |
    //   9        2       14
    //   v        |        v
    //   5 --11-> 4 --6--> 3
    //   ^                 |
    //   +------9----------+
    //
    // Edges:
    //   0->1: 7, 0->5: 9
    //   1->2: 1, 1->4: 2  (wait, let's define a clean graph)
    //
    // Let's use a classic textbook graph:
    //   0 -> 1 (w=7), 0 -> 2 (w=9), 0 -> 5 (w=14)
    //   1 -> 2 (w=10), 1 -> 3 (w=15)
    //   2 -> 3 (w=11), 2 -> 5 (w=2)
    //   3 -> 4 (w=6)
    //   4 -> 5 (w=9)
    //
    // Shortest paths from 0:
    //   0: 0
    //   1: 7   (0->1)
    //   2: 9   (0->2)
    //   3: 20  (0->2->3)
    //   4: 26  (0->2->3->4)
    //   5: 11  (0->2->5)

    let graph = csr_from_triplets(
        6,
        &[
            (0, 1, 7.0),
            (0, 2, 9.0),
            (0, 5, 14.0),
            (1, 2, 10.0),
            (1, 3, 15.0),
            (2, 3, 11.0),
            (2, 5, 2.0),
            (3, 4, 6.0),
            (4, 5, 9.0),
        ],
    );

    let result =
        sparse_graph::dijkstra(&graph, 0).expect("Dijkstra should succeed on a valid graph");

    let expected_distances = [0.0, 7.0, 9.0, 20.0, 26.0, 11.0];
    for (i, &expected) in expected_distances.iter().enumerate() {
        assert_abs_diff_eq!(result.distances[i], expected, epsilon = 1e-12,);
    }
}

/// 10-node graph with 3 known components: {0,1,2,3}, {4,5,6}, {7,8,9}.
/// Verify n_components = 3 and labels are consistent.
#[test]
fn connected_components_known() {
    // Component 0: nodes 0, 1, 2, 3 (chain: 0-1, 1-2, 2-3)
    // Component 1: nodes 4, 5, 6 (chain: 4-5, 5-6)
    // Component 2: nodes 7, 8, 9 (chain: 7-8, 8-9)
    let graph = csr_from_triplets(
        10,
        &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (4, 5, 1.0),
            (5, 6, 1.0),
            (7, 8, 1.0),
            (8, 9, 1.0),
        ],
    );

    let result =
        sparse_graph::connected_components(&graph).expect("connected_components should succeed");

    assert_eq!(
        result.n_components, 3,
        "expected 3 connected components, got {}",
        result.n_components
    );

    // Verify nodes within the same component share the same label.
    // Component {0,1,2,3}
    let label_a = result.labels[0];
    for &node in &[1, 2, 3] {
        assert_eq!(
            result.labels[node], label_a,
            "node {node} should be in the same component as node 0"
        );
    }

    // Component {4,5,6}
    let label_b = result.labels[4];
    for &node in &[5, 6] {
        assert_eq!(
            result.labels[node], label_b,
            "node {node} should be in the same component as node 4"
        );
    }

    // Component {7,8,9}
    let label_c = result.labels[7];
    for &node in &[8, 9] {
        assert_eq!(
            result.labels[node], label_c,
            "node {node} should be in the same component as node 7"
        );
    }

    // Verify the three components have distinct labels.
    assert_ne!(
        label_a, label_b,
        "components {{0,1,2,3}} and {{4,5,6}} should differ"
    );
    assert_ne!(
        label_a, label_c,
        "components {{0,1,2,3}} and {{7,8,9}} should differ"
    );
    assert_ne!(
        label_b, label_c,
        "components {{4,5,6}} and {{7,8,9}} should differ"
    );
}

// ===========================================================================
// Distribution Accuracy (vs scipy reference values)
// ===========================================================================

/// N(0,1): verify pdf(0), cdf(1.96), ppf(0.975) against scipy reference values.
#[test]
fn normal_matches_scipy() {
    let n = Normal::new(0.0, 1.0).expect("standard normal should be valid");

    // scipy.stats.norm.pdf(0) = 0.3989422804014327
    assert_abs_diff_eq!(n.pdf(0.0), 0.3989422804014327, epsilon = 1e-12,);

    // scipy.stats.norm.cdf(1.96) = 0.9750021048517796
    // statrs and scipy may differ at the ~13th decimal place due to different
    // erf implementations; verify within 1e-10 which is still excellent.
    assert_abs_diff_eq!(n.cdf(1.96), 0.9750021048517796, epsilon = 1e-10,);

    // scipy.stats.norm.ppf(0.975) = 1.959963984540054
    assert_abs_diff_eq!(n.ppf(0.975), 1.959963984540054, epsilon = 1e-10,);
}

/// Chi-squared(5): verify cdf and sf at x=11.0705 against scipy reference values.
#[test]
fn chi2_matches_scipy() {
    let c = ChiSquared::new(5.0).expect("chi-squared(5) should be valid");

    // scipy.stats.chi2.cdf(11.0705, 5) ~ 0.9500003848...
    let cdf_val = c.cdf(11.0705);
    assert_abs_diff_eq!(cdf_val, 0.95, epsilon = 1e-4,);

    // scipy.stats.chi2.sf(11.0705, 5) ~ 0.04999961...
    let sf_val = c.sf(11.0705);
    assert_abs_diff_eq!(sf_val, 0.05, epsilon = 1e-4,);

    // Verify sf + cdf = 1 within machine precision.
    assert_abs_diff_eq!(cdf_val + sf_val, 1.0, epsilon = 1e-12,);
}

/// F(3,40): verify sf(2.84) against scipy reference value.
/// scipy.stats.f.sf(2.84, 3, 40) ~ 0.0499...
#[test]
fn f_dist_matches_scipy() {
    let f = FDist::new(3.0, 40.0).expect("F(3,40) should be valid");

    // scipy.stats.f.sf(2.84, 3, 40) is approximately 0.05
    let sf_val = f.sf(2.84);
    assert_abs_diff_eq!(sf_val, 0.05, epsilon = 1e-2,);

    // Also check cdf + sf = 1.
    let cdf_val = f.cdf(2.84);
    assert_abs_diff_eq!(cdf_val + sf_val, 1.0, epsilon = 1e-12,);
}

/// t(10): verify cdf(2.228) against scipy reference value.
/// scipy.stats.t.cdf(2.228, 10) ~ 0.975
#[test]
fn t_dist_matches_scipy() {
    let t = StudentsT::new(10.0).expect("t(10) should be valid");

    // scipy.stats.t.cdf(2.228, 10) ~ 0.975
    let cdf_val = t.cdf(2.228);
    assert_abs_diff_eq!(cdf_val, 0.975, epsilon = 1e-3,);

    // Verify symmetry: cdf(x) + cdf(-x) = 1
    let cdf_neg = t.cdf(-2.228);
    assert_abs_diff_eq!(cdf_val + cdf_neg, 1.0, epsilon = 1e-12,);
}

/// Beta(2,5): mean = 2/7, variance = 10/294.
/// Gamma(3, rate=2): mean = 1.5, variance = 0.75.
/// Verify within 1e-12.
#[test]
fn beta_gamma_matches_scipy() {
    // Beta(2, 5)
    let b = Beta::new(2.0, 5.0).expect("Beta(2,5) should be valid");
    assert_abs_diff_eq!(b.mean(), 2.0 / 7.0, epsilon = 1e-12,);
    assert_abs_diff_eq!(
        b.variance(),
        10.0 / (49.0 * 8.0), // = 10/392, which equals alpha*beta / ((a+b)^2 * (a+b+1))
        epsilon = 1e-12,
    );

    // Gamma(shape=3, rate=2): mean = shape/rate = 1.5, variance = shape/rate^2 = 0.75
    let g = Gamma::new(3.0, 2.0).expect("Gamma(3, 2) should be valid");
    assert_abs_diff_eq!(g.mean(), 1.5, epsilon = 1e-12,);
    assert_abs_diff_eq!(g.variance(), 0.75, epsilon = 1e-12,);
}

// ===========================================================================
// Optimizer Accuracy
// ===========================================================================

/// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2.
/// Start from (-1, 1) with both NewtonCG and TrustRegionNCG.
/// Verify solution is within 1e-4 of (1, 1) and f(solution) < 1e-8.
#[test]
fn rosenbrock_converges_to_minimum() {
    let fun_grad = |x: &Array1<f64>| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        let f = a * a + 100.0 * b * b;
        let g = array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b];
        (f, g)
    };

    let hessp = |x: &Array1<f64>, p: &Array1<f64>| {
        let h00 = 2.0 - 400.0 * (x[1] - x[0] * x[0]) + 800.0 * x[0] * x[0];
        let h01 = -400.0 * x[0];
        let h11 = 200.0;
        array![h00 * p[0] + h01 * p[1], h01 * p[0] + h11 * p[1]]
    };

    let x0 = array![-1.0, 1.0];

    // NewtonCG
    let result_ncg = NewtonCG::new()
        .with_max_iter(500)
        .with_tol(1e-10)
        .minimize(fun_grad, hessp, x0.clone())
        .expect("NewtonCG should succeed on Rosenbrock");

    assert_abs_diff_eq!(result_ncg.x[0], 1.0, epsilon = 1e-4,);
    assert_abs_diff_eq!(result_ncg.x[1], 1.0, epsilon = 1e-4,);
    assert!(
        result_ncg.fun < 1e-8,
        "NewtonCG: f(solution) = {} should be < 1e-8",
        result_ncg.fun
    );

    // TrustRegionNCG
    let result_tr = TrustRegionNCG::new()
        .with_max_iter(500)
        .with_tol(1e-10)
        .minimize(fun_grad, hessp, x0)
        .expect("TrustRegionNCG should succeed on Rosenbrock");

    assert_abs_diff_eq!(result_tr.x[0], 1.0, epsilon = 1e-4,);
    assert_abs_diff_eq!(result_tr.x[1], 1.0, epsilon = 1e-4,);
    assert!(
        result_tr.fun < 1e-8,
        "TrustRegionNCG: f(solution) = {} should be < 1e-8",
        result_tr.fun
    );
}

/// Quadratic f(x) = 0.5 * x^T @ diag(1,2,...,10) @ x - b^T @ x with b = [1,1,...,1].
/// Exact solution: x_i = 1/i.
/// Verify within 1e-8.
#[test]
fn quadratic_exact_solution() {
    let n = 10;
    let diag_vals: Array1<f64> = (1..=n).map(|i| i as f64).collect();
    let b = Array1::ones(n);

    let diag_for_fg = diag_vals.clone();
    let b_for_fg = b.clone();
    let fun_grad = move |x: &Array1<f64>| {
        let ax = &diag_for_fg * x;
        let f_val = 0.5 * x.dot(&ax) - x.dot(&b_for_fg);
        let g = &ax - &b_for_fg;
        (f_val, g)
    };

    let diag_for_hp = diag_vals.clone();
    let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &diag_for_hp * p;

    let x0 = Array1::from_elem(n, 5.0);

    let result = TrustRegionNCG::new()
        .with_tol(1e-12)
        .minimize(fun_grad, hessp, x0)
        .expect("TrustRegionNCG should converge on a quadratic");

    assert!(
        result.converged,
        "optimizer should converge on a quadratic objective"
    );

    for i in 0..n {
        let expected = 1.0 / (i + 1) as f64;
        assert_abs_diff_eq!(result.x[i], expected, epsilon = 1e-8,);
    }
}

// ===========================================================================
// Interpolation Accuracy
// ===========================================================================

/// 100 points of sin(x) on [0, 2*pi].
/// Evaluate spline at 1000 intermediate points.
/// Max error should be < 1e-4 (cubic interpolation of smooth function).
#[test]
fn cubic_spline_sin_accuracy() {
    let n_knots = 100;
    let x_knots: Vec<f64> = (0..n_knots)
        .map(|i| 2.0 * std::f64::consts::PI * i as f64 / (n_knots - 1) as f64)
        .collect();
    let y_knots: Vec<f64> = x_knots.iter().map(|&xi| xi.sin()).collect();

    let spline = CubicSpline::new(&x_knots, &y_knots, BoundaryCondition::NotAKnot)
        .expect("CubicSpline should succeed on sin data");

    let n_test = 1000;
    let mut max_error = 0.0_f64;
    for i in 0..n_test {
        // Test at points offset from knots to ensure we're evaluating between them.
        let x_test = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / n_test as f64;
        let spline_val = spline.eval(x_test);
        let exact_val = x_test.sin();
        let error = (spline_val - exact_val).abs();
        max_error = max_error.max(error);
    }

    assert!(
        max_error < 1e-4,
        "cubic spline of sin(x) with {n_knots} knots: max error = {max_error:.2e}, \
         expected < 1e-4"
    );
}

/// Fit not-a-knot spline to y = x^3 - 2x^2 + x - 1 at 20 points on [-2, 2].
/// Evaluate at 100 test points.
/// Max error should be < 1e-10 (cubic spline reproduces cubic polynomials
/// exactly with not-a-knot BC).
#[test]
fn spline_reproduces_cubic_exactly() {
    let cubic_poly = |x: f64| x * x * x - 2.0 * x * x + x - 1.0;

    let n_knots = 20;
    let x_knots: Vec<f64> = (0..n_knots)
        .map(|i| -2.0 + 4.0 * i as f64 / (n_knots - 1) as f64)
        .collect();
    let y_knots: Vec<f64> = x_knots.iter().map(|&xi| cubic_poly(xi)).collect();

    let spline = CubicSpline::new(&x_knots, &y_knots, BoundaryCondition::NotAKnot)
        .expect("CubicSpline should succeed on cubic polynomial data");

    let n_test = 100;
    let mut max_error = 0.0_f64;
    for i in 0..n_test {
        let x_test = -2.0 + 4.0 * (i as f64 + 0.5) / n_test as f64;
        let spline_val = spline.eval(x_test);
        let exact_val = cubic_poly(x_test);
        let error = (spline_val - exact_val).abs();
        max_error = max_error.max(error);
    }

    assert!(
        max_error < 1e-10,
        "not-a-knot spline of cubic polynomial: max error = {max_error:.2e}, \
         expected < 1e-10 (cubic splines reproduce cubics exactly)"
    );
}

// ===========================================================================
// Quadrature Accuracy
// ===========================================================================

/// Verify adaptive Simpson quadrature matches known integral values:
///   - integral_0^pi sin(x) dx = 2.0
///   - integral_{-5}^{5} exp(-x^2) dx = sqrt(pi)
///   - integral_0^1 x^10 dx = 1/11
///   - integral_0^1 1/(1+x^2) dx = pi/4
#[test]
fn quad_matches_scipy_integrals() {
    // Integral of sin(x) from 0 to pi = 2.0
    let r1 = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-14);
    assert_abs_diff_eq!(r1.value, 2.0, epsilon = 1e-12,);

    // Integral of exp(-x^2) from -5 to 5 ~ sqrt(pi)
    let r2 = quad(|x| (-x * x).exp(), -5.0, 5.0, 1e-12);
    assert_abs_diff_eq!(r2.value, std::f64::consts::PI.sqrt(), epsilon = 1e-8,);

    // Integral of x^10 from 0 to 1 = 1/11
    let r3 = quad(|x| x.powi(10), 0.0, 1.0, 1e-14);
    assert_abs_diff_eq!(r3.value, 1.0 / 11.0, epsilon = 1e-10,);

    // Integral of 1/(1+x^2) from 0 to 1 = pi/4
    let r4 = quad(|x| 1.0 / (1.0 + x * x), 0.0, 1.0, 1e-14);
    assert_abs_diff_eq!(r4.value, std::f64::consts::FRAC_PI_4, epsilon = 1e-10,);
}

/// Gauss-Legendre polynomial exactness:
///   - GL(3) on integral_{-1}^{1} x^4 dx should be exact (degree 4, 2*3-1=5 >= 4).
///   - GL(5) on integral_0^1 x^9 dx = 0.1 should be exact (degree 9, 2*5-1=9).
/// Verify within 1e-14.
#[test]
fn gauss_legendre_polynomial_exact() {
    // GL(3): exact for polynomials of degree <= 5.
    // integral_{-1}^{1} x^4 dx = 2/5
    let r1 = gauss_legendre(|x| x.powi(4), -1.0, 1.0, 3).expect("GL(3) should succeed");
    assert_abs_diff_eq!(r1.value, 2.0 / 5.0, epsilon = 1e-14,);

    // GL(5): exact for polynomials of degree <= 9.
    // integral_0^1 x^9 dx = 1/10 = 0.1
    let r2 = gauss_legendre(|x| x.powi(9), 0.0, 1.0, 5).expect("GL(5) should succeed");
    assert_abs_diff_eq!(r2.value, 0.1, epsilon = 1e-14,);
}
