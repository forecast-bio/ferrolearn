//! Proof-of-API integration test for ferrolearn-numerical.
//!
//! Audit deliverable for crosslink #323 (under #251). Exercises every
//! public type and function in the 6 numerical modules.

use ferrolearn_numerical::distributions::{
    Beta, ChiSquared, ContinuousDistribution, Dirichlet, FDist, Gamma, Normal, StudentsT, chi2_sf,
    f_sf, norm_sf, t_test_two_tailed,
};
use ferrolearn_numerical::integrate::{
    QuadratureResult, gauss_legendre, gauss_legendre_composite, quad, quad_with_limit,
};
use ferrolearn_numerical::interpolate::{BoundaryCondition, CubicSpline};
use ferrolearn_numerical::optimize::{
    Minimize1DResult, NewtonCG, OptimizeResult, TrustRegionNCG, brent_bounded,
};
use ferrolearn_numerical::sparse_eig::{LanczosSolver, WhichEigenvalues, eigsh};
use ferrolearn_numerical::sparse_graph::{
    connected_components, dijkstra, dijkstra_all_pairs, minimum_spanning_tree,
};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sprs::TriMat;

#[test]
fn api_proof_distributions_continuous() {
    let n = Normal::new(0.0, 1.0).unwrap();
    assert!((n.cdf(0.0) - 0.5).abs() < 1e-9);
    let _ = n.pdf(0.5);
    let _ = n.sf(1.0);
    let _ = n.ppf(0.975);

    let c = ChiSquared::new(3.0).unwrap();
    let _ = c.pdf(1.0);
    let _ = c.cdf(2.0);
    let _ = c.sf(2.0);
    let _ = c.ppf(0.5);

    let f = FDist::new(5.0, 10.0).unwrap();
    let _ = f.pdf(1.0);
    let _ = f.cdf(2.0);

    let t = StudentsT::new(5.0).unwrap();
    let _ = t.pdf(0.0);
    let _ = t.cdf(0.5);

    let b = Beta::new(2.0, 5.0).unwrap();
    let _ = b.pdf(0.3);
    let _ = b.cdf(0.5);

    let g = Gamma::new(2.0, 1.5).unwrap();
    let _ = g.pdf(1.0);
    let _ = g.cdf(2.0);
}

#[test]
fn api_proof_distributions_dirichlet_and_helpers() {
    let mut rng = SmallRng::seed_from_u64(7);
    let d = Dirichlet::new(&[1.0, 2.0, 3.0]).unwrap();
    let sample: Array1<f64> = d.sample(&mut rng);
    assert_eq!(sample.len(), 3);
    let _ = d.ln_pdf(&[0.2, 0.3, 0.5]);

    let _ = chi2_sf(2.0, 3.0);
    let _ = f_sf(1.5, 5.0, 10.0);
    let _ = t_test_two_tailed(0.5, 5.0);
    let _ = norm_sf(1.96);
}

#[test]
fn api_proof_integrate() {
    // integrate sin(x) from 0 to pi -> 2.0
    let result: QuadratureResult = quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-9);
    assert!((result.value - 2.0).abs() < 1e-6);

    let result = quad_with_limit(|x: f64| x * x, 0.0, 1.0, 1e-9, 12);
    assert!((result.value - 1.0 / 3.0).abs() < 1e-6);

    let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 8).unwrap();
    assert!((result.value - 1.0 / 3.0).abs() < 1e-9);

    let result = gauss_legendre_composite(|x: f64| x.exp(), 0.0, 1.0, 4, 4).unwrap();
    let exact = std::f64::consts::E - 1.0;
    assert!((result.value - exact).abs() < 1e-6);
}

#[test]
fn api_proof_interpolate() {
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let ys = [0.0_f64, 1.0, 4.0, 9.0, 16.0];
    let cs = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();
    let _ = cs.eval(2.5);
    let _ = cs.eval_array(&[0.5, 1.5, 2.5]);
    let _ = cs.derivative(2.0);
    let _ = cs.second_derivative(2.0);
    let _ = cs.integrate(0.0, 4.0);

    let cs2 = CubicSpline::new(&xs, &ys, BoundaryCondition::NotAKnot).unwrap();
    let _ = cs2.eval(2.0);
}

#[test]
fn api_proof_optimize() {
    // Minimize f(x) = (x[0] - 3)^2 + (x[1] + 1)^2
    let fg = |x: &Array1<f64>| -> (f64, Array1<f64>) {
        let g = ndarray::array![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 1.0)];
        let v = (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        (v, g)
    };
    let hp = |_: &Array1<f64>, p: &Array1<f64>| -> Array1<f64> { p.mapv(|v| 2.0 * v) };

    let opt = NewtonCG::new()
        .with_max_iter(50)
        .with_tol(1e-10)
        .with_max_cg_iter(20);
    let res: OptimizeResult = opt.minimize(fg, hp, ndarray::array![0.0_f64, 0.0]).unwrap();
    assert!((res.x[0] - 3.0).abs() < 1e-3);

    let opt2 = TrustRegionNCG::new()
        .with_max_iter(50)
        .with_tol(1e-10)
        .with_initial_radius(1.0)
        .with_max_radius(100.0);
    let res2 = opt2
        .minimize(fg, hp, ndarray::array![0.0_f64, 0.0])
        .unwrap();
    assert!((res2.x[0] - 3.0).abs() < 1e-3);

    let res3: Minimize1DResult = brent_bounded(|x: f64| (x - 2.0).powi(2), 0.0, 5.0, 1e-9, 100);
    assert!((res3.x - 2.0).abs() < 1e-6);
}

#[test]
fn api_proof_sparse_eig_and_graph() {
    // Build small symmetric sparse matrix
    let mut tri = TriMat::<f64>::new((4, 4));
    tri.add_triplet(0, 0, 4.0);
    tri.add_triplet(1, 1, 3.0);
    tri.add_triplet(2, 2, 2.0);
    tri.add_triplet(3, 3, 1.0);
    let mat = tri.to_csr();

    let solver = LanczosSolver::new(2)
        .with_which(WhichEigenvalues::LargestAlgebraic)
        .with_tol(1e-8)
        .with_max_iter(100);
    let eig = solver.solve_sparse(&mat).unwrap();
    assert_eq!(eig.eigenvalues.len(), 2);

    let _eig = eigsh(&mat, 2, WhichEigenvalues::LargestAlgebraic).unwrap();

    // Graph algorithms
    let mut g = TriMat::<f64>::new((4, 4));
    g.add_triplet(0, 1, 1.0);
    g.add_triplet(1, 2, 2.0);
    g.add_triplet(2, 3, 3.0);
    let graph = g.to_csr();
    let dij = dijkstra(&graph, 0).unwrap();
    assert_eq!(dij.distances.len(), 4);
    let _ = dijkstra_all_pairs(&graph).unwrap();
    let cc = connected_components(&graph).unwrap();
    assert!(cc.n_components >= 1);
    let mst = minimum_spanning_tree(&graph).unwrap();
    // MST of a 4-node connected graph has 3 edges; storage may symmetrise.
    assert!(mst.nnz() >= 3);
}
