use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, array};
use sprs::{CsMat, TriMat};

use ferrolearn_numerical::distributions::{ChiSquared, ContinuousDistribution, Normal};
use ferrolearn_numerical::integrate::{gauss_legendre, quad};
use ferrolearn_numerical::interpolate::{BoundaryCondition, CubicSpline};
use ferrolearn_numerical::optimize::TrustRegionNCG;
use ferrolearn_numerical::sparse_eig::{WhichEigenvalues, eigsh};
use ferrolearn_numerical::sparse_graph::{
    connected_components, dijkstra, dijkstra_all_pairs, minimum_spanning_tree,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a sparse tridiagonal (1, -2, 1) matrix of size n x n.
fn tridiagonal_matrix(n: usize) -> CsMat<f64> {
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, -2.0);
        if i + 1 < n {
            tri.add_triplet(i, i + 1, 1.0);
            tri.add_triplet(i + 1, i, 1.0);
        }
    }
    tri.to_csr()
}

/// Build a deterministic random sparse graph with ~5% density and positive
/// weights in [0.1, 10.0]. The graph is symmetric (undirected).
fn random_graph_1000() -> CsMat<f64> {
    let n = 1000;
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let hash = ((i * 1000 + j).wrapping_mul(2654435761)) % 1000;
            if hash < 50 {
                let weight = 0.1 + 9.9 * ((i * 997 + j * 991) as f64 % 1000.0) / 1000.0;
                tri.add_triplet(i, j, weight);
                tri.add_triplet(j, i, weight);
            }
        }
    }
    tri.to_csr()
}

// ---------------------------------------------------------------------------
// Sparse eigensolver benchmarks
// ---------------------------------------------------------------------------

fn sparse_eig_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_eig");

    // 1000x1000 tridiagonal, k=10, SmallestAlgebraic
    let mat_1000 = tridiagonal_matrix(1000);
    group.bench_function("eigsh_1000_sa", |b| {
        b.iter(|| {
            eigsh(
                black_box(&mat_1000),
                10,
                WhichEigenvalues::SmallestAlgebraic,
            )
            .unwrap()
        });
    });

    // 1000x1000 tridiagonal, k=10, LargestAlgebraic
    group.bench_function("eigsh_1000_la", |b| {
        b.iter(|| eigsh(black_box(&mat_1000), 10, WhichEigenvalues::LargestAlgebraic).unwrap());
    });

    // 5000x5000 tridiagonal, k=10, SmallestAlgebraic
    let mat_5000 = tridiagonal_matrix(5000);
    group.sample_size(10);
    group.bench_function("eigsh_5000_sa", |b| {
        b.iter(|| {
            eigsh(
                black_box(&mat_5000),
                10,
                WhichEigenvalues::SmallestAlgebraic,
            )
            .unwrap()
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Sparse graph benchmarks
// ---------------------------------------------------------------------------

fn sparse_graph_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_graph");
    let graph = random_graph_1000();

    group.bench_function("dijkstra_single_1000", |b| {
        b.iter(|| dijkstra(black_box(&graph), 0).unwrap());
    });

    group.sample_size(10);
    group.bench_function("dijkstra_all_1000", |b| {
        b.iter(|| dijkstra_all_pairs(black_box(&graph)).unwrap());
    });

    group.sample_size(100);
    group.bench_function("connected_components_1000", |b| {
        b.iter(|| connected_components(black_box(&graph)).unwrap());
    });

    group.bench_function("mst_1000", |b| {
        b.iter(|| minimum_spanning_tree(black_box(&graph)).unwrap());
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Distribution benchmarks
// ---------------------------------------------------------------------------

fn distribution_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributions");
    let n = 100_000;
    let xs: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * f64::from(i) / (f64::from(n) - 1.0))
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    group.bench_function("normal_pdf_100k", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &xs {
                sum += normal.pdf(black_box(x));
            }
            black_box(sum)
        });
    });

    group.bench_function("normal_cdf_100k", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &xs {
                sum += normal.cdf(black_box(x));
            }
            black_box(sum)
        });
    });

    let chi2 = ChiSquared::new(5.0).unwrap();
    // Chi-squared CDF only meaningful for x >= 0; use linspace 0..20
    let chi2_xs: Vec<f64> = (0..n)
        .map(|i| 20.0 * f64::from(i) / (f64::from(n) - 1.0))
        .collect();
    group.bench_function("chi2_cdf_100k", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for &x in &chi2_xs {
                sum += chi2.cdf(black_box(x));
            }
            black_box(sum)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Optimizer benchmarks
// ---------------------------------------------------------------------------

fn optimizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer");

    // Rosenbrock from (-1, 1)
    group.bench_function("rosenbrock_trust_ncg", |b| {
        b.iter(|| {
            let fun_grad = |x: &Array1<f64>| {
                let a = 1.0 - x[0];
                let b_val = x[1] - x[0] * x[0];
                let f = a * a + 100.0 * b_val * b_val;
                let g = array![-2.0 * a - 400.0 * x[0] * b_val, 200.0 * b_val];
                (f, g)
            };
            let hessp = |x: &Array1<f64>, p: &Array1<f64>| {
                let h00 = 2.0 - 400.0 * (x[1] - x[0] * x[0]) + 800.0 * x[0] * x[0];
                let h01 = -400.0 * x[0];
                let h11 = 200.0;
                array![h00 * p[0] + h01 * p[1], h01 * p[0] + h11 * p[1]]
            };
            let x0 = array![-1.0, 1.0];
            black_box(
                TrustRegionNCG::new()
                    .with_max_iter(500)
                    .with_tol(1e-10)
                    .minimize(fun_grad, hessp, x0)
                    .unwrap(),
            )
        });
    });

    // 50-dimensional diagonal quadratic: f(x) = 0.5 * sum_i (i+1)*x_i^2
    group.bench_function("quadratic_50d", |b| {
        b.iter(|| {
            let n = 50;
            let diag: Array1<f64> = (1..=n).map(|i| i as f64).collect();
            let diag2 = diag.clone();

            let fun_grad = move |x: &Array1<f64>| {
                let ax = &diag * x;
                let f_val = 0.5 * x.dot(&ax);
                let g = ax;
                (f_val, g)
            };
            let hessp = move |_x: &Array1<f64>, p: &Array1<f64>| &diag2 * p;

            let x0 = Array1::from_elem(n, 5.0);
            black_box(TrustRegionNCG::new().minimize(fun_grad, hessp, x0).unwrap())
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Interpolation benchmarks
// ---------------------------------------------------------------------------

fn interpolation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    // 1000 sin(x) data points on [0, 2*pi]
    let n_build = 1000;
    let x_build: Vec<f64> = (0..n_build)
        .map(|i| 2.0 * std::f64::consts::PI * f64::from(i) / (f64::from(n_build) - 1.0))
        .collect();
    let y_build: Vec<f64> = x_build.iter().map(|&x| x.sin()).collect();

    group.bench_function("spline_build_1000", |b| {
        b.iter(|| {
            black_box(
                CubicSpline::new(
                    black_box(&x_build),
                    black_box(&y_build),
                    BoundaryCondition::Natural,
                )
                .unwrap(),
            )
        });
    });

    // Evaluate at 10000 points
    let spline_1000 = CubicSpline::new(&x_build, &y_build, BoundaryCondition::Natural).unwrap();
    let n_eval = 10_000;
    let x_eval: Vec<f64> = (0..n_eval)
        .map(|i| 2.0 * std::f64::consts::PI * f64::from(i) / (f64::from(n_eval) - 1.0))
        .collect();

    group.bench_function("spline_eval_10000", |b| {
        b.iter(|| black_box(spline_1000.eval_array(black_box(&x_eval))));
    });

    // Build from 10000 points
    let n_build_large = 10_000;
    let x_build_large: Vec<f64> = (0..n_build_large)
        .map(|i| 2.0 * std::f64::consts::PI * f64::from(i) / (f64::from(n_build_large) - 1.0))
        .collect();
    let y_build_large: Vec<f64> = x_build_large.iter().map(|&x| x.sin()).collect();

    group.bench_function("spline_build_10000", |b| {
        b.iter(|| {
            black_box(
                CubicSpline::new(
                    black_box(&x_build_large),
                    black_box(&y_build_large),
                    BoundaryCondition::Natural,
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Quadrature benchmarks
// ---------------------------------------------------------------------------

fn quadrature_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadrature");

    // Adaptive Simpson: integral of sin(x) from 0 to pi = 2
    group.bench_function("quad_sin", |b| {
        b.iter(|| black_box(quad(f64::sin, 0.0, std::f64::consts::PI, 1e-10)));
    });

    // Adaptive Simpson: integral of exp(-x^2) from -5 to 5 ~ sqrt(pi)
    group.bench_function("quad_gaussian", |b| {
        b.iter(|| black_box(quad(|x| (-x * x).exp(), -5.0, 5.0, 1e-10)));
    });

    // Gauss-Legendre 10-point on sin integral
    group.bench_function("gauss_legendre_10_sin", |b| {
        b.iter(|| black_box(gauss_legendre(f64::sin, 0.0, std::f64::consts::PI, 10).unwrap()));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    sparse_eig_benchmarks,
    sparse_graph_benchmarks,
    distribution_benchmarks,
    optimizer_benchmarks,
    interpolation_benchmarks,
    quadrature_benchmarks
);
criterion_main!(benches);
