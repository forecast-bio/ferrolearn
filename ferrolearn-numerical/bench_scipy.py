#!/usr/bin/env python3
"""
Benchmark and accuracy reference script for ferrolearn-numerical.

Exercises scipy equivalents of every module in ferrolearn-numerical and
produces timing data plus reference values that Rust tests compare against.

Usage:
    python bench_scipy.py

Requires: numpy, scipy
"""

import time
import json
import statistics
import numpy as np
from scipy.sparse import diags, random as sparse_random
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import dijkstra, connected_components, minimum_spanning_tree
from scipy.stats import norm, chi2, f as f_dist, t as t_dist, beta, gamma
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, fixed_quad

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
NUM_ITERATIONS = 10


def _fmt_list(arr, precision=12):
    """Format a list/array of floats for output."""
    return "[" + ", ".join(f"{v:.{precision}g}" for v in arr) + "]"


def _fmt_dict(d, precision=12):
    """Format a dict of floats for output."""
    items = []
    for k, v in d.items():
        items.append(f'"{k}": {v:.{precision}g}')
    return "{" + ", ".join(items) + "}"


def benchmark(func, iterations=NUM_ITERATIONS):
    """Run func `iterations` times and return median elapsed time in ms."""
    times = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
    median_ms = statistics.median(times)
    return median_ms, result


# ===========================================================================
# 1. Sparse Eigensolver
# ===========================================================================

def bench_sparse_eigensolver():
    print("=== SPARSE EIGENSOLVER ===")

    def make_tridiag(n):
        """1, -2, 1 symmetric tridiagonal matrix."""
        diagonals = [
            np.ones(n - 1),       # super-diagonal
            -2.0 * np.ones(n),    # main diagonal
            np.ones(n - 1),       # sub-diagonal
        ]
        return diags(diagonals, offsets=[1, 0, -1], format="csc")

    # --- 1000x1000 ---
    mat_1000 = make_tridiag(1000)

    ms, vals_sa_1000 = benchmark(lambda: eigsh(mat_1000, k=10, which="SA")[0])
    vals_sa_1000 = np.sort(vals_sa_1000)
    print(f"eigsh_1000_SA: time_ms={ms:.2f}")

    ms, vals_la_1000 = benchmark(lambda: eigsh(mat_1000, k=10, which="LA")[0])
    vals_la_1000 = np.sort(vals_la_1000)
    print(f"eigsh_1000_LA: time_ms={ms:.2f}")

    # --- 5000x5000 ---
    mat_5000 = make_tridiag(5000)

    ms, _ = benchmark(lambda: eigsh(mat_5000, k=10, which="SA")[0])
    print(f"eigsh_5000_SA: time_ms={ms:.2f}")

    # Reference eigenvalues
    print(f"eigsh_1000_SA_eigenvalues: {_fmt_list(vals_sa_1000)}")
    print(f"eigsh_1000_LA_eigenvalues: {_fmt_list(vals_la_1000)}")
    print()


# ===========================================================================
# 2. Sparse Graph
# ===========================================================================

def bench_sparse_graph():
    print("=== SPARSE GRAPH ===")

    rng = np.random.default_rng(SEED)
    n = 1000
    density = 0.05

    # Build a sparse random graph (symmetric for undirected).
    # Create random sparse matrix, then symmetrise.
    rows, cols, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.uniform(0.1, 10.0)
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([w, w])

    from scipy.sparse import csr_matrix
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Single-source Dijkstra
    ms, dists_single = benchmark(lambda: dijkstra(graph, indices=0))
    print(f"dijkstra_single_1000: time_ms={ms:.2f}")

    # All-pairs Dijkstra
    ms, _ = benchmark(lambda: dijkstra(graph), iterations=NUM_ITERATIONS)
    print(f"dijkstra_all_1000: time_ms={ms:.2f}")

    # Connected components
    ms, cc_result = benchmark(lambda: connected_components(graph, directed=False))
    n_components = cc_result[0]
    print(f"connected_components_1000: time_ms={ms:.2f}")
    print(f"connected_components_count: {n_components}")

    # Minimum spanning tree
    ms, mst = benchmark(lambda: minimum_spanning_tree(graph))
    print(f"mst_1000: time_ms={ms:.2f}")

    # Reference distances: first 20 from source 0
    d0 = dists_single.ravel()[:20]
    print(f"dijkstra_distances_0_20: {_fmt_list(d0)}")
    print()


# ===========================================================================
# 3. Distributions
# ===========================================================================

def bench_distributions():
    print("=== DISTRIBUTIONS ===")

    rng = np.random.default_rng(SEED)
    n = 100_000

    # Distribution configs: (name, frozen_dist, sample_points_for_pdf_cdf, sample_points_for_ppf)
    distributions = [
        ("norm",  norm(0, 1),    rng.standard_normal(n),              rng.uniform(0.001, 0.999, n)),
        ("chi2",  chi2(5),       np.abs(rng.standard_normal(n)) * 5,  rng.uniform(0.001, 0.999, n)),
        ("f",     f_dist(3, 40), np.abs(rng.standard_normal(n)) * 3,  rng.uniform(0.001, 0.999, n)),
        ("t",     t_dist(10),    rng.standard_normal(n) * 3,          rng.uniform(0.001, 0.999, n)),
        ("beta",  beta(2, 5),    rng.uniform(0.001, 0.999, n),        rng.uniform(0.001, 0.999, n)),
        ("gamma", gamma(3, scale=2), np.abs(rng.standard_normal(n)) * 5, rng.uniform(0.001, 0.999, n)),
    ]

    for name, dist, x_vals, p_vals in distributions:
        ms_pdf, _ = benchmark(lambda d=dist, x=x_vals: d.pdf(x))
        print(f"pdf_100k_{name}: time_ms={ms_pdf:.2f}")

        ms_cdf, _ = benchmark(lambda d=dist, x=x_vals: d.cdf(x))
        print(f"cdf_100k_{name}: time_ms={ms_cdf:.2f}")

        ms_ppf, _ = benchmark(lambda d=dist, p=p_vals: d.ppf(p))
        print(f"ppf_100k_{name}: time_ms={ms_ppf:.2f}")

    # Reference values
    ref = {}
    ref["norm_pdf_0"]       = norm.pdf(0)
    ref["norm_cdf_1.96"]    = norm.cdf(1.96)
    ref["norm_ppf_0.975"]   = norm.ppf(0.975)
    ref["chi2_sf_11.07_5"]  = chi2.sf(11.07, 5)
    ref["f_sf_2.84_3_40"]   = f_dist.sf(2.84, 3, 40)
    ref["t_cdf_2.228_10"]   = t_dist.cdf(2.228, 10)
    ref["beta_mean_2_5"]    = beta.mean(2, 5)
    ref["beta_var_2_5"]     = beta.var(2, 5)
    ref["gamma_ppf_0.95_3_0.5"] = gamma.ppf(0.95, 3, scale=0.5)

    print(f"reference_values: {_fmt_dict(ref)}")
    print()


# ===========================================================================
# 4. Optimizer
# ===========================================================================

def bench_optimizer():
    print("=== OPTIMIZER ===")

    # --- Rosenbrock ---
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def rosenbrock_jac(x):
        dx = -2 * (1 - x[0]) + 100 * 2 * (x[1] - x[0] ** 2) * (-2 * x[0])
        dy = 100 * 2 * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    def rosenbrock_hessp(x, p):
        """Hessian-vector product for Rosenbrock."""
        h11 = 2 - 400 * x[1] + 1200 * x[0] ** 2
        h12 = -400 * x[0]
        h22 = 200.0
        return np.array([
            h11 * p[0] + h12 * p[1],
            h12 * p[0] + h22 * p[1],
        ])

    x0_rosen = np.array([-1.0, 1.0])

    ms, res_rosen = benchmark(
        lambda: minimize(
            rosenbrock, x0_rosen, method="trust-ncg",
            jac=rosenbrock_jac, hessp=rosenbrock_hessp,
        )
    )
    print(f"trust_ncg_rosenbrock: time_ms={ms:.2f}")

    # --- 50-dimensional quadratic ---
    diag_vals = np.arange(1, 51, dtype=np.float64)

    def quad50(x):
        return 0.5 * np.sum(diag_vals * x ** 2)

    def quad50_jac(x):
        return diag_vals * x

    def quad50_hessp(x, p):
        return diag_vals * p

    x0_quad = np.ones(50)

    ms, res_quad = benchmark(
        lambda: minimize(
            quad50, x0_quad, method="trust-ncg",
            jac=quad50_jac, hessp=quad50_hessp,
        )
    )
    print(f"trust_ncg_quad50: time_ms={ms:.2f}")

    # Reference solutions
    print(f"rosenbrock_solution: {_fmt_list(res_rosen.x)}")
    print(f"quad50_solution_norm: {np.linalg.norm(res_quad.x):.12g}")
    print()


# ===========================================================================
# 5. Interpolation
# ===========================================================================

def bench_interpolation():
    print("=== INTERPOLATION ===")

    # --- 1000 data points ---
    x_1000 = np.linspace(0, 2 * np.pi, 1000)
    y_1000 = np.sin(x_1000)

    ms_build_1000, cs_1000 = benchmark(lambda: CubicSpline(x_1000, y_1000))
    print(f"cubicspline_build_1000: time_ms={ms_build_1000:.2f}")

    x_eval = np.linspace(0, 2 * np.pi, 10000)
    ms_eval, _ = benchmark(lambda: cs_1000(x_eval))
    print(f"cubicspline_eval_10000: time_ms={ms_eval:.2f}")

    # --- 10000 data points ---
    x_10000 = np.linspace(0, 2 * np.pi, 10000)
    y_10000 = np.sin(x_10000)

    ms_build_10000, _ = benchmark(lambda: CubicSpline(x_10000, y_10000))
    print(f"cubicspline_build_10000: time_ms={ms_build_10000:.2f}")

    # Reference predictions at 20 evenly-spaced test points
    x_test_20 = np.linspace(0, 2 * np.pi, 20)
    preds_20 = cs_1000(x_test_20)
    print(f"cubicspline_predictions_20: {_fmt_list(preds_20)}")
    print()


# ===========================================================================
# 6. Quadrature
# ===========================================================================

def bench_quadrature():
    print("=== QUADRATURE ===")

    # --- quad(sin, 0, pi) ---
    ms, (val_sin, err_sin) = benchmark(lambda: quad(np.sin, 0, np.pi))
    print(f"quad_sin: time_ms={ms:.2f}, value={val_sin:.15g}, error={err_sin:.6g}")

    # --- quad(exp(-x^2), -5, 5) ---
    ms, (val_gauss, err_gauss) = benchmark(
        lambda: quad(lambda x: np.exp(-x ** 2), -5, 5)
    )
    print(f"quad_gaussian: time_ms={ms:.2f}, value={val_gauss:.15g}, error={err_gauss:.6g}")

    # --- quad(x^10, 0, 1) ---
    ms, (val_poly, err_poly) = benchmark(
        lambda: quad(lambda x: x ** 10, 0, 1)
    )
    print(f"quad_polynomial: time_ms={ms:.2f}, value={val_poly:.15g}, error={err_poly:.6g}")

    # --- fixed_quad (Gauss-Legendre) for comparison ---
    ms_fq_sin, (val_fq_sin, _) = benchmark(
        lambda: fixed_quad(np.sin, 0, np.pi, n=50)
    )
    print(f"fixed_quad_sin_n50: time_ms={ms_fq_sin:.2f}, value={val_fq_sin:.15g}")

    ms_fq_gauss, (val_fq_gauss, _) = benchmark(
        lambda: fixed_quad(lambda x: np.exp(-x ** 2), -5, 5, n=50)
    )
    print(f"fixed_quad_gaussian_n50: time_ms={ms_fq_gauss:.2f}, value={val_fq_gauss:.15g}")

    ms_fq_poly, (val_fq_poly, _) = benchmark(
        lambda: fixed_quad(lambda x: x ** 10, 0, 1, n=50)
    )
    print(f"fixed_quad_polynomial_n50: time_ms={ms_fq_poly:.2f}, value={val_fq_poly:.15g}")

    # Exact reference values for verification
    print(f"exact_sin_0_pi: {2.0:.15g}")
    print(f"exact_gaussian_neg5_5: {np.sqrt(np.pi):.15g}")
    print(f"exact_polynomial_x10_0_1: {1.0 / 11.0:.15g}")
    print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    np.random.seed(SEED)

    print(f"NumPy version: {np.__version__}")
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    print(f"Seed: {SEED}")
    print(f"Iterations per benchmark: {NUM_ITERATIONS}")
    print()

    bench_sparse_eigensolver()
    bench_sparse_graph()
    bench_distributions()
    bench_optimizer()
    bench_interpolation()
    bench_quadrature()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
