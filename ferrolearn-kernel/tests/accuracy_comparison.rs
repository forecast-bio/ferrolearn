//! Accuracy comparison tests — Rust vs Python numerical agreement.
//!
//! Each test reproduces a Python accuracy scenario and verifies that
//! Rust matches or exceeds Python's accuracy against known ground truth.

use ndarray::{Array1, Array2};

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::bandwidth::BandwidthStrategy;
use ferrolearn_kernel::kernels::GaussianKernel;
use ferrolearn_kernel::local_polynomial::LocalPolynomialRegression;
use ferrolearn_kernel::weights;

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

// =========================================================================
// Scenario 1: NW on sin(x) with deterministic noise
// =========================================================================

// Noise generator uses 2.718 as a deterministic seed (matches the Python reference
// constants below); not intended as the E constant. Hence `clippy::approx_constant`.
#[allow(clippy::approx_constant)]
#[test]
fn nw_sin_accuracy_vs_ground_truth() {
    let n = 200;
    let x_vec = linspace(0.0, 2.0 * std::f64::consts::PI, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y_true: Array1<f64> = Array1::from_vec(x_vec.iter().map(|&xi| xi.sin()).collect());
    let y_noisy: Array1<f64> = y_true
        .iter()
        .enumerate()
        .map(|(i, &yt)| yt + 0.05 * (i as f64 * 2.718).sin())
        .collect();

    let bw = ndarray::array![0.3f64];
    let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
    let pred = weights::nw_predict_from_weights(&w, &y_noisy);

    let mae: f64 = pred
        .iter()
        .zip(y_true.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64;
    let max_err: f64 = pred
        .iter()
        .zip(y_true.iter())
        .map(|(p, t)| (p - t).abs())
        .fold(0.0f64, f64::max);

    // Python: MAE = 0.04121, max_err = 0.22317
    // Rust should match within floating-point tolerance
    let python_mae = 0.04121210774076802;
    let python_max = 0.22317327274515178;

    assert!(
        (mae - python_mae).abs() < 1e-10,
        "NW sin MAE: Rust={mae:.12e} Python={python_mae:.12e} diff={:.2e}",
        (mae - python_mae).abs()
    );
    assert!(
        (max_err - python_max).abs() < 1e-10,
        "NW sin max_err: Rust={max_err:.12e} Python={python_max:.12e} diff={:.2e}",
        (max_err - python_max).abs()
    );
}

// =========================================================================
// Scenario 2: LPR order 1 on sin(x)
// =========================================================================

#[allow(clippy::approx_constant)]
#[test]
fn lpr_sin_accuracy_vs_ground_truth() {
    let n = 200;
    let x_vec = linspace(0.0, 2.0 * std::f64::consts::PI, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y_true: Array1<f64> = Array1::from_vec(x_vec.iter().map(|&xi| xi.sin()).collect());
    let y_noisy: Array1<f64> = y_true
        .iter()
        .enumerate()
        .map(|(i, &yt)| yt + 0.05 * (i as f64 * 2.718).sin())
        .collect();

    let lpr =
        LocalPolynomialRegression::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.3), 1);
    let fitted = lpr.fit(&x, &y_noisy).unwrap();
    let pred = fitted.predict(&x).unwrap();

    let mae: f64 = pred
        .iter()
        .zip(y_true.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64;
    let max_err: f64 = pred
        .iter()
        .zip(y_true.iter())
        .map(|(p, t)| (p - t).abs())
        .fold(0.0f64, f64::max);

    // Python: MAE = 0.02717, max_err = 0.04400
    let python_mae = 0.02717294511744801;
    let python_max = 0.04400109288303855;

    // LPR solvers may differ slightly — allow 1e-6 tolerance
    assert!(
        (mae - python_mae).abs() < 1e-6,
        "LPR sin MAE: Rust={mae:.10e} Python={python_mae:.10e} diff={:.2e}",
        (mae - python_mae).abs()
    );
    assert!(
        (max_err - python_max).abs() < 1e-5,
        "LPR sin max_err: Rust={max_err:.10e} Python={python_max:.10e} diff={:.2e}",
        (max_err - python_max).abs()
    );
}

// =========================================================================
// Scenario 3: NW boundary bias on y=x
// =========================================================================

#[test]
fn nw_linear_boundary_bias_matches_python() {
    let n = 200;
    let x_vec = linspace(0.0, 1.0, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y: Array1<f64> = Array1::from_vec(x_vec);

    let python_results = [
        (
            0.05,
            0.03832403996395557,
            -0.038324039963955614,
            0.0034470730731548916,
        ),
        (
            0.1,
            0.07820388320521897,
            -0.07820388320521876,
            0.013792285682146544,
        ),
        (
            0.2,
            0.15798445863928115,
            -0.15798445863928123,
            0.05421375272106817,
        ),
    ];

    for &(bw_val, py_bias_left, py_bias_right, py_mae) in &python_results {
        let bw = ndarray::array![bw_val];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let bias_left = pred[0] - y[0];
        let bias_right = pred[n - 1] - y[n - 1];
        let mae: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n as f64;

        assert!(
            (bias_left - py_bias_left).abs() < 1e-10,
            "bw={bw_val} bias_left: Rust={bias_left:.12e} Python={py_bias_left:.12e}"
        );
        assert!(
            (bias_right - py_bias_right).abs() < 1e-10,
            "bw={bw_val} bias_right: Rust={bias_right:.12e} Python={py_bias_right:.12e}"
        );
        assert!(
            (mae - py_mae).abs() < 1e-10,
            "bw={bw_val} MAE: Rust={mae:.12e} Python={py_mae:.12e}"
        );
    }
}

// =========================================================================
// Scenario 4: LPR on linear data — should be near-exact
// =========================================================================

#[test]
fn lpr_linear_near_exact() {
    let n = 200;
    let x_vec = linspace(0.0, 1.0, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y: Array1<f64> = Array1::from_vec(x_vec);

    // Python max errors: 4.2e-11, 2.4e-11, 1.3e-11 for bw=0.05, 0.1, 0.2
    for bw_val in [0.05, 0.1, 0.2] {
        let lpr = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(bw_val),
            1,
        );
        let fitted = lpr.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        let max_err: f64 = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0f64, f64::max);

        // Both should be near machine epsilon — allow 1e-8
        assert!(
            max_err < 1e-8,
            "LPR linear bw={bw_val}: max_err={max_err:.2e} should be < 1e-8"
        );
    }
}

// =========================================================================
// Scenario 5: NW on quadratic — vertex bias
// =========================================================================

#[test]
fn nw_quadratic_vertex_bias_matches_python() {
    let n = 200;
    let x_vec = linspace(-2.0, 2.0, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y: Array1<f64> = Array1::from_vec(x_vec.iter().map(|&xi| xi * xi).collect());

    let bw = ndarray::array![0.3f64];
    let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
    let pred = weights::nw_predict_from_weights(&w, &y);

    let vertex_bias = pred[n / 2] - y[n / 2];
    let mae: f64 = pred
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64;

    // Python: vertex_bias = 0.09000, MAE = 0.14897
    let py_vertex_bias = 0.08999999991255508;
    let py_mae = 0.14896803602061237;

    assert!(
        (vertex_bias - py_vertex_bias).abs() < 1e-8,
        "Vertex bias: Rust={vertex_bias:.12e} Python={py_vertex_bias:.12e}"
    );
    assert!(
        (mae - py_mae).abs() < 1e-8,
        "Quadratic MAE: Rust={mae:.12e} Python={py_mae:.12e}"
    );
}

// =========================================================================
// Scenario 6: LPR order 2 on quadratic — should be near-exact
// =========================================================================

#[test]
fn lpr_quadratic_order2_near_exact() {
    let n = 200;
    let x_vec = linspace(-2.0, 2.0, n);
    let x = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();
    let y: Array1<f64> = Array1::from_vec(x_vec.iter().map(|&xi| xi * xi).collect());

    let lpr =
        LocalPolynomialRegression::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.3), 2);
    let fitted = lpr.fit(&x, &y).unwrap();
    let pred = fitted.predict(&x).unwrap();

    let vertex_bias = (pred[n / 2] - y[n / 2]).abs();
    let mae: f64 = pred
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64;

    // Python: vertex_bias = 1.1e-11, MAE = 3.1e-11
    // Both should be near machine epsilon
    assert!(
        vertex_bias < 1e-8,
        "LPR order 2 vertex bias={vertex_bias:.2e} should be < 1e-8"
    );
    assert!(mae < 1e-8, "LPR order 2 MAE={mae:.2e} should be < 1e-8");
}

// =========================================================================
// Scenario 7: Cross-implementation ULP on fixture data
// =========================================================================

#[test]
fn fixture_nw_ulp_agreement() {
    // Load the fixture and verify Rust matches Python at ULP level
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        #[serde(rename = "X")]
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        predictions: Vec<f64>,
        bandwidth: Vec<f64>,
    }

    let path = format!(
        "{}/fixtures/nw_gaussian_sin.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read_to_string(&path).unwrap();
    let fix: Fixture = serde_json::from_str(&data).unwrap();

    let n = fix.x.len();
    let flat: Vec<f64> = fix.x.iter().flat_map(|r| r.iter().copied()).collect();
    let x = Array2::from_shape_vec((n, 1), flat).unwrap();
    let y = Array1::from_vec(fix.y);
    let bw = Array1::from_vec(fix.bandwidth);

    let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
    let pred = weights::nw_predict_from_weights(&w, &y);

    let python_pred = Array1::from_vec(fix.predictions);

    let max_diff: f64 = pred
        .iter()
        .zip(python_pred.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f64, f64::max);
    let mean_diff: f64 = pred
        .iter()
        .zip(python_pred.iter())
        .map(|(r, p)| (r - p).abs())
        .sum::<f64>()
        / n as f64;

    // Count ULPs
    let mut max_ulps: u64 = 0;
    for (&r, &p) in pred.iter().zip(python_pred.iter()) {
        let diff_ulps = (r.to_bits() as i64 - p.to_bits() as i64).unsigned_abs();
        max_ulps = max_ulps.max(diff_ulps);
    }

    // Report
    eprintln!("Fixture NW ULP analysis:");
    eprintln!("  Max absolute diff: {max_diff:.2e}");
    eprintln!("  Mean absolute diff: {mean_diff:.2e}");
    eprintln!("  Max ULP distance: {max_ulps}");

    // NW is identical algorithm — should match to near machine precision
    assert!(
        max_diff < 1e-12,
        "Fixture max diff {max_diff:.2e} exceeds 1e-12"
    );
}
