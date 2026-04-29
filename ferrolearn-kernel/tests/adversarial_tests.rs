//! Adversarial tests for ferrolearn-kernel.
//!
//! These tests exercise corner cases and failure modes:
//! 1. Boundary bias trap — NW biases at edges, LPR corrects
//! 2. Heteroscedasticity ghost — White test detects non-constant variance
//! 3. Curse of irrelevance — adding noise dimensions degrades prediction
//! 4. Matrix kill — collinear features don't crash LPR

use ndarray::{Array1, Array2, array};

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::bandwidth::BandwidthStrategy;
use ferrolearn_kernel::diagnostics::{HeteroscedasticityTest, heteroscedasticity_test};
use ferrolearn_kernel::kernels::GaussianKernel;
use ferrolearn_kernel::local_polynomial::LocalPolynomialRegression;
use ferrolearn_kernel::nadaraya_watson::NadarayaWatson;
use ferrolearn_kernel::weights;

/// Test 1: Boundary bias trap.
///
/// On y = x (linear), NW pulls predictions toward the global mean at
/// boundaries (x=0 and x=1). LPR order 1 should have much less bias.
#[test]
fn boundary_bias_trap() {
    let n = 100;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
    let y: Array1<f64> = x.column(0).to_owned();

    // NW
    let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.15));
    let fitted_nw = nw.fit(&x, &y).unwrap();
    let pred_nw = fitted_nw.predict(&x).unwrap();

    // LPR order 1
    let lpr =
        LocalPolynomialRegression::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.15), 1);
    let fitted_lpr = lpr.fit(&x, &y).unwrap();
    let pred_lpr = fitted_lpr.predict(&x).unwrap();

    // NW boundary bias: at x=0, prediction pulled above 0; at x=1, pulled below 1
    let nw_bias_left = (pred_nw[0] - y[0]).abs();
    let nw_bias_right = (pred_nw[n - 1] - y[n - 1]).abs();
    let nw_boundary_bias = nw_bias_left + nw_bias_right;

    let lpr_bias_left = (pred_lpr[0] - y[0]).abs();
    let lpr_bias_right = (pred_lpr[n - 1] - y[n - 1]).abs();
    let lpr_boundary_bias = lpr_bias_left + lpr_bias_right;

    assert!(
        nw_boundary_bias > 0.05,
        "NW should have substantial boundary bias ({nw_boundary_bias:.4})"
    );
    assert!(
        lpr_boundary_bias < nw_boundary_bias * 0.5,
        "LPR boundary bias ({lpr_boundary_bias:.4}) should be much less than NW ({nw_boundary_bias:.4})"
    );
}

/// Test 2: Heteroscedasticity ghost.
///
/// Generate data where variance grows with x. The White test should
/// detect this and reject homoscedasticity.
#[test]
fn heteroscedasticity_ghost() {
    let n = 200;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
    let x = Array2::from_shape_vec((n, 1), x_data).unwrap();

    // y = sin(x) + x * deterministic_noise
    let y: Array1<f64> = x
        .column(0)
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + xi * 0.3 * (i as f64 * 2.7).sin())
        .collect();

    let bw = array![0.5f64];
    let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
    let pred = weights::nw_predict_from_weights(&w, &y);

    let result =
        heteroscedasticity_test(&x, &y, &pred, HeteroscedasticityTest::White, 0.05).unwrap();

    // White test should reject homoscedasticity (p < 0.05)
    assert!(
        result.p_value < 0.05,
        "White test p={:.4} should be < 0.05 for heteroscedastic data",
        result.p_value
    );
    assert!(
        result.is_heteroscedastic,
        "Should reject null of homoscedasticity"
    );
}

/// Test 3: Noise variable robustness.
///
/// With 1 signal variable and 5 noise variables, the model should
/// still produce finite, reasonable predictions. Follows the Python
/// package's approach — the key check is graceful degradation.
#[test]
fn noise_variable_robustness() {
    let n = 200;

    // 1 signal variable + 5 noise variables
    let mut x_data: Vec<f64> = Vec::with_capacity(n * 6);
    for i in 0..n {
        let signal = (i as f64 / n as f64) * 4.0 - 2.0; // [-2, 2]
        x_data.push(signal);
        for d in 0..5 {
            // Deterministic pseudo-random noise
            let noise = ((i * 97 + d * 53 + 7) as f64 * 0.618033988).fract() * 4.0 - 2.0;
            x_data.push(noise);
        }
    }
    let x = Array2::from_shape_vec((n, 6), x_data).unwrap();
    // y depends only on the first variable
    let y: Array1<f64> = x.column(0).mapv(f64::sin);

    let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Silverman);
    let fitted = nw.fit(&x, &y).unwrap();
    let pred = fitted.predict(&x).unwrap();

    // All predictions should be finite
    for &p in &pred {
        assert!(p.is_finite(), "Prediction should be finite, got {p}");
    }

    // Should still capture some signal (R² > 0)
    let y_mean = y.mean().unwrap();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(pred.iter())
        .map(|(&yi, &pi)| (yi - pi).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    assert!(
        r2 > 0.0,
        "R² ({r2:.4}) should be positive — model should capture some signal"
    );
}

/// Test 4: Matrix kill — collinear features.
///
/// When features are perfectly collinear, LPR should still produce
/// finite predictions (via Tikhonov regularization), not crash or NaN.
#[test]
fn matrix_kill_collinear() {
    let n = 50;
    // x1 = linear, x2 = 2*x1 (perfectly collinear)
    let mut x_data: Vec<f64> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let val = i as f64 * 0.1;
        x_data.push(val);
        x_data.push(val * 2.0);
    }
    let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
    let y: Array1<f64> = x.column(0).mapv(f64::sin);

    let lpr = LocalPolynomialRegression::with_kernel(
        GaussianKernel,
        BandwidthStrategy::PerDimension(array![0.5, 1.0]),
        1,
    );
    let fitted = lpr.fit(&x, &y).unwrap();
    let pred = fitted.predict(&x).unwrap();

    // All predictions should be finite
    for (i, &p) in pred.iter().enumerate() {
        assert!(
            p.is_finite(),
            "Prediction at index {i} should be finite, got {p}"
        );
    }

    // Should also be reasonable (within the range of y)
    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let margin = (y_max - y_min) * 0.5;
    for &p in &pred {
        assert!(
            p >= y_min - margin && p <= y_max + margin,
            "Prediction {p:.4} outside reasonable range [{:.4}, {:.4}]",
            y_min - margin,
            y_max + margin
        );
    }
}

/// Test 5: Constant data — predictions should equal the constant.
#[test]
fn constant_data_exact() {
    let n = 50;
    let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64 * 0.1).collect()).unwrap();
    let y: Array1<f64> = Array1::from_elem(n, 42.0);

    let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
    let fitted = nw.fit(&x, &y).unwrap();
    let pred = fitted.predict(&x).unwrap();

    for (i, &p) in pred.iter().enumerate() {
        assert!(
            (p - 42.0).abs() < 1e-10,
            "Prediction at {i} should be 42.0, got {p}"
        );
    }
}

/// Test 6: Single training point — should predict that point everywhere.
#[test]
fn single_training_point() {
    let x_train = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let y_train = array![7.0f64];
    let bw = array![1.0f64];

    let w = weights::compute_kernel_weights(&x_train, &x_train, &bw, &GaussianKernel);
    let pred = weights::nw_predict_from_weights(&w, &y_train);

    assert!(
        (pred[0] - 7.0).abs() < 1e-10,
        "Single-point prediction should be 7.0, got {}",
        pred[0]
    );
}
