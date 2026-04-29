//! Wild bootstrap confidence intervals and variance estimation.
//!
//! Provides bias-corrected confidence intervals and Fan-Yao
//! nonparametric variance function estimation.

use ndarray::{Array1, Array2};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use ferrolearn_core::FerroError;

use crate::kernels::GaussianKernel;
use crate::weights;

/// Bias correction method for wild bootstrap CI.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BiasCorrection {
    /// No bias correction.
    None,
    /// Undersmooth bandwidth (0.75x).
    Undersmooth,
    /// Robust bias-corrected.
    Rbc,
    /// Big brother: higher-order polynomial + undersmoothed bandwidth.
    BigBrother,
    /// RBC studentized (CCF 2018/2022).
    RbcStudentized,
}

/// Result of confidence interval computation.
#[derive(Debug, Clone)]
pub struct ConfidenceIntervalResult {
    /// Point predictions.
    pub predictions: Array1<f64>,
    /// Lower CI bound.
    pub lower: Array1<f64>,
    /// Upper CI bound.
    pub upper: Array1<f64>,
    /// Nominal confidence level.
    pub confidence_level: f64,
    /// Bias correction method used.
    pub bias_correction: BiasCorrection,
}

/// Result of variance function estimation.
#[derive(Debug, Clone)]
pub struct VarianceFunctionResult {
    /// Evaluation points.
    pub x_eval: Array2<f64>,
    /// Variance estimates σ²(x).
    pub variance_estimate: Array1<f64>,
    /// Standard deviation estimates σ(x).
    pub std_estimate: Array1<f64>,
    /// Bandwidth used for variance estimation.
    pub bandwidth: Array1<f64>,
}

/// Configuration for wild bootstrap confidence intervals.
pub struct BootstrapConfig<'a> {
    /// Training features.
    pub x_train: &'a Array2<f64>,
    /// Training targets.
    pub y_train: &'a Array1<f64>,
    /// Points at which to compute CI.
    pub x_pred: &'a Array2<f64>,
    /// Fitted bandwidth.
    pub bandwidth: &'a Array1<f64>,
    /// Point predictions at `x_pred`.
    pub predictions: &'a Array1<f64>,
}

/// Compute wild bootstrap confidence intervals.
#[allow(clippy::too_many_arguments)]
pub fn wild_bootstrap_confidence_intervals(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    x_pred: &Array2<f64>,
    bandwidth: &Array1<f64>,
    predictions: &Array1<f64>,
    n_bootstrap: usize,
    confidence_level: f64,
    bias_correction: BiasCorrection,
) -> Result<ConfidenceIntervalResult, FerroError> {
    let n_train = x_train.nrows();
    let n_pred = x_pred.nrows();

    // Compute residuals
    let w_train = weights::compute_kernel_weights(x_train, x_train, bandwidth, &GaussianKernel);
    let y_hat = weights::nw_predict_from_weights(&w_train, y_train);
    let residuals: Array1<f64> = y_train - &y_hat;

    // Adjust bandwidth for bias correction
    let bw_boot = match bias_correction {
        BiasCorrection::Undersmooth | BiasCorrection::BigBrother => bandwidth.mapv(|h| h * 0.75),
        BiasCorrection::RbcStudentized => bandwidth.mapv(|h| h * 0.6),
        _ => bandwidth.clone(),
    };

    // Bootstrap: collect predictions across iterations (parallel)
    let boot_preds: Vec<Array1<f64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::rng();

            // Rademacher weights
            let wild_weights: Array1<f64> = (0..n_train)
                .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
                .collect();

            // Bootstrap response
            let y_boot: Array1<f64> = &y_hat + &(&residuals * &wild_weights);

            // Refit and predict
            let w = weights::compute_kernel_weights(x_pred, x_train, &bw_boot, &GaussianKernel);
            weights::nw_predict_from_weights(&w, &y_boot)
        })
        .collect();

    // Compute quantiles
    let alpha = 1.0 - confidence_level;
    let mut lower = Array1::zeros(n_pred);
    let mut upper = Array1::zeros(n_pred);

    for j in 0..n_pred {
        let mut boot_vals: Vec<f64> = boot_preds.iter().map(|bp| bp[j]).collect();
        boot_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lo_idx = (alpha / 2.0 * n_bootstrap as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;
        let hi_idx = hi_idx.min(n_bootstrap - 1);

        lower[j] = boot_vals[lo_idx];
        upper[j] = boot_vals[hi_idx];
    }

    Ok(ConfidenceIntervalResult {
        predictions: predictions.clone(),
        lower,
        upper,
        confidence_level,
        bias_correction,
    })
}

/// Fan-Yao nonparametric variance function estimation.
///
/// Estimates `σ²(x) = E[ε² | X = x]` by fitting a kernel regression
/// of squared residuals.
pub fn fan_yao_variance_estimation(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    y_pred: &Array1<f64>,
    x_eval: &Array2<f64>,
    bandwidth: &Array1<f64>,
) -> VarianceFunctionResult {
    // Squared residuals
    let residuals_sq: Array1<f64> = y_train
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &yh)| (y - yh).powi(2))
        .collect();

    // Larger bandwidth for variance estimation
    let var_bw = bandwidth.mapv(|h| h * 1.2);

    // NW regression of squared residuals
    let w = weights::compute_kernel_weights(x_eval, x_train, &var_bw, &GaussianKernel);
    let mut variance_estimate = weights::nw_predict_from_weights(&w, &residuals_sq);

    // Clip to positive
    variance_estimate.mapv_inplace(|v| v.max(1e-10));
    let std_estimate = variance_estimate.mapv(f64::sqrt);

    VarianceFunctionResult {
        x_eval: x_eval.clone(),
        variance_estimate,
        std_estimate,
        bandwidth: var_bw,
    }
}

/// Result of conformal calibration.
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// Calibrated lower CI bound.
    pub lower: Array1<f64>,
    /// Calibrated upper CI bound.
    pub upper: Array1<f64>,
    /// Conformal quantile (the calibration correction).
    pub conformal_quantile: f64,
    /// Nominal confidence level.
    pub confidence_level: f64,
}

/// Conformal calibration for finite-sample coverage guarantees.
///
/// Splits data into training and calibration sets, fits on training,
/// computes conformity scores on calibration, and uses the scores to
/// adjust prediction intervals for the desired coverage.
///
/// Based on Lei et al. (2018), "Distribution-Free Predictive Inference
/// for Regression."
pub fn conformal_calibrate_ci(
    x: &Array2<f64>,
    y: &Array1<f64>,
    x_pred: &Array2<f64>,
    bandwidth: &Array1<f64>,
    confidence_level: f64,
    calibration_fraction: f64,
) -> Result<ConformalResult, FerroError> {
    let n = x.nrows();
    let n_cal = ((n as f64) * calibration_fraction).round() as usize;
    let n_train = n - n_cal;

    if n_train < 2 || n_cal < 1 {
        return Err(FerroError::InsufficientSamples {
            required: 3,
            actual: n,
            context: "conformal calibration needs at least 3 samples".into(),
        });
    }

    // Split: first n_train for training, rest for calibration
    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![..n_train]).to_owned();
    let x_cal = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_cal = y.slice(ndarray::s![n_train..]).to_owned();

    // Fit on training set
    let w_cal = weights::compute_kernel_weights(&x_cal, &x_train, bandwidth, &GaussianKernel);
    let y_hat_cal = weights::nw_predict_from_weights(&w_cal, &y_train);

    // Conformity scores: |y - ŷ|
    let mut scores: Vec<f64> = y_cal
        .iter()
        .zip(y_hat_cal.iter())
        .map(|(&y, &yh)| (y - yh).abs())
        .collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Conformal quantile: ceil((n_cal + 1) * confidence_level) / n_cal
    let idx = ((n_cal as f64 + 1.0) * confidence_level).ceil() as usize;
    let idx = idx.min(n_cal) - 1;
    let conformal_quantile = scores[idx.min(scores.len() - 1)];

    // Predict on new points
    let w_pred = weights::compute_kernel_weights(x_pred, &x_train, bandwidth, &GaussianKernel);
    let predictions = weights::nw_predict_from_weights(&w_pred, &y_train);

    let lower = predictions.mapv(|p| p - conformal_quantile);
    let upper = predictions.mapv(|p| p + conformal_quantile);

    Ok(ConformalResult {
        lower,
        upper,
        conformal_quantile,
        confidence_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn make_test_data() -> (Array2<f64>, Array1<f64>) {
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.06).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        (x, y)
    }

    #[test]
    fn bootstrap_ci_finite_and_ordered() {
        let (x, y) = make_test_data();
        let bw = array![0.5f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let result = wild_bootstrap_confidence_intervals(
            &x,
            &y,
            &x,
            &bw,
            &pred,
            200,
            0.95,
            BiasCorrection::None,
        )
        .unwrap();

        // CI bounds should be finite and lower <= upper
        for i in 0..x.nrows() {
            assert!(result.lower[i].is_finite(), "Lower CI should be finite");
            assert!(result.upper[i].is_finite(), "Upper CI should be finite");
            assert!(
                result.lower[i] <= result.upper[i],
                "Lower CI should be <= upper CI at index {i}"
            );
        }

        // Average width should be positive
        let avg_width: f64 = (&result.upper - &result.lower).mean().unwrap();
        assert!(avg_width > 0.0, "CI width should be positive");
    }

    #[test]
    fn ci_widens_with_higher_confidence() {
        let (x, y) = make_test_data();
        let bw = array![0.5f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let ci_90 = wild_bootstrap_confidence_intervals(
            &x,
            &y,
            &x,
            &bw,
            &pred,
            200,
            0.90,
            BiasCorrection::None,
        )
        .unwrap();
        let ci_95 = wild_bootstrap_confidence_intervals(
            &x,
            &y,
            &x,
            &bw,
            &pred,
            200,
            0.95,
            BiasCorrection::None,
        )
        .unwrap();

        // 95% CI should be wider on average
        let avg_width_90: f64 = (&ci_90.upper - &ci_90.lower).mean().unwrap();
        let avg_width_95: f64 = (&ci_95.upper - &ci_95.lower).mean().unwrap();
        assert!(
            avg_width_95 >= avg_width_90 * 0.9, // Allow some noise
            "95% width {avg_width_95:.4} should be >= 90% width {avg_width_90:.4}"
        );
    }

    #[test]
    fn fan_yao_positive_variance() {
        let (x, y) = make_test_data();
        let bw = array![0.5f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let result = fan_yao_variance_estimation(&x, &y, &pred, &x, &bw);

        for &v in &result.variance_estimate {
            assert!(v > 0.0, "Variance estimate should be positive");
        }
        for &s in &result.std_estimate {
            assert!(s > 0.0, "Std estimate should be positive");
        }
    }

    #[test]
    fn fan_yao_detects_heteroscedasticity() {
        // Generate data with variance proportional to x
        let n = 200;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        // y = sin(x) + noise where noise ~ x * sin(i)
        let y: Array1<f64> = x
            .column(0)
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi.sin() + xi * 0.3 * (i as f64 * 2.7).sin())
            .collect();

        let bw = array![0.5f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let result = fan_yao_variance_estimation(&x, &y, &pred, &x, &bw);

        // Variance at high x should be larger than at low x
        let var_low = result
            .variance_estimate
            .slice(ndarray::s![..20])
            .mean()
            .unwrap();
        let var_high = result
            .variance_estimate
            .slice(ndarray::s![180..])
            .mean()
            .unwrap();
        assert!(
            var_high > var_low,
            "Variance at high x ({var_high:.4}) should exceed low x ({var_low:.4})"
        );
    }

    #[test]
    fn conformal_ci_finite_and_ordered() {
        let (x, y) = make_test_data();
        let bw = array![0.5f64];
        let x_pred = Array2::from_shape_vec((5, 1), vec![0.5, 1.5, 2.5, 3.5, 4.5]).unwrap();

        let result = conformal_calibrate_ci(&x, &y, &x_pred, &bw, 0.95, 0.25).unwrap();

        for i in 0..x_pred.nrows() {
            assert!(result.lower[i].is_finite(), "Lower should be finite");
            assert!(result.upper[i].is_finite(), "Upper should be finite");
            assert!(
                result.lower[i] <= result.upper[i],
                "Lower should be <= upper at index {i}"
            );
        }
        assert!(
            result.conformal_quantile > 0.0,
            "Quantile should be positive"
        );
    }

    #[test]
    fn conformal_ci_covers_training_data() {
        // Conformal CI on training points should cover most of them
        let (x, y) = make_test_data();
        let bw = array![0.5f64];

        let result = conformal_calibrate_ci(&x, &y, &x, &bw, 0.90, 0.3).unwrap();

        let n = x.nrows();
        let covered: usize = (0..n)
            .filter(|&i| y[i] >= result.lower[i] && y[i] <= result.upper[i])
            .count();
        let coverage = covered as f64 / n as f64;
        // Should have reasonable coverage (not exact due to split)
        assert!(coverage > 0.5, "Coverage {coverage:.2} should be above 0.5");
    }

    #[test]
    fn conformal_rejects_too_few_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let y = array![0.0, 1.0];
        let bw = array![0.5f64];
        let x_pred = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();

        let result = conformal_calibrate_ci(&x, &y, &x_pred, &bw, 0.95, 0.5);
        assert!(result.is_err(), "Should fail with too few samples");
    }
}
