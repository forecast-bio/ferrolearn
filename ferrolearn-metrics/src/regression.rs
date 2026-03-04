//! Regression evaluation metrics.
//!
//! This module provides standard regression metrics used to evaluate the
//! performance of supervised regression models:
//!
//! - [`mean_absolute_error`] — mean of absolute residuals
//! - [`mean_squared_error`] — mean of squared residuals
//! - [`root_mean_squared_error`] — square root of MSE
//! - [`r2_score`] — coefficient of determination
//! - [`mean_absolute_percentage_error`] — mean of absolute percentage errors
//! - [`explained_variance_score`] — fraction of variance explained by the model
//!
//! All functions are generic over `F: num_traits::Float + Send + Sync + 'static`.

use ferrolearn_core::FerroError;
use ndarray::Array1;
use num_traits::Float;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that two arrays have the same length.
fn check_same_length<F: Float>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    context: &str,
) -> Result<(), FerroError> {
    let n = y_true.len();
    let m = y_pred.len();
    if n != m {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![m],
            context: context.into(),
        });
    }
    Ok(())
}

/// Validate that the array has at least one element.
fn check_non_empty(n: usize, context: &str) -> Result<(), FerroError> {
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the mean absolute error (MAE).
///
/// `MAE = (1/n) * sum |y_true - y_pred|`
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_absolute_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.5_f64, 2.0, 2.5];
/// let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
/// assert!((mae - 1.0 / 3.0).abs() < 1e-10);
/// ```
pub fn mean_absolute_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_absolute_error: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_absolute_error")?;

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| acc + (t - p).abs());

    Ok(sum / F::from(n).unwrap())
}

/// Compute the mean squared error (MSE).
///
/// `MSE = (1/n) * sum (y_true - y_pred)^2`
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_squared_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 4.0];
/// let mse = mean_squared_error(&y_true, &y_pred).unwrap();
/// assert!((mse - 1.0 / 3.0).abs() < 1e-10);
/// ```
pub fn mean_squared_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_squared_error: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_squared_error")?;

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            let diff = t - p;
            acc + diff * diff
        });

    Ok(sum / F::from(n).unwrap())
}

/// Compute the root mean squared error (RMSE).
///
/// `RMSE = sqrt(MSE)`
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::root_mean_squared_error;
/// use ndarray::array;
///
/// let y_true = array![0.0_f64, 0.0];
/// let y_pred = array![1.0_f64, 1.0];
/// let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
/// assert!((rmse - 1.0).abs() < 1e-10);
/// ```
pub fn root_mean_squared_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Compute the coefficient of determination R².
///
/// `R² = 1 - SS_res / SS_tot`
///
/// where `SS_res = sum (y_true - y_pred)^2` and
/// `SS_tot = sum (y_true - mean(y_true))^2`.
///
/// Returns `1.0` when predictions are perfect. Returns `0.0` when the model
/// is equivalent to predicting the mean. Can be negative.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::NumericalInstability`] if `SS_tot` is zero (all
/// `y_true` values are identical), making R² undefined.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::r2_score;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let r2 = r2_score(&y_true, &y_pred).unwrap();
/// assert!((r2 - 1.0).abs() < 1e-10);
/// ```
pub fn r2_score<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "r2_score: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "r2_score")?;

    let n_f = F::from(n).unwrap();
    let mean_true = y_true.iter().copied().fold(F::zero(), |a, v| a + v) / n_f;

    let ss_tot = y_true.iter().fold(F::zero(), |acc, &t| {
        let diff = t - mean_true;
        acc + diff * diff
    });

    if ss_tot == F::zero() {
        return Err(FerroError::NumericalInstability {
            message: "r2_score: SS_tot is zero — all y_true values are constant".into(),
        });
    }

    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            let diff = t - p;
            acc + diff * diff
        });

    Ok(F::one() - ss_res / ss_tot)
}

/// Compute the mean absolute percentage error (MAPE).
///
/// `MAPE = (1/n) * sum |( y_true - y_pred ) / y_true| * 100`
///
/// Note: samples where `y_true == 0` are skipped to avoid division by zero.
/// If all `y_true` values are zero, `MAPE` is returned as `F::infinity()`.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values (non-zero for meaningful MAPE).
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_absolute_percentage_error;
/// use ndarray::array;
///
/// let y_true = array![100.0_f64, 200.0, 300.0];
/// let y_pred = array![110.0_f64, 190.0, 300.0];
/// let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
/// // |10/100| + |10/200| + |0/300| = 0.1 + 0.05 + 0.0 = 0.15 / 3 * 100 = 5.0
/// assert!((mape - 5.0).abs() < 1e-10);
/// ```
pub fn mean_absolute_percentage_error<F>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(
        y_true,
        y_pred,
        "mean_absolute_percentage_error: y_true vs y_pred",
    )?;
    let n = y_true.len();
    check_non_empty(n, "mean_absolute_percentage_error")?;

    let hundred = F::from(100.0).unwrap();
    let mut sum = F::zero();
    let mut count = 0usize;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t != F::zero() {
            sum = sum + ((t - p) / t).abs();
            count += 1;
        }
    }

    if count == 0 {
        return Ok(F::infinity());
    }

    Ok(sum * hundred / F::from(count).unwrap())
}

/// Compute the explained variance score.
///
/// `EVS = 1 - Var(y_true - y_pred) / Var(y_true)`
///
/// where `Var(x) = mean(x^2) - mean(x)^2` (population variance).
///
/// Returns `1.0` for perfect predictions. Returns `0.0` when the model
/// accounts for no variance. Can be negative.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::NumericalInstability`] if `Var(y_true)` is zero
/// (all targets are identical), making EVS undefined.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::explained_variance_score;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let evs = explained_variance_score(&y_true, &y_pred).unwrap();
/// assert!((evs - 1.0).abs() < 1e-10);
/// ```
pub fn explained_variance_score<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "explained_variance_score: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "explained_variance_score")?;

    let n_f = F::from(n).unwrap();

    // Variance of y_true.
    let mean_true = y_true.iter().copied().fold(F::zero(), |a, v| a + v) / n_f;
    let var_true = y_true.iter().fold(F::zero(), |acc, &t| {
        let diff = t - mean_true;
        acc + diff * diff
    }) / n_f;

    if var_true == F::zero() {
        return Err(FerroError::NumericalInstability {
            message: "explained_variance_score: Var(y_true) is zero — all targets are constant"
                .into(),
        });
    }

    // Variance of residuals (y_true - y_pred).
    let residuals: Vec<F> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect();

    let mean_res = residuals.iter().copied().fold(F::zero(), |a, v| a + v) / n_f;
    let var_res = residuals.iter().fold(F::zero(), |acc, &r| {
        let diff = r - mean_res;
        acc + diff * diff
    }) / n_f;

    Ok(F::one() - var_res / var_true)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // mean_absolute_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_mae_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(mean_absolute_error(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_basic() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.0, 2.5];
        // |0.5| + |0| + |0.5| = 1.0 / 3
        assert_abs_diff_eq!(
            mean_absolute_error(&y_true, &y_pred).unwrap(),
            1.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mae_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(mean_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mae_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(mean_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mae_f32() {
        let y_true = array![1.0_f32, 2.0, 3.0];
        let y_pred = array![2.0_f32, 2.0, 2.0];
        assert_abs_diff_eq!(
            mean_absolute_error(&y_true, &y_pred).unwrap(),
            2.0_f32 / 3.0,
            epsilon = 1e-6
        );
    }

    // -----------------------------------------------------------------------
    // mean_squared_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(mean_squared_error(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_basic() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.0_f64, 2.0, 4.0];
        // 0 + 0 + 1 = 1 / 3
        assert_abs_diff_eq!(
            mean_squared_error(&y_true, &y_pred).unwrap(),
            1.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mse_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(mean_squared_error(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // root_mean_squared_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_rmse_basic() {
        let y_true = array![0.0_f64, 0.0];
        let y_pred = array![1.0_f64, 1.0];
        assert_abs_diff_eq!(
            root_mean_squared_error(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_rmse_consistent_with_mse() {
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0];
        let y_pred = array![1.5_f64, 2.5, 2.5, 3.5];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(rmse, mse.sqrt(), epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // r2_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_r2_perfect() {
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(r2_score(&y_true, &y_pred).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_mean_predictor() {
        // When predictions equal the mean, R² should be 0.
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0_f64;
        let y_pred = array![mean, mean, mean, mean, mean];
        assert_abs_diff_eq!(r2_score(&y_true, &y_pred).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_negative() {
        // A terrible predictor can give R² < 0.
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![3.0_f64, 2.0, 1.0];
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 < 0.0);
    }

    #[test]
    fn test_r2_constant_y_true() {
        let y_true = array![3.0_f64, 3.0, 3.0];
        let y_pred = array![3.0_f64, 3.0, 3.0];
        assert!(r2_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_r2_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(r2_score(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // mean_absolute_percentage_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_mape_basic() {
        let y_true = array![100.0_f64, 200.0, 300.0];
        let y_pred = array![110.0_f64, 190.0, 300.0];
        // (|10/100| + |10/200| + |0/300|) / 3 * 100 = (0.1 + 0.05 + 0.0) / 3 * 100 = 5.0
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y_true, &y_pred).unwrap(),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mape_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y, &y).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mape_skips_zero_true() {
        // y_true=0 at index 1 should be skipped.
        let y_true = array![100.0_f64, 0.0, 200.0];
        let y_pred = array![110.0_f64, 999.0, 200.0];
        // index 0: |10/100| = 0.1; index 2: |0/200| = 0.0; count=2
        // MAPE = (0.1 + 0.0) / 2 * 100 = 5.0
        assert_abs_diff_eq!(
            mean_absolute_percentage_error(&y_true, &y_pred).unwrap(),
            5.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mape_all_zero_true_returns_inf() {
        let y_true = array![0.0_f64, 0.0];
        let y_pred = array![1.0_f64, 2.0];
        let mape = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!(mape.is_infinite());
    }

    #[test]
    fn test_mape_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(mean_absolute_percentage_error(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // explained_variance_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_evs_perfect() {
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evs_constant_error() {
        // When residuals are constant (same for all samples), EVS = 1.0
        // because Var(residuals) = 0 when residual is constant.
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![2.0_f64, 3.0, 4.0]; // constant bias of +1
        // residuals = [-1, -1, -1], Var(res) = 0 => EVS = 1
        assert_abs_diff_eq!(
            explained_variance_score(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_evs_constant_y_true() {
        let y_true = array![5.0_f64, 5.0, 5.0];
        let y_pred = array![1.0_f64, 2.0, 3.0];
        assert!(explained_variance_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_evs_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(explained_variance_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_evs_vs_r2_perfect_predictions() {
        // For perfect predictions, EVS == R² == 1.0.
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0];
        let y_pred = array![1.0_f64, 2.0, 3.0, 4.0];
        let evs = explained_variance_score(&y_true, &y_pred).unwrap();
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(evs, r2, epsilon = 1e-10);
    }
}
