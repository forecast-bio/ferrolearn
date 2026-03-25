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
//! - [`median_absolute_error`] — median of absolute residuals (robust to outliers)
//! - [`max_error`] — maximum absolute error (worst-case prediction)
//! - [`mean_squared_log_error`] — MSE on log-transformed values
//! - [`root_mean_squared_log_error`] — square root of MSLE
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
/// **Convention note:** This function returns MAPE as a **percentage** (multiplied
/// by 100), so perfect predictions yield 0.0 and predictions that are off by 10%
/// on average yield 10.0. This differs from scikit-learn's
/// `mean_absolute_percentage_error`, which returns a **fraction** (not multiplied
/// by 100), where the same 10% error would be 0.1. Divide by 100 to convert to
/// the sklearn convention.
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
// median_absolute_error
// ---------------------------------------------------------------------------

/// Compute the median absolute error.
///
/// `MedAE = median(|y_true - y_pred|)`
///
/// Unlike MAE, the median is robust to outliers.
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
/// use ferrolearn_metrics::regression::median_absolute_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0, 100.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0, 3.0];
/// let medae = median_absolute_error(&y_true, &y_pred).unwrap();
/// // errors: [0, 0, 0, 97] — median of sorted [0,0,0,97] = (0+0)/2 = 0
/// assert!((medae - 0.0).abs() < 1e-10);
/// ```
pub fn median_absolute_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "median_absolute_error: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "median_absolute_error")?;

    let mut errors: Vec<F> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if n % 2 == 0 {
        (errors[n / 2 - 1] + errors[n / 2]) / F::from(2.0).unwrap()
    } else {
        errors[n / 2]
    };

    Ok(median)
}

// ---------------------------------------------------------------------------
// max_error
// ---------------------------------------------------------------------------

/// Compute the maximum absolute error.
///
/// `max_error = max |y_true - y_pred|`
///
/// This metric captures the worst-case prediction error.
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
/// use ferrolearn_metrics::regression::max_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.5_f64, 2.0, 5.0];
/// let me = max_error(&y_true, &y_pred).unwrap();
/// assert!((me - 2.0).abs() < 1e-10);
/// ```
pub fn max_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "max_error: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "max_error")?;

    let max = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .fold(F::zero(), |acc, v| if v > acc { v } else { acc });

    Ok(max)
}

// ---------------------------------------------------------------------------
// mean_squared_log_error
// ---------------------------------------------------------------------------

/// Compute the mean squared logarithmic error (MSLE).
///
/// `MSLE = (1/n) * sum (log(1 + y_true) - log(1 + y_pred))^2`
///
/// This metric is useful for targets with exponential growth where you
/// want to penalize under-prediction more than over-prediction. All values
/// must be non-negative.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values (must be non-negative).
/// * `y_pred` — predicted values (must be non-negative).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if any value is negative.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_squared_log_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
/// assert!((msle - 0.0).abs() < 1e-10);
/// ```
pub fn mean_squared_log_error<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_squared_log_error: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_squared_log_error")?;

    let one = F::one();

    for (i, &v) in y_true.iter().enumerate() {
        if v < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "mean_squared_log_error requires non-negative values, found negative at index {i}"
                ),
            });
        }
    }
    for (i, &v) in y_pred.iter().enumerate() {
        if v < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_pred".into(),
                reason: format!(
                    "mean_squared_log_error requires non-negative predictions, found negative at index {i}"
                ),
            });
        }
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            let diff = (one + t).ln() - (one + p).ln();
            acc + diff * diff
        });

    Ok(sum / F::from(n).unwrap())
}

// ---------------------------------------------------------------------------
// root_mean_squared_log_error
// ---------------------------------------------------------------------------

/// Compute the root mean squared logarithmic error (RMSLE).
///
/// `RMSLE = sqrt(MSLE) = sqrt((1/n) * sum (log(1 + y_true) - log(1 + y_pred))^2)`
///
/// # Arguments
///
/// * `y_true` — ground-truth target values (must be non-negative).
/// * `y_pred` — predicted values (must be non-negative).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if any value is negative.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::root_mean_squared_log_error;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let rmsle = root_mean_squared_log_error(&y_true, &y_pred).unwrap();
/// assert!((rmsle - 0.0).abs() < 1e-10);
/// ```
pub fn root_mean_squared_log_error<F>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let msle = mean_squared_log_error(y_true, y_pred)?;
    Ok(msle.sqrt())
}

// ---------------------------------------------------------------------------
// mean_pinball_loss
// ---------------------------------------------------------------------------

/// Compute the mean pinball (quantile) loss.
///
/// For a given quantile `alpha` in `[0, 1]`:
///
/// ```text
/// pinball = (1/n) * sum(alpha * max(y_true - y_pred, 0)
///                     + (1 - alpha) * max(y_pred - y_true, 0))
/// ```
///
/// When `alpha = 0.5`, this is equivalent to half the MAE.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
/// * `alpha`  — quantile parameter in `[0, 1]`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if `alpha` is not in `[0, 1]`.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_pinball_loss;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let loss = mean_pinball_loss(&y_true, &y_pred, 0.5).unwrap();
/// assert!((loss - 0.0).abs() < 1e-10);
/// ```
pub fn mean_pinball_loss<F>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    alpha: F,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_pinball_loss: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_pinball_loss")?;

    if alpha < F::zero() || alpha > F::one() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "alpha must be in [0, 1]".into(),
        });
    }

    let one_minus_alpha = F::one() - alpha;

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            let diff = t - p;
            if diff >= F::zero() {
                acc + alpha * diff
            } else {
                acc + one_minus_alpha * (-diff)
            }
        });

    Ok(sum / F::from(n).unwrap())
}

// ---------------------------------------------------------------------------
// mean_poisson_deviance
// ---------------------------------------------------------------------------

/// Compute the mean Poisson deviance.
///
/// ```text
/// D = (2/n) * sum(y_true * ln(y_true / y_pred) + y_pred - y_true)
/// ```
///
/// For samples where `y_true == 0`, the term reduces to `y_pred` (since
/// `0 * ln(0/x)` is defined as `0` by the limit convention).
///
/// # Arguments
///
/// * `y_true` — ground-truth target values (must be non-negative).
/// * `y_pred` — predicted values (must be strictly positive).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if any `y_true < 0` or any `y_pred <= 0`.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_poisson_deviance;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let dev = mean_poisson_deviance(&y_true, &y_pred).unwrap();
/// assert!((dev - 0.0).abs() < 1e-10);
/// ```
pub fn mean_poisson_deviance<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_poisson_deviance: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_poisson_deviance")?;

    for (i, &v) in y_true.iter().enumerate() {
        if v < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "mean_poisson_deviance requires y_true >= 0, found negative at index {i}"
                ),
            });
        }
    }
    for (i, &v) in y_pred.iter().enumerate() {
        if v <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_pred".into(),
                reason: format!(
                    "mean_poisson_deviance requires y_pred > 0, found non-positive at index {i}"
                ),
            });
        }
    }

    let two = F::from(2.0).unwrap();
    let n_f = F::from(n).unwrap();

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            if t == F::zero() {
                // 0 * ln(0/p) is 0 by convention, so term = 0 + p - 0 = p
                acc + p
            } else {
                // t * ln(t/p) + p - t
                acc + t * (t / p).ln() + p - t
            }
        });

    Ok(two * sum / n_f)
}

// ---------------------------------------------------------------------------
// mean_gamma_deviance
// ---------------------------------------------------------------------------

/// Compute the mean Gamma deviance.
///
/// ```text
/// D = (2/n) * sum(ln(y_pred / y_true) + y_true / y_pred - 1)
/// ```
///
/// # Arguments
///
/// * `y_true` — ground-truth target values (must be strictly positive).
/// * `y_pred` — predicted values (must be strictly positive).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if any `y_true <= 0` or any `y_pred <= 0`.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_gamma_deviance;
/// use ndarray::array;
///
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let dev = mean_gamma_deviance(&y_true, &y_pred).unwrap();
/// assert!((dev - 0.0).abs() < 1e-10);
/// ```
pub fn mean_gamma_deviance<F>(y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_gamma_deviance: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_gamma_deviance")?;

    for (i, &v) in y_true.iter().enumerate() {
        if v <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "mean_gamma_deviance requires y_true > 0, found non-positive at index {i}"
                ),
            });
        }
    }
    for (i, &v) in y_pred.iter().enumerate() {
        if v <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "y_pred".into(),
                reason: format!(
                    "mean_gamma_deviance requires y_pred > 0, found non-positive at index {i}"
                ),
            });
        }
    }

    let two = F::from(2.0).unwrap();
    let n_f = F::from(n).unwrap();

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(F::zero(), |acc, (&t, &p)| {
            acc + (p / t).ln() + t / p - F::one()
        });

    Ok(two * sum / n_f)
}

// ---------------------------------------------------------------------------
// mean_tweedie_deviance
// ---------------------------------------------------------------------------

/// Compute the mean Tweedie deviance with power parameter `p`.
///
/// The Tweedie family generalises several distributions:
///
/// | `power` | Distribution | `y_true` constraint | `y_pred` constraint |
/// |---------|-------------|---------------------|---------------------|
/// | 0       | Normal      | any                 | any                 |
/// | 1       | Poisson     | >= 0                | > 0                 |
/// | 2       | Gamma       | > 0                 | > 0                 |
/// | 3       | Inverse Gaussian | > 0            | > 0                 |
///
/// The unit deviance for each observation is:
///
/// ```text
/// d(y, mu) = 2 * (y^(2-p) / ((1-p)(2-p)) - y*mu^(1-p) / (1-p) + mu^(2-p) / (2-p))
/// ```
///
/// Special cases for `p = 0`, `p = 1`, and `p = 2` are handled explicitly.
///
/// # Arguments
///
/// * `y_true` — ground-truth target values.
/// * `y_pred` — predicted values.
/// * `power`  — the Tweedie power parameter.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if the inputs violate the constraints
/// for the given `power`, or if `power` is in `(0, 1)` (which is not a valid
/// Tweedie distribution).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::regression::mean_tweedie_deviance;
/// use ndarray::array;
///
/// // power=0 should give 2*MSE
/// let y_true = array![1.0_f64, 2.0, 3.0];
/// let y_pred = array![1.0_f64, 2.0, 3.0];
/// let dev = mean_tweedie_deviance(&y_true, &y_pred, 0.0).unwrap();
/// assert!((dev - 0.0).abs() < 1e-10);
/// ```
pub fn mean_tweedie_deviance<F>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    power: F,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_same_length(y_true, y_pred, "mean_tweedie_deviance: y_true vs y_pred")?;
    let n = y_true.len();
    check_non_empty(n, "mean_tweedie_deviance")?;

    let zero = F::zero();
    let one = F::one();
    let two = F::from(2.0).unwrap();

    // Reject power in (0, 1) — not a valid Tweedie distribution.
    if power > zero && power < one {
        return Err(FerroError::InvalidParameter {
            name: "power".into(),
            reason: "power must not be in (0, 1); valid values are <= 0 or >= 1".into(),
        });
    }

    // Check input constraints based on power.
    if power >= one {
        // y_pred > 0 required for power >= 1
        for (i, &v) in y_pred.iter().enumerate() {
            if v <= zero {
                return Err(FerroError::InvalidParameter {
                    name: "y_pred".into(),
                    reason: format!(
                        "mean_tweedie_deviance with power >= 1 requires y_pred > 0, found non-positive at index {i}"
                    ),
                });
            }
        }
    }
    if power >= one && power < two {
        // y_true >= 0 for Poisson-like
        for (i, &v) in y_true.iter().enumerate() {
            if v < zero {
                return Err(FerroError::InvalidParameter {
                    name: "y_true".into(),
                    reason: format!(
                        "mean_tweedie_deviance with 1 <= power < 2 requires y_true >= 0, found negative at index {i}"
                    ),
                });
            }
        }
    }
    if power >= two {
        // y_true > 0 for Gamma-like and beyond
        for (i, &v) in y_true.iter().enumerate() {
            if v <= zero {
                return Err(FerroError::InvalidParameter {
                    name: "y_true".into(),
                    reason: format!(
                        "mean_tweedie_deviance with power >= 2 requires y_true > 0, found non-positive at index {i}"
                    ),
                });
            }
        }
    }

    let n_f = F::from(n).unwrap();

    // Special case: power == 0 => Normal distribution => deviance = (y - mu)^2
    if power == zero {
        let sum = y_true
            .iter()
            .zip(y_pred.iter())
            .fold(zero, |acc, (&t, &p)| {
                let diff = t - p;
                acc + diff * diff
            });
        return Ok(sum / n_f);
    }

    // Special case: power == 1 => Poisson
    if power == one {
        return mean_poisson_deviance(y_true, y_pred);
    }

    // Special case: power == 2 => Gamma
    if power == two {
        return mean_gamma_deviance(y_true, y_pred);
    }

    // General case
    let one_minus_p = one - power;
    let two_minus_p = two - power;

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .fold(zero, |acc, (&t, &p)| {
            let term = if t == zero {
                // When y_true = 0: y^(2-p) / ((1-p)(2-p)) = 0 (since 2-p < 0 for p > 2,
                // but for p in [1,2), y_true >= 0 is fine and 0^(2-p) = 0 since 2-p > 0)
                // Actually for p < 2, 0^(2-p) = 0. For p > 2, y_true must be > 0 so this
                // branch only fires for 1 <= p < 2.
                p.powf(two_minus_p) / two_minus_p
            } else {
                t.powf(two_minus_p) / (one_minus_p * two_minus_p)
                    - t * p.powf(one_minus_p) / one_minus_p
                    + p.powf(two_minus_p) / two_minus_p
            };
            acc + term
        });

    Ok(two * sum / n_f)
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

    // -----------------------------------------------------------------------
    // median_absolute_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_median_absolute_error_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(median_absolute_error(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_absolute_error_odd() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.0, 2.0];
        // errors: [0.5, 0.0, 1.0], sorted: [0.0, 0.5, 1.0], median = 0.5
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_even() {
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0];
        let y_pred = array![1.0_f64, 2.0, 5.0, 6.0];
        // errors: [0, 0, 2, 2], sorted: [0, 0, 2, 2], median = (0+2)/2 = 1
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_robust_to_outlier() {
        let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0_f64, 2.0, 3.0, 4.0, 1000.0];
        // errors: [0, 0, 0, 0, 995], sorted, median = 0.0
        assert_abs_diff_eq!(
            median_absolute_error(&y_true, &y_pred).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_median_absolute_error_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(median_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_median_absolute_error_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(median_absolute_error(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // max_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_error_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(max_error(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_basic() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.0, 5.0];
        assert_abs_diff_eq!(max_error(&y_true, &y_pred).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_negative_residual() {
        let y_true = array![5.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert_abs_diff_eq!(max_error(&y_true, &y_pred).unwrap(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_error_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(max_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_max_error_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64];
        assert!(max_error(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // mean_squared_log_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_msle_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(
            mean_squared_log_error(&y, &y).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_basic() {
        let y_true = array![3.0_f64, 5.0, 2.5, 7.0];
        let y_pred = array![2.5_f64, 5.0, 4.0, 8.0];
        // manual: (ln(4)-ln(3.5))^2 + 0 + (ln(3.5)-ln(5))^2 + (ln(8)-ln(9))^2
        let expected = ((4.0_f64.ln() - 3.5_f64.ln()).powi(2)
            + 0.0
            + (3.5_f64.ln() - 5.0_f64.ln()).powi(2)
            + (8.0_f64.ln() - 9.0_f64.ln()).powi(2))
            / 4.0;
        assert_abs_diff_eq!(
            mean_squared_log_error(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_msle_negative_y_true() {
        let y_true = array![-1.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_negative_y_pred() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64, -2.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_msle_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // root_mean_squared_log_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_rmsle_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(
            root_mean_squared_log_error(&y, &y).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_rmsle_consistent_with_msle() {
        let y_true = array![3.0_f64, 5.0, 2.5, 7.0];
        let y_pred = array![2.5_f64, 5.0, 4.0, 8.0];
        let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
        let rmsle = root_mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(rmsle, msle.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_rmsle_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(root_mean_squared_log_error(&y_true, &y_pred).is_err());
    }
}

// ---------------------------------------------------------------------------
// Kani formal verification harnesses
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use ndarray::Array1;

    /// Helper: generate a symbolic f64 that is finite (not NaN or Inf)
    /// and within a reasonable magnitude range to avoid overflow.
    fn any_finite_f64() -> f64 {
        let val: f64 = kani::any();
        kani::assume(!val.is_nan() && !val.is_infinite());
        kani::assume(val.abs() < 1e6);
        val
    }

    /// Prove that mean_absolute_error output is >= 0.0 for all finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_mae_non_negative() {
        const N: usize = 4;
        let mut y_true_data = [0.0f64; N];
        let mut y_pred_data = [0.0f64; N];
        for i in 0..N {
            y_true_data[i] = any_finite_f64();
            y_pred_data[i] = any_finite_f64();
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = mean_absolute_error(&y_true, &y_pred);
        if let Ok(mae) = result {
            assert!(mae >= 0.0, "MAE must be >= 0.0");
        }
    }

    /// Prove that mean_squared_error output is >= 0.0 for all finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_mse_non_negative() {
        const N: usize = 4;
        let mut y_true_data = [0.0f64; N];
        let mut y_pred_data = [0.0f64; N];
        for i in 0..N {
            y_true_data[i] = any_finite_f64();
            y_pred_data[i] = any_finite_f64();
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = mean_squared_error(&y_true, &y_pred);
        if let Ok(mse) = result {
            assert!(mse >= 0.0, "MSE must be >= 0.0");
        }
    }

    /// Prove that root_mean_squared_error output is >= 0.0 for all finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_rmse_non_negative() {
        const N: usize = 4;
        let mut y_true_data = [0.0f64; N];
        let mut y_pred_data = [0.0f64; N];
        for i in 0..N {
            y_true_data[i] = any_finite_f64();
            y_pred_data[i] = any_finite_f64();
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = root_mean_squared_error(&y_true, &y_pred);
        if let Ok(rmse) = result {
            assert!(rmse >= 0.0, "RMSE must be >= 0.0");
        }
    }

    /// Prove that r2_score does not produce NaN when y_true is non-constant
    /// (i.e., when SS_tot > 0 and the function returns Ok).
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_r2_score_no_nan_on_non_constant() {
        const N: usize = 4;
        let mut y_true_data = [0.0f64; N];
        let mut y_pred_data = [0.0f64; N];
        for i in 0..N {
            y_true_data[i] = any_finite_f64();
            y_pred_data[i] = any_finite_f64();
        }

        // Ensure y_true is non-constant: at least one pair of values differs.
        kani::assume(y_true_data[0] != y_true_data[1]);

        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = r2_score(&y_true, &y_pred);
        if let Ok(r2) = result {
            assert!(!r2.is_nan(), "R² must not be NaN for non-constant y_true");
        }
    }

    /// Prove that explained_variance_score does not produce NaN when y_true
    /// is non-constant (Var(y_true) > 0).
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_evs_no_nan_on_non_constant() {
        const N: usize = 4;
        let mut y_true_data = [0.0f64; N];
        let mut y_pred_data = [0.0f64; N];
        for i in 0..N {
            y_true_data[i] = any_finite_f64();
            y_pred_data[i] = any_finite_f64();
        }

        // Ensure y_true is non-constant: at least one pair of values differs.
        kani::assume(y_true_data[0] != y_true_data[1]);

        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = explained_variance_score(&y_true, &y_pred);
        if let Ok(evs) = result {
            assert!(!evs.is_nan(), "EVS must not be NaN for non-constant y_true");
        }
    }

    // -----------------------------------------------------------------------
    // mean_pinball_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_pinball_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(
            mean_pinball_loss(&y, &y, 0.5_f64).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_pinball_half_is_half_mae() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.0, 2.5];
        let pinball = mean_pinball_loss(&y_true, &y_pred, 0.5_f64).unwrap();
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(pinball, mae / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pinball_alpha_high() {
        // alpha=0.9: penalises under-prediction heavily
        let y_true = array![3.0_f64];
        let y_pred = array![1.0_f64]; // under-prediction by 2
        // diff = 3 - 1 = 2 > 0 => alpha * diff = 0.9 * 2 = 1.8
        let loss = mean_pinball_loss(&y_true, &y_pred, 0.9_f64).unwrap();
        assert_abs_diff_eq!(loss, 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_pinball_alpha_low() {
        // alpha=0.1: penalises over-prediction heavily
        let y_true = array![1.0_f64];
        let y_pred = array![3.0_f64]; // over-prediction by 2
        // diff = 1 - 3 = -2 < 0 => (1-alpha) * |diff| = 0.9 * 2 = 1.8
        let loss = mean_pinball_loss(&y_true, &y_pred, 0.1_f64).unwrap();
        assert_abs_diff_eq!(loss, 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_pinball_invalid_alpha() {
        let y = array![1.0_f64];
        assert!(mean_pinball_loss(&y, &y, 1.5_f64).is_err());
        assert!(mean_pinball_loss(&y, &y, -0.1_f64).is_err());
    }

    #[test]
    fn test_pinball_empty() {
        let y_true = Array1::<f64>::from_vec(vec![]);
        let y_pred = Array1::<f64>::from_vec(vec![]);
        assert!(mean_pinball_loss(&y_true, &y_pred, 0.5_f64).is_err());
    }

    #[test]
    fn test_pinball_f32() {
        let y_true = array![1.0_f32, 2.0, 3.0];
        let y_pred = array![1.5_f32, 2.0, 2.5];
        let loss = mean_pinball_loss(&y_true, &y_pred, 0.5_f32).unwrap();
        assert!(loss >= 0.0_f32);
    }

    // -----------------------------------------------------------------------
    // mean_poisson_deviance
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_deviance_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(mean_poisson_deviance(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_deviance_basic() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![2.0_f64, 1.0];
        // term1: 1*ln(1/2) + 2 - 1 = -ln(2) + 1
        // term2: 2*ln(2/1) + 1 - 2 = 2*ln(2) - 1
        // sum = ln(2)
        // deviance = 2 * ln(2) / 2 = ln(2)
        let expected = 2.0_f64.ln();
        assert_abs_diff_eq!(
            mean_poisson_deviance(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_poisson_deviance_zero_true() {
        let y_true = array![0.0_f64, 1.0];
        let y_pred = array![1.0_f64, 1.0];
        // term1: y_true=0 => contributes y_pred=1.0
        // term2: 1*ln(1/1) + 1 - 1 = 0
        // sum = 1, deviance = 2*1/2 = 1
        assert_abs_diff_eq!(
            mean_poisson_deviance(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_poisson_deviance_negative_true() {
        let y_true = array![-1.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert!(mean_poisson_deviance(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_poisson_deviance_zero_pred() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![0.0_f64, 2.0];
        assert!(mean_poisson_deviance(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // mean_gamma_deviance
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_deviance_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        assert_abs_diff_eq!(mean_gamma_deviance(&y, &y).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_deviance_basic() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![2.0_f64, 1.0];
        // term1: ln(2/1) + 1/2 - 1 = ln(2) - 0.5
        // term2: ln(1/2) + 2/1 - 1 = -ln(2) + 1
        // sum = 0.5
        // deviance = 2 * 0.5 / 2 = 0.5
        let expected = 0.5;
        assert_abs_diff_eq!(
            mean_gamma_deviance(&y_true, &y_pred).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_gamma_deviance_zero_true() {
        let y_true = array![0.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert!(mean_gamma_deviance(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_gamma_deviance_zero_pred() {
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![0.0_f64, 2.0];
        assert!(mean_gamma_deviance(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // mean_tweedie_deviance
    // -----------------------------------------------------------------------

    #[test]
    fn test_tweedie_power_0_equals_mse() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.5, 2.5];
        let tweedie = mean_tweedie_deviance(&y_true, &y_pred, 0.0).unwrap();
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(tweedie, mse, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_power_1_equals_poisson() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.5, 2.5];
        let tweedie = mean_tweedie_deviance(&y_true, &y_pred, 1.0).unwrap();
        let poisson = mean_poisson_deviance(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(tweedie, poisson, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_power_2_equals_gamma() {
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.5, 2.5];
        let tweedie = mean_tweedie_deviance(&y_true, &y_pred, 2.0).unwrap();
        let gamma = mean_gamma_deviance(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(tweedie, gamma, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_perfect() {
        let y = array![1.0_f64, 2.0, 3.0];
        // For all power values, perfect predictions should give 0 deviance.
        for &p in &[0.0, 1.0, 1.5, 2.0, 3.0] {
            let dev = mean_tweedie_deviance(&y, &y, p).unwrap();
            assert_abs_diff_eq!(dev, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tweedie_invalid_power() {
        let y = array![1.0_f64, 2.0];
        assert!(mean_tweedie_deviance(&y, &y, 0.5).is_err());
    }

    #[test]
    fn test_tweedie_negative_true_power_1() {
        let y_true = array![-1.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert!(mean_tweedie_deviance(&y_true, &y_pred, 1.0).is_err());
    }

    #[test]
    fn test_tweedie_zero_true_power_2() {
        let y_true = array![0.0_f64, 2.0];
        let y_pred = array![1.0_f64, 2.0];
        assert!(mean_tweedie_deviance(&y_true, &y_pred, 2.0).is_err());
    }
}
