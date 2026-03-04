//! Classification evaluation metrics.
//!
//! This module provides standard classification metrics used to evaluate the
//! performance of supervised classification models:
//!
//! - [`accuracy_score`] — fraction of correctly classified samples
//! - [`precision_score`] — positive predictive value
//! - [`recall_score`] — sensitivity / true positive rate
//! - [`f1_score`] — harmonic mean of precision and recall
//! - [`roc_auc_score`] — area under the ROC curve (binary classification)
//! - [`confusion_matrix`] — matrix of true/predicted class counts
//! - [`log_loss`] — cross-entropy loss for probabilistic classifiers

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};

/// Averaging strategy for multi-class precision, recall, and F1.
///
/// This enum controls how per-class scores are aggregated into a single
/// scalar metric when there are more than two classes.
///
/// # Variants
///
/// | Variant    | Description |
/// |------------|-------------|
/// | `Binary`   | Report for the positive class only (class label 1). Requires exactly two distinct classes. |
/// | `Macro`    | Unweighted mean of per-class scores. Treats all classes equally regardless of support. |
/// | `Micro`    | Compute counts globally (sum TPs, FPs, FNs across classes) then compute the metric. |
/// | `Weighted` | Mean of per-class scores weighted by the number of true instances per class. |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Use the positive class (label 1) only. Requires binary labels.
    Binary,
    /// Unweighted mean over all classes.
    Macro,
    /// Global micro-averaged score.
    Micro,
    /// Class-support-weighted mean over all classes.
    Weighted,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that two label arrays have the same length.
fn check_same_length(n_true: usize, n_pred: usize, context: &str) -> Result<(), FerroError> {
    if n_true != n_pred {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_true],
            actual: vec![n_pred],
            context: context.into(),
        });
    }
    Ok(())
}

/// Return sorted unique class labels found in `y_true` or `y_pred`.
fn unique_classes(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Vec<usize> {
    let mut classes: Vec<usize> = y_true.iter().chain(y_pred.iter()).copied().collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

/// Compute per-class TP, FP, FN counts.
///
/// Returns `(tp, fp, fn_count)` for each class in `classes`.
fn per_class_counts(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    classes: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = classes.len();
    let mut tp = vec![0usize; n];
    let mut fp = vec![0usize; n];
    let mut fn_count = vec![0usize; n];

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        for (k, &c) in classes.iter().enumerate() {
            let is_true_c = t == c;
            let is_pred_c = p == c;
            if is_true_c && is_pred_c {
                tp[k] += 1;
            } else if !is_true_c && is_pred_c {
                fp[k] += 1;
            } else if is_true_c && !is_pred_c {
                fn_count[k] += 1;
            }
        }
    }
    (tp, fp, fn_count)
}

/// Safe division returning 0.0 on divide-by-zero.
#[inline]
fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the fraction of correctly classified samples.
///
/// # Arguments
///
/// * `y_true` — ground-truth class labels.
/// * `y_pred` — predicted class labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_pred` have
/// different lengths.
///
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::accuracy_score;
/// use ndarray::array;
///
/// let y_true = array![0, 1, 2, 1, 0];
/// let y_pred = array![0, 1, 2, 0, 0];
/// let acc = accuracy_score(&y_true, &y_pred).unwrap();
/// assert!((acc - 0.8).abs() < 1e-10);
/// ```
pub fn accuracy_score(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<f64, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_pred.len(), "accuracy_score: y_true vs y_pred")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "accuracy_score".into(),
        });
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(&t, &p)| t == p)
        .count();
    Ok(correct as f64 / n as f64)
}

/// Compute the precision score.
///
/// Precision is the ratio `TP / (TP + FP)`. When the denominator is zero,
/// the per-class precision defaults to `0.0`.
///
/// # Arguments
///
/// * `y_true`   — ground-truth class labels.
/// * `y_pred`   — predicted class labels.
/// * `average`  — aggregation strategy (see [`Average`]).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] when `Average::Binary` is
/// requested but the label set does not contain exactly two classes.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::{precision_score, Average};
/// use ndarray::array;
///
/// let y_true = array![0, 1, 1, 0, 1];
/// let y_pred = array![0, 1, 0, 0, 1];
/// let p = precision_score(&y_true, &y_pred, Average::Binary).unwrap();
/// assert!((p - 1.0).abs() < 1e-10);
/// ```
pub fn precision_score(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    average: Average,
) -> Result<f64, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_pred.len(), "precision_score: y_true vs y_pred")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "precision_score".into(),
        });
    }

    let classes = unique_classes(y_true, y_pred);
    let (tp, fp, _fn) = per_class_counts(y_true, y_pred, &classes);

    aggregate_metric(&tp, &fp, &_fn, y_true, &classes, average, "precision_score")
}

/// Compute the recall (sensitivity) score.
///
/// Recall is the ratio `TP / (TP + FN)`. When the denominator is zero,
/// the per-class recall defaults to `0.0`.
///
/// # Arguments
///
/// * `y_true`   — ground-truth class labels.
/// * `y_pred`   — predicted class labels.
/// * `average`  — aggregation strategy (see [`Average`]).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] when `Average::Binary` is
/// requested but the label set is not binary.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::{recall_score, Average};
/// use ndarray::array;
///
/// let y_true = array![0, 1, 1, 0, 1];
/// let y_pred = array![0, 1, 0, 0, 1];
/// let r = recall_score(&y_true, &y_pred, Average::Binary).unwrap();
/// assert!((r - 2.0 / 3.0).abs() < 1e-10);
/// ```
pub fn recall_score(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    average: Average,
) -> Result<f64, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_pred.len(), "recall_score: y_true vs y_pred")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "recall_score".into(),
        });
    }

    let classes = unique_classes(y_true, y_pred);
    let (tp, fp, fn_count) = per_class_counts(y_true, y_pred, &classes);

    aggregate_recall(
        &tp,
        &fp,
        &fn_count,
        y_true,
        &classes,
        average,
        "recall_score",
    )
}

/// Compute the F1 score (harmonic mean of precision and recall).
///
/// F1 = `2 * precision * recall / (precision + recall)`. Defaults to `0.0`
/// when both precision and recall are zero.
///
/// # Arguments
///
/// * `y_true`   — ground-truth class labels.
/// * `y_pred`   — predicted class labels.
/// * `average`  — aggregation strategy (see [`Average`]).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] when `Average::Binary` is
/// requested but the label set is not binary.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::{f1_score, Average};
/// use ndarray::array;
///
/// let y_true = array![0, 1, 1, 0, 1];
/// let y_pred = array![0, 1, 0, 0, 1];
/// let f1 = f1_score(&y_true, &y_pred, Average::Binary).unwrap();
/// // precision=1.0, recall=2/3 => f1 = 2*(1*2/3)/(1+2/3) = 4/5 = 0.8
/// assert!((f1 - 0.8).abs() < 1e-10);
/// ```
pub fn f1_score(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
    average: Average,
) -> Result<f64, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_pred.len(), "f1_score: y_true vs y_pred")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "f1_score".into(),
        });
    }

    let classes = unique_classes(y_true, y_pred);
    let (tp, fp, fn_count) = per_class_counts(y_true, y_pred, &classes);

    aggregate_f1(&tp, &fp, &fn_count, y_true, &classes, average, "f1_score")
}

/// Compute the ROC AUC score for binary classification.
///
/// Uses the trapezoidal rule on the empirical ROC curve. Only binary
/// classification is supported: `y_true` must contain only labels `0` and `1`.
///
/// # Arguments
///
/// * `y_true`  — ground-truth binary labels (0 or 1).
/// * `y_score` — predicted probability or decision score for class 1.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if `y_true` contains labels
/// other than 0 and 1, or if there is only one class present.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::roc_auc_score;
/// use ndarray::array;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_score = array![0.1, 0.4, 0.35, 0.8];
/// let auc = roc_auc_score(&y_true, &y_score).unwrap();
/// assert!((auc - 0.75).abs() < 1e-10);
/// ```
pub fn roc_auc_score(y_true: &Array1<usize>, y_score: &Array1<f64>) -> Result<f64, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_score.len(), "roc_auc_score: y_true vs y_score")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "roc_auc_score".into(),
        });
    }

    // Validate that all labels are 0 or 1.
    for &label in y_true.iter() {
        if label > 1 {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "roc_auc_score requires binary labels (0 or 1), found label {label}"
                ),
            });
        }
    }

    let n_pos: usize = y_true.iter().filter(|&&v| v == 1).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err(FerroError::InvalidParameter {
            name: "y_true".into(),
            reason: "roc_auc_score requires at least one positive and one negative sample".into(),
        });
    }

    // Sort by descending score to trace the ROC curve.
    let mut pairs: Vec<(f64, usize)> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(&s, &t)| (s, t))
        .collect();
    // Stable sort descending by score; ties broken by label descending (positives first).
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.cmp(&a.1))
    });

    // Compute AUC via the trapezoidal rule on the ROC curve.
    // At each threshold, track cumulative TP and FP counts.
    let mut auc = 0.0_f64;
    let mut tp_prev = 0usize;
    let mut fp_prev = 0usize;
    let mut tp = 0usize;
    let mut fp = 0usize;

    let mut i = 0;
    while i < pairs.len() {
        // Consume all tied scores as one batch.
        let score = pairs[i].0;
        let batch_start_tp = tp;
        let batch_start_fp = fp;
        while i < pairs.len() && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        // Trapezoid area contribution.
        let _ = (batch_start_tp, batch_start_fp); // used below via tp_prev/fp_prev
        let _ = (tp_prev, fp_prev);
        auc += (fp as f64 - fp_prev as f64) * (tp as f64 + tp_prev as f64) / 2.0;
        tp_prev = tp;
        fp_prev = fp;
    }

    auc /= (n_pos * n_neg) as f64;
    Ok(auc)
}

/// Compute the confusion matrix.
///
/// The matrix `C` has shape `(n_classes, n_classes)` where `C[i, j]` is the
/// number of samples with true label `i` that were predicted as class `j`.
/// Classes are the sorted union of labels seen in `y_true` and `y_pred`.
///
/// # Arguments
///
/// * `y_true` — ground-truth class labels.
/// * `y_pred` — predicted class labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::confusion_matrix;
/// use ndarray::array;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 2, 0, 0, 2];
/// let cm = confusion_matrix(&y_true, &y_pred).unwrap();
/// assert_eq!(cm[[0, 0]], 2);
/// assert_eq!(cm[[1, 0]], 1);
/// assert_eq!(cm[[1, 2]], 1);
/// assert_eq!(cm[[2, 2]], 2);
/// ```
pub fn confusion_matrix(
    y_true: &Array1<usize>,
    y_pred: &Array1<usize>,
) -> Result<Array2<usize>, FerroError> {
    let n = y_true.len();
    check_same_length(n, y_pred.len(), "confusion_matrix: y_true vs y_pred")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "confusion_matrix".into(),
        });
    }

    let classes = unique_classes(y_true, y_pred);
    let k = classes.len();
    let mut matrix = Array2::<usize>::zeros((k, k));

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        // binary-search to find class index
        let row = classes.partition_point(|&c| c < t);
        let col = classes.partition_point(|&c| c < p);
        matrix[[row, col]] += 1;
    }

    Ok(matrix)
}

/// Compute the log-loss (cross-entropy loss) for probabilistic classifiers.
///
/// `log_loss = -1/n * sum_i sum_k y_{i,k} * log(p_{i,k})`
///
/// Labels in `y_true` are used as column indices into `y_prob`. Probabilities
/// are clipped to `[eps, 1-eps]` with `eps = 1e-15` to avoid `log(0)`.
///
/// # Arguments
///
/// * `y_true` — ground-truth class labels (integers from `0` to `n_classes-1`).
/// * `y_prob` — predicted probability matrix of shape `(n_samples, n_classes)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true.len()` does not equal
/// `y_prob.nrows()`.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples.
/// Returns [`FerroError::InvalidParameter`] if any label index is out of
/// bounds for the number of columns in `y_prob`.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::log_loss;
/// use ndarray::{array, Array2};
///
/// let y_true = array![0usize, 1, 1, 0];
/// let y_prob = Array2::from_shape_vec(
///     (4, 2),
///     vec![0.9, 0.1, 0.2, 0.8, 0.3, 0.7, 0.8, 0.2],
/// ).unwrap();
/// let loss = log_loss(&y_true, &y_prob).unwrap();
/// assert!(loss > 0.0);
/// ```
pub fn log_loss(y_true: &Array1<usize>, y_prob: &Array2<f64>) -> Result<f64, FerroError> {
    let n = y_true.len();
    if n != y_prob.nrows() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_prob.nrows()],
            context: "log_loss: y_true length vs y_prob rows".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "log_loss".into(),
        });
    }

    let n_classes = y_prob.ncols();
    const EPS: f64 = 1e-15;

    let mut total = 0.0_f64;
    for (i, &label) in y_true.iter().enumerate() {
        if label >= n_classes {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "label {label} at index {i} is out of bounds for y_prob with {n_classes} columns"
                ),
            });
        }
        let p = y_prob[[i, label]].clamp(EPS, 1.0 - EPS);
        total += p.ln();
    }

    Ok(-total / n as f64)
}

// ---------------------------------------------------------------------------
// Aggregation helpers (precision / recall / F1)
// ---------------------------------------------------------------------------

/// Aggregate per-class precision counts into a single score.
fn aggregate_metric(
    tp: &[usize],
    fp: &[usize],
    _fn_count: &[usize],
    y_true: &Array1<usize>,
    classes: &[usize],
    average: Average,
    context: &str,
) -> Result<f64, FerroError> {
    match average {
        Average::Binary => {
            if classes.len() != 2 {
                return Err(FerroError::InvalidParameter {
                    name: "average".into(),
                    reason: format!(
                        "{context}: Average::Binary requires exactly 2 classes, found {}",
                        classes.len()
                    ),
                });
            }
            // Positive class is the larger of the two (index 1).
            Ok(safe_div(tp[1] as f64, (tp[1] + fp[1]) as f64))
        }
        Average::Macro => {
            let sum: f64 = tp
                .iter()
                .zip(fp.iter())
                .map(|(&t, &f)| safe_div(t as f64, (t + f) as f64))
                .sum();
            Ok(sum / classes.len() as f64)
        }
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fp: usize = fp.iter().sum();
            Ok(safe_div(total_tp as f64, (total_tp + total_fp) as f64))
        }
        Average::Weighted => {
            let n = y_true.len() as f64;
            let mut weighted_sum = 0.0_f64;
            for (k, &c) in classes.iter().enumerate() {
                let support = y_true.iter().filter(|&&v| v == c).count();
                let prec = safe_div(tp[k] as f64, (tp[k] + fp[k]) as f64);
                weighted_sum += prec * support as f64;
            }
            Ok(weighted_sum / n)
        }
    }
}

/// Aggregate per-class recall counts into a single score.
fn aggregate_recall(
    tp: &[usize],
    _fp: &[usize],
    fn_count: &[usize],
    y_true: &Array1<usize>,
    classes: &[usize],
    average: Average,
    context: &str,
) -> Result<f64, FerroError> {
    match average {
        Average::Binary => {
            if classes.len() != 2 {
                return Err(FerroError::InvalidParameter {
                    name: "average".into(),
                    reason: format!(
                        "{context}: Average::Binary requires exactly 2 classes, found {}",
                        classes.len()
                    ),
                });
            }
            Ok(safe_div(tp[1] as f64, (tp[1] + fn_count[1]) as f64))
        }
        Average::Macro => {
            let sum: f64 = tp
                .iter()
                .zip(fn_count.iter())
                .map(|(&t, &f)| safe_div(t as f64, (t + f) as f64))
                .sum();
            Ok(sum / classes.len() as f64)
        }
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fn: usize = fn_count.iter().sum();
            Ok(safe_div(total_tp as f64, (total_tp + total_fn) as f64))
        }
        Average::Weighted => {
            let n = y_true.len() as f64;
            let mut weighted_sum = 0.0_f64;
            for (k, &c) in classes.iter().enumerate() {
                let support = y_true.iter().filter(|&&v| v == c).count();
                let rec = safe_div(tp[k] as f64, (tp[k] + fn_count[k]) as f64);
                weighted_sum += rec * support as f64;
            }
            Ok(weighted_sum / n)
        }
    }
}

/// Aggregate per-class F1 counts into a single score.
fn aggregate_f1(
    tp: &[usize],
    fp: &[usize],
    fn_count: &[usize],
    y_true: &Array1<usize>,
    classes: &[usize],
    average: Average,
    context: &str,
) -> Result<f64, FerroError> {
    match average {
        Average::Binary => {
            if classes.len() != 2 {
                return Err(FerroError::InvalidParameter {
                    name: "average".into(),
                    reason: format!(
                        "{context}: Average::Binary requires exactly 2 classes, found {}",
                        classes.len()
                    ),
                });
            }
            let prec = safe_div(tp[1] as f64, (tp[1] + fp[1]) as f64);
            let rec = safe_div(tp[1] as f64, (tp[1] + fn_count[1]) as f64);
            Ok(safe_div(2.0 * prec * rec, prec + rec))
        }
        Average::Macro => {
            let sum: f64 = tp
                .iter()
                .zip(fp.iter())
                .zip(fn_count.iter())
                .map(|((&t, &f_p), &f_n)| {
                    let prec = safe_div(t as f64, (t + f_p) as f64);
                    let rec = safe_div(t as f64, (t + f_n) as f64);
                    safe_div(2.0 * prec * rec, prec + rec)
                })
                .sum();
            Ok(sum / classes.len() as f64)
        }
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fp: usize = fp.iter().sum();
            let total_fn: usize = fn_count.iter().sum();
            let prec = safe_div(total_tp as f64, (total_tp + total_fp) as f64);
            let rec = safe_div(total_tp as f64, (total_tp + total_fn) as f64);
            Ok(safe_div(2.0 * prec * rec, prec + rec))
        }
        Average::Weighted => {
            let n = y_true.len() as f64;
            let mut weighted_sum = 0.0_f64;
            for (k, &c) in classes.iter().enumerate() {
                let support = y_true.iter().filter(|&&v| v == c).count();
                let prec = safe_div(tp[k] as f64, (tp[k] + fp[k]) as f64);
                let rec = safe_div(tp[k] as f64, (tp[k] + fn_count[k]) as f64);
                let f1 = safe_div(2.0 * prec * rec, prec + rec);
                weighted_sum += f1 * support as f64;
            }
            Ok(weighted_sum / n)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    // -----------------------------------------------------------------------
    // accuracy_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_accuracy_perfect() {
        let y_true = array![0usize, 1, 2, 1, 0];
        let y_pred = array![0usize, 1, 2, 1, 0];
        assert_abs_diff_eq!(
            accuracy_score(&y_true, &y_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_accuracy_partial() {
        let y_true = array![0usize, 1, 2, 1, 0];
        let y_pred = array![0usize, 1, 2, 0, 0]; // 4 correct out of 5
        assert_abs_diff_eq!(
            accuracy_score(&y_true, &y_pred).unwrap(),
            0.8,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_accuracy_zero() {
        let y_true = array![0usize, 1, 2];
        let y_pred = array![1usize, 2, 0];
        assert_abs_diff_eq!(
            accuracy_score(&y_true, &y_pred).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_accuracy_shape_mismatch() {
        let y_true = array![0usize, 1, 2];
        let y_pred = array![0usize, 1];
        assert!(accuracy_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_accuracy_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_pred = Array1::<usize>::from_vec(vec![]);
        assert!(accuracy_score(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // precision_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_precision_binary_perfect() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 1, 0, 1];
        assert_abs_diff_eq!(
            precision_score(&y_true, &y_pred, Average::Binary).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_precision_binary_partial() {
        // TP=2, FP=1 for class 1
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 1, 1];
        assert_abs_diff_eq!(
            precision_score(&y_true, &y_pred, Average::Binary).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_precision_macro() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        // Trace per sample:
        //   idx0: true=0,pred=0 → class0 TP++
        //   idx1: true=1,pred=2 → class1 FN++, class2 FP++
        //   idx2: true=2,pred=2 → class2 TP++
        //   idx3: true=0,pred=0 → class0 TP++
        //   idx4: true=1,pred=0 → class1 FN++, class0 FP++
        //   idx5: true=2,pred=2 → class2 TP++
        // class 0: TP=2 FP=1 => prec=2/3
        // class 1: TP=0 FP=0 => prec=0 (safe_div: 0/0=0)
        // class 2: TP=2 FP=1 => prec=2/3
        // macro = (2/3 + 0 + 2/3) / 3 = 4/9
        let expected = (2.0 / 3.0 + 0.0 + 2.0 / 3.0) / 3.0;
        assert_abs_diff_eq!(
            precision_score(&y_true, &y_pred, Average::Macro).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_precision_micro() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        // total TP=4 (2 for class0, 0 for class1, 2 for class2)
        // total FP=2 (class0 FP=1, class2 FP=1)
        // micro prec = 4 / (4+2) = 4/6 = 2/3
        assert_abs_diff_eq!(
            precision_score(&y_true, &y_pred, Average::Micro).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_precision_weighted() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        // prec class0=2/3 (support=2), class1=0 (support=2), class2=2/3 (support=2)
        let expected = (2.0 / 3.0 * 2.0 + 0.0 * 2.0 + 2.0 / 3.0 * 2.0) / 6.0;
        assert_abs_diff_eq!(
            precision_score(&y_true, &y_pred, Average::Weighted).unwrap(),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_precision_binary_error_multiclass() {
        let y_true = array![0usize, 1, 2];
        let y_pred = array![0usize, 1, 2];
        assert!(precision_score(&y_true, &y_pred, Average::Binary).is_err());
    }

    // -----------------------------------------------------------------------
    // recall_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_recall_binary_perfect() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 1, 0, 1];
        assert_abs_diff_eq!(
            recall_score(&y_true, &y_pred, Average::Binary).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_recall_binary_partial() {
        // TP=2, FN=1 for class 1
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 0, 1];
        assert_abs_diff_eq!(
            recall_score(&y_true, &y_pred, Average::Binary).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_recall_macro() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        // class0: TP=2,FN=0 => 1.0; class1: TP=0,FN=2 => 0.0; class2: TP=2,FN=0 => 1.0
        // Wait: class 2 has y_true=2 at indices 2 and 5. y_pred=2 at indices 2 and 5.
        // TP_2=2, FN_2=0 => recall=1.0
        // macro = (1+0+1)/3 = 2/3
        assert_abs_diff_eq!(
            recall_score(&y_true, &y_pred, Average::Macro).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_recall_micro() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        // TP=4 (2+0+2), FN=2 (0+2+0) => 4/6 = 2/3
        assert_abs_diff_eq!(
            recall_score(&y_true, &y_pred, Average::Micro).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-10
        );
    }

    // -----------------------------------------------------------------------
    // f1_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_f1_binary() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 0, 1];
        // precision=1.0, recall=2/3 => f1 = 2*(1*2/3)/(1+2/3) = (4/3)/(5/3) = 4/5
        assert_abs_diff_eq!(
            f1_score(&y_true, &y_pred, Average::Binary).unwrap(),
            0.8,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_f1_macro_perfect() {
        let y_true = array![0usize, 1, 2];
        let y_pred = array![0usize, 1, 2];
        assert_abs_diff_eq!(
            f1_score(&y_true, &y_pred, Average::Macro).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_f1_micro_equals_accuracy_for_balanced() {
        // For balanced classes, micro-F1 = accuracy.
        let y_true = array![0usize, 1, 0, 1];
        let y_pred = array![0usize, 1, 1, 0];
        let acc = accuracy_score(&y_true, &y_pred).unwrap();
        let f1 = f1_score(&y_true, &y_pred, Average::Micro).unwrap();
        assert_abs_diff_eq!(acc, f1, epsilon = 1e-10);
    }

    #[test]
    fn test_f1_weighted() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 0, 1];
        // Trace per sample:
        //   idx0: true=0,pred=0 → class0 TP++
        //   idx1: true=1,pred=1 → class1 TP++
        //   idx2: true=1,pred=0 → class0 FP++, class1 FN++
        //   idx3: true=0,pred=0 → class0 TP++
        //   idx4: true=1,pred=1 → class1 TP++
        // class0: TP=2 FP=1 FN=0 support=2 => prec=2/3 rec=1 f1=2*(2/3*1)/(2/3+1)=4/5
        // class1: TP=2 FP=0 FN=1 support=3 => prec=1 rec=2/3 f1=4/5
        // weighted = (4/5*2 + 4/5*3)/5 = (8/5+12/5)/5 = (20/5)/5 = 4/5 = 0.8
        assert_abs_diff_eq!(
            f1_score(&y_true, &y_pred, Average::Weighted).unwrap(),
            0.8,
            epsilon = 1e-10
        );
    }

    // -----------------------------------------------------------------------
    // roc_auc_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_roc_auc_basic() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1, 0.4, 0.35, 0.8];
        assert_abs_diff_eq!(
            roc_auc_score(&y_true, &y_score).unwrap(),
            0.75,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1, 0.2, 0.8, 0.9];
        assert_abs_diff_eq!(
            roc_auc_score(&y_true, &y_score).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_roc_auc_random() {
        // For reversed ordering: AUC should be 0.
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.9, 0.8, 0.2, 0.1];
        assert_abs_diff_eq!(
            roc_auc_score(&y_true, &y_score).unwrap(),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_roc_auc_invalid_label() {
        let y_true = array![0usize, 2, 1];
        let y_score = array![0.1, 0.5, 0.8];
        assert!(roc_auc_score(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_auc_only_one_class() {
        let y_true = array![1usize, 1, 1];
        let y_score = array![0.1, 0.5, 0.8];
        assert!(roc_auc_score(&y_true, &y_score).is_err());
    }

    // -----------------------------------------------------------------------
    // confusion_matrix
    // -----------------------------------------------------------------------

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = array![0usize, 1, 1, 0, 1];
        let y_pred = array![0usize, 1, 0, 0, 1];
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm[[0, 0]], 2); // TN
        assert_eq!(cm[[0, 1]], 0); // FP
        assert_eq!(cm[[1, 0]], 1); // FN
        assert_eq!(cm[[1, 1]], 2); // TP
    }

    #[test]
    fn test_confusion_matrix_multiclass() {
        let y_true = array![0usize, 1, 2, 0, 1, 2];
        let y_pred = array![0usize, 2, 2, 0, 0, 2];
        let cm = confusion_matrix(&y_true, &y_pred).unwrap();
        assert_eq!(cm[[0, 0]], 2);
        assert_eq!(cm[[1, 0]], 1);
        assert_eq!(cm[[1, 2]], 1);
        assert_eq!(cm[[2, 2]], 2);
    }

    #[test]
    fn test_confusion_matrix_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_pred = array![0usize, 1, 2];
        assert!(confusion_matrix(&y_true, &y_pred).is_err());
    }

    // -----------------------------------------------------------------------
    // log_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_loss_near_perfect() {
        let y_true = array![0usize, 1, 1, 0];
        let y_prob =
            Array2::from_shape_vec((4, 2), vec![0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1]).unwrap();
        let loss = log_loss(&y_true, &y_prob).unwrap();
        assert!(loss > 0.0);
        assert!(loss < 0.2); // very small for near-perfect predictions
    }

    #[test]
    fn test_log_loss_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_prob = Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.1, 0.9, 0.5, 0.5]).unwrap();
        assert!(log_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_log_loss_out_of_bounds_label() {
        let y_true = array![0usize, 5]; // 5 is out of bounds for 2 columns
        let y_prob = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap();
        assert!(log_loss(&y_true, &y_prob).is_err());
    }

    #[test]
    fn test_log_loss_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_prob = Array2::<f64>::zeros((0, 2));
        assert!(log_loss(&y_true, &y_prob).is_err());
    }
}
