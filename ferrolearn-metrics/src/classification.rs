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
//! - [`roc_curve`] — compute ROC curve (FPR, TPR, thresholds)
//! - [`precision_recall_curve`] — compute precision-recall curve
//! - [`auc`] — area under an arbitrary curve via trapezoidal rule
//! - [`average_precision_score`] — average precision from precision-recall curve
//! - [`confusion_matrix`] — matrix of true/predicted class counts
//! - [`log_loss`] — cross-entropy loss for probabilistic classifiers

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Result type for curve functions that return `(x, y, thresholds)` arrays.
type CurveResult<F> = Result<(Array1<F>, Array1<F>, Array1<F>), FerroError>;

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
// ROC / Precision-Recall curve functions
// ---------------------------------------------------------------------------

/// Validate binary labels and count positives/negatives.
fn validate_binary_scores<F: Float + Send + Sync + 'static>(
    y_true: &Array1<usize>,
    y_score: &Array1<F>,
    context: &str,
) -> Result<(usize, usize), FerroError> {
    let n = y_true.len();
    check_same_length(n, y_score.len(), &format!("{context}: y_true vs y_score"))?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }

    for &label in y_true.iter() {
        if label > 1 {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "{context} requires binary labels (0 or 1), found label {label}"
                ),
            });
        }
    }

    let n_pos: usize = y_true.iter().filter(|&&v| v == 1).count();
    let n_neg = n - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return Err(FerroError::InvalidParameter {
            name: "y_true".into(),
            reason: format!(
                "{context} requires at least one positive and one negative sample"
            ),
        });
    }

    Ok((n_pos, n_neg))
}

/// Sort by descending score, returning `(score, label)` pairs.
fn sort_by_score_desc<F: Float>(
    y_true: &Array1<usize>,
    y_score: &Array1<F>,
) -> Vec<(F, usize)> {
    let mut pairs: Vec<(F, usize)> = y_score
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
    pairs
}

/// Compute the Receiver Operating Characteristic (ROC) curve.
///
/// Returns `(fpr, tpr, thresholds)` where `fpr` and `tpr` are the false
/// positive rate and true positive rate at each distinct threshold value.
/// The curve starts at `(0, 0)` with threshold `+inf` and ends at `(1, 1)`.
///
/// # Arguments
///
/// * `y_true`  — ground-truth binary labels (0 or 1).
/// * `y_score` — predicted probability or decision score for the positive class.
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
/// use ferrolearn_metrics::classification::roc_curve;
/// use ndarray::array;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
/// let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();
/// // First point is (0, 0) with threshold +inf
/// assert!((fpr[0] - 0.0).abs() < 1e-10);
/// assert!((tpr[0] - 0.0).abs() < 1e-10);
/// // Last point is (1, 1)
/// let last = fpr.len() - 1;
/// assert!((fpr[last] - 1.0).abs() < 1e-10);
/// assert!((tpr[last] - 1.0).abs() < 1e-10);
/// ```
pub fn roc_curve<F>(
    y_true: &Array1<usize>,
    y_score: &Array1<F>,
) -> CurveResult<F>
where
    F: Float + Send + Sync + 'static,
{
    let (n_pos, n_neg) = validate_binary_scores(y_true, y_score, "roc_curve")?;
    let pairs = sort_by_score_desc(y_true, y_score);

    let mut fpr_vec: Vec<F> = Vec::new();
    let mut tpr_vec: Vec<F> = Vec::new();
    let mut thresh_vec: Vec<F> = Vec::new();

    // Start at (0, 0) with threshold = +infinity.
    fpr_vec.push(F::zero());
    tpr_vec.push(F::zero());
    thresh_vec.push(F::infinity());

    let n_pos_f = F::from(n_pos).unwrap();
    let n_neg_f = F::from(n_neg).unwrap();

    let mut tp: usize = 0;
    let mut fp: usize = 0;
    let mut i = 0;

    while i < pairs.len() {
        let score = pairs[i].0;
        // Consume all tied scores as one batch.
        while i < pairs.len() && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        fpr_vec.push(F::from(fp).unwrap() / n_neg_f);
        tpr_vec.push(F::from(tp).unwrap() / n_pos_f);
        thresh_vec.push(score);
    }

    Ok((
        Array1::from_vec(fpr_vec),
        Array1::from_vec(tpr_vec),
        Array1::from_vec(thresh_vec),
    ))
}

/// Compute the precision-recall curve.
///
/// Returns `(precision, recall, thresholds)` where precision and recall are
/// computed at each distinct threshold in descending order. The last point
/// has recall=0 and precision=1 (by convention).
///
/// # Arguments
///
/// * `y_true`  — ground-truth binary labels (0 or 1).
/// * `y_score` — predicted probability or decision score for the positive class.
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
/// use ferrolearn_metrics::classification::precision_recall_curve;
/// use ndarray::array;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
/// let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();
/// // Last point has recall=0, precision=1
/// let last = precision.len() - 1;
/// assert!((precision[last] - 1.0).abs() < 1e-10);
/// assert!((recall[last] - 0.0).abs() < 1e-10);
/// ```
pub fn precision_recall_curve<F>(
    y_true: &Array1<usize>,
    y_score: &Array1<F>,
) -> CurveResult<F>
where
    F: Float + Send + Sync + 'static,
{
    let (n_pos, _n_neg) = validate_binary_scores(y_true, y_score, "precision_recall_curve")?;
    let pairs = sort_by_score_desc(y_true, y_score);

    let mut prec_vec: Vec<F> = Vec::new();
    let mut rec_vec: Vec<F> = Vec::new();
    let mut thresh_vec: Vec<F> = Vec::new();

    let n_pos_f = F::from(n_pos).unwrap();

    let mut tp: usize = 0;
    let mut fp: usize = 0;
    let mut i = 0;

    while i < pairs.len() {
        let score = pairs[i].0;
        // Consume all tied scores as one batch.
        while i < pairs.len() && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        let tp_f = F::from(tp).unwrap();
        let fp_f = F::from(fp).unwrap();
        let precision = if tp + fp > 0 {
            tp_f / (tp_f + fp_f)
        } else {
            F::zero()
        };
        let recall = tp_f / n_pos_f;
        prec_vec.push(precision);
        rec_vec.push(recall);
        thresh_vec.push(score);
    }

    // Reverse so that recall goes from high to low (sklearn convention: recall descending).
    // Then append the sentinel point: precision=1, recall=0.
    prec_vec.reverse();
    rec_vec.reverse();
    thresh_vec.reverse();

    prec_vec.push(F::one());
    rec_vec.push(F::zero());
    // thresholds array has one fewer element than precision/recall (sklearn convention).
    // We already reversed thresholds; don't append a sentinel threshold.

    Ok((
        Array1::from_vec(prec_vec),
        Array1::from_vec(rec_vec),
        Array1::from_vec(thresh_vec),
    ))
}

/// Compute the area under an arbitrary curve using the trapezoidal rule.
///
/// Given monotonic `x` and corresponding `y` values, this function computes
/// the area under the piecewise-linear curve through the points `(x[i], y[i])`.
///
/// # Arguments
///
/// * `x` — x-coordinates of the curve (assumed sorted).
/// * `y` — y-coordinates of the curve.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if there are fewer than 2 points.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::auc;
/// use ndarray::array;
///
/// let x = array![0.0_f64, 0.5, 1.0];
/// let y = array![0.0_f64, 0.5, 1.0];
/// let area = auc(&x, &y).unwrap();
/// assert!((area - 0.5).abs() < 1e-10);
/// ```
pub fn auc<F>(x: &Array1<F>, y: &Array1<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = x.len();
    if n != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y.len()],
            context: "auc: x vs y".into(),
        });
    }
    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "auc requires at least 2 points".into(),
        });
    }

    let two = F::from(2.0).unwrap();
    let mut area = F::zero();
    for i in 1..n {
        let dx = x[i] - x[i - 1];
        let avg_y = (y[i] + y[i - 1]) / two;
        area = area + dx * avg_y;
    }

    Ok(area)
}

/// Compute the average precision score from binary classification scores.
///
/// Average precision summarises the precision-recall curve as the weighted
/// mean of precisions at each threshold, with the increase in recall from
/// the previous threshold used as the weight:
///
/// `AP = sum_k (R_k - R_{k-1}) * P_k`
///
/// # Arguments
///
/// * `y_true`  — ground-truth binary labels (0 or 1).
/// * `y_score` — predicted probability or decision score for the positive class.
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
/// use ferrolearn_metrics::classification::average_precision_score;
/// use ndarray::array;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
/// let ap = average_precision_score(&y_true, &y_score).unwrap();
/// assert!(ap > 0.0 && ap <= 1.0);
/// ```
pub fn average_precision_score<F>(
    y_true: &Array1<usize>,
    y_score: &Array1<F>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (precision, recall, _thresholds) = precision_recall_curve(y_true, y_score)?;

    // AP = sum_k (R_{k-1} - R_k) * P_{k-1}
    // recall is in decreasing order (high recall first), so R_{k-1} >= R_k.
    // We iterate from k=1 to n-1.
    let mut ap = F::zero();
    for k in 1..recall.len() {
        let delta_recall = (recall[k - 1] - recall[k]).abs();
        ap = ap + delta_recall * precision[k - 1];
    }

    Ok(ap)
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
// top_k_accuracy_score
// ---------------------------------------------------------------------------

/// Compute the top-k accuracy score.
///
/// The top-k accuracy is the fraction of samples for which the true label is
/// among the `k` classes with the highest predicted score.
///
/// # Arguments
///
/// * `y_true` — ground-truth class labels (values in `0..n_classes`).
/// * `y_score` — predicted score matrix of shape `(n_samples, n_classes)`.
///   Each row contains a score for every class.
/// * `k` — number of top predictions to consider.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true.len() != y_score.nrows()`.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples.
/// Returns [`FerroError::InvalidParameter`] if `k == 0` or any label is out
/// of bounds for the number of columns in `y_score`.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::top_k_accuracy_score;
/// use ndarray::{array, Array2};
///
/// let y_true = array![0usize, 1, 2];
/// let y_score = Array2::from_shape_vec(
///     (3, 3),
///     vec![0.8, 0.1, 0.1,  // class 0 highest
///          0.1, 0.7, 0.2,  // class 1 highest
///          0.1, 0.3, 0.6], // class 2 highest
/// ).unwrap();
/// let acc = top_k_accuracy_score(&y_true, &y_score, 1).unwrap();
/// assert!((acc - 1.0).abs() < 1e-10);
/// ```
pub fn top_k_accuracy_score(
    y_true: &Array1<usize>,
    y_score: &Array2<f64>,
    k: usize,
) -> Result<f64, FerroError> {
    let n = y_true.len();
    if n != y_score.nrows() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_score.nrows()],
            context: "top_k_accuracy_score: y_true vs y_score".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "top_k_accuracy_score".into(),
        });
    }
    if k == 0 {
        return Err(FerroError::InvalidParameter {
            name: "k".into(),
            reason: "k must be >= 1".into(),
        });
    }

    let n_classes = y_score.ncols();
    let mut correct = 0usize;

    for (i, &label) in y_true.iter().enumerate() {
        if label >= n_classes {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "label {label} at index {i} is out of bounds for y_score with {n_classes} columns"
                ),
            });
        }

        // Get scores for this sample and find the top-k class indices.
        let row = y_score.row(i);
        let mut indices: Vec<usize> = (0..n_classes).collect();
        indices.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = &indices[..k.min(n_classes)];
        if top_k.contains(&label) {
            correct += 1;
        }
    }

    Ok(correct as f64 / n as f64)
}

// ---------------------------------------------------------------------------
// calibration_curve
// ---------------------------------------------------------------------------

/// Compute a calibration curve (reliability diagram data).
///
/// Predictions are binned into `n_bins` uniform bins. For each bin that
/// contains at least one sample, the function computes the fraction of
/// positive samples (class 1) and the mean predicted probability.
///
/// Bins that contain no samples are omitted from the output.
///
/// # Arguments
///
/// * `y_true` — binary ground-truth labels (0 or 1).
/// * `y_prob` — predicted probabilities for the positive class.
/// * `n_bins` — number of bins to divide the `[0, 1]` probability range.
///
/// # Returns
///
/// `(fraction_of_positives, mean_predicted_value)` — each is an `Array1<F>`
/// of length equal to the number of non-empty bins.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if array lengths differ.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples.
/// Returns [`FerroError::InvalidParameter`] if `n_bins == 0` or labels
/// are not binary (0 or 1).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::classification::calibration_curve;
/// use ndarray::array;
///
/// let y_true = array![0usize, 0, 1, 1, 1];
/// let y_prob = array![0.1_f64, 0.3, 0.6, 0.8, 0.9];
/// let (frac_pos, mean_pred) = calibration_curve(&y_true, &y_prob, 2).unwrap();
/// // Bin [0, 0.5): samples with probs 0.1, 0.3 -> frac_pos = 0/2, mean_pred = 0.2
/// // Bin [0.5, 1]: samples with probs 0.6, 0.8, 0.9 -> frac_pos = 3/3, mean_pred = 0.7667
/// assert_eq!(frac_pos.len(), 2);
/// ```
pub fn calibration_curve<F>(
    y_true: &Array1<usize>,
    y_prob: &Array1<F>,
    n_bins: usize,
) -> Result<(Array1<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = y_true.len();
    if n != y_prob.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_prob.len()],
            context: "calibration_curve: y_true vs y_prob".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "calibration_curve".into(),
        });
    }
    if n_bins == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_bins".into(),
            reason: "n_bins must be >= 1".into(),
        });
    }

    // Validate binary labels.
    for &label in y_true.iter() {
        if label > 1 {
            return Err(FerroError::InvalidParameter {
                name: "y_true".into(),
                reason: format!(
                    "calibration_curve requires binary labels (0 or 1), found {label}"
                ),
            });
        }
    }

    let n_bins_f = F::from(n_bins).unwrap();
    let mut bin_pos_sum = vec![F::zero(); n_bins];
    let mut bin_prob_sum = vec![F::zero(); n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for (&label, &prob) in y_true.iter().zip(y_prob.iter()) {
        // Determine bin index: floor(prob * n_bins), clamped to [0, n_bins-1].
        let mut bin = (prob * n_bins_f).to_usize().unwrap_or(0);
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        bin_count[bin] += 1;
        bin_prob_sum[bin] = bin_prob_sum[bin] + prob;
        if label == 1 {
            bin_pos_sum[bin] = bin_pos_sum[bin] + F::one();
        }
    }

    // Collect non-empty bins.
    let mut frac_positives = Vec::new();
    let mut mean_predictions = Vec::new();

    for bin in 0..n_bins {
        if bin_count[bin] > 0 {
            let count_f = F::from(bin_count[bin]).unwrap();
            frac_positives.push(bin_pos_sum[bin] / count_f);
            mean_predictions.push(bin_prob_sum[bin] / count_f);
        }
    }

    Ok((
        Array1::from_vec(frac_positives),
        Array1::from_vec(mean_predictions),
    ))
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

    // -----------------------------------------------------------------------
    // roc_curve
    // -----------------------------------------------------------------------

    #[test]
    fn test_roc_curve_basic() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
        let (fpr, tpr, thresholds) = roc_curve(&y_true, &y_score).unwrap();

        // First point: (0, 0) with threshold = +inf
        assert_abs_diff_eq!(fpr[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[0], 0.0, epsilon = 1e-10);
        assert!(thresholds[0].is_infinite());

        // Last point: (1, 1)
        let last = fpr.len() - 1;
        assert_abs_diff_eq!(fpr[last], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tpr[last], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_curve_perfect() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.2, 0.8, 0.9];
        let (fpr, tpr, _thresholds) = roc_curve(&y_true, &y_score).unwrap();

        // Perfect classifier: should go (0,0) -> (0,0.5) -> (0,1) -> ...
        // FPR stays 0 while TPR increases.
        // Verify there's a point with FPR=0, TPR=1.
        let has_perfect_point = fpr.iter().zip(tpr.iter()).any(|(&f, &t)| {
            (f - 0.0).abs() < 1e-10 && (t - 1.0).abs() < 1e-10
        });
        assert!(has_perfect_point);
    }

    #[test]
    fn test_roc_curve_auc_consistency() {
        // AUC computed from roc_curve should match roc_auc_score.
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
        let (fpr, tpr, _thresholds) = roc_curve(&y_true, &y_score).unwrap();
        let area = auc(&fpr, &tpr).unwrap();
        let auc_direct = roc_auc_score(&y_true, &y_score).unwrap();
        assert_abs_diff_eq!(area, auc_direct, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_curve_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_score = Array1::<f64>::from_vec(vec![]);
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_single_class() {
        let y_true = array![1usize, 1, 1];
        let y_score = array![0.5_f64, 0.6, 0.7];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_invalid_label() {
        let y_true = array![0usize, 2, 1];
        let y_score = array![0.1_f64, 0.5, 0.8];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_roc_curve_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_score = array![0.1_f64, 0.5, 0.8];
        assert!(roc_curve(&y_true, &y_score).is_err());
    }

    // -----------------------------------------------------------------------
    // precision_recall_curve
    // -----------------------------------------------------------------------

    #[test]
    fn test_precision_recall_curve_basic() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
        let (precision, recall, _thresholds) =
            precision_recall_curve(&y_true, &y_score).unwrap();

        // Last point is (precision=1, recall=0) sentinel.
        let last = precision.len() - 1;
        assert_abs_diff_eq!(precision[last], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(recall[last], 0.0, epsilon = 1e-10);

        // All precisions should be in [0, 1].
        for &p in precision.iter() {
            assert!(p >= 0.0 && p <= 1.0 + 1e-10);
        }
        // All recalls should be in [0, 1].
        for &r in recall.iter() {
            assert!(r >= -1e-10 && r <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_precision_recall_curve_perfect() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.2, 0.8, 0.9];
        let (precision, recall, _thresholds) =
            precision_recall_curve(&y_true, &y_score).unwrap();

        // For a perfect classifier, at the highest threshold both predictions
        // are positives and correct.
        // First recall values should include 1.0 with high precision.
        let first_recall = recall[0];
        let first_precision = precision[0];
        // At the lowest threshold everything is predicted positive.
        assert_abs_diff_eq!(first_recall, 1.0, epsilon = 1e-10);
        assert!(first_precision > 0.0);
    }

    #[test]
    fn test_precision_recall_curve_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_score = Array1::<f64>::from_vec(vec![]);
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_precision_recall_curve_single_class() {
        let y_true = array![0usize, 0, 0];
        let y_score = array![0.1_f64, 0.5, 0.9];
        assert!(precision_recall_curve(&y_true, &y_score).is_err());
    }

    // -----------------------------------------------------------------------
    // auc
    // -----------------------------------------------------------------------

    #[test]
    fn test_auc_triangle() {
        // Triangle: (0,0) -> (1,1) => area = 0.5
        let x = array![0.0_f64, 1.0];
        let y = array![0.0_f64, 1.0];
        assert_abs_diff_eq!(auc(&x, &y).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_auc_rectangle() {
        // Rectangle: (0,1) -> (1,1) => area = 1.0
        let x = array![0.0_f64, 1.0];
        let y = array![1.0_f64, 1.0];
        assert_abs_diff_eq!(auc(&x, &y).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_auc_trapezoid() {
        // (0,0) -> (0.5, 0.5) -> (1.0, 1.0) => area = 0.5
        let x = array![0.0_f64, 0.5, 1.0];
        let y = array![0.0_f64, 0.5, 1.0];
        assert_abs_diff_eq!(auc(&x, &y).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_auc_shape_mismatch() {
        let x = array![0.0_f64, 1.0];
        let y = array![0.0_f64, 0.5, 1.0];
        assert!(auc(&x, &y).is_err());
    }

    #[test]
    fn test_auc_single_point() {
        let x = array![0.0_f64];
        let y = array![1.0_f64];
        assert!(auc(&x, &y).is_err());
    }

    #[test]
    fn test_auc_f32() {
        let x = array![0.0_f32, 1.0];
        let y = array![0.0_f32, 1.0];
        assert_abs_diff_eq!(auc(&x, &y).unwrap(), 0.5_f32, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // average_precision_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_average_precision_basic() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.4, 0.35, 0.8];
        let ap = average_precision_score(&y_true, &y_score).unwrap();
        assert!(ap > 0.0);
        assert!(ap <= 1.0);
    }

    #[test]
    fn test_average_precision_perfect() {
        let y_true = array![0usize, 0, 1, 1];
        let y_score = array![0.1_f64, 0.2, 0.8, 0.9];
        let ap = average_precision_score(&y_true, &y_score).unwrap();
        // Perfect ranking: AP should be 1.0.
        assert_abs_diff_eq!(ap, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_precision_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_score = Array1::<f64>::from_vec(vec![]);
        assert!(average_precision_score(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_average_precision_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_score = array![0.1_f64, 0.5, 0.8];
        assert!(average_precision_score(&y_true, &y_score).is_err());
    }

    #[test]
    fn test_average_precision_single_class() {
        let y_true = array![1usize, 1, 1];
        let y_score = array![0.5_f64, 0.6, 0.7];
        assert!(average_precision_score(&y_true, &y_score).is_err());
    }
}

// ---------------------------------------------------------------------------
// Kani formal verification harnesses
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Prove that accuracy_score output is in [0.0, 1.0] for any non-empty
    /// input with matching lengths and valid labels.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_accuracy_score_range() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 3;
            y_pred_data[i] = kani::any::<usize>() % 3;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = accuracy_score(&y_true, &y_pred);
        if let Ok(acc) = result {
            assert!(acc >= 0.0, "accuracy must be >= 0.0");
            assert!(acc <= 1.0, "accuracy must be <= 1.0");
        }
    }

    /// Prove that precision_score output is in [0.0, 1.0] for binary labels
    /// with Macro averaging (covers all code paths including zero denominator).
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_precision_score_range() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 3;
            y_pred_data[i] = kani::any::<usize>() % 3;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = precision_score(&y_true, &y_pred, Average::Macro);
        if let Ok(prec) = result {
            assert!(prec >= 0.0, "precision must be >= 0.0");
            assert!(prec <= 1.0, "precision must be <= 1.0");
        }
    }

    /// Prove that precision_score does not panic on zero denominator
    /// (all predictions wrong class) with Binary averaging.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_precision_no_panic_zero_denom() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 2;
            y_pred_data[i] = kani::any::<usize>() % 2;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        // Must not panic regardless of input — may return Ok or Err.
        let _ = precision_score(&y_true, &y_pred, Average::Binary);
    }

    /// Prove that recall_score output is in [0.0, 1.0] with Macro averaging.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_recall_score_range() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 3;
            y_pred_data[i] = kani::any::<usize>() % 3;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = recall_score(&y_true, &y_pred, Average::Macro);
        if let Ok(rec) = result {
            assert!(rec >= 0.0, "recall must be >= 0.0");
            assert!(rec <= 1.0, "recall must be <= 1.0");
        }
    }

    /// Prove that recall_score does not panic on zero denominator
    /// (no true positives for a class) with Binary averaging.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_recall_no_panic_zero_denom() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 2;
            y_pred_data[i] = kani::any::<usize>() % 2;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        // Must not panic regardless of input.
        let _ = recall_score(&y_true, &y_pred, Average::Binary);
    }

    /// Prove that f1_score output is in [0.0, 1.0] with Macro averaging.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_f1_score_range() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 3;
            y_pred_data[i] = kani::any::<usize>() % 3;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = f1_score(&y_true, &y_pred, Average::Macro);
        if let Ok(f1) = result {
            assert!(f1 >= 0.0, "f1 must be >= 0.0");
            assert!(f1 <= 1.0, "f1 must be <= 1.0");
        }
    }

    /// Prove that log_loss output is >= 0.0 and not NaN for valid inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_log_loss_non_negative_no_nan() {
        const N: usize = 4;
        const C: usize = 2;

        let mut y_true_data = [0usize; N];
        let mut y_prob_data = [0.0f64; N * C];

        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % C;
        }

        for i in 0..(N * C) {
            let val: f64 = kani::any();
            kani::assume(!val.is_nan() && !val.is_infinite());
            kani::assume(val >= 0.0 && val <= 1.0);
            y_prob_data[i] = val;
        }

        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_prob = Array2::from_shape_vec((N, C), y_prob_data.to_vec()).unwrap();

        let result = log_loss(&y_true, &y_prob);
        if let Ok(loss) = result {
            assert!(loss >= 0.0, "log_loss must be >= 0.0");
            assert!(!loss.is_nan(), "log_loss must not be NaN");
        }
    }

    /// Prove that all entries in the confusion matrix are >= 0.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_confusion_matrix_non_negative() {
        const N: usize = 4;
        let mut y_true_data = [0usize; N];
        let mut y_pred_data = [0usize; N];
        for i in 0..N {
            y_true_data[i] = kani::any::<usize>() % 3;
            y_pred_data[i] = kani::any::<usize>() % 3;
        }
        let y_true = Array1::from_vec(y_true_data.to_vec());
        let y_pred = Array1::from_vec(y_pred_data.to_vec());

        let result = confusion_matrix(&y_true, &y_pred);
        if let Ok(cm) = result {
            // usize entries are always >= 0 by type, but verify the matrix
            // is well-formed and entries sum to N.
            let total: usize = cm.iter().sum();
            assert!(total == N, "confusion matrix entries must sum to N");
            // All entries are >= 0 by construction (usize), but explicitly check
            // the invariant for documentation.
            for &entry in cm.iter() {
                assert!(entry <= N, "no entry can exceed total sample count");
            }
        }
    }

    // -----------------------------------------------------------------------
    // top_k_accuracy_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_k_accuracy_perfect_top1() {
        let y_true = array![0usize, 1, 2];
        let y_score = Array2::from_shape_vec(
            (3, 3),
            vec![0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8],
        )
        .unwrap();
        let acc = top_k_accuracy_score(&y_true, &y_score, 1).unwrap();
        assert_abs_diff_eq!(acc, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_top_k_accuracy_top2() {
        let y_true = array![0usize, 1, 2];
        let y_score = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.8, 0.1, 0.1, // correct top-1
                0.1, 0.7, 0.2, // correct top-1
                0.3, 0.4, 0.3, // class 2 not in top-1 but in top-2
            ],
        )
        .unwrap();
        // top-1: samples 0,1 correct, sample 2 wrong -> 2/3
        let acc_1 = top_k_accuracy_score(&y_true, &y_score, 1).unwrap();
        assert_abs_diff_eq!(acc_1, 2.0 / 3.0, epsilon = 1e-10);
        // top-2: all correct
        let acc_2 = top_k_accuracy_score(&y_true, &y_score, 2).unwrap();
        assert_abs_diff_eq!(acc_2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_top_k_accuracy_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_score = Array2::<f64>::zeros((0, 3));
        assert!(top_k_accuracy_score(&y_true, &y_score, 1).is_err());
    }

    #[test]
    fn test_top_k_accuracy_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_score = Array2::<f64>::zeros((3, 2));
        assert!(top_k_accuracy_score(&y_true, &y_score, 1).is_err());
    }

    #[test]
    fn test_top_k_accuracy_k_zero() {
        let y_true = array![0usize, 1];
        let y_score = Array2::<f64>::zeros((2, 2));
        assert!(top_k_accuracy_score(&y_true, &y_score, 0).is_err());
    }

    #[test]
    fn test_top_k_accuracy_label_out_of_bounds() {
        let y_true = array![0usize, 5];
        let y_score = Array2::<f64>::zeros((2, 3));
        assert!(top_k_accuracy_score(&y_true, &y_score, 1).is_err());
    }

    #[test]
    fn test_top_k_accuracy_k_equals_n_classes() {
        // When k = n_classes, all samples should be correct
        let y_true = array![0usize, 1, 2];
        let y_score = Array2::from_shape_vec(
            (3, 3),
            vec![0.1, 0.2, 0.7, 0.5, 0.3, 0.2, 0.3, 0.3, 0.4],
        )
        .unwrap();
        let acc = top_k_accuracy_score(&y_true, &y_score, 3).unwrap();
        assert_abs_diff_eq!(acc, 1.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // calibration_curve
    // -----------------------------------------------------------------------

    #[test]
    fn test_calibration_curve_basic() {
        let y_true = array![0usize, 0, 1, 1, 1];
        let y_prob = array![0.1_f64, 0.3, 0.6, 0.8, 0.9];
        let (frac_pos, mean_pred) = calibration_curve(&y_true, &y_prob, 2).unwrap();
        // Bin [0, 0.5): probs 0.1, 0.3 -> frac_pos=0/2=0, mean_pred=(0.1+0.3)/2=0.2
        // Bin [0.5, 1.0]: probs 0.6, 0.8, 0.9 -> frac_pos=3/3=1, mean_pred=(0.6+0.8+0.9)/3
        assert_eq!(frac_pos.len(), 2);
        assert_abs_diff_eq!(frac_pos[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(frac_pos[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean_pred[0], 0.2, epsilon = 1e-10);
        let expected_mean = (0.6 + 0.8 + 0.9) / 3.0;
        assert_abs_diff_eq!(mean_pred[1], expected_mean, epsilon = 1e-10);
    }

    #[test]
    fn test_calibration_curve_empty_bins() {
        // All predictions in one bin
        let y_true = array![1usize, 0, 1];
        let y_prob = array![0.1_f64, 0.2, 0.3];
        let (frac_pos, _mean_pred) = calibration_curve(&y_true, &y_prob, 5).unwrap();
        // Empty bins are omitted
        assert!(frac_pos.len() <= 5);
        assert!(frac_pos.len() >= 1);
    }

    #[test]
    fn test_calibration_curve_empty() {
        let y_true = Array1::<usize>::from_vec(vec![]);
        let y_prob = Array1::<f64>::from_vec(vec![]);
        assert!(calibration_curve(&y_true, &y_prob, 5).is_err());
    }

    #[test]
    fn test_calibration_curve_shape_mismatch() {
        let y_true = array![0usize, 1];
        let y_prob = array![0.1_f64, 0.2, 0.3];
        assert!(calibration_curve(&y_true, &y_prob, 5).is_err());
    }

    #[test]
    fn test_calibration_curve_zero_bins() {
        let y_true = array![0usize, 1];
        let y_prob = array![0.1_f64, 0.9];
        assert!(calibration_curve(&y_true, &y_prob, 0).is_err());
    }

    #[test]
    fn test_calibration_curve_non_binary() {
        let y_true = array![0usize, 2];
        let y_prob = array![0.1_f64, 0.9];
        assert!(calibration_curve(&y_true, &y_prob, 5).is_err());
    }

    #[test]
    fn test_calibration_curve_f32() {
        let y_true = array![0usize, 1, 1];
        let y_prob = array![0.2_f32, 0.7, 0.9];
        let (frac_pos, mean_pred) = calibration_curve(&y_true, &y_prob, 2).unwrap();
        assert!(!frac_pos.is_empty());
        assert!(!mean_pred.is_empty());
    }
}
