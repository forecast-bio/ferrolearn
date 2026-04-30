//! Ranking evaluation metrics.
//!
//! This module provides ranking metrics commonly used to evaluate recommendation
//! and information retrieval systems:
//!
//! - [`dcg_score`] — Discounted Cumulative Gain
//! - [`ndcg_score`] — Normalized Discounted Cumulative Gain

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sort indices by descending `y_score`, breaking ties by index.
fn argsort_desc<F: Float>(arr: &Array1<F>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&a, &b| {
        arr[b]
            .partial_cmp(&arr[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    indices
}

/// Compute the raw DCG at position `k` given a relevance ordering.
fn compute_dcg<F: Float>(relevances: &[F], k: usize) -> F {
    let mut dcg = F::zero();
    for (i, &rel) in relevances.iter().take(k).enumerate() {
        // DCG_i = rel_i / log2(i + 2)
        let denom = F::from(i + 2).unwrap().log2();
        dcg = dcg + rel / denom;
    }
    dcg
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Discounted Cumulative Gain (DCG).
///
/// Items are ranked by descending `y_score`. The DCG is defined as:
///
/// ```text
/// DCG@k = sum_{i=0}^{k-1} y_true[ranked_i] / log2(i + 2)
/// ```
///
/// # Arguments
///
/// * `y_true` — relevance scores (ground-truth gains).
/// * `y_score` — predicted scores used to rank items.
/// * `k` — optional cutoff; if `None`, all items are used.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` have
/// different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::ranking::dcg_score;
/// use ndarray::array;
///
/// let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
/// let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let dcg = dcg_score(&y_true, &y_score, None).unwrap();
/// assert!(dcg > 0.0);
/// ```
pub fn dcg_score<F: Float + Send + Sync + 'static>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
    k: Option<usize>,
) -> Result<F, FerroError> {
    let n = y_true.len();
    if n != y_score.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_score.len()],
            context: "dcg_score: y_true vs y_score".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "dcg_score".into(),
        });
    }

    let k = k.unwrap_or(n).min(n);
    let ranked_indices = argsort_desc(y_score);
    let ranked_relevances: Vec<F> = ranked_indices.iter().map(|&i| y_true[i]).collect();

    Ok(compute_dcg(&ranked_relevances, k))
}

/// Compute the Normalized Discounted Cumulative Gain (NDCG).
///
/// NDCG is the ratio of the DCG to the ideal DCG (computed by sorting
/// `y_true` in descending order). NDCG is always in `[0, 1]` when
/// relevances are non-negative.
///
/// ```text
/// NDCG@k = DCG@k / ideal_DCG@k
/// ```
///
/// When the ideal DCG is zero (all relevances are zero), the NDCG is
/// defined as `0.0`.
///
/// # Arguments
///
/// * `y_true` — relevance scores (ground-truth gains).
/// * `y_score` — predicted scores used to rank items.
/// * `k` — optional cutoff; if `None`, all items are used.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` have
/// different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::ranking::ndcg_score;
/// use ndarray::array;
///
/// let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
/// let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
/// assert!(ndcg > 0.0 && ndcg <= 1.0);
///
/// // Perfect ranking yields NDCG = 1.0
/// let y_perfect = array![3.0_f64, 3.0, 2.0, 2.0, 1.0, 0.0];
/// let y_score_perf = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let ndcg_perf = ndcg_score(&y_perfect, &y_score_perf, None).unwrap();
/// assert!((ndcg_perf - 1.0).abs() < 1e-10);
/// ```
pub fn ndcg_score<F: Float + Send + Sync + 'static>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
    k: Option<usize>,
) -> Result<F, FerroError> {
    let n = y_true.len();
    if n != y_score.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_score.len()],
            context: "ndcg_score: y_true vs y_score".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "ndcg_score".into(),
        });
    }

    let k = k.unwrap_or(n).min(n);

    // Actual DCG: sort by y_score descending.
    let ranked_indices = argsort_desc(y_score);
    let ranked_relevances: Vec<F> = ranked_indices.iter().map(|&i| y_true[i]).collect();
    let dcg = compute_dcg(&ranked_relevances, k);

    // Ideal DCG: sort y_true descending.
    let mut ideal_relevances: Vec<F> = y_true.iter().copied().collect();
    ideal_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let ideal_dcg = compute_dcg(&ideal_relevances, k);

    if ideal_dcg == F::zero() {
        return Ok(F::zero());
    }

    Ok(dcg / ideal_dcg)
}

// ---------------------------------------------------------------------------
// Multilabel ranking metrics
// ---------------------------------------------------------------------------

fn check_ranking_inputs<F: Float>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
    context: &str,
) -> Result<(), FerroError> {
    if y_true.shape() != y_score.shape() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![y_true.nrows(), y_true.ncols()],
            actual: vec![y_score.nrows(), y_score.ncols()],
            context: format!("{context}: y_true and y_score must have the same shape"),
        });
    }
    if y_true.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    Ok(())
}

/// Compute the coverage error: how far down the ranked list we have to go
/// to cover all true labels, averaged over samples.
///
/// Lower is better; 1.0 is perfect (every true label sits at the top of the
/// score-sorted list for its sample).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn coverage_error<F>(y_true: &Array2<usize>, y_score: &Array2<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "coverage_error")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    let n_f = F::from(n_samples).ok_or_else(|| FerroError::InvalidParameter {
        name: "n_samples".into(),
        reason: "could not convert".into(),
    })?;

    let mut acc = F::zero();
    for i in 0..n_samples {
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        // Find the lowest score among the true (positive) labels for this row.
        let mut min_pos_score = F::infinity();
        let mut any = false;
        for j in 0..n_labels {
            if row_true[j] > 0 {
                any = true;
                if row_score[j] < min_pos_score {
                    min_pos_score = row_score[j];
                }
            }
        }
        if !any {
            // No positive labels — sklearn convention skips this sample.
            continue;
        }
        // Coverage = number of labels with score >= min_pos_score
        let mut cov = 0usize;
        for j in 0..n_labels {
            if row_score[j] >= min_pos_score {
                cov += 1;
            }
        }
        acc = acc + F::from(cov).unwrap_or(F::zero());
    }
    Ok(acc / n_f)
}

/// Compute the label-ranking average precision (LRAP) score.
///
/// For each sample, considers the rank of each positive label among all
/// labels and averages the per-positive-label precision. Higher is better;
/// `1.0` is perfect.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn label_ranking_average_precision_score<F>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "label_ranking_average_precision_score")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    let mut sample_scores: Vec<F> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        let positives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] > 0).collect();
        if positives.is_empty() || positives.len() == n_labels {
            // sklearn convention: degenerate samples (all-positive or
            // all-negative) score 1.0.
            sample_scores.push(F::one());
            continue;
        }
        let mut sample_acc = F::zero();
        for &pos in &positives {
            let pos_score = row_score[pos];
            // Rank of `pos` is the number of labels with score >= its own.
            let mut rank = 0usize;
            let mut tp = 0usize;
            for j in 0..n_labels {
                if row_score[j] >= pos_score {
                    rank += 1;
                    if row_true[j] > 0 {
                        tp += 1;
                    }
                }
            }
            sample_acc =
                sample_acc + F::from(tp).unwrap_or(F::zero()) / F::from(rank).unwrap_or(F::one());
        }
        sample_scores.push(sample_acc / F::from(positives.len()).unwrap_or(F::one()));
    }
    let n_f = F::from(sample_scores.len()).ok_or_else(|| FerroError::InvalidParameter {
        name: "n_samples".into(),
        reason: "could not convert".into(),
    })?;
    let total = sample_scores.iter().copied().fold(F::zero(), |a, b| a + b);
    Ok(total / n_f)
}

/// Compute the label-ranking loss: the average number of badly-ordered
/// label pairs per sample, normalised by the number of possible label pairs.
///
/// Lower is better; `0.0` is perfect (all positives outrank all negatives
/// for every sample).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn label_ranking_loss<F>(y_true: &Array2<usize>, y_score: &Array2<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "label_ranking_loss")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    let mut totals = F::zero();
    let mut counted = 0usize;
    for i in 0..n_samples {
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        let positives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] > 0).collect();
        let negatives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] == 0).collect();
        if positives.is_empty() || negatives.is_empty() {
            // Degenerate samples contribute 0 to the loss in sklearn.
            continue;
        }
        let mut bad_pairs = 0usize;
        for &p in &positives {
            for &n in &negatives {
                if row_score[p] <= row_score[n] {
                    bad_pairs += 1;
                }
            }
        }
        let denom = positives.len() * negatives.len();
        let frac = F::from(bad_pairs).unwrap_or(F::zero()) / F::from(denom).unwrap_or(F::one());
        totals = totals + frac;
        counted += 1;
    }
    if counted == 0 {
        return Ok(F::zero());
    }
    let n_f = F::from(counted).ok_or_else(|| FerroError::InvalidParameter {
        name: "counted".into(),
        reason: "could not convert".into(),
    })?;
    Ok(totals / n_f)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dcg_basic() {
        // y_true = [3, 2, 3, 0, 1, 2], y_score = [6, 5, 4, 3, 2, 1]
        // Ranking by y_score descending gives relevances: [3, 2, 3, 0, 1, 2]
        // DCG = 3/log2(0+2) + 2/log2(1+2) + 3/log2(2+2) + 0/log2(3+2) + 1/log2(4+2) + 2/log2(5+2)
        let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        let dcg = dcg_score(&y_true, &y_score, None).unwrap();
        let expected = 3.0 / 2.0_f64.log2()
            + 2.0 / 3.0_f64.log2()
            + 3.0 / 4.0_f64.log2()
            + 0.0 / 5.0_f64.log2()
            + 1.0 / 6.0_f64.log2()
            + 2.0 / 7.0_f64.log2();
        assert!((dcg - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dcg_with_k() {
        let y_true = array![3.0_f64, 2.0, 1.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let dcg_k2 = dcg_score(&y_true, &y_score, Some(2)).unwrap();
        // Only first 2: 3/log2(2) + 2/log2(3)
        let expected = 3.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2();
        assert!((dcg_k2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let y_true = array![3.0_f64, 2.0, 1.0, 0.0];
        let y_score = array![4.0_f64, 3.0, 2.0, 1.0];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_all_zero_relevance() {
        let y_true = array![0.0_f64, 0.0, 0.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_with_k() {
        let y_true = array![0.0_f64, 1.0, 3.0, 2.0];
        let y_score = array![1.0_f64, 4.0, 3.0, 2.0];
        // Ranking by y_score desc: indices [1, 2, 3, 0] -> relevances [1, 3, 2, 0]
        // DCG@2 = 1/log2(0+2) + 3/log2(1+2) = 1/1 + 3/log2(3)
        // Ideal order of relevances: [3, 2, 1, 0]
        // ideal DCG@2 = 3/log2(2) + 2/log2(3) = 3/1 + 2/log2(3)
        let ndcg = ndcg_score(&y_true, &y_score, Some(2)).unwrap();
        let dcg = 1.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2();
        let ideal = 3.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2();
        assert!((ndcg - dcg / ideal).abs() < 1e-10);
    }

    #[test]
    fn test_dcg_empty() {
        let y_true: Array1<f64> = Array1::zeros(0);
        let y_score: Array1<f64> = Array1::zeros(0);
        assert!(dcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_dcg_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_score = array![1.0_f64, 2.0, 3.0];
        assert!(dcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_ndcg_empty() {
        let y_true: Array1<f64> = Array1::zeros(0);
        let y_score: Array1<f64> = Array1::zeros(0);
        assert!(ndcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_ndcg_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_score = array![1.0_f64];
        assert!(ndcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_dcg_single_element() {
        let y_true = array![5.0_f64];
        let y_score = array![1.0_f64];
        let dcg = dcg_score(&y_true, &y_score, None).unwrap();
        // 5 / log2(0+2) = 5 / log2(2) = 5 / 1 = 5
        assert!((dcg - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_single_element() {
        let y_true = array![5.0_f64];
        let y_score = array![1.0_f64];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dcg_f32() {
        let y_true = array![3.0_f32, 2.0, 1.0];
        let y_score = array![3.0_f32, 2.0, 1.0];
        let dcg = dcg_score(&y_true, &y_score, None).unwrap();
        assert!(dcg > 0.0_f32);
    }

    #[test]
    fn test_ndcg_k_larger_than_n() {
        let y_true = array![3.0_f64, 2.0, 1.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let ndcg_all = ndcg_score(&y_true, &y_score, None).unwrap();
        let ndcg_big_k = ndcg_score(&y_true, &y_score, Some(100)).unwrap();
        assert!((ndcg_all - ndcg_big_k).abs() < 1e-10);
    }
}
