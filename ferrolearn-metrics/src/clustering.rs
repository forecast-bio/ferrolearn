//! Clustering evaluation metrics.
//!
//! This module provides standard clustering metrics used to evaluate the
//! quality of unsupervised clustering results:
//!
//! - [`silhouette_score`] — mean silhouette coefficient across all samples
//! - [`silhouette_samples`] — per-sample silhouette coefficients
//! - [`adjusted_rand_score`] — Adjusted Rand Index comparing two labelings
//! - [`adjusted_mutual_info`] — Adjusted Mutual Information between two labelings
//! - [`davies_bouldin_score`] — Davies-Bouldin index for cluster separation
//! - [`calinski_harabasz_score`] — Variance Ratio Criterion
//! - [`homogeneity_score`] — each cluster contains only one class
//! - [`completeness_score`] — all samples of a class are in one cluster
//! - [`v_measure_score`] — harmonic mean of homogeneity and completeness
//!
//! Noise points (label == -1, as used by DBSCAN) are excluded from all
//! silhouette and Calinski-Harabasz computations but are counted in
//! contingency-based metrics.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that two label arrays have the same length.
fn check_labels_same_length(n_a: usize, n_b: usize, context: &str) -> Result<(), FerroError> {
    if n_a != n_b {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_a],
            actual: vec![n_b],
            context: context.into(),
        });
    }
    Ok(())
}

/// Validate that `x` (n_samples, n_features) and `labels` (n_samples,)
/// have compatible lengths.
fn check_x_labels_compat(
    n_samples: usize,
    n_labels: usize,
    context: &str,
) -> Result<(), FerroError> {
    if n_samples != n_labels {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![n_labels],
            context: context.into(),
        });
    }
    Ok(())
}

/// Euclidean distance between two row views represented as slices.
#[inline]
fn euclidean_dist<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| {
            let d = ai - bi;
            acc + d * d
        })
        .sqrt()
}

/// Return a sorted, deduplicated list of non-noise cluster labels.
fn unique_cluster_labels(labels: &Array1<isize>) -> Vec<isize> {
    let mut v: Vec<isize> = labels.iter().copied().filter(|&l| l != -1).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// n choose 2.
#[inline]
fn n_choose_2(n: u64) -> u64 {
    if n < 2 { 0 } else { n * (n - 1) / 2 }
}

// ---------------------------------------------------------------------------
// silhouette_score
// ---------------------------------------------------------------------------

/// Compute the mean Silhouette Coefficient for all non-noise samples.
///
/// For each sample `i` belonging to cluster `C_i`:
/// - `a(i)` = mean distance from `i` to all other samples in `C_i`
/// - `b(i)` = mean distance from `i` to samples in the nearest other cluster
/// - `s(i)` = `(b(i) - a(i)) / max(a(i), b(i))`
///
/// The score returned is the mean of `s(i)` over all non-noise samples.
///
/// Noise points (label == -1) are ignored entirely.
///
/// # Arguments
///
/// * `x`      — feature matrix of shape `(n_samples, n_features)`.
/// * `labels` — cluster label for each sample. Use `-1` for noise.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != labels.len()`.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples or all
/// samples are noise.
/// Returns [`FerroError::InvalidParameter`] if there is only one cluster
/// (after excluding noise).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::silhouette_score;
/// use ndarray::{array, Array2};
///
/// // Two well-separated clusters.
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     0.1, 0.1,
///     10.0, 10.0,
///     10.1, 10.1,
/// ]).unwrap();
/// let labels = array![0isize, 0, 1, 1];
/// let score = silhouette_score(&x, &labels).unwrap();
/// assert!(score > 0.9);
/// ```
pub fn silhouette_score<F>(x: &Array2<F>, labels: &Array1<isize>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = x.nrows();
    check_x_labels_compat(n, labels.len(), "silhouette_score: x rows vs labels")?;

    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "silhouette_score".into(),
        });
    }

    let cluster_labels = unique_cluster_labels(labels);
    let n_clusters = cluster_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: format!(
                "silhouette_score requires at least 2 clusters (excluding noise), found {n_clusters}"
            ),
        });
    }

    // Pre-compute cluster membership lists (indices of samples per cluster).
    let cluster_indices: Vec<Vec<usize>> = cluster_labels
        .iter()
        .map(|&cl| {
            labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == cl { Some(i) } else { None })
                .collect()
        })
        .collect();

    // Map from cluster label to its position in `cluster_labels`.
    let label_to_idx = |lbl: isize| -> Option<usize> {
        cluster_labels
            .partition_point(|&c| c < lbl)
            .let_if(|&pos| pos < cluster_labels.len() && cluster_labels[pos] == lbl)
    };

    let mut sum_s = F::zero();
    let mut count = 0usize;

    for i in 0..n {
        let li = labels[i];
        if li == -1 {
            continue; // skip noise
        }

        let ci_idx = match label_to_idx(li) {
            Some(idx) => idx,
            None => continue,
        };

        let ci_members = &cluster_indices[ci_idx];

        // a(i): mean intra-cluster distance (exclude self)
        let a_i = if ci_members.len() <= 1 {
            F::zero()
        } else {
            let mut dist_sum = F::zero();
            for &j in ci_members {
                if j == i {
                    continue;
                }
                dist_sum = dist_sum + row_euclidean_dist(x, i, j);
            }
            dist_sum / F::from(ci_members.len() - 1).unwrap()
        };

        // b(i): mean distance to the nearest other cluster
        let mut b_i = F::infinity();
        for (k, &cl_k) in cluster_labels.iter().enumerate() {
            if cl_k == li {
                continue; // same cluster
            }
            let other_members = &cluster_indices[k];
            if other_members.is_empty() {
                continue;
            }
            let mut dist_sum = F::zero();
            for &j in other_members {
                dist_sum = dist_sum + row_euclidean_dist(x, i, j);
            }
            let mean_dist = dist_sum / F::from(other_members.len()).unwrap();
            if mean_dist < b_i {
                b_i = mean_dist;
            }
        }

        // s(i) = (b - a) / max(a, b)
        let max_ab = if a_i > b_i { a_i } else { b_i };
        let s_i = if max_ab == F::zero() {
            F::zero()
        } else {
            (b_i - a_i) / max_ab
        };

        sum_s = sum_s + s_i;
        count += 1;
    }

    if count == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "silhouette_score: all samples are noise (label == -1)".into(),
        });
    }

    Ok(sum_s / F::from(count).unwrap())
}

/// Euclidean distance between row `i` and row `j` of a 2D array.
fn row_euclidean_dist<F: Float>(x: &Array2<F>, i: usize, j: usize) -> F {
    let n_features = x.ncols();
    let mut sq_sum = F::zero();
    for f in 0..n_features {
        let d = x[[i, f]] - x[[j, f]];
        sq_sum = sq_sum + d * d;
    }
    sq_sum.sqrt()
}

// Helper trait to make partition_point + check more ergonomic via closure.
trait LetIf: Sized {
    fn let_if(self, pred: impl FnOnce(&Self) -> bool) -> Option<Self>;
}
impl LetIf for usize {
    fn let_if(self, pred: impl FnOnce(&Self) -> bool) -> Option<Self> {
        if pred(&self) { Some(self) } else { None }
    }
}

// ---------------------------------------------------------------------------
// adjusted_rand_score
// ---------------------------------------------------------------------------

/// Compute the Adjusted Rand Index (ARI) between two clusterings.
///
/// ARI measures the similarity between two label assignments, corrected for
/// chance. A score of `1.0` means perfect agreement; `0.0` is the expected
/// value for random labelings.
///
/// The combinatorial formula used is:
///
/// ```text
/// ARI = (sum_ij C(n_ij, 2) - (sum_i C(a_i, 2) * sum_j C(b_j, 2)) / C(n, 2))
///       / ((sum_i C(a_i, 2) + sum_j C(b_j, 2)) / 2
///          - (sum_i C(a_i, 2) * sum_j C(b_j, 2)) / C(n, 2))
/// ```
///
/// where `n_ij` is the contingency table entry, `a_i` is the row sum, and
/// `b_j` is the column sum.
///
/// # Arguments
///
/// * `labels_true` — ground-truth cluster labels.
/// * `labels_pred` — predicted cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::adjusted_rand_score;
/// use ndarray::array;
///
/// let labels = array![0isize, 0, 1, 1, 2, 2];
/// let ari = adjusted_rand_score(&labels, &labels).unwrap();
/// assert!((ari - 1.0).abs() < 1e-10);
/// ```
pub fn adjusted_rand_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "adjusted_rand_score")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "adjusted_rand_score".into(),
        });
    }

    // Collect unique sorted labels for each.
    let mut classes_true: Vec<isize> = labels_true.iter().copied().collect();
    classes_true.sort_unstable();
    classes_true.dedup();

    let mut classes_pred: Vec<isize> = labels_pred.iter().copied().collect();
    classes_pred.sort_unstable();
    classes_pred.dedup();

    let r = classes_true.len();
    let s = classes_pred.len();

    // Build contingency table.
    let mut contingency = vec![vec![0u64; s]; r];
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = classes_true.partition_point(|&c| c < lt);
        let ci = classes_pred.partition_point(|&c| c < lp);
        contingency[ri][ci] += 1;
    }

    // Row sums a_i and column sums b_j.
    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<u64> = (0..s)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    let n_u64 = n as u64;
    let sum_comb_c: u64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| n_choose_2(v))
        .sum();
    let sum_comb_a: u64 = a.iter().map(|&ai| n_choose_2(ai)).sum();
    let sum_comb_b: u64 = b.iter().map(|&bj| n_choose_2(bj)).sum();
    let comb_n = n_choose_2(n_u64);

    if comb_n == 0 {
        // Only one sample — convention: ARI = 1 if labels agree, else 0.
        return Ok(if labels_true[0] == labels_pred[0] {
            1.0
        } else {
            0.0
        });
    }

    let prod_ab = sum_comb_a as f64 * sum_comb_b as f64;
    let expected = prod_ab / comb_n as f64;
    let max_val = f64::midpoint(sum_comb_a as f64, sum_comb_b as f64);
    let numerator = sum_comb_c as f64 - expected;
    let denominator = max_val - expected;

    if denominator == 0.0 {
        // Degenerate case: all samples in one cluster or all in separate clusters.
        return Ok(if numerator == 0.0 { 1.0 } else { 0.0 });
    }

    Ok(numerator / denominator)
}

// ---------------------------------------------------------------------------
// adjusted_mutual_info
// ---------------------------------------------------------------------------

/// Compute the Adjusted Mutual Information (AMI) between two clusterings.
///
/// AMI corrects the Mutual Information (MI) for chance. A score of `1.0`
/// indicates perfect agreement; `0.0` is the expected value for random
/// labelings.
///
/// The formula used is:
///
/// ```text
/// MI = sum_{i,j} p_{ij} * log(p_{ij} / (p_i * p_j))
/// AMI = (MI - E[MI]) / (max(H_true, H_pred) - E[MI])
/// ```
///
/// where `E[MI]` is the expected MI under random permutations (computed using
/// the exact formula from Vinh et al., 2010).
///
/// # Arguments
///
/// * `labels_true` — ground-truth cluster labels.
/// * `labels_pred` — predicted cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::adjusted_mutual_info;
/// use ndarray::array;
///
/// let labels = array![0isize, 0, 1, 1, 2, 2];
/// let ami = adjusted_mutual_info(&labels, &labels).unwrap();
/// assert!((ami - 1.0).abs() < 1e-10);
/// ```
pub fn adjusted_mutual_info(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "adjusted_mutual_info")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "adjusted_mutual_info".into(),
        });
    }

    let n_f = n as f64;

    // Unique sorted labels.
    let mut classes_true: Vec<isize> = labels_true.iter().copied().collect();
    classes_true.sort_unstable();
    classes_true.dedup();

    let mut classes_pred: Vec<isize> = labels_pred.iter().copied().collect();
    classes_pred.sort_unstable();
    classes_pred.dedup();

    let r = classes_true.len();
    let s = classes_pred.len();

    // Contingency table.
    let mut contingency = vec![vec![0u64; s]; r];
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = classes_true.partition_point(|&c| c < lt);
        let ci = classes_pred.partition_point(|&c| c < lp);
        contingency[ri][ci] += 1;
    }

    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<u64> = (0..s)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Mutual Information.
    let mut mi = 0.0_f64;
    for i in 0..r {
        for j in 0..s {
            let n_ij = contingency[i][j] as f64;
            if n_ij == 0.0 {
                continue;
            }
            let ai = a[i] as f64;
            let bj = b[j] as f64;
            mi += n_ij / n_f * (n_ij * n_f / (ai * bj)).ln();
        }
    }

    // Entropies H(U) and H(V).
    let h_true = entropy_from_counts(&a, n_f);
    let h_pred = entropy_from_counts(&b, n_f);

    // Expected MI (exact formula from Vinh et al., 2010).
    let e_mi = expected_mutual_info(&a, &b, n as u64);

    let denominator = f64::max(h_true, h_pred) - e_mi;

    if denominator.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    Ok((mi - e_mi) / denominator)
}

/// Shannon entropy from raw counts.
fn entropy_from_counts(counts: &[u64], n: f64) -> f64 {
    counts.iter().fold(0.0, |acc, &c| {
        if c == 0 {
            acc
        } else {
            let p = c as f64 / n;
            acc - p * p.ln()
        }
    })
}

/// Expected Mutual Information under random permutations.
///
/// Uses the exact combinatorial formula:
/// E[MI] = sum_{i,j} sum_{n_ij} p(n_ij) * (n_ij/n) * log((n * n_ij) / (a_i * b_j))
///
/// where the sum over n_ij runs from max(1, a_i + b_j - n) to min(a_i, b_j).
fn expected_mutual_info(a: &[u64], b: &[u64], n: u64) -> f64 {
    let n_f = n as f64;
    let mut e_mi = 0.0_f64;

    // Precompute log factorials for n up to n.
    let log_fact = precompute_log_factorials(n as usize);

    for &ai in a {
        for &bj in b {
            let lo = ai.saturating_add(bj).saturating_sub(n).max(1);
            let hi = ai.min(bj);
            if lo > hi {
                continue;
            }
            for nij in lo..=hi {
                let nij_f = nij as f64;
                let ai_f = ai as f64;
                let bj_f = bj as f64;

                // log hypergeometric probability:
                // log P(n_ij) = log C(a_i, n_ij) + log C(b_j, n - a_i) - log C(n, b_j)
                //             + correction for n_ij being the actual value in numerator

                // log (n_ij / n) * log(n * n_ij / (a_i * b_j)) term
                // Full formula (as in sklearn):
                // term = nij/n * log(n*nij/(ai*bj))
                //        * C(ai, nij) * C(n-ai, bj-nij) / C(n, bj)
                // log hypergeometric probability using log-factorial table.
                // P(n_ij) = C(a_i, n_ij) * C(n - a_i, b_j - n_ij) / C(n, b_j)
                // The term (n - ai - bj + nij) must be non-negative by
                // construction (nij >= max(1, ai+bj-n)), so saturating sub is safe.
                let rem = n.saturating_sub(ai).saturating_sub(bj).saturating_add(nij);
                let log_num = log_fact[ai as usize]
                    + log_fact[(n - ai) as usize]
                    + log_fact[bj as usize]
                    + log_fact[(n - bj) as usize];
                let log_den = log_fact[nij as usize]
                    + log_fact[(ai - nij) as usize]
                    + log_fact[(bj - nij) as usize]
                    + log_fact[rem as usize]
                    + log_fact[n as usize];
                let log_p = log_num - log_den;
                let p = log_p.exp();

                let mi_term = nij_f / n_f * (n_f * nij_f / (ai_f * bj_f)).ln();
                e_mi += mi_term * p;
            }
        }
    }

    e_mi
}

/// Precompute log(k!) for k = 0..=n.
fn precompute_log_factorials(n: usize) -> Vec<f64> {
    let mut lf = vec![0.0_f64; n + 1];
    for k in 1..=n {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    lf
}

// ---------------------------------------------------------------------------
// davies_bouldin_score
// ---------------------------------------------------------------------------

/// Compute the Davies-Bouldin Index for a clustering.
///
/// The Davies-Bouldin Index measures clustering quality based on the ratio of
/// within-cluster scatter to between-cluster separation. Lower values indicate
/// better clustering.
///
/// For each cluster `i`, let `s_i` be the mean distance from cluster members
/// to their centroid. For each pair of clusters `(i, j)`, define:
///
/// ```text
/// R_ij = (s_i + s_j) / d(c_i, c_j)
/// ```
///
/// where `d(c_i, c_j)` is the Euclidean distance between centroids. Then:
///
/// ```text
/// DB = (1/k) * sum_i max_{j != i} R_ij
/// ```
///
/// Noise points (label == -1) are excluded.
///
/// # Arguments
///
/// * `x`      — feature matrix of shape `(n_samples, n_features)`.
/// * `labels` — cluster label for each sample. Use `-1` for noise.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != labels.len()`.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples or all
/// samples are noise.
/// Returns [`FerroError::InvalidParameter`] if fewer than 2 clusters are found.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::davies_bouldin_score;
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     0.1, 0.1,
///     10.0, 10.0,
///     10.1, 10.1,
/// ]).unwrap();
/// let labels = array![0isize, 0, 1, 1];
/// let score = davies_bouldin_score(&x, &labels).unwrap();
/// assert!(score < 0.05);
/// ```
pub fn davies_bouldin_score<F>(x: &Array2<F>, labels: &Array1<isize>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = x.nrows();
    check_x_labels_compat(n, labels.len(), "davies_bouldin_score: x rows vs labels")?;

    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "davies_bouldin_score".into(),
        });
    }

    let cluster_labels = unique_cluster_labels(labels);
    let n_clusters = cluster_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: format!(
                "davies_bouldin_score requires at least 2 clusters (excluding noise), found {n_clusters}"
            ),
        });
    }

    let n_features = x.ncols();

    // Compute centroids for each cluster.
    let cluster_indices: Vec<Vec<usize>> = cluster_labels
        .iter()
        .map(|&cl| {
            labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == cl { Some(i) } else { None })
                .collect()
        })
        .collect();

    // Centroids: average position.
    let centroids: Vec<Vec<F>> = cluster_indices
        .iter()
        .map(|members| {
            let mut centroid = vec![F::zero(); n_features];
            for &i in members {
                for f in 0..n_features {
                    centroid[f] = centroid[f] + x[[i, f]];
                }
            }
            let cnt = F::from(members.len()).unwrap();
            centroid.iter_mut().for_each(|v| *v = *v / cnt);
            centroid
        })
        .collect();

    // s_i: mean distance from each member to its centroid.
    let s: Vec<F> = cluster_indices
        .iter()
        .enumerate()
        .map(|(k, members)| {
            let c = &centroids[k];
            let total: F = members.iter().fold(F::zero(), |acc, &i| {
                acc + euclidean_dist(&c[..], &x.row(i).to_vec()[..])
            });
            if members.is_empty() {
                F::zero()
            } else {
                total / F::from(members.len()).unwrap()
            }
        })
        .collect();

    // Pairwise centroid distances.
    let mut db_sum = F::zero();
    for i in 0..n_clusters {
        let mut max_r = F::zero();
        for j in 0..n_clusters {
            if i == j {
                continue;
            }
            let d_ij = euclidean_dist(&centroids[i][..], &centroids[j][..]);
            if d_ij == F::zero() {
                // Coincident centroids — R_ij is undefined; treat as infinity.
                max_r = F::infinity();
                break;
            }
            let r_ij = (s[i] + s[j]) / d_ij;
            if r_ij > max_r {
                max_r = r_ij;
            }
        }
        db_sum = db_sum + max_r;
    }

    Ok(db_sum / F::from(n_clusters).unwrap())
}

// ---------------------------------------------------------------------------
// silhouette_samples
// ---------------------------------------------------------------------------

/// Compute per-sample Silhouette Coefficients.
///
/// For each sample `i` belonging to cluster `C_i`:
/// - `a(i)` = mean distance from `i` to all other samples in `C_i`
/// - `b(i)` = mean distance from `i` to samples in the nearest other cluster
/// - `s(i)` = `(b(i) - a(i)) / max(a(i), b(i))`
///
/// Unlike [`silhouette_score`], this returns the per-sample array rather than
/// the mean. Noise points (label == -1) receive a silhouette value of `0.0`.
///
/// # Arguments
///
/// * `x`      — feature matrix of shape `(n_samples, n_features)`.
/// * `labels` — cluster label for each sample. Use `-1` for noise.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != labels.len()`.
/// Returns [`FerroError::InsufficientSamples`] if there are no samples.
/// Returns [`FerroError::InvalidParameter`] if there is only one cluster
/// (after excluding noise).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::silhouette_samples;
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     0.1, 0.1,
///     10.0, 10.0,
///     10.1, 10.1,
/// ]).unwrap();
/// let labels = array![0isize, 0, 1, 1];
/// let samples = silhouette_samples(&x, &labels).unwrap();
/// assert_eq!(samples.len(), 4);
/// assert!(samples[0] > 0.9);
/// ```
pub fn silhouette_samples<F>(x: &Array2<F>, labels: &Array1<isize>) -> Result<Array1<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = x.nrows();
    check_x_labels_compat(n, labels.len(), "silhouette_samples: x rows vs labels")?;

    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "silhouette_samples".into(),
        });
    }

    let cluster_labels = unique_cluster_labels(labels);
    let n_clusters = cluster_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: format!(
                "silhouette_samples requires at least 2 clusters (excluding noise), found {n_clusters}"
            ),
        });
    }

    let cluster_indices: Vec<Vec<usize>> = cluster_labels
        .iter()
        .map(|&cl| {
            labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == cl { Some(i) } else { None })
                .collect()
        })
        .collect();

    let label_to_idx = |lbl: isize| -> Option<usize> {
        cluster_labels
            .partition_point(|&c| c < lbl)
            .let_if(|&pos| pos < cluster_labels.len() && cluster_labels[pos] == lbl)
    };

    let mut result = Array1::from_elem(n, F::zero());

    for i in 0..n {
        let li = labels[i];
        if li == -1 {
            continue; // noise points get 0.0
        }

        let ci_idx = match label_to_idx(li) {
            Some(idx) => idx,
            None => continue,
        };

        let ci_members = &cluster_indices[ci_idx];

        let a_i = if ci_members.len() <= 1 {
            F::zero()
        } else {
            let mut dist_sum = F::zero();
            for &j in ci_members {
                if j == i {
                    continue;
                }
                dist_sum = dist_sum + row_euclidean_dist(x, i, j);
            }
            dist_sum / F::from(ci_members.len() - 1).unwrap()
        };

        let mut b_i = F::infinity();
        for (k, &cl_k) in cluster_labels.iter().enumerate() {
            if cl_k == li {
                continue;
            }
            let other_members = &cluster_indices[k];
            if other_members.is_empty() {
                continue;
            }
            let mut dist_sum = F::zero();
            for &j in other_members {
                dist_sum = dist_sum + row_euclidean_dist(x, i, j);
            }
            let mean_dist = dist_sum / F::from(other_members.len()).unwrap();
            if mean_dist < b_i {
                b_i = mean_dist;
            }
        }

        let max_ab = if a_i > b_i { a_i } else { b_i };
        let s_i = if max_ab == F::zero() {
            F::zero()
        } else {
            (b_i - a_i) / max_ab
        };

        result[i] = s_i;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// calinski_harabasz_score
// ---------------------------------------------------------------------------

/// Compute the Calinski-Harabasz Index (Variance Ratio Criterion).
///
/// The score is the ratio of the between-cluster dispersion to the
/// within-cluster dispersion:
///
/// ```text
/// CH = (B / (k - 1)) / (W / (n - k))
/// ```
///
/// where `B` is the between-group sum of squares, `W` is the within-group
/// sum of squares, `k` is the number of clusters, and `n` is the number of
/// samples.
///
/// Higher values indicate better-defined clustering. Noise points (label == -1)
/// are excluded.
///
/// # Arguments
///
/// * `x`      — feature matrix of shape `(n_samples, n_features)`.
/// * `labels` — cluster label for each sample. Use `-1` for noise.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != labels.len()`.
/// Returns [`FerroError::InsufficientSamples`] if there are fewer than 2
/// non-noise samples.
/// Returns [`FerroError::InvalidParameter`] if fewer than 2 clusters are found,
/// or if `n == k` (which would make the within-group degrees of freedom zero).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::calinski_harabasz_score;
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     0.1, 0.1,
///     10.0, 10.0,
///     10.1, 10.1,
/// ]).unwrap();
/// let labels = array![0isize, 0, 1, 1];
/// let score = calinski_harabasz_score(&x, &labels).unwrap();
/// assert!(score > 100.0); // well-separated clusters
/// ```
pub fn calinski_harabasz_score<F>(x: &Array2<F>, labels: &Array1<isize>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n_total = x.nrows();
    check_x_labels_compat(
        n_total,
        labels.len(),
        "calinski_harabasz_score: x rows vs labels",
    )?;

    let cluster_labels = unique_cluster_labels(labels);
    let n_clusters = cluster_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: format!(
                "calinski_harabasz_score requires at least 2 clusters (excluding noise), found {n_clusters}"
            ),
        });
    }

    let n_features = x.ncols();

    // Collect indices per cluster.
    let cluster_indices: Vec<Vec<usize>> = cluster_labels
        .iter()
        .map(|&cl| {
            labels
                .iter()
                .enumerate()
                .filter_map(|(i, &l)| if l == cl { Some(i) } else { None })
                .collect()
        })
        .collect();

    // Total non-noise samples.
    let n: usize = cluster_indices.iter().map(std::vec::Vec::len).sum();

    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "calinski_harabasz_score".into(),
        });
    }

    if n == n_clusters {
        return Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: "calinski_harabasz_score: n_samples equals n_clusters, within-group dispersion is zero".into(),
        });
    }

    // Global centroid (non-noise samples only).
    let mut global_centroid = vec![F::zero(); n_features];
    for members in &cluster_indices {
        for &i in members {
            for f in 0..n_features {
                global_centroid[f] = global_centroid[f] + x[[i, f]];
            }
        }
    }
    let n_f = F::from(n).unwrap();
    for v in &mut global_centroid {
        *v = *v / n_f;
    }

    // Per-cluster centroids.
    let centroids: Vec<Vec<F>> = cluster_indices
        .iter()
        .map(|members| {
            let mut centroid = vec![F::zero(); n_features];
            for &i in members {
                for f in 0..n_features {
                    centroid[f] = centroid[f] + x[[i, f]];
                }
            }
            let cnt = F::from(members.len()).unwrap();
            centroid.iter_mut().for_each(|v| *v = *v / cnt);
            centroid
        })
        .collect();

    // Between-group sum of squares.
    let mut b_ss = F::zero();
    for (k, members) in cluster_indices.iter().enumerate() {
        let n_k = F::from(members.len()).unwrap();
        let mut sq_dist = F::zero();
        for f in 0..n_features {
            let d = centroids[k][f] - global_centroid[f];
            sq_dist = sq_dist + d * d;
        }
        b_ss = b_ss + n_k * sq_dist;
    }

    // Within-group sum of squares.
    let mut w_ss = F::zero();
    for (k, members) in cluster_indices.iter().enumerate() {
        for &i in members {
            let mut sq_dist = F::zero();
            for f in 0..n_features {
                let d = x[[i, f]] - centroids[k][f];
                sq_dist = sq_dist + d * d;
            }
            w_ss = w_ss + sq_dist;
        }
    }

    let k_f = F::from(n_clusters).unwrap();
    let one = F::one();

    // CH = (B / (k-1)) / (W / (n-k))
    if w_ss == F::zero() {
        // Perfect clustering — all points coincide with centroids.
        return Ok(F::infinity());
    }

    Ok((b_ss / (k_f - one)) / (w_ss / (n_f - k_f)))
}

// ---------------------------------------------------------------------------
// homogeneity_score, completeness_score, v_measure_score
// ---------------------------------------------------------------------------

/// Build a contingency table between true labels and predicted labels (both isize).
///
/// Returns `(classes_true, classes_pred, contingency)`.
fn build_contingency_table(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> (Vec<isize>, Vec<isize>, Vec<Vec<u64>>) {
    let mut classes_true: Vec<isize> = labels_true.iter().copied().collect();
    classes_true.sort_unstable();
    classes_true.dedup();

    let mut classes_pred: Vec<isize> = labels_pred.iter().copied().collect();
    classes_pred.sort_unstable();
    classes_pred.dedup();

    let r = classes_true.len();
    let s = classes_pred.len();

    let mut contingency = vec![vec![0u64; s]; r];
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = classes_true.partition_point(|&c| c < lt);
        let ci = classes_pred.partition_point(|&c| c < lp);
        contingency[ri][ci] += 1;
    }

    (classes_true, classes_pred, contingency)
}

/// Compute the homogeneity score.
///
/// A clustering result satisfies homogeneity if all clusters contain only
/// samples from a single class. The score is:
///
/// ```text
/// homogeneity = 1 - H(C|K) / H(C)
/// ```
///
/// where `H(C|K)` is the conditional entropy of the class distribution
/// given the cluster assignments and `H(C)` is the entropy of the class
/// distribution.
///
/// # Arguments
///
/// * `labels_true` — ground-truth class labels.
/// * `labels_pred` — cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::homogeneity_score;
/// use ndarray::array;
///
/// let labels_true = array![0isize, 0, 1, 1, 2, 2];
/// let labels_pred = array![0isize, 0, 1, 1, 2, 2];
/// let h = homogeneity_score(&labels_true, &labels_pred).unwrap();
/// assert!((h - 1.0).abs() < 1e-10);
/// ```
pub fn homogeneity_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "homogeneity_score")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "homogeneity_score".into(),
        });
    }

    let n_f = n as f64;
    let (_classes_true, _classes_pred, contingency) =
        build_contingency_table(labels_true, labels_pred);

    let n_cols = contingency[0].len();

    // Row sums (class totals).
    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();

    // H(C) = entropy of classes.
    let h_c = entropy_from_counts(&a, n_f);
    if h_c == 0.0 {
        // Only one class — homogeneity is 1.0 by convention.
        return Ok(1.0);
    }

    // Column sums (cluster totals).
    let b: Vec<u64> = (0..n_cols)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // H(C|K) = conditional entropy of classes given clusters.
    let mut h_c_given_k = 0.0_f64;
    for j in 0..n_cols {
        let bj = b[j] as f64;
        if bj == 0.0 {
            continue;
        }
        for row in &contingency {
            let n_ij = row[j] as f64;
            if n_ij == 0.0 {
                continue;
            }
            h_c_given_k -= (n_ij / n_f) * (n_ij / bj).ln();
        }
    }

    Ok(1.0 - h_c_given_k / h_c)
}

/// Compute the completeness score.
///
/// A clustering result satisfies completeness if all samples of a given
/// class are assigned to the same cluster. The score is:
///
/// ```text
/// completeness = 1 - H(K|C) / H(K)
/// ```
///
/// where `H(K|C)` is the conditional entropy of the cluster distribution
/// given the class labels and `H(K)` is the entropy of the cluster
/// distribution.
///
/// # Arguments
///
/// * `labels_true` — ground-truth class labels.
/// * `labels_pred` — cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::completeness_score;
/// use ndarray::array;
///
/// let labels_true = array![0isize, 0, 1, 1, 2, 2];
/// let labels_pred = array![0isize, 0, 1, 1, 2, 2];
/// let c = completeness_score(&labels_true, &labels_pred).unwrap();
/// assert!((c - 1.0).abs() < 1e-10);
/// ```
pub fn completeness_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "completeness_score")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "completeness_score".into(),
        });
    }

    let n_f = n as f64;
    let (_classes_true, _classes_pred, contingency) =
        build_contingency_table(labels_true, labels_pred);

    let n_cols = contingency[0].len();

    // Column sums (cluster totals).
    let b: Vec<u64> = (0..n_cols)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // H(K) = entropy of clusters.
    let h_k = entropy_from_counts(&b, n_f);
    if h_k == 0.0 {
        // Only one cluster — completeness is 1.0 by convention.
        return Ok(1.0);
    }

    // Row sums (class totals).
    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();

    // H(K|C) = conditional entropy of clusters given classes.
    let mut h_k_given_c = 0.0_f64;
    for (row, &ai_raw) in contingency.iter().zip(a.iter()) {
        let ai = ai_raw as f64;
        if ai == 0.0 {
            continue;
        }
        for &cell in row {
            let n_ij = cell as f64;
            if n_ij == 0.0 {
                continue;
            }
            h_k_given_c -= (n_ij / n_f) * (n_ij / ai).ln();
        }
    }

    Ok(1.0 - h_k_given_c / h_k)
}

/// Compute the V-measure score.
///
/// The V-measure is the harmonic mean of homogeneity and completeness:
///
/// ```text
/// v = 2 * (homogeneity * completeness) / (homogeneity + completeness)
/// ```
///
/// # Arguments
///
/// * `labels_true` — ground-truth class labels.
/// * `labels_pred` — cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::v_measure_score;
/// use ndarray::array;
///
/// let labels_true = array![0isize, 0, 1, 1, 2, 2];
/// let labels_pred = array![0isize, 0, 1, 1, 2, 2];
/// let v = v_measure_score(&labels_true, &labels_pred).unwrap();
/// assert!((v - 1.0).abs() < 1e-10);
/// ```
pub fn v_measure_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let h = homogeneity_score(labels_true, labels_pred)?;
    let c = completeness_score(labels_true, labels_pred)?;

    if h + c == 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * h * c / (h + c))
}

// ---------------------------------------------------------------------------
// rand_score
// ---------------------------------------------------------------------------

/// Compute the unadjusted Rand Index between two clusterings.
///
/// The Rand Index measures the proportion of sample pairs that are either in the
/// same cluster in both labelings or in different clusters in both labelings:
///
/// ```text
/// RI = (a + d) / C(n, 2)
/// ```
///
/// where `a` is the number of pairs in the same cluster in both, and `d` is the
/// number of pairs in different clusters in both.
///
/// Unlike [`adjusted_rand_score`], this is not corrected for chance: random
/// labelings may receive a high score.
///
/// # Arguments
///
/// * `labels_true` — ground-truth cluster labels.
/// * `labels_pred` — predicted cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays have fewer than 2 elements.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::rand_score;
/// use ndarray::array;
///
/// let labels = array![0isize, 0, 1, 1, 2, 2];
/// let ri = rand_score(&labels, &labels).unwrap();
/// assert!((ri - 1.0).abs() < 1e-10);
/// ```
pub fn rand_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "rand_score")?;
    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "rand_score requires at least 2 samples".into(),
        });
    }

    // Build the contingency table.
    let mut classes_true: Vec<isize> = labels_true.iter().copied().collect();
    classes_true.sort_unstable();
    classes_true.dedup();

    let mut classes_pred: Vec<isize> = labels_pred.iter().copied().collect();
    classes_pred.sort_unstable();
    classes_pred.dedup();

    let r = classes_true.len();
    let s = classes_pred.len();

    let mut contingency = vec![vec![0u64; s]; r];
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = classes_true.partition_point(|&c| c < lt);
        let ci = classes_pred.partition_point(|&c| c < lp);
        contingency[ri][ci] += 1;
    }

    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<u64> = (0..s)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    let comb_n = n_choose_2(n as u64);

    // a = number of pairs in same cluster in both = sum of C(n_ij, 2)
    let sum_comb_c: u64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| n_choose_2(v))
        .sum();

    let sum_comb_a: u64 = a.iter().map(|&ai| n_choose_2(ai)).sum();
    let sum_comb_b: u64 = b.iter().map(|&bj| n_choose_2(bj)).sum();

    // d = C(n,2) - sum_comb_a - sum_comb_b + sum_comb_c
    // (pairs that differ in both = total pairs - pairs_same_in_true - pairs_same_in_pred + pairs_same_in_both)
    let d = comb_n - sum_comb_a - sum_comb_b + sum_comb_c;

    Ok((sum_comb_c + d) as f64 / comb_n as f64)
}

// ---------------------------------------------------------------------------
// normalized_mutual_info_score
// ---------------------------------------------------------------------------

/// Normalization method for [`normalized_mutual_info_score`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmiMethod {
    /// Normalize by the geometric mean of entropies: `sqrt(H(U) * H(V))`.
    Geometric,
    /// Normalize by the arithmetic mean of entropies: `(H(U) + H(V)) / 2`.
    Arithmetic,
    /// Normalize by the minimum entropy: `min(H(U), H(V))`.
    Min,
    /// Normalize by the maximum entropy: `max(H(U), H(V))`.
    Max,
}

/// Compute the Normalized Mutual Information (NMI) between two clusterings.
///
/// NMI scales the Mutual Information to `[0, 1]` by dividing by a function of
/// the entropies:
///
/// ```text
/// NMI = MI(U, V) / norm(H(U), H(V))
/// ```
///
/// The normalization function is chosen via the [`NmiMethod`] parameter.
///
/// # Arguments
///
/// * `labels_true` — ground-truth cluster labels.
/// * `labels_pred` — predicted cluster labels.
/// * `method`      — normalization strategy.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::{normalized_mutual_info_score, NmiMethod};
/// use ndarray::array;
///
/// let labels = array![0isize, 0, 1, 1, 2, 2];
/// let nmi = normalized_mutual_info_score(&labels, &labels, NmiMethod::Geometric).unwrap();
/// assert!((nmi - 1.0).abs() < 1e-10);
/// ```
pub fn normalized_mutual_info_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
    method: NmiMethod,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "normalized_mutual_info_score")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "normalized_mutual_info_score".into(),
        });
    }

    let n_f = n as f64;

    let (_, _, contingency) = build_contingency_table(labels_true, labels_pred);

    let r = contingency.len();
    let s = contingency[0].len();

    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<u64> = (0..s)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Mutual Information.
    let mut mi = 0.0_f64;
    for i in 0..r {
        for j in 0..s {
            let n_ij = contingency[i][j] as f64;
            if n_ij == 0.0 {
                continue;
            }
            let ai = a[i] as f64;
            let bj = b[j] as f64;
            mi += n_ij / n_f * (n_ij * n_f / (ai * bj)).ln();
        }
    }

    let h_true = entropy_from_counts(&a, n_f);
    let h_pred = entropy_from_counts(&b, n_f);

    let normalizer = match method {
        NmiMethod::Geometric => (h_true * h_pred).sqrt(),
        NmiMethod::Arithmetic => f64::midpoint(h_true, h_pred),
        NmiMethod::Min => h_true.min(h_pred),
        NmiMethod::Max => h_true.max(h_pred),
    };

    if normalizer.abs() < f64::EPSILON {
        // Both entropies are zero => single cluster in each => perfect by convention.
        return Ok(1.0);
    }

    Ok(mi / normalizer)
}

// ---------------------------------------------------------------------------
// fowlkes_mallows_score
// ---------------------------------------------------------------------------

/// Compute the Fowlkes-Mallows Index (FMI) between two clusterings.
///
/// FMI is the geometric mean of pairwise precision and recall:
///
/// ```text
/// FMI = TP / sqrt((TP + FP) * (TP + FN))
/// ```
///
/// where `TP` is the number of pairs in the same cluster in both, `FP` is pairs
/// in the same cluster only in the predicted labels, and `FN` is pairs in the
/// same cluster only in the true labels.
///
/// # Arguments
///
/// * `labels_true` — ground-truth cluster labels.
/// * `labels_pred` — predicted cluster labels.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::clustering::fowlkes_mallows_score;
/// use ndarray::array;
///
/// let labels = array![0isize, 0, 1, 1, 2, 2];
/// let fmi = fowlkes_mallows_score(&labels, &labels).unwrap();
/// assert!((fmi - 1.0).abs() < 1e-10);
/// ```
pub fn fowlkes_mallows_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "fowlkes_mallows_score")?;
    if n < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n,
            context: "fowlkes_mallows_score requires at least 2 samples".into(),
        });
    }

    // Build contingency table.
    let mut classes_true: Vec<isize> = labels_true.iter().copied().collect();
    classes_true.sort_unstable();
    classes_true.dedup();

    let mut classes_pred: Vec<isize> = labels_pred.iter().copied().collect();
    classes_pred.sort_unstable();
    classes_pred.dedup();

    let r = classes_true.len();
    let s = classes_pred.len();

    let mut contingency = vec![vec![0u64; s]; r];
    for (&lt, &lp) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = classes_true.partition_point(|&c| c < lt);
        let ci = classes_pred.partition_point(|&c| c < lp);
        contingency[ri][ci] += 1;
    }

    let a: Vec<u64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<u64> = (0..s)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // TP = sum C(n_ij, 2)
    let tp: u64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| n_choose_2(v))
        .sum();

    // TP + FP = sum C(b_j, 2)
    let tp_plus_fp: u64 = b.iter().map(|&bj| n_choose_2(bj)).sum();

    // TP + FN = sum C(a_i, 2)
    let tp_plus_fn: u64 = a.iter().map(|&ai| n_choose_2(ai)).sum();

    if tp_plus_fp == 0 || tp_plus_fn == 0 {
        return Ok(0.0);
    }

    Ok(tp as f64 / ((tp_plus_fp as f64) * (tp_plus_fn as f64)).sqrt())
}

// ---------------------------------------------------------------------------
// Mutual information / contingency / homogeneity_completeness_v_measure /
// pair_confusion_matrix
// ---------------------------------------------------------------------------

/// Build the contingency matrix between two clusterings.
///
/// Returns a 2-D array of shape `(n_classes_true, n_classes_pred)` where
/// entry `(i, j)` counts the samples assigned to true class `i` and
/// predicted class `j`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
pub fn contingency_matrix(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<Array2<u64>, FerroError> {
    check_labels_same_length(
        labels_true.len(),
        labels_pred.len(),
        "contingency_matrix: labels_true vs labels_pred",
    )?;
    if labels_true.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "contingency_matrix".into(),
        });
    }
    let (_ct, _cp, table) = build_contingency_table(labels_true, labels_pred);
    let r = table.len();
    let s = if r == 0 { 0 } else { table[0].len() };
    let mut out = Array2::<u64>::zeros((r, s));
    for i in 0..r {
        for j in 0..s {
            out[[i, j]] = table[i][j];
        }
    }
    Ok(out)
}

/// Compute the (raw, unnormalised) mutual information score between two
/// clusterings.
///
/// `MI(U, V) = sum_{i,j} (n_ij / n) * log(n * n_ij / (a_i * b_j))`
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
pub fn mutual_info_score(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<f64, FerroError> {
    check_labels_same_length(
        labels_true.len(),
        labels_pred.len(),
        "mutual_info_score: labels_true vs labels_pred",
    )?;
    if labels_true.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "mutual_info_score".into(),
        });
    }
    let (_ct, _cp, contingency) = build_contingency_table(labels_true, labels_pred);
    let r = contingency.len();
    let s = if r == 0 { 0 } else { contingency[0].len() };

    let mut row_sum = vec![0u64; r];
    let mut col_sum = vec![0u64; s];
    let mut total: u64 = 0;
    for i in 0..r {
        for j in 0..s {
            row_sum[i] += contingency[i][j];
            col_sum[j] += contingency[i][j];
            total += contingency[i][j];
        }
    }
    if total == 0 {
        return Ok(0.0);
    }
    let n = total as f64;
    let mut mi = 0.0_f64;
    for i in 0..r {
        for j in 0..s {
            let nij = contingency[i][j] as f64;
            if nij > 0.0 {
                let ai = row_sum[i] as f64;
                let bj = col_sum[j] as f64;
                mi += (nij / n) * ((n * nij) / (ai * bj)).ln();
            }
        }
    }
    Ok(mi)
}

/// Compute homogeneity, completeness, and V-measure together.
///
/// Returns `(homogeneity, completeness, v_measure)`. `beta` controls the
/// weight of homogeneity in the V-measure: `beta < 1` favours homogeneity,
/// `beta > 1` favours completeness.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `beta <= 0`.
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
pub fn homogeneity_completeness_v_measure(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
    beta: f64,
) -> Result<(f64, f64, f64), FerroError> {
    if !(beta > 0.0 && beta.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "beta".into(),
            reason: format!("homogeneity_completeness_v_measure: beta must be > 0, got {beta}"),
        });
    }
    let h = homogeneity_score(labels_true, labels_pred)?;
    let c = completeness_score(labels_true, labels_pred)?;
    let v = if h + c == 0.0 {
        0.0
    } else {
        (1.0 + beta) * h * c / (beta * h + c)
    };
    Ok((h, c, v))
}

/// Compute the pair-confusion matrix between two clusterings.
///
/// Returns a 2x2 matrix `[[C00, C01], [C10, C11]]` over all `(n choose 2)`
/// sample pairs:
/// - `C00` — pairs in different clusters in both clusterings
/// - `C11` — pairs in the same cluster in both clusterings
/// - `C01` — pairs in different clusters in true, same in pred
/// - `C10` — pairs in same cluster in true, different in pred
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if the arrays have different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
pub fn pair_confusion_matrix(
    labels_true: &Array1<isize>,
    labels_pred: &Array1<isize>,
) -> Result<Array2<u64>, FerroError> {
    let n = labels_true.len();
    check_labels_same_length(n, labels_pred.len(), "pair_confusion_matrix")?;
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "pair_confusion_matrix".into(),
        });
    }
    let (_ct, _cp, contingency) = build_contingency_table(labels_true, labels_pred);
    let r = contingency.len();
    let s = if r == 0 { 0 } else { contingency[0].len() };

    let mut row_sum = vec![0u64; r];
    let mut col_sum = vec![0u64; s];
    let mut sum_nij_sq: u64 = 0;
    for i in 0..r {
        for j in 0..s {
            let nij = contingency[i][j];
            row_sum[i] += nij;
            col_sum[j] += nij;
            sum_nij_sq += nij * nij;
        }
    }
    let sum_a_sq: u64 = row_sum.iter().map(|x| x * x).sum();
    let sum_b_sq: u64 = col_sum.iter().map(|x| x * x).sum();
    let total = n as u64;
    let total_sq = total * total;

    let c11 = sum_nij_sq.saturating_sub(total);
    let c10 = sum_a_sq.saturating_sub(sum_nij_sq);
    let c01 = sum_b_sq.saturating_sub(sum_nij_sq);
    let c00 = total_sq + sum_nij_sq - sum_a_sq - sum_b_sq;
    let mut out = Array2::<u64>::zeros((2, 2));
    out[[0, 0]] = c00;
    out[[0, 1]] = c01;
    out[[1, 0]] = c10;
    out[[1, 1]] = c11;
    Ok(out)
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
    // silhouette_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_silhouette_perfect_clustering() {
        // Two clusters far apart — score should be very close to 1.0.
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 0.1, 0.0, 100.0, 0.0, 100.1, 0.0])
                .unwrap();
        let labels = array![0isize, 0, 1, 1];
        let score = silhouette_score(&x, &labels).unwrap();
        assert!(score > 0.99, "expected near 1.0, got {score}");
    }

    #[test]
    fn test_silhouette_identical_labels_returns_score() {
        // Identical labels: well-separated clusters score close to 1.
        let x =
            Array2::from_shape_vec((6, 1), vec![0.0_f64, 0.5, 1.0, 100.0, 100.5, 101.0]).unwrap();
        let labels = array![0isize, 0, 0, 1, 1, 1];
        let score = silhouette_score(&x, &labels).unwrap();
        assert!(score > 0.9, "expected > 0.9, got {score}");
    }

    #[test]
    fn test_silhouette_noise_points_ignored() {
        // Noise points (label -1) must be skipped.
        let x = Array2::from_shape_vec((5, 1), vec![0.0_f64, 0.1, 50.0, 100.0, 100.1]).unwrap();
        // point at index 2 is noise
        let labels = array![0isize, 0, -1, 1, 1];
        let score = silhouette_score(&x, &labels).unwrap();
        assert!(score > 0.9, "expected > 0.9, got {score}");
    }

    #[test]
    fn test_silhouette_all_noise_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![-1isize, -1, -1];
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_single_cluster_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0, 0];
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_shape_mismatch_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0];
        assert!(silhouette_score(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_overlapping_clusters_lower_score() {
        // Clusters that overlap should yield a lower silhouette than well-separated ones.
        let x_sep = Array2::from_shape_vec((4, 1), vec![0.0_f64, 0.1, 100.0, 100.1]).unwrap();
        let x_ov = Array2::from_shape_vec((4, 1), vec![0.0_f64, 1.0, 0.5, 1.5]).unwrap();
        let labels = array![0isize, 0, 1, 1];
        let score_sep = silhouette_score(&x_sep, &labels).unwrap();
        let score_ov = silhouette_score(&x_ov, &labels).unwrap();
        assert!(
            score_sep > score_ov,
            "separated ({score_sep}) should beat overlapping ({score_ov})"
        );
    }

    // -----------------------------------------------------------------------
    // adjusted_rand_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_ari_identical_labels_is_one() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        assert_abs_diff_eq!(
            adjusted_rand_score(&labels, &labels).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_ari_permuted_labels_is_one() {
        // Relabeling clusters should not change ARI.
        let lt = array![0isize, 0, 1, 1];
        let lp = array![1isize, 1, 0, 0]; // same partition, different names
        assert_abs_diff_eq!(adjusted_rand_score(&lt, &lp).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ari_all_in_one_cluster() {
        // All samples in a single predicted cluster, true has two clusters.
        let lt = array![0isize, 0, 1, 1];
        let lp = array![0isize, 0, 0, 0];
        let ari = adjusted_rand_score(&lt, &lp).unwrap();
        // ARI should be <= 0 for this degenerate case.
        assert!(ari <= 0.0, "expected <= 0, got {ari}");
    }

    #[test]
    fn test_ari_shape_mismatch_returns_error() {
        let lt = array![0isize, 0, 1];
        let lp = array![0isize, 0];
        assert!(adjusted_rand_score(&lt, &lp).is_err());
    }

    #[test]
    fn test_ari_empty_returns_error() {
        let lt = Array1::<isize>::from_vec(vec![]);
        let lp = Array1::<isize>::from_vec(vec![]);
        assert!(adjusted_rand_score(&lt, &lp).is_err());
    }

    #[test]
    fn test_ari_opposite_labeling_near_zero() {
        // Each sample in its own cluster vs all in one: near 0 or negative.
        let lt = array![0isize, 1, 2, 3];
        let lp = array![0isize, 0, 0, 0];
        let ari = adjusted_rand_score(&lt, &lp).unwrap();
        assert!(ari <= 0.1, "expected near 0 or negative, got {ari}");
    }

    // -----------------------------------------------------------------------
    // adjusted_mutual_info
    // -----------------------------------------------------------------------

    #[test]
    fn test_ami_identical_labels_is_one() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        assert_abs_diff_eq!(
            adjusted_mutual_info(&labels, &labels).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_ami_permuted_labels_is_one() {
        let lt = array![0isize, 0, 1, 1];
        let lp = array![1isize, 1, 0, 0];
        assert_abs_diff_eq!(
            adjusted_mutual_info(&lt, &lp).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_ami_shape_mismatch_returns_error() {
        let lt = array![0isize, 0, 1];
        let lp = array![0isize, 0];
        assert!(adjusted_mutual_info(&lt, &lp).is_err());
    }

    #[test]
    fn test_ami_empty_returns_error() {
        let lt = Array1::<isize>::from_vec(vec![]);
        let lp = Array1::<isize>::from_vec(vec![]);
        assert!(adjusted_mutual_info(&lt, &lp).is_err());
    }

    #[test]
    fn test_ami_all_same_predicted_label_near_zero() {
        // Predicting all samples as one cluster has near-zero AMI.
        let lt = array![0isize, 0, 1, 1, 2, 2];
        let lp = array![0isize, 0, 0, 0, 0, 0];
        let ami = adjusted_mutual_info(&lt, &lp).unwrap();
        assert!(ami <= 0.1, "expected near 0, got {ami}");
    }

    // -----------------------------------------------------------------------
    // davies_bouldin_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_db_well_separated_is_low() {
        // Clusters far apart and compact → very low DB index.
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 0.1, 0.0, 100.0, 0.0, 100.1, 0.0])
                .unwrap();
        let labels = array![0isize, 0, 1, 1];
        let score = davies_bouldin_score(&x, &labels).unwrap();
        assert!(score < 0.01, "expected very low DB, got {score}");
    }

    #[test]
    fn test_db_shape_mismatch_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0];
        assert!(davies_bouldin_score(&x, &labels).is_err());
    }

    #[test]
    fn test_db_single_cluster_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0, 0];
        assert!(davies_bouldin_score(&x, &labels).is_err());
    }

    #[test]
    fn test_db_all_noise_returns_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![-1isize, -1, -1];
        assert!(davies_bouldin_score(&x, &labels).is_err());
    }

    #[test]
    fn test_db_noise_points_ignored() {
        // Noise label -1 should be ignored; result same as without noise point.
        let x_no_noise = Array2::from_shape_vec((4, 1), vec![0.0_f64, 0.1, 100.0, 100.1]).unwrap();
        let x_with_noise =
            Array2::from_shape_vec((5, 1), vec![0.0_f64, 0.1, 50.0, 100.0, 100.1]).unwrap();
        let labels_no_noise = array![0isize, 0, 1, 1];
        let labels_with_noise = array![0isize, 0, -1, 1, 1];

        let db_no = davies_bouldin_score(&x_no_noise, &labels_no_noise).unwrap();
        let db_with = davies_bouldin_score(&x_with_noise, &labels_with_noise).unwrap();
        assert_abs_diff_eq!(db_no, db_with, epsilon = 1e-10);
    }

    #[test]
    fn test_db_worse_clustering_has_higher_score() {
        // A poor clustering (large scatter relative to separation) yields higher DB.
        let x = Array2::from_shape_vec((6, 1), vec![0.0_f64, 5.0, 10.0, 15.0, 20.0, 25.0]).unwrap();
        // Good: [0,5] vs [10..25]  — tight cluster 0, wider cluster 1
        let labels_good = array![0isize, 0, 1, 1, 1, 1];
        // Bad: alternating assignments
        let labels_bad = array![0isize, 1, 0, 1, 0, 1];

        let db_good = davies_bouldin_score(&x, &labels_good).unwrap();
        let db_bad = davies_bouldin_score(&x, &labels_bad).unwrap();
        assert!(
            db_good < db_bad,
            "good clustering ({db_good}) should have lower DB than bad ({db_bad})"
        );
    }

    // -----------------------------------------------------------------------
    // silhouette_samples
    // -----------------------------------------------------------------------

    #[test]
    fn test_silhouette_samples_well_separated() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1])
                .unwrap();
        let labels = array![0isize, 0, 1, 1];
        let samples = silhouette_samples(&x, &labels).unwrap();
        assert_eq!(samples.len(), 4);
        for &s in &samples {
            assert!(s > 0.9, "expected > 0.9, got {s}");
        }
    }

    #[test]
    fn test_silhouette_samples_mean_matches_score() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0_f64, 0.1, 100.0, 100.1]).unwrap();
        let labels = array![0isize, 0, 1, 1];
        let samples = silhouette_samples(&x, &labels).unwrap();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let score: f64 = silhouette_score(&x, &labels).unwrap();
        assert_abs_diff_eq!(mean, score, epsilon = 1e-10);
    }

    #[test]
    fn test_silhouette_samples_noise_gets_zero() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0_f64, 0.1, 50.0, 100.0, 100.1]).unwrap();
        let labels = array![0isize, 0, -1, 1, 1];
        let samples = silhouette_samples(&x, &labels).unwrap();
        assert_abs_diff_eq!(samples[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_silhouette_samples_single_cluster_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0, 0];
        assert!(silhouette_samples(&x, &labels).is_err());
    }

    #[test]
    fn test_silhouette_samples_empty() {
        let x = Array2::<f64>::zeros((0, 2));
        let labels = Array1::<isize>::from_vec(vec![]);
        assert!(silhouette_samples(&x, &labels).is_err());
    }

    // -----------------------------------------------------------------------
    // calinski_harabasz_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_calinski_harabasz_well_separated() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1])
                .unwrap();
        let labels = array![0isize, 0, 1, 1];
        let score = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(score > 100.0, "expected high CH, got {score}");
    }

    #[test]
    fn test_calinski_harabasz_single_cluster_error() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0, 0];
        assert!(calinski_harabasz_score(&x, &labels).is_err());
    }

    #[test]
    fn test_calinski_harabasz_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0_f64, 1.0, 2.0]).unwrap();
        let labels = array![0isize, 0];
        assert!(calinski_harabasz_score(&x, &labels).is_err());
    }

    #[test]
    fn test_calinski_harabasz_better_for_good_clustering() {
        let x =
            Array2::from_shape_vec((6, 1), vec![0.0_f64, 1.0, 2.0, 100.0, 101.0, 102.0]).unwrap();
        let labels_good = array![0isize, 0, 0, 1, 1, 1];
        let labels_bad = array![0isize, 1, 0, 1, 0, 1];
        let ch_good = calinski_harabasz_score(&x, &labels_good).unwrap();
        let ch_bad = calinski_harabasz_score(&x, &labels_bad).unwrap();
        assert!(
            ch_good > ch_bad,
            "good ({ch_good}) should beat bad ({ch_bad})"
        );
    }

    // -----------------------------------------------------------------------
    // homogeneity_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_homogeneity_perfect() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![0isize, 0, 1, 1, 2, 2];
        assert_abs_diff_eq!(
            homogeneity_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_homogeneity_permuted_labels() {
        let labels_true = array![0isize, 0, 1, 1];
        let labels_pred = array![1isize, 1, 0, 0];
        assert_abs_diff_eq!(
            homogeneity_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_homogeneity_single_class_is_one() {
        let labels_true = array![0isize, 0, 0, 0];
        let labels_pred = array![0isize, 1, 0, 1];
        assert_abs_diff_eq!(
            homogeneity_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_homogeneity_empty() {
        let lt = Array1::<isize>::from_vec(vec![]);
        let lp = Array1::<isize>::from_vec(vec![]);
        assert!(homogeneity_score(&lt, &lp).is_err());
    }

    // -----------------------------------------------------------------------
    // completeness_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_completeness_perfect() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![0isize, 0, 1, 1, 2, 2];
        assert_abs_diff_eq!(
            completeness_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_completeness_single_cluster_is_one() {
        let labels_true = array![0isize, 1, 2, 0];
        let labels_pred = array![0isize, 0, 0, 0];
        assert_abs_diff_eq!(
            completeness_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_completeness_empty() {
        let lt = Array1::<isize>::from_vec(vec![]);
        let lp = Array1::<isize>::from_vec(vec![]);
        assert!(completeness_score(&lt, &lp).is_err());
    }

    // -----------------------------------------------------------------------
    // v_measure_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_v_measure_perfect() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![0isize, 0, 1, 1, 2, 2];
        assert_abs_diff_eq!(
            v_measure_score(&labels_true, &labels_pred).unwrap(),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_v_measure_harmonic_mean() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![0isize, 0, 0, 1, 1, 2];
        let h = homogeneity_score(&labels_true, &labels_pred).unwrap();
        let c = completeness_score(&labels_true, &labels_pred).unwrap();
        let v = v_measure_score(&labels_true, &labels_pred).unwrap();
        let expected = 2.0 * h * c / (h + c);
        assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_v_measure_empty() {
        let lt = Array1::<isize>::from_vec(vec![]);
        let lp = Array1::<isize>::from_vec(vec![]);
        assert!(v_measure_score(&lt, &lp).is_err());
    }
}

// ---------------------------------------------------------------------------
// Kani formal verification harnesses
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Helper: generate a symbolic f64 that is finite and within a reasonable
    /// magnitude range to avoid overflow in distance calculations.
    fn any_finite_f64() -> f64 {
        let val: f64 = kani::any();
        kani::assume(!val.is_nan() && !val.is_infinite());
        kani::assume(val.abs() < 1e3);
        val
    }

    /// Prove that silhouette_score output is in [-1.0, 1.0] for valid inputs
    /// with two well-formed clusters.
    ///
    /// We use 4 samples with 1 feature and 2 clusters (labels 0 and 1) to
    /// keep the state space tractable for bounded model checking.
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_silhouette_score_range() {
        const N: usize = 4;
        const D: usize = 1;

        let mut x_data = [0.0f64; N * D];
        for i in 0..(N * D) {
            x_data[i] = any_finite_f64();
        }

        // Assign labels: first two samples to cluster 0, last two to cluster 1.
        // This guarantees exactly 2 clusters each with 2 members.
        let labels_data: [isize; N] = [0, 0, 1, 1];

        let x = Array2::from_shape_vec((N, D), x_data.to_vec()).unwrap();
        let labels = Array1::from_vec(labels_data.to_vec());

        let result = silhouette_score(&x, &labels);
        if let Ok(score) = result {
            assert!(score >= -1.0, "silhouette score must be >= -1.0");
            assert!(score <= 1.0, "silhouette score must be <= 1.0");
        }
    }

    /// Prove that davies_bouldin_score output is >= 0.0 for valid inputs
    /// with two clusters and non-coincident centroids.
    ///
    /// We use 4 samples with 1 feature and 2 clusters (labels 0 and 1).
    #[kani::proof]
    #[kani::unwind(6)]
    fn prove_davies_bouldin_score_non_negative() {
        const N: usize = 4;
        const D: usize = 1;

        let mut x_data = [0.0f64; N * D];
        for i in 0..(N * D) {
            x_data[i] = any_finite_f64();
        }

        // Assign labels: first two samples to cluster 0, last two to cluster 1.
        let labels_data: [isize; N] = [0, 0, 1, 1];

        let x = Array2::from_shape_vec((N, D), x_data.to_vec()).unwrap();
        let labels = Array1::from_vec(labels_data.to_vec());

        let result = davies_bouldin_score(&x, &labels);
        if let Ok(score) = result {
            assert!(score >= 0.0, "Davies-Bouldin score must be >= 0.0");
        }
    }

    // -----------------------------------------------------------------------
    // rand_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_rand_score_perfect() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let ri = rand_score(&labels, &labels).unwrap();
        assert_abs_diff_eq!(ri, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rand_score_all_different() {
        // Every sample in its own cluster in both
        let labels = array![0isize, 1, 2, 3];
        let ri = rand_score(&labels, &labels).unwrap();
        assert_abs_diff_eq!(ri, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rand_score_all_same() {
        // All samples in one cluster
        let labels = array![0isize, 0, 0, 0];
        let ri = rand_score(&labels, &labels).unwrap();
        assert_abs_diff_eq!(ri, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rand_score_known_value() {
        // Manually computed:
        // true: [0,0,1,1], pred: [0,1,0,1]
        // Pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        // same_in_true: (0,1), (2,3)
        // same_in_pred: (0,2), (1,3)
        // agree: pairs same in both = 0, pairs different in both: (0,3), (1,2) = 2
        // RI = (0 + 2) / 6 = 1/3
        let labels_true = array![0isize, 0, 1, 1];
        let labels_pred = array![0isize, 1, 0, 1];
        let ri = rand_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ri, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rand_score_bounds() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![2isize, 1, 0, 0, 1, 2];
        let ri = rand_score(&labels_true, &labels_pred).unwrap();
        assert!(ri >= 0.0 && ri <= 1.0);
    }

    #[test]
    fn test_rand_score_too_few_samples() {
        let labels = array![0isize];
        assert!(rand_score(&labels, &labels).is_err());
    }

    // -----------------------------------------------------------------------
    // normalized_mutual_info_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_nmi_perfect_geometric() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_info_score(&labels, &labels, NmiMethod::Geometric).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_perfect_arithmetic() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_info_score(&labels, &labels, NmiMethod::Arithmetic).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_perfect_min() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_info_score(&labels, &labels, NmiMethod::Min).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_perfect_max() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_info_score(&labels, &labels, NmiMethod::Max).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_bounds() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![2isize, 1, 0, 0, 1, 2];
        let nmi =
            normalized_mutual_info_score(&labels_true, &labels_pred, NmiMethod::Geometric).unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);
    }

    #[test]
    fn test_nmi_single_cluster() {
        // All in one cluster => H(true) = 0 => NMI = 1.0 by convention
        let labels_true = array![0isize, 0, 0, 0];
        let labels_pred = array![0isize, 0, 0, 0];
        let nmi =
            normalized_mutual_info_score(&labels_true, &labels_pred, NmiMethod::Geometric).unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_empty() {
        let labels_true = Array1::<isize>::from_vec(vec![]);
        let labels_pred = Array1::<isize>::from_vec(vec![]);
        assert!(
            normalized_mutual_info_score(&labels_true, &labels_pred, NmiMethod::Geometric).is_err()
        );
    }

    // -----------------------------------------------------------------------
    // fowlkes_mallows_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_fmi_perfect() {
        let labels = array![0isize, 0, 1, 1, 2, 2];
        let fmi = fowlkes_mallows_score(&labels, &labels).unwrap();
        assert_abs_diff_eq!(fmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fmi_known_value() {
        // true: [0,0,1,1], pred: [0,1,0,1]
        // TP = 0 (no pairs same in both)
        // TP + FP = C(2,2) for pred clusters {0,2} and {1,3} => 1 + 1 = 2
        // TP + FN = C(2,2) for true clusters {0,1} and {2,3} => 1 + 1 = 2
        // FMI = 0 / sqrt(2 * 2) = 0
        let labels_true = array![0isize, 0, 1, 1];
        let labels_pred = array![0isize, 1, 0, 1];
        let fmi = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(fmi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fmi_bounds() {
        let labels_true = array![0isize, 0, 1, 1, 2, 2];
        let labels_pred = array![2isize, 1, 0, 0, 1, 2];
        let fmi = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert!(fmi >= 0.0 && fmi <= 1.0);
    }

    #[test]
    fn test_fmi_too_few_samples() {
        let labels = array![0isize];
        assert!(fowlkes_mallows_score(&labels, &labels).is_err());
    }

    #[test]
    fn test_fmi_all_singletons() {
        // Every sample is its own cluster => no pairs in same cluster
        // TP=0, TP+FP=0, TP+FN=0 => FMI = 0.0
        let labels = array![0isize, 1, 2, 3];
        let fmi = fowlkes_mallows_score(&labels, &labels).unwrap();
        assert_abs_diff_eq!(fmi, 0.0, epsilon = 1e-10);
    }
}
