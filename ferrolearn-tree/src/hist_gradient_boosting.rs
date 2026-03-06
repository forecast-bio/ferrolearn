//! Histogram-based gradient boosting classifiers and regressors.
//!
//! This module provides [`HistGradientBoostingClassifier`] and
//! [`HistGradientBoostingRegressor`], which implement histogram-based gradient
//! boosting trees inspired by LightGBM / scikit-learn's HistGradientBoosting*.
//!
//! # Key design choices
//!
//! - **Feature binning**: continuous features are discretised into up to
//!   `max_bins` (default 256) bins using quantile-based bin edges. NaN values
//!   are assigned a dedicated bin.
//! - **Histogram-based split finding**: at each node, gradient/hessian sums are
//!   accumulated into per-bin histograms, making split finding O(n_bins) instead
//!   of O(n log n).
//! - **Subtraction trick**: the child histogram with more samples is computed by
//!   subtracting the smaller child's histogram from the parent, halving the work.
//! - **Missing value support**: NaN values are routed to the bin that yields the
//!   best split, enabling native handling of missing data.
//!
//! # Regression Losses
//!
//! - **`LeastSquares`**: mean squared error; gradient = `y - F(x)`, hessian = 1.
//! - **`LeastAbsoluteDeviation`**: gradient = `sign(y - F(x))`, hessian = 1.
//!
//! # Classification Loss
//!
//! - **`LogLoss`**: binary and multiclass logistic loss (one-vs-rest via softmax).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::HistGradientBoostingRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 1), vec![
//!     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//! ]).unwrap();
//! let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
//!
//! let model = HistGradientBoostingRegressor::<f64>::new()
//!     .with_n_estimators(50)
//!     .with_learning_rate(0.1)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 8);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Loss enums
// ---------------------------------------------------------------------------

/// Loss function for histogram-based gradient boosting regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistRegressionLoss {
    /// Least squares (L2) loss.
    LeastSquares,
    /// Least absolute deviation (L1) loss.
    LeastAbsoluteDeviation,
}

/// Loss function for histogram-based gradient boosting classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistClassificationLoss {
    /// Log-loss (logistic / cross-entropy) for binary and multiclass.
    LogLoss,
}

// ---------------------------------------------------------------------------
// Internal: histogram tree node
// ---------------------------------------------------------------------------

/// A node in a histogram-based gradient boosting tree.
///
/// Unlike [`crate::decision_tree::Node`], thresholds are stored as bin indices
/// rather than raw feature values. Prediction at a split checks whether the
/// sample's bin index for the feature is `<= threshold_bin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HistNode<F> {
    /// An internal split node.
    Split {
        /// Feature index used for the split.
        feature: usize,
        /// Bin index threshold; samples with `bin[feature] <= threshold_bin`
        /// go left.
        threshold_bin: u16,
        /// Whether NaN values should go left (`true`) or right (`false`).
        nan_goes_left: bool,
        /// Index of the left child node in the flat vec.
        left: usize,
        /// Index of the right child node in the flat vec.
        right: usize,
        /// Weighted gain from this split (for feature importance).
        gain: F,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
    /// A leaf node that stores a prediction value.
    Leaf {
        /// Predicted value (raw leaf output, e.g. gradient step).
        value: F,
        /// Number of samples that reached this node during training.
        n_samples: usize,
    },
}

// ---------------------------------------------------------------------------
// Internal: binning
// ---------------------------------------------------------------------------

/// The special bin index reserved for NaN / missing values.
const NAN_BIN: u16 = u16::MAX;

/// Bin edges for a single feature, plus the number of non-NaN bins.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureBinInfo<F> {
    /// Sorted bin edges (upper thresholds). `edges[b]` is the upper bound
    /// for bin `b`. The last bin captures everything above `edges[len-2]`.
    edges: Vec<F>,
    /// Number of non-NaN bins for this feature (at most `max_bins`).
    n_bins: u16,
    /// Whether any NaN was observed for this feature during binning.
    has_nan: bool,
}

/// Compute quantile-based bin edges for every feature.
fn compute_bin_edges<F: Float>(x: &Array2<F>, max_bins: u16) -> Vec<FeatureBinInfo<F>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();
    let mut infos = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        // Separate non-NaN values and sort them.
        let mut vals: Vec<F> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let has_nan = vals.len() < n_samples;

        if vals.is_empty() {
            // All NaN.
            infos.push(FeatureBinInfo {
                edges: Vec::new(),
                n_bins: 0,
                has_nan: true,
            });
            continue;
        }

        // Deduplicate to find unique values.
        let mut unique: Vec<F> = Vec::new();
        for &v in &vals {
            if unique.is_empty() || (v - *unique.last().unwrap()).abs() > F::epsilon() {
                unique.push(v);
            }
        }

        let n_unique = unique.len();
        let actual_bins = (max_bins as usize).min(n_unique);

        if actual_bins <= 1 {
            // Only one unique value — one bin, edge is that value.
            infos.push(FeatureBinInfo {
                edges: vec![unique[0]],
                n_bins: 1,
                has_nan,
            });
            continue;
        }

        // Compute quantile-based bin edges.
        let mut edges = Vec::with_capacity(actual_bins);
        for b in 1..actual_bins {
            let frac = b as f64 / actual_bins as f64;
            let idx_f = frac * (n_unique as f64 - 1.0);
            let lo = idx_f.floor() as usize;
            let hi = (lo + 1).min(n_unique - 1);
            let t = F::from(idx_f - lo as f64).unwrap();
            let edge = unique[lo] * (F::one() - t) + unique[hi] * t;
            // Avoid duplicate edges.
            if edges.is_empty() || (edge - *edges.last().unwrap()).abs() > F::epsilon() {
                edges.push(edge);
            }
        }
        // Add a final edge for the upper bound (max value).
        let last = *unique.last().unwrap();
        if edges.is_empty() || (last - *edges.last().unwrap()).abs() > F::epsilon() {
            edges.push(last);
        }

        let n_bins = edges.len() as u16;
        infos.push(FeatureBinInfo {
            edges,
            n_bins,
            has_nan,
        });
    }

    infos
}

/// Map a single feature value to its bin index given bin edges.
#[inline]
fn map_to_bin<F: Float>(value: F, info: &FeatureBinInfo<F>) -> u16 {
    if value.is_nan() {
        return NAN_BIN;
    }
    if info.n_bins == 0 {
        return NAN_BIN;
    }
    // Binary search: find the first edge >= value.
    let edges = &info.edges;
    let mut lo: usize = 0;
    let mut hi: usize = edges.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if edges[mid] < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo >= edges.len() {
        (info.n_bins - 1).min(edges.len() as u16 - 1)
    } else {
        lo as u16
    }
}

/// Bin all samples for all features.
fn bin_data<F: Float>(x: &Array2<F>, bin_infos: &[FeatureBinInfo<F>]) -> Vec<Vec<u16>> {
    let (n_samples, n_features) = x.dim();
    // bins[i][j] = bin index for sample i, feature j.
    let mut bins = vec![vec![0u16; n_features]; n_samples];
    for j in 0..n_features {
        for i in 0..n_samples {
            bins[i][j] = map_to_bin(x[[i, j]], &bin_infos[j]);
        }
    }
    bins
}

// ---------------------------------------------------------------------------
// Internal: histogram-based tree building
// ---------------------------------------------------------------------------

/// Gradient/hessian accumulator for a single bin.
#[derive(Debug, Clone, Copy, Default)]
struct BinEntry<F> {
    grad_sum: F,
    hess_sum: F,
    count: usize,
}

/// A histogram for one feature: one entry per bin, plus an optional NaN entry.
struct FeatureHistogram<F> {
    bins: Vec<BinEntry<F>>,
    nan_entry: BinEntry<F>,
}

impl<F: Float> FeatureHistogram<F> {
    fn new(n_bins: u16) -> Self {
        Self {
            bins: vec![
                BinEntry {
                    grad_sum: F::zero(),
                    hess_sum: F::zero(),
                    count: 0,
                };
                n_bins as usize
            ],
            nan_entry: BinEntry {
                grad_sum: F::zero(),
                hess_sum: F::zero(),
                count: 0,
            },
        }
    }
}

/// Build histograms for all features from scratch over the given sample indices.
fn build_histograms<F: Float>(
    bin_data: &[Vec<u16>],
    gradients: &[F],
    hessians: &[F],
    sample_indices: &[usize],
    bin_infos: &[FeatureBinInfo<F>],
) -> Vec<FeatureHistogram<F>> {
    let n_features = bin_infos.len();
    let mut histograms: Vec<FeatureHistogram<F>> = bin_infos
        .iter()
        .map(|info| FeatureHistogram::new(info.n_bins))
        .collect();

    for &i in sample_indices {
        let g = gradients[i];
        let h = hessians[i];
        for j in 0..n_features {
            let b = bin_data[i][j];
            if b == NAN_BIN {
                histograms[j].nan_entry.grad_sum = histograms[j].nan_entry.grad_sum + g;
                histograms[j].nan_entry.hess_sum = histograms[j].nan_entry.hess_sum + h;
                histograms[j].nan_entry.count += 1;
            } else {
                let entry = &mut histograms[j].bins[b as usize];
                entry.grad_sum = entry.grad_sum + g;
                entry.hess_sum = entry.hess_sum + h;
                entry.count += 1;
            }
        }
    }

    histograms
}

/// Subtraction trick: compute child histogram from parent and sibling.
fn subtract_histograms<F: Float>(
    parent: &[FeatureHistogram<F>],
    sibling: &[FeatureHistogram<F>],
    bin_infos: &[FeatureBinInfo<F>],
) -> Vec<FeatureHistogram<F>> {
    let n_features = bin_infos.len();
    let mut result: Vec<FeatureHistogram<F>> = bin_infos
        .iter()
        .map(|info| FeatureHistogram::new(info.n_bins))
        .collect();

    for j in 0..n_features {
        let n_bins = bin_infos[j].n_bins as usize;
        for b in 0..n_bins {
            result[j].bins[b].grad_sum = parent[j].bins[b].grad_sum - sibling[j].bins[b].grad_sum;
            result[j].bins[b].hess_sum = parent[j].bins[b].hess_sum - sibling[j].bins[b].hess_sum;
            // count is usize so we need saturating_sub for safety
            result[j].bins[b].count = parent[j].bins[b]
                .count
                .saturating_sub(sibling[j].bins[b].count);
        }
        // NaN bin
        result[j].nan_entry.grad_sum = parent[j].nan_entry.grad_sum - sibling[j].nan_entry.grad_sum;
        result[j].nan_entry.hess_sum = parent[j].nan_entry.hess_sum - sibling[j].nan_entry.hess_sum;
        result[j].nan_entry.count = parent[j]
            .nan_entry
            .count
            .saturating_sub(sibling[j].nan_entry.count);
    }

    result
}

/// Result of finding the best split for a node.
struct SplitCandidate<F> {
    feature: usize,
    threshold_bin: u16,
    gain: F,
    nan_goes_left: bool,
}

/// Find the best split across all features from histograms.
///
/// Uses the standard gain formula:
///   gain = (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - (G_parent^2 / (H_parent + lambda))
fn find_best_split_from_histograms<F: Float>(
    histograms: &[FeatureHistogram<F>],
    bin_infos: &[FeatureBinInfo<F>],
    total_grad: F,
    total_hess: F,
    total_count: usize,
    l2_regularization: F,
    min_samples_leaf: usize,
) -> Option<SplitCandidate<F>> {
    let n_features = bin_infos.len();
    let parent_gain = total_grad * total_grad / (total_hess + l2_regularization);

    let mut best: Option<SplitCandidate<F>> = None;

    for j in 0..n_features {
        let n_bins = bin_infos[j].n_bins as usize;
        if n_bins <= 1 {
            continue;
        }
        let nan = &histograms[j].nan_entry;

        // Try scanning left-to-right through bins.
        // For each split position b, left contains bins 0..=b, right contains b+1..n_bins-1.
        // NaN samples can go either left or right; we try both.
        let mut left_grad = F::zero();
        let mut left_hess = F::zero();
        let mut left_count: usize = 0;

        for b in 0..(n_bins - 1) {
            let entry = &histograms[j].bins[b];
            left_grad = left_grad + entry.grad_sum;
            left_hess = left_hess + entry.hess_sum;
            left_count += entry.count;

            let right_grad_no_nan = total_grad - left_grad - nan.grad_sum;
            let right_hess_no_nan = total_hess - left_hess - nan.hess_sum;
            let right_count_no_nan = total_count
                .saturating_sub(left_count)
                .saturating_sub(nan.count);

            // Try NaN goes left.
            {
                let lg = left_grad + nan.grad_sum;
                let lh = left_hess + nan.hess_sum;
                let lc = left_count + nan.count;
                let rg = right_grad_no_nan;
                let rh = right_hess_no_nan;
                let rc = right_count_no_nan;

                if lc >= min_samples_leaf && rc >= min_samples_leaf {
                    let gain = lg * lg / (lh + l2_regularization)
                        + rg * rg / (rh + l2_regularization)
                        - parent_gain;
                    if gain > F::zero() {
                        let better = match &best {
                            None => true,
                            Some(curr) => gain > curr.gain,
                        };
                        if better {
                            best = Some(SplitCandidate {
                                feature: j,
                                threshold_bin: b as u16,
                                gain,
                                nan_goes_left: true,
                            });
                        }
                    }
                }
            }

            // Try NaN goes right.
            {
                let lg = left_grad;
                let lh = left_hess;
                let lc = left_count;
                let rg = right_grad_no_nan + nan.grad_sum;
                let rh = right_hess_no_nan + nan.hess_sum;
                let rc = right_count_no_nan + nan.count;

                if lc >= min_samples_leaf && rc >= min_samples_leaf {
                    let gain = lg * lg / (lh + l2_regularization)
                        + rg * rg / (rh + l2_regularization)
                        - parent_gain;
                    if gain > F::zero() {
                        let better = match &best {
                            None => true,
                            Some(curr) => gain > curr.gain,
                        };
                        if better {
                            best = Some(SplitCandidate {
                                feature: j,
                                threshold_bin: b as u16,
                                gain,
                                nan_goes_left: false,
                            });
                        }
                    }
                }
            }
        }
    }

    best
}

/// Compute leaf value: -G / (H + lambda).
#[inline]
fn compute_leaf_value<F: Float>(grad_sum: F, hess_sum: F, l2_reg: F) -> F {
    if hess_sum.abs() < F::epsilon() {
        F::zero()
    } else {
        -grad_sum / (hess_sum + l2_reg)
    }
}

/// Parameters for histogram tree building.
struct HistTreeParams<F> {
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    max_leaf_nodes: Option<usize>,
    l2_regularization: F,
}

/// Build a single histogram-based regression tree.
///
/// Returns a vector of `HistNode`s (flat representation).
fn build_hist_tree<F: Float>(
    binned: &[Vec<u16>],
    gradients: &[F],
    hessians: &[F],
    sample_indices: &[usize],
    bin_infos: &[FeatureBinInfo<F>],
    params: &HistTreeParams<F>,
) -> Vec<HistNode<F>> {
    let n_features = bin_infos.len();
    let _ = n_features; // used implicitly through bin_infos

    if sample_indices.is_empty() {
        return vec![HistNode::Leaf {
            value: F::zero(),
            n_samples: 0,
        }];
    }

    // Use a work queue approach for best-first (leaf-wise) or depth-first growth.
    // If max_leaf_nodes is set, use best-first; otherwise depth-first.
    if params.max_leaf_nodes.is_some() {
        build_hist_tree_best_first(
            binned,
            gradients,
            hessians,
            sample_indices,
            bin_infos,
            params,
        )
    } else {
        let mut nodes = Vec::new();
        let hist = build_histograms(binned, gradients, hessians, sample_indices, bin_infos);
        build_hist_tree_recursive(
            binned,
            gradients,
            hessians,
            sample_indices,
            bin_infos,
            params,
            &hist,
            0,
            &mut nodes,
        );
        nodes
    }
}

/// Recursive depth-first histogram tree building.
#[allow(clippy::too_many_arguments)]
fn build_hist_tree_recursive<F: Float>(
    binned: &[Vec<u16>],
    gradients: &[F],
    hessians: &[F],
    sample_indices: &[usize],
    bin_infos: &[FeatureBinInfo<F>],
    params: &HistTreeParams<F>,
    histograms: &[FeatureHistogram<F>],
    depth: usize,
    nodes: &mut Vec<HistNode<F>>,
) -> usize {
    let n = sample_indices.len();
    let grad_sum: F = sample_indices
        .iter()
        .map(|&i| gradients[i])
        .fold(F::zero(), |a, b| a + b);
    let hess_sum: F = sample_indices
        .iter()
        .map(|&i| hessians[i])
        .fold(F::zero(), |a, b| a + b);

    // Check stopping conditions.
    let at_max_depth = params.max_depth.is_some_and(|d| depth >= d);
    let too_few = n < 2 * params.min_samples_leaf;

    if at_max_depth || too_few {
        let idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: compute_leaf_value(grad_sum, hess_sum, params.l2_regularization),
            n_samples: n,
        });
        return idx;
    }

    // Find best split.
    let split = find_best_split_from_histograms(
        histograms,
        bin_infos,
        grad_sum,
        hess_sum,
        n,
        params.l2_regularization,
        params.min_samples_leaf,
    );

    let split = match split {
        Some(s) => s,
        None => {
            let idx = nodes.len();
            nodes.push(HistNode::Leaf {
                value: compute_leaf_value(grad_sum, hess_sum, params.l2_regularization),
                n_samples: n,
            });
            return idx;
        }
    };

    // Partition samples into left and right.
    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
        sample_indices.iter().partition(|&&i| {
            let b = binned[i][split.feature];
            if b == NAN_BIN {
                split.nan_goes_left
            } else {
                b <= split.threshold_bin
            }
        });

    if left_indices.is_empty() || right_indices.is_empty() {
        let idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: compute_leaf_value(grad_sum, hess_sum, params.l2_regularization),
            n_samples: n,
        });
        return idx;
    }

    // Build histograms for the smaller child, then use subtraction trick.
    let (small_indices, _large_indices, small_is_left) =
        if left_indices.len() <= right_indices.len() {
            (&left_indices, &right_indices, true)
        } else {
            (&right_indices, &left_indices, false)
        };

    let small_hist = build_histograms(binned, gradients, hessians, small_indices, bin_infos);
    let large_hist = subtract_histograms(histograms, &small_hist, bin_infos);

    let (left_hist, right_hist) = if small_is_left {
        (small_hist, large_hist)
    } else {
        (large_hist, small_hist)
    };

    // Reserve a placeholder for this split node.
    let node_idx = nodes.len();
    nodes.push(HistNode::Leaf {
        value: F::zero(),
        n_samples: 0,
    }); // placeholder

    // Recurse.
    let left_idx = build_hist_tree_recursive(
        binned,
        gradients,
        hessians,
        &left_indices,
        bin_infos,
        params,
        &left_hist,
        depth + 1,
        nodes,
    );
    let right_idx = build_hist_tree_recursive(
        binned,
        gradients,
        hessians,
        &right_indices,
        bin_infos,
        params,
        &right_hist,
        depth + 1,
        nodes,
    );

    nodes[node_idx] = HistNode::Split {
        feature: split.feature,
        threshold_bin: split.threshold_bin,
        nan_goes_left: split.nan_goes_left,
        left: left_idx,
        right: right_idx,
        gain: split.gain,
        n_samples: n,
    };

    node_idx
}

/// Entry in the best-first priority queue.
struct SplitTask {
    /// Indices of samples at this node.
    sample_indices: Vec<usize>,
    /// The node index in the flat node vec (a leaf placeholder).
    node_idx: usize,
    /// Depth of this node.
    depth: usize,
    /// The gain of the best split at this node.
    gain: f64,
    /// Feature of the best split.
    feature: usize,
    /// Bin threshold of the best split.
    threshold_bin: u16,
    /// Whether NaN goes left.
    nan_goes_left: bool,
}

/// Build a histogram tree using best-first (leaf-wise) growth with max_leaf_nodes.
fn build_hist_tree_best_first<F: Float>(
    binned: &[Vec<u16>],
    gradients: &[F],
    hessians: &[F],
    sample_indices: &[usize],
    bin_infos: &[FeatureBinInfo<F>],
    params: &HistTreeParams<F>,
) -> Vec<HistNode<F>> {
    let max_leaves = params.max_leaf_nodes.unwrap_or(usize::MAX);
    let mut nodes: Vec<HistNode<F>> = Vec::new();

    // Root node.
    let n = sample_indices.len();
    let grad_sum: F = sample_indices
        .iter()
        .map(|&i| gradients[i])
        .fold(F::zero(), |a, b| a + b);
    let hess_sum: F = sample_indices
        .iter()
        .map(|&i| hessians[i])
        .fold(F::zero(), |a, b| a + b);

    let root_idx = nodes.len();
    nodes.push(HistNode::Leaf {
        value: compute_leaf_value(grad_sum, hess_sum, params.l2_regularization),
        n_samples: n,
    });

    let root_hist = build_histograms(binned, gradients, hessians, sample_indices, bin_infos);
    let root_split = find_best_split_from_histograms(
        &root_hist,
        bin_infos,
        grad_sum,
        hess_sum,
        n,
        params.l2_regularization,
        params.min_samples_leaf,
    );

    let mut pending: Vec<(SplitTask, Vec<FeatureHistogram<F>>)> = Vec::new();
    let mut n_leaves: usize = 1;

    if let Some(split) = root_split {
        let at_max_depth = params.max_depth.is_some_and(|d| d == 0);
        if !at_max_depth {
            pending.push((
                SplitTask {
                    sample_indices: sample_indices.to_vec(),
                    node_idx: root_idx,
                    depth: 0,
                    gain: split.gain.to_f64().unwrap_or(0.0),
                    feature: split.feature,
                    threshold_bin: split.threshold_bin,
                    nan_goes_left: split.nan_goes_left,
                },
                root_hist,
            ));
        }
    }

    while !pending.is_empty() && n_leaves < max_leaves {
        // Pick the task with the highest gain.
        let best_idx = pending
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.0.gain
                    .partial_cmp(&b.0.gain)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        let (task, parent_hist) = pending.swap_remove(best_idx);

        // Partition.
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
            task.sample_indices.iter().partition(|&&i| {
                let b = binned[i][task.feature];
                if b == NAN_BIN {
                    task.nan_goes_left
                } else {
                    b <= task.threshold_bin
                }
            });

        if left_indices.is_empty() || right_indices.is_empty() {
            continue;
        }

        // Build histograms using subtraction trick.
        let (small_indices, _large_indices, small_is_left) =
            if left_indices.len() <= right_indices.len() {
                (&left_indices, &right_indices, true)
            } else {
                (&right_indices, &left_indices, false)
            };

        let small_hist = build_histograms(binned, gradients, hessians, small_indices, bin_infos);
        let large_hist = subtract_histograms(&parent_hist, &small_hist, bin_infos);

        let (left_hist, right_hist) = if small_is_left {
            (small_hist, large_hist)
        } else {
            (large_hist, small_hist)
        };

        // Create left and right leaf nodes.
        let left_grad: F = left_indices
            .iter()
            .map(|&i| gradients[i])
            .fold(F::zero(), |a, b| a + b);
        let left_hess: F = left_indices
            .iter()
            .map(|&i| hessians[i])
            .fold(F::zero(), |a, b| a + b);
        let right_grad: F = right_indices
            .iter()
            .map(|&i| gradients[i])
            .fold(F::zero(), |a, b| a + b);
        let right_hess: F = right_indices
            .iter()
            .map(|&i| hessians[i])
            .fold(F::zero(), |a, b| a + b);

        let left_idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: compute_leaf_value(left_grad, left_hess, params.l2_regularization),
            n_samples: left_indices.len(),
        });
        let right_idx = nodes.len();
        nodes.push(HistNode::Leaf {
            value: compute_leaf_value(right_grad, right_hess, params.l2_regularization),
            n_samples: right_indices.len(),
        });

        // Convert the parent leaf placeholder into a split node.
        nodes[task.node_idx] = HistNode::Split {
            feature: task.feature,
            threshold_bin: task.threshold_bin,
            nan_goes_left: task.nan_goes_left,
            left: left_idx,
            right: right_idx,
            gain: F::from(task.gain).unwrap(),
            n_samples: task.sample_indices.len(),
        };

        // One leaf became two, so net +1 leaf.
        n_leaves += 1;

        let child_depth = task.depth + 1;
        let at_max_depth = params.max_depth.is_some_and(|d| child_depth >= d);

        if !at_max_depth && n_leaves < max_leaves {
            // Try to split left child.
            if left_indices.len() >= 2 * params.min_samples_leaf {
                let left_split = find_best_split_from_histograms(
                    &left_hist,
                    bin_infos,
                    left_grad,
                    left_hess,
                    left_indices.len(),
                    params.l2_regularization,
                    params.min_samples_leaf,
                );
                if let Some(s) = left_split {
                    pending.push((
                        SplitTask {
                            sample_indices: left_indices,
                            node_idx: left_idx,
                            depth: child_depth,
                            gain: s.gain.to_f64().unwrap_or(0.0),
                            feature: s.feature,
                            threshold_bin: s.threshold_bin,
                            nan_goes_left: s.nan_goes_left,
                        },
                        left_hist,
                    ));
                }
            }

            // Try to split right child.
            if right_indices.len() >= 2 * params.min_samples_leaf {
                let right_split = find_best_split_from_histograms(
                    &right_hist,
                    bin_infos,
                    right_grad,
                    right_hess,
                    right_indices.len(),
                    params.l2_regularization,
                    params.min_samples_leaf,
                );
                if let Some(s) = right_split {
                    pending.push((
                        SplitTask {
                            sample_indices: right_indices,
                            node_idx: right_idx,
                            depth: child_depth,
                            gain: s.gain.to_f64().unwrap_or(0.0),
                            feature: s.feature,
                            threshold_bin: s.threshold_bin,
                            nan_goes_left: s.nan_goes_left,
                        },
                        right_hist,
                    ));
                }
            }
        }
    }

    nodes
}

/// Traverse a histogram tree to find the leaf for a single binned sample.
#[inline]
fn traverse_hist_tree<F: Float>(nodes: &[HistNode<F>], sample_bins: &[u16]) -> usize {
    let mut idx = 0;
    loop {
        match &nodes[idx] {
            HistNode::Split {
                feature,
                threshold_bin,
                nan_goes_left,
                left,
                right,
                ..
            } => {
                let b = sample_bins[*feature];
                if b == NAN_BIN {
                    idx = if *nan_goes_left { *left } else { *right };
                } else if b <= *threshold_bin {
                    idx = *left;
                } else {
                    idx = *right;
                }
            }
            HistNode::Leaf { .. } => return idx,
        }
    }
}

/// Compute feature importances from a histogram tree's gain values.
fn compute_hist_feature_importances<F: Float>(
    nodes: &[HistNode<F>],
    n_features: usize,
) -> Array1<F> {
    let mut importances = Array1::zeros(n_features);
    for node in nodes {
        if let HistNode::Split { feature, gain, .. } = node {
            importances[*feature] = importances[*feature] + *gain;
        }
    }
    importances
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// Compute softmax probabilities for each class across all samples.
///
/// Returns `probs[k][i]` = probability of class k for sample i.
fn softmax_matrix<F: Float>(
    f_vals: &[Array1<F>],
    n_samples: usize,
    n_classes: usize,
) -> Vec<Vec<F>> {
    let mut probs: Vec<Vec<F>> = vec![vec![F::zero(); n_samples]; n_classes];
    for i in 0..n_samples {
        let max_val = (0..n_classes)
            .map(|k| f_vals[k][i])
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        let mut sum = F::zero();
        let mut exps = vec![F::zero(); n_classes];
        for k in 0..n_classes {
            exps[k] = (f_vals[k][i] - max_val).exp();
            sum = sum + exps[k];
        }
        let eps = F::from(1e-15).unwrap();
        if sum < eps {
            sum = eps;
        }
        for k in 0..n_classes {
            probs[k][i] = exps[k] / sum;
        }
    }
    probs
}

// ---------------------------------------------------------------------------
// HistGradientBoostingRegressor
// ---------------------------------------------------------------------------

/// Histogram-based gradient boosting regressor.
///
/// Uses quantile-based feature binning and gradient/hessian histograms for
/// O(n_bins) split finding per node. This is significantly faster than the
/// standard [`GradientBoostingRegressor`](crate::GradientBoostingRegressor)
/// for larger datasets.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistGradientBoostingRegressor<F> {
    /// Number of boosting stages (trees).
    pub n_estimators: usize,
    /// Learning rate (shrinkage) applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Maximum number of bins for feature discretisation (at most 256).
    pub max_bins: u16,
    /// L2 regularization term on weights.
    pub l2_regularization: f64,
    /// Maximum number of leaf nodes per tree (best-first growth).
    /// If `None`, depth-first growth is used with `max_depth`.
    pub max_leaf_nodes: Option<usize>,
    /// Loss function.
    pub loss: HistRegressionLoss,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> HistGradientBoostingRegressor<F> {
    /// Create a new `HistGradientBoostingRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `learning_rate = 0.1`,
    /// `max_depth = None`, `min_samples_leaf = 20`,
    /// `max_bins = 255`, `l2_regularization = 0.0`,
    /// `max_leaf_nodes = Some(31)`, `loss = LeastSquares`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: None,
            min_samples_leaf: 20,
            max_bins: 255,
            l2_regularization: 0.0,
            max_leaf_nodes: Some(31),
            loss: HistRegressionLoss::LeastSquares,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate (shrinkage).
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set the maximum number of bins for feature discretisation.
    #[must_use]
    pub fn with_max_bins(mut self, bins: u16) -> Self {
        self.max_bins = bins;
        self
    }

    /// Set the L2 regularization term.
    #[must_use]
    pub fn with_l2_regularization(mut self, reg: f64) -> Self {
        self.l2_regularization = reg;
        self
    }

    /// Set the maximum number of leaf nodes (best-first growth).
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, n: Option<usize>) -> Self {
        self.max_leaf_nodes = n;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: HistRegressionLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for HistGradientBoostingRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedHistGradientBoostingRegressor
// ---------------------------------------------------------------------------

/// A fitted histogram-based gradient boosting regressor.
///
/// Stores the binning information, initial prediction, and the sequence of
/// fitted histogram trees. Predictions are computed by binning the input
/// features and traversing each tree.
#[derive(Debug, Clone)]
pub struct FittedHistGradientBoostingRegressor<F> {
    /// Bin edge information for each feature.
    bin_infos: Vec<FeatureBinInfo<F>>,
    /// Initial prediction (baseline).
    init: F,
    /// Learning rate used during training.
    learning_rate: F,
    /// Sequence of fitted histogram trees.
    trees: Vec<Vec<HistNode<F>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>>
    for HistGradientBoostingRegressor<F>
{
    type Fitted = FittedHistGradientBoostingRegressor<F>;
    type Error = FerroError;

    /// Fit the histogram-based gradient boosting regressor.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedHistGradientBoostingRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate inputs.
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "HistGradientBoostingRegressor requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.max_bins < 2 {
            return Err(FerroError::InvalidParameter {
                name: "max_bins".into(),
                reason: "must be at least 2".into(),
            });
        }

        let lr = F::from(self.learning_rate).unwrap();
        let l2_reg = F::from(self.l2_regularization).unwrap();

        // Compute bin edges and bin the data.
        let bin_infos = compute_bin_edges(x, self.max_bins);
        let binned = bin_data(x, &bin_infos);

        // Initial prediction.
        let init = match self.loss {
            HistRegressionLoss::LeastSquares => {
                let sum: F = y.iter().copied().fold(F::zero(), |a, b| a + b);
                sum / F::from(n_samples).unwrap()
            }
            HistRegressionLoss::LeastAbsoluteDeviation => median_f(y),
        };

        let mut f_vals = Array1::from_elem(n_samples, init);
        let all_indices: Vec<usize> = (0..n_samples).collect();

        let tree_params = HistTreeParams {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            max_leaf_nodes: self.max_leaf_nodes,
            l2_regularization: l2_reg,
        };

        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute gradients and hessians.
            let (gradients, hessians) = match self.loss {
                HistRegressionLoss::LeastSquares => {
                    let grads: Vec<F> = (0..n_samples).map(|i| -(y[i] - f_vals[i])).collect();
                    let hess: Vec<F> = vec![F::one(); n_samples];
                    (grads, hess)
                }
                HistRegressionLoss::LeastAbsoluteDeviation => {
                    let grads: Vec<F> = (0..n_samples)
                        .map(|i| {
                            let diff = y[i] - f_vals[i];
                            if diff > F::zero() {
                                -F::one()
                            } else if diff < F::zero() {
                                F::one()
                            } else {
                                F::zero()
                            }
                        })
                        .collect();
                    let hess: Vec<F> = vec![F::one(); n_samples];
                    (grads, hess)
                }
            };

            let tree = build_hist_tree(
                &binned,
                &gradients,
                &hessians,
                &all_indices,
                &bin_infos,
                &tree_params,
            );

            // Update predictions.
            for i in 0..n_samples {
                let leaf_idx = traverse_hist_tree(&tree, &binned[i]);
                if let HistNode::Leaf { value, .. } = tree[leaf_idx] {
                    f_vals[i] = f_vals[i] + lr * value;
                }
            }

            trees.push(tree);
        }

        // Compute feature importances.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees {
            total_importances =
                total_importances + compute_hist_feature_importances(tree_nodes, n_features);
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedHistGradientBoostingRegressor {
            bin_infos,
            init,
            learning_rate: lr,
            trees,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>>
    for FittedHistGradientBoostingRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let binned = bin_data(x, &self.bin_infos);
        let mut predictions = Array1::from_elem(n_samples, self.init);

        for i in 0..n_samples {
            for tree_nodes in &self.trees {
                let leaf_idx = traverse_hist_tree(tree_nodes, &binned[i]);
                if let HistNode::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    predictions[i] = predictions[i] + self.learning_rate * value;
                }
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedHistGradientBoostingRegressor<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for HistGradientBoostingRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedHistGradientBoostingRegressor<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// HistGradientBoostingClassifier
// ---------------------------------------------------------------------------

/// Histogram-based gradient boosting classifier.
///
/// For binary classification a single model is trained on log-odds residuals.
/// For multiclass (*K* classes), *K* histogram trees are built per boosting
/// round (one-vs-rest in probability space via softmax).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistGradientBoostingClassifier<F> {
    /// Number of boosting stages.
    pub n_estimators: usize,
    /// Learning rate (shrinkage).
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Maximum number of bins for feature discretisation (at most 256).
    pub max_bins: u16,
    /// L2 regularization term on weights.
    pub l2_regularization: f64,
    /// Maximum number of leaf nodes per tree (best-first growth).
    pub max_leaf_nodes: Option<usize>,
    /// Classification loss function.
    pub loss: HistClassificationLoss,
    /// Random seed for reproducibility (reserved for future subsampling).
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> HistGradientBoostingClassifier<F> {
    /// Create a new `HistGradientBoostingClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `learning_rate = 0.1`,
    /// `max_depth = None`, `min_samples_leaf = 20`,
    /// `max_bins = 255`, `l2_regularization = 0.0`,
    /// `max_leaf_nodes = Some(31)`, `loss = LogLoss`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: None,
            min_samples_leaf: 20,
            max_bins: 255,
            l2_regularization: 0.0,
            max_leaf_nodes: Some(31),
            loss: HistClassificationLoss::LogLoss,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate (shrinkage).
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set the maximum number of bins for feature discretisation.
    #[must_use]
    pub fn with_max_bins(mut self, bins: u16) -> Self {
        self.max_bins = bins;
        self
    }

    /// Set the L2 regularization term.
    #[must_use]
    pub fn with_l2_regularization(mut self, reg: f64) -> Self {
        self.l2_regularization = reg;
        self
    }

    /// Set the maximum number of leaf nodes (best-first growth).
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, n: Option<usize>) -> Self {
        self.max_leaf_nodes = n;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for HistGradientBoostingClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedHistGradientBoostingClassifier
// ---------------------------------------------------------------------------

/// A fitted histogram-based gradient boosting classifier.
///
/// For binary classification, stores a single sequence of trees predicting log-odds.
/// For multiclass, stores `K` sequences of trees (one per class).
#[derive(Debug, Clone)]
pub struct FittedHistGradientBoostingClassifier<F> {
    /// Bin edge information for each feature.
    bin_infos: Vec<FeatureBinInfo<F>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Initial predictions per class (log-odds or log-prior).
    init: Vec<F>,
    /// Learning rate.
    learning_rate: F,
    /// Trees: for binary, `trees[0]` has all trees. For multiclass,
    /// `trees[k]` has trees for class k.
    trees: Vec<Vec<Vec<HistNode<F>>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for HistGradientBoostingClassifier<F>
{
    type Fitted = FittedHistGradientBoostingClassifier<F>;
    type Error = FerroError;

    /// Fit the histogram-based gradient boosting classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedHistGradientBoostingClassifier<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "HistGradientBoostingClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.max_bins < 2 {
            return Err(FerroError::InvalidParameter {
                name: "max_bins".into(),
                reason: "must be at least 2".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "need at least 2 distinct classes".into(),
            });
        }

        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        let lr = F::from(self.learning_rate).unwrap();
        let l2_reg = F::from(self.l2_regularization).unwrap();

        // Bin the data.
        let bin_infos = compute_bin_edges(x, self.max_bins);
        let binned = bin_data(x, &bin_infos);

        let tree_params = HistTreeParams {
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            max_leaf_nodes: self.max_leaf_nodes,
            l2_regularization: l2_reg,
        };

        let all_indices: Vec<usize> = (0..n_samples).collect();

        if n_classes == 2 {
            self.fit_binary(
                &binned,
                &y_mapped,
                n_samples,
                n_features,
                &classes,
                lr,
                &tree_params,
                &all_indices,
                &bin_infos,
            )
        } else {
            self.fit_multiclass(
                &binned,
                &y_mapped,
                n_samples,
                n_features,
                n_classes,
                &classes,
                lr,
                &tree_params,
                &all_indices,
                &bin_infos,
            )
        }
    }
}

impl<F: Float + Send + Sync + 'static> HistGradientBoostingClassifier<F> {
    /// Fit binary classification (log-loss on log-odds).
    #[allow(clippy::too_many_arguments)]
    fn fit_binary(
        &self,
        binned: &[Vec<u16>],
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        classes: &[usize],
        lr: F,
        tree_params: &HistTreeParams<F>,
        all_indices: &[usize],
        bin_infos: &[FeatureBinInfo<F>],
    ) -> Result<FittedHistGradientBoostingClassifier<F>, FerroError> {
        // Initial log-odds.
        let pos_count = y_mapped.iter().filter(|&&c| c == 1).count();
        let p = F::from(pos_count).unwrap() / F::from(n_samples).unwrap();
        let eps = F::from(1e-15).unwrap();
        let p_clipped = p.max(eps).min(F::one() - eps);
        let init_val = (p_clipped / (F::one() - p_clipped)).ln();

        let mut f_vals = Array1::from_elem(n_samples, init_val);
        let mut trees_seq: Vec<Vec<HistNode<F>>> = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute probabilities.
            let probs: Vec<F> = f_vals.iter().map(|&fv| sigmoid(fv)).collect();

            // Gradients and hessians for log-loss:
            //   gradient = p - y (we negate pseudo-residual because tree fits -gradient)
            //   hessian = p * (1 - p)
            let gradients: Vec<F> = (0..n_samples)
                .map(|i| {
                    let yi = F::from(y_mapped[i]).unwrap();
                    probs[i] - yi
                })
                .collect();
            let hessians: Vec<F> = (0..n_samples)
                .map(|i| {
                    let pi = probs[i].max(eps).min(F::one() - eps);
                    pi * (F::one() - pi)
                })
                .collect();

            let tree = build_hist_tree(
                binned,
                &gradients,
                &hessians,
                all_indices,
                bin_infos,
                tree_params,
            );

            // Update f_vals.
            for i in 0..n_samples {
                let leaf_idx = traverse_hist_tree(&tree, &binned[i]);
                if let HistNode::Leaf { value, .. } = tree[leaf_idx] {
                    f_vals[i] = f_vals[i] + lr * value;
                }
            }

            trees_seq.push(tree);
        }

        // Feature importances.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees_seq {
            total_importances =
                total_importances + compute_hist_feature_importances(tree_nodes, n_features);
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedHistGradientBoostingClassifier {
            bin_infos: bin_infos.to_vec(),
            classes: classes.to_vec(),
            init: vec![init_val],
            learning_rate: lr,
            trees: vec![trees_seq],
            n_features,
            feature_importances: total_importances,
        })
    }

    /// Fit multiclass classification (K trees per round, softmax).
    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass(
        &self,
        binned: &[Vec<u16>],
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
        lr: F,
        tree_params: &HistTreeParams<F>,
        all_indices: &[usize],
        bin_infos: &[FeatureBinInfo<F>],
    ) -> Result<FittedHistGradientBoostingClassifier<F>, FerroError> {
        // Initial log-prior for each class.
        let mut class_counts = vec![0usize; n_classes];
        for &c in y_mapped {
            class_counts[c] += 1;
        }
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-15).unwrap();
        let init_vals: Vec<F> = class_counts
            .iter()
            .map(|&cnt| {
                let p = (F::from(cnt).unwrap() / n_f).max(eps);
                p.ln()
            })
            .collect();

        let mut f_vals: Vec<Array1<F>> = init_vals
            .iter()
            .map(|&init| Array1::from_elem(n_samples, init))
            .collect();

        let mut trees_per_class: Vec<Vec<Vec<HistNode<F>>>> = (0..n_classes)
            .map(|_| Vec::with_capacity(self.n_estimators))
            .collect();

        for _ in 0..self.n_estimators {
            let probs = softmax_matrix(&f_vals, n_samples, n_classes);

            for k in 0..n_classes {
                // Gradients and hessians for softmax cross-entropy:
                //   gradient_k = p_k - y_k
                //   hessian_k  = p_k * (1 - p_k)
                let gradients: Vec<F> = (0..n_samples)
                    .map(|i| {
                        let yi_k = if y_mapped[i] == k {
                            F::one()
                        } else {
                            F::zero()
                        };
                        probs[k][i] - yi_k
                    })
                    .collect();
                let hessians: Vec<F> = (0..n_samples)
                    .map(|i| {
                        let pk = probs[k][i].max(eps).min(F::one() - eps);
                        pk * (F::one() - pk)
                    })
                    .collect();

                let tree = build_hist_tree(
                    binned,
                    &gradients,
                    &hessians,
                    all_indices,
                    bin_infos,
                    tree_params,
                );

                // Update f_vals for class k.
                for (i, fv) in f_vals[k].iter_mut().enumerate() {
                    let leaf_idx = traverse_hist_tree(&tree, &binned[i]);
                    if let HistNode::Leaf { value, .. } = tree[leaf_idx] {
                        *fv = *fv + lr * value;
                    }
                }

                trees_per_class[k].push(tree);
            }
        }

        // Feature importances aggregated across all classes and rounds.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for class_trees in &trees_per_class {
            for tree_nodes in class_trees {
                total_importances =
                    total_importances + compute_hist_feature_importances(tree_nodes, n_features);
            }
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedHistGradientBoostingClassifier {
            bin_infos: bin_infos.to_vec(),
            classes: classes.to_vec(),
            init: init_vals,
            learning_rate: lr,
            trees: trees_per_class,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>>
    for FittedHistGradientBoostingClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let binned = bin_data(x, &self.bin_infos);

        if n_classes == 2 {
            let init = self.init[0];
            let mut predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let mut f_val = init;
                for tree_nodes in &self.trees[0] {
                    let leaf_idx = traverse_hist_tree(tree_nodes, &binned[i]);
                    if let HistNode::Leaf { value, .. } = tree_nodes[leaf_idx] {
                        f_val = f_val + self.learning_rate * value;
                    }
                }
                let prob = sigmoid(f_val);
                let class_idx = if prob >= F::from(0.5).unwrap() { 1 } else { 0 };
                predictions[i] = self.classes[class_idx];
            }
            Ok(predictions)
        } else {
            let mut predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let mut scores = Vec::with_capacity(n_classes);
                for k in 0..n_classes {
                    let mut f_val = self.init[k];
                    for tree_nodes in &self.trees[k] {
                        let leaf_idx = traverse_hist_tree(tree_nodes, &binned[i]);
                        if let HistNode::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            f_val = f_val + self.learning_rate * value;
                        }
                    }
                    scores.push(f_val);
                }
                let best_k = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(k, _)| k)
                    .unwrap_or(0);
                predictions[i] = self.classes[best_k];
            }
            Ok(predictions)
        }
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedHistGradientBoostingClassifier<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedHistGradientBoostingClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for HistGradientBoostingClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedHgbcPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedHistGradientBoostingClassifier<F>`.
struct FittedHgbcPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedHistGradientBoostingClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedHgbcPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the median of an `Array1`.
fn median_f<F: Float>(arr: &Array1<F>) -> F {
    let mut sorted: Vec<F> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / F::from(2.0).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- Binning tests --

    #[test]
    fn test_bin_edges_simple() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let infos = compute_bin_edges(&x, 4);
        assert_eq!(infos.len(), 1);
        // Should have up to 4 bins.
        assert!(infos[0].n_bins <= 4);
        assert!(infos[0].n_bins >= 2);
        assert!(!infos[0].has_nan);
    }

    #[test]
    fn test_bin_edges_with_nan() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]).unwrap();
        let infos = compute_bin_edges(&x, 4);
        assert_eq!(infos.len(), 1);
        assert!(infos[0].has_nan);
        assert!(infos[0].n_bins >= 1);
    }

    #[test]
    fn test_bin_edges_all_nan() {
        let x = Array2::from_shape_vec((3, 1), vec![f64::NAN, f64::NAN, f64::NAN]).unwrap();
        let infos = compute_bin_edges(&x, 4);
        assert_eq!(infos[0].n_bins, 0);
        assert!(infos[0].has_nan);
    }

    #[test]
    fn test_map_to_bin_basic() {
        let info = FeatureBinInfo {
            edges: vec![2.0, 4.0, 6.0, 8.0],
            n_bins: 4,
            has_nan: false,
        };
        assert_eq!(map_to_bin(1.0, &info), 0);
        assert_eq!(map_to_bin(2.0, &info), 0);
        assert_eq!(map_to_bin(3.0, &info), 1);
        assert_eq!(map_to_bin(5.0, &info), 2);
        assert_eq!(map_to_bin(7.0, &info), 3);
        assert_eq!(map_to_bin(9.0, &info), 3);
        assert_eq!(map_to_bin(f64::NAN, &info), NAN_BIN);
    }

    #[test]
    fn test_bin_data_roundtrip() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();
        let infos = compute_bin_edges(&x, 255);
        let binned = bin_data(&x, &infos);
        assert_eq!(binned.len(), 4);
        assert_eq!(binned[0].len(), 2);
        // Values should be monotonically non-decreasing since input is sorted.
        for j in 0..2 {
            for i in 1..4 {
                assert!(binned[i][j] >= binned[i - 1][j]);
            }
        }
    }

    // -- Subtraction trick test --

    #[test]
    fn test_subtraction_trick() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bin_infos = compute_bin_edges(&x, 255);
        let binned = bin_data(&x, &bin_infos);

        let gradients = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let hessians = vec![1.0; 6];
        let all_indices: Vec<usize> = (0..6).collect();
        let left_indices: Vec<usize> = vec![0, 1, 2];
        let right_indices: Vec<usize> = vec![3, 4, 5];

        let parent_hist =
            build_histograms(&binned, &gradients, &hessians, &all_indices, &bin_infos);
        let left_hist = build_histograms(&binned, &gradients, &hessians, &left_indices, &bin_infos);
        let right_from_sub = subtract_histograms(&parent_hist, &left_hist, &bin_infos);
        let right_direct =
            build_histograms(&binned, &gradients, &hessians, &right_indices, &bin_infos);

        // The subtraction trick result should match direct computation.
        for j in 0..bin_infos.len() {
            let n_bins = bin_infos[j].n_bins as usize;
            for b in 0..n_bins {
                assert_relative_eq!(
                    right_from_sub[j].bins[b].grad_sum,
                    right_direct[j].bins[b].grad_sum,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    right_from_sub[j].bins[b].hess_sum,
                    right_direct[j].bins[b].hess_sum,
                    epsilon = 1e-10
                );
                assert_eq!(
                    right_from_sub[j].bins[b].count,
                    right_direct[j].bins[b].count
                );
            }
        }
    }

    // -- Regressor tests --

    #[test]
    fn test_hgbr_simple_least_squares() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert!(preds[i] < 3.0, "Expected ~1.0, got {}", preds[i]);
        }
        for i in 4..8 {
            assert!(preds[i] > 3.0, "Expected ~5.0, got {}", preds[i]);
        }
    }

    #[test]
    fn test_hgbr_lad_loss() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_loss(HistRegressionLoss::LeastAbsoluteDeviation)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert!(preds[i] < 3.5, "LAD expected <3.5, got {}", preds[i]);
        }
        for i in 4..8 {
            assert!(preds[i] > 2.5, "LAD expected >2.5, got {}", preds[i]);
        }
    }

    #[test]
    fn test_hgbr_reproducibility() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (p1, p2) in preds1.iter().zip(preds2.iter()) {
            assert_relative_eq!(*p1, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hgbr_feature_importances() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 3);
        // First feature should be most important since it's the only one with variance.
        assert!(
            importances[0] > importances[1],
            "Expected imp[0]={} > imp[1]={}",
            importances[0],
            importances[1]
        );
    }

    #[test]
    fn test_hgbr_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = HistGradientBoostingRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbr_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_hgbr_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = HistGradientBoostingRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbr_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = HistGradientBoostingRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbr_invalid_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbr_invalid_max_bins() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_max_bins(1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbr_default_trait() {
        let model = HistGradientBoostingRegressor::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert!((model.learning_rate - 0.1).abs() < 1e-10);
        assert_eq!(model.max_bins, 255);
    }

    #[test]
    fn test_hgbr_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_hgbr_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let model = HistGradientBoostingRegressor::<f32>::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_hgbr_nan_handling() {
        // Some features have NaN — the model should handle them gracefully.
        let x = Array2::from_shape_vec(
            (8, 1),
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN, 7.0, 8.0],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
        // Should still produce finite predictions for all samples (including NaN inputs).
        for p in preds.iter() {
            assert!(p.is_finite(), "Expected finite prediction, got {}", p);
        }
    }

    #[test]
    fn test_hgbr_convergence() {
        // MSE should decrease as we add more estimators.
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let mse = |preds: &Array1<f64>, y: &Array1<f64>| -> f64 {
            preds
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / y.len() as f64
        };

        let model_few = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted_few = model_few.fit(&x, &y).unwrap();
        let preds_few = fitted_few.predict(&x).unwrap();
        let mse_few = mse(&preds_few, &y);

        let model_many = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted_many = model_many.fit(&x, &y).unwrap();
        let preds_many = fitted_many.predict(&x).unwrap();
        let mse_many = mse(&preds_many, &y);

        assert!(
            mse_many < mse_few,
            "Expected MSE to decrease with more estimators: {} (50) vs {} (5)",
            mse_many,
            mse_few
        );
    }

    #[test]
    fn test_hgbr_max_leaf_nodes() {
        // Test that best-first growth with max_leaf_nodes works.
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(Some(4))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_hgbr_l2_regularization() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        // With very high regularization, predictions should be closer to the mean.
        let model_noreg = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_l2_regularization(0.0)
            .with_random_state(42);
        let fitted_noreg = model_noreg.fit(&x, &y).unwrap();
        let preds_noreg = fitted_noreg.predict(&x).unwrap();

        let model_highreg = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_l2_regularization(100.0)
            .with_random_state(42);
        let fitted_highreg = model_highreg.fit(&x, &y).unwrap();
        let preds_highreg = fitted_highreg.predict(&x).unwrap();

        // With high reg, variance of predictions should be smaller.
        let var = |preds: &Array1<f64>| -> f64 {
            let mean = preds.mean().unwrap();
            preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / preds.len() as f64
        };

        assert!(
            var(&preds_highreg) < var(&preds_noreg),
            "High regularization should reduce prediction variance"
        );
    }

    // -- Classifier tests --

    #[test]
    fn test_hgbc_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0, "Expected 0 at index {}, got {}", i, preds[i]);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1, "Expected 1 at index {}, got {}", i, preds[i]);
        }
    }

    #[test]
    fn test_hgbc_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/9 correct, got {}/9",
            correct
        );
    }

    #[test]
    fn test_hgbc_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_hgbc_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_hgbc_feature_importances() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 3);
        assert!(importances[0] > importances[1]);
    }

    #[test]
    fn test_hgbc_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = HistGradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbc_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_hgbc_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = HistGradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbc_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = HistGradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbc_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_hgbc_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_hgbc_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = HistGradientBoostingClassifier::<f32>::new()
            .with_n_estimators(10)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_hgbc_default_trait() {
        let model = HistGradientBoostingClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert!((model.learning_rate - 0.1).abs() < 1e-10);
        assert_eq!(model.max_bins, 255);
    }

    #[test]
    fn test_hgbc_non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![10, 10, 10, 20, 20, 20];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for &p in preds.iter() {
            assert!(p == 10 || p == 20);
        }
    }

    #[test]
    fn test_hgbc_nan_handling() {
        // Classifier should handle NaN features.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0,
                f64::NAN,
                2.0,
                3.0,
                f64::NAN,
                3.0,
                4.0,
                4.0,
                5.0,
                6.0,
                6.0,
                f64::NAN,
                7.0,
                8.0,
                f64::NAN,
                9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
        // All predictions should be valid class labels.
        for &p in preds.iter() {
            assert!(p == 0 || p == 1);
        }
    }

    // -- Comparison with standard GBM --

    #[test]
    fn test_hist_vs_standard_gbm_similar_accuracy() {
        // Both models should achieve comparable accuracy on a simple task.
        use crate::GradientBoostingRegressor;

        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];

        let mse = |preds: &Array1<f64>, y: &Array1<f64>| -> f64 {
            preds
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / y.len() as f64
        };

        // Standard GBM.
        let std_model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let std_fitted = std_model.fit(&x, &y).unwrap();
        let std_preds = std_fitted.predict(&x).unwrap();
        let std_mse = mse(&std_preds, &y);

        // Histogram GBM.
        let hist_model = HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(1)
            .with_max_leaf_nodes(None)
            .with_max_depth(Some(3))
            .with_random_state(42);
        let hist_fitted = hist_model.fit(&x, &y).unwrap();
        let hist_preds = hist_fitted.predict(&x).unwrap();
        let hist_mse = mse(&hist_preds, &y);

        // Both should have low MSE on this simple task.
        assert!(std_mse < 1.0, "Standard GBM MSE too high: {}", std_mse);
        assert!(hist_mse < 1.0, "Hist GBM MSE too high: {}", hist_mse);
    }
}
