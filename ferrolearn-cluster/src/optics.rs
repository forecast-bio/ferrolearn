//! OPTICS — Ordering Points To Identify the Clustering Structure.
//!
//! This module provides [`OPTICS`], a density-based algorithm that computes a
//! **reachability ordering** of the data.  Unlike DBSCAN, OPTICS does not
//! require a global density threshold; instead it produces a reachability plot
//! from which clusters at various density levels can be extracted.
//!
//! # Algorithm
//!
//! 1. For each unprocessed point `p`:
//!    - Compute its **core distance** — the distance to the `min_samples`-th
//!      nearest neighbour (within `max_eps`), or `∞` if there are fewer than
//!      `min_samples` neighbours.
//!    - If `p` is a core point, update the reachability distances of all
//!      unprocessed neighbours and add them to an ordered seed list.
//!    - Append `p` to the ordering with its final reachability distance.
//!
//! 2. **Cluster extraction** via the Xi method (see [`FittedOPTICS::extract_clusters`]):
//!    steep descents in the reachability plot define cluster boundaries.
//!
//! OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! produces a reachability ordering and reachability distances from which
//! cluster memberships can be derived post-hoc.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::OPTICS;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((9, 2), vec![
//!     0.0, 0.0,  0.1, 0.1,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//!     10.0, 0.0, 10.1, 0.0, 10.0, 0.1,
//! ]).unwrap();
//!
//! let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
//! assert_eq!(fitted.ordering().len(), 9);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: ordered float wrapper for the priority queue
// ─────────────────────────────────────────────────────────────────────────────

/// A `(reachability_distance, point_index)` pair that forms a min-heap entry.
#[derive(Clone, Copy, PartialEq)]
struct SeedEntry<F: Float> {
    reach_dist: F,
    idx: usize,
}

impl<F: Float> Eq for SeedEntry<F> {}

/// We want a **min**-heap, so we reverse the comparison.
impl<F: Float> Ord for SeedEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN is treated as greater than anything (put last).
        other
            .reach_dist
            .partial_cmp(&self.reach_dist)
            .unwrap_or(Ordering::Less)
    }
}

impl<F: Float> PartialOrd for SeedEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration struct
// ─────────────────────────────────────────────────────────────────────────────

/// OPTICS clustering configuration (unfitted).
///
/// Holds hyperparameters.  Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedOPTICS`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct OPTICS<F> {
    /// Minimum number of points required to form a core point (including
    /// the point itself).
    pub min_samples: usize,
    /// Maximum radius considered for neighbourhood queries.  Points beyond
    /// this distance are not considered neighbours.  Defaults to `F::infinity()`.
    pub max_eps: F,
    /// Xi steep-point threshold used by [`FittedOPTICS::extract_clusters`].
    /// A value in `(0, 1)`.  Defaults to `0.05`.
    pub xi: F,
    /// Minimum number of points required for a cluster to be kept.
    /// Clusters smaller than this are relabelled as noise (`-1`).
    /// Defaults to `None`, meaning use `min_samples`.
    pub min_cluster_size: Option<usize>,
}

impl<F: Float> OPTICS<F> {
    /// Create a new `OPTICS` with the given `min_samples`.
    ///
    /// Defaults: `max_eps = F::infinity()`, `xi = 0.05`, `min_cluster_size = None`.
    #[must_use]
    pub fn new(min_samples: usize) -> Self {
        Self {
            min_samples,
            max_eps: F::infinity(),
            xi: F::from(0.05).unwrap_or_else(|| F::from(5e-2).unwrap()),
            min_cluster_size: None,
        }
    }

    /// Set the maximum neighbourhood radius.
    #[must_use]
    pub fn with_max_eps(mut self, max_eps: F) -> Self {
        self.max_eps = max_eps;
        self
    }

    /// Set the Xi steep-point threshold.
    ///
    /// Must be in `(0, 1)`.
    #[must_use]
    pub fn with_xi(mut self, xi: F) -> Self {
        self.xi = xi;
        self
    }

    /// Set the minimum cluster size.
    ///
    /// Clusters with fewer than `min_cluster_size` points are relabelled as
    /// noise (`-1`).  When `None` (the default), `min_samples` is used as
    /// the minimum cluster size, matching scikit-learn's default behaviour.
    #[must_use]
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = Some(size);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitted struct
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted OPTICS model.
///
/// Stores the reachability ordering, reachability distances, core distances,
/// predecessor tracking, and cluster labels (extracted via the Xi method).
///
/// OPTICS does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedOPTICS<F> {
    /// Indices of data points in the reachability order.
    ordering_: Vec<usize>,
    /// Reachability distance for each data point (indexed by original point
    /// index, not by ordering position).  The first point processed always
    /// has reachability distance `∞`.
    reachability_: Array1<F>,
    /// Core distance for each point (indexed by original point index).
    /// Equals `∞` for non-core points.
    core_distances_: Array1<F>,
    /// Cluster label for each training sample (0-indexed for clusters; `-1`
    /// for noise).  Extracted using the Xi method.
    labels_: Array1<isize>,
    /// Predecessor for each point in the OPTICS ordering.
    /// `predecessors_[i] = Some(j)` means point `i` was reached from point `j`.
    /// The first point in each connected component has `None`.
    predecessors_: Vec<Option<usize>>,
    /// The `min_samples` value used during fitting (needed for Xi extraction).
    min_samples_: usize,
}

impl<F: Float> FittedOPTICS<F> {
    /// Return the reachability ordering (indices into the training data).
    #[must_use]
    pub fn ordering(&self) -> &[usize] {
        &self.ordering_
    }

    /// Return the reachability distances, indexed by original point index.
    #[must_use]
    pub fn reachability(&self) -> &Array1<F> {
        &self.reachability_
    }

    /// Return the core distances, indexed by original point index.
    #[must_use]
    pub fn core_distances(&self) -> &Array1<F> {
        &self.core_distances_
    }

    /// Return the cluster labels (Xi-method extraction).
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the predecessors for each point.
    ///
    /// `predecessors()[i] = Some(j)` means point `i` was reached from point `j`
    /// during the OPTICS ordering phase. The first point in each connected
    /// component has `None`.
    #[must_use]
    pub fn predecessors(&self) -> &[Option<usize>] {
        &self.predecessors_
    }

    /// Return the number of clusters found (excluding noise).
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels_.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }

    /// Extract flat clusters from the reachability plot using the Xi method.
    ///
    /// The Xi method identifies *steep up* and *steep down* areas in the
    /// reachability plot.  Clusters are formed between matching steep-down /
    /// steep-up pairs.
    ///
    /// `xi` must be in `(0, 1)`.  Returns a vector of cluster labels
    /// (length == `n_samples`); noise has label `-1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `xi` is outside `(0, 1)`.
    pub fn extract_clusters(&self, xi: F) -> Result<Array1<isize>, FerroError> {
        if xi <= F::zero() || xi >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        Ok(xi_cluster_extraction(
            &self.ordering_,
            &self.reachability_,
            &self.predecessors_,
            xi,
            self.min_samples_,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two slices.
#[inline]
fn euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
        .sqrt()
}

/// Return all neighbours of `idx` within distance `max_eps` (sorted by distance).
///
/// Returns `(neighbor_indices, distances)` in ascending distance order.
fn get_neighbors<F: Float>(x: &Array2<F>, idx: usize, max_eps: F) -> (Vec<usize>, Vec<F>) {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);
    let mut pairs: Vec<(F, usize)> = (0..x.nrows())
        .filter_map(|j| {
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps && j != idx {
                Some((d, j))
            } else {
                None
            }
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let indices = pairs.iter().map(|p| p.1).collect();
    let dists = pairs.iter().map(|p| p.0).collect();
    (indices, dists)
}

/// Compute core distance of `idx`: distance to the `min_samples`-th nearest
/// neighbour within `max_eps`.  Returns `F::infinity()` if fewer than
/// `min_samples` neighbours exist.
fn core_distance<F: Float>(x: &Array2<F>, idx: usize, max_eps: F, min_samples: usize) -> F {
    let row = x.row(idx);
    let rs = row.as_slice().unwrap_or(&[]);

    let mut dists: Vec<F> = (0..x.nrows())
        .filter_map(|j| {
            if j == idx {
                return None;
            }
            let other = x.row(j);
            let os = other.as_slice().unwrap_or(&[]);
            let d = euclidean(rs, os);
            if d <= max_eps { Some(d) } else { None }
        })
        .collect();

    if dists.len() < min_samples.saturating_sub(1) {
        // Not enough neighbours (need min_samples - 1 others).
        return F::infinity();
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    // The core distance is the distance to the (min_samples-1)-th other point
    // (0-indexed), i.e., the min_samples-th point overall including self.
    let k = min_samples.saturating_sub(1);
    if k == 0 {
        F::zero()
    } else if k <= dists.len() {
        dists[k - 1]
    } else {
        F::infinity()
    }
}

/// Update the seed list with neighbours that have improved reachability,
/// and track predecessors.
///
/// For each unprocessed neighbour `q`, the new reachability distance is
/// `max(core_dist_p, dist(p, q))`.  If this improves the current value, the
/// seed list is updated and `current_point` is recorded as the predecessor
/// of `q`.
#[allow(clippy::too_many_arguments)]
fn update_seeds<F: Float>(
    core_dist_p: F,
    current_point: usize,
    neighbors: &[usize],
    neighbor_dists: &[F],
    processed: &[bool],
    reachability: &mut Array1<F>,
    predecessors: &mut [Option<usize>],
    seeds: &mut BinaryHeap<SeedEntry<F>>,
) {
    for (i, &q) in neighbors.iter().enumerate() {
        if processed[q] {
            continue;
        }
        let new_reach = if core_dist_p > neighbor_dists[i] {
            core_dist_p
        } else {
            neighbor_dists[i]
        };
        if new_reach < reachability[q] {
            reachability[q] = new_reach;
            predecessors[q] = Some(current_point);
            seeds.push(SeedEntry {
                reach_dist: new_reach,
                idx: q,
            });
        }
    }
}

/// A steep-down area (SDA) tracked during Xi extraction.
#[derive(Debug, Clone)]
struct SteepDownArea {
    start: usize,
    end: usize,
    mib: f64,
}

/// Extend a steep region (downward or upward) maximally.
///
/// `steep_point[i]` is true if position `i` is steep in the primary direction.
/// `xward_point[i]` is true if the reachability is moving in the *opposite*
/// direction.  We allow up to `min_samples` consecutive non-steep positions
/// that are still going in the correct direction.
fn extend_region(
    steep_point: &[bool],
    xward_point: &[bool],
    start: usize,
    min_samples: usize,
) -> usize {
    let n = steep_point.len();
    let mut non_xward_points = 0usize;
    let mut index = start;
    let mut end = start;

    while index < n {
        if steep_point[index] {
            non_xward_points = 0;
            end = index;
        } else if !xward_point[index] {
            // Not steep, but still going in the right direction.
            non_xward_points += 1;
            if non_xward_points > min_samples {
                break;
            }
        } else {
            // Going the wrong direction — stop.
            return end;
        }
        index += 1;
    }
    end
}

/// Predecessor correction (Algorithm 2 of Schubert & Gertz 2018).
///
/// Returns `Some((c_start, c_end))` if the corrected cluster is valid,
/// or `None` if predecessor correction eliminates it.
fn correct_predecessor(
    r_plot: &[f64],
    pred_plot: &[Option<usize>],
    ordering: &[usize],
    mut s: usize,
    mut e: usize,
) -> Option<(usize, usize)> {
    while s < e {
        if r_plot[s] > r_plot[e] {
            return Some((s, e));
        }
        let p_e = pred_plot[ordering[e]];
        for i in s..e {
            if p_e == Some(ordering[i]) {
                return Some((s, e));
            }
        }
        e -= 1;
    }
    None
}

/// Xi-method cluster extraction from the reachability plot.
///
/// Implements Figure 19 of the OPTICS paper (Ankerst et al., 1999) with
/// corrections from Schubert & Gertz (2018).  This matches sklearn's
/// `cluster_optics_xi` implementation.
///
/// The algorithm:
/// 1. Builds steep-up/steep-down boolean arrays from the reachability plot.
/// 2. Iterates through steep points, extending them into *areas*.
/// 3. Maintains a stack of steep-down areas (SDAs) with max-in-between (MIB).
/// 4. When encountering a steep-up area, attempts to match it with each SDA.
/// 5. Applies boundary correction and predecessor correction per cluster.
fn xi_cluster_extraction<F: Float>(
    ordering: &[usize],
    reachability: &Array1<F>,
    predecessors: &[Option<usize>],
    xi: F,
    min_samples: usize,
) -> Array1<isize> {
    let n_ordered = ordering.len();
    let n_total = reachability.len();

    if n_ordered == 0 {
        return Array1::from_elem(n_total, -1isize);
    }

    // Build the reachability plot in ordering order (as f64 for simplicity).
    // Append +inf sentinel at the end (helps detect clusters at the tail).
    let mut r_plot: Vec<f64> = ordering
        .iter()
        .map(|&i| {
            let v = reachability[i];
            if v.is_finite() {
                v.to_f64().unwrap_or(f64::INFINITY)
            } else {
                f64::INFINITY
            }
        })
        .collect();
    r_plot.push(f64::INFINITY);

    // Predecessor plot in ordering order.
    let pred_plot: Vec<Option<usize>> = ordering.iter().map(|&i| predecessors[i]).collect();

    let xi_f64 = xi.to_f64().unwrap_or(0.05);
    let xi_complement = 1.0 - xi_f64;
    let min_samples = min_samples.max(1);

    // Compute steep_upward, steep_downward, upward, downward arrays.
    // ratio[i] = r_plot[i] / r_plot[i+1]
    let n_plot = r_plot.len() - 1; // last element is the sentinel
    let mut steep_upward = vec![false; n_plot];
    let mut steep_downward = vec![false; n_plot];
    let mut upward = vec![false; n_plot];
    let mut downward = vec![false; n_plot];

    for i in 0..n_plot {
        if r_plot[i + 1] == 0.0 {
            // Avoid division by zero; treat as downward.
            if r_plot[i] > 0.0 {
                steep_downward[i] = true;
                downward[i] = true;
            }
            continue;
        }
        let ratio = r_plot[i] / r_plot[i + 1];
        if ratio <= xi_complement {
            steep_upward[i] = true;
        }
        if ratio >= 1.0 / xi_complement {
            steep_downward[i] = true;
        }
        if ratio > 1.0 {
            downward[i] = true;
        }
        if ratio < 1.0 {
            upward[i] = true;
        }
    }

    // Main loop: Figure 19 of the OPTICS paper.
    let mut sdas: Vec<SteepDownArea> = Vec::new();
    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut index = 0usize;
    let mut mib = 0.0_f64;

    // Collect indices that are steep upward or steep downward.
    let steep_indices: Vec<usize> = (0..n_plot)
        .filter(|&i| steep_upward[i] || steep_downward[i])
        .collect();

    for &steep_index in &steep_indices {
        if steep_index < index {
            continue;
        }

        // Update MIB with the max reachability between the last processed
        // index and the current steep index.
        for k in index..=steep_index {
            if r_plot[k] > mib {
                mib = r_plot[k];
            }
        }

        if steep_downward[steep_index] {
            // --- Steep downward area ---
            // Filter existing SDAs whose start reachability * xi_complement < mib.
            sdas = update_filter_sdas(sdas, mib, xi_complement, &r_plot);

            let d_start = steep_index;
            let d_end = extend_region(&steep_downward, &upward, d_start, min_samples);
            sdas.push(SteepDownArea {
                start: d_start,
                end: d_end,
                mib: 0.0,
            });
            index = d_end + 1;
            if index < r_plot.len() {
                mib = r_plot[index];
            }
        } else {
            // --- Steep upward area ---
            sdas = update_filter_sdas(sdas, mib, xi_complement, &r_plot);

            let u_start = steep_index;
            let u_end = extend_region(&steep_upward, &downward, u_start, min_samples);
            index = u_end + 1;
            if index < r_plot.len() {
                mib = r_plot[index];
            }

            // Try to form clusters by matching this upward area with each SDA.
            let mut u_clusters: Vec<(usize, usize)> = Vec::new();
            for sda in &sdas {
                let mut c_start = sda.start;
                let c_end_initial = u_end;

                // Line (**), sc2*: skip if the point after the cluster end
                // times xi_complement is less than the SDA's MIB.
                let r_after = if c_end_initial + 1 < r_plot.len() {
                    r_plot[c_end_initial + 1]
                } else {
                    f64::INFINITY
                };
                if r_after * xi_complement < sda.mib {
                    continue;
                }

                // Definition 11, criterion 4: boundary correction.
                let d_max = r_plot[sda.start];
                let mut c_end = c_end_initial;

                if d_max * xi_complement >= r_after {
                    // Adjust start: find the first index from the left
                    // at a similar level as the end.
                    while c_start < sda.end
                        && c_start + 1 < r_plot.len()
                        && r_plot[c_start + 1] > r_after
                    {
                        c_start += 1;
                    }
                } else if r_after * xi_complement >= d_max {
                    // Adjust end: find the first index from the right
                    // at a similar level as the start.
                    while c_end > u_start && c_end > 0 && r_plot[c_end - 1] > d_max {
                        c_end -= 1;
                    }
                }

                // Predecessor correction.
                if let Some((cs, ce)) =
                    correct_predecessor(&r_plot, &pred_plot, ordering, c_start, c_end)
                {
                    c_start = cs;
                    c_end = ce;
                } else {
                    continue;
                }

                // Definition 11, criterion 3a: minimum size (checked later
                // by filter_small_clusters, but we can skip tiny ones here).
                if c_end < c_start + 1 {
                    continue;
                }

                // Definition 11, criterion 1: c_start must be within the SDA.
                if c_start > sda.end {
                    continue;
                }

                // Definition 11, criterion 2: c_end must be within the SUA.
                if c_end < u_start {
                    continue;
                }

                u_clusters.push((c_start, c_end));
            }

            // Add smaller clusters first (so larger encompassing clusters
            // come after when we process them).
            u_clusters.reverse();
            clusters.extend(u_clusters);
        }
    }

    // Convert cluster intervals to labels.
    // Clusters are ordered with smaller (leaf) clusters before larger
    // encompassing ones.  A cluster interval is assigned a label ONLY if
    // all positions in [c_start, c_end] are currently unassigned (-1).
    // This selects the leaf-level clusters from the hierarchy.
    //
    // Labels are initially in ordering-space; we remap to point-space at the
    // end.
    let mut ord_labels = vec![-1isize; n_ordered];
    let mut label = 0isize;
    for &(c_start, c_end) in &clusters {
        let end = c_end.min(n_ordered - 1);
        // Only assign if the entire interval is unassigned.
        let all_unassigned = (c_start..=end).all(|pos| ord_labels[pos] == -1);
        if all_unassigned {
            for pos in c_start..=end {
                ord_labels[pos] = label;
            }
            label += 1;
        }
    }

    // Remap from ordering-space labels to point-space labels.
    let mut labels = Array1::from_elem(n_total, -1isize);
    for (ord_pos, &pt) in ordering.iter().enumerate() {
        labels[pt] = ord_labels[ord_pos];
    }

    labels
}

/// Update and filter steep-down areas based on maximum-in-between (MIB).
///
/// Removes SDAs whose start reachability * xi_complement is less than the
/// current MIB.  Updates the MIB of surviving SDAs.
fn update_filter_sdas(
    sdas: Vec<SteepDownArea>,
    mib: f64,
    xi_complement: f64,
    r_plot: &[f64],
) -> Vec<SteepDownArea> {
    if mib.is_infinite() {
        return Vec::new();
    }
    let mut result: Vec<SteepDownArea> = sdas
        .into_iter()
        .filter(|sda| mib <= r_plot[sda.start] * xi_complement)
        .collect();
    for sda in &mut result {
        if mib > sda.mib {
            sda.mib = mib;
        }
    }
    result
}

/// Filter small clusters and renumber labels to be contiguous.
///
/// Clusters with fewer than `min_cluster_size` points are relabelled as
/// noise (`-1`). Remaining clusters are renumbered `0, 1, 2, ...`.
fn filter_small_clusters(labels: &mut Array1<isize>, min_cluster_size: usize) {
    // Count cluster sizes.
    let mut cluster_sizes: HashMap<isize, usize> = HashMap::new();
    for &l in labels.iter() {
        if l >= 0 {
            *cluster_sizes.entry(l).or_insert(0) += 1;
        }
    }

    // Relabel small clusters as noise.
    for label in labels.iter_mut() {
        if *label >= 0 {
            if let Some(&size) = cluster_sizes.get(label) {
                if size < min_cluster_size {
                    *label = -1;
                }
            }
        }
    }

    // Renumber clusters to be contiguous.
    let mut unique_labels: Vec<isize> = cluster_sizes
        .keys()
        .filter(|&&k| {
            cluster_sizes
                .get(&k)
                .is_some_and(|&sz| sz >= min_cluster_size)
        })
        .copied()
        .collect();
    unique_labels.sort_unstable();

    let mut remap: HashMap<isize, isize> = HashMap::new();
    for (new_id, &old_id) in unique_labels.iter().enumerate() {
        remap.insert(old_id, new_id as isize);
    }

    for label in labels.iter_mut() {
        if *label >= 0 {
            if let Some(&new_id) = remap.get(label) {
                *label = new_id;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for OPTICS<F> {
    type Fitted = FittedOPTICS<F>;
    type Error = FerroError;

    /// Fit the OPTICS model to the data.
    ///
    /// Computes the reachability ordering and distances for all training points.
    /// Cluster labels are extracted using the Xi method with the configured `xi`
    /// parameter.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `min_samples == 0`, `max_eps <= 0`,
    ///   or `xi` is outside `(0, 1)`.
    /// - [`FerroError::InsufficientSamples`] if the dataset is empty.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOPTICS<F>, FerroError> {
        let n_samples = x.nrows();

        // Validate parameters.
        if self.min_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.max_eps <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "max_eps".into(),
                reason: "must be positive".into(),
            });
        }
        if self.xi <= F::zero() || self.xi >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "xi".into(),
                reason: "must be in (0, 1)".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OPTICS requires at least 1 sample".into(),
            });
        }

        // Initialise state.
        let mut reachability = Array1::from_elem(n_samples, F::infinity());
        let mut core_distances = Array1::from_elem(n_samples, F::infinity());
        let mut processed = vec![false; n_samples];
        let mut ordering: Vec<usize> = Vec::with_capacity(n_samples);
        let mut predecessors: Vec<Option<usize>> = vec![None; n_samples];

        // Pre-compute core distances.
        for i in 0..n_samples {
            core_distances[i] = core_distance(x, i, self.max_eps, self.min_samples);
        }

        // Main OPTICS loop.
        for start in 0..n_samples {
            if processed[start] {
                continue;
            }

            processed[start] = true;
            ordering.push(start);

            if core_distances[start].is_infinite() {
                // Not a core point — just record it as noise candidate.
                continue;
            }

            // Priority queue (min-heap by reachability).
            let mut seeds: BinaryHeap<SeedEntry<F>> = BinaryHeap::new();

            let (nbrs, nbr_dists) = get_neighbors(x, start, self.max_eps);
            update_seeds(
                core_distances[start],
                start,
                &nbrs,
                &nbr_dists,
                &processed,
                &mut reachability,
                &mut predecessors,
                &mut seeds,
            );

            while let Some(entry) = seeds.pop() {
                let p = entry.idx;
                // Stale entry: a better reachability may have been inserted later.
                if processed[p] {
                    continue;
                }
                // Skip entries whose reachability is outdated.
                if entry.reach_dist > reachability[p] {
                    continue;
                }

                processed[p] = true;
                ordering.push(p);

                if core_distances[p].is_finite() {
                    let (p_nbrs, p_nbr_dists) = get_neighbors(x, p, self.max_eps);
                    update_seeds(
                        core_distances[p],
                        p,
                        &p_nbrs,
                        &p_nbr_dists,
                        &processed,
                        &mut reachability,
                        &mut predecessors,
                        &mut seeds,
                    );
                }
            }
        }

        // Extract cluster labels via the Xi method.
        let mut labels = xi_cluster_extraction(
            &ordering,
            &reachability,
            &predecessors,
            self.xi,
            self.min_samples,
        );

        // Apply min_cluster_size filtering.
        let min_size = self.min_cluster_size.unwrap_or(self.min_samples);
        filter_small_clusters(&mut labels, min_size);

        Ok(FittedOPTICS {
            ordering_: ordering,
            reachability_: reachability,
            core_distances_: core_distances,
            labels_: labels,
            predecessors_: predecessors,
            min_samples_: self.min_samples,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Three tight 2-D clusters.
    fn three_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 10.0, 0.0, 10.1, 0.0,
                10.0, 0.1,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_ordering_covers_all_points() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();

        let mut sorted = fitted.ordering().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_reachability_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.reachability().len(), 9);
    }

    #[test]
    fn test_core_distances_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.core_distances().len(), 9);
    }

    #[test]
    fn test_labels_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 9);
    }

    #[test]
    fn test_core_points_have_finite_core_distance() {
        let x = three_blobs();
        // With min_samples=2 all tight-cluster points should be core points.
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // Points 0-8 each have at least one close neighbour.
        for i in 0..9 {
            // Core distance is finite because each point has neighbours.
            assert!(
                fitted.core_distances()[i].is_finite(),
                "expected finite core distance for point {i}"
            );
        }
    }

    #[test]
    fn test_isolated_point_infinite_core_distance() {
        // Add an isolated point far from the clusters.
        let mut data = three_blobs().into_raw_vec_and_offset().0;
        data.extend_from_slice(&[100.0, 100.0]);
        let x = Array2::from_shape_vec((10, 2), data).unwrap();

        // With max_eps=2.0, the isolated point has no neighbours, so its core
        // distance must be infinite regardless of min_samples.
        let fitted = OPTICS::<f64>::new(3)
            .with_max_eps(2.0)
            .fit(&x, &())
            .unwrap();
        assert!(
            fitted.core_distances()[9].is_infinite(),
            "isolated point should have infinite core distance"
        );
    }

    #[test]
    fn test_reachability_first_point_infinite() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let first = fitted.ordering()[0];
        assert!(
            fitted.reachability()[first].is_infinite(),
            "first point in ordering should have infinite reachability"
        );
    }

    #[test]
    fn test_extract_clusters_valid_xi() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let labels = fitted.extract_clusters(0.05).unwrap();
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn test_extract_clusters_invalid_xi_zero() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(0.0).is_err());
    }

    #[test]
    fn test_extract_clusters_invalid_xi_one() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert!(fitted.extract_clusters(1.0).is_err());
    }

    #[test]
    fn test_invalid_min_samples_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_eps_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_eps_negative() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_max_eps(-1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_zero() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(0.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_xi_one() {
        let x = three_blobs();
        let result = OPTICS::<f64>::new(2).with_xi(1.0).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let result = OPTICS::<f64>::new(2).fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
        let fitted = OPTICS::<f64>::new(1).fit(&x, &()).unwrap();
        assert_eq!(fitted.ordering().len(), 1);
        assert_eq!(fitted.ordering()[0], 0);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();

        let fitted = OPTICS::<f32>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.ordering().len(), 6);
    }

    #[test]
    fn test_n_clusters_non_negative() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        // n_clusters() just counts distinct non-noise labels.
        let _ = fitted.n_clusters(); // Should not panic.
    }

    #[test]
    fn test_ordering_unique_indices() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let ordering = fitted.ordering();
        let mut seen = std::collections::HashSet::new();
        for &idx in ordering {
            assert!(seen.insert(idx), "duplicate index {idx} in ordering");
        }
    }

    #[test]
    fn test_with_max_eps_limits_reachability() {
        let x = three_blobs();
        let max_eps = 0.5;
        let fitted = OPTICS::<f64>::new(2)
            .with_max_eps(max_eps)
            .fit(&x, &())
            .unwrap();
        // All finite reachability values must be <= max_eps.
        for &r in fitted.reachability().iter() {
            if r.is_finite() {
                assert!(r <= max_eps + 1e-10);
            }
        }
    }

    #[test]
    fn test_predecessors_length() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        assert_eq!(fitted.predecessors().len(), 9);
    }

    #[test]
    fn test_first_point_has_no_predecessor() {
        let x = three_blobs();
        let fitted = OPTICS::<f64>::new(2).fit(&x, &()).unwrap();
        let first = fitted.ordering()[0];
        assert!(
            fitted.predecessors()[first].is_none(),
            "first point in ordering should have no predecessor"
        );
    }

    #[test]
    fn test_min_cluster_size_filters_small_clusters() {
        // Create data where one "cluster" is just 2 points and others are larger.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.05, 0.05, // cluster of 4
                10.0, 10.0, 10.05, 10.0, // cluster of 2
                20.0, 20.0, 20.05, 20.0, // cluster of 2
            ],
        )
        .unwrap();

        let fitted = OPTICS::<f64>::new(2)
            .with_min_cluster_size(3)
            .fit(&x, &())
            .unwrap();

        // The small clusters (size 2) should be filtered out as noise.
        for &l in fitted.labels().iter() {
            if l >= 0 {
                // Count how many points share this label.
                let count = fitted.labels().iter().filter(|&&c| c == l).count();
                assert!(
                    count >= 3,
                    "cluster with label {l} has only {count} points, expected >= 3"
                );
            }
        }
    }

    #[test]
    fn test_with_min_cluster_size_builder() {
        let optics = OPTICS::<f64>::new(5).with_min_cluster_size(10);
        assert_eq!(optics.min_cluster_size, Some(10));
    }
}
