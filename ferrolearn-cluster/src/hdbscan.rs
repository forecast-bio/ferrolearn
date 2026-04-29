//! HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
//!
//! This module provides [`Hdbscan`], a hierarchical extension of DBSCAN that
//! automatically discovers clusters of varying densities. Unlike DBSCAN, which
//! uses a single `eps` threshold, HDBSCAN builds a hierarchy of clusterings
//! and selects the most persistent clusters.
//!
//! # Algorithm
//!
//! 1. Compute the **core distance** for each point (distance to the
//!    `min_samples`-th nearest neighbor).
//! 2. Build the **mutual reachability graph**: for each pair `(i, j)`, the
//!    mutual reachability distance is
//!    `max(core_dist[i], core_dist[j], d(i, j))`.
//! 3. Compute the **minimum spanning tree** (MST) of this graph using Prim's
//!    algorithm.
//! 4. Sort MST edges by weight and build a **condensed cluster tree**.
//! 5. Extract flat clusters from the condensed tree by maximizing cluster
//!    stability.
//!
//! # Notes
//!
//! HDBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict) — it
//! only labels the training data. Use the fitted labels directly.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::Hdbscan;
//! use ferrolearn_core::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,  5.1, 5.1,
//! ]).unwrap();
//!
//! let model = Hdbscan::<f64>::new();
//! let fitted = model.with_min_cluster_size(3).fit(&x, &()).unwrap();
//! let labels = fitted.labels();
//! assert_eq!(labels.len(), 8);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// HDBSCAN clustering configuration (unfitted).
///
/// Holds hyperparameters for the HDBSCAN algorithm. Call [`Fit::fit`]
/// to run the algorithm and produce a [`FittedHdbscan`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct Hdbscan<F> {
    /// The minimum number of points required to form a cluster.
    pub min_cluster_size: usize,
    /// The number of neighbors used to compute core distances.
    /// Defaults to `min_cluster_size` if `None`.
    pub min_samples: Option<usize>,
    /// A distance threshold. Clusters below this epsilon are not split.
    /// Default is `0.0` (no epsilon constraint).
    pub cluster_selection_epsilon: F,
}

impl<F: Float> Hdbscan<F> {
    /// Create a new `Hdbscan` with default parameters.
    ///
    /// Defaults: `min_cluster_size = 5`, `min_samples = None`,
    /// `cluster_selection_epsilon = 0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
            cluster_selection_epsilon: F::zero(),
        }
    }

    /// Set the minimum cluster size.
    #[must_use]
    pub fn with_min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    /// Set the number of samples for core distance computation.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(min_samples);
        self
    }

    /// Set the cluster selection epsilon.
    #[must_use]
    pub fn with_cluster_selection_epsilon(mut self, eps: F) -> Self {
        self.cluster_selection_epsilon = eps;
        self
    }
}

impl<F: Float> Default for Hdbscan<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted HDBSCAN model.
///
/// Stores the cluster labels and cluster membership probabilities from the
/// training run. Noise points are labeled with `-1`.
///
/// HDBSCAN does **not** implement [`Predict`](ferrolearn_core::Predict).
#[derive(Debug, Clone)]
pub struct FittedHdbscan<F> {
    /// Cluster label for each training sample. Noise points have label -1.
    labels_: Array1<isize>,
    /// Cluster membership probabilities for each training sample.
    /// Values range from 0.0 (noise) to 1.0 (strong membership).
    probabilities_: Array1<F>,
}

impl<F: Float> FittedHdbscan<F> {
    /// Return the cluster labels for the training data.
    ///
    /// Noise points have label `-1`.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the cluster membership probabilities.
    ///
    /// Values range from `0.0` (noise) to `1.0` (strong cluster membership).
    #[must_use]
    pub fn probabilities(&self) -> &Array1<F> {
        &self.probabilities_
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared Euclidean distance between two slices.
#[inline]
fn sq_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Compute the core distance for each point: distance to the k-th nearest
/// neighbor (0-indexed, so we need the (k-1)-th element after sorting).
fn compute_core_distances<F: Float>(x: &Array2<F>, min_samples: usize) -> Vec<F> {
    let n = x.nrows();
    let mut core_dists = vec![F::zero(); n];

    for (i, cd) in core_dists.iter_mut().enumerate() {
        let row_i = x.row(i);
        let si = row_i.as_slice().unwrap_or(&[]);
        let mut dists: Vec<F> = (0..n)
            .map(|j| {
                if i == j {
                    F::zero()
                } else {
                    sq_euclidean(si, x.row(j).as_slice().unwrap_or(&[])).sqrt()
                }
            })
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // k-th nearest neighbor (index min_samples since index 0 is self with dist 0)
        let k = min_samples.min(n - 1);
        *cd = dists[k];
    }
    core_dists
}

/// An MST edge.
#[derive(Debug, Clone, Copy)]
struct MstEdge<F> {
    /// Source node.
    u: usize,
    /// Destination node.
    v: usize,
    /// Weight (mutual reachability distance).
    weight: F,
}

/// Build the minimum spanning tree of the mutual reachability graph using
/// Prim's algorithm.
fn build_mst<F: Float>(x: &Array2<F>, core_dists: &[F]) -> Vec<MstEdge<F>> {
    let n = x.nrows();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_weight = vec![F::max_value(); n];
    let mut min_source = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    // Start from node 0.
    in_tree[0] = true;
    for j in 1..n {
        let d = mutual_reachability(x, core_dists, 0, j);
        min_weight[j] = d;
        min_source[j] = 0;
    }

    for _ in 0..(n - 1) {
        // Find the node not yet in the tree with the minimum edge weight.
        let mut best_node = 0;
        let mut best_weight = F::max_value();
        for j in 0..n {
            if !in_tree[j] && min_weight[j] < best_weight {
                best_weight = min_weight[j];
                best_node = j;
            }
        }

        in_tree[best_node] = true;
        edges.push(MstEdge {
            u: min_source[best_node],
            v: best_node,
            weight: best_weight,
        });

        // Update min_weight for remaining nodes.
        for j in 0..n {
            if !in_tree[j] {
                let d = mutual_reachability(x, core_dists, best_node, j);
                if d < min_weight[j] {
                    min_weight[j] = d;
                    min_source[j] = best_node;
                }
            }
        }
    }

    edges
}

/// Compute the mutual reachability distance between points i and j.
#[inline]
fn mutual_reachability<F: Float>(x: &Array2<F>, core_dists: &[F], i: usize, j: usize) -> F {
    let d = sq_euclidean(
        x.row(i).as_slice().unwrap_or(&[]),
        x.row(j).as_slice().unwrap_or(&[]),
    )
    .sqrt();
    // max(core_dist[i], core_dist[j], d(i, j))
    let mut result = d;
    if core_dists[i] > result {
        result = core_dists[i];
    }
    if core_dists[j] > result {
        result = core_dists[j];
    }
    result
}

/// A node in the condensed tree.
#[derive(Debug, Clone)]
struct CondensedNode {
    /// Child cluster IDs (could be leaf points represented as negative IDs).
    children: Vec<usize>,
    /// Lambda (1/distance) at which each child fell out.
    child_lambdas: Vec<f64>,
    /// Lambda at which this cluster was born (parent split).
    birth_lambda: f64,
    /// Lambda at which this cluster dies (further split).
    death_lambda: f64,
    /// The stability of this cluster.
    stability: f64,
    /// Number of points in this cluster.
    size: usize,
}

/// Build the condensed cluster tree from sorted MST edges, and extract
/// clusters. Returns (labels, probabilities).
fn extract_clusters<F: Float>(
    n_samples: usize,
    mst_edges: &mut [MstEdge<F>],
    min_cluster_size: usize,
    cluster_selection_epsilon: F,
) -> (Array1<isize>, Array1<F>) {
    if n_samples == 0 {
        return (Array1::zeros(0), Array1::zeros(0));
    }

    // Sort MST edges by weight (ascending).
    mst_edges.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Union-Find for tracking component membership.
    let mut parent: Vec<usize> = (0..n_samples).collect();
    let mut size = vec![1usize; n_samples];

    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        // Path compression.
        let mut current = i;
        while parent[current] != root {
            let next = parent[current];
            parent[current] = root;
            current = next;
        }
        root
    }

    // Build the single-linkage dendrogram by processing edges in order.
    // We track a condensed tree: only splits where both children have
    // at least min_cluster_size points are considered real cluster splits.

    // For each point, track which condensed cluster it belongs to.
    let mut point_cluster = vec![0usize; n_samples]; // condensed cluster ID
    let mut condensed_clusters: Vec<CondensedNode> = Vec::new();

    // Start: all points are in one big cluster (root).
    condensed_clusters.push(CondensedNode {
        children: Vec::new(),
        child_lambdas: Vec::new(),
        birth_lambda: 0.0,
        death_lambda: 0.0,
        stability: 0.0,
        size: n_samples,
    });

    // Process edges from largest to smallest (we want to split, so reverse).
    // Actually, for building the hierarchy top-down, we process from largest
    // distance down. But the standard approach is:
    // 1. Build single-linkage tree bottom-up (process edges smallest to largest).
    // 2. Convert to condensed tree.
    // Let me do the standard approach: build the dendrogram bottom-up.

    // Dendrogram: each merge is (child_a, child_b, distance, new_size).
    let mut dendrogram: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n_samples - 1);
    let mut next_cluster_id = n_samples; // IDs >= n_samples are internal nodes.

    // Map from UF root to current dendrogram cluster ID.
    let mut uf_to_cluster: Vec<usize> = (0..n_samples).collect();

    for edge in mst_edges.iter() {
        let ru = find(&mut parent, edge.u);
        let rv = find(&mut parent, edge.v);
        if ru == rv {
            continue;
        }

        let su = size[ru];
        let sv = size[rv];
        let cu = uf_to_cluster[ru];
        let cv = uf_to_cluster[rv];
        let dist = edge.weight.to_f64().unwrap_or(0.0);

        dendrogram.push((cu, cv, dist, su + sv));

        // Merge: smaller into larger.
        if su < sv {
            parent[ru] = rv;
            size[rv] += su;
            uf_to_cluster[rv] = next_cluster_id;
        } else {
            parent[rv] = ru;
            size[ru] += sv;
            uf_to_cluster[ru] = next_cluster_id;
        }
        next_cluster_id += 1;
    }

    // Now build the condensed tree from the dendrogram.
    // Total nodes: n_samples (leaves) + dendrogram.len() (internal).
    let total_nodes = n_samples + dendrogram.len();

    // For each internal node, record its children and merge distance.
    let mut node_children: Vec<(usize, usize)> = vec![(0, 0); total_nodes];
    let mut node_distance: Vec<f64> = vec![0.0; total_nodes];
    let mut node_size: Vec<usize> = vec![1; total_nodes];

    for (i, &(left, right, dist, sz)) in dendrogram.iter().enumerate() {
        let node_id = n_samples + i;
        node_children[node_id] = (left, right);
        node_distance[node_id] = dist;
        node_size[node_id] = sz;
    }

    // Traverse the dendrogram top-down to build the condensed tree.
    // The root is the last internal node.
    let root_node = if dendrogram.is_empty() {
        0
    } else {
        n_samples + dendrogram.len() - 1
    };

    // Condensed tree: track which cluster each point ends up in.
    // Start fresh.
    condensed_clusters.clear();
    let mut cond_cluster_counter: usize = 0;

    // Map from dendrogram node to condensed cluster.
    let mut node_to_cond: Vec<Option<usize>> = vec![None; total_nodes];

    // BFS/DFS from root.
    let root_cond = cond_cluster_counter;
    condensed_clusters.push(CondensedNode {
        children: Vec::new(),
        child_lambdas: Vec::new(),
        birth_lambda: 0.0,
        death_lambda: 0.0,
        stability: 0.0,
        size: n_samples,
    });
    cond_cluster_counter += 1;
    node_to_cond[root_node] = Some(root_cond);

    // Stack for DFS: (dendrogram_node_id, parent_condensed_cluster_id).
    let mut stack: Vec<(usize, usize)> = vec![(root_node, root_cond)];

    let eps_f64 = cluster_selection_epsilon.to_f64().unwrap_or(0.0);

    while let Some((node_id, parent_cond)) = stack.pop() {
        if node_id < n_samples {
            // This is a leaf (individual point). Add it to the parent condensed cluster.
            condensed_clusters[parent_cond].children.push(node_id);
            let lambda = if condensed_clusters[parent_cond].death_lambda > 0.0 {
                condensed_clusters[parent_cond].death_lambda
            } else {
                condensed_clusters[parent_cond].birth_lambda
            };
            condensed_clusters[parent_cond].child_lambdas.push(lambda);
            point_cluster[node_id] = parent_cond;
            continue;
        }

        let (left, right) = node_children[node_id];
        let left_size = node_size.get(left).copied().unwrap_or(1);
        let right_size = node_size.get(right).copied().unwrap_or(1);
        let split_dist = node_distance[node_id];
        let lambda = if split_dist > 0.0 {
            1.0 / split_dist
        } else {
            f64::MAX
        };

        let both_large = left_size >= min_cluster_size && right_size >= min_cluster_size;
        let above_epsilon = split_dist > eps_f64;

        if both_large && above_epsilon {
            // True split: create two new condensed clusters.
            condensed_clusters[parent_cond].death_lambda = lambda;

            let left_cond = cond_cluster_counter;
            condensed_clusters.push(CondensedNode {
                children: Vec::new(),
                child_lambdas: Vec::new(),
                birth_lambda: lambda,
                death_lambda: 0.0,
                stability: 0.0,
                size: left_size,
            });
            cond_cluster_counter += 1;

            let right_cond = cond_cluster_counter;
            condensed_clusters.push(CondensedNode {
                children: Vec::new(),
                child_lambdas: Vec::new(),
                birth_lambda: lambda,
                death_lambda: 0.0,
                stability: 0.0,
                size: right_size,
            });
            cond_cluster_counter += 1;

            condensed_clusters[parent_cond]
                .children
                .push(left_cond + n_samples);
            condensed_clusters[parent_cond].child_lambdas.push(lambda);
            condensed_clusters[parent_cond]
                .children
                .push(right_cond + n_samples);
            condensed_clusters[parent_cond].child_lambdas.push(lambda);

            node_to_cond[left] = Some(left_cond);
            node_to_cond[right] = Some(right_cond);

            stack.push((left, left_cond));
            stack.push((right, right_cond));
        } else {
            // Not a true split. One or both sides are too small, or below epsilon.
            // Points from the small side become noise (fall out of parent).
            // Points from the large side continue in the parent cluster.

            if left_size >= min_cluster_size && !above_epsilon {
                // Both above min_cluster_size but below epsilon — keep in parent.
                stack.push((left, parent_cond));
                stack.push((right, parent_cond));
            } else if left_size < min_cluster_size && right_size < min_cluster_size {
                // Both sides too small — all points fall out.
                // Add all leaves to parent as fallen-out.
                collect_leaves(left, n_samples, &node_children, &mut |leaf| {
                    condensed_clusters[parent_cond].children.push(leaf);
                    condensed_clusters[parent_cond].child_lambdas.push(lambda);
                    point_cluster[leaf] = parent_cond;
                });
                collect_leaves(right, n_samples, &node_children, &mut |leaf| {
                    condensed_clusters[parent_cond].children.push(leaf);
                    condensed_clusters[parent_cond].child_lambdas.push(lambda);
                    point_cluster[leaf] = parent_cond;
                });
                condensed_clusters[parent_cond].death_lambda = lambda;
            } else {
                // One side is large, the other is small.
                let (large, small) = if left_size >= min_cluster_size {
                    (left, right)
                } else {
                    (right, left)
                };

                // Small side: points fall out of parent cluster.
                collect_leaves(small, n_samples, &node_children, &mut |leaf| {
                    condensed_clusters[parent_cond].children.push(leaf);
                    condensed_clusters[parent_cond].child_lambdas.push(lambda);
                    point_cluster[leaf] = parent_cond;
                });

                // Large side continues in parent.
                stack.push((large, parent_cond));
            }
        }
    }

    // Compute stability for each condensed cluster.
    // Stability = sum over points p in cluster: (lambda_p - birth_lambda)
    for cluster in &mut condensed_clusters {
        let birth = cluster.birth_lambda;
        let mut stab = 0.0;
        for &child_lambda in &cluster.child_lambdas {
            stab += child_lambda - birth;
        }
        cluster.stability = if stab > 0.0 { stab } else { 0.0 };
    }

    // Select clusters: bottom-up, each cluster is selected if its own stability
    // exceeds the sum of its children's stabilities.
    let n_cond = condensed_clusters.len();
    let mut selected = vec![true; n_cond];
    let mut total_stability = vec![0.0f64; n_cond];

    // Initialize: leaf condensed clusters have their own stability.
    for i in 0..n_cond {
        total_stability[i] = condensed_clusters[i].stability;
    }

    // Process bottom-up. Find parent-child relationships among condensed clusters.
    // A condensed cluster's children (that are clusters, not points) are those
    // that have child IDs >= n_samples in the children vector.
    for i in (0..n_cond).rev() {
        let mut child_sum = 0.0;
        let mut has_child_clusters = false;
        for &child_id in &condensed_clusters[i].children {
            if child_id >= n_samples {
                let cond_child = child_id - n_samples;
                if cond_child < n_cond {
                    child_sum += total_stability[cond_child];
                    has_child_clusters = true;
                }
            }
        }

        if has_child_clusters {
            if condensed_clusters[i].stability >= child_sum {
                // Select this cluster, deselect all descendants.
                total_stability[i] = condensed_clusters[i].stability;
                // Mark all descendant clusters as not selected.
                deselect_descendants(i, n_samples, &condensed_clusters, &mut selected);
            } else {
                // Children are better; don't select this cluster.
                selected[i] = false;
                total_stability[i] = child_sum;
            }
        }
        // Leaf condensed clusters (no child clusters) remain selected.
    }

    // Root cluster is never selected as a "real" cluster (it contains everything).
    if n_cond > 0 {
        selected[0] = false;
    }

    // Assign labels: map condensed cluster IDs to label IDs.
    let mut cluster_label_map: Vec<isize> = vec![-1; n_cond];
    let mut label_counter: isize = 0;
    for (i, &sel) in selected.iter().enumerate() {
        if sel && condensed_clusters[i].size >= min_cluster_size {
            cluster_label_map[i] = label_counter;
            label_counter += 1;
        }
    }

    // Assign each point a label based on which selected cluster it belongs to.
    let mut labels = Array1::from_elem(n_samples, -1isize);
    let mut probabilities = Array1::from_elem(n_samples, F::zero());

    for (pt, &cond_id) in point_cluster.iter().enumerate() {
        // Walk up from cond_id to find the nearest selected ancestor.
        let label =
            find_selected_cluster(cond_id, &cluster_label_map, &condensed_clusters, n_samples);
        labels[pt] = label;
    }

    // Compute probabilities based on the lambda at which each point falls out
    // vs. the cluster's birth and death lambdas.
    for cond_id in 0..n_cond {
        if cluster_label_map[cond_id] < 0 {
            continue;
        }
        let birth = condensed_clusters[cond_id].birth_lambda;
        let death = condensed_clusters[cond_id].death_lambda;
        let range = if death > birth { death - birth } else { 1.0 };

        for (idx, &child) in condensed_clusters[cond_id].children.iter().enumerate() {
            if child < n_samples {
                let child_lambda = condensed_clusters[cond_id].child_lambdas[idx];
                let prob = if range > 0.0 {
                    ((child_lambda - birth) / range).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if labels[child] == cluster_label_map[cond_id] {
                    probabilities[child] = F::from(prob).unwrap_or_else(F::zero);
                }
            }
        }
    }

    // Points that were assigned to a cluster but didn't get a probability
    // from the loop above (because they're in a descendant cluster that was
    // absorbed) get probability 1.0.
    for i in 0..n_samples {
        if labels[i] >= 0 && probabilities[i] == F::zero() {
            probabilities[i] = F::one();
        }
    }

    (labels, probabilities)
}

/// Collect all leaf (point) indices under a dendrogram node.
fn collect_leaves(
    node_id: usize,
    n_samples: usize,
    node_children: &[(usize, usize)],
    callback: &mut dyn FnMut(usize),
) {
    if node_id < n_samples {
        callback(node_id);
        return;
    }
    let (left, right) = node_children[node_id];
    collect_leaves(left, n_samples, node_children, callback);
    collect_leaves(right, n_samples, node_children, callback);
}

/// Mark all descendant condensed clusters as not selected.
fn deselect_descendants(
    cond_id: usize,
    n_samples: usize,
    condensed_clusters: &[CondensedNode],
    selected: &mut [bool],
) {
    for &child_id in &condensed_clusters[cond_id].children {
        if child_id >= n_samples {
            let child_cond = child_id - n_samples;
            if child_cond < condensed_clusters.len() {
                selected[child_cond] = false;
                deselect_descendants(child_cond, n_samples, condensed_clusters, selected);
            }
        }
    }
}

/// Walk up from a condensed cluster to find the nearest selected ancestor.
fn find_selected_cluster(
    cond_id: usize,
    cluster_label_map: &[isize],
    condensed_clusters: &[CondensedNode],
    n_samples: usize,
) -> isize {
    // Check if this cluster is selected.
    if cluster_label_map[cond_id] >= 0 {
        return cluster_label_map[cond_id];
    }

    // Check ancestors. We need to find which condensed cluster contains
    // cond_id as a child. Build a parent map.
    // Since this is called for each point, we do a simple search.
    for (i, cluster) in condensed_clusters.iter().enumerate() {
        for &child_id in &cluster.children {
            if child_id >= n_samples && child_id - n_samples == cond_id {
                return find_selected_cluster(i, cluster_label_map, condensed_clusters, n_samples);
            }
        }
    }

    -1 // Noise.
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impl: Fit
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for Hdbscan<F> {
    type Fitted = FittedHdbscan<F>;
    type Error = FerroError;

    /// Fit the HDBSCAN model to the data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `min_cluster_size` is less
    /// than 2.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedHdbscan<F>, FerroError> {
        if self.min_cluster_size < 2 {
            return Err(FerroError::InvalidParameter {
                name: "min_cluster_size".into(),
                reason: "must be at least 2".into(),
            });
        }

        let min_samples = self.min_samples.unwrap_or(self.min_cluster_size);
        if min_samples == 0 {
            return Err(FerroError::InvalidParameter {
                name: "min_samples".into(),
                reason: "must be at least 1".into(),
            });
        }

        if self.cluster_selection_epsilon < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "cluster_selection_epsilon".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedHdbscan {
                labels_: Array1::zeros(0),
                probabilities_: Array1::zeros(0),
            });
        }

        if n_samples < self.min_cluster_size {
            // Not enough points to form any cluster: all noise.
            return Ok(FittedHdbscan {
                labels_: Array1::from_elem(n_samples, -1isize),
                probabilities_: Array1::from_elem(n_samples, F::zero()),
            });
        }

        // Step 1: Compute core distances.
        let core_dists = compute_core_distances(x, min_samples);

        // Step 2-3: Build MST of mutual reachability graph.
        let mut mst_edges = build_mst(x, &core_dists);

        // Step 4-5: Build condensed tree and extract clusters.
        let (labels, probabilities) = extract_clusters(
            n_samples,
            &mut mst_edges,
            self.min_cluster_size,
            self.cluster_selection_epsilon,
        );

        Ok(FittedHdbscan {
            labels_: labels,
            probabilities_: probabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two well-separated blobs.
    fn make_two_blobs() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                // Cluster A near (0, 0)
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, -0.05, 0.05,
                // Cluster B near (10, 10)
                10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 10.05, 10.05, 9.95, 10.05,
            ],
        )
        .unwrap()
    }

    /// Make a dense cluster dataset for HDBSCAN testing.
    fn make_dense_clusters() -> Array2<f64> {
        // Two dense clusters, well-separated.
        Array2::from_shape_vec(
            (20, 2),
            vec![
                // Cluster A: 10 points tightly packed near (0, 0)
                0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.05, 0.05, -0.05, 0.0, 0.0, -0.05, -0.05, -0.05,
                0.03, 0.02, -0.02, 0.03, 0.04, -0.01,
                // Cluster B: 10 points tightly packed near (5, 5)
                5.0, 5.0, 5.05, 5.0, 5.0, 5.05, 5.05, 5.05, 4.95, 5.0, 5.0, 4.95, 4.95, 4.95, 5.03,
                5.02, 4.98, 5.03, 5.04, 4.99,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_two_clusters() {
        let x = make_two_blobs();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(3);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 12);

        // First 6 should be in the same cluster.
        let first_label = labels[0];
        assert!(
            first_label >= 0,
            "expected cluster, got noise for first blob"
        );
        for i in 0..6 {
            assert_eq!(
                labels[i], first_label,
                "point {i} should be in the same cluster as point 0"
            );
        }

        // Last 6 should be in the same cluster.
        let second_label = labels[6];
        assert!(
            second_label >= 0,
            "expected cluster, got noise for second blob"
        );
        for i in 6..12 {
            assert_eq!(
                labels[i], second_label,
                "point {i} should be in same cluster as point 6"
            );
        }

        // The two clusters should have different labels.
        assert_ne!(first_label, second_label);
        assert_eq!(fitted.n_clusters(), 2);
    }

    #[test]
    fn test_noise_detection() {
        // Two tight blobs + outliers.
        let x = Array2::from_shape_vec(
            (14, 2),
            vec![
                // Cluster A
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, // Cluster B
                10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 10.05, 10.05, // Outliers
                50.0, 50.0, -50.0, -50.0, 100.0, 0.0, 0.0, 100.0,
            ],
        )
        .unwrap();

        let model = Hdbscan::<f64>::new().with_min_cluster_size(3);
        let fitted = model.fit(&x, &()).unwrap();

        let labels = fitted.labels();
        // Outliers should be noise.
        assert_eq!(labels[10], -1, "outlier at (50,50) should be noise");
        assert_eq!(labels[11], -1, "outlier at (-50,-50) should be noise");
        assert_eq!(labels[12], -1, "outlier at (100,0) should be noise");
        assert_eq!(labels[13], -1, "outlier at (0,100) should be noise");

        // Cluster points should not be noise.
        assert!(labels[0] >= 0, "cluster A point should not be noise");
        assert!(labels[5] >= 0, "cluster B point should not be noise");
    }

    #[test]
    fn test_min_cluster_size_effect() {
        let x = make_two_blobs();

        // With small min_cluster_size, should find clusters.
        let model_small = Hdbscan::<f64>::new().with_min_cluster_size(2);
        let fitted_small = model_small.fit(&x, &()).unwrap();
        assert!(
            fitted_small.n_clusters() >= 1,
            "should find at least 1 cluster"
        );

        // With very large min_cluster_size, everything becomes noise.
        let model_large = Hdbscan::<f64>::new().with_min_cluster_size(100);
        let fitted_large = model_large.fit(&x, &()).unwrap();
        for &label in fitted_large.labels() {
            assert_eq!(
                label, -1,
                "all points should be noise with large min_cluster_size"
            );
        }
    }

    #[test]
    fn test_probabilities_range() {
        let x = make_two_blobs();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(3);
        let fitted = model.fit(&x, &()).unwrap();

        for (i, &prob) in fitted.probabilities().iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&prob),
                "probability at index {i} is {prob}, expected [0, 1]"
            );
        }

        // Noise points should have probability 0.
        for i in 0..x.nrows() {
            if fitted.labels()[i] == -1 {
                assert_relative_eq!(fitted.probabilities()[i], 0.0);
            }
        }
    }

    #[test]
    fn test_dense_clusters() {
        let x = make_dense_clusters();
        let model = Hdbscan::<f64>::new()
            .with_min_cluster_size(3)
            .with_min_samples(3);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 20);
        // Should find 2 clusters in the dense, well-separated data.
        assert_eq!(
            fitted.n_clusters(),
            2,
            "should find 2 clusters in well-separated dense data"
        );

        // Points 0-9 should be in one cluster, 10-19 in another.
        let first_label = fitted.labels()[0];
        assert!(first_label >= 0, "first cluster points should not be noise");
        for i in 0..10 {
            assert_eq!(
                fitted.labels()[i],
                first_label,
                "point {i} should be in cluster A"
            );
        }
        let second_label = fitted.labels()[10];
        assert!(
            second_label >= 0,
            "second cluster points should not be noise"
        );
        for i in 10..20 {
            assert_eq!(
                fitted.labels()[i],
                second_label,
                "point {i} should be in cluster B"
            );
        }
        assert_ne!(
            first_label, second_label,
            "two clusters should have different labels"
        );
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = Hdbscan::<f64>::new();
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels().len(), 0);
        assert_eq!(fitted.probabilities().len(), 0);
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_single_point() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(2);
        let fitted = model.fit(&x, &()).unwrap();

        assert_eq!(fitted.labels()[0], -1);
        assert_eq!(fitted.n_clusters(), 0);
    }

    #[test]
    fn test_too_few_for_cluster() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1]).unwrap();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(5);
        let fitted = model.fit(&x, &()).unwrap();

        // With min_cluster_size=5 and only 3 points, all should be noise.
        for &label in fitted.labels() {
            assert_eq!(label, -1);
        }
    }

    #[test]
    fn test_invalid_min_cluster_size() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(1);
        let result = model.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_cluster_selection_epsilon() {
        let x = make_two_blobs();

        // With very large epsilon, no splits should happen (everything is one blob or noise).
        let model = Hdbscan::<f64>::new()
            .with_min_cluster_size(3)
            .with_cluster_selection_epsilon(1000.0);
        let fitted = model.fit(&x, &()).unwrap();

        // Should either be all one cluster or all noise (depending on tree structure).
        let n = fitted.n_clusters();
        assert!(
            n <= 1,
            "with large epsilon, should have at most 1 cluster, got {n}"
        );
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 10.0, 10.0, 10.1, 10.0,
                10.0, 10.1, 10.1, 10.1, 10.05, 10.05,
            ],
        )
        .unwrap();

        let model = Hdbscan::<f32>::new().with_min_cluster_size(3);
        let fitted = model.fit(&x, &()).unwrap();
        assert_eq!(fitted.labels().len(), 10);
    }

    #[test]
    fn test_identical_points() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let model = Hdbscan::<f64>::new().with_min_cluster_size(2);
        let fitted = model.fit(&x, &()).unwrap();

        // All identical points should be in the same cluster or all noise.
        let labels = fitted.labels();
        let first = labels[0];
        for &l in labels {
            assert_eq!(l, first, "identical points should have the same label");
        }
    }

    #[test]
    fn test_n_clusters_accessor() {
        let x = make_two_blobs();
        let model = Hdbscan::<f64>::new().with_min_cluster_size(3);
        let fitted = model.fit(&x, &()).unwrap();
        let n = fitted.n_clusters();
        assert!(n > 0, "should find clusters");
    }

    #[test]
    fn test_default_constructor() {
        let model = Hdbscan::<f64>::default();
        assert_eq!(model.min_cluster_size, 5);
        assert!(model.min_samples.is_none());
    }
}
