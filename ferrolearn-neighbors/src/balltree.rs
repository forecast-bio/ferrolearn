//! Ball tree for efficient nearest neighbor search in moderate-to-high dimensions.
//!
//! Unlike KD-Trees which partition along axis-aligned hyperplanes, ball trees
//! partition data into nested hyperspheres. This degrades more gracefully with
//! dimensionality, making ball trees effective for d > 15 where KD-Trees become
//! equivalent to brute force.
//!
//! # Implementation
//!
//! Nodes are stored in a flat `Vec` for cache-friendly traversal. All internal
//! computations use squared Euclidean distance to avoid unnecessary `sqrt` calls.
//! The k-NN search uses a `BinaryHeap` (max-heap) for O(log k) insertion.
//!
//! # Complexity
//!
//! - Build: O(n log n)
//! - Query: O(n^(1-1/d) log n) amortized, much better than O(n) brute force
//!   for moderate dimensions
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::balltree::BallTree;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).unwrap();
//!
//! let tree = BallTree::build(&data);
//! let neighbors = tree.query(&data, &[0.1, 0.1], 2);
//! assert_eq!(neighbors[0].0, 0); // closest is (0,0)
//! ```

use ndarray::Array2;
use num_traits::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Default leaf size — the maximum number of points in a leaf node.
const DEFAULT_LEAF_SIZE: usize = 40;

/// A node in the flat ball tree.
#[derive(Debug)]
struct Node {
    /// Centroid of the points in this node (true centroid, not nearest point).
    centroid: Vec<f64>,
    /// Squared bounding radius from centroid to farthest point.
    radius_sq: f64,
    /// Start index (inclusive) into the permuted index array.
    start: usize,
    /// End index (exclusive) into the permuted index array.
    end: usize,
    /// Node type: leaf or branch with child indices.
    kind: NodeKind,
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    Leaf,
    Branch { left: usize, right: usize },
}

/// A ball tree spatial index for nearest neighbor queries.
///
/// Stores data as flattened f64 for cache-friendly access. Nodes are stored
/// in a flat `Vec` with index-based child references.
#[derive(Debug)]
pub struct BallTree {
    /// Flat array of tree nodes.
    nodes: Vec<Node>,
    /// Flattened data: `data[i * n_features + j]` is feature j of point i.
    data: Vec<f64>,
    /// Number of features per point.
    n_features: usize,
    /// Permuted index array — maps internal positions to original dataset indices.
    indices: Vec<usize>,
    /// Leaf size used during construction (accessible for introspection).
    _leaf_size: usize,
}

/// Entry in the max-heap used during k-NN search.
/// The largest squared distance sits at the top and gets evicted first.
struct KnnCandidate {
    dist_sq: f64,
    orig_idx: usize,
}

impl PartialEq for KnnCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl Eq for KnnCandidate {}

impl PartialOrd for KnnCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KnnCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
    }
}

/// Squared Euclidean distance between two f64 slices.
#[inline]
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Squared lower-bound distance from a query point to the nearest possible
/// point inside a ball defined by `center` and `radius_sq`.
///
/// Returns zero when the query is inside the ball.
#[inline]
fn ball_lower_bound_sq(query: &[f64], center: &[f64], radius_sq: f64) -> f64 {
    let d2 = dist_sq(query, center);
    let d = d2.sqrt();
    let r = radius_sq.sqrt();
    let gap = d - r;
    if gap > 0.0 { gap * gap } else { 0.0 }
}

impl BallTree {
    /// Build a ball tree from a dataset with the default leaf size.
    pub fn build<F: Float + Send + Sync + 'static>(data: &Array2<F>) -> Self {
        Self::build_with_leaf_size(data, DEFAULT_LEAF_SIZE)
    }

    /// Build a ball tree with a custom leaf size.
    pub fn build_with_leaf_size<F: Float + Send + Sync + 'static>(
        data: &Array2<F>,
        leaf_size: usize,
    ) -> Self {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let leaf_size = leaf_size.max(1);

        if n_samples == 0 {
            return Self {
                nodes: Vec::new(),
                data: Vec::new(),
                n_features,
                indices: Vec::new(),
                _leaf_size: leaf_size,
            };
        }

        // Flatten data to f64 for cache-friendly access.
        let flat_data: Vec<f64> = (0..n_samples)
            .flat_map(|i| (0..n_features).map(move |j| data[[i, j]].to_f64().unwrap()))
            .collect();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut nodes = Vec::new();

        build_recursive(
            &flat_data,
            &mut indices,
            0,
            n_samples,
            n_features,
            &mut nodes,
            leaf_size,
        );

        Self {
            nodes,
            data: flat_data,
            n_features,
            indices,
            _leaf_size: leaf_size,
        }
    }

    /// Query the k nearest neighbors of a point.
    ///
    /// Returns `(original_index, distance)` pairs sorted by distance ascending.
    /// The `_data` parameter is accepted for API compatibility but not used
    /// (data is stored internally during build).
    pub fn query<F: Float + Send + Sync + 'static>(
        &self,
        _data: &Array2<F>,
        query: &[f64],
        k: usize,
    ) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KnnCandidate> = BinaryHeap::with_capacity(k);
        self.knn_search(0, query, k, &mut heap);

        let mut results: Vec<(usize, f64)> = heap
            .into_iter()
            .map(|c| (c.orig_idx, c.dist_sq.sqrt()))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Find all points within `radius` of `query`.
    ///
    /// Returns `(original_index, distance)` pairs (unsorted).
    pub fn within_radius(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut results = Vec::new();
        self.radius_search(0, query, radius, radius_sq, &mut results);
        results
    }

    fn knn_search(
        &self,
        node_idx: usize,
        query: &[f64],
        k: usize,
        heap: &mut BinaryHeap<KnnCandidate>,
    ) {
        let node = &self.nodes[node_idx];
        let nf = self.n_features;

        // Prune: if the closest possible point in this ball is farther than
        // our current worst, skip this subtree.
        if heap.len() == k {
            let worst_sq = heap.peek().unwrap().dist_sq;
            if ball_lower_bound_sq(query, &node.centroid, node.radius_sq) > worst_sq {
                return;
            }
        }

        match node.kind {
            NodeKind::Leaf => {
                for &idx in &self.indices[node.start..node.end] {
                    let point = &self.data[idx * nf..(idx + 1) * nf];
                    let d2 = dist_sq(query, point);
                    if heap.len() < k {
                        heap.push(KnnCandidate {
                            dist_sq: d2,
                            orig_idx: idx,
                        });
                    } else if d2 < heap.peek().unwrap().dist_sq {
                        heap.pop();
                        heap.push(KnnCandidate {
                            dist_sq: d2,
                            orig_idx: idx,
                        });
                    }
                }
            }
            NodeKind::Branch { left, right } => {
                // Search the closer child first for better pruning.
                let dl = dist_sq(query, &self.nodes[left].centroid);
                let dr = dist_sq(query, &self.nodes[right].centroid);

                let (first, second) = if dl <= dr {
                    (left, right)
                } else {
                    (right, left)
                };

                self.knn_search(first, query, k, heap);
                self.knn_search(second, query, k, heap);
            }
        }
    }

    fn radius_search(
        &self,
        node_idx: usize,
        query: &[f64],
        radius: f64,
        radius_sq: f64,
        results: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let nf = self.n_features;

        let dist_to_center = dist_sq(query, &node.centroid).sqrt();
        let node_radius = node.radius_sq.sqrt();

        // Prune: ball is entirely outside the search radius.
        if dist_to_center - node_radius > radius {
            return;
        }

        // Bulk include: ball is entirely within the search radius.
        if dist_to_center + node_radius <= radius {
            for &idx in &self.indices[node.start..node.end] {
                let point = &self.data[idx * nf..(idx + 1) * nf];
                let d = dist_sq(query, point).sqrt();
                results.push((idx, d));
            }
            return;
        }

        match node.kind {
            NodeKind::Leaf => {
                for &idx in &self.indices[node.start..node.end] {
                    let point = &self.data[idx * nf..(idx + 1) * nf];
                    let d2 = dist_sq(query, point);
                    if d2 <= radius_sq {
                        results.push((idx, d2.sqrt()));
                    }
                }
            }
            NodeKind::Branch { left, right } => {
                self.radius_search(left, query, radius, radius_sq, results);
                self.radius_search(right, query, radius, radius_sq, results);
            }
        }
    }
}

/// Recursively build the ball tree, appending nodes to the flat `nodes` vec.
/// Returns the index of the created node.
fn build_recursive(
    data: &[f64],
    indices: &mut [usize],
    start: usize,
    end: usize,
    n_features: usize,
    nodes: &mut Vec<Node>,
    leaf_size: usize,
) -> usize {
    let count = end - start;
    debug_assert!(count > 0);

    // Single pass: compute centroid and per-dimension min/max for split axis.
    let mut centroid = vec![0.0; n_features];
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];

    for &idx in &indices[start..end] {
        let base = idx * n_features;
        for j in 0..n_features {
            let v = data[base + j];
            centroid[j] += v;
            if v < mins[j] {
                mins[j] = v;
            }
            if v > maxs[j] {
                maxs[j] = v;
            }
        }
    }

    let inv_n = 1.0 / count as f64;
    for v in &mut centroid {
        *v *= inv_n;
    }

    // Compute squared radius: max squared distance from centroid to any point.
    let mut radius_sq = 0.0_f64;
    for &idx in &indices[start..end] {
        let point = &data[idx * n_features..(idx + 1) * n_features];
        let d2 = dist_sq(&centroid, point);
        if d2 > radius_sq {
            radius_sq = d2;
        }
    }

    let node_idx = nodes.len();

    // Leaf node.
    if count <= leaf_size {
        nodes.push(Node {
            centroid,
            radius_sq,
            start,
            end,
            kind: NodeKind::Leaf,
        });
        return node_idx;
    }

    // Reserve slot — children will be appended after.
    nodes.push(Node {
        centroid,
        radius_sq,
        start,
        end,
        kind: NodeKind::Leaf, // placeholder
    });

    // Split along the dimension of greatest spread.
    let split_dim = mins
        .iter()
        .zip(maxs.iter())
        .enumerate()
        .max_by(|(_, (a_min, a_max)), (_, (b_min, b_max))| {
            (*a_max - *a_min)
                .partial_cmp(&(*b_max - *b_min))
                .unwrap_or(Ordering::Equal)
        })
        .map_or(0, |(i, _)| i);

    let mid = start + count / 2;
    indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
        let va = data[a * n_features + split_dim];
        let vb = data[b * n_features + split_dim];
        va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
    });

    let left = build_recursive(data, indices, start, mid, n_features, nodes, leaf_size);
    let right = build_recursive(data, indices, mid, end, n_features, nodes, leaf_size);

    nodes[node_idx].kind = NodeKind::Branch { left, right };

    node_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kdtree;
    use ndarray::Array2;

    #[test]
    fn test_build_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let tree = BallTree::build(&data);
        assert!(tree.nodes.is_empty());
    }

    #[test]
    fn test_build_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let tree = BallTree::build(&data);
        assert_eq!(tree.nodes.len(), 1);

        let neighbors = tree.query(&data, &[1.0, 2.0], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!(neighbors[0].1 < 1e-10);
    }

    #[test]
    fn test_query_simple() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = BallTree::build(&data);
        let neighbors = tree.query(&data, &[0.1, 0.1], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
    }

    #[test]
    fn test_query_k_neighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 10.0],
        )
        .unwrap();

        let tree = BallTree::build(&data);
        let neighbors = tree.query(&data, &[0.5, 0.5], 4);
        assert_eq!(neighbors.len(), 4);

        // The 4 closest should be indices 0-3 (not 4, at (10,10)).
        let indices: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
        assert!(!indices.contains(&4));
    }

    #[test]
    fn test_balltree_matches_brute_force() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        let tree = BallTree::build(&data);
        let query = [0.5, 0.5];

        for k in 1..=8 {
            let bt_result = tree.query(&data, &query, k);
            let bf_result = kdtree::brute_force_knn(&data, &query, k);

            assert_eq!(bt_result.len(), bf_result.len(), "k={k}: length mismatch");

            for (i, (bt, bf)) in bt_result.iter().zip(bf_result.iter()).enumerate() {
                assert!(
                    (bt.1 - bf.1).abs() < 1e-10,
                    "k={k}, neighbor {i}: bt dist={}, bf dist={}",
                    bt.1,
                    bf.1
                );
            }
        }
    }

    #[test]
    fn test_high_dimensional() {
        // 50 points in 100 dimensions — the main use case.
        let n = 50;
        let d = 100;
        let flat: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.01).collect();
        let data = Array2::from_shape_vec((n, d), flat).unwrap();

        let tree = BallTree::build(&data);
        let query: Vec<f64> = vec![0.5; d];

        let bt_result = tree.query(&data, &query, 5);
        let bf_result = kdtree::brute_force_knn(&data, &query, 5);

        assert_eq!(bt_result.len(), bf_result.len());
        for (bt, bf) in bt_result.iter().zip(bf_result.iter()) {
            assert!(
                (bt.1 - bf.1).abs() < 1e-10,
                "bt dist={}, bf dist={}",
                bt.1,
                bf.1
            );
        }
    }

    #[test]
    fn test_custom_leaf_size() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        for leaf_size in [1, 2, 4, 8, 16] {
            let tree = BallTree::build_with_leaf_size(&data, leaf_size);
            let query = [0.5, 0.5];

            for k in 1..=8 {
                let bt_result = tree.query(&data, &query, k);
                let bf_result = kdtree::brute_force_knn(&data, &query, k);

                assert_eq!(
                    bt_result.len(),
                    bf_result.len(),
                    "leaf_size={leaf_size}, k={k}"
                );
                for (bt, bf) in bt_result.iter().zip(bf_result.iter()) {
                    assert!(
                        (bt.1 - bf.1).abs() < 1e-10,
                        "leaf_size={leaf_size}, k={k}: bt={}, bf={}",
                        bt.1,
                        bf.1
                    );
                }
            }
        }
    }

    #[test]
    fn test_within_radius_basic() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0]).unwrap();

        let tree = BallTree::build(&data);
        // Points within distance 1.5 of origin: (0,0), (1,0), (0,1)
        let results = tree.within_radius(&[0.0, 0.0], 1.5);
        let mut indices: Vec<usize> = results.iter().map(|r| r.0).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_within_radius_empty() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = BallTree::build(&data);
        let results = tree.within_radius(&[10.0, 10.0], 0.1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_within_radius_all() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let tree = BallTree::build(&data);
        let results = tree.within_radius(&[0.5, 0.5], 100.0);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_within_radius_brute_force_comparison() {
        let n = 50;
        let d = 5;
        let flat: Vec<f64> = (0..n * d).map(|i| (i as f64) * 0.1).collect();
        let data = Array2::from_shape_vec((n, d), flat.clone()).unwrap();

        let tree = BallTree::build(&data);
        let query = vec![1.0; d];
        let radius = 3.0;

        let mut bt_results = tree.within_radius(&query, radius);
        bt_results.sort_by_key(|r| r.0);

        // Brute force
        let mut bf_results: Vec<(usize, f64)> = (0..n)
            .filter_map(|i| {
                let point = &flat[i * d..(i + 1) * d];
                let d2: f64 = point
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let dist = d2.sqrt();
                if dist <= radius {
                    Some((i, dist))
                } else {
                    None
                }
            })
            .collect();
        bf_results.sort_by_key(|r| r.0);

        assert_eq!(bt_results.len(), bf_results.len(), "count mismatch");
        for (bt, bf) in bt_results.iter().zip(bf_results.iter()) {
            assert_eq!(bt.0, bf.0);
            assert!((bt.1 - bf.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_query_empty_tree() {
        let data = Array2::<f64>::zeros((0, 2));
        let tree = BallTree::build(&data);
        let results = tree.query(&data, &[0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_k_zero() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let tree = BallTree::build(&data);
        let results = tree.query(&data, &[0.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_large_dataset() {
        // 500 points in 10 dimensions
        let n = 500;
        let d = 10;
        let flat: Vec<f64> = (0..n * d)
            .map(|i| ((i * 7 + 13) % 100) as f64 * 0.01)
            .collect();
        let data = Array2::from_shape_vec((n, d), flat).unwrap();

        let tree = BallTree::build(&data);
        let query: Vec<f64> = vec![0.5; d];

        let bt_result = tree.query(&data, &query, 10);
        let bf_result = kdtree::brute_force_knn(&data, &query, 10);

        assert_eq!(bt_result.len(), bf_result.len());
        for (bt, bf) in bt_result.iter().zip(bf_result.iter()) {
            assert!((bt.1 - bf.1).abs() < 1e-9, "bt={}, bf={}", bt.1, bf.1);
        }
    }

    #[test]
    fn test_bounding_invariant() {
        // Every point should be within the root node's bounding ball.
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        let tree = BallTree::build(&data);
        let root = &tree.nodes[0];

        for i in 0..8 {
            let point = &tree.data[i * 2..(i + 1) * 2];
            let d2 = dist_sq(&root.centroid, point);
            assert!(
                d2 <= root.radius_sq + 1e-10,
                "point {i} at dist_sq={d2} outside root radius_sq={}",
                root.radius_sq
            );
        }
    }

    #[test]
    fn test_duplicate_points() {
        let flat: Vec<f64> = std::iter::repeat_n([5.0, 5.0], 20).flatten().collect();
        let data = Array2::from_shape_vec((20, 2), flat).unwrap();

        let tree = BallTree::build(&data);
        let results = tree.query(&data, &[5.0, 5.0], 5);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.1 < 1e-10);
        }
    }
}
