//! Graph algorithms on sparse adjacency matrices.
//!
//! This module provides scipy.sparse.csgraph equivalents for the ferrolearn
//! framework, operating on sparse CSR adjacency/weight matrices (`sprs::CsMat<f64>`):
//!
//! - **[`dijkstra`]** — Single-source shortest paths (Dijkstra's algorithm).
//! - **[`dijkstra_all_pairs`]** — All-pairs shortest paths via repeated Dijkstra.
//! - **[`connected_components`]** — BFS-based connected component labelling
//!   (treating the graph as undirected).
//! - **[`minimum_spanning_tree`]** — Minimum spanning tree via Kruskal's algorithm
//!   with union-find.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use ndarray::{Array1, Array2};
use sprs::CsMat;

/// Result of single-source Dijkstra's algorithm.
#[derive(Debug, Clone)]
pub struct DijkstraResult {
    /// Shortest distance from the source to each vertex.
    ///
    /// Set to [`f64::INFINITY`] for vertices that are unreachable from the
    /// source.
    pub distances: Array1<f64>,

    /// Predecessor of each vertex on the shortest-path tree.
    ///
    /// Set to [`usize::MAX`] when the vertex has no predecessor (i.e. it is
    /// the source itself, or it is unreachable).
    pub predecessors: Array1<usize>,
}

/// Result of connected-components analysis.
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    /// Number of connected components.
    pub n_components: usize,

    /// Component label for each vertex (0-indexed, contiguous).
    pub labels: Array1<usize>,
}

// ---------------------------------------------------------------------------
// Dijkstra — single-source
// ---------------------------------------------------------------------------

/// Compute single-source shortest paths using Dijkstra's algorithm.
///
/// # Arguments
///
/// * `graph` — Square CSR weight matrix. Entry `(i, j)` is the edge weight
///   from vertex `i` to vertex `j`. All stored weights must be non-negative.
/// * `source` — Index of the source vertex.
///
/// # Errors
///
/// Returns an error if:
/// - The matrix is not square.
/// - `source` is out of bounds.
/// - Any stored edge weight is negative.
pub fn dijkstra(graph: &CsMat<f64>, source: usize) -> Result<DijkstraResult, String> {
    let (rows, cols) = graph.shape();
    if rows != cols {
        return Err(format!(
            "adjacency matrix must be square, got shape ({rows}, {cols})"
        ));
    }
    let n = rows;
    if source >= n {
        return Err(format!(
            "source vertex {source} is out of bounds for graph with {n} vertices"
        ));
    }

    // Validate non-negative weights.
    for (&w, _) in graph.iter() {
        if w < 0.0 {
            return Err(format!(
                "negative edge weight {w} is not allowed in Dijkstra's algorithm"
            ));
        }
    }

    let mut dist = Array1::from_elem(n, f64::INFINITY);
    let mut pred = Array1::from_elem(n, usize::MAX);
    dist[source] = 0.0;

    // Min-heap of (distance, vertex).
    let mut heap: BinaryHeap<Reverse<(OrderedF64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((OrderedF64(0.0), source)));

    while let Some(Reverse((OrderedF64(d_u), u))) = heap.pop() {
        // Skip stale entries.
        if d_u > dist[u] {
            continue;
        }

        if let Some(row) = graph.outer_view(u) {
            for (v, &w) in row.iter() {
                let new_dist = d_u + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    pred[v] = u;
                    heap.push(Reverse((OrderedF64(new_dist), v)));
                }
            }
        }
    }

    Ok(DijkstraResult {
        distances: dist,
        predecessors: pred,
    })
}

// ---------------------------------------------------------------------------
// Dijkstra — all-pairs
// ---------------------------------------------------------------------------

/// Compute all-pairs shortest paths by running Dijkstra from every vertex.
///
/// Returns an `n x n` matrix where entry `(i, j)` is the shortest distance
/// from vertex `i` to vertex `j` ([`f64::INFINITY`] if unreachable).
///
/// # Errors
///
/// Returns an error if the matrix is not square or contains negative weights.
pub fn dijkstra_all_pairs(graph: &CsMat<f64>) -> Result<Array2<f64>, String> {
    let (rows, cols) = graph.shape();
    if rows != cols {
        return Err(format!(
            "adjacency matrix must be square, got shape ({rows}, {cols})"
        ));
    }
    let n = rows;
    let mut result = Array2::from_elem((n, n), f64::INFINITY);

    for src in 0..n {
        let single = dijkstra(graph, src)?;
        result.row_mut(src).assign(&single.distances);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Connected components (undirected BFS)
// ---------------------------------------------------------------------------

/// Find connected components of a graph, treating it as undirected.
///
/// An edge exists between vertices `i` and `j` whenever `graph[(i, j)] != 0`
/// **or** `graph[(j, i)] != 0`. The algorithm uses BFS to label every vertex
/// with a component id (0-indexed).
///
/// # Errors
///
/// Returns an error if the matrix is not square.
pub fn connected_components(graph: &CsMat<f64>) -> Result<ConnectedComponentsResult, String> {
    let (rows, cols) = graph.shape();
    if rows != cols {
        return Err(format!(
            "adjacency matrix must be square, got shape ({rows}, {cols})"
        ));
    }
    let n = rows;

    // Build undirected adjacency lists from the CSR matrix.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&val, (i, j)) in graph.iter() {
        if val != 0.0 {
            adj[i].push(j);
            adj[j].push(i);
        }
    }

    let mut labels = Array1::from_elem(n, usize::MAX);
    let mut component = 0usize;
    let mut queue = VecDeque::new();

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        // BFS from `start`.
        labels[start] = component;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if labels[v] == usize::MAX {
                    labels[v] = component;
                    queue.push_back(v);
                }
            }
        }
        component += 1;
    }

    Ok(ConnectedComponentsResult {
        n_components: component,
        labels,
    })
}

// ---------------------------------------------------------------------------
// Minimum spanning tree (Kruskal)
// ---------------------------------------------------------------------------

/// Compute the minimum spanning tree of an undirected, weighted graph using
/// Kruskal's algorithm.
///
/// The input should be a symmetric CSR weight matrix (undirected graph). Each
/// non-zero entry `(i, j)` with `i < j` is treated as one undirected edge of
/// the given weight.
///
/// Returns a symmetric CSR matrix containing only the MST edges. If the graph
/// is disconnected the result is a minimum spanning *forest*.
///
/// # Errors
///
/// Returns an error if the matrix is not square or contains negative weights.
pub fn minimum_spanning_tree(graph: &CsMat<f64>) -> Result<CsMat<f64>, String> {
    let (rows, cols) = graph.shape();
    if rows != cols {
        return Err(format!(
            "adjacency matrix must be square, got shape ({rows}, {cols})"
        ));
    }
    let n = rows;

    // Collect edges (deduplicate by requiring i < j).
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for (&val, (i, j)) in graph.iter() {
        if val < 0.0 {
            return Err(format!(
                "negative edge weight {val} is not allowed in minimum_spanning_tree"
            ));
        }
        if i < j && val != 0.0 {
            edges.push((val, i, j));
        }
    }

    // Sort by weight.
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-find with path compression and union by rank.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return false;
        }
        match rank[rx].cmp(&rank[ry]) {
            std::cmp::Ordering::Less => parent[rx] = ry,
            std::cmp::Ordering::Greater => parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                parent[ry] = rx;
                rank[rx] += 1;
            }
        }
        true
    }

    // Build MST edges.
    let mut mst_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for (w, u, v) in edges {
        if union(&mut parent, &mut rank, u, v) {
            // Store both directions for a symmetric matrix.
            mst_triplets.push((u, v, w));
            mst_triplets.push((v, u, w));
        }
    }

    // Build CSR matrix from triplets.
    let mut tri = sprs::TriMat::new((n, n));
    for (i, j, w) in &mst_triplets {
        tri.add_triplet(*i, *j, *w);
    }
    Ok(tri.to_csr())
}

// ---------------------------------------------------------------------------
// Helper: ordered f64 wrapper for BinaryHeap
// ---------------------------------------------------------------------------

/// A wrapper around `f64` that implements [`Ord`] via [`f64::total_cmp`],
/// allowing it to be used in a [`BinaryHeap`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use sprs::TriMat;

    /// Helper: build a CSR matrix from a list of (row, col, value) triplets.
    fn csr_from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
        let mut tri = TriMat::new((n, n));
        for &(r, c, v) in triplets {
            tri.add_triplet(r, c, v);
        }
        tri.to_csr()
    }

    // -----------------------------------------------------------------------
    // Dijkstra tests
    // -----------------------------------------------------------------------

    #[test]
    fn dijkstra_simple_path() {
        // Linear graph: 0 --1--> 1 --2--> 2 --3--> 3
        let graph = csr_from_triplets(4, &[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)]);
        let res = dijkstra(&graph, 0).unwrap();

        assert_abs_diff_eq!(res.distances[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(res.distances[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(res.distances[2], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(res.distances[3], 6.0, epsilon = 1e-12);

        assert_eq!(res.predecessors[0], usize::MAX); // source has no predecessor
        assert_eq!(res.predecessors[1], 0);
        assert_eq!(res.predecessors[2], 1);
        assert_eq!(res.predecessors[3], 2);
    }

    #[test]
    fn dijkstra_disconnected() {
        // Two disconnected components: {0, 1} and {2, 3}
        let graph = csr_from_triplets(4, &[(0, 1, 1.0), (1, 0, 1.0), (2, 3, 2.0), (3, 2, 2.0)]);
        let res = dijkstra(&graph, 0).unwrap();

        assert_abs_diff_eq!(res.distances[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(res.distances[1], 1.0, epsilon = 1e-12);
        assert!(res.distances[2].is_infinite());
        assert!(res.distances[3].is_infinite());

        assert_eq!(res.predecessors[2], usize::MAX);
        assert_eq!(res.predecessors[3], usize::MAX);
    }

    #[test]
    fn dijkstra_all_pairs_triangle() {
        // Triangle: 0 --1-- 1 --1-- 2 --1-- 0 (bidirectional)
        let graph = csr_from_triplets(
            3,
            &[
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (0, 2, 1.0),
                (2, 0, 1.0),
            ],
        );
        let dist = dijkstra_all_pairs(&graph).unwrap();

        // All distances should be 0 on diagonal and 1 off-diagonal.
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_abs_diff_eq!(dist[[i, j]], 0.0, epsilon = 1e-12);
                } else {
                    assert_abs_diff_eq!(dist[[i, j]], 1.0, epsilon = 1e-12);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Connected components tests
    // -----------------------------------------------------------------------

    #[test]
    fn connected_components_two_clusters() {
        // Cluster 1: {0, 1, 2} — fully connected
        // Cluster 2: {3, 4}    — connected
        let graph = csr_from_triplets(5, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0)]);
        let res = connected_components(&graph).unwrap();

        assert_eq!(res.n_components, 2);
        // Vertices 0, 1, 2 should share a label, 3 and 4 another.
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[1], res.labels[2]);
        assert_eq!(res.labels[3], res.labels[4]);
        assert_ne!(res.labels[0], res.labels[3]);
    }

    #[test]
    fn connected_components_single() {
        // Fully connected 4-node graph.
        let mut triplets = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    triplets.push((i, j, 1.0));
                }
            }
        }
        let graph = csr_from_triplets(4, &triplets);
        let res = connected_components(&graph).unwrap();

        assert_eq!(res.n_components, 1);
        for i in 0..4 {
            assert_eq!(res.labels[i], 0);
        }
    }

    // -----------------------------------------------------------------------
    // Minimum spanning tree tests
    // -----------------------------------------------------------------------

    #[test]
    fn mst_simple() {
        // 4-node graph (symmetric):
        //   0 --1-- 1
        //   |       |
        //   4       2
        //   |       |
        //   3 --3-- 2
        //
        // Also add edge 0--2 with weight 5 (should NOT be in MST).
        let graph = csr_from_triplets(
            4,
            &[
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 2, 2.0),
                (2, 1, 2.0),
                (2, 3, 3.0),
                (3, 2, 3.0),
                (0, 3, 4.0),
                (3, 0, 4.0),
                (0, 2, 5.0),
                (2, 0, 5.0),
            ],
        );
        let mst = minimum_spanning_tree(&graph).unwrap();

        // MST should have edges: 0-1 (1), 1-2 (2), 2-3 (3). Total weight = 6.
        // Since the MST is stored symmetrically, sum of all entries = 12.
        let total_weight: f64 = mst.iter().map(|(&w, _)| w).sum();
        assert_abs_diff_eq!(total_weight, 12.0, epsilon = 1e-12);

        // MST should have exactly 3 undirected edges = 6 entries in symmetric matrix.
        let nnz = mst.nnz();
        assert_eq!(nnz, 6);

        // Verify the specific edges are present.
        let mst_dense = mst.to_dense();
        assert_abs_diff_eq!(mst_dense[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mst_dense[[1, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mst_dense[[1, 2]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mst_dense[[2, 1]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mst_dense[[2, 3]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mst_dense[[3, 2]], 3.0, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn negative_weight_error() {
        let graph = csr_from_triplets(3, &[(0, 1, 1.0), (1, 2, -3.0)]);
        let res = dijkstra(&graph, 0);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("negative"));

        let res2 = minimum_spanning_tree(&graph);
        assert!(res2.is_err());
        assert!(res2.unwrap_err().contains("negative"));
    }
}
