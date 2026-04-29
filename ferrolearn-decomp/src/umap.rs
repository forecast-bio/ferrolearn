//! Uniform Manifold Approximation and Projection (UMAP).
//!
//! [`Umap`] performs non-linear dimensionality reduction based on the
//! mathematical framework of Riemannian geometry and algebraic topology,
//! as described by McInnes, Healy, and Melville (2018).
//!
//! # Algorithm
//!
//! 1. Build a k-nearest-neighbor graph with `n_neighbors` neighbors.
//! 2. Compute the fuzzy simplicial set by smoothing kNN distances: for each
//!    point find a local connectivity parameter `rho` (distance to nearest
//!    neighbor) and a bandwidth `sigma` such that the sum of the membership
//!    strengths equals `log2(n_neighbors)`.
//! 3. Symmetrise the fuzzy graph: `w_ij = w_i|j + w_j|i - w_i|j * w_j|i`.
//! 4. Determine curve parameters `a` and `b` from `min_dist` and `spread`
//!    that define the target distribution in the embedding space:
//!    `phi(d) = 1 / (1 + a * d^(2b))`.
//! 5. Initialise the embedding (spectral or random).
//! 6. Optimise via SGD with attractive forces on positive edges and
//!    repulsive forces via negative sampling (5 negatives per positive).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::Umap;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::<f64>::from_shape_fn((30, 5), |(i, j)| (i + j) as f64);
//! let umap = Umap::new().with_random_state(42).with_n_epochs(50);
//! let fitted = umap.fit(&x, &()).unwrap();
//! let emb = fitted.embedding();
//! assert_eq!(emb.ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

// ---------------------------------------------------------------------------
// Metric enum
// ---------------------------------------------------------------------------

/// Distance metric for UMAP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UmapMetric {
    /// Standard Euclidean distance.
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
}

// ---------------------------------------------------------------------------
// Umap (unfitted)
// ---------------------------------------------------------------------------

/// UMAP configuration.
///
/// Holds hyperparameters for the UMAP algorithm. Calling [`Fit::fit`]
/// computes the embedding and returns a [`FittedUmap`].
#[derive(Debug, Clone)]
pub struct Umap {
    /// Number of embedding dimensions (default 2).
    n_components: usize,
    /// Number of nearest neighbors for the kNN graph (default 15).
    n_neighbors: usize,
    /// Minimum distance in the embedding space (default 0.1).
    min_dist: f64,
    /// Spread of the embedding (default 1.0).
    spread: f64,
    /// Learning rate for SGD (default 1.0).
    learning_rate: f64,
    /// Number of SGD epochs (default 200).
    n_epochs: usize,
    /// Distance metric (default Euclidean).
    metric: UmapMetric,
    /// Number of negative samples per positive edge (default 5).
    negative_sample_rate: usize,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl Umap {
    /// Create a new `Umap` with default parameters.
    ///
    /// Defaults: `n_components=2`, `n_neighbors=15`, `min_dist=0.1`,
    /// `spread=1.0`, `learning_rate=1.0`, `n_epochs=200`, metric=`Euclidean`,
    /// `negative_sample_rate=5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            learning_rate: 1.0,
            n_epochs: 200,
            metric: UmapMetric::Euclidean,
            negative_sample_rate: 5,
            random_state: None,
        }
    }

    /// Set the number of embedding dimensions.
    #[must_use]
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set the number of nearest neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Set the minimum distance in the embedding.
    #[must_use]
    pub fn with_min_dist(mut self, d: f64) -> Self {
        self.min_dist = d;
        self
    }

    /// Set the spread.
    #[must_use]
    pub fn with_spread(mut self, s: f64) -> Self {
        self.spread = s;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of SGD epochs.
    #[must_use]
    pub fn with_n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_metric(mut self, m: UmapMetric) -> Self {
        self.metric = m;
        self
    }

    /// Set the negative sample rate.
    #[must_use]
    pub fn with_negative_sample_rate(mut self, rate: usize) -> Self {
        self.negative_sample_rate = rate;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured number of neighbors.
    #[must_use]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Return the configured minimum distance.
    #[must_use]
    pub fn min_dist(&self) -> f64 {
        self.min_dist
    }

    /// Return the configured spread.
    #[must_use]
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Return the configured learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Return the configured number of epochs.
    #[must_use]
    pub fn n_epochs(&self) -> usize {
        self.n_epochs
    }

    /// Return the configured metric.
    #[must_use]
    pub fn metric(&self) -> UmapMetric {
        self.metric
    }

    /// Return the configured negative sample rate.
    #[must_use]
    pub fn negative_sample_rate(&self) -> usize {
        self.negative_sample_rate
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

impl Default for Umap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedUmap
// ---------------------------------------------------------------------------

/// A fitted UMAP model holding the learned embedding and training data.
///
/// Created by calling [`Fit::fit`] on a [`Umap`]. Implements
/// [`Transform<Array2<f64>>`] for projecting new data via nearest-neighbor
/// lookup.
#[derive(Debug, Clone)]
pub struct FittedUmap {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
    /// Training data, stored for out-of-sample extension.
    x_train_: Array2<f64>,
    /// Curve parameter `a`.
    a_: f64,
    /// Curve parameter `b`.
    b_: f64,
    /// Number of neighbors used.
    n_neighbors_: usize,
    /// The metric used.
    metric_: UmapMetric,
}

impl FittedUmap {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }

    /// The curve parameter `a`.
    #[must_use]
    pub fn a(&self) -> f64 {
        self.a_
    }

    /// The curve parameter `b`.
    #[must_use]
    pub fn b(&self) -> f64 {
        self.b_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute distance between two rows of a matrix using the given metric.
fn compute_distance(x: &Array2<f64>, i: usize, j: usize, metric: UmapMetric) -> f64 {
    let ncols = x.ncols();
    match metric {
        UmapMetric::Euclidean => {
            let mut sq = 0.0;
            for k in 0..ncols {
                let diff = x[[i, k]] - x[[j, k]];
                sq += diff * diff;
            }
            sq.sqrt()
        }
        UmapMetric::Manhattan => {
            let mut sum = 0.0;
            for k in 0..ncols {
                sum += (x[[i, k]] - x[[j, k]]).abs();
            }
            sum
        }
        UmapMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_i = 0.0;
            let mut norm_j = 0.0;
            for k in 0..ncols {
                dot += x[[i, k]] * x[[j, k]];
                norm_i += x[[i, k]] * x[[i, k]];
                norm_j += x[[j, k]] * x[[j, k]];
            }
            let denom = (norm_i * norm_j).sqrt();
            if denom < 1e-16 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }
}

/// Compute distance between a point (row of x_new) and a training point.
fn compute_distance_cross(
    x_new: &Array2<f64>,
    i: usize,
    x_train: &Array2<f64>,
    j: usize,
    metric: UmapMetric,
) -> f64 {
    let ncols = x_new.ncols();
    match metric {
        UmapMetric::Euclidean => {
            let mut sq = 0.0;
            for k in 0..ncols {
                let diff = x_new[[i, k]] - x_train[[j, k]];
                sq += diff * diff;
            }
            sq.sqrt()
        }
        UmapMetric::Manhattan => {
            let mut sum = 0.0;
            for k in 0..ncols {
                sum += (x_new[[i, k]] - x_train[[j, k]]).abs();
            }
            sum
        }
        UmapMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_i = 0.0;
            let mut norm_j = 0.0;
            for k in 0..ncols {
                dot += x_new[[i, k]] * x_train[[j, k]];
                norm_i += x_new[[i, k]] * x_new[[i, k]];
                norm_j += x_train[[j, k]] * x_train[[j, k]];
            }
            let denom = (norm_i * norm_j).sqrt();
            if denom < 1e-16 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }
}

/// Build k-nearest-neighbor graph. Returns for each point the sorted list of
/// (neighbor_index, distance) pairs.
fn build_knn(x: &Array2<f64>, k: usize, metric: UmapMetric) -> Vec<Vec<(usize, f64)>> {
    let n = x.nrows();
    let mut knn = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, compute_distance(x, i, j, metric)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        knn.push(dists);
    }
    knn
}

/// Compute the fuzzy simplicial set: smooth kNN distances to get membership
/// strengths.
///
/// For each point i, find `rho_i` (distance to nearest neighbor) and
/// `sigma_i` such that `sum_j exp(-(d(i,j) - rho_i) / sigma_i) = log2(k)`.
///
/// Returns a sparse-ish weighted graph as a list of (i, j, weight) edges.
fn compute_fuzzy_simplicial_set(knn: &[Vec<(usize, f64)>], n: usize) -> Vec<(usize, usize, f64)> {
    let k = if knn.is_empty() { 0 } else { knn[0].len() };
    let target = (k as f64).ln() / std::f64::consts::LN_2; // log2(k)

    // For each point, compute rho and sigma.
    let mut rho = vec![0.0; n];
    let mut sigma = vec![1.0; n];

    for i in 0..n {
        if knn[i].is_empty() {
            continue;
        }
        // rho_i = distance to nearest neighbor.
        rho[i] = knn[i][0].1;
        if rho[i] < 1e-16 {
            // If nearest neighbor is at distance 0, find first non-zero.
            for &(_, d) in &knn[i] {
                if d > 1e-16 {
                    rho[i] = d;
                    break;
                }
            }
        }

        // Binary search for sigma.
        let mut lo = 1e-20_f64;
        let mut hi = 1e4_f64;
        for _iter in 0..64 {
            let mid = f64::midpoint(lo, hi);
            let mut val = 0.0;
            for &(_, d) in &knn[i] {
                let adjusted = (d - rho[i]).max(0.0);
                val += (-adjusted / mid).exp();
            }
            if val > target {
                hi = mid;
            } else {
                lo = mid;
            }
            if (hi - lo) / (lo + 1e-16) < 1e-5 {
                break;
            }
        }
        sigma[i] = f64::midpoint(lo, hi);
    }

    // Build directed graph with membership strengths.
    // w_{i|j} = exp(-(d(i,j) - rho_i) / sigma_i)  for j in knn(i)
    let mut directed: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, neighbors) in knn.iter().enumerate() {
        for &(j, d) in neighbors {
            let adjusted = (d - rho[i]).max(0.0);
            let w = (-adjusted / sigma[i]).exp();
            directed[i].push((j, w));
        }
    }

    // Symmetrise: w_ij = w_{i|j} + w_{j|i} - w_{i|j} * w_{j|i}
    // Use a hash map approach for efficiency.
    // Collect directed weights for each undirected edge.
    let mut forward: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();
    let mut backward: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();

    for (i, neighbors) in directed.iter().enumerate() {
        for &(j, w) in neighbors {
            let key = if i < j { (i, j) } else { (j, i) };
            if i < j {
                forward.insert(key, w);
            } else {
                backward.insert(key, w);
            }
        }
    }

    // Combine keys.
    let mut all_keys: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for &k in forward.keys() {
        all_keys.insert(k);
    }
    for &k in backward.keys() {
        all_keys.insert(k);
    }

    let mut edges = Vec::with_capacity(all_keys.len());
    for key in all_keys {
        let w_fwd = forward.get(&key).copied().unwrap_or(0.0);
        let w_bwd = backward.get(&key).copied().unwrap_or(0.0);
        let w = w_fwd + w_bwd - w_fwd * w_bwd;
        if w > 1e-16 {
            edges.push((key.0, key.1, w));
        }
    }

    edges
}

/// Find curve parameters `a` and `b` from `min_dist` and `spread`.
///
/// We want `1 / (1 + a * d^(2b)) ~ 1` when `d < min_dist` and
/// `exp(-(d - min_dist) / spread)` when `d >= min_dist`.
///
/// This is solved by a simple grid search / least squares fit.
fn find_ab_params(min_dist: f64, spread: f64) -> (f64, f64) {
    // Sample distances and target values.
    let n_samples = 300;
    let d_max = 3.0 * spread;
    let mut best_a = 1.0;
    let mut best_b = 1.0;
    let mut best_err = f64::MAX;

    // Grid search over a and b.
    let a_range: Vec<f64> = (1..=40).map(|i| f64::from(i) * 0.25).collect();
    let b_range: Vec<f64> = (1..=30).map(|i| f64::from(i) * 0.1).collect();

    for &a in &a_range {
        for &b in &b_range {
            let mut err = 0.0;
            for k in 0..n_samples {
                let d = (f64::from(k) + 0.5) / f64::from(n_samples) * d_max;
                let target = if d <= min_dist {
                    1.0
                } else {
                    (-(d - min_dist) / spread).exp()
                };
                let pred = 1.0 / (1.0 + a * d.powf(2.0 * b));
                let diff = pred - target;
                err += diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Refine with a finer grid around the best.
    let a_lo = (best_a - 0.3).max(0.01);
    let a_hi = best_a + 0.3;
    let b_lo = (best_b - 0.15).max(0.01);
    let b_hi = best_b + 0.15;

    for ia in 0..20 {
        let a = a_lo + (a_hi - a_lo) * f64::from(ia) / 19.0;
        for ib in 0..20 {
            let b = b_lo + (b_hi - b_lo) * f64::from(ib) / 19.0;
            let mut err = 0.0;
            for k in 0..n_samples {
                let d = (f64::from(k) + 0.5) / f64::from(n_samples) * d_max;
                let target = if d <= min_dist {
                    1.0
                } else {
                    (-(d - min_dist) / spread).exp()
                };
                let pred = 1.0 / (1.0 + a * d.powf(2.0 * b));
                let diff = pred - target;
                err += diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    (best_a, best_b)
}

/// Clip a value to prevent overflow/underflow in gradient computation.
fn clip(val: f64, lo: f64, hi: f64) -> f64 {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for Umap {
    type Fitted = FittedUmap;
    type Error = FerroError;

    /// Fit UMAP by computing the fuzzy simplicial set and optimising the
    /// low-dimensional embedding via SGD.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero,
    ///   `n_neighbors` is zero or too large, `min_dist` is negative,
    ///   `spread` is non-positive, or `learning_rate` is non-positive.
    /// - [`FerroError::InsufficientSamples`] if there are fewer samples than
    ///   `n_neighbors + 1`.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedUmap, FerroError> {
        let n = x.nrows();

        // Validate parameters.
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "Umap::fit requires at least 2 samples".into(),
            });
        }
        let effective_k = self.n_neighbors.min(n - 1);
        if self.min_dist < 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "min_dist".into(),
                reason: "must be non-negative".into(),
            });
        }
        if self.spread <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "spread".into(),
                reason: "must be positive".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }

        let dim = self.n_components;
        let seed = self.random_state.unwrap_or(0);

        // Step 1: Build kNN graph.
        let knn = build_knn(x, effective_k, self.metric);

        // Step 2: Compute fuzzy simplicial set.
        let edges = compute_fuzzy_simplicial_set(&knn, n);

        // Step 3: Find a, b parameters.
        let (a, b) = find_ab_params(self.min_dist, self.spread);

        // Step 4: Initialise embedding (random uniform).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let uniform = Uniform::new(-10.0, 10.0).unwrap();
        let mut y = Array2::<f64>::zeros((n, dim));
        for elem in &mut y {
            *elem = uniform.sample(&mut rng);
        }

        // Pre-compute epochs per edge: spread epochs proportional to weight.
        if edges.is_empty() {
            return Ok(FittedUmap {
                embedding_: y,
                x_train_: x.to_owned(),
                a_: a,
                b_: b,
                n_neighbors_: effective_k,
                metric_: self.metric,
            });
        }

        let max_weight = edges.iter().map(|e| e.2).fold(0.0_f64, f64::max);

        // Each edge gets `n_epochs * (weight / max_weight)` total updates.
        let epochs_per_sample: Vec<f64> = edges
            .iter()
            .map(|e| {
                let ratio = e.2 / max_weight;
                if ratio > 0.0 {
                    (self.n_epochs as f64) / ((self.n_epochs as f64) * ratio).max(1.0)
                } else {
                    f64::MAX
                }
            })
            .collect();

        let mut epoch_of_next_sample: Vec<f64> = epochs_per_sample.clone();

        let neg_rate = self.negative_sample_rate;
        let idx_uniform = Uniform::new(0usize, n).unwrap();

        // Step 5: SGD optimisation.
        for epoch in 0..self.n_epochs {
            let alpha = self.learning_rate * (1.0 - epoch as f64 / self.n_epochs as f64);
            let alpha = alpha.max(0.0);

            for (edge_idx, &(ei, ej, _weight)) in edges.iter().enumerate() {
                if epoch_of_next_sample[edge_idx] > epoch as f64 {
                    continue;
                }

                // Attractive force.
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = y[[ei, d]] - y[[ej, d]];
                    dist_sq += diff * diff;
                }
                let dist_sq = dist_sq.max(1e-16);

                let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));

                for d in 0..dim {
                    let diff = y[[ei, d]] - y[[ej, d]];
                    let grad = clip(grad_coeff * diff, -4.0, 4.0);
                    y[[ei, d]] += alpha * grad;
                    y[[ej, d]] -= alpha * grad;
                }

                // Negative sampling.
                for _ in 0..neg_rate {
                    let neg = idx_uniform.sample(&mut rng);
                    if neg == ei {
                        continue;
                    }
                    let mut neg_dist_sq = 0.0;
                    for d in 0..dim {
                        let diff = y[[ei, d]] - y[[neg, d]];
                        neg_dist_sq += diff * diff;
                    }
                    let neg_dist_sq = neg_dist_sq.max(1e-16);

                    let rep_coeff =
                        2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_dist_sq.powf(b)));

                    for d in 0..dim {
                        let diff = y[[ei, d]] - y[[neg, d]];
                        let grad = clip(rep_coeff * diff, -4.0, 4.0);
                        y[[ei, d]] += alpha * grad;
                    }
                }

                epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];
            }
        }

        Ok(FittedUmap {
            embedding_: y,
            x_train_: x.to_owned(),
            a_: a,
            b_: b,
            n_neighbors_: effective_k,
            metric_: self.metric,
        })
    }
}

impl Transform<Array2<f64>> for FittedUmap {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Project new data into the UMAP embedding space.
    ///
    /// For each new point, find the nearest neighbors in the training data
    /// and compute a weighted average of their embeddings (weighted by the
    /// UMAP kernel).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the training data.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_features = self.x_train_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedUmap::transform".into(),
            });
        }

        let n_test = x.nrows();
        let n_train = self.x_train_.nrows();
        let dim = self.embedding_.ncols();
        let k = self.n_neighbors_.min(n_train);

        let mut result = Array2::<f64>::zeros((n_test, dim));

        for t in 0..n_test {
            // Find k nearest training neighbors.
            let mut dists: Vec<(usize, f64)> = (0..n_train)
                .map(|j| {
                    (
                        j,
                        compute_distance_cross(x, t, &self.x_train_, j, self.metric_),
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);

            // Compute weights using the UMAP kernel: 1/(1 + a * d^(2b)).
            let mut weights = Vec::with_capacity(k);
            let mut weight_sum = 0.0;
            for &(_, d) in &dists {
                let w = 1.0 / (1.0 + self.a_ * d.powf(2.0 * self.b_));
                weights.push(w);
                weight_sum += w;
            }

            if weight_sum < 1e-16 {
                // Fallback: uniform weights.
                weight_sum = k as f64;
                weights = vec![1.0; k];
            }

            // Weighted average of neighbor embeddings.
            for (idx, &(train_idx, _)) in dists.iter().enumerate() {
                let w = weights[idx] / weight_sum;
                for d in 0..dim {
                    result[[t, d]] += w * self.embedding_[[train_idx, d]];
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Generate small blobs dataset.
    fn make_blobs(seed: u64) -> (Array2<f64>, Vec<usize>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 0.3).unwrap();
        let n_per_cluster = 10;
        let n_features = 5;
        let centers = [
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![5.0, 5.0, 5.0, 5.0, 5.0],
            vec![10.0, 0.0, 10.0, 0.0, 10.0],
        ];
        let n = centers.len() * n_per_cluster;
        let mut x = Array2::<f64>::zeros((n, n_features));
        let mut labels = Vec::with_capacity(n);
        for (c_idx, center) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let row = c_idx * n_per_cluster + i;
                for (f, &c) in center.iter().enumerate() {
                    x[[row, f]] = c + normal.sample(&mut rng);
                }
                labels.push(c_idx);
            }
        }
        (x, labels)
    }

    #[test]
    fn test_umap_basic_shape() {
        let x = Array2::<f64>::from_shape_fn((30, 5), |(i, j)| (i + j) as f64);
        let umap = Umap::new().with_n_epochs(10).with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (30, 2));
    }

    #[test]
    fn test_umap_3d_embedding() {
        let x = Array2::<f64>::from_shape_fn((20, 4), |(i, j)| (i + j) as f64);
        let umap = Umap::new()
            .with_n_components(3)
            .with_n_epochs(10)
            .with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 3);
    }

    #[test]
    fn test_umap_separates_clusters() {
        let (x, labels) = make_blobs(42);
        let umap = Umap::new()
            .with_n_neighbors(5)
            .with_n_epochs(100)
            .with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        let emb = fitted.embedding();

        // Check cluster separation with k-NN accuracy (k=3).
        let n = emb.nrows();
        let mut correct = 0;
        for i in 0..n {
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut d = 0.0;
                    for dd in 0..emb.ncols() {
                        let diff = emb[[i, dd]] - emb[[j, dd]];
                        d += diff * diff;
                    }
                    (d, labels[j])
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut votes = [0usize; 3];
            for &(_, lbl) in dists.iter().take(3) {
                votes[lbl] += 1;
            }
            let pred = votes.iter().enumerate().max_by_key(|&(_, v)| v).unwrap().0;
            if pred == labels[i] {
                correct += 1;
            }
        }
        let accuracy = f64::from(correct) / n as f64;
        assert!(
            accuracy > 0.8,
            "UMAP k-NN accuracy should be > 80%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_umap_transform_new_data() {
        let (x, _) = make_blobs(42);
        let umap = Umap::new()
            .with_n_neighbors(5)
            .with_n_epochs(50)
            .with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();

        // Transform a subset of training data.
        let x_test = x.slice(ndarray::s![0..5, ..]).to_owned();
        let projected = fitted.transform(&x_test).unwrap();
        assert_eq!(projected.dim(), (5, 2));
    }

    #[test]
    fn test_umap_transform_shape_mismatch() {
        let x = Array2::<f64>::from_shape_fn((20, 4), |(i, j)| (i + j) as f64);
        let umap = Umap::new().with_n_epochs(10).with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        let x_bad = Array2::<f64>::zeros((5, 3)); // wrong number of features
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_umap_ab_params_reasonable() {
        let (a, b) = find_ab_params(0.1, 1.0);
        // a and b should be positive.
        assert!(a > 0.0, "a should be positive, got {a}");
        assert!(b > 0.0, "b should be positive, got {b}");
        // At d=0, 1/(1+a*0) = 1, which is correct.
        // At d=min_dist, should be close to 1.
        let val_at_min = 1.0 / (1.0 + a * (0.1_f64).powf(2.0 * b));
        assert!(
            val_at_min > 0.5,
            "kernel at min_dist should be > 0.5, got {val_at_min}"
        );
    }

    #[test]
    fn test_umap_invalid_n_components_zero() {
        let x = Array2::<f64>::zeros((10, 3));
        let umap = Umap::new().with_n_components(0);
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_invalid_n_neighbors_zero() {
        let x = Array2::<f64>::zeros((10, 3));
        let umap = Umap::new().with_n_neighbors(0);
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_invalid_min_dist() {
        let x = Array2::<f64>::zeros((10, 3));
        let umap = Umap::new().with_min_dist(-0.1);
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_invalid_spread() {
        let x = Array2::<f64>::zeros((10, 3));
        let umap = Umap::new().with_spread(0.0);
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_invalid_learning_rate() {
        let x = Array2::<f64>::zeros((10, 3));
        let umap = Umap::new().with_learning_rate(-1.0);
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_insufficient_samples() {
        let x = Array2::<f64>::zeros((1, 3));
        let umap = Umap::new();
        assert!(umap.fit(&x, &()).is_err());
    }

    #[test]
    fn test_umap_getters() {
        let umap = Umap::new()
            .with_n_components(3)
            .with_n_neighbors(10)
            .with_min_dist(0.2)
            .with_spread(1.5)
            .with_learning_rate(0.5)
            .with_n_epochs(100)
            .with_metric(UmapMetric::Manhattan)
            .with_negative_sample_rate(3)
            .with_random_state(99);
        assert_eq!(umap.n_components(), 3);
        assert_eq!(umap.n_neighbors(), 10);
        assert!((umap.min_dist() - 0.2).abs() < 1e-10);
        assert!((umap.spread() - 1.5).abs() < 1e-10);
        assert!((umap.learning_rate() - 0.5).abs() < 1e-10);
        assert_eq!(umap.n_epochs(), 100);
        assert_eq!(umap.metric(), UmapMetric::Manhattan);
        assert_eq!(umap.negative_sample_rate(), 3);
        assert_eq!(umap.random_state(), Some(99));
    }

    #[test]
    fn test_umap_default() {
        let umap = Umap::default();
        assert_eq!(umap.n_components(), 2);
        assert_eq!(umap.n_neighbors(), 15);
    }

    #[test]
    fn test_umap_cosine_metric() {
        let x = Array2::<f64>::from_shape_fn((20, 4), |(i, j)| (i + j + 1) as f64);
        let umap = Umap::new()
            .with_metric(UmapMetric::Cosine)
            .with_n_epochs(10)
            .with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
    }

    #[test]
    fn test_umap_small_n_neighbors_capped() {
        // n_neighbors > n-1 should be automatically capped
        let x = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| (i + j) as f64);
        let umap = Umap::new()
            .with_n_neighbors(100)
            .with_n_epochs(10)
            .with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (5, 2));
    }

    #[test]
    fn test_umap_fitted_accessors() {
        let x = Array2::<f64>::from_shape_fn((20, 4), |(i, j)| (i + j) as f64);
        let umap = Umap::new().with_n_epochs(10).with_random_state(42);
        let fitted = umap.fit(&x, &()).unwrap();
        assert!(fitted.a() > 0.0);
        assert!(fitted.b() > 0.0);
    }
}
