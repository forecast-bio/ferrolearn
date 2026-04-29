// Numeric/geometric code uses index-based loops extensively for clarity.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

//! t-distributed Stochastic Neighbor Embedding (t-SNE).
//!
//! [`Tsne`] performs non-linear dimensionality reduction by modelling pairwise
//! similarities in the high-dimensional space as conditional probabilities and
//! finding a low-dimensional embedding that preserves those similarities under
//! a Student-t distribution.
//!
//! # Algorithm
//!
//! 1. Compute pairwise affinities in the input space using a Gaussian kernel
//!    whose bandwidth (sigma) is set per point via binary search to match a
//!    target perplexity.
//! 2. Symmetrise the affinity matrix: `P_ij = (P_i|j + P_j|i) / (2n)`.
//! 3. Initialise a low-dimensional embedding (random Gaussian).
//! 4. Optimise with gradient descent using:
//!    - **Early exaggeration** (multiply P by a factor) for the first 250
//!      iterations.
//!    - **Momentum** (0.5 early, 0.8 later).
//!    - **Barnes-Hut approximation** with a spatial tree for O(n log n)
//!      repulsive force computation.
//!
//! # Important
//!
//! t-SNE does **not** support projecting new data — there is no `Transform`
//! implementation. Use [`FittedTsne::embedding`] to retrieve the result.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::Tsne;
//! use ferrolearn_core::traits::Fit;
//! use ndarray::Array2;
//!
//! let x = Array2::<f64>::zeros((20, 5));
//! let tsne = Tsne::new().with_perplexity(5.0).with_random_state(42);
//! let fitted = tsne.fit(&x, &()).unwrap();
//! assert_eq!(fitted.embedding().dim(), (20, 2));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

// ---------------------------------------------------------------------------
// Tsne (unfitted)
// ---------------------------------------------------------------------------

/// t-SNE configuration.
///
/// Holds hyperparameters for the t-SNE algorithm. Calling [`Fit::fit`]
/// computes the embedding and returns a [`FittedTsne`].
#[derive(Debug, Clone)]
pub struct Tsne {
    /// Number of embedding dimensions (default 2).
    n_components: usize,
    /// Perplexity target (default 30.0).
    perplexity: f64,
    /// Learning rate for gradient descent (default 200.0).
    learning_rate: f64,
    /// Number of gradient descent iterations (default 1000).
    n_iter: usize,
    /// Early exaggeration factor (default 12.0).
    early_exaggeration: f64,
    /// Barnes-Hut theta parameter controlling the accuracy/speed trade-off
    /// (default 0.5). A value of 0 disables the approximation.
    theta: f64,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl Tsne {
    /// Create a new `Tsne` with default parameters.
    ///
    /// Defaults: `n_components=2`, `perplexity=30.0`, `learning_rate=200.0`,
    /// `n_iter=1000`, `early_exaggeration=12.0`, `theta=0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            theta: 0.5,
            random_state: None,
        }
    }

    /// Set the number of embedding dimensions.
    #[must_use]
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set the perplexity.
    #[must_use]
    pub fn with_perplexity(mut self, perp: f64) -> Self {
        self.perplexity = perp;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of iterations.
    #[must_use]
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the early exaggeration factor.
    #[must_use]
    pub fn with_early_exaggeration(mut self, ex: f64) -> Self {
        self.early_exaggeration = ex;
        self
    }

    /// Set the Barnes-Hut theta parameter. Use 0 for exact computation.
    #[must_use]
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Set the random seed for reproducible results.
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

    /// Return the configured perplexity.
    #[must_use]
    pub fn perplexity(&self) -> f64 {
        self.perplexity
    }

    /// Return the configured learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Return the configured number of iterations.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Return the configured early exaggeration factor.
    #[must_use]
    pub fn early_exaggeration(&self) -> f64 {
        self.early_exaggeration
    }

    /// Return the configured theta.
    #[must_use]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

impl Default for Tsne {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedTsne
// ---------------------------------------------------------------------------

/// A fitted t-SNE model holding the learned embedding.
///
/// Created by calling [`Fit::fit`] on a [`Tsne`]. There is **no**
/// `Transform` implementation because t-SNE does not support out-of-sample
/// projection.
#[derive(Debug, Clone)]
pub struct FittedTsne {
    /// The embedding, shape `(n_samples, n_components)`.
    embedding_: Array2<f64>,
    /// Final Kullback-Leibler divergence.
    kl_divergence_: f64,
    /// Number of iterations actually performed.
    n_iter_: usize,
}

impl FittedTsne {
    /// The embedding coordinates, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding_
    }

    /// The Kullback-Leibler divergence at convergence.
    #[must_use]
    pub fn kl_divergence(&self) -> f64 {
        self.kl_divergence_
    }

    /// Number of iterations actually performed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Barnes-Hut spatial tree (quad-tree / oct-tree generalisation)
// ---------------------------------------------------------------------------

/// A spatial tree node for Barnes-Hut approximation.
///
/// This is a generic k-d tree where each internal node stores the centre of
/// mass and total count of the points in its region, and each leaf stores a
/// single point (or is empty). The tree recursively subdivides space into
/// 2^d children (d = number of embedding dimensions, typically 2 or 3).
struct BHTree {
    /// Centre of mass of all points in this node.
    center_of_mass: Vec<f64>,
    /// Number of points contained.
    count: usize,
    /// Width of this cell (side length).
    width: f64,
    /// Children nodes. Empty means leaf.
    children: Vec<BHTree>,
    /// Dimensionality.
    dim: usize,
}

impl BHTree {
    /// Create an empty tree node centred at `center` with given `width`.
    fn new(_center: Vec<f64>, width: f64, dim: usize) -> Self {
        Self {
            center_of_mass: vec![0.0; dim],
            count: 0,
            width,
            children: Vec::new(),
            dim,
        }
    }

    /// Insert a point into the tree.
    fn insert(&mut self, point: &[f64], center: &[f64]) {
        if self.count == 0 && self.children.is_empty() {
            // Empty leaf: store the point here.
            self.center_of_mass = point.to_vec();
            self.count = 1;
            return;
        }

        if self.children.is_empty() && self.count == 1 {
            // Occupied leaf: subdivide and re-insert existing point.
            self.subdivide(center);
            let old = self.center_of_mass.clone();
            self.reinsert(&old, center);
        }

        // Update aggregate centre of mass.
        let c = self.count as f64;
        for d in 0..self.dim {
            self.center_of_mass[d] = (self.center_of_mass[d] * c + point[d]) / (c + 1.0);
        }
        self.count += 1;

        // Insert into appropriate child.
        if !self.children.is_empty() {
            let idx = self.child_index(point, center);
            let child_center = self.child_center(center, idx);
            self.children[idx].insert(point, &child_center);
        }
    }

    /// Determine which child quadrant a point belongs to.
    fn child_index(&self, point: &[f64], center: &[f64]) -> usize {
        let mut idx = 0;
        for d in 0..self.dim {
            if point[d] >= center[d] {
                idx |= 1 << d;
            }
        }
        idx
    }

    /// Compute the centre of a child quadrant.
    fn child_center(&self, parent_center: &[f64], idx: usize) -> Vec<f64> {
        let quarter = self.width / 4.0;
        let mut c = parent_center.to_vec();
        for d in 0..self.dim {
            if idx & (1 << d) != 0 {
                c[d] += quarter;
            } else {
                c[d] -= quarter;
            }
        }
        c
    }

    /// Subdivide this node into 2^dim children.
    fn subdivide(&mut self, _center: &[f64]) {
        let n_children = 1 << self.dim;
        let child_width = self.width / 2.0;
        self.children.reserve_exact(n_children);
        for _ in 0..n_children {
            self.children
                .push(BHTree::new(vec![0.0; self.dim], child_width, self.dim));
        }
    }

    /// Re-insert a point that was stored in this node into a child.
    fn reinsert(&mut self, point: &[f64], center: &[f64]) {
        let idx = self.child_index(point, center);
        let child_center = self.child_center(center, idx);
        self.children[idx].insert(point, &child_center);
    }

    /// Compute the repulsive force on a point using the Barnes-Hut criterion.
    ///
    /// Accumulates into `force` (length = dim) and returns the sum of
    /// 1/(1 + ||y_i - y_j||^2) for normalisation (the Z factor).
    fn compute_repulsive(
        &self,
        point: &[f64],
        theta: f64,
        force: &mut [f64],
        center: &[f64],
    ) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Squared distance from point to this node's centre of mass.
        let mut dist_sq = 0.0;
        for d in 0..self.dim {
            let diff = point[d] - self.center_of_mass[d];
            dist_sq += diff * diff;
        }

        // If this is a leaf with a single point that is not the query itself
        // (dist > 0), or the cell is far enough away, use the approximation.
        if self.children.is_empty() || (self.width / dist_sq.sqrt() < theta && dist_sq > 1e-16) {
            let inv = 1.0 / (1.0 + dist_sq);
            let mult = (self.count as f64) * inv * inv;
            for d in 0..self.dim {
                force[d] += mult * (point[d] - self.center_of_mass[d]);
            }
            return (self.count as f64) * inv;
        }

        // Otherwise, recurse into children.
        let mut z = 0.0;
        for (idx, child) in self.children.iter().enumerate() {
            let child_center = self.child_center(center, idx);
            z += child.compute_repulsive(point, theta, force, &child_center);
        }
        z
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute pairwise squared Euclidean distances.
fn pairwise_sq_distances(x: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let d = x.ncols();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..d {
                let diff = x[[i, k]] - x[[j, k]];
                sq += diff * diff;
            }
            dist[i][j] = sq;
            dist[j][i] = sq;
        }
    }
    dist
}

/// Binary search for the sigma that produces the target perplexity for point i.
///
/// Returns a row of conditional probabilities `P(j|i)` for all j.
fn compute_pij_row(dist_sq: &[f64], i: usize, target_perp: f64) -> Vec<f64> {
    let n = dist_sq.len();
    let target_entropy = target_perp.ln();

    let mut lo = 1e-20_f64;
    let mut hi = 1e5_f64;
    let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

    let mut p = vec![0.0; n];

    for _iter in 0..100 {
        // Compute un-normalised probabilities.
        let mut sum = 0.0;
        for j in 0..n {
            if j == i {
                p[j] = 0.0;
            } else {
                p[j] = (-beta * dist_sq[j]).exp();
                sum += p[j];
            }
        }

        if sum < 1e-16 {
            // All probabilities near zero — widen the kernel.
            hi = beta;
            beta = f64::midpoint(lo, hi);
            continue;
        }

        // Normalise and compute entropy.
        let inv_sum = 1.0 / sum;
        let mut entropy = 0.0;
        for j in 0..n {
            p[j] *= inv_sum;
            if p[j] > 1e-16 {
                entropy -= p[j] * p[j].ln();
            }
        }

        let diff = entropy - target_entropy;
        if diff.abs() < 1e-5 {
            break;
        }
        if diff > 0.0 {
            // Entropy too high — need narrower kernel (larger beta).
            lo = beta;
        } else {
            // Entropy too low — need wider kernel (smaller beta).
            hi = beta;
        }
        beta = f64::midpoint(lo, hi);
    }

    p
}

/// Compute the symmetric joint probability matrix P.
fn compute_joint_probabilities(x: &Array2<f64>, perplexity: f64) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let dist = pairwise_sq_distances(x);

    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        let p_row = compute_pij_row(&dist[i], i, perplexity);
        for j in 0..n {
            p[i][j] = p_row[j];
        }
    }

    // Symmetrise: P_ij = (P_i|j + P_j|i) / (2n)
    let inv_2n = 1.0 / (2.0 * n as f64);
    for i in 0..n {
        for j in (i + 1)..n {
            let sym = (p[i][j] + p[j][i]) * inv_2n;
            // Clamp to small positive value for numerical stability.
            let sym = sym.max(1e-12);
            p[i][j] = sym;
            p[j][i] = sym;
        }
        p[i][i] = 0.0;
    }

    p
}

/// Compute the Kullback-Leibler divergence KL(P || Q).
fn compute_kl_divergence(p: &[Vec<f64>], y: &Array2<f64>) -> f64 {
    let n = p.len();
    // Compute Q (student-t distribution).
    let mut z = 0.0;
    let mut q_raw = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for d in 0..y.ncols() {
                let diff = y[[i, d]] - y[[j, d]];
                dist_sq += diff * diff;
            }
            let val = 1.0 / (1.0 + dist_sq);
            q_raw[i][j] = val;
            q_raw[j][i] = val;
            z += 2.0 * val;
        }
    }

    let mut kl = 0.0;
    if z > 0.0 {
        let inv_z = 1.0 / z;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p_ij = p[i][j];
                let q_ij = (q_raw[i][j] * inv_z).max(1e-16);
                if p_ij > 1e-16 {
                    kl += p_ij * (p_ij / q_ij).ln();
                }
            }
        }
    }
    kl
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for Tsne {
    type Fitted = FittedTsne;
    type Error = FerroError;

    /// Fit t-SNE by computing the low-dimensional embedding.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero,
    ///   perplexity is non-positive, or learning rate is non-positive.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2
    ///   samples or perplexity is too large for the number of samples.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedTsne, FerroError> {
        let n = x.nrows();

        // Validate parameters.
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "Tsne::fit requires at least 2 samples".into(),
            });
        }
        if self.perplexity <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "perplexity".into(),
                reason: "must be positive".into(),
            });
        }
        if self.perplexity >= n as f64 {
            return Err(FerroError::InvalidParameter {
                name: "perplexity".into(),
                reason: format!(
                    "perplexity ({}) must be less than n_samples ({})",
                    self.perplexity, n
                ),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.theta < 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "theta".into(),
                reason: "must be non-negative".into(),
            });
        }

        let dim = self.n_components;

        // Step 1: Compute joint probabilities.
        let p = compute_joint_probabilities(x, self.perplexity);

        // Step 2: Initialise embedding from a small Gaussian.
        let seed = self.random_state.unwrap_or(0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1e-4).unwrap();

        let mut y = Array2::<f64>::zeros((n, dim));
        for elem in &mut y {
            *elem = normal.sample(&mut rng);
        }

        // Gradient descent state.
        let mut gains = Array2::from_elem((n, dim), 1.0);
        let mut velocity = Array2::<f64>::zeros((n, dim));

        let early_exag_end = 250.min(self.n_iter);
        let lr = self.learning_rate;
        let use_bh = self.theta > 0.0 && dim <= 3;

        for iteration in 0..self.n_iter {
            let momentum = if iteration < early_exag_end { 0.5 } else { 0.8 };
            let exaggeration = if iteration < early_exag_end {
                self.early_exaggeration
            } else {
                1.0
            };

            // Compute gradient.
            let mut gradient = Array2::<f64>::zeros((n, dim));

            if use_bh {
                // Barnes-Hut: attractive forces from P, repulsive from tree.
                self.bh_gradient(&p, &y, exaggeration, &mut gradient);
            } else {
                // Exact gradient computation.
                exact_gradient(&p, &y, exaggeration, &mut gradient);
            }

            // Update with momentum and adaptive gains.
            for i in 0..n {
                for d in 0..dim {
                    let g = gradient[[i, d]];
                    let v = velocity[[i, d]];
                    // Adaptive gain: increase if gradient and velocity disagree.
                    if (g > 0.0) == (v > 0.0) {
                        gains[[i, d]] = (gains[[i, d]] * 0.8_f64).max(0.01);
                    } else {
                        gains[[i, d]] += 0.2;
                    }
                    velocity[[i, d]] = momentum * v - lr * gains[[i, d]] * g;
                    y[[i, d]] += velocity[[i, d]];
                }
            }

            // Centre the embedding.
            for d in 0..dim {
                let mean: f64 = y.column(d).sum() / n as f64;
                for i in 0..n {
                    y[[i, d]] -= mean;
                }
            }
        }

        let kl = compute_kl_divergence(&p, &y);

        Ok(FittedTsne {
            embedding_: y,
            kl_divergence_: kl,
            n_iter_: self.n_iter,
        })
    }
}

impl Tsne {
    /// Compute the gradient using Barnes-Hut approximation.
    fn bh_gradient(
        &self,
        p: &[Vec<f64>],
        y: &Array2<f64>,
        exaggeration: f64,
        gradient: &mut Array2<f64>,
    ) {
        let n = y.nrows();
        let dim = y.ncols();

        // Build the spatial tree.
        let mut min_vals = vec![f64::MAX; dim];
        let mut max_vals = vec![f64::MIN; dim];
        for i in 0..n {
            for d in 0..dim {
                min_vals[d] = min_vals[d].min(y[[i, d]]);
                max_vals[d] = max_vals[d].max(y[[i, d]]);
            }
        }

        let mut width = 0.0_f64;
        let mut tree_center = vec![0.0; dim];
        for d in 0..dim {
            let range = max_vals[d] - min_vals[d];
            width = width.max(range);
            tree_center[d] = f64::midpoint(min_vals[d], max_vals[d]);
        }
        width *= 1.01; // Small padding.

        let mut tree = BHTree::new(vec![0.0; dim], width, dim);
        for i in 0..n {
            let point: Vec<f64> = (0..dim).map(|d| y[[i, d]]).collect();
            tree.insert(&point, &tree_center);
        }

        // Attractive forces: iterate over non-zero P entries.
        // Also compute repulsive forces via tree.
        let mut repulsive = Array2::<f64>::zeros((n, dim));
        let mut z_sum = 0.0;

        for i in 0..n {
            let point: Vec<f64> = (0..dim).map(|d| y[[i, d]]).collect();
            let mut rep_force = vec![0.0; dim];
            let z_i = tree.compute_repulsive(&point, self.theta, &mut rep_force, &tree_center);
            z_sum += z_i;
            for d in 0..dim {
                repulsive[[i, d]] = rep_force[d];
            }
        }

        // Attractive gradient.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p_ij = p[i][j] * exaggeration;
                if p_ij < 1e-16 {
                    continue;
                }
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = y[[i, d]] - y[[j, d]];
                    dist_sq += diff * diff;
                }
                let inv = 1.0 / (1.0 + dist_sq);
                for d in 0..dim {
                    gradient[[i, d]] += 4.0 * p_ij * inv * (y[[i, d]] - y[[j, d]]);
                }
            }
        }

        // Repulsive gradient (normalised by Z).
        if z_sum > 0.0 {
            let inv_z = 1.0 / z_sum;
            for i in 0..n {
                for d in 0..dim {
                    gradient[[i, d]] -= 4.0 * inv_z * repulsive[[i, d]];
                }
            }
        }
    }
}

/// Exact gradient computation (no Barnes-Hut).
fn exact_gradient(p: &[Vec<f64>], y: &Array2<f64>, exaggeration: f64, gradient: &mut Array2<f64>) {
    let n = y.nrows();
    let dim = y.ncols();

    // Compute Q distribution (Student-t).
    let mut q_raw = vec![vec![0.0; n]; n];
    let mut z = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for d in 0..dim {
                let diff = y[[i, d]] - y[[j, d]];
                dist_sq += diff * diff;
            }
            let val = 1.0 / (1.0 + dist_sq);
            q_raw[i][j] = val;
            q_raw[j][i] = val;
            z += 2.0 * val;
        }
    }

    if z < 1e-16 {
        return;
    }
    let inv_z = 1.0 / z;

    // Gradient: 4 * sum_j (p_ij - q_ij) * (y_i - y_j) / (1 + ||y_i - y_j||^2)
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let p_ij = p[i][j] * exaggeration;
            let q_ij = q_raw[i][j] * inv_z;
            let inv = q_raw[i][j]; // = 1/(1+dist_sq)
            let mult = 4.0 * (p_ij - q_ij) * inv;
            for d in 0..dim {
                gradient[[i, d]] += mult * (y[[i, d]] - y[[j, d]]);
            }
        }
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

    /// Generate small blobs dataset: 3 clusters of 10 points each in 5D.
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
    fn test_tsne_basic_shape() {
        let x = Array2::<f64>::zeros((20, 5));
        let tsne = Tsne::new()
            .with_perplexity(5.0)
            .with_n_iter(50)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (20, 2));
    }

    #[test]
    fn test_tsne_3d_embedding() {
        let x = Array2::<f64>::zeros((15, 4));
        let tsne = Tsne::new()
            .with_n_components(3)
            .with_perplexity(4.0)
            .with_n_iter(50)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().ncols(), 3);
    }

    #[test]
    fn test_tsne_separates_clusters() {
        let (x, labels) = make_blobs(42);
        let tsne = Tsne::new()
            .with_perplexity(5.0)
            .with_n_iter(500)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
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
            // Majority vote of 3 nearest.
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
            "t-SNE k-NN accuracy should be > 80%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_tsne_kl_divergence_non_negative() {
        let (x, _) = make_blobs(42);
        let tsne = Tsne::new()
            .with_perplexity(5.0)
            .with_n_iter(100)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
        assert!(
            fitted.kl_divergence() >= 0.0,
            "KL divergence should be non-negative, got {}",
            fitted.kl_divergence()
        );
    }

    #[test]
    fn test_tsne_reproducibility() {
        let x = Array2::<f64>::from_shape_fn((10, 3), |(i, j)| (i + j) as f64);
        let t1 = Tsne::new()
            .with_perplexity(3.0)
            .with_n_iter(50)
            .with_random_state(42);
        let t2 = Tsne::new()
            .with_perplexity(3.0)
            .with_n_iter(50)
            .with_random_state(42);
        let f1 = t1.fit(&x, &()).unwrap();
        let f2 = t2.fit(&x, &()).unwrap();
        for (a, b) in f1.embedding().iter().zip(f2.embedding().iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "embeddings should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_tsne_exact_mode() {
        // theta=0 forces exact computation
        let x = Array2::<f64>::from_shape_fn((10, 3), |(i, j)| (i + j) as f64);
        let tsne = Tsne::new()
            .with_theta(0.0)
            .with_perplexity(3.0)
            .with_n_iter(50)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
        assert_eq!(fitted.embedding().dim(), (10, 2));
    }

    #[test]
    fn test_tsne_invalid_n_components_zero() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new().with_n_components(0);
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_invalid_perplexity_zero() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new().with_perplexity(0.0);
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_perplexity_too_large() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new().with_perplexity(10.0);
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_invalid_learning_rate() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new().with_learning_rate(-1.0);
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_invalid_theta() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new().with_theta(-0.1);
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_insufficient_samples() {
        let x = Array2::<f64>::zeros((1, 3));
        let tsne = Tsne::new();
        assert!(tsne.fit(&x, &()).is_err());
    }

    #[test]
    fn test_tsne_getters() {
        let tsne = Tsne::new()
            .with_n_components(3)
            .with_perplexity(20.0)
            .with_learning_rate(100.0)
            .with_n_iter(500)
            .with_early_exaggeration(8.0)
            .with_theta(0.3)
            .with_random_state(99);
        assert_eq!(tsne.n_components(), 3);
        assert!((tsne.perplexity() - 20.0).abs() < 1e-10);
        assert!((tsne.learning_rate() - 100.0).abs() < 1e-10);
        assert_eq!(tsne.n_iter(), 500);
        assert!((tsne.early_exaggeration() - 8.0).abs() < 1e-10);
        assert!((tsne.theta() - 0.3).abs() < 1e-10);
        assert_eq!(tsne.random_state(), Some(99));
    }

    #[test]
    fn test_tsne_default() {
        let tsne = Tsne::default();
        assert_eq!(tsne.n_components(), 2);
        assert!((tsne.perplexity() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_tsne_n_iter_recorded() {
        let x = Array2::<f64>::zeros((10, 3));
        let tsne = Tsne::new()
            .with_perplexity(3.0)
            .with_n_iter(50)
            .with_random_state(42);
        let fitted = tsne.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_iter(), 50);
    }
}
