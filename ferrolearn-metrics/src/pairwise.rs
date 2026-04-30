//! Pairwise distance metrics.
//!
//! This module computes distance matrices between rows of two 2-D arrays.
//! Every function returns an `(n, m)` matrix where entry `[i, j]` is the
//! distance between row `i` of the first input and row `j` of the second.
//!
//! Supported distance metrics:
//!
//! - [`euclidean_distances`] — L2 (Euclidean) distance, optimised via the
//!   identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b`
//! - [`manhattan_distances`] — L1 (Manhattan / city-block) distance
//! - [`cosine_distances`] — `1 - cosine_similarity`
//! - [`pairwise_distances`] — dispatcher that selects the above via the
//!   [`Metric`] enum

use ferrolearn_core::FerroError;
use ndarray::Array2;
use num_traits::Float;

/// Distance metric selector for [`pairwise_distances`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Cosine distance: `1 - cosine_similarity`.
    Cosine,
    /// Chebyshev (L-infinity) distance.
    Chebyshev,
}

/// Trait for computing pairwise distances between two matrices.
///
/// This is the dyn-friendly equivalent of scikit-learn's `DistanceMetric`
/// class. Implementors take two `(n_a, d) × (n_b, d)` matrices and return
/// the `(n_a, n_b)` distance matrix.
pub trait DistanceMetric<F>: Send + Sync
where
    F: Float + Send + Sync + 'static,
{
    /// Compute the pairwise distance matrix between every row of `x` and
    /// every row of `y`.
    fn pairwise(&self, x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>;

    /// Convenience: compute the distance from a single point `a` to a single
    /// point `b`. Default implementation calls [`pairwise`] on `1 × d`
    /// matrices and returns the single entry.
    fn distance(&self, a: &ndarray::Array1<F>, b: &ndarray::Array1<F>) -> Result<F, FerroError> {
        let mut x = Array2::<F>::zeros((1, a.len()));
        let mut y = Array2::<F>::zeros((1, b.len()));
        for i in 0..a.len() {
            x[[0, i]] = a[i];
        }
        for i in 0..b.len() {
            y[[0, i]] = b[i];
        }
        let m = self.pairwise(&x, &y)?;
        Ok(m[[0, 0]])
    }
}

impl<F> DistanceMetric<F> for Metric
where
    F: Float + Send + Sync + 'static,
{
    fn pairwise(&self, x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError> {
        pairwise_distances(x, y, *self)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that two matrices have the same number of columns (features).
fn check_feature_dim<F: Float>(
    x: &Array2<F>,
    y: &Array2<F>,
    context: &str,
) -> Result<(), FerroError> {
    if x.ncols() != y.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows(), x.ncols()],
            actual: vec![y.nrows(), y.ncols()],
            context: format!(
                "{context}: feature dimensions must match (X has {} cols, Y has {} cols)",
                x.ncols(),
                y.ncols()
            ),
        });
    }
    Ok(())
}

/// Validate that the matrices are non-empty.
fn check_non_empty<F: Float>(
    x: &Array2<F>,
    y: &Array2<F>,
    context: &str,
) -> Result<(), FerroError> {
    if x.nrows() == 0 || y.nrows() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: format!("{context}: both X and Y must have at least one row"),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: format!("{context}: feature dimension must be at least 1"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute pairwise Euclidean distances using the optimised identity.
///
/// Uses the identity `||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b` which
/// replaces the inner-loop subtraction with a matrix multiplication, offering
/// better cache locality for large inputs.
///
/// Negative squared distances (arising from floating-point rounding) are
/// clamped to zero before taking the square root.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::euclidean_distances;
/// use ndarray::array;
///
/// let x = array![[0.0_f64, 0.0], [1.0, 0.0]];
/// let y = array![[0.0_f64, 0.0], [0.0, 1.0]];
/// let d = euclidean_distances(&x, &y).unwrap();
/// assert!((d[[0, 0]] - 0.0).abs() < 1e-10);
/// assert!((d[[0, 1]] - 1.0).abs() < 1e-10);
/// assert!((d[[1, 0]] - 1.0).abs() < 1e-10);
/// assert!((d[[1, 1]] - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
pub fn euclidean_distances<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "euclidean_distances")?;
    check_non_empty(x, y, "euclidean_distances")?;

    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();
    let two = F::from(2.0).unwrap();

    // Precompute ||x_i||^2 and ||y_j||^2.
    let x_sq: Vec<F> = (0..n)
        .map(|i| (0..d).fold(F::zero(), |acc, k| acc + x[[i, k]] * x[[i, k]]))
        .collect();
    let y_sq: Vec<F> = (0..m)
        .map(|j| (0..d).fold(F::zero(), |acc, k| acc + y[[j, k]] * y[[j, k]]))
        .collect();

    let mut result = Array2::<F>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            // dot product x_i . y_j
            let dot = (0..d).fold(F::zero(), |acc, k| acc + x[[i, k]] * y[[j, k]]);
            let sq_dist = (x_sq[i] + y_sq[j] - two * dot).max(F::zero());
            result[[i, j]] = sq_dist.sqrt();
        }
    }

    Ok(result)
}

/// Compute pairwise Manhattan (L1) distances.
///
/// For each pair of rows `(x_i, y_j)`, the Manhattan distance is
/// `sum_k |x_i[k] - y_j[k]|`.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::manhattan_distances;
/// use ndarray::array;
///
/// let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
/// let y = array![[1.0_f64, 1.0]];
/// let d = manhattan_distances(&x, &y).unwrap();
/// assert!((d[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((d[[1, 0]] - 0.0).abs() < 1e-10);
/// ```
pub fn manhattan_distances<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "manhattan_distances")?;
    check_non_empty(x, y, "manhattan_distances")?;

    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    let mut result = Array2::<F>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let dist = (0..d).fold(F::zero(), |acc, k| acc + (x[[i, k]] - y[[j, k]]).abs());
            result[[i, j]] = dist;
        }
    }

    Ok(result)
}

/// Compute pairwise cosine distances.
///
/// Cosine distance is defined as `1 - cosine_similarity`, where
/// `cosine_similarity(a, b) = a . b / (||a|| * ||b||)`.
///
/// If either vector has zero norm, the cosine distance is defined as `1.0`.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::cosine_distances;
/// use ndarray::array;
///
/// let x = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let y = array![[1.0_f64, 0.0]];
/// let d = cosine_distances(&x, &y).unwrap();
/// assert!((d[[0, 0]] - 0.0).abs() < 1e-10); // same direction
/// assert!((d[[1, 0]] - 1.0).abs() < 1e-10); // orthogonal
/// ```
pub fn cosine_distances<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "cosine_distances")?;
    check_non_empty(x, y, "cosine_distances")?;

    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    // Precompute norms.
    let x_norms: Vec<F> = (0..n)
        .map(|i| {
            (0..d)
                .fold(F::zero(), |acc, k| acc + x[[i, k]] * x[[i, k]])
                .sqrt()
        })
        .collect();
    let y_norms: Vec<F> = (0..m)
        .map(|j| {
            (0..d)
                .fold(F::zero(), |acc, k| acc + y[[j, k]] * y[[j, k]])
                .sqrt()
        })
        .collect();

    let mut result = Array2::<F>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            if x_norms[i] == F::zero() || y_norms[j] == F::zero() {
                result[[i, j]] = F::one();
            } else {
                let dot = (0..d).fold(F::zero(), |acc, k| acc + x[[i, k]] * y[[j, k]]);
                let cos_sim = dot / (x_norms[i] * y_norms[j]);
                // Clamp to avoid floating-point rounding pushing above 1.
                let cos_sim_clamped = cos_sim.min(F::one()).max(-F::one());
                result[[i, j]] = F::one() - cos_sim_clamped;
            }
        }
    }

    Ok(result)
}

/// Compute pairwise Chebyshev (L-infinity) distances.
///
/// For each pair of rows `(x_i, y_j)`, the Chebyshev distance is
/// `max_k |x_i[k] - y_j[k]|`.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::chebyshev_distances;
/// use ndarray::array;
///
/// let x = array![[0.0_f64, 0.0], [3.0, 4.0]];
/// let y = array![[1.0_f64, 2.0]];
/// let d = chebyshev_distances(&x, &y).unwrap();
/// assert!((d[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((d[[1, 0]] - 2.0).abs() < 1e-10);
/// ```
fn chebyshev_distances_inner<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();

    let mut result = Array2::<F>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let dist = (0..d).fold(F::zero(), |max_val, k| {
                let diff = (x[[i, k]] - y[[j, k]]).abs();
                if diff > max_val { diff } else { max_val }
            });
            result[[i, j]] = dist;
        }
    }

    Ok(result)
}

/// Compute pairwise Chebyshev (L-infinity) distances.
///
/// For each pair of rows `(x_i, y_j)`, the Chebyshev distance is
/// `max_k |x_i[k] - y_j[k]|`.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::chebyshev_distances;
/// use ndarray::array;
///
/// let x = array![[0.0_f64, 0.0], [3.0, 4.0]];
/// let y = array![[1.0_f64, 2.0]];
/// let d = chebyshev_distances(&x, &y).unwrap();
/// assert!((d[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((d[[1, 0]] - 2.0).abs() < 1e-10);
/// ```
pub fn chebyshev_distances<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "chebyshev_distances")?;
    check_non_empty(x, y, "chebyshev_distances")?;
    chebyshev_distances_inner(x, y)
}

/// Compute a pairwise distance matrix using the specified metric.
///
/// Returns an `(n, m)` matrix where entry `[i, j]` is the distance between
/// row `i` of `x` and row `j` of `y` according to the chosen [`Metric`].
///
/// # Arguments
///
/// * `x`      — array of shape `(n, d)`.
/// * `y`      — array of shape `(m, d)`.
/// * `metric` — the distance metric to use.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::{pairwise_distances, Metric};
/// use ndarray::array;
///
/// let x = array![[0.0_f64, 0.0], [1.0, 0.0]];
/// let y = array![[0.0_f64, 0.0], [0.0, 1.0]];
/// let d = pairwise_distances(&x, &y, Metric::Euclidean).unwrap();
/// assert!((d[[0, 0]] - 0.0).abs() < 1e-10);
/// assert!((d[[1, 0]] - 1.0).abs() < 1e-10);
/// ```
pub fn pairwise_distances<F>(
    x: &Array2<F>,
    y: &Array2<F>,
    metric: Metric,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    match metric {
        Metric::Euclidean => euclidean_distances(x, y),
        Metric::Manhattan => manhattan_distances(x, y),
        Metric::Cosine => cosine_distances(x, y),
        Metric::Chebyshev => chebyshev_distances(x, y),
    }
}

/// Compute pairwise Euclidean distances with NaN handling.
///
/// For each pair of rows `(x_i, y_j)`, the distance is computed over the
/// features where both values are non-NaN, then scaled by
/// `sqrt(n_features / n_valid)` to compensate for the missing features.
///
/// If all features are NaN for a given pair, the distance is `NaN`.
///
/// # Arguments
///
/// * `x` — array of shape `(n, d)`.
/// * `y` — array of shape `(m, d)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
/// numbers of columns.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::pairwise::nan_euclidean_distances;
/// use ndarray::array;
///
/// let x = array![[f64::NAN, 0.0], [1.0, 0.0]];
/// let y = array![[0.0_f64, 0.0]];
/// let d = nan_euclidean_distances(&x, &y).unwrap();
/// // x[0] vs y[0]: only feature 1 is valid, dist = |0-0| * sqrt(2/1) = 0
/// assert!((d[[0, 0]] - 0.0).abs() < 1e-10);
/// // x[1] vs y[0]: both valid, dist = sqrt(1+0) = 1
/// assert!((d[[1, 0]] - 1.0).abs() < 1e-10);
/// ```
pub fn nan_euclidean_distances<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "nan_euclidean_distances")?;
    check_non_empty(x, y, "nan_euclidean_distances")?;

    let n = x.nrows();
    let m = y.nrows();
    let d = x.ncols();
    let d_f = F::from(d).unwrap();

    let mut result = Array2::<F>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut sq_sum = F::zero();
            let mut n_valid = 0usize;
            for k in 0..d {
                let xi = x[[i, k]];
                let yj = y[[j, k]];
                if xi.is_nan() || yj.is_nan() {
                    continue;
                }
                let diff = xi - yj;
                sq_sum = sq_sum + diff * diff;
                n_valid += 1;
            }
            if n_valid == 0 {
                result[[i, j]] = F::nan();
            } else {
                let scale = d_f / F::from(n_valid).unwrap();
                result[[i, j]] = (sq_sum * scale).sqrt();
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// pairwise_distances_argmin / argmin_min / pairwise_kernels
// ---------------------------------------------------------------------------

/// Kernel function for [`pairwise_kernels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PairwiseKernel<F> {
    /// `K(x, y) = x . y`
    Linear,
    /// `K(x, y) = (gamma * x . y + coef0)^degree`
    Polynomial {
        /// Degree of the polynomial.
        degree: u32,
        /// Multiplier on the dot product (default `1 / n_features`).
        gamma: F,
        /// Additive bias term.
        coef0: F,
    },
    /// `K(x, y) = exp(-gamma * ||x - y||^2)`
    Rbf {
        /// Bandwidth parameter (default `1 / n_features`).
        gamma: F,
    },
    /// `K(x, y) = tanh(gamma * x . y + coef0)`
    Sigmoid {
        /// Multiplier on the dot product.
        gamma: F,
        /// Additive bias term.
        coef0: F,
    },
    /// `K(x, y) = exp(-gamma * ||x - y||_1)`
    Laplacian {
        /// Bandwidth parameter.
        gamma: F,
    },
}

/// For each row in `X`, return the index of the closest row in `Y` under the
/// given metric.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `X` and `Y` have different feature
/// dimensions.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
pub fn pairwise_distances_argmin<F>(
    x: &Array2<F>,
    y: &Array2<F>,
    metric: Metric,
) -> Result<ndarray::Array1<usize>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let d = pairwise_distances(x, y, metric)?;
    let n = d.nrows();
    let mut out = ndarray::Array1::<usize>::zeros(n);
    for i in 0..n {
        let row = d.row(i);
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v < best_val {
                best_val = v;
                best_idx = j;
            }
        }
        out[i] = best_idx;
    }
    Ok(out)
}

/// Like [`pairwise_distances_argmin`] but also returns the minimum distance
/// for each row.
///
/// # Errors
///
/// Returns the same errors as [`pairwise_distances_argmin`].
pub fn pairwise_distances_argmin_min<F>(
    x: &Array2<F>,
    y: &Array2<F>,
    metric: Metric,
) -> Result<(ndarray::Array1<usize>, ndarray::Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let d = pairwise_distances(x, y, metric)?;
    let n = d.nrows();
    let mut idx = ndarray::Array1::<usize>::zeros(n);
    let mut mins = ndarray::Array1::<F>::from_elem(n, F::zero());
    for i in 0..n {
        let row = d.row(i);
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (j, &v) in row.iter().enumerate().skip(1) {
            if v < best_val {
                best_val = v;
                best_idx = j;
            }
        }
        idx[i] = best_idx;
        mins[i] = best_val;
    }
    Ok((idx, mins))
}

/// Compute the pairwise kernel matrix between rows of `X` and rows of `Y`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `X` and `Y` have different feature
/// dimensions.
/// Returns [`FerroError::InsufficientSamples`] if either matrix is empty.
pub fn pairwise_kernels<F>(
    x: &Array2<F>,
    y: &Array2<F>,
    kernel: PairwiseKernel<F>,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_feature_dim(x, y, "pairwise_kernels")?;
    check_non_empty(x, y, "pairwise_kernels")?;
    let n = x.nrows();
    let m = y.nrows();
    let mut out = Array2::<F>::zeros((n, m));
    match kernel {
        PairwiseKernel::Linear => {
            for i in 0..n {
                for j in 0..m {
                    let mut s = F::zero();
                    for k in 0..x.ncols() {
                        s = s + x[[i, k]] * y[[j, k]];
                    }
                    out[[i, j]] = s;
                }
            }
        }
        PairwiseKernel::Polynomial {
            degree,
            gamma,
            coef0,
        } => {
            for i in 0..n {
                for j in 0..m {
                    let mut s = F::zero();
                    for k in 0..x.ncols() {
                        s = s + x[[i, k]] * y[[j, k]];
                    }
                    let v = gamma * s + coef0;
                    out[[i, j]] = v.powi(degree as i32);
                }
            }
        }
        PairwiseKernel::Rbf { gamma } => {
            for i in 0..n {
                for j in 0..m {
                    let mut s = F::zero();
                    for k in 0..x.ncols() {
                        let d = x[[i, k]] - y[[j, k]];
                        s = s + d * d;
                    }
                    out[[i, j]] = (-gamma * s).exp();
                }
            }
        }
        PairwiseKernel::Sigmoid { gamma, coef0 } => {
            for i in 0..n {
                for j in 0..m {
                    let mut s = F::zero();
                    for k in 0..x.ncols() {
                        s = s + x[[i, k]] * y[[j, k]];
                    }
                    out[[i, j]] = (gamma * s + coef0).tanh();
                }
            }
        }
        PairwiseKernel::Laplacian { gamma } => {
            for i in 0..n {
                for j in 0..m {
                    let mut s = F::zero();
                    for k in 0..x.ncols() {
                        s = s + (x[[i, k]] - y[[j, k]]).abs();
                    }
                    out[[i, j]] = (-gamma * s).exp();
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // euclidean_distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_euclidean_identity() {
        let x = array![[1.0_f64, 2.0, 3.0]];
        let d = euclidean_distances(&x, &x).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_basic() {
        let x = array![[0.0_f64, 0.0], [3.0, 0.0]];
        let y = array![[0.0_f64, 4.0]];
        let d = euclidean_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 0]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_symmetry() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let y = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let d_xy = euclidean_distances(&x, &y).unwrap();
        let d_yx = euclidean_distances(&y, &x).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(d_xy[[i, j]], d_yx[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_euclidean_shape_mismatch() {
        let x = array![[1.0_f64, 2.0]];
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(euclidean_distances(&x, &y).is_err());
    }

    #[test]
    fn test_euclidean_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(euclidean_distances(&x, &y).is_err());
    }

    #[test]
    fn test_euclidean_f32() {
        let x = array![[0.0_f32, 0.0], [1.0, 0.0]];
        let y = array![[0.0_f32, 0.0]];
        let d = euclidean_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 0.0_f32, epsilon = 1e-6);
        assert_abs_diff_eq!(d[[1, 0]], 1.0_f32, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // manhattan_distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_manhattan_identity() {
        let x = array![[1.0_f64, 2.0, 3.0]];
        let d = manhattan_distances(&x, &x).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_manhattan_basic() {
        let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let y = array![[1.0_f64, 1.0]];
        let d = manhattan_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_manhattan_shape_mismatch() {
        let x = array![[1.0_f64, 2.0]];
        let y = array![[1.0_f64]];
        assert!(manhattan_distances(&x, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // cosine_distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_same_direction() {
        let x = array![[1.0_f64, 0.0]];
        let y = array![[2.0_f64, 0.0]];
        let d = cosine_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let x = array![[1.0_f64, 0.0]];
        let y = array![[0.0_f64, 1.0]];
        let d = cosine_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let x = array![[1.0_f64, 0.0]];
        let y = array![[-1.0_f64, 0.0]];
        let d = cosine_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let x = array![[0.0_f64, 0.0]];
        let y = array![[1.0_f64, 0.0]];
        let d = cosine_distances(&x, &y).unwrap();
        // Zero vector => cosine distance = 1.0 by convention.
        assert_abs_diff_eq!(d[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_shape_mismatch() {
        let x = array![[1.0_f64, 2.0]];
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(cosine_distances(&x, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // chebyshev_distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_chebyshev_basic() {
        let x = array![[0.0_f64, 0.0], [3.0, 4.0]];
        let y = array![[1.0_f64, 2.0]];
        let d = chebyshev_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 0]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_identity() {
        let x = array![[1.0_f64, 5.0, 3.0]];
        let d = chebyshev_distances(&x, &x).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_shape_mismatch() {
        let x = array![[1.0_f64, 2.0]];
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(chebyshev_distances(&x, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // pairwise_distances dispatcher
    // -----------------------------------------------------------------------

    #[test]
    fn test_pairwise_euclidean_dispatch() {
        let x = array![[0.0_f64, 0.0], [1.0, 0.0]];
        let y = array![[0.0_f64, 0.0]];
        let d_direct = euclidean_distances(&x, &y).unwrap();
        let d_dispatch = pairwise_distances(&x, &y, Metric::Euclidean).unwrap();
        assert_eq!(d_direct, d_dispatch);
    }

    #[test]
    fn test_pairwise_manhattan_dispatch() {
        let x = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let y = array![[1.0_f64, 1.0]];
        let d_direct = manhattan_distances(&x, &y).unwrap();
        let d_dispatch = pairwise_distances(&x, &y, Metric::Manhattan).unwrap();
        assert_eq!(d_direct, d_dispatch);
    }

    #[test]
    fn test_pairwise_cosine_dispatch() {
        let x = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let y = array![[1.0_f64, 0.0]];
        let d_direct = cosine_distances(&x, &y).unwrap();
        let d_dispatch = pairwise_distances(&x, &y, Metric::Cosine).unwrap();
        assert_eq!(d_direct, d_dispatch);
    }

    #[test]
    fn test_pairwise_chebyshev_dispatch() {
        let x = array![[0.0_f64, 0.0], [3.0, 4.0]];
        let y = array![[1.0_f64, 2.0]];
        let d_direct = chebyshev_distances(&x, &y).unwrap();
        let d_dispatch = pairwise_distances(&x, &y, Metric::Chebyshev).unwrap();
        assert_eq!(d_direct, d_dispatch);
    }

    #[test]
    fn test_pairwise_multi_row_multi_col() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[7.0_f64, 8.0], [9.0, 10.0]];
        let d = pairwise_distances(&x, &y, Metric::Manhattan).unwrap();
        assert_eq!(d.shape(), &[3, 2]);
        // x[0] vs y[0]: |1-7| + |2-8| = 12
        assert_abs_diff_eq!(d[[0, 0]], 12.0, epsilon = 1e-10);
        // x[2] vs y[1]: |5-9| + |6-10| = 8
        assert_abs_diff_eq!(d[[2, 1]], 8.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // nan_euclidean_distances
    // -----------------------------------------------------------------------

    #[test]
    fn test_nan_euclidean_no_nans() {
        // Without NaN, should match euclidean_distances exactly
        let x = array![[0.0_f64, 0.0], [3.0, 0.0]];
        let y = array![[0.0_f64, 4.0]];
        let d_nan = nan_euclidean_distances(&x, &y).unwrap();
        let d_ref = euclidean_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d_nan[[0, 0]], d_ref[[0, 0]], epsilon = 1e-10);
        assert_abs_diff_eq!(d_nan[[1, 0]], d_ref[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn test_nan_euclidean_with_nan() {
        let x = array![[f64::NAN, 0.0], [1.0, 0.0]];
        let y = array![[0.0_f64, 0.0]];
        let d = nan_euclidean_distances(&x, &y).unwrap();
        // x[0] vs y[0]: only feature 1 valid => sq_dist = 0, scale = 2/1 => sqrt(0) = 0
        assert_abs_diff_eq!(d[[0, 0]], 0.0, epsilon = 1e-10);
        // x[1] vs y[0]: both valid => sqrt((1-0)^2 + (0-0)^2) = 1
        assert_abs_diff_eq!(d[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_euclidean_scaling() {
        // x=[NaN, 3], y=[0, 0] => only feature 1 valid
        // sq_dist = 9, n_valid = 1, n_features = 2
        // result = sqrt(9 * 2/1) = sqrt(18) = 3*sqrt(2)
        let x = array![[f64::NAN, 3.0_f64]];
        let y = array![[0.0_f64, 0.0]];
        let d = nan_euclidean_distances(&x, &y).unwrap();
        assert_abs_diff_eq!(d[[0, 0]], 3.0 * 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_nan_euclidean_all_nan() {
        let x = array![[f64::NAN, f64::NAN]];
        let y = array![[1.0_f64, 2.0]];
        let d = nan_euclidean_distances(&x, &y).unwrap();
        assert!(d[[0, 0]].is_nan());
    }

    #[test]
    fn test_nan_euclidean_shape_mismatch() {
        let x = array![[1.0_f64, 2.0]];
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(nan_euclidean_distances(&x, &y).is_err());
    }

    #[test]
    fn test_nan_euclidean_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = array![[1.0_f64, 2.0, 3.0]];
        assert!(nan_euclidean_distances(&x, &y).is_err());
    }
}
