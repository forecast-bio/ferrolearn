//! KNN imputer: fill missing (NaN) values using K-nearest neighbors.
//!
//! [`KNNImputer`] replaces each missing value by computing the weighted average
//! of the corresponding feature from the `k` nearest non-missing neighbors.
//! Distance is computed only on the features that are non-missing in both the
//! query row and the candidate row (partial Euclidean distance).
//!
//! # Weighting
//!
//! - [`KNNWeights::Uniform`] — all neighbors contribute equally.
//! - [`KNNWeights::Distance`] — neighbors are weighted by the inverse of their
//!   distance (closer neighbors contribute more).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// KNNWeights
// ---------------------------------------------------------------------------

/// Weighting strategy for k-nearest neighbor imputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KNNWeights {
    /// All neighbors contribute equally.
    Uniform,
    /// Neighbors contribute proportionally to the inverse of their distance.
    Distance,
}

// ---------------------------------------------------------------------------
// KNNImputer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted KNN imputer.
///
/// Calling [`Fit::fit`] stores the training data and returns a
/// [`FittedKNNImputer`] that can impute missing values in new data.
///
/// # Parameters
///
/// - `n_neighbors` — number of nearest neighbors to use (default 5).
/// - `weights` — how to weight neighbor contributions (default `Uniform`).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::knn_imputer::{KNNImputer, KNNWeights};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN]];
/// let fitted = imputer.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert!(!out[[2, 1]].is_nan());
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct KNNImputer<F> {
    /// Number of nearest neighbors to use.
    n_neighbors: usize,
    /// Weighting strategy.
    weights: KNNWeights,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> KNNImputer<F> {
    /// Create a new `KNNImputer` with the given parameters.
    pub fn new(n_neighbors: usize, weights: KNNWeights) -> Self {
        Self {
            n_neighbors,
            weights,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of neighbors.
    #[must_use]
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Return the weighting strategy.
    #[must_use]
    pub fn weights(&self) -> KNNWeights {
        self.weights
    }
}

impl<F: Float + Send + Sync + 'static> Default for KNNImputer<F> {
    fn default() -> Self {
        Self::new(5, KNNWeights::Uniform)
    }
}

// ---------------------------------------------------------------------------
// FittedKNNImputer
// ---------------------------------------------------------------------------

/// A fitted KNN imputer holding the training data used for neighbor lookup.
///
/// Created by calling [`Fit::fit`] on a [`KNNImputer`].
#[derive(Debug, Clone)]
pub struct FittedKNNImputer<F> {
    /// The training data (used for neighbor lookup).
    train_data: Array2<F>,
    /// Number of neighbors.
    n_neighbors: usize,
    /// Weighting strategy.
    weights: KNNWeights,
}

impl<F: Float + Send + Sync + 'static> FittedKNNImputer<F> {
    /// Return the number of training samples.
    #[must_use]
    pub fn n_train_samples(&self) -> usize {
        self.train_data.nrows()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute partial Euclidean distance between two rows, using only features
/// that are non-missing in both rows.
///
/// Returns `(distance, n_valid)`. If no valid features exist, returns
/// `(F::infinity(), 0)`.
fn partial_euclidean_distance<F: Float>(row_a: &[F], row_b: &[F]) -> (F, usize) {
    let mut sum_sq = F::zero();
    let mut n_valid = 0usize;
    for (&a, &b) in row_a.iter().zip(row_b.iter()) {
        if !a.is_nan() && !b.is_nan() {
            let d = a - b;
            sum_sq = sum_sq + d * d;
            n_valid += 1;
        }
    }
    if n_valid == 0 {
        (F::infinity(), 0)
    } else {
        // Scale distance to account for missing dimensions:
        // d_full = d_partial * sqrt(n_total / n_valid)
        // But we keep it simple here: just use sqrt(sum_sq)
        (sum_sq.sqrt(), n_valid)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for KNNImputer<F> {
    type Fitted = FittedKNNImputer<F>;
    type Error = FerroError;

    /// Fit the imputer by storing the training data.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::InvalidParameter`] if `n_neighbors` is zero or exceeds
    ///   the number of samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedKNNImputer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "KNNImputer::fit".into(),
            });
        }
        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "n_neighbors must be at least 1".into(),
            });
        }
        if self.n_neighbors > n_samples {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: format!(
                    "n_neighbors ({}) exceeds the number of training samples ({})",
                    self.n_neighbors, n_samples
                ),
            });
        }

        Ok(FittedKNNImputer {
            train_data: x.to_owned(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedKNNImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Impute missing values in `x` using the k-nearest neighbors from the
    /// training data.
    ///
    /// For each missing value `x[i, j]`, the method finds the `k` nearest
    /// training rows (based on partial Euclidean distance over non-missing
    /// features) that also have a non-missing value at feature `j`, then
    /// computes a (optionally distance-weighted) average.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the training data.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.train_data.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedKNNImputer::transform".into(),
            });
        }

        let mut out = x.to_owned();
        let n_train = self.train_data.nrows();

        for i in 0..out.nrows() {
            // Check if this row has any missing values
            let row_slice: Vec<F> = out.row(i).to_vec();
            let has_missing = row_slice.iter().any(|v| v.is_nan());
            if !has_missing {
                continue;
            }

            // Compute distances to all training rows
            let mut dists: Vec<(usize, F)> = Vec::with_capacity(n_train);
            for t in 0..n_train {
                let train_row: Vec<F> = self.train_data.row(t).to_vec();
                let (d, n_valid) = partial_euclidean_distance(&row_slice, &train_row);
                if n_valid > 0 {
                    dists.push((t, d));
                }
            }
            // Sort by distance
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // For each missing feature, impute from the k nearest neighbors
            // that have a non-missing value for that feature
            for j in 0..n_features {
                if !row_slice[j].is_nan() {
                    continue;
                }

                // Collect up to k neighbors that have a valid value at feature j
                let mut neighbor_vals: Vec<(F, F)> = Vec::new(); // (value, distance)
                for &(t_idx, dist) in &dists {
                    let val = self.train_data[[t_idx, j]];
                    if !val.is_nan() {
                        neighbor_vals.push((val, dist));
                        if neighbor_vals.len() >= self.n_neighbors {
                            break;
                        }
                    }
                }

                if neighbor_vals.is_empty() {
                    // No valid neighbors found — leave as NaN or fill with zero
                    out[[i, j]] = F::zero();
                    continue;
                }

                let imputed = match self.weights {
                    KNNWeights::Uniform => {
                        let sum = neighbor_vals
                            .iter()
                            .map(|&(v, _)| v)
                            .fold(F::zero(), |acc, v| acc + v);
                        sum / F::from(neighbor_vals.len()).unwrap_or_else(F::one)
                    }
                    KNNWeights::Distance => {
                        // Inverse distance weighting
                        let mut weight_sum = F::zero();
                        let mut val_sum = F::zero();
                        let epsilon = F::from(1e-12).unwrap_or_else(F::min_positive_value);
                        for &(val, dist) in &neighbor_vals {
                            let w = if dist <= epsilon {
                                // Exact match — give very high weight
                                F::from(1e12).unwrap_or_else(F::max_value)
                            } else {
                                F::one() / dist
                            };
                            weight_sum = weight_sum + w;
                            val_sum = val_sum + w * val;
                        }
                        if weight_sum > F::zero() {
                            val_sum / weight_sum
                        } else {
                            neighbor_vals[0].0
                        }
                    }
                };

                out[[i, j]] = imputed;
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted imputer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for KNNImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the imputer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "KNNImputer".into(),
            reason: "imputer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for KNNImputer<F> {
    type FitError = FerroError;

    /// Fit the imputer on `x` and return the imputed output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_knn_imputer_uniform_basic() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        // Row 2 has NaN in column 1
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Missing value at [2,1]: nearest 2 neighbors are rows 0 and 1
        // with values 2.0 and 4.0 → mean = 3.0
        assert_abs_diff_eq!(out[[2, 1]], 3.0, epsilon = 1e-10);
        // Non-missing values unchanged
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_imputer_distance_weighted() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Distance);
        // Rows 0 and 1 have known feature 1; row 2 is missing feature 1
        // Row 2 feature 0 = 4.0, row 0 feature 0 = 1.0, row 1 feature 0 = 3.0
        // Distance to row 0: |4 - 1| = 3.0
        // Distance to row 1: |4 - 3| = 1.0
        // Weighted: (2.0 * 1/3 + 6.0 * 1/1) / (1/3 + 1/1) = (0.667 + 6.0) / 1.333 ≈ 5.0
        let x = array![[1.0, 2.0], [3.0, 6.0], [4.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // w0 = 1/3, w1 = 1/1
        let w0 = 1.0 / 3.0;
        let w1 = 1.0 / 1.0;
        let expected = (2.0 * w0 + 6.0 * w1) / (w0 + w1);
        assert_abs_diff_eq!(out[[2, 1]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_imputer_no_missing() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_imputer_multiple_missing() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x = array![
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, f64::NAN, f64::NAN]
        ];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All imputed values should be finite
        assert!(!out[[2, 1]].is_nan());
        assert!(!out[[2, 2]].is_nan());
    }

    #[test]
    fn test_knn_imputer_fit_transform() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN]];
        let out = imputer.fit_transform(&x).unwrap();
        assert!(!out[[2, 1]].is_nan());
    }

    #[test]
    fn test_knn_imputer_zero_rows_error() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_knn_imputer_zero_neighbors_error() {
        let imputer = KNNImputer::<f64>::new(0, KNNWeights::Uniform);
        let x = array![[1.0, 2.0]];
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_knn_imputer_too_many_neighbors_error() {
        let imputer = KNNImputer::<f64>::new(10, KNNWeights::Uniform);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_knn_imputer_shape_mismatch_error() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_knn_imputer_unfitted_transform_error() {
        let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
        let x = array![[1.0, 2.0]];
        assert!(imputer.transform(&x).is_err());
    }

    #[test]
    fn test_knn_imputer_default() {
        let imputer = KNNImputer::<f64>::default();
        assert_eq!(imputer.n_neighbors(), 5);
        assert_eq!(imputer.weights(), KNNWeights::Uniform);
    }

    #[test]
    fn test_knn_imputer_single_neighbor() {
        let imputer = KNNImputer::<f64>::new(1, KNNWeights::Uniform);
        // Row 0 is closest to row 2 (distance on col 0 = |5 - 4| = 1)
        let x = array![[1.0, 10.0], [4.0, 40.0], [5.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Nearest neighbor to row 2 by col 0: row 1 (dist = 1) vs row 0 (dist = 4)
        assert_abs_diff_eq!(out[[2, 1]], 40.0, epsilon = 1e-10);
    }

    #[test]
    fn test_knn_imputer_f32() {
        let imputer = KNNImputer::<f32>::new(2, KNNWeights::Uniform);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, f32::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[2, 1]].is_nan());
    }
}
