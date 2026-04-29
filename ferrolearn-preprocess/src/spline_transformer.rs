//! Spline transformer: generate B-spline basis functions for each feature.
//!
//! [`SplineTransformer`] expands each input feature into a set of B-spline
//! basis columns. This is a nonlinear feature expansion technique that
//! represents each feature as a combination of piecewise polynomial functions.
//!
//! # Knot Placement
//!
//! - [`KnotStrategy::Uniform`] — knots are evenly spaced between min and max.
//! - [`KnotStrategy::Quantile`] — knots are placed at quantiles of the data.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// KnotStrategy
// ---------------------------------------------------------------------------

/// Strategy for placing knots in the spline transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotStrategy {
    /// Knots are evenly spaced between the min and max of each feature.
    Uniform,
    /// Knots are placed at quantiles of the data.
    Quantile,
}

// ---------------------------------------------------------------------------
// SplineTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted spline transformer.
///
/// Calling [`Fit::fit`] computes the knot positions and returns a
/// [`FittedSplineTransformer`] that generates B-spline basis functions.
///
/// # Parameters
///
/// - `n_knots` — number of interior knots (default 5).
/// - `degree` — degree of the B-spline (default 3, i.e. cubic).
/// - `knots` — knot placement strategy (default `Uniform`).
///
/// The number of output columns per feature is `n_knots + degree - 1`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::spline_transformer::{SplineTransformer, KnotStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
/// let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
/// let fitted = st.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // 5 + 3 - 1 = 7 basis columns per feature
/// assert_eq!(out.ncols(), 7);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SplineTransformer<F> {
    /// Number of interior knots.
    n_knots: usize,
    /// Degree of the B-spline.
    degree: usize,
    /// Knot placement strategy.
    knots: KnotStrategy,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SplineTransformer<F> {
    /// Create a new `SplineTransformer`.
    pub fn new(n_knots: usize, degree: usize, knots: KnotStrategy) -> Self {
        Self {
            n_knots,
            degree,
            knots,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of interior knots.
    #[must_use]
    pub fn n_knots(&self) -> usize {
        self.n_knots
    }

    /// Return the B-spline degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return the knot placement strategy.
    #[must_use]
    pub fn knot_strategy(&self) -> KnotStrategy {
        self.knots
    }
}

impl<F: Float + Send + Sync + 'static> Default for SplineTransformer<F> {
    fn default() -> Self {
        Self::new(5, 3, KnotStrategy::Uniform)
    }
}

// ---------------------------------------------------------------------------
// FittedSplineTransformer
// ---------------------------------------------------------------------------

/// A fitted spline transformer holding per-feature knot positions.
///
/// Created by calling [`Fit::fit`] on a [`SplineTransformer`].
#[derive(Debug, Clone)]
pub struct FittedSplineTransformer<F> {
    /// Full knot vector per feature (including boundary knots with multiplicity).
    knot_vectors: Vec<Vec<F>>,
    /// Degree of the B-spline.
    degree: usize,
    /// Number of basis functions per feature.
    n_basis: usize,
}

impl<F: Float + Send + Sync + 'static> FittedSplineTransformer<F> {
    /// Return the knot vectors.
    #[must_use]
    pub fn knot_vectors(&self) -> &[Vec<F>] {
        &self.knot_vectors
    }

    /// Return the number of basis functions per feature.
    #[must_use]
    pub fn n_basis_per_feature(&self) -> usize {
        self.n_basis
    }

    /// Return the total number of output columns.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.knot_vectors.len() * self.n_basis
    }
}

// ---------------------------------------------------------------------------
// B-spline evaluation (Cox-de Boor recursion)
// ---------------------------------------------------------------------------

/// Evaluate all B-spline basis functions at a given value `x` using the
/// Cox-de Boor recursion.
///
/// `knots` is the full knot vector of length `n_basis + degree + 1`.
/// Returns a vector of length `n_basis` containing the basis values.
fn bspline_basis<F: Float>(x: F, knots: &[F], degree: usize, n_basis: usize) -> Vec<F> {
    // Start with degree-0 basis functions
    let n_intervals = knots.len() - 1;
    let mut basis = vec![F::zero(); n_intervals];

    // Degree 0: indicator functions using half-open intervals [t_i, t_{i+1}).
    // Special case: when x equals the rightmost knot, assign it to the last
    // non-degenerate interval so the Cox-de Boor recursion can propagate
    // the value up through knot spans with non-zero width.
    let last_knot = knots[knots.len() - 1];
    if x >= last_knot {
        // Find the last interval with non-zero width and activate it
        let mut found = false;
        for i in (0..n_intervals).rev() {
            if knots[i] < knots[i + 1] {
                basis[i] = F::one();
                found = true;
                break;
            }
        }
        // Fallback: if all intervals are degenerate, activate the last one
        if !found {
            basis[n_intervals - 1] = F::one();
        }
    } else {
        for i in 0..n_intervals {
            // Half-open: [t_i, t_{i+1})
            basis[i] = if x >= knots[i] && x < knots[i + 1] {
                F::one()
            } else {
                F::zero()
            };
        }
    }

    // Build up to the desired degree
    for d in 1..=degree {
        let n_current = n_intervals - d;
        let mut new_basis = vec![F::zero(); n_current];
        for i in 0..n_current {
            let denom1 = knots[i + d] - knots[i];
            let denom2 = knots[i + d + 1] - knots[i + 1];

            let left = if denom1 > F::zero() {
                (x - knots[i]) / denom1 * basis[i]
            } else {
                F::zero()
            };

            let right = if denom2 > F::zero() {
                (knots[i + d + 1] - x) / denom2 * basis[i + 1]
            } else {
                F::zero()
            };

            new_basis[i] = left + right;
        }
        basis = new_basis;
    }

    // Truncate or pad to n_basis
    basis.truncate(n_basis);
    while basis.len() < n_basis {
        basis.push(F::zero());
    }

    basis
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SplineTransformer<F> {
    type Fitted = FittedSplineTransformer<F>;
    type Error = FerroError;

    /// Fit by computing knot positions for each feature.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has fewer than 2 rows.
    /// - [`FerroError::InvalidParameter`] if `n_knots` < 2 or `degree` is 0.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSplineTransformer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "SplineTransformer::fit".into(),
            });
        }
        if self.n_knots < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_knots".into(),
                reason: "n_knots must be at least 2".into(),
            });
        }
        if self.degree == 0 {
            return Err(FerroError::InvalidParameter {
                name: "degree".into(),
                reason: "degree must be at least 1".into(),
            });
        }

        let n_features = x.ncols();
        let n_basis = self.n_knots + self.degree - 1;
        let mut knot_vectors = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col_vals: Vec<F> = x.column(j).iter().copied().collect();
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = col_vals[0];
            let max_val = col_vals[col_vals.len() - 1];

            // Compute interior knots
            let interior_knots: Vec<F> = match self.knots {
                KnotStrategy::Uniform => (0..self.n_knots)
                    .map(|i| {
                        min_val
                            + (max_val - min_val) * F::from(i).unwrap()
                                / F::from(self.n_knots - 1).unwrap()
                    })
                    .collect(),
                KnotStrategy::Quantile => {
                    let n = col_vals.len();
                    (0..self.n_knots)
                        .map(|i| {
                            let frac = F::from(i).unwrap()
                                / F::from(self.n_knots - 1).unwrap_or_else(F::one);
                            let pos = frac * F::from(n.saturating_sub(1)).unwrap();
                            let lo = pos.floor().to_usize().unwrap_or(0).min(n - 1);
                            let hi = pos.ceil().to_usize().unwrap_or(0).min(n - 1);
                            let f = pos - F::from(lo).unwrap();
                            col_vals[lo] * (F::one() - f) + col_vals[hi] * f
                        })
                        .collect()
                }
            };

            // Build full knot vector with boundary knots of multiplicity (degree + 1)
            let mut full_knots = Vec::new();
            // Left boundary knots
            for _ in 0..self.degree {
                full_knots.push(min_val);
            }
            // Interior knots
            full_knots.extend_from_slice(&interior_knots);
            // Right boundary knots
            for _ in 0..self.degree {
                full_knots.push(max_val);
            }

            knot_vectors.push(full_knots);
        }

        Ok(FittedSplineTransformer {
            knot_vectors,
            degree: self.degree,
            n_basis,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSplineTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate B-spline basis functions for each feature.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.knot_vectors.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSplineTransformer::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let n_out = n_features * self.n_basis;
        let mut out = Array2::zeros((n_samples, n_out));

        for j in 0..n_features {
            let knots = &self.knot_vectors[j];
            let col_offset = j * self.n_basis;

            for i in 0..n_samples {
                let val = x[[i, j]];
                let basis_vals = bspline_basis(val, knots, self.degree, self.n_basis);
                for (k, &bv) in basis_vals.iter().enumerate() {
                    out[[i, col_offset + k]] = bv;
                }
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted transformer.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SplineTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the transformer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "SplineTransformer".into(),
            reason: "transformer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for SplineTransformer<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
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
    fn test_spline_output_dimensions() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis = n_knots + degree - 1 = 5 + 3 - 1 = 7
        assert_eq!(out.ncols(), 7);
        assert_eq!(out.nrows(), 5);
    }

    #[test]
    fn test_spline_partition_of_unity() {
        // B-spline basis functions should sum to 1 at any interior point
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spline_non_negative() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.1], [0.5], [0.9], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for v in &out {
            assert!(*v >= -1e-10, "Basis value should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_spline_quantile_knots() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Quantile);
        let x = array![[0.0], [0.1], [0.2], [0.5], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 7);
        // Partition of unity should still hold
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_spline_multi_feature() {
        let st = SplineTransformer::<f64>::new(3, 2, KnotStrategy::Uniform);
        let x = array![[0.0, 10.0], [0.5, 15.0], [1.0, 20.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis per feature = 3 + 2 - 1 = 4, total = 2 * 4 = 8
        assert_eq!(out.ncols(), 8);
    }

    #[test]
    fn test_spline_fit_transform() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [0.5], [1.0]];
        let out = st.fit_transform(&x).unwrap();
        assert_eq!(out.ncols(), 7);
    }

    #[test]
    fn test_spline_insufficient_samples_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_too_few_knots_error() {
        let st = SplineTransformer::<f64>::new(1, 3, KnotStrategy::Uniform);
        let x = array![[0.0], [1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_zero_degree_error() {
        let st = SplineTransformer::<f64>::new(5, 0, KnotStrategy::Uniform);
        let x = array![[0.0], [1.0]];
        assert!(st.fit(&x, &()).is_err());
    }

    #[test]
    fn test_spline_shape_mismatch_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x_train = array![[0.0, 1.0], [0.5, 1.5]];
        let fitted = st.fit(&x_train, &()).unwrap();
        let x_bad = array![[0.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_spline_unfitted_error() {
        let st = SplineTransformer::<f64>::new(5, 3, KnotStrategy::Uniform);
        let x = array![[0.0]];
        assert!(st.transform(&x).is_err());
    }

    #[test]
    fn test_spline_default() {
        let st = SplineTransformer::<f64>::default();
        assert_eq!(st.n_knots(), 5);
        assert_eq!(st.degree(), 3);
        assert_eq!(st.knot_strategy(), KnotStrategy::Uniform);
    }

    #[test]
    fn test_spline_degree1() {
        // Linear splines: should produce piecewise linear basis
        let st = SplineTransformer::<f64>::new(3, 1, KnotStrategy::Uniform);
        let x = array![[0.0], [0.5], [1.0]];
        let fitted = st.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // n_basis = 3 + 1 - 1 = 3
        assert_eq!(out.ncols(), 3);
        // Partition of unity
        for i in 0..out.nrows() {
            let row_sum: f64 = out.row(i).iter().sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }
}
