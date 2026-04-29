//! Quantile transformer: map features to a uniform or normal distribution.
//!
//! [`QuantileTransformer`] transforms features by mapping each value through
//! its empirical cumulative distribution function (CDF), producing values
//! uniformly distributed in `[0, 1]`. Optionally, the result can be mapped
//! to a standard normal distribution using the inverse normal CDF (probit).
//!
//! This is useful for making features more Gaussian-like, which can improve
//! the performance of many machine learning algorithms.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// OutputDistribution
// ---------------------------------------------------------------------------

/// Target output distribution for the quantile transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDistribution {
    /// Map to the uniform distribution on `[0, 1]`.
    Uniform,
    /// Map to the standard normal distribution via the probit function.
    Normal,
}

// ---------------------------------------------------------------------------
// QuantileTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted quantile transformer.
///
/// Calling [`Fit::fit`] computes the quantiles for each feature and returns a
/// [`FittedQuantileTransformer`].
///
/// # Parameters
///
/// - `n_quantiles` — number of quantile reference points (default 1000).
/// - `output_distribution` — target distribution (default `Uniform`).
/// - `subsample` — maximum number of samples used to compute quantiles
///   (default 100_000; set to 0 to use all samples).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::quantile_transformer::{QuantileTransformer, OutputDistribution};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
/// let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
/// let fitted = qt.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // Values should be in [0, 1]
/// for v in out.iter() {
///     assert!(*v >= 0.0 && *v <= 1.0);
/// }
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct QuantileTransformer<F> {
    /// Number of quantile reference points.
    n_quantiles: usize,
    /// Target output distribution.
    output_distribution: OutputDistribution,
    /// Maximum number of samples for quantile computation (0 = all).
    subsample: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> QuantileTransformer<F> {
    /// Create a new `QuantileTransformer`.
    pub fn new(
        n_quantiles: usize,
        output_distribution: OutputDistribution,
        subsample: usize,
    ) -> Self {
        Self {
            n_quantiles,
            output_distribution,
            subsample,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of quantiles.
    #[must_use]
    pub fn n_quantiles(&self) -> usize {
        self.n_quantiles
    }

    /// Return the target output distribution.
    #[must_use]
    pub fn output_distribution(&self) -> OutputDistribution {
        self.output_distribution
    }

    /// Return the subsample size.
    #[must_use]
    pub fn subsample(&self) -> usize {
        self.subsample
    }
}

impl<F: Float + Send + Sync + 'static> Default for QuantileTransformer<F> {
    fn default() -> Self {
        Self::new(1000, OutputDistribution::Uniform, 100_000)
    }
}

// ---------------------------------------------------------------------------
// FittedQuantileTransformer
// ---------------------------------------------------------------------------

/// A fitted quantile transformer holding per-feature quantile references.
///
/// Created by calling [`Fit::fit`] on a [`QuantileTransformer`].
#[derive(Debug, Clone)]
pub struct FittedQuantileTransformer<F> {
    /// Quantile reference values per feature: `quantiles[j]` is a sorted
    /// vector of reference values for feature `j`.
    quantiles: Vec<Vec<F>>,
    /// The reference quantile levels (evenly spaced in [0, 1]).
    references: Vec<F>,
    /// Target output distribution.
    output_distribution: OutputDistribution,
}

impl<F: Float + Send + Sync + 'static> FittedQuantileTransformer<F> {
    /// Return the computed quantile reference values per feature.
    #[must_use]
    pub fn quantiles(&self) -> &[Vec<F>] {
        &self.quantiles
    }

    /// Return the number of features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.quantiles.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Approximate the inverse normal CDF (probit function) using the rational
/// approximation by Abramowitz and Stegun.
fn probit<F: Float>(p: F) -> F {
    // Clamp to avoid infinities
    let eps = F::from(1e-7).unwrap_or_else(F::min_positive_value);
    let p = if p < eps {
        eps
    } else if p > F::one() - eps {
        F::one() - eps
    } else {
        p
    };

    // Rational approximation for the probit function
    let half = F::from(0.5).unwrap();
    if p < half {
        // Use symmetry: probit(p) = -probit(1-p)
        let t = (-F::from(2.0).unwrap() * p.ln()).sqrt();
        let c0 = F::from(2.515517).unwrap();
        let c1 = F::from(0.802853).unwrap();
        let c2 = F::from(0.010328).unwrap();
        let d1 = F::from(1.432788).unwrap();
        let d2 = F::from(0.189269).unwrap();
        let d3 = F::from(0.001308).unwrap();
        let num = c0 + c1 * t + c2 * t * t;
        let den = F::one() + d1 * t + d2 * t * t + d3 * t * t * t;
        -(t - num / den)
    } else {
        let t = (-F::from(2.0).unwrap() * (F::one() - p).ln()).sqrt();
        let c0 = F::from(2.515517).unwrap();
        let c1 = F::from(0.802853).unwrap();
        let c2 = F::from(0.010328).unwrap();
        let d1 = F::from(1.432788).unwrap();
        let d2 = F::from(0.189269).unwrap();
        let d3 = F::from(0.001308).unwrap();
        let num = c0 + c1 * t + c2 * t * t;
        let den = F::one() + d1 * t + d2 * t * t + d3 * t * t * t;
        t - num / den
    }
}

/// Linearly interpolate: find the quantile level for a given value in a
/// sorted quantile reference vector.
fn interpolate_cdf<F: Float>(value: F, quantiles: &[F], references: &[F]) -> F {
    if quantiles.is_empty() {
        return F::from(0.5).unwrap();
    }

    // Clamp to range
    if value <= quantiles[0] {
        return references[0];
    }
    if value >= quantiles[quantiles.len() - 1] {
        return references[references.len() - 1];
    }

    // Binary search for the interval
    let mut lo = 0;
    let mut hi = quantiles.len() - 1;
    while lo < hi - 1 {
        let mid = usize::midpoint(lo, hi);
        if quantiles[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation
    let denom = quantiles[hi] - quantiles[lo];
    if denom == F::zero() {
        references[lo]
    } else {
        let frac = (value - quantiles[lo]) / denom;
        references[lo] + frac * (references[hi] - references[lo])
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for QuantileTransformer<F> {
    type Fitted = FittedQuantileTransformer<F>;
    type Error = FerroError;

    /// Fit by computing per-feature quantile reference values.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has fewer than 2 rows.
    /// - [`FerroError::InvalidParameter`] if `n_quantiles` is less than 2.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedQuantileTransformer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "QuantileTransformer::fit".into(),
            });
        }
        if self.n_quantiles < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_quantiles".into(),
                reason: "n_quantiles must be at least 2".into(),
            });
        }

        let n_features = x.ncols();
        let effective_quantiles = self.n_quantiles.min(n_samples);

        // Build evenly spaced reference levels in [0, 1]
        let references: Vec<F> = (0..effective_quantiles)
            .map(|i| F::from(i).unwrap() / F::from(effective_quantiles - 1).unwrap_or_else(F::one))
            .collect();

        let mut quantiles = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col_vals: Vec<F> = x.column(j).iter().copied().collect();
            // Remove NaN values
            col_vals.retain(|v| !v.is_nan());
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Subsample if needed
            if self.subsample > 0 && col_vals.len() > self.subsample {
                let step = col_vals.len() as f64 / self.subsample as f64;
                let mut sampled = Vec::with_capacity(self.subsample);
                for i in 0..self.subsample {
                    let idx = (i as f64 * step) as usize;
                    sampled.push(col_vals[idx.min(col_vals.len() - 1)]);
                }
                col_vals = sampled;
            }

            // Compute quantile reference values
            let n = col_vals.len();
            let mut feature_quantiles = Vec::with_capacity(effective_quantiles);
            for &ref_level in &references {
                let pos = ref_level * F::from(n.saturating_sub(1)).unwrap();
                let lo = pos.floor().to_usize().unwrap_or(0).min(n.saturating_sub(1));
                let hi = pos.ceil().to_usize().unwrap_or(0).min(n.saturating_sub(1));
                let frac = pos - F::from(lo).unwrap();
                let val = if lo == hi {
                    col_vals[lo]
                } else {
                    col_vals[lo] * (F::one() - frac) + col_vals[hi] * frac
                };
                feature_quantiles.push(val);
            }

            quantiles.push(feature_quantiles);
        }

        Ok(FittedQuantileTransformer {
            quantiles,
            references,
            output_distribution: self.output_distribution,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedQuantileTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by mapping each value through the empirical CDF.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.quantiles.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedQuantileTransformer::transform".into(),
            });
        }

        let mut out = x.to_owned();

        for j in 0..n_features {
            let feature_quantiles = &self.quantiles[j];
            for i in 0..out.nrows() {
                let val = out[[i, j]];
                if val.is_nan() {
                    continue;
                }
                let cdf_val = interpolate_cdf(val, feature_quantiles, &self.references);

                out[[i, j]] = match self.output_distribution {
                    OutputDistribution::Uniform => cdf_val,
                    OutputDistribution::Normal => probit(cdf_val),
                };
            }
        }

        Ok(out)
    }
}

/// Implement `Transform` on the unfitted transformer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for QuantileTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the transformer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "QuantileTransformer".into(),
            reason: "transformer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for QuantileTransformer<F> {
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
    fn test_quantile_transformer_uniform() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All values should be in [0, 1]
        for v in &out {
            assert!(*v >= 0.0 && *v <= 1.0, "Value {v} not in [0,1]");
        }
        // First should be 0, last should be 1
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[4, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_transformer_normal() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Normal, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Middle value should be close to 0 (median → 0 in normal)
        assert!(out[[2, 0]].abs() < 0.5, "Median should map near 0");
        // First should be negative, last positive
        assert!(out[[0, 0]] < out[[4, 0]]);
    }

    #[test]
    fn test_quantile_transformer_monotonic() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[5.0], [3.0], [1.0], [4.0], [2.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Transform should preserve ordering: rank(5) > rank(3) > rank(1)
        assert!(out[[0, 0]] > out[[1, 0]]); // 5 > 3
        assert!(out[[1, 0]] > out[[2, 0]]); // 3 > 1
    }

    #[test]
    fn test_quantile_transformer_multiple_features() {
        let qt = QuantileTransformer::<f64>::new(50, OutputDistribution::Uniform, 0);
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        // Each feature independently transformed
        for j in 0..2 {
            assert!(out[[0, j]] <= out[[2, j]]);
        }
    }

    #[test]
    fn test_quantile_transformer_fit_transform() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let out = qt.fit_transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[4, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile_transformer_insufficient_samples_error() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0]];
        assert!(qt.fit(&x, &()).is_err());
    }

    #[test]
    fn test_quantile_transformer_too_few_quantiles_error() {
        let qt = QuantileTransformer::<f64>::new(1, OutputDistribution::Uniform, 0);
        let x = array![[1.0], [2.0], [3.0]];
        assert!(qt.fit(&x, &()).is_err());
    }

    #[test]
    fn test_quantile_transformer_shape_mismatch() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = qt.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_quantile_transformer_unfitted_error() {
        let qt = QuantileTransformer::<f64>::new(100, OutputDistribution::Uniform, 0);
        let x = array![[1.0]];
        assert!(qt.transform(&x).is_err());
    }

    #[test]
    fn test_quantile_transformer_default() {
        let qt = QuantileTransformer::<f64>::default();
        assert_eq!(qt.n_quantiles(), 1000);
        assert_eq!(qt.output_distribution(), OutputDistribution::Uniform);
        assert_eq!(qt.subsample(), 100_000);
    }

    #[test]
    fn test_quantile_transformer_f32() {
        let qt = QuantileTransformer::<f32>::new(50, OutputDistribution::Uniform, 0);
        let x: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let fitted = qt.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(out[[0, 0]] >= 0.0f32);
        assert!(out[[4, 0]] <= 1.0f32);
    }
}
