//! Polynomial features: generate polynomial and interaction features.
//!
//! Given input features `[a, b]` and degree 2, this transformer generates:
//! - `[1, a, b, a², a·b, b²]` (default — `interaction_only = false`)
//! - `[1, a, b, a·b]` (with `interaction_only = true`)
//!
//! With `include_bias = false`, the constant column `1` is omitted.
//!
//! This transformer is **stateless** — no fitting is needed. Call
//! [`Transform::transform`] directly.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::Transform;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

/// A stateless polynomial feature generator.
///
/// Generates all polynomial combinations of the input features up to the
/// specified degree.
///
/// # Configuration
///
/// - `degree`: maximum polynomial degree (default `2`).
/// - `interaction_only`: if `true`, only cross-product terms are generated
///   (no pure powers like `a²`). Default `false`.
/// - `include_bias`: if `true`, a constant column of ones is prepended.
///   Default `true`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::polynomial_features::PolynomialFeatures;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
/// let x = array![[2.0, 3.0]];
/// let out = poly.transform(&x).unwrap();
/// // out = [[1, 2, 3, 4, 6, 9]]
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeatures<F> {
    /// Maximum polynomial degree.
    pub(crate) degree: usize,
    /// If `true`, only interaction terms are produced (no pure powers).
    pub(crate) interaction_only: bool,
    /// If `true`, prepend a bias (constant ones) column.
    pub(crate) include_bias: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PolynomialFeatures<F> {
    /// Create a new `PolynomialFeatures` transformer.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `degree == 0`.
    pub fn new(
        degree: usize,
        interaction_only: bool,
        include_bias: bool,
    ) -> Result<Self, FerroError> {
        if degree == 0 {
            return Err(FerroError::InvalidParameter {
                name: "degree".into(),
                reason: "degree must be at least 1".into(),
            });
        }
        Ok(Self {
            degree,
            interaction_only,
            include_bias,
            _marker: std::marker::PhantomData,
        })
    }

    /// Create a `PolynomialFeatures` with default settings:
    /// degree=2, interaction_only=false, include_bias=true.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the configured degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Return whether only interaction terms are generated.
    #[must_use]
    pub fn interaction_only(&self) -> bool {
        self.interaction_only
    }

    /// Return whether a bias column is included.
    #[must_use]
    pub fn include_bias(&self) -> bool {
        self.include_bias
    }

    /// Generate all combinations (with repetition unless `interaction_only`)
    /// of feature indices up to `degree`.
    ///
    /// Returns a list of index-tuples, where each tuple specifies which
    /// feature indices to multiply together to produce one output column.
    fn feature_combinations(&self, n_features: usize) -> Vec<Vec<usize>> {
        let mut combos: Vec<Vec<usize>> = Vec::new();

        // Bias term: empty product = 1
        if self.include_bias {
            combos.push(vec![]);
        }

        // Generate combinations of degrees 1..=self.degree
        let mut stack: Vec<(Vec<usize>, usize)> = Vec::new();

        // Start with each feature at degree 1
        for i in 0..n_features {
            stack.push((vec![i], i));
        }

        while let Some((combo, last_idx)) = stack.pop() {
            combos.push(combo.clone());

            if combo.len() < self.degree {
                // Extend with another feature
                let start = if self.interaction_only {
                    // Strictly increasing indices — no repeated features
                    last_idx + 1
                } else {
                    // Non-decreasing indices — allows repeated features (pure powers)
                    last_idx
                };
                for i in start..n_features {
                    let mut new_combo = combo.clone();
                    new_combo.push(i);
                    stack.push((new_combo, i));
                }
            }
        }

        // Sort: bias first, then by combo length, then lexicographically
        combos.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        combos
    }
}

impl<F: Float + Send + Sync + 'static> Default for PolynomialFeatures<F> {
    fn default() -> Self {
        Self::default_config()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for PolynomialFeatures<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Generate polynomial and interaction features.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the input has zero columns.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "x".into(),
                reason: "input must have at least one column".into(),
            });
        }

        let combos = self.feature_combinations(n_features);
        let n_out = combos.len();

        let mut out = Array2::zeros((n_samples, n_out));

        for (k, combo) in combos.iter().enumerate() {
            for i in 0..n_samples {
                let val = combo.iter().fold(F::one(), |acc, &j| acc * x[[i, j]]);
                out[[i, k]] = val;
            }
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for PolynomialFeatures<F> {
    /// Fit the polynomial features transformer using the pipeline interface.
    ///
    /// Because `PolynomialFeatures` is stateless, this simply boxes `self`
    /// as a [`FittedPipelineTransformer`].
    ///
    /// # Errors
    ///
    /// This implementation never returns an error.
    fn fit_pipeline(
        &self,
        _x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        Ok(Box::new(self.clone()))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for PolynomialFeatures<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
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
    fn test_degree2_two_features_with_bias() {
        // degree=2, interaction_only=false, include_bias=true
        // Expected: [1, a, b, a², a·b, b²]
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 6); // 1 + 2 + 3 combinations
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // bias
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 4.0, epsilon = 1e-10); // a²
        assert_abs_diff_eq!(out[[0, 4]], 6.0, epsilon = 1e-10); // a·b
        assert_abs_diff_eq!(out[[0, 5]], 9.0, epsilon = 1e-10); // b²
    }

    #[test]
    fn test_degree2_interaction_only() {
        // degree=2, interaction_only=true, include_bias=true
        // Expected: [1, a, b, a·b]
        let poly = PolynomialFeatures::<f64>::new(2, true, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 4);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10); // bias
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10); // a
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10); // b
        assert_abs_diff_eq!(out[[0, 3]], 6.0, epsilon = 1e-10); // a·b
    }

    #[test]
    fn test_no_bias() {
        // degree=2, interaction_only=false, include_bias=false
        // Expected: [a, b, a², a·b, b²]
        let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 5);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-10); // a
    }

    #[test]
    fn test_degree1_only_linear() {
        let poly = PolynomialFeatures::<f64>::new(1, false, true).unwrap();
        let x = array![[2.0, 3.0]];
        let out = poly.transform(&x).unwrap();
        // [1, a, b]
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multiple_rows() {
        let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5]);
        // Row 0: a=1, b=2 → [1, 2, 1, 2, 4]
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 3]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 4]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_feature_degree2() {
        // [a] → [1, a, a²]
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[3.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_degree_zero() {
        assert!(PolynomialFeatures::<f64>::new(0, false, true).is_err());
    }

    #[test]
    fn test_default_config() {
        let poly = PolynomialFeatures::<f64>::default();
        assert_eq!(poly.degree(), 2);
        assert!(!poly.interaction_only());
        assert!(poly.include_bias());
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let poly = PolynomialFeatures::<f64>::new(2, false, true).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = Array1::zeros(2);
        let fitted = poly.fit_pipeline(&x, &y).unwrap();
        let result = fitted.transform_pipeline(&x).unwrap();
        assert_eq!(result.shape(), &[2, 6]);
    }

    #[test]
    fn test_degree3_single_feature() {
        // [a] with degree=3, no bias → [a, a², a³]
        let poly = PolynomialFeatures::<f64>::new(3, false, false).unwrap();
        let x = array![[2.0]];
        let out = poly.transform(&x).unwrap();
        assert_eq!(out.shape()[1], 3);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 8.0, epsilon = 1e-10);
    }
}
