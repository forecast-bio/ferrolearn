//! Feature selection driven by a model's feature importance weights.
//!
//! [`SelectFromModel`](super::feature_selection::SelectFromModel) provides
//! basic mean/explicit-threshold selection.  This module provides a richer
//! API via [`SelectFromModelExt`], which supports four threshold strategies
//! (mean, median, explicit value, percentile) and an optional
//! `max_features` cap.
//!
//! # Threshold Strategies
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`ThresholdStrategy::Mean`] | Threshold = arithmetic mean of importances |
//! | [`ThresholdStrategy::Median`] | Threshold = median of importances |
//! | [`ThresholdStrategy::Value`] | User-supplied explicit threshold |
//! | [`ThresholdStrategy::Percentile`] | Keep features in the top *p*% by importance |
//!
//! When `max_features` is set, at most that many features are retained
//! (in descending importance order) regardless of the threshold.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// ThresholdStrategy
// ---------------------------------------------------------------------------

/// Strategy for computing the importance threshold in [`SelectFromModelExt`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ThresholdStrategy {
    /// Threshold equals the arithmetic mean of all feature importances.
    #[default]
    Mean,
    /// Threshold equals the median of all feature importances.
    Median,
    /// User-supplied explicit threshold value.
    Value(f64),
    /// Keep features in the top `p`% of importance scores (0 < p <= 100).
    ///
    /// For example, `Percentile(25.0)` retains features whose importance is
    /// at or above the 75th-percentile value (i.e., the top 25%).
    Percentile(f64),
}

// ---------------------------------------------------------------------------
// SelectFromModelExt (unfitted)
// ---------------------------------------------------------------------------

/// An extended model-importance-based feature selector.
///
/// Like [`SelectFromModel`](super::feature_selection::SelectFromModel) but
/// supports four threshold strategies and an optional `max_features` cap.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::select_from_model::{SelectFromModelExt, ThresholdStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
/// let importances = array![0.1, 0.5, 0.4];
/// let fitted = sel.fit(&importances, &()).unwrap();
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let out = fitted.transform(&x).unwrap();
/// // Mean importance = (0.1+0.5+0.4)/3 ≈ 0.333; columns 1 and 2 kept
/// assert_eq!(out.ncols(), 2);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SelectFromModelExt<F> {
    /// The threshold strategy.
    threshold: ThresholdStrategy,
    /// Optional cap on number of features to select.
    max_features: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectFromModelExt<F> {
    /// Create a new `SelectFromModelExt`.
    ///
    /// # Parameters
    ///
    /// - `threshold` — the strategy for computing the importance threshold.
    /// - `max_features` — optional maximum number of features to retain.
    pub fn new(threshold: ThresholdStrategy, max_features: Option<usize>) -> Self {
        Self {
            threshold,
            max_features,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the threshold strategy.
    #[must_use]
    pub fn threshold_strategy(&self) -> ThresholdStrategy {
        self.threshold
    }

    /// Return the maximum number of features (if set).
    #[must_use]
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }
}

impl<F: Float + Send + Sync + 'static> Default for SelectFromModelExt<F> {
    fn default() -> Self {
        Self::new(ThresholdStrategy::Mean, None)
    }
}

// ---------------------------------------------------------------------------
// FittedSelectFromModelExt
// ---------------------------------------------------------------------------

/// A fitted model-importance selector produced by [`SelectFromModelExt::fit`].
#[derive(Debug, Clone)]
pub struct FittedSelectFromModelExt<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// The computed threshold value.
    threshold_value: F,
    /// Feature importances supplied during fitting.
    importances: Array1<F>,
    /// Indices of selected columns (sorted).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectFromModelExt<F> {
    /// Return the computed threshold value.
    #[must_use]
    pub fn threshold_value(&self) -> F {
        self.threshold_value
    }

    /// Return the feature importances.
    #[must_use]
    pub fn importances(&self) -> &Array1<F> {
        &self.importances
    }

    /// Return the indices of the selected columns.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Return the number of selected features.
    #[must_use]
    pub fn n_features_selected(&self) -> usize {
        self.selected_indices.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the median of a slice of floats.
fn compute_median<F: Float>(values: &[F]) -> F {
    let mut sorted: Vec<F> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        let two = F::one() + F::one();
        (sorted[n / 2 - 1] + sorted[n / 2]) / two
    } else {
        sorted[n / 2]
    }
}

/// Compute the percentile threshold. `pct` is the percentage of features to
/// keep (e.g., 25.0 means top 25%).
fn compute_percentile_threshold<F: Float>(values: &[F], pct: f64) -> F {
    let mut sorted: Vec<F> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    // The threshold is set at the (100 - pct) percentile of the sorted values.
    // E.g., for top 25% we want the value at the 75th percentile.
    let rank = ((100.0 - pct) / 100.0) * (n.saturating_sub(1)) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let lower = lower.min(n.saturating_sub(1));
    let upper = upper.min(n.saturating_sub(1));
    if lower == upper {
        sorted[lower]
    } else {
        let frac = F::from(rank - rank.floor()).unwrap_or(F::zero());
        sorted[lower] * (F::one() - frac) + sorted[upper] * frac
    }
}

/// Build a new `Array2<F>` containing only the columns listed in `indices`.
fn select_columns<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let nrows = x.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        return Array2::zeros((nrows, 0));
    }
    let mut out = Array2::zeros((nrows, ncols));
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..nrows {
            out[[i, new_j]] = x[[i, old_j]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array1<F>, ()> for SelectFromModelExt<F> {
    type Fitted = FittedSelectFromModelExt<F>;
    type Error = FerroError;

    /// Fit by computing the threshold from the given feature importances.
    ///
    /// # Parameters
    ///
    /// - `x` — per-feature importance scores (one value per feature).
    /// - `_y` — ignored (unsupervised).
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if the importance vector is empty,
    ///   or if `Percentile` value is not in `(0, 100]`.
    fn fit(
        &self,
        x: &Array1<F>,
        _y: &(),
    ) -> Result<FittedSelectFromModelExt<F>, FerroError> {
        let n = x.len();
        if n == 0 {
            return Err(FerroError::InvalidParameter {
                name: "importances".into(),
                reason: "importance vector must not be empty".into(),
            });
        }

        let values: Vec<F> = x.iter().copied().collect();

        // Compute threshold
        let threshold_value = match self.threshold {
            ThresholdStrategy::Mean => {
                values.iter().copied().fold(F::zero(), |acc, v| acc + v)
                    / F::from(n).unwrap_or(F::one())
            }
            ThresholdStrategy::Median => compute_median(&values),
            ThresholdStrategy::Value(v) => F::from(v).unwrap_or(F::zero()),
            ThresholdStrategy::Percentile(pct) => {
                if pct <= 0.0 || pct > 100.0 {
                    return Err(FerroError::InvalidParameter {
                        name: "percentile".into(),
                        reason: format!(
                            "percentile must be in (0, 100], got {}",
                            pct
                        ),
                    });
                }
                compute_percentile_threshold(&values, pct)
            }
        };

        // Select features whose importance >= threshold
        let mut selected_indices: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|&(_, &imp)| imp >= threshold_value)
            .map(|(j, _)| j)
            .collect();

        // Apply max_features cap: keep only the top-k by importance
        if let Some(max_f) = self.max_features {
            if selected_indices.len() > max_f {
                // Sort selected by importance descending, keep top max_f
                selected_indices.sort_by(|&a, &b| {
                    values[b]
                        .partial_cmp(&values[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                selected_indices.truncate(max_f);
                // Re-sort in column order
                selected_indices.sort_unstable();
            }
        }

        Ok(FittedSelectFromModelExt {
            n_features_in: n,
            threshold_value,
            importances: x.clone(),
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectFromModelExt<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSelectFromModelExt::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for FittedSelectFromModelExt<F> {
    /// Clone the fitted selector and box it as a pipeline transformer.
    ///
    /// Because the selector is already fitted (importances supplied at fit
    /// time), `fit_pipeline` simply boxes the existing fitted state.
    ///
    /// # Errors
    ///
    /// This implementation never fails.
    fn fit_pipeline(
        &self,
        _x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        Ok(Box::new(self.clone()))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedSelectFromModelExt<F>
{
    /// Transform using the pipeline interface.
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
    fn test_mean_threshold() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
        let importances = array![0.1, 0.5, 0.4];
        let fitted = sel.fit(&importances, &()).unwrap();
        // Mean = (0.1+0.5+0.4)/3 ≈ 0.333; cols 1 and 2 kept
        assert_eq!(fitted.selected_indices(), &[1, 2]);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 1]], 3.0, epsilon = 1e-15);
    }

    #[test]
    fn test_median_threshold() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Median, None);
        // Sorted: [0.1, 0.3, 0.5] → median = 0.3
        let importances = array![0.1, 0.5, 0.3];
        let fitted = sel.fit(&importances, &()).unwrap();
        // Features with importance >= 0.3: indices 1 (0.5) and 2 (0.3)
        assert_eq!(fitted.selected_indices(), &[1, 2]);
    }

    #[test]
    fn test_median_threshold_even() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Median, None);
        // Sorted: [0.1, 0.2, 0.5, 0.6] → median = (0.2+0.5)/2 = 0.35
        let importances = array![0.1, 0.5, 0.2, 0.6];
        let fitted = sel.fit(&importances, &()).unwrap();
        // Features >= 0.35: 1 (0.5) and 3 (0.6)
        assert_eq!(fitted.selected_indices(), &[1, 3]);
    }

    #[test]
    fn test_explicit_value_threshold() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.45), None);
        let importances = array![0.1, 0.5, 0.4];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[1]);
    }

    #[test]
    fn test_percentile_threshold_top_50() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(50.0), None);
        // Sorted: [0.1, 0.3, 0.5, 0.7]
        // Top 50% → threshold at 50th percentile = sorted[1.5] interp = 0.4
        let importances = array![0.5, 0.1, 0.7, 0.3];
        let fitted = sel.fit(&importances, &()).unwrap();
        // Features >= threshold: 0 (0.5), 2 (0.7)
        assert!(fitted.selected_indices().contains(&0));
        assert!(fitted.selected_indices().contains(&2));
        assert_eq!(fitted.n_features_selected(), 2);
    }

    #[test]
    fn test_percentile_100_keeps_all() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(100.0), None);
        let importances = array![0.1, 0.5, 0.3];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 3);
    }

    #[test]
    fn test_percentile_invalid() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(0.0), None);
        let importances = array![0.1, 0.5, 0.3];
        assert!(sel.fit(&importances, &()).is_err());

        let sel2 = SelectFromModelExt::<f64>::new(ThresholdStrategy::Percentile(101.0), None);
        assert!(sel2.fit(&importances, &()).is_err());
    }

    #[test]
    fn test_max_features_cap() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.0), Some(2));
        // All features pass threshold=0, but max_features=2
        let importances = array![0.3, 0.5, 0.1, 0.7];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 2);
        // Should keep top-2: indices 1 (0.5) and 3 (0.7)
        assert_eq!(fitted.selected_indices(), &[1, 3]);
    }

    #[test]
    fn test_max_features_not_needed() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.4), Some(5));
        let importances = array![0.1, 0.5, 0.4];
        let fitted = sel.fit(&importances, &()).unwrap();
        // Only 2 pass threshold, max_features=5 doesn't limit
        assert_eq!(fitted.n_features_selected(), 2);
    }

    #[test]
    fn test_empty_importances_error() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
        let importances: Array1<f64> = Array1::zeros(0);
        assert!(sel.fit(&importances, &()).is_err());
    }

    #[test]
    fn test_shape_mismatch_on_transform() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
        let importances = array![0.5, 0.5];
        let fitted = sel.fit(&importances, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]]; // 3 cols, 2 expected
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_threshold_value_accessor() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(0.42), None);
        let importances = array![0.1, 0.5];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_abs_diff_eq!(fitted.threshold_value(), 0.42, epsilon = 1e-15);
    }

    #[test]
    fn test_default() {
        let sel = SelectFromModelExt::<f64>::default();
        assert_eq!(sel.threshold_strategy(), ThresholdStrategy::Mean);
        assert_eq!(sel.max_features(), None);
    }

    #[test]
    fn test_pipeline_integration() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean, None);
        let importances = array![0.1, 0.9];
        let fitted = sel.fit(&importances, &()).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let fitted_box = fitted.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_f32() {
        let sel = SelectFromModelExt::<f32>::new(ThresholdStrategy::Mean, None);
        let importances: Array1<f32> = array![0.1f32, 0.5, 0.4];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 2);
    }

    #[test]
    fn test_none_selected_high_threshold() {
        let sel = SelectFromModelExt::<f64>::new(ThresholdStrategy::Value(10.0), None);
        let importances = array![0.1, 0.5, 0.4];
        let fitted = sel.fit(&importances, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 0);
        let x = array![[1.0, 2.0, 3.0]];
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 0);
        assert_eq!(out.nrows(), 1);
    }
}
