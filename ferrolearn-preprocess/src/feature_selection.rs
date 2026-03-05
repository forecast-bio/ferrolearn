//! Feature selection transformers.
//!
//! This module provides three feature selection strategies:
//!
//! - [`VarianceThreshold`] — remove features whose variance falls below a
//!   configurable threshold (default 0.0 removes zero-variance features).
//! - [`SelectKBest`] — keep the *K* features with the highest ANOVA F-scores
//!   computed against a class label vector.
//! - [`SelectFromModel`] — keep features whose importance weight (provided by
//!   a previously fitted model) exceeds a configurable threshold.
//!
//! All three implement the standard ferrolearn `Fit` / `Transform` pattern
//! and integrate with the dynamic [`ferrolearn_core::pipeline::Pipeline`].

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Shared helper: collect selected columns
// ---------------------------------------------------------------------------

/// Build a new `Array2<F>` containing only the columns listed in `indices`.
///
/// Columns are emitted in the order they appear in `indices`.
fn select_columns<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let nrows = x.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        // Return empty matrix with correct row count
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

// ===========================================================================
// VarianceThreshold
// ===========================================================================

/// An unfitted variance-threshold feature selector.
///
/// During fitting the population variance of every column is computed (NaN
/// values are treated as zero — use an imputer upstream if needed).  Columns
/// whose variance is *less than or equal to* the configured threshold are
/// discarded during transformation.
///
/// The default threshold is `0.0`, which removes features with exactly zero
/// variance (i.e. constant columns).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_selection::VarianceThreshold;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let sel = VarianceThreshold::<f64>::new(0.0);
/// // Column 1 is constant — will be removed
/// let x = array![[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]];
/// let fitted = sel.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct VarianceThreshold<F> {
    /// Features with variance strictly less than this threshold are removed.
    threshold: F,
}

impl<F: Float + Send + Sync + 'static> VarianceThreshold<F> {
    /// Create a new `VarianceThreshold` with the given threshold.
    ///
    /// Pass `F::zero()` (the default) to remove only constant features.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `threshold` is negative.
    pub fn new(threshold: F) -> Self {
        Self { threshold }
    }

    /// Return the variance threshold.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }
}

impl<F: Float + Send + Sync + 'static> Default for VarianceThreshold<F> {
    fn default() -> Self {
        Self::new(F::zero())
    }
}

// ---------------------------------------------------------------------------
// FittedVarianceThreshold
// ---------------------------------------------------------------------------

/// A fitted variance-threshold selector holding the selected column indices
/// and the per-column variances observed during fitting.
///
/// Created by calling [`Fit::fit`] on a [`VarianceThreshold`].
#[derive(Debug, Clone)]
pub struct FittedVarianceThreshold<F> {
    /// Column indices (into the *original* feature matrix) that were selected.
    selected_indices: Vec<usize>,
    /// Per-column population variances computed during fitting.
    variances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedVarianceThreshold<F> {
    /// Return the indices of the selected columns.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Return the per-column variances computed during fitting.
    #[must_use]
    pub fn variances(&self) -> &Array1<F> {
        &self.variances
    }
}

// ---------------------------------------------------------------------------
// Trait implementations — VarianceThreshold
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for VarianceThreshold<F> {
    type Fitted = FittedVarianceThreshold<F>;
    type Error = FerroError;

    /// Fit by computing per-column population variances.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// Returns [`FerroError::InvalidParameter`] if the threshold is negative.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedVarianceThreshold<F>, FerroError> {
        if self.threshold < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "threshold".into(),
                reason: "variance threshold must be non-negative".into(),
            });
        }
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "VarianceThreshold::fit".into(),
            });
        }

        let n = F::from(n_samples).unwrap_or(F::one());
        let n_features = x.ncols();
        let mut variances = Array1::zeros(n_features);
        let mut selected_indices = Vec::new();

        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n;
            let var = col
                .iter()
                .copied()
                .map(|v| (v - mean) * (v - mean))
                .fold(F::zero(), |acc, v| acc + v)
                / n;
            variances[j] = var;
            if var > self.threshold {
                selected_indices.push(j);
            }
        }

        Ok(FittedVarianceThreshold {
            selected_indices,
            variances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedVarianceThreshold<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected (high-variance) columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_original = self.variances.len();
        if x.ncols() != n_original {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_original],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedVarianceThreshold::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration — VarianceThreshold (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for VarianceThreshold<F> {
    /// Fit using the pipeline interface; `y` is ignored.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedVarianceThreshold<F> {
    /// Transform using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ===========================================================================
// SelectKBest
// ===========================================================================

/// Scoring function variants for [`SelectKBest`].
///
/// Currently only ANOVA F-value scoring is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunc {
    /// ANOVA F-value: ratio of between-class variance to within-class variance.
    ///
    /// This is analogous to scikit-learn's `f_classif`.
    FClassif,
}

/// An unfitted K-best feature selector.
///
/// Requires class labels (`Array1<usize>`) at fit time to compute per-feature
/// ANOVA F-scores.  The top *K* features (by score) are retained.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_selection::{SelectKBest, ScoreFunc};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::{array, Array1};
///
/// let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
/// let x = array![[1.0, 10.0], [1.0, 20.0], [2.0, 10.0], [2.0, 20.0]];
/// let y: Array1<usize> = array![0, 0, 1, 1];
/// let fitted = sel.fit(&x, &y).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct SelectKBest<F> {
    /// Number of top-scoring features to keep.
    k: usize,
    /// The scoring function to use.
    score_func: ScoreFunc,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectKBest<F> {
    /// Create a new `SelectKBest` selector.
    ///
    /// # Parameters
    ///
    /// - `k` — the number of features to retain.
    /// - `score_func` — the scoring function; currently only
    ///   [`ScoreFunc::FClassif`] is available.
    #[must_use]
    pub fn new(k: usize, score_func: ScoreFunc) -> Self {
        Self {
            k,
            score_func,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return *k*.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the score function.
    #[must_use]
    pub fn score_func(&self) -> ScoreFunc {
        self.score_func
    }
}

// ---------------------------------------------------------------------------
// FittedSelectKBest
// ---------------------------------------------------------------------------

/// A fitted K-best selector holding per-feature scores and selected indices.
///
/// Created by calling [`Fit::fit`] on a [`SelectKBest`].
#[derive(Debug, Clone)]
pub struct FittedSelectKBest<F> {
    /// The original number of features (used for shape checking on transform).
    n_features_in: usize,
    /// Per-feature ANOVA F-scores computed during fitting.
    scores: Array1<F>,
    /// Indices of the selected columns, sorted in original column order.
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectKBest<F> {
    /// Return the per-feature F-scores computed during fitting.
    #[must_use]
    pub fn scores(&self) -> &Array1<F> {
        &self.scores
    }

    /// Return the indices of the selected columns.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

// ---------------------------------------------------------------------------
// ANOVA F-value helper
// ---------------------------------------------------------------------------

/// Compute per-feature ANOVA F-scores given a feature matrix `x` and integer
/// class labels `y`.
///
/// For each feature column the F-statistic is:
///
/// ```text
/// F = (between-class variance / (n_classes - 1))
///   / (within-class variance  / (n_samples - n_classes))
/// ```
///
/// Features for which the within-class variance is zero (perfectly separable)
/// get an F-score of `F::infinity()`.  Features that have zero between-class
/// variance get a score of `F::zero()`.
fn anova_f_scores<F: Float>(x: &Array2<F>, y: &Array1<usize>) -> Vec<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Collect unique classes and build per-class row-index lists.
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }
    let n_classes = class_indices.len();

    let mut scores = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let col = x.column(j);

        // Overall mean of this feature
        let grand_mean =
            col.iter().copied().fold(F::zero(), |acc, v| acc + v) / F::from(n_samples).unwrap();

        // Between-class sum of squares: sum_k n_k * (mean_k - grand_mean)^2
        let mut ss_between = F::zero();
        // Within-class sum of squares: sum_k sum_{i in k} (x_i - mean_k)^2
        let mut ss_within = F::zero();

        for rows in class_indices.values() {
            let n_k = F::from(rows.len()).unwrap();
            let class_mean = rows
                .iter()
                .map(|&i| col[i])
                .fold(F::zero(), |acc, v| acc + v)
                / n_k;
            let diff = class_mean - grand_mean;
            ss_between = ss_between + n_k * diff * diff;
            for &i in rows {
                let d = col[i] - class_mean;
                ss_within = ss_within + d * d;
            }
        }

        let df_between = F::from(n_classes.saturating_sub(1)).unwrap();
        let df_within = F::from(n_samples.saturating_sub(n_classes)).unwrap();

        let f = if df_between == F::zero() || df_within == F::zero() {
            F::zero()
        } else {
            let ms_between = ss_between / df_between;
            let ms_within = ss_within / df_within;
            if ms_within == F::zero() {
                F::infinity()
            } else {
                ms_between / ms_within
            }
        };

        scores.push(f);
    }

    scores
}

// ---------------------------------------------------------------------------
// Trait implementations — SelectKBest
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for SelectKBest<F> {
    type Fitted = FittedSelectKBest<F>;
    type Error = FerroError;

    /// Fit by computing per-feature ANOVA F-scores against the class labels.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::InvalidParameter`] if `k` exceeds the number of features.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers
    ///   of rows.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedSelectKBest<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SelectKBest::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "SelectKBest::fit — y must have the same length as x has rows".into(),
            });
        }
        let n_features = x.ncols();
        if self.k > n_features {
            return Err(FerroError::InvalidParameter {
                name: "k".into(),
                reason: format!(
                    "k ({}) cannot exceed the number of features ({})",
                    self.k, n_features
                ),
            });
        }

        let raw_scores = match self.score_func {
            ScoreFunc::FClassif => anova_f_scores(x, y),
        };

        let scores = Array1::from_vec(raw_scores.clone());

        // Determine the top-k indices (stable: break ties by preferring the
        // lower column index so results are deterministic).
        let mut ranked: Vec<usize> = (0..n_features).collect();
        ranked.sort_by(|&a, &b| {
            raw_scores[b]
                .partial_cmp(&raw_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                // Tie: keep lower column index
                .then(a.cmp(&b))
        });

        let mut selected_indices: Vec<usize> = ranked[..self.k].to_vec();
        // Return in original column order for a stable output layout
        selected_indices.sort_unstable();

        Ok(FittedSelectKBest {
            n_features_in: n_features,
            scores,
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectKBest<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the K selected columns.
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
                context: "FittedSelectKBest::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration — SelectKBest (generic)
//
// NOTE: The pipeline interface uses a *fixed* `y = Array1<f64>`, so we cannot
// use the actual class-label vector from the pipeline.  We therefore refit
// using the pipeline `y` converted to `usize` labels by rounding.
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for SelectKBest<F> {
    /// Fit using the pipeline interface.
    ///
    /// The continuous `y` labels are rounded to `usize` class indices.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.round().to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedSelectKBest<F> {
    /// Transform using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ===========================================================================
// SelectFromModel
// ===========================================================================

/// A feature selector driven by external feature-importance weights.
///
/// The importance vector is typically obtained from a fitted model (e.g. a
/// decision-tree model's `feature_importances_` field).  Features whose
/// importance is *strictly greater than or equal to* the threshold are kept.
///
/// The default threshold is the **mean importance** of all features.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_selection::SelectFromModel;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let importances = array![0.1, 0.5, 0.4];
/// let sel = SelectFromModel::<f64>::new_from_importances(&importances, None).unwrap();
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let out = sel.transform(&x).unwrap();
/// // Mean importance = (0.1+0.5+0.4)/3 ≈ 0.333; columns 1 and 2 are kept
/// assert_eq!(out.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SelectFromModel<F> {
    /// Importance weight for each feature.
    importances: Array1<F>,
    /// The threshold: features with importance >= threshold are kept.
    threshold: F,
    /// Indices of selected features (original column order).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> SelectFromModel<F> {
    /// Create a `SelectFromModel` from a pre-computed importance vector.
    ///
    /// # Parameters
    ///
    /// - `importances` — one importance weight per feature.
    /// - `threshold` — optional explicit threshold; if `None` the mean
    ///   importance is used.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `importances` is empty.
    pub fn new_from_importances(
        importances: &Array1<F>,
        threshold: Option<F>,
    ) -> Result<Self, FerroError> {
        let n = importances.len();
        if n == 0 {
            return Err(FerroError::InvalidParameter {
                name: "importances".into(),
                reason: "importance vector must not be empty".into(),
            });
        }

        let thr = threshold.unwrap_or_else(|| {
            importances
                .iter()
                .copied()
                .fold(F::zero(), |acc, v| acc + v)
                / F::from(n).unwrap_or(F::one())
        });

        let selected_indices: Vec<usize> = importances
            .iter()
            .enumerate()
            .filter(|&(_, &imp)| imp >= thr)
            .map(|(j, _)| j)
            .collect();

        Ok(Self {
            importances: importances.clone(),
            threshold: thr,
            selected_indices,
        })
    }

    /// Return the threshold used to select features.
    #[must_use]
    pub fn threshold(&self) -> F {
        self.threshold
    }

    /// Return the importance vector supplied at construction time.
    #[must_use]
    pub fn importances(&self) -> &Array1<F> {
        &self.importances
    }

    /// Return the indices of the selected columns.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

// ---------------------------------------------------------------------------
// Trait implementation — SelectFromModel
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SelectFromModel<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the columns whose importance exceeds
    /// the threshold.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the length of the importance vector supplied at construction.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.importances.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "SelectFromModel::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration — SelectFromModel (generic)
//
// `SelectFromModel` is already "fitted" (importance weights are provided at
// construction time), so `fit_pipeline` merely boxes `self.clone()`.
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for SelectFromModel<F> {
    /// Clone the selector and box it as a fitted pipeline transformer.
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

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for SelectFromModel<F> {
    /// Transform using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ========================================================================
    // VarianceThreshold tests
    // ========================================================================

    #[test]
    fn test_variance_threshold_removes_constant_column() {
        let sel = VarianceThreshold::<f64>::new(0.0);
        // Column 1 is constant (all 7.0)
        let x = array![[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]];
        let fitted = sel.fit(&x, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0usize]);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        // Column 0 values preserved
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[1, 0]], 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_variance_threshold_keeps_all_when_above() {
        let sel = VarianceThreshold::<f64>::new(0.0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = sel.fit(&x, &()).unwrap();
        assert_eq!(fitted.selected_indices().len(), 2);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_variance_threshold_custom_threshold() {
        let sel = VarianceThreshold::<f64>::new(1.5);
        // Column 0: values [1,2,3], variance = 2/3 ≈ 0.667 → removed
        // Column 1: values [10,20,30], variance = 200/3 ≈ 66.7 → kept
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let fitted = sel.fit(&x, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[1usize]);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_variance_threshold_stores_variances() {
        let sel = VarianceThreshold::<f64>::default();
        let x = array![[0.0], [0.0], [0.0]]; // constant → var = 0
        let fitted = sel.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.variances()[0], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_variance_threshold_zero_rows_error() {
        let sel = VarianceThreshold::<f64>::new(0.0);
        let x: Array2<f64> = Array2::zeros((0, 2));
        assert!(sel.fit(&x, &()).is_err());
    }

    #[test]
    fn test_variance_threshold_negative_threshold_error() {
        let sel = VarianceThreshold::<f64>::new(-0.1);
        let x = array![[1.0], [2.0]];
        assert!(sel.fit(&x, &()).is_err());
    }

    #[test]
    fn test_variance_threshold_shape_mismatch_on_transform() {
        let sel = VarianceThreshold::<f64>::new(0.0);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = sel.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_variance_threshold_all_constant_columns() {
        let sel = VarianceThreshold::<f64>::new(0.0);
        let x = array![[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]];
        let fitted = sel.fit(&x, &()).unwrap();
        // Both columns are constant: both removed
        assert_eq!(fitted.selected_indices().len(), 0);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 0);
        assert_eq!(out.nrows(), 3);
    }

    #[test]
    fn test_variance_threshold_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let sel = VarianceThreshold::<f64>::new(0.0);
        let x = array![[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]];
        let y = ndarray::array![0.0, 1.0, 0.0];
        let fitted_box = sel.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_variance_threshold_f32() {
        let sel = VarianceThreshold::<f32>::new(0.0f32);
        let x: Array2<f32> = array![[1.0f32, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let fitted = sel.fit(&x, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0usize]);
    }

    // ========================================================================
    // SelectKBest tests
    // ========================================================================

    #[test]
    fn test_select_k_best_selects_highest_scoring_feature() {
        // Feature 0 separates classes perfectly; feature 1 does not.
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[1.0, 5.0], [1.0, 6.0], [10.0, 5.0], [10.0, 6.0]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        // Column 0 should be selected (high F-score)
        assert_eq!(fitted.selected_indices(), &[0usize]);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_select_k_best_k_equals_n_features_keeps_all() {
        let sel = SelectKBest::<f64>::new(2, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y: Array1<usize> = array![0, 1, 0];
        let fitted = sel.fit(&x, &y).unwrap();
        assert_eq!(fitted.selected_indices().len(), 2);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_select_k_best_scores_stored() {
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [1.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        assert_eq!(fitted.scores().len(), 2);
    }

    #[test]
    fn test_select_k_best_zero_rows_error() {
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x: Array2<f64> = Array2::zeros((0, 3));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_k_best_k_exceeds_n_features_error() {
        let sel = SelectKBest::<f64>::new(5, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_k_best_y_length_mismatch_error() {
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0]; // wrong length
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_k_best_shape_mismatch_on_transform() {
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_select_k_best_selected_indices_in_column_order() {
        // Both features selected — indices should be [0, 1] not reversed
        let sel = SelectKBest::<f64>::new(2, ScoreFunc::FClassif);
        let x = array![[1.0, 100.0], [2.0, 200.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let indices = fitted.selected_indices();
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_select_k_best_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[1.0, 5.0], [1.0, 6.0], [10.0, 5.0], [10.0, 6.0]];
        let y = ndarray::array![0.0, 0.0, 1.0, 1.0];
        let fitted_box = sel.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        assert_eq!(out.ncols(), 1);
    }

    #[test]
    fn test_select_k_best_f_score_zero_within_class_variance() {
        // Perfectly separating feature → F should be infinity
        let sel = SelectKBest::<f64>::new(1, ScoreFunc::FClassif);
        let x = array![[0.0], [0.0], [10.0], [10.0]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        assert!(fitted.scores()[0].is_infinite());
    }

    // ========================================================================
    // SelectFromModel tests
    // ========================================================================

    #[test]
    fn test_select_from_model_mean_threshold() {
        // Mean importance = (0.1 + 0.5 + 0.4) / 3 ≈ 0.333
        // Features 1 (0.5) and 2 (0.4) are >= threshold; feature 0 (0.1) is not
        let importances = array![0.1, 0.5, 0.4];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, None).unwrap();
        assert_eq!(sel.selected_indices(), &[1usize, 2]);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = sel.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 1]], 3.0, epsilon = 1e-15);
    }

    #[test]
    fn test_select_from_model_explicit_threshold() {
        let importances = array![0.1, 0.5, 0.4];
        // Only feature 1 (0.5 >= 0.45) is selected
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, Some(0.45)).unwrap();
        assert_eq!(sel.selected_indices(), &[1usize]);
        let x = array![[1.0, 2.0, 3.0]];
        let out = sel.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_select_from_model_all_selected_when_threshold_zero() {
        let importances = array![0.1, 0.2, 0.3];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, Some(0.0)).unwrap();
        assert_eq!(sel.selected_indices().len(), 3);
    }

    #[test]
    fn test_select_from_model_none_selected_when_threshold_high() {
        let importances = array![0.1, 0.2, 0.3];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, Some(1.0)).unwrap();
        assert_eq!(sel.selected_indices().len(), 0);
        let x = array![[1.0, 2.0, 3.0]];
        let out = sel.transform(&x).unwrap();
        assert_eq!(out.ncols(), 0);
    }

    #[test]
    fn test_select_from_model_empty_importances_error() {
        let importances: Array1<f64> = Array1::zeros(0);
        assert!(SelectFromModel::<f64>::new_from_importances(&importances, None).is_err());
    }

    #[test]
    fn test_select_from_model_shape_mismatch_on_transform() {
        let importances = array![0.3, 0.7];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, None).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]]; // 3 cols, but 2 features expected
        assert!(sel.transform(&x_bad).is_err());
    }

    #[test]
    fn test_select_from_model_threshold_accessor() {
        let importances = array![0.3, 0.7];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, Some(0.5)).unwrap();
        assert_abs_diff_eq!(sel.threshold(), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_select_from_model_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;
        let importances = array![0.1, 0.9];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, None).unwrap();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = ndarray::array![0.0, 1.0];
        let fitted_box = sel.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        // Mean importance = 0.5; only feature 1 (0.9 >= 0.5) kept
        assert_eq!(out.ncols(), 1);
        assert_abs_diff_eq!(out[[0, 0]], 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_select_from_model_importances_accessor() {
        let importances = array![0.2, 0.8];
        let sel = SelectFromModel::<f64>::new_from_importances(&importances, None).unwrap();
        assert_abs_diff_eq!(sel.importances()[0], 0.2, epsilon = 1e-15);
        assert_abs_diff_eq!(sel.importances()[1], 0.8, epsilon = 1e-15);
    }
}
