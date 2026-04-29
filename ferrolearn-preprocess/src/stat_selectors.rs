//! Statistical-test-based feature selectors.
//!
//! Three selectors that choose features based on p-values obtained from a
//! statistical test (e.g., ANOVA F-test, chi-squared test):
//!
//! - [`SelectFpr`] — **False Positive Rate**: selects every feature whose
//!   p-value is below `alpha`.
//! - [`SelectFdr`] — **False Discovery Rate**: applies the Benjamini-Hochberg
//!   procedure to control the expected proportion of false positives.
//! - [`SelectFwe`] — **Family-Wise Error**: applies the Bonferroni correction
//!   (`alpha / n_features`) to control the probability of any false positive.
//!
//! All three take a pre-computed vector of p-values (one per feature) at fit
//! time, allowing integration with any upstream scoring function.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

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

/// Validate common inputs for all three selectors.
fn validate_inputs(n_features: usize, alpha: f64) -> Result<(), FerroError> {
    if n_features == 0 {
        return Err(FerroError::InvalidParameter {
            name: "p_values".into(),
            reason: "p-value vector must not be empty".into(),
        });
    }
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("alpha must be in (0, 1], got {alpha}"),
        });
    }
    Ok(())
}

// ===========================================================================
// SelectFpr — False Positive Rate
// ===========================================================================

/// Select features with p-values below `alpha`.
///
/// A feature is selected if its p-value is strictly less than `alpha`.
/// This controls the per-feature false positive rate but does not adjust
/// for multiple comparisons.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::stat_selectors::SelectFpr;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let sel = SelectFpr::<f64>::new(0.05);
/// let p_values = array![0.01, 0.5, 0.03, 0.9];
/// let fitted = sel.fit(&p_values, &()).unwrap();
/// // Features 0 (p=0.01) and 2 (p=0.03) are below alpha=0.05
/// assert_eq!(fitted.selected_indices(), &[0, 2]);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SelectFpr<F> {
    /// Significance threshold.
    alpha: f64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectFpr<F> {
    /// Create a new `SelectFpr` with the given significance level.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the significance level.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// A fitted `SelectFpr` holding the selected indices.
#[derive(Debug, Clone)]
pub struct FittedSelectFpr<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// P-values supplied during fitting.
    p_values: Array1<F>,
    /// Indices of selected columns (sorted).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectFpr<F> {
    /// Return the p-values.
    #[must_use]
    pub fn p_values(&self) -> &Array1<F> {
        &self.p_values
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

impl<F: Float + Send + Sync + 'static> Fit<Array1<F>, ()> for SelectFpr<F> {
    type Fitted = FittedSelectFpr<F>;
    type Error = FerroError;

    /// Fit by selecting features whose p-value is below `alpha`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if p-values are empty or alpha is
    ///   not in `(0, 1]`.
    fn fit(&self, x: &Array1<F>, _y: &()) -> Result<FittedSelectFpr<F>, FerroError> {
        let n = x.len();
        validate_inputs(n, self.alpha)?;

        let alpha_f = F::from(self.alpha).unwrap_or_else(F::zero);
        let selected_indices: Vec<usize> = x
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p < alpha_f)
            .map(|(j, _)| j)
            .collect();

        Ok(FittedSelectFpr {
            n_features_in: n,
            p_values: x.clone(),
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectFpr<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if column count does not match.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSelectFpr::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ===========================================================================
// SelectFdr — False Discovery Rate (Benjamini-Hochberg)
// ===========================================================================

/// Select features controlling the false discovery rate via the
/// Benjamini-Hochberg procedure.
///
/// Features are sorted by p-value. Feature *i* (0-indexed, sorted ascending)
/// is selected if `p_value[i] <= alpha * (i+1) / n_features`. All features
/// with rank at or below the highest qualifying rank are selected.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::stat_selectors::SelectFdr;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let sel = SelectFdr::<f64>::new(0.05);
/// let p_values = array![0.01, 0.5, 0.03, 0.9];
/// let fitted = sel.fit(&p_values, &()).unwrap();
/// assert!(fitted.selected_indices().contains(&0));
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SelectFdr<F> {
    /// Target false discovery rate.
    alpha: f64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectFdr<F> {
    /// Create a new `SelectFdr` with the given FDR level.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the FDR level.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// A fitted `SelectFdr` holding the selected indices.
#[derive(Debug, Clone)]
pub struct FittedSelectFdr<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// P-values supplied during fitting.
    p_values: Array1<F>,
    /// Indices of selected columns (sorted in original order).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectFdr<F> {
    /// Return the p-values.
    #[must_use]
    pub fn p_values(&self) -> &Array1<F> {
        &self.p_values
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

impl<F: Float + Send + Sync + 'static> Fit<Array1<F>, ()> for SelectFdr<F> {
    type Fitted = FittedSelectFdr<F>;
    type Error = FerroError;

    /// Fit using the Benjamini-Hochberg procedure.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if p-values are empty or alpha is
    ///   not in `(0, 1]`.
    fn fit(&self, x: &Array1<F>, _y: &()) -> Result<FittedSelectFdr<F>, FerroError> {
        let n = x.len();
        validate_inputs(n, self.alpha)?;

        let alpha_f = F::from(self.alpha).unwrap_or_else(F::zero);
        let n_f = F::from(n).unwrap_or_else(F::one);

        // Sort features by p-value (ascending), keeping original indices
        let mut ranked: Vec<(usize, F)> = x.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find the largest rank k where p_(k) <= alpha * (k+1) / n
        let mut max_qualifying_rank: Option<usize> = None;
        for (rank, &(_, p_val)) in ranked.iter().enumerate() {
            let bh_threshold = alpha_f * F::from(rank + 1).unwrap_or_else(F::one) / n_f;
            if p_val <= bh_threshold {
                max_qualifying_rank = Some(rank);
            }
        }

        // Select all features at or below the max qualifying rank
        let mut selected_indices: Vec<usize> = match max_qualifying_rank {
            Some(max_rank) => ranked[..=max_rank]
                .iter()
                .map(|&(idx, _)| idx)
                .collect(),
            None => Vec::new(),
        };
        selected_indices.sort_unstable();

        Ok(FittedSelectFdr {
            n_features_in: n,
            p_values: x.clone(),
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectFdr<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if column count does not match.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSelectFdr::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ===========================================================================
// SelectFwe — Family-Wise Error (Bonferroni)
// ===========================================================================

/// Select features controlling the family-wise error rate via the
/// Bonferroni correction.
///
/// A feature is selected if its p-value is strictly less than
/// `alpha / n_features`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::stat_selectors::SelectFwe;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let sel = SelectFwe::<f64>::new(0.05);
/// let p_values = array![0.001, 0.5, 0.03, 0.9];
/// let fitted = sel.fit(&p_values, &()).unwrap();
/// // Bonferroni threshold = 0.05/4 = 0.0125; only feature 0 qualifies
/// assert_eq!(fitted.selected_indices(), &[0]);
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SelectFwe<F> {
    /// Significance level before Bonferroni correction.
    alpha: f64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectFwe<F> {
    /// Create a new `SelectFwe` with the given significance level.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the significance level.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// A fitted `SelectFwe` holding the selected indices.
#[derive(Debug, Clone)]
pub struct FittedSelectFwe<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// P-values supplied during fitting.
    p_values: Array1<F>,
    /// Indices of selected columns (sorted).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectFwe<F> {
    /// Return the p-values.
    #[must_use]
    pub fn p_values(&self) -> &Array1<F> {
        &self.p_values
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

impl<F: Float + Send + Sync + 'static> Fit<Array1<F>, ()> for SelectFwe<F> {
    type Fitted = FittedSelectFwe<F>;
    type Error = FerroError;

    /// Fit using the Bonferroni correction: `p < alpha / n_features`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if p-values are empty or alpha is
    ///   not in `(0, 1]`.
    fn fit(&self, x: &Array1<F>, _y: &()) -> Result<FittedSelectFwe<F>, FerroError> {
        let n = x.len();
        validate_inputs(n, self.alpha)?;

        let adjusted_alpha = self.alpha / n as f64;
        let adjusted_alpha_f = F::from(adjusted_alpha).unwrap_or_else(F::zero);

        let selected_indices: Vec<usize> = x
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p < adjusted_alpha_f)
            .map(|(j, _)| j)
            .collect();

        Ok(FittedSelectFwe {
            n_features_in: n,
            p_values: x.clone(),
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectFwe<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if column count does not match.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSelectFwe::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ========================================================================
    // SelectFpr tests
    // ========================================================================

    #[test]
    fn test_fpr_selects_below_alpha() {
        let sel = SelectFpr::<f64>::new(0.05);
        let p = array![0.01, 0.5, 0.03, 0.9];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0, 2]);
    }

    #[test]
    fn test_fpr_none_below_alpha() {
        let sel = SelectFpr::<f64>::new(0.001);
        let p = array![0.01, 0.5, 0.03];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 0);
    }

    #[test]
    fn test_fpr_all_below_alpha() {
        let sel = SelectFpr::<f64>::new(0.99);
        let p = array![0.01, 0.5, 0.03];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 3);
    }

    #[test]
    fn test_fpr_transform() {
        let sel = SelectFpr::<f64>::new(0.05);
        let p = array![0.01, 0.5, 0.03];
        let fitted = sel.fit(&p, &()).unwrap();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2); // features 0 and 2
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 3.0);
    }

    #[test]
    fn test_fpr_empty_error() {
        let sel = SelectFpr::<f64>::new(0.05);
        let p: Array1<f64> = Array1::zeros(0);
        assert!(sel.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fpr_invalid_alpha() {
        let sel = SelectFpr::<f64>::new(0.0);
        let p = array![0.01];
        assert!(sel.fit(&p, &()).is_err());

        let sel2 = SelectFpr::<f64>::new(1.5);
        assert!(sel2.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fpr_shape_mismatch() {
        let sel = SelectFpr::<f64>::new(0.05);
        let p = array![0.01, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_fpr_accessor() {
        let sel = SelectFpr::<f64>::new(0.05);
        assert_eq!(sel.alpha(), 0.05);
    }

    #[test]
    fn test_fpr_p_values_accessor() {
        let sel = SelectFpr::<f64>::new(0.05);
        let p = array![0.01, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.p_values().len(), 2);
    }

    // ========================================================================
    // SelectFdr tests (Benjamini-Hochberg)
    // ========================================================================

    #[test]
    fn test_fdr_basic() {
        let sel = SelectFdr::<f64>::new(0.05);
        // Sorted p-values: 0.01 (feat 0), 0.03 (feat 2), 0.5 (feat 1), 0.9 (feat 3)
        // BH thresholds: 0.05*1/4=0.0125, 0.05*2/4=0.025, 0.05*3/4=0.0375, 0.05*4/4=0.05
        // 0.01 <= 0.0125 ✓ (rank 0)
        // 0.03 <= 0.025  ✗ → but check all: max qualifying rank = 0
        let p = array![0.01, 0.5, 0.03, 0.9];
        let fitted = sel.fit(&p, &()).unwrap();
        assert!(fitted.selected_indices().contains(&0));
    }

    #[test]
    fn test_fdr_multiple_pass() {
        let sel = SelectFdr::<f64>::new(0.10);
        // Sorted: 0.005 (rank 0), 0.02 (rank 1), 0.04 (rank 2), 0.5 (rank 3)
        // BH: 0.1*1/4=0.025, 0.1*2/4=0.05, 0.1*3/4=0.075, 0.1*4/4=0.1
        // 0.005 <= 0.025 ✓
        // 0.02  <= 0.05  ✓
        // 0.04  <= 0.075 ✓ → max rank = 2 → select rank 0,1,2
        let p = array![0.02, 0.5, 0.005, 0.04];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 3);
        assert!(fitted.selected_indices().contains(&0)); // 0.02
        assert!(fitted.selected_indices().contains(&2)); // 0.005
        assert!(fitted.selected_indices().contains(&3)); // 0.04
    }

    #[test]
    fn test_fdr_none_selected() {
        let sel = SelectFdr::<f64>::new(0.001);
        let p = array![0.01, 0.5, 0.03];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 0);
    }

    #[test]
    fn test_fdr_transform() {
        let sel = SelectFdr::<f64>::new(0.10);
        let p = array![0.001, 0.5, 0.9];
        let fitted = sel.fit(&p, &()).unwrap();
        let x = array![[1.0, 2.0, 3.0]];
        let out = fitted.transform(&x).unwrap();
        // Feature 0 (p=0.001) selected: BH threshold = 0.1*1/3 ≈ 0.033
        assert!(out.ncols() >= 1);
    }

    #[test]
    fn test_fdr_empty_error() {
        let sel = SelectFdr::<f64>::new(0.05);
        let p: Array1<f64> = Array1::zeros(0);
        assert!(sel.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fdr_invalid_alpha() {
        let sel = SelectFdr::<f64>::new(0.0);
        let p = array![0.01];
        assert!(sel.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fdr_shape_mismatch() {
        let sel = SelectFdr::<f64>::new(0.05);
        let p = array![0.01, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_fdr_accessor() {
        let sel = SelectFdr::<f64>::new(0.05);
        assert_eq!(sel.alpha(), 0.05);
    }

    // ========================================================================
    // SelectFwe tests (Bonferroni)
    // ========================================================================

    #[test]
    fn test_fwe_basic() {
        let sel = SelectFwe::<f64>::new(0.05);
        // Bonferroni threshold = 0.05/4 = 0.0125
        let p = array![0.001, 0.5, 0.03, 0.9];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0]);
    }

    #[test]
    fn test_fwe_two_features() {
        let sel = SelectFwe::<f64>::new(0.10);
        // Bonferroni: 0.1/3 ≈ 0.0333
        let p = array![0.01, 0.02, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0, 1]);
    }

    #[test]
    fn test_fwe_none_selected() {
        let sel = SelectFwe::<f64>::new(0.01);
        // Bonferroni: 0.01/3 ≈ 0.00333
        let p = array![0.005, 0.5, 0.03];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.n_features_selected(), 0);
    }

    #[test]
    fn test_fwe_transform() {
        let sel = SelectFwe::<f64>::new(0.05);
        let p = array![0.001, 0.5, 0.9];
        let fitted = sel.fit(&p, &()).unwrap();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_eq!(out[[0, 0]], 1.0);
    }

    #[test]
    fn test_fwe_empty_error() {
        let sel = SelectFwe::<f64>::new(0.05);
        let p: Array1<f64> = Array1::zeros(0);
        assert!(sel.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fwe_invalid_alpha() {
        let sel = SelectFwe::<f64>::new(0.0);
        let p = array![0.01];
        assert!(sel.fit(&p, &()).is_err());
    }

    #[test]
    fn test_fwe_shape_mismatch() {
        let sel = SelectFwe::<f64>::new(0.05);
        let p = array![0.01, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_fwe_accessor() {
        let sel = SelectFwe::<f64>::new(0.05);
        assert_eq!(sel.alpha(), 0.05);
    }

    #[test]
    fn test_fwe_single_feature() {
        let sel = SelectFwe::<f64>::new(0.05);
        // Bonferroni: 0.05/1 = 0.05; p=0.01 < 0.05 ✓
        let p = array![0.01];
        let fitted = sel.fit(&p, &()).unwrap();
        assert_eq!(fitted.selected_indices(), &[0]);
    }

    #[test]
    fn test_fwe_f32() {
        let sel = SelectFwe::<f32>::new(0.05);
        let p: Array1<f32> = array![0.001f32, 0.5];
        let fitted = sel.fit(&p, &()).unwrap();
        // Bonferroni: 0.05/2 = 0.025; p=0.001 < 0.025 ✓
        assert_eq!(fitted.selected_indices(), &[0]);
    }
}
