//! Recursive Feature Elimination (RFE) and RFE with Cross-Validation (RFECV).
//!
//! [`RFE`] recursively removes the least-important features, ranking features
//! by their importance at each elimination step. The importance is determined by
//! an external importance vector that the user supplies via a callback.
//!
//! [`RFECV`] extends RFE by using cross-validation to find the optimal number
//! of features to retain.
//!
//! Because `ferrolearn-preprocess` cannot depend on estimator crates (to avoid
//! circular dependencies), these implementations accept feature importance
//! vectors directly rather than wrapping fitted estimators.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::{Array1, Array2};
use num_traits::Float;

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

// ===========================================================================
// RFE
// ===========================================================================

/// Recursive Feature Elimination.
///
/// Starting from all features, repeatedly removes the `step` least-important
/// features until `n_features_to_select` features remain. The ranking is
/// determined by the importance vector supplied at construction.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::rfe::RFE;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// // Feature importances: feature 0 is most important, feature 2 least
/// let importances = array![0.6, 0.3, 0.1];
/// let rfe = RFE::<f64>::new(&importances, 1, 1).unwrap();
/// assert_eq!(rfe.support(), &[true, false, false]);
/// assert_eq!(rfe.ranking(), &[1, 2, 3]);
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let out = rfe.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct RFE<F> {
    /// Feature ranking: `ranking[j]` is the rank of feature `j` (1 = selected).
    ranking: Vec<usize>,
    /// Boolean mask: `support[j]` is `true` if feature `j` is selected.
    support: Vec<bool>,
    /// Indices of the selected features (sorted).
    selected_indices: Vec<usize>,
    /// Original number of features.
    n_features_in: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> RFE<F> {
    /// Create a new `RFE` from pre-computed feature importances.
    ///
    /// # Parameters
    ///
    /// - `importances` — per-feature importance scores (higher = more important).
    /// - `n_features_to_select` — number of features to keep.
    /// - `step` — number of features to remove per iteration.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `importances` is empty, `step` is
    ///   zero, or `n_features_to_select` exceeds the number of features.
    pub fn new(
        importances: &Array1<F>,
        n_features_to_select: usize,
        step: usize,
    ) -> Result<Self, FerroError> {
        let n_features = importances.len();
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "importances".into(),
                reason: "importance vector must not be empty".into(),
            });
        }
        if step == 0 {
            return Err(FerroError::InvalidParameter {
                name: "step".into(),
                reason: "step must be at least 1".into(),
            });
        }
        if n_features_to_select == 0 || n_features_to_select > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_features_to_select".into(),
                reason: format!(
                    "n_features_to_select ({n_features_to_select}) must be in [1, {n_features}]"
                ),
            });
        }

        // Simulate the elimination process.
        // Track which round each feature is eliminated in; features removed in
        // the same step share the same rank. Selected features get rank 1,
        // features removed in the last elimination round get rank 2, etc.
        let mut ranking = vec![0usize; n_features];
        let mut remaining: Vec<usize> = (0..n_features).collect();
        let mut elimination_rounds: Vec<Vec<usize>> = Vec::new();

        // Working copy of importances
        let imp: Vec<F> = importances.iter().copied().collect();

        while remaining.len() > n_features_to_select {
            // Sort remaining by importance (ascending)
            remaining.sort_by(|&a, &b| {
                imp[a]
                    .partial_cmp(&imp[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Remove the `step` least important features
            let n_to_remove = step.min(remaining.len() - n_features_to_select);
            let removed: Vec<usize> = remaining[..n_to_remove].to_vec();
            elimination_rounds.push(removed);
            remaining = remaining[n_to_remove..].to_vec();
        }

        // Assign ranks: selected features get rank 1, features removed in the
        // last round get rank 2, second-to-last round gets rank 3, etc.
        for &idx in &remaining {
            ranking[idx] = 1;
        }
        for (round_idx, round) in elimination_rounds.iter().rev().enumerate() {
            let rank = round_idx + 2;
            for &idx in round {
                ranking[idx] = rank;
            }
        }

        let support: Vec<bool> = ranking.iter().map(|&r| r == 1).collect();
        let mut selected_indices: Vec<usize> = remaining;
        selected_indices.sort_unstable();

        Ok(Self {
            ranking,
            support,
            selected_indices,
            n_features_in: n_features,
            _marker: std::marker::PhantomData,
        })
    }

    /// Return the feature ranking (1 = best, higher = eliminated earlier).
    #[must_use]
    pub fn ranking(&self) -> &[usize] {
        &self.ranking
    }

    /// Return the boolean support mask.
    #[must_use]
    pub fn support(&self) -> &[bool] {
        &self.support
    }

    /// Return the indices of the selected features.
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

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for RFE<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected features.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features used at construction.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "RFE::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ===========================================================================
// RFECV
// ===========================================================================

/// Recursive Feature Elimination with Cross-Validation.
///
/// Like [`RFE`], but uses cross-validation scores to determine the optimal
/// number of features. The user supplies a vector of per-feature-count CV
/// scores (e.g., from running RFE with different `n_features_to_select`
/// values), and RFECV picks the number that maximises the score.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::rfe::RFECV;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// let importances = array![0.5, 0.3, 0.2];
/// // CV scores for selecting 1, 2, 3 features:
/// let cv_scores = vec![0.85, 0.95, 0.90];
/// let rfecv = RFECV::<f64>::new(&importances, &cv_scores, 1).unwrap();
/// // Best is 2 features (score 0.95)
/// assert_eq!(rfecv.n_features_selected(), 2);
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let out = rfecv.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct RFECV<F> {
    /// The underlying RFE with the optimal number of features.
    rfe: RFE<F>,
    /// CV scores for each number of features (1..=n_features).
    cv_scores: Vec<f64>,
    /// The optimal number of features (1-indexed).
    optimal_n_features: usize,
}

impl<F: Float + Send + Sync + 'static> RFECV<F> {
    /// Create a new `RFECV` from pre-computed importances and CV scores.
    ///
    /// # Parameters
    ///
    /// - `importances` — per-feature importance scores.
    /// - `cv_scores` — CV score for each possible number of features
    ///   (index 0 = 1 feature, index 1 = 2 features, ...).
    /// - `step` — features removed per iteration.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if inputs are empty or mismatched.
    pub fn new(
        importances: &Array1<F>,
        cv_scores: &[f64],
        step: usize,
    ) -> Result<Self, FerroError> {
        let n_features = importances.len();
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "importances".into(),
                reason: "importance vector must not be empty".into(),
            });
        }
        if cv_scores.len() != n_features {
            return Err(FerroError::InvalidParameter {
                name: "cv_scores".into(),
                reason: format!(
                    "cv_scores length ({}) must equal number of features ({})",
                    cv_scores.len(),
                    n_features
                ),
            });
        }

        // Find the optimal number of features (1-indexed)
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &score) in cv_scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        let optimal_n_features = best_idx + 1;

        let rfe = RFE::new(importances, optimal_n_features, step)?;

        Ok(Self {
            rfe,
            cv_scores: cv_scores.to_vec(),
            optimal_n_features,
        })
    }

    /// Return the CV scores.
    #[must_use]
    pub fn cv_scores(&self) -> &[f64] {
        &self.cv_scores
    }

    /// Return the optimal number of features.
    #[must_use]
    pub fn optimal_n_features(&self) -> usize {
        self.optimal_n_features
    }

    /// Return the number of selected features.
    #[must_use]
    pub fn n_features_selected(&self) -> usize {
        self.rfe.n_features_selected()
    }

    /// Return the feature ranking.
    #[must_use]
    pub fn ranking(&self) -> &[usize] {
        self.rfe.ranking()
    }

    /// Return the boolean support mask.
    #[must_use]
    pub fn support(&self) -> &[bool] {
        self.rfe.support()
    }

    /// Return the indices of the selected features.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        self.rfe.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for RFECV<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the optimally selected features.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if column count does not match.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.rfe.transform(x)
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

    // ========================================================================
    // RFE tests
    // ========================================================================

    #[test]
    fn test_rfe_basic_ranking() {
        // Importances: [0.6, 0.3, 0.1]
        // Select 1 feature, step 1
        // Round 1: remove feature 2 (lowest 0.1) → remaining [0, 1]
        // Round 2: remove feature 1 (lowest 0.3) → remaining [0]
        let imp = array![0.6, 0.3, 0.1];
        let rfe = RFE::<f64>::new(&imp, 1, 1).unwrap();
        assert_eq!(rfe.ranking(), &[1, 2, 3]);
        assert_eq!(rfe.support(), &[true, false, false]);
        assert_eq!(rfe.selected_indices(), &[0]);
    }

    #[test]
    fn test_rfe_select_two() {
        let imp = array![0.5, 0.3, 0.2];
        let rfe = RFE::<f64>::new(&imp, 2, 1).unwrap();
        assert_eq!(rfe.n_features_selected(), 2);
        // Feature 2 (0.2) should be eliminated first
        assert_eq!(rfe.ranking()[2], 2); // eliminated in round 1
        assert_eq!(rfe.ranking()[0], 1);
        assert_eq!(rfe.ranking()[1], 1);
    }

    #[test]
    fn test_rfe_step_two() {
        let imp = array![0.5, 0.3, 0.2, 0.1];
        // Select 2, step 2: remove 2 features at once
        let rfe = RFE::<f64>::new(&imp, 2, 2).unwrap();
        assert_eq!(rfe.n_features_selected(), 2);
        assert!(rfe.support()[0]);
        assert!(rfe.support()[1]);
        assert!(!rfe.support()[2]);
        assert!(!rfe.support()[3]);
    }

    #[test]
    fn test_rfe_transform() {
        let imp = array![0.6, 0.3, 0.1];
        let rfe = RFE::<f64>::new(&imp, 1, 1).unwrap();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = rfe.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        // Feature 0 is selected
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[1, 0]], 4.0, epsilon = 1e-15);
    }

    #[test]
    fn test_rfe_all_features_selected() {
        let imp = array![0.5, 0.3, 0.2];
        let rfe = RFE::<f64>::new(&imp, 3, 1).unwrap();
        assert_eq!(rfe.n_features_selected(), 3);
        assert!(rfe.support().iter().all(|&s| s));
    }

    #[test]
    fn test_rfe_empty_importances_error() {
        let imp: Array1<f64> = Array1::zeros(0);
        assert!(RFE::<f64>::new(&imp, 1, 1).is_err());
    }

    #[test]
    fn test_rfe_zero_step_error() {
        let imp = array![0.5, 0.3];
        assert!(RFE::<f64>::new(&imp, 1, 0).is_err());
    }

    #[test]
    fn test_rfe_n_features_too_large_error() {
        let imp = array![0.5, 0.3];
        assert!(RFE::<f64>::new(&imp, 5, 1).is_err());
    }

    #[test]
    fn test_rfe_n_features_zero_error() {
        let imp = array![0.5, 0.3];
        assert!(RFE::<f64>::new(&imp, 0, 1).is_err());
    }

    #[test]
    fn test_rfe_shape_mismatch_error() {
        let imp = array![0.5, 0.3];
        let rfe = RFE::<f64>::new(&imp, 1, 1).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(rfe.transform(&x_bad).is_err());
    }

    // ========================================================================
    // RFECV tests
    // ========================================================================

    #[test]
    fn test_rfecv_selects_optimal() {
        let imp = array![0.5, 0.3, 0.2];
        // Best CV score at 2 features
        let cv_scores = vec![0.85, 0.95, 0.90];
        let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
        assert_eq!(rfecv.optimal_n_features(), 2);
        assert_eq!(rfecv.n_features_selected(), 2);
    }

    #[test]
    fn test_rfecv_transform() {
        let imp = array![0.5, 0.3, 0.2];
        let cv_scores = vec![0.85, 0.95, 0.90];
        let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = rfecv.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_rfecv_cv_scores_accessor() {
        let imp = array![0.5, 0.3];
        let cv_scores = vec![0.9, 0.8];
        let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
        assert_eq!(rfecv.cv_scores(), &[0.9, 0.8]);
        // Best is 1 feature (score 0.9)
        assert_eq!(rfecv.optimal_n_features(), 1);
    }

    #[test]
    fn test_rfecv_mismatched_scores_error() {
        let imp = array![0.5, 0.3, 0.2];
        let cv_scores = vec![0.85, 0.95]; // wrong length
        assert!(RFECV::<f64>::new(&imp, &cv_scores, 1).is_err());
    }

    #[test]
    fn test_rfecv_empty_importances_error() {
        let imp: Array1<f64> = Array1::zeros(0);
        let cv_scores: Vec<f64> = vec![];
        assert!(RFECV::<f64>::new(&imp, &cv_scores, 1).is_err());
    }

    #[test]
    fn test_rfecv_ranking_and_support() {
        let imp = array![0.5, 0.3, 0.2];
        let cv_scores = vec![0.80, 0.95, 0.90];
        let rfecv = RFECV::<f64>::new(&imp, &cv_scores, 1).unwrap();
        assert_eq!(rfecv.n_features_selected(), 2);
        let support = rfecv.support();
        assert_eq!(support.iter().filter(|&&s| s).count(), 2);
    }
}
