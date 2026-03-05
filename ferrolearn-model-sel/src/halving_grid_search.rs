//! Successive halving hyperparameter search with cross-validation.
//!
//! [`HalvingGridSearchCV`] implements the successive-halving algorithm:
//!
//! 1. Start all candidate parameter sets on a small budget (subset) of data.
//! 2. Keep the top `1/factor` fraction of candidates.
//! 3. Multiply the data budget by `factor`.
//! 4. Repeat until the maximum budget is reached or only one candidate remains.
//!
//! This is more efficient than exhaustive grid search when many candidates are
//! clearly inferior on small data budgets.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{HalvingGridSearchCV, KFold, param_grid, ParamValue};
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
//!     let diff = y_true - y_pred;
//!     Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // let factory = |_: &_| Pipeline::new().estimator_step("est", Box::new(MyEst));
//! // let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64, 10.0_f64] };
//! // let mut hs = HalvingGridSearchCV::new(Box::new(factory), grid, Box::new(KFold::new(3)), neg_mse);
//! // hs.fit(&x, &y).unwrap();
//! // println!("Best: {:?}", hs.best_params());
//! ```

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ndarray::{Array1, Array2};

use crate::cross_validation::{CrossValidator, cross_val_score};
use crate::grid_search::CvResults;
use crate::param_grid::ParamSet;

// ---------------------------------------------------------------------------
// HalvingGridSearchCV
// ---------------------------------------------------------------------------

/// Successive-halving hyperparameter search using cross-validation.
///
/// Each *round* evaluates the surviving candidates on an increasing fraction of
/// the training data. Candidates are ranked by mean cross-validation score, and
/// the bottom `(factor-1)/factor` fraction are eliminated before the next round.
///
/// # Fields (set via builder methods)
///
/// - `factor` — halving factor (default 3).
/// - `min_resources` — smallest data budget per round (default
///   `ceil(n_samples / factor^ceil(log_factor(n_candidates)))`).
/// - `max_resources` — largest data budget (default `n_samples`).
/// - `aggressive_elimination` — when `true`, always eliminate until only one
///   candidate remains, even if `max_resources` has been reached (default
///   `false`).
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::{HalvingGridSearchCV, KFold, param_grid};
/// use ferrolearn_core::pipeline::Pipeline;
/// use ferrolearn_core::FerroError;
/// use ndarray::{Array1, Array2};
///
/// fn neg_mse(y: &Array1<f64>, p: &Array1<f64>) -> Result<f64, FerroError> {
///     let d = y - p; Ok(-d.mapv(|v| v*v).mean().unwrap_or(0.0))
/// }
/// // let mut hs = HalvingGridSearchCV::new(Box::new(|_| todo!()), grid, Box::new(KFold::new(3)), neg_mse);
/// // hs.fit(&x, &y).unwrap();
/// ```
pub struct HalvingGridSearchCV<'a> {
    /// Factory that builds a [`Pipeline`] from a [`ParamSet`].
    pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
    /// All parameter combinations to search.
    param_grid: Vec<ParamSet>,
    /// Cross-validator used to evaluate each combination on each budget.
    cv: Box<dyn CrossValidator>,
    /// Scoring function; higher is better.
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    /// Halving factor (default 3).
    factor: usize,
    /// Minimum number of samples per round (auto-computed if `None`).
    min_resources: Option<usize>,
    /// Maximum number of samples per round (defaults to `n_samples`).
    max_resources: Option<usize>,
    /// Whether to keep halving past `max_resources` until one candidate remains.
    aggressive_elimination: bool,
    /// Results populated after [`fit`](HalvingGridSearchCV::fit) is called.
    results: Option<CvResults>,
}

impl<'a> HalvingGridSearchCV<'a> {
    /// Create a new [`HalvingGridSearchCV`] with default options.
    ///
    /// # Parameters
    ///
    /// - `pipeline_factory` — closure that accepts a [`ParamSet`] and returns
    ///   an unfitted [`Pipeline`].
    /// - `param_grid` — the parameter combinations to search.
    /// - `cv` — the cross-validator.
    /// - `scoring` — scoring function; higher is better.
    pub fn new(
        pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
        param_grid: Vec<ParamSet>,
        cv: Box<dyn CrossValidator>,
        scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    ) -> Self {
        Self {
            pipeline_factory,
            param_grid,
            cv,
            scoring,
            factor: 3,
            min_resources: None,
            max_resources: None,
            aggressive_elimination: false,
            results: None,
        }
    }

    /// Set the halving factor (default 3).
    ///
    /// After each round, only `ceil(n_candidates / factor)` candidates survive.
    /// The data budget is multiplied by `factor` each round.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid `factor` values are caught at
    /// [`fit`](HalvingGridSearchCV::fit) time.
    #[must_use]
    pub fn factor(mut self, factor: usize) -> Self {
        self.factor = factor;
        self
    }

    /// Set the minimum number of samples used in the first round.
    ///
    /// If `None` (default), it is auto-computed as
    /// `max(1, floor(n_samples / factor^n_rounds))`.
    #[must_use]
    pub fn min_resources(mut self, min_resources: Option<usize>) -> Self {
        self.min_resources = min_resources;
        self
    }

    /// Set the maximum number of samples per round (defaults to `n_samples`).
    #[must_use]
    pub fn max_resources(mut self, max_resources: Option<usize>) -> Self {
        self.max_resources = max_resources;
        self
    }

    /// When `true`, keep halving until only one candidate remains, even after
    /// reaching `max_resources`.
    #[must_use]
    pub fn aggressive_elimination(mut self, aggressive: bool) -> Self {
        self.aggressive_elimination = aggressive;
        self
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the minimum resources if not specified.
    ///
    /// We aim for `factor^n_rounds * min_res ≈ max_res`, so
    /// `min_res = max(1, floor(max_res / factor^n_rounds))`.
    fn compute_min_resources(&self, n_samples: usize, n_candidates: usize) -> usize {
        if let Some(mr) = self.min_resources {
            return mr;
        }
        let max_res = self.max_resources.unwrap_or(n_samples);
        // Number of rounds needed to reduce n_candidates to 1.
        let n_rounds = compute_n_rounds(n_candidates, self.factor).max(1);
        let factor_pow = (self.factor as f64).powi(n_rounds as i32);
        let min_res = (max_res as f64 / factor_pow).ceil() as usize;
        min_res.max(1)
    }

    /// Evaluate a list of candidates on the first `budget` rows of `x`/`y`.
    ///
    /// Returns a `Vec<(candidate_index, mean_score)>` sorted best-first.
    fn evaluate_candidates(
        &self,
        candidates: &[usize],
        budget: usize,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Vec<(usize, f64)>, FerroError> {
        // Slice the first `budget` samples.
        let x_sub = x.slice(ndarray::s![..budget, ..]).to_owned();
        let y_sub: Array1<f64> = y.iter().take(budget).copied().collect();

        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
        for &cand_idx in candidates {
            let params = &self.param_grid[cand_idx];
            let pipeline = (self.pipeline_factory)(params);
            let scores = cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)
                .map_err(|e| FerroError::InvalidParameter {
                    name: "cross_val_score".into(),
                    reason: format!("candidate {cand_idx}: {e}"),
                })?;
            let mean = scores.mean().unwrap_or(f64::NEG_INFINITY);
            scored.push((cand_idx, mean));
        }
        // Sort descending by score (best first).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Run the successive-halving search.
    ///
    /// For each round, candidates are evaluated on an increasing budget of
    /// data; the bottom fraction are eliminated.  The final
    /// [`CvResults`] contains only the candidates that reached the last round.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `param_grid` is empty, `factor < 2`,
    ///   or cross-validation fails for any candidate.
    /// - [`FerroError::InsufficientSamples`] if `budget > n_samples` on any round.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), FerroError> {
        if self.param_grid.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "param_grid".into(),
                reason: "parameter grid must not be empty".into(),
            });
        }
        if self.factor < 2 {
            return Err(FerroError::InvalidParameter {
                name: "factor".into(),
                reason: format!("must be >= 2, got {}", self.factor),
            });
        }

        let n_samples = x.nrows();
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "HalvingGridSearchCV: x rows and y length must match".into(),
            });
        }

        let n_candidates = self.param_grid.len();
        let max_res = self.max_resources.unwrap_or(n_samples).min(n_samples);
        let min_res = self.compute_min_resources(n_samples, n_candidates);

        if min_res == 0 || min_res > max_res {
            return Err(FerroError::InvalidParameter {
                name: "min_resources".into(),
                reason: format!(
                    "computed min_resources={min_res} is invalid (max_resources={max_res})"
                ),
            });
        }

        // Candidates are tracked by their index into `self.param_grid`.
        let mut active: Vec<usize> = (0..n_candidates).collect();
        let mut budget = min_res;
        let mut results = CvResults::new();

        loop {
            // Cap budget at max_resources.
            let effective_budget = budget.min(max_res);

            // Evaluate all active candidates on this budget.
            let scored = self.evaluate_candidates(&active, effective_budget, x, y)?;

            // Determine how many survive into the next round.
            let n_survive = if self.aggressive_elimination || effective_budget < max_res {
                // Keep top ceil(n / factor) but at least 1.
                let n = scored.len();
                n.div_ceil(self.factor).max(1)
            } else {
                // Final round: keep all remaining (or just 1 if aggressive).
                if self.aggressive_elimination {
                    1
                } else {
                    scored.len()
                }
            };

            let is_final = effective_budget >= max_res || scored.len() <= 1;

            if is_final {
                // Record all surviving candidates in CvResults.
                for (cand_idx, _mean) in &scored {
                    let params = self.param_grid[*cand_idx].clone();
                    // Re-evaluate on the full budget for the CvResults entry.
                    let x_sub = x.slice(ndarray::s![..effective_budget, ..]).to_owned();
                    let y_sub: Array1<f64> = y.iter().take(effective_budget).copied().collect();
                    let pipeline = (self.pipeline_factory)(&params);
                    let fold_scores =
                        cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)?;
                    results.push(params, fold_scores);
                }
                break;
            }

            // Eliminate the bottom candidates; carry forward top `n_survive`.
            active = scored
                .into_iter()
                .take(n_survive)
                .map(|(idx, _)| idx)
                .collect();

            if active.len() <= 1 {
                // Only one candidate left – record it and finish.
                if let Some(&cand_idx) = active.first() {
                    let params = self.param_grid[cand_idx].clone();
                    let x_sub = x.slice(ndarray::s![..effective_budget, ..]).to_owned();
                    let y_sub: Array1<f64> = y.iter().take(effective_budget).copied().collect();
                    let pipeline = (self.pipeline_factory)(&params);
                    let fold_scores =
                        cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)?;
                    results.push(params, fold_scores);
                }
                break;
            }

            // Multiply the budget for the next round.
            budget = budget.saturating_mul(self.factor).min(max_res + 1);
        }

        self.results = Some(results);
        Ok(())
    }

    /// Return a reference to the cross-validation results from the final round.
    ///
    /// Returns `None` if [`fit`](HalvingGridSearchCV::fit) has not been called.
    pub fn cv_results(&self) -> Option<&CvResults> {
        self.results.as_ref()
    }

    /// Return the parameter set that achieved the highest mean score in the
    /// final round.
    ///
    /// Returns `None` if [`fit`](HalvingGridSearchCV::fit) has not been called.
    pub fn best_params(&self) -> Option<&ParamSet> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.params.get(idx)
    }

    /// Return the best mean cross-validation score from the final round.
    ///
    /// Returns `None` if [`fit`](HalvingGridSearchCV::fit) has not been called.
    pub fn best_score(&self) -> Option<f64> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.mean_scores.get(idx).copied()
    }
}

// ---------------------------------------------------------------------------
// Internal utility
// ---------------------------------------------------------------------------

/// Return `ceil(log_base(n))` — the number of halving rounds needed to reduce
/// `n` candidates to 1.
fn compute_n_rounds(n: usize, base: usize) -> usize {
    if n <= 1 || base < 2 {
        return 0;
    }
    let mut rounds = 0_usize;
    let mut remaining = n;
    while remaining > 1 {
        remaining = remaining.div_ceil(base);
        rounds += 1;
    }
    rounds
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
    use ndarray::{Array1, Array2};

    use crate::KFold;
    use crate::param_grid::ParamValue;

    // -----------------------------------------------------------------------
    // Test fixtures
    // -----------------------------------------------------------------------

    struct ConstantEstimator {
        value: f64,
    }

    struct FittedConstant {
        value: f64,
    }

    impl PipelineEstimator<f64> for ConstantEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedConstant { value: self.value }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedConstant {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.value))
        }
    }

    struct MeanEstimator;
    struct FittedMean {
        mean: f64,
    }

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedMean {
                mean: y.mean().unwrap_or(0.0),
            }))
        }
    }

    impl FittedPipelineEstimator<f64> for FittedMean {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    fn constant_factory(params: &ParamSet) -> Pipeline {
        let val = match params.get("constant") {
            Some(ParamValue::Float(v)) => *v,
            _ => 0.0,
        };
        Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
    }

    fn mean_factory(_params: &ParamSet) -> Pipeline {
        Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
    }

    // -----------------------------------------------------------------------
    // compute_n_rounds utility
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_n_rounds_basic() {
        assert_eq!(compute_n_rounds(1, 3), 0);
        assert_eq!(compute_n_rounds(3, 3), 1);
        assert_eq!(compute_n_rounds(9, 3), 2);
        assert_eq!(compute_n_rounds(10, 3), 3); // ceil(log3(10)) = 3
    }

    // -----------------------------------------------------------------------
    // HalvingGridSearchCV tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_halving_basic_runs() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.cv_results().is_some());
    }

    #[test]
    fn test_halving_finds_best_constant() {
        // y = 1.0; estimator predicting 1.0 should win.
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 5.0_f64, 10.0_f64, 20.0_f64, 50.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .factor(2)
        .min_resources(Some(9));
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();

        let best = hs.best_params().unwrap();
        assert_eq!(best.get("constant"), Some(&ParamValue::Float(1.0)));
    }

    #[test]
    fn test_halving_best_score_near_zero_for_perfect() {
        let grid = crate::param_grid! {
            "dummy" => [true],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(mean_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 3.0);
        hs.fit(&x, &y).unwrap();

        let score = hs.best_score().unwrap();
        assert!(score.abs() < 1e-10, "expected ~0 score, got {score}");
    }

    #[test]
    fn test_halving_empty_grid_returns_error() {
        let mut hs = HalvingGridSearchCV::new(
            Box::new(mean_factory),
            vec![],
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::zeros(60);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_factor_less_than_2_returns_error() {
        let grid = crate::param_grid! { "dummy" => [true] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(mean_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .factor(1);
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_returns_none_before_fit() {
        let grid = crate::param_grid! { "dummy" => [true] };
        let hs = HalvingGridSearchCV::new(
            Box::new(mean_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        assert!(hs.best_params().is_none());
        assert!(hs.best_score().is_none());
        assert!(hs.cv_results().is_none());
    }

    #[test]
    fn test_halving_shape_mismatch_returns_error() {
        let grid = crate::param_grid! { "dummy" => [true] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(mean_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(25); // length mismatch
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_single_candidate() {
        // With only one candidate there is nothing to halve.
        let grid = crate::param_grid! { "constant" => [1.0_f64] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_score().is_some());
    }

    #[test]
    fn test_halving_cv_results_non_empty_after_fit() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .factor(3)
        .min_resources(Some(9));
        let x = Array2::<f64>::zeros((90, 2));
        let y = Array1::<f64>::from_elem(90, 1.0);
        hs.fit(&x, &y).unwrap();

        let results = hs.cv_results().unwrap();
        // At least one candidate should have been evaluated.
        assert!(!results.params.is_empty());
    }

    #[test]
    fn test_halving_custom_max_resources() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .max_resources(Some(30))
        .min_resources(Some(9));
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        // Should not use more than 30 samples even though 60 are available.
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_score().is_some());
    }

    #[test]
    fn test_halving_factor_2() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(2)),
            neg_mse,
        )
        .factor(2)
        .min_resources(Some(6));
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_params().is_some());
    }

    #[test]
    fn test_halving_aggressive_elimination() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(2)),
            neg_mse,
        )
        .factor(3)
        .min_resources(Some(6))
        .aggressive_elimination(true);
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();
        // With aggressive elimination, results should have exactly 1 candidate.
        let results = hs.cv_results().unwrap();
        assert_eq!(results.params.len(), 1);
    }

    #[test]
    fn test_halving_best_score_is_finite() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64],
        };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 0.5);
        hs.fit(&x, &y).unwrap();
        let score = hs.best_score().unwrap();
        assert!(
            score.is_finite(),
            "best score should be finite, got {score}"
        );
    }

    #[test]
    fn test_halving_cv_results_all_scores_have_fold_scores() {
        let grid = crate::param_grid! {
            "constant" => [0.0_f64, 1.0_f64, 2.0_f64],
        };
        let n_folds = 3_usize;
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(n_folds)),
            neg_mse,
        )
        .min_resources(Some(9));
        let x = Array2::<f64>::zeros((90, 2));
        let y = Array1::<f64>::from_elem(90, 1.0);
        hs.fit(&x, &y).unwrap();

        let results = hs.cv_results().unwrap();
        for fold_scores in &results.all_scores {
            assert_eq!(
                fold_scores.len(),
                n_folds,
                "each entry should have {n_folds} fold scores"
            );
        }
    }
}
