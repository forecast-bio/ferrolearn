//! Successive halving hyperparameter search with random sampling.
//!
//! [`HalvingRandomSearchCV`] combines successive halving with random
//! parameter sampling. Instead of exhaustively enumerating a grid, it
//! draws `n_candidates` random parameter sets from user-supplied
//! distributions, then runs the successive-halving elimination tournament
//! described in [`HalvingGridSearchCV`](crate::HalvingGridSearchCV).
//!
//! This is more efficient than either exhaustive grid search or plain
//! randomized search when the parameter space is large and many candidates
//! can be eliminated early on small data budgets.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{
//!     HalvingRandomSearchCV, KFold,
//!     distributions::{Distribution, Uniform},
//! };
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y: &Array1<f64>, p: &Array1<f64>) -> Result<f64, FerroError> {
//!     let d = y - p; Ok(-d.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // let dists = vec![("alpha".into(), Box::new(Uniform::new(0.01, 10.0)) as Box<dyn Distribution>)];
//! // let mut hs = HalvingRandomSearchCV::new(
//! //     Box::new(|_| Pipeline::new().estimator_step("est", Box::new(my_est))),
//! //     dists, 20, Box::new(KFold::new(3)), neg_mse, Some(42),
//! // );
//! // hs.fit(&x, &y).unwrap();
//! // println!("Best: {:?}", hs.best_params());
//! ```

use rand::SeedableRng;
use rand::rngs::SmallRng;

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ndarray::{Array1, Array2};

use crate::cross_validation::{CrossValidator, cross_val_score};
use crate::distributions::Distribution;
use crate::grid_search::CvResults;
use crate::param_grid::ParamSet;

// ---------------------------------------------------------------------------
// HalvingRandomSearchCV
// ---------------------------------------------------------------------------

/// Successive-halving hyperparameter search with random sampling.
///
/// This meta-estimator draws `n_candidates` parameter sets from the supplied
/// distributions, then runs successive halving: in each round, candidates are
/// evaluated on an increasing fraction of data, and the worst performers are
/// eliminated.
///
/// # Fields (set via builder methods)
///
/// - `factor` — halving factor (default 3).
/// - `min_resources` — smallest data budget per round (auto-computed if `None`).
/// - `max_resources` — largest data budget (default `n_samples`).
pub struct HalvingRandomSearchCV<'a> {
    /// Factory that builds a [`Pipeline`] from a [`ParamSet`].
    pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
    /// One `(name, distribution)` pair per hyperparameter.
    param_distributions: Vec<(String, Box<dyn Distribution>)>,
    /// Number of initial candidates to sample.
    n_candidates: usize,
    /// Cross-validator used to evaluate each combination on each budget.
    cv: Box<dyn CrossValidator>,
    /// Scoring function; higher is better.
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    /// Halving factor (default 3).
    factor: usize,
    /// Minimum number of samples per round (auto-computed if `None`).
    min_resources: Option<usize>,
    /// Maximum number of samples per round (defaults to `n_samples`).
    max_resources: Option<usize>,
    /// Results populated after [`fit`](HalvingRandomSearchCV::fit) is called.
    results: Option<CvResults>,
}

impl<'a> HalvingRandomSearchCV<'a> {
    /// Create a new [`HalvingRandomSearchCV`] with default options.
    ///
    /// # Parameters
    ///
    /// - `pipeline_factory` — closure that accepts a [`ParamSet`] and returns
    ///   an unfitted [`Pipeline`].
    /// - `param_distributions` — list of `(name, distribution)` pairs.
    /// - `n_candidates` — the number of initial random candidates to sample.
    /// - `cv` — the cross-validator.
    /// - `scoring` — scoring function; higher is better.
    /// - `random_state` — optional seed for the RNG.
    pub fn new(
        pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
        param_distributions: Vec<(String, Box<dyn Distribution>)>,
        n_candidates: usize,
        cv: Box<dyn CrossValidator>,
        scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            pipeline_factory,
            param_distributions,
            n_candidates,
            cv,
            scoring,
            random_state,
            factor: 3,
            min_resources: None,
            max_resources: None,
            results: None,
        }
    }

    /// Set the halving factor (default 3).
    ///
    /// After each round, only `ceil(n_candidates / factor)` candidates survive.
    /// The data budget is multiplied by `factor` each round.
    #[must_use]
    pub fn factor(mut self, factor: usize) -> Self {
        self.factor = factor;
        self
    }

    /// Set the minimum number of samples used in the first round.
    ///
    /// If `None` (default), it is auto-computed.
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

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Sample a single [`ParamSet`] by drawing one value from each distribution.
    fn sample_params(&self, rng: &mut SmallRng) -> ParamSet {
        self.param_distributions
            .iter()
            .map(|(name, dist)| (name.clone(), dist.sample(rng)))
            .collect()
    }

    /// Compute the minimum resources if not specified.
    fn compute_min_resources(&self, n_samples: usize, n_cands: usize) -> usize {
        if let Some(mr) = self.min_resources {
            return mr;
        }
        let max_res = self.max_resources.unwrap_or(n_samples);
        let n_rounds = compute_n_rounds(n_cands, self.factor).max(1);
        let factor_pow = (self.factor as f64).powi(n_rounds as i32);
        let min_res = (max_res as f64 / factor_pow).ceil() as usize;
        min_res.max(1)
    }

    /// Evaluate a list of candidates on the first `budget` rows of `x`/`y`.
    ///
    /// Returns `Vec<(candidate_index, mean_score)>` sorted best-first.
    fn evaluate_candidates(
        &self,
        candidate_params: &[ParamSet],
        candidate_indices: &[usize],
        budget: usize,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Vec<(usize, f64)>, FerroError> {
        let x_sub = x.slice(ndarray::s![..budget, ..]).to_owned();
        let y_sub: Array1<f64> = y.iter().take(budget).copied().collect();

        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(candidate_indices.len());
        for &cand_idx in candidate_indices {
            let params = &candidate_params[cand_idx];
            let pipeline = (self.pipeline_factory)(params);
            let scores = cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)
                .map_err(|e| FerroError::InvalidParameter {
                    name: "cross_val_score".into(),
                    reason: format!("candidate {cand_idx}: {e}"),
                })?;
            let mean = scores.mean().unwrap_or(f64::NEG_INFINITY);
            scored.push((cand_idx, mean));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Run the successive-halving search with randomly sampled candidates.
    ///
    /// 1. Sample `n_candidates` parameter sets from the distributions.
    /// 2. Run successive halving: evaluate on increasing data budgets,
    ///    eliminating the worst `(factor-1)/factor` fraction each round.
    /// 3. Store the final-round results.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_candidates` is zero,
    ///   `param_distributions` is empty, `factor < 2`, or any CV fails.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have incompatible shapes.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), FerroError> {
        if self.n_candidates == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_candidates".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.param_distributions.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "param_distributions".into(),
                reason: "distribution list must not be empty".into(),
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
                context: "HalvingRandomSearchCV: x rows and y length must match".into(),
            });
        }

        // Sample all candidates up-front.
        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };
        let all_params: Vec<ParamSet> = (0..self.n_candidates)
            .map(|_| self.sample_params(&mut rng))
            .collect();

        let max_res = self.max_resources.unwrap_or(n_samples).min(n_samples);
        let min_res = self.compute_min_resources(n_samples, self.n_candidates);

        if min_res == 0 || min_res > max_res {
            return Err(FerroError::InvalidParameter {
                name: "min_resources".into(),
                reason: format!(
                    "computed min_resources={min_res} is invalid (max_resources={max_res})"
                ),
            });
        }

        let mut active: Vec<usize> = (0..self.n_candidates).collect();
        let mut budget = min_res;
        let mut results = CvResults::new();

        loop {
            let effective_budget = budget.min(max_res);

            let scored = self.evaluate_candidates(&all_params, &active, effective_budget, x, y)?;

            let is_final = effective_budget >= max_res || scored.len() <= 1;

            if is_final {
                // Record all surviving candidates in CvResults.
                for (cand_idx, _mean) in &scored {
                    let params = all_params[*cand_idx].clone();
                    let x_sub = x.slice(ndarray::s![..effective_budget, ..]).to_owned();
                    let y_sub: Array1<f64> = y.iter().take(effective_budget).copied().collect();
                    let pipeline = (self.pipeline_factory)(&params);
                    let fold_scores =
                        cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)?;
                    results.push(params, fold_scores);
                }
                break;
            }

            // Eliminate: keep top ceil(n / factor) but at least 1.
            let n_survive = scored.len().div_ceil(self.factor).max(1);

            active = scored
                .into_iter()
                .take(n_survive)
                .map(|(idx, _)| idx)
                .collect();

            if active.len() <= 1 {
                if let Some(&cand_idx) = active.first() {
                    let params = all_params[cand_idx].clone();
                    let x_sub = x.slice(ndarray::s![..effective_budget, ..]).to_owned();
                    let y_sub: Array1<f64> = y.iter().take(effective_budget).copied().collect();
                    let pipeline = (self.pipeline_factory)(&params);
                    let fold_scores =
                        cross_val_score(&pipeline, &x_sub, &y_sub, self.cv.as_ref(), self.scoring)?;
                    results.push(params, fold_scores);
                }
                break;
            }

            budget = budget.saturating_mul(self.factor).min(max_res + 1);
        }

        self.results = Some(results);
        Ok(())
    }

    /// Return a reference to the cross-validation results from the final round.
    ///
    /// Returns `None` if [`fit`](HalvingRandomSearchCV::fit) has not been called.
    pub fn cv_results(&self) -> Option<&CvResults> {
        self.results.as_ref()
    }

    /// Return the parameter set that achieved the highest mean score in the
    /// final round.
    ///
    /// Returns `None` if [`fit`](HalvingRandomSearchCV::fit) has not been called.
    pub fn best_params(&self) -> Option<&ParamSet> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.params.get(idx)
    }

    /// Return the best mean cross-validation score from the final round.
    ///
    /// Returns `None` if [`fit`](HalvingRandomSearchCV::fit) has not been called.
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
    use crate::distributions::{Choice, Uniform};
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
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_halving_random_basic_runs() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("constant".into(), Box::new(Uniform::new(-5.0, 5.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dists,
            6,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(42),
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.cv_results().is_some());
    }

    #[test]
    fn test_halving_random_deterministic_with_seed() {
        let make_hs = || {
            let dists: Vec<(String, Box<dyn Distribution>)> =
                vec![("constant".into(), Box::new(Uniform::new(-5.0, 5.0)))];
            HalvingRandomSearchCV::new(
                Box::new(constant_factory),
                dists,
                6,
                Box::new(KFold::new(3)),
                neg_mse,
                Some(99),
            )
        };

        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);

        let mut hs1 = make_hs();
        hs1.fit(&x, &y).unwrap();

        let mut hs2 = make_hs();
        hs2.fit(&x, &y).unwrap();

        let s1 = hs1.best_score().unwrap();
        let s2 = hs2.best_score().unwrap();
        assert!(
            (s1 - s2).abs() < 1e-10,
            "scores should match with same seed: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_halving_random_best_score_near_zero_for_perfect() {
        let dists: Vec<(String, Box<dyn Distribution>)> = vec![(
            "dummy".into(),
            Box::new(Choice::new(vec![ParamValue::Bool(true)])),
        )];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            dists,
            3,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(0),
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 3.0);
        hs.fit(&x, &y).unwrap();

        let score = hs.best_score().unwrap();
        assert!(score.abs() < 1e-10, "expected ~0 score, got {score}");
    }

    #[test]
    fn test_halving_random_n_candidates_zero_error() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("a".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            dists,
            0,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::zeros(60);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_random_empty_distributions_error() {
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            vec![],
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        );
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::zeros(60);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_random_factor_less_than_2_error() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("a".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        )
        .factor(1);
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::zeros(60);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_random_returns_none_before_fit() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("a".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        );
        assert!(hs.best_params().is_none());
        assert!(hs.best_score().is_none());
        assert!(hs.cv_results().is_none());
    }

    #[test]
    fn test_halving_random_shape_mismatch_error() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("a".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(mean_factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(25);
        assert!(hs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_halving_random_single_candidate() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("constant".into(), Box::new(Uniform::new(0.9, 1.1)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dists,
            1,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(7),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_score().is_some());
    }

    #[test]
    fn test_halving_random_custom_factor() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("constant".into(), Box::new(Uniform::new(-10.0, 10.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dists,
            8,
            Box::new(KFold::new(2)),
            neg_mse,
            Some(42),
        )
        .factor(2)
        .min_resources(Some(6));
        let x = Array2::<f64>::zeros((60, 2));
        let y = Array1::<f64>::from_elem(60, 1.0);
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_params().is_some());
    }

    #[test]
    fn test_halving_random_best_score_is_finite() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("constant".into(), Box::new(Uniform::new(-5.0, 5.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(13),
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
    fn test_halving_random_cv_results_non_empty_after_fit() {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("constant".into(), Box::new(Uniform::new(-5.0, 5.0)))];
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dists,
            9,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(55),
        )
        .min_resources(Some(9));
        let x = Array2::<f64>::zeros((90, 2));
        let y = Array1::<f64>::from_elem(90, 1.0);
        hs.fit(&x, &y).unwrap();

        let results = hs.cv_results().unwrap();
        assert!(!results.params.is_empty());
    }
}
