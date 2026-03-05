//! Randomized hyperparameter search with cross-validation.
//!
//! [`RandomizedSearchCV`] samples `n_iter` random parameter combinations from
//! the supplied distributions, evaluates each using cross-validation, and
//! records the results in a [`CvResults`] struct (re-exported from
//! [`grid_search`]).
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrolearn_model_sel::{
//!     RandomizedSearchCV, KFold, ParamValue,
//!     distributions::{Distribution, Uniform, IntUniform},
//! };
//! use ferrolearn_core::pipeline::Pipeline;
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
//!     let diff = y_true - y_pred;
//!     Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
//! }
//!
//! // let factory = |_: &_| Pipeline::new().estimator_step("m", Box::new(MyEst));
//! // let param_dists = vec![
//! //     ("alpha".to_string(), Box::new(Uniform::new(0.0, 1.0)) as Box<dyn Distribution>),
//! // ];
//! // let mut rs = RandomizedSearchCV::new(
//! //     Box::new(factory), param_dists, 10, Box::new(KFold::new(3)), neg_mse, Some(42),
//! // );
//! // rs.fit(&x, &y).unwrap();
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
// RandomizedSearchCV
// ---------------------------------------------------------------------------

/// Randomized search over hyperparameter distributions using cross-validation.
///
/// Instead of exhaustively searching a grid, [`RandomizedSearchCV`] samples
/// `n_iter` random parameter combinations from user-supplied [`Distribution`]
/// objects. This is more efficient when the grid is large or when only a
/// fraction of the space needs to be explored.
///
/// After calling [`fit`](RandomizedSearchCV::fit) the best parameters and
/// score can be retrieved via the accessor methods.
pub struct RandomizedSearchCV<'a> {
    /// Factory that builds a [`Pipeline`] from a [`ParamSet`].
    pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
    /// One `(name, distribution)` pair per hyperparameter.
    param_distributions: Vec<(String, Box<dyn Distribution>)>,
    /// Number of parameter combinations to sample.
    n_iter: usize,
    /// Cross-validator used to evaluate each combination.
    cv: Box<dyn CrossValidator>,
    /// Scoring function; higher is better.
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    /// Results populated after [`fit`](RandomizedSearchCV::fit) is called.
    results: Option<CvResults>,
}

impl<'a> RandomizedSearchCV<'a> {
    /// Create a new [`RandomizedSearchCV`].
    ///
    /// # Parameters
    ///
    /// - `pipeline_factory` — closure that accepts a [`ParamSet`] and returns
    ///   an unfitted [`Pipeline`].
    /// - `param_distributions` — list of `(name, distribution)` pairs.
    /// - `n_iter` — number of random parameter combinations to try.
    /// - `cv` — the cross-validator.
    /// - `scoring` — scoring function; higher is better.
    /// - `random_state` — optional seed for the RNG.
    pub fn new(
        pipeline_factory: Box<dyn Fn(&ParamSet) -> Pipeline + 'a>,
        param_distributions: Vec<(String, Box<dyn Distribution>)>,
        n_iter: usize,
        cv: Box<dyn CrossValidator>,
        scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            pipeline_factory,
            param_distributions,
            n_iter,
            cv,
            scoring,
            random_state,
            results: None,
        }
    }

    /// Sample a single [`ParamSet`] by drawing one value from each distribution.
    fn sample_params(&self, rng: &mut SmallRng) -> ParamSet {
        self.param_distributions
            .iter()
            .map(|(name, dist)| (name.clone(), dist.sample(rng)))
            .collect()
    }

    /// Run the randomized search.
    ///
    /// Samples `n_iter` parameter combinations, builds a pipeline for each,
    /// runs cross-validation, and stores the results internally.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if `n_iter` is zero, if the distribution list
    /// is empty, if any pipeline fails to fit, or if the cross-validator fails.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), FerroError> {
        if self.n_iter == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_iter".into(),
                reason: "n_iter must be > 0".into(),
            });
        }
        if self.param_distributions.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "param_distributions".into(),
                reason: "distribution list must not be empty".into(),
            });
        }

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let mut results = CvResults::new();

        for _ in 0..self.n_iter {
            let params = self.sample_params(&mut rng);
            let pipeline = (self.pipeline_factory)(&params);
            let scores = cross_val_score(&pipeline, x, y, self.cv.as_ref(), self.scoring)?;
            results.push(params, scores);
        }

        self.results = Some(results);
        Ok(())
    }

    /// Return a reference to the full cross-validation results.
    ///
    /// Returns `None` if [`fit`](RandomizedSearchCV::fit) has not been called.
    pub fn cv_results(&self) -> Option<&CvResults> {
        self.results.as_ref()
    }

    /// Return the parameter set that achieved the highest mean score.
    ///
    /// Returns `None` if [`fit`](RandomizedSearchCV::fit) has not been called.
    pub fn best_params(&self) -> Option<&ParamSet> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.params.get(idx)
    }

    /// Return the best mean cross-validation score.
    ///
    /// Returns `None` if [`fit`](RandomizedSearchCV::fit) has not been called.
    pub fn best_score(&self) -> Option<f64> {
        let results = self.results.as_ref()?;
        let idx = results.best_index()?;
        results.mean_scores.get(idx).copied()
    }
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
    use crate::distributions::{Choice, IntUniform, LogUniform, Uniform};
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

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_random_search_samples_correct_n_iter() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            7,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(42),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        rs.fit(&x, &y).unwrap();

        let results = rs.cv_results().unwrap();
        assert_eq!(results.params.len(), 7);
    }

    #[test]
    fn test_random_search_deterministic_with_seed() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let make_dists = || -> Vec<(String, Box<dyn Distribution>)> {
            vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))]
        };

        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);

        let mut rs1 = RandomizedSearchCV::new(
            Box::new(factory),
            make_dists(),
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(99),
        );
        rs1.fit(&x, &y).unwrap();

        let factory2 =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let mut rs2 = RandomizedSearchCV::new(
            Box::new(factory2),
            make_dists(),
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(99),
        );
        rs2.fit(&x, &y).unwrap();

        let r1 = rs1.cv_results().unwrap();
        let r2 = rs2.cv_results().unwrap();

        // With the same seed the sampled alpha values should be identical.
        for (p1, p2) in r1.params.iter().zip(r2.params.iter()) {
            assert_eq!(p1.get("alpha"), p2.get("alpha"));
        }
    }

    #[test]
    fn test_random_search_returns_none_before_fit() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(1),
        );
        assert!(rs.best_params().is_none());
        assert!(rs.best_score().is_none());
        assert!(rs.cv_results().is_none());
    }

    #[test]
    fn test_random_search_n_iter_zero_error() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            0,
            Box::new(KFold::new(3)),
            neg_mse,
            None,
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        assert!(rs.fit(&x, &y).is_err());
    }

    #[test]
    fn test_random_search_with_log_uniform() {
        let factory = |params: &ParamSet| {
            let val = match params.get("lr") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.01,
            };
            Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
        };
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("lr".into(), Box::new(LogUniform::new(1e-4, 1e-1)))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(7),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        rs.fit(&x, &y).unwrap();

        // All sampled lr values should be in [1e-4, 1e-1).
        let results = rs.cv_results().unwrap();
        for p in &results.params {
            if let Some(ParamValue::Float(lr)) = p.get("lr") {
                assert!(*lr >= 1e-4 && *lr < 1e-1, "lr {lr} out of range");
            }
        }
    }

    #[test]
    fn test_random_search_with_int_uniform() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("n_trees".into(), Box::new(IntUniform::new(1, 100)))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            10,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(13),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        rs.fit(&x, &y).unwrap();

        let results = rs.cv_results().unwrap();
        for p in &results.params {
            if let Some(ParamValue::Int(n)) = p.get("n_trees") {
                assert!(*n >= 1 && *n <= 100, "n_trees {n} out of [1, 100]");
            }
        }
    }

    #[test]
    fn test_random_search_with_choice() {
        let factory =
            |_: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let options = vec![
            ParamValue::String("relu".into()),
            ParamValue::String("tanh".into()),
            ParamValue::String("sigmoid".into()),
        ];
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("activation".into(), Box::new(Choice::new(options.clone())))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            15,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(55),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::zeros(30);
        rs.fit(&x, &y).unwrap();

        let results = rs.cv_results().unwrap();
        for p in &results.params {
            assert!(options.contains(p.get("activation").unwrap()));
        }
    }

    #[test]
    fn test_random_search_best_params_selected_correctly() {
        // y = 1.0; best constant predictor is 1.0 → neg_mse = 0.
        // We include 1.0 in the Choice so it can be sampled.
        let factory = |params: &ParamSet| {
            let val = match params.get("c") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
        };
        // Choice always returns 1.0 (single option).
        let dists: Vec<(String, Box<dyn Distribution>)> = vec![(
            "c".into(),
            Box::new(Choice::new(vec![ParamValue::Float(1.0)])),
        )];
        let mut rs = RandomizedSearchCV::new(
            Box::new(factory),
            dists,
            3,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(0),
        );
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        rs.fit(&x, &y).unwrap();

        let score = rs.best_score().unwrap();
        assert!(score.abs() < 1e-10, "expected ~0 score, got {score}");
    }
}
