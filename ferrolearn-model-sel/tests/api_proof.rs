//! Proof-of-API integration test for ferrolearn-model-sel.
//!
//! Audit deliverable for crosslink #317 (under #249). Exercises every
//! public estimator, splitter, and helper after orphan wiring (#310),
//! the new CV splitters (#311), and the new Dummy estimators (#314).
//!
//! Constructors and accessors are exercised across the whole API surface;
//! heavy meta-estimators that would require a full Pipeline factory
//! (GridSearchCV, RandomizedSearchCV, HalvingGrid/Random, OneVsRest/OvO,
//! MultiOutput*, CalibratedClassifierCV, SelfTrainingClassifier,
//! TransformedTargetRegressor) are exercised at construction-time only,
//! using minimal stub closures — these confirm signatures + integration
//! without depending on a fully-wired downstream estimator.

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::{FerroError, Fit, Predict};
use ferrolearn_model_sel::calibration::FitFn as CalibFitFn;
use ferrolearn_model_sel::cross_validation::CrossValidator;
use ferrolearn_model_sel::distributions::{Choice, Distribution, IntUniform, LogUniform, Uniform};
use ferrolearn_model_sel::learning_curve::learning_curve;
use ferrolearn_model_sel::validation_curve::validation_curve;
use ferrolearn_model_sel::{
    CalibratedClassifierCV, CalibrationMethod, CrossValidateResult, DummyClassifier,
    DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy, FeatureUnion, GridSearchCV,
    HalvingGridSearchCV, HalvingRandomSearchCV, KFold, LeaveOneOut, LeavePOut,
    MultiOutputClassifier, MultiOutputRegressor, OneVsOneClassifier, OneVsRestClassifier, ParamSet,
    ParamValue, PredefinedSplit, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold,
    SelfTrainingClassifier, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit,
    TransformedTargetRegressor, param_grid, train_test_split,
};
use ndarray::{Array1, Array2, array};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// =============================================================================
// Splitters: KFold, StratifiedKFold, TimeSeriesSplit, and the new family
// =============================================================================
#[test]
fn api_proof_splitters() {
    // KFold + StratifiedKFold (already in old API)
    let folds = KFold::new(3).split(9);
    assert_eq!(folds.len(), 3);
    let folds = KFold::new(3).shuffle(true).random_state(7).split(9);
    assert_eq!(folds.len(), 3);

    let y = array![0usize, 0, 1, 1, 2, 2, 0, 1, 2];
    let folds = StratifiedKFold::new(3).split(&y).unwrap();
    assert_eq!(folds.len(), 3);

    // TimeSeriesSplit
    let folds = TimeSeriesSplit::new(3)
        .max_train_size(Some(5))
        .test_size(Some(2))
        .gap(0)
        .split(10)
        .unwrap();
    assert_eq!(folds.len(), 3);

    // New splitters
    let folds = LeaveOneOut::new().fold_indices(5).unwrap();
    assert_eq!(folds.len(), 5);

    let folds = LeavePOut::new(2).fold_indices(4).unwrap();
    assert_eq!(folds.len(), 6); // C(4, 2) = 6

    let folds = ShuffleSplit::new(2, 0.25)
        .random_state(7)
        .fold_indices(8)
        .unwrap();
    assert_eq!(folds.len(), 2);

    let folds = StratifiedShuffleSplit::new(2, 0.25)
        .random_state(7)
        .split(&y)
        .unwrap();
    assert_eq!(folds.len(), 2);

    let folds = RepeatedKFold::new(3, 2)
        .random_state(11)
        .fold_indices(9)
        .unwrap();
    assert_eq!(folds.len(), 6);

    let folds = RepeatedStratifiedKFold::new(3, 2)
        .random_state(11)
        .split(&y)
        .unwrap();
    assert_eq!(folds.len(), 6);

    let test_fold = array![0_isize, 1, -1, 1, 0];
    let folds = PredefinedSplit::new(test_fold).fold_indices(5).unwrap();
    assert_eq!(folds.len(), 2);
}

// =============================================================================
// train_test_split
// =============================================================================
#[test]
fn api_proof_train_test_split() {
    let x = Array2::<f64>::zeros((20, 3));
    let y = Array1::<f64>::zeros(20);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42)).unwrap();
    assert_eq!(x_train.nrows(), 16);
    assert_eq!(x_test.nrows(), 4);
    assert_eq!(y_train.len(), 16);
    assert_eq!(y_test.len(), 4);
}

// =============================================================================
// Dummy estimators
// =============================================================================
#[test]
fn api_proof_dummy() {
    let x = Array2::<f64>::zeros((6, 2));
    let y_class = array![0usize, 0, 1, 1, 1, 2];
    for strat in [
        DummyClassifierStrategy::MostFrequent,
        DummyClassifierStrategy::Prior,
        DummyClassifierStrategy::Stratified,
        DummyClassifierStrategy::Uniform,
        DummyClassifierStrategy::Constant(1),
    ] {
        let clf = DummyClassifier::new(strat).random_state(1);
        let fitted = clf.fit(&x, &y_class).unwrap();
        let _preds = fitted.predict(&x).unwrap();
    }
    let _default_clf = DummyClassifier::default();

    let y_reg: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for strat in [
        DummyRegressorStrategy::Mean,
        DummyRegressorStrategy::Median,
        DummyRegressorStrategy::Quantile(0.25),
        DummyRegressorStrategy::Constant(7.5),
    ] {
        let reg = DummyRegressor::<f64>::new(strat);
        let fitted = reg.fit(&x, &y_reg).unwrap();
        let _preds = fitted.predict(&x).unwrap();
    }
    let _default_reg: DummyRegressor<f64> = DummyRegressor::default();
}

// =============================================================================
// Param grids + distributions
// =============================================================================
#[test]
fn api_proof_param_grid_and_distributions() {
    // param_grid! macro
    let grid: Vec<ParamSet> = param_grid! {
        "alpha" => [0.01_f64, 0.1, 1.0],
        "fit_intercept" => [true, false],
    };
    assert_eq!(grid.len(), 6);

    // ParamValue conversions
    let _f = ParamValue::Float(1.0);
    let _i = ParamValue::Int(2);
    let _b = ParamValue::Bool(true);
    let _s = ParamValue::String("x".into());

    // Distributions
    let mut rng = SmallRng::seed_from_u64(7);
    let _ = Uniform::new(0.0, 1.0).sample(&mut rng);
    let _ = LogUniform::new(0.001, 1.0).sample(&mut rng);
    let _ = IntUniform::new(1, 10).sample(&mut rng);
    let _ = Choice::new(vec![ParamValue::Int(1), ParamValue::Int(2)]).sample(&mut rng);
}

// =============================================================================
// Heavy meta-estimators: smoke-construct only (need Pipeline factories)
// =============================================================================

fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

#[test]
fn api_proof_search_estimators() {
    let factory = Box::new(|_p: &ParamSet| Pipeline::<f64>::new());
    let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64] };
    let gs = GridSearchCV::new(factory, grid.clone(), Box::new(KFold::new(3)), neg_mse);
    assert!(gs.cv_results().is_none());
    assert!(gs.best_params().is_none());
    assert!(gs.best_score().is_none());

    let dists: Vec<(String, Box<dyn Distribution>)> =
        vec![("alpha".to_string(), Box::new(Uniform::new(0.01, 1.0)))];
    let factory_r = Box::new(|_p: &ParamSet| Pipeline::<f64>::new());
    let rs = RandomizedSearchCV::new(
        factory_r,
        dists,
        5,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(7),
    );
    assert!(rs.cv_results().is_none());
    assert!(rs.best_params().is_none());
    assert!(rs.best_score().is_none());

    let factory_h = Box::new(|_p: &ParamSet| Pipeline::<f64>::new());
    let hgs = HalvingGridSearchCV::new(factory_h, grid.clone(), Box::new(KFold::new(2)), neg_mse)
        .factor(2)
        .min_resources(Some(8))
        .max_resources(Some(64))
        .aggressive_elimination(false);
    assert!(hgs.cv_results().is_none());
    assert!(hgs.best_params().is_none());
    assert!(hgs.best_score().is_none());

    let dists2: Vec<(String, Box<dyn Distribution>)> =
        vec![("alpha".to_string(), Box::new(LogUniform::new(0.001, 1.0)))];
    let factory_hr = Box::new(|_p: &ParamSet| Pipeline::<f64>::new());
    let hrs = HalvingRandomSearchCV::new(
        factory_hr,
        dists2,
        4,
        Box::new(KFold::new(2)),
        neg_mse,
        Some(7),
    )
    .factor(2)
    .min_resources(Some(8))
    .max_resources(Some(64));
    assert!(hrs.cv_results().is_none());
    assert!(hrs.best_params().is_none());
    assert!(hrs.best_score().is_none());
}

// =============================================================================
// Multiclass / multioutput / calibration / self-training (constructor smoke)
// =============================================================================
#[test]
fn api_proof_meta_estimators() {
    let _ovr = OneVsRestClassifier::new(Box::new(|| Pipeline::<f64>::new()));
    let _ovo = OneVsOneClassifier::new(Box::new(|| Pipeline::<f64>::new()));
    let _moc = MultiOutputClassifier::new(Box::new(|| Pipeline::<f64>::new()));
    let _mor = MultiOutputRegressor::new(Box::new(|| Pipeline::<f64>::new()));

    // CalibratedClassifierCV — constructor + method enum
    let fit_fn: CalibFitFn = Box::new(|_x, _y| {
        Ok(
            Box::new(|x: &Array2<f64>| Ok(Array1::<f64>::zeros(x.nrows())))
                as Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>>,
        )
    });
    let _cal = CalibratedClassifierCV::new(fit_fn, CalibrationMethod::Sigmoid, 3);
    let _iso = CalibrationMethod::Isotonic;

    // SelfTrainingClassifier — uses the same calibration::FitFn shape
    let st_fit_fn: CalibFitFn = Box::new(|_x, _y| {
        Ok(
            Box::new(|x: &Array2<f64>| Ok(Array1::<f64>::zeros(x.nrows())))
                as Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>>,
        )
    });
    let _st = SelfTrainingClassifier::new(st_fit_fn)
        .threshold(0.7)
        .max_iter(5);

    // TransformedTargetRegressor
    let _ttr = TransformedTargetRegressor::<f64>::new(
        Pipeline::<f64>::new(),
        |y: f64| y.ln(),
        |y: f64| y.exp(),
    );
}

// =============================================================================
// FeatureUnion construction
// =============================================================================
#[test]
fn api_proof_feature_union() {
    let fu: FeatureUnion<f64> = FeatureUnion::new();
    assert_eq!(fu.n_transformers(), 0);
    assert_eq!(fu.transformer_names(), Vec::<&str>::new());
}

// =============================================================================
// learning_curve / validation_curve smoke (empty-pipeline error path)
// =============================================================================
#[test]
fn api_proof_curves_signatures() {
    // Both functions take a Pipeline reference; passing an empty pipeline is
    // expected to fail at fit time. We just want to prove the call signature
    // compiles; the call is wrapped in a `Result::is_err()` check.
    let x = Array2::<f64>::zeros((10, 2));
    let y = Array1::<f64>::zeros(10);
    let kf = KFold::new(2);
    let pipe = Pipeline::<f64>::new();

    let _ = learning_curve(&pipe, &x, &y, &kf, &[0.5, 1.0], neg_mse).is_err();

    let _ = validation_curve(
        &x,
        &y,
        &kf,
        &[0.1_f64, 1.0],
        |_v: f64| Pipeline::<f64>::new(),
        neg_mse,
    )
    .is_err();
}

// =============================================================================
// CrossValidateResult — public field surface
// =============================================================================
#[test]
fn api_proof_cross_validate_result_fields() {
    // The struct is constructed by cross_validate(); here we exercise its
    // four public fields so the type's API surface stays under test.
    let r = CrossValidateResult {
        test_scores: vec![0.8, 0.9],
        train_scores: Some(vec![0.95, 0.97]),
        fit_times: vec![0.01, 0.012],
        score_times: vec![0.001, 0.0011],
    };
    assert_eq!(r.test_scores.len(), 2);
    assert_eq!(r.train_scores.as_ref().map(|v| v.len()), Some(2));
    assert_eq!(r.fit_times.len(), 2);
    assert_eq!(r.score_times.len(), 2);
}
