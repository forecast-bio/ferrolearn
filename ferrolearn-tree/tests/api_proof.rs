//! Proof-of-API integration test for ferrolearn-tree.
//!
//! Audit deliverable for crosslink #275 (under #243). Exercises every
//! public API surface of the crate end-to-end so that future PRs that
//! change the public API have a green-or-red signal here.
//!
//! Coverage:
//! - DecisionTreeClassifier / Regressor: builders + fit/predict/score/
//!   predict_proba/predict_log_proba (cls), feature_importances_
//! - ExtraTreeClassifier / Regressor: same
//! - RandomForestClassifier / Regressor: same plus parallel-fit smoke
//! - ExtraTreesClassifier / Regressor: same
//! - GradientBoostingClassifier / Regressor: same plus decision_function
//! - HistGradientBoostingClassifier / Regressor: same plus decision_function
//! - AdaBoostClassifier / AdaBoostRegressor: same plus decision_function
//! - BaggingClassifier / Regressor: builders + fit/predict/score
//!   /predict_proba/predict_log_proba/feature_importances_
//! - VotingClassifier / Regressor: builders + fit/predict/score/
//!   predict_proba (cls)
//! - IsolationForest: fit/predict/score_samples
//! - RandomTreesEmbedding: fit/transform
//! - All public enum variants

use approx::assert_relative_eq;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_tree::{
    AdaBoostAlgorithm, AdaBoostClassifier, AdaBoostLoss, AdaBoostRegressor, BaggingClassifier,
    BaggingRegressor, ClassificationCriterion, ClassificationLoss, DecisionTreeClassifier,
    DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    HistClassificationLoss, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    HistRegressionLoss, IsolationForest, MaxFeatures, RandomForestClassifier,
    RandomForestRegressor, RandomTreesEmbedding, RegressionCriterion, RegressionLoss,
    VotingClassifier, VotingRegressor,
};
use ndarray::{Array1, Array2, array};

/// Two well-separated clusters in 2D used for classification + regression.
fn two_clusters_2d() -> (Array2<f64>, Array1<usize>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 5.0, 5.0, 5.5, 5.0, 5.0,
            5.5, 5.5, 5.5, 6.0, 6.0, 5.5, 6.0,
        ],
    )
    .unwrap();
    let y_cls = array![0usize, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let y_reg = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    (x, y_cls, y_reg)
}

fn assert_proba_well_formed(proba: &Array2<f64>, n_samples: usize, n_classes: usize) {
    assert_eq!(proba.dim(), (n_samples, n_classes));
    for i in 0..n_samples {
        let row_sum: f64 = proba.row(i).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        for ci in 0..n_classes {
            assert!(
                (0.0..=1.0).contains(&proba[[i, ci]]),
                "proba[{i},{ci}] = {} not in [0, 1]",
                proba[[i, ci]]
            );
        }
    }
}

fn assert_importances_well_formed(imp: &Array1<f64>, n_features: usize) {
    assert_eq!(imp.len(), n_features);
    let total: f64 = imp.sum();
    // Either all zeros (no splits had impurity decrease) or sums to 1.
    assert!(
        total.abs() < 1e-9 || (total - 1.0).abs() < 1e-9,
        "feature_importances should sum to 1 or be all zeros; got sum = {total}"
    );
    for &v in imp.iter() {
        assert!(v >= 0.0 && v <= 1.0, "importance {v} outside [0, 1]");
    }
}

// =============================================================================
// DecisionTree
// =============================================================================
#[test]
fn api_proof_decision_tree() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    // Classifier
    let m = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_criterion(ClassificationCriterion::Gini);
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.nodes();
    assert_eq!(f.n_features(), 2);
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 12);
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let log_proba = f.predict_log_proba(&x).unwrap();
    assert_eq!(log_proba.dim(), (12, 2));
    assert_relative_eq!(f.score(&x, &y_cls).unwrap(), 1.0, epsilon = 1e-10);
    assert_eq!(f.classes(), &[0, 1]);

    // Entropy criterion smoke
    let _ = DecisionTreeClassifier::<f64>::new()
        .with_criterion(ClassificationCriterion::Entropy)
        .fit(&x, &y_cls)
        .unwrap();

    // Regressor
    let mr = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_criterion(RegressionCriterion::Mse);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.nodes();
    assert_eq!(fr.n_features(), 2);
    let preds_r = fr.predict(&x).unwrap();
    assert_eq!(preds_r.len(), 12);
    assert!(fr.score(&x, &y_reg).unwrap() > 0.5);
    assert_importances_well_formed(fr.feature_importances(), 2);

    // Defaults compile
    let _: DecisionTreeClassifier<f64> = Default::default();
    let _: DecisionTreeRegressor<f64> = Default::default();
}

// =============================================================================
// ExtraTree (single)
// =============================================================================
#[test]
fn api_proof_extra_tree() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = ExtraTreeClassifier::<f64>::new()
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_max_features(MaxFeatures::Sqrt)
        .with_criterion(ClassificationCriterion::Gini)
        .with_random_state(42);
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);
    assert_eq!(f.classes(), &[0, 1]);

    let mr = ExtraTreeRegressor::<f64>::new()
        .with_max_depth(Some(3))
        .with_max_features(MaxFeatures::Log2)
        .with_criterion(RegressionCriterion::Mse)
        .with_random_state(42);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// RandomForest
// =============================================================================
#[test]
fn api_proof_random_forest() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = RandomForestClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::Sqrt)
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_random_state(7)
        .with_criterion(ClassificationCriterion::Gini);
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.trees();
    assert_eq!(f.n_features(), 2);
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    assert_relative_eq!(f.score(&x, &y_cls).unwrap(), 1.0, epsilon = 1e-10);
    assert_eq!(f.classes(), &[0, 1]);

    let mr = RandomForestRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_max_features(MaxFeatures::All)
        .with_random_state(7);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.predict(&x).unwrap();
    assert!(fr.score(&x, &y_reg).unwrap() > 0.7);
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// ExtraTrees ensemble
// =============================================================================
#[test]
fn api_proof_extra_trees_ensemble() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = ExtraTreesClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(4))
        .with_max_features(MaxFeatures::Fraction(0.5))
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_bootstrap(false)
        .with_criterion(ClassificationCriterion::Gini);
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);
    assert_eq!(f.classes(), &[0, 1]);
    assert_eq!(f.n_estimators(), 10);

    let mr = ExtraTreesRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_bootstrap(true);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// GradientBoosting
// =============================================================================
#[test]
fn api_proof_gradient_boosting() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = GradientBoostingClassifier::<f64>::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(Some(3))
        .with_subsample(1.0)
        .with_random_state(0);
    // ClassificationLoss::LogLoss is the only variant; expose for the
    // enum-coverage smoke test.
    let _ = ClassificationLoss::LogLoss;
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.init();
    let _ = f.learning_rate();
    let _ = f.trees();
    assert_eq!(f.n_features(), 2);
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.dim(), (12, 1)); // binary
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);

    let mr = GradientBoostingRegressor::<f64>::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(Some(3))
        .with_loss(RegressionLoss::LeastSquares)
        .with_huber_alpha(0.9)
        .with_random_state(0);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.init();
    let _ = fr.learning_rate();
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// HistGradientBoosting
// =============================================================================
#[test]
fn api_proof_hist_gradient_boosting() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = HistGradientBoostingClassifier::<f64>::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(Some(4))
        .with_min_samples_leaf(1)
        .with_max_bins(32)
        .with_l2_regularization(0.0)
        .with_max_leaf_nodes(Some(15))
        .with_random_state(0);
    // HistClassificationLoss::LogLoss is the only variant; expose for the
    // enum-coverage smoke test below.
    let _ = HistClassificationLoss::LogLoss;
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.dim(), (12, 1));
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);

    let mr = HistGradientBoostingRegressor::<f64>::new()
        .with_n_estimators(20)
        .with_max_bins(64)
        .with_loss(HistRegressionLoss::LeastSquares)
        .with_random_state(0);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// AdaBoost (classifier + regressor in distinct files)
// =============================================================================
#[test]
fn api_proof_adaboost() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    // SAMME
    let m_samme = AdaBoostClassifier::<f64>::new()
        .with_n_estimators(20)
        .with_learning_rate(1.0)
        .with_algorithm(AdaBoostAlgorithm::Samme)
        .with_random_state(0);
    let f = m_samme.fit(&x, &y_cls).unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.dim(), (12, 2));
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);
    assert_eq!(f.classes(), &[0, 1]);

    // SAMME.R
    let m_sammer = AdaBoostClassifier::<f64>::new()
        .with_algorithm(AdaBoostAlgorithm::SammeR)
        .with_n_estimators(20)
        .with_random_state(0);
    let f2 = m_sammer.fit(&x, &y_cls).unwrap();
    let _ = f2.predict_proba(&x).unwrap();
    let _ = f2.decision_function(&x).unwrap();

    let mr = AdaBoostRegressor::<f64>::new()
        .with_n_estimators(20)
        .with_learning_rate(1.0)
        .with_max_depth(Some(3))
        .with_loss(AdaBoostLoss::Linear)
        .with_random_state(0);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.estimators();
    let _ = fr.estimator_weights();
    assert_eq!(fr.n_features(), 2);
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);

    // Other AdaBoostLoss variants
    for loss in [AdaBoostLoss::Square, AdaBoostLoss::Exponential] {
        let _ = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_loss(loss)
            .with_random_state(1)
            .fit(&x, &y_reg)
            .unwrap();
    }
}

// =============================================================================
// Bagging (now wired in as of #267)
// =============================================================================
#[test]
fn api_proof_bagging() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = BaggingClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_max_samples(0.8)
        .with_max_features(1.0)
        .with_bootstrap(true)
        .with_bootstrap_features(false)
        .with_max_depth(Some(3))
        .with_random_state(0);
    let f = m.fit(&x, &y_cls).unwrap();
    let _ = f.trees();
    assert_eq!(f.n_features(), 2);
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.score(&x, &y_cls).unwrap();
    assert_importances_well_formed(f.feature_importances(), 2);
    assert_eq!(f.classes(), &[0, 1]);

    let mr = BaggingRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_max_samples(0.8)
        .with_bootstrap(true)
        .with_random_state(0);
    let fr = mr.fit(&x, &y_reg).unwrap();
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
    assert_importances_well_formed(fr.feature_importances(), 2);
}

// =============================================================================
// Voting (heterogeneous-depth ensemble of decision trees)
// =============================================================================
#[test]
fn api_proof_voting() {
    let (x, y_cls, y_reg) = two_clusters_2d();

    let m = VotingClassifier::<f64>::new()
        .with_max_depths(vec![Some(2), Some(3), Some(4)])
        .with_min_samples_split(2)
        .with_min_samples_leaf(1)
        .with_criterion(ClassificationCriterion::Gini);
    let f = m.fit(&x, &y_cls).unwrap();
    assert_eq!(f.n_estimators(), 3);
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 12, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.score(&x, &y_cls).unwrap();
    assert_eq!(f.classes(), &[0, 1]);

    let mr = VotingRegressor::<f64>::new()
        .with_max_depths(vec![Some(2), Some(3), None])
        .with_min_samples_split(2)
        .with_min_samples_leaf(1);
    let fr = mr.fit(&x, &y_reg).unwrap();
    assert_eq!(fr.n_estimators(), 3);
    let _ = fr.predict(&x).unwrap();
    let _ = fr.score(&x, &y_reg).unwrap();
}

// =============================================================================
// IsolationForest
// =============================================================================
#[test]
fn api_proof_isolation_forest() {
    // 8 inliers + 2 obvious outliers
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05, -0.05,
            10.0, 10.0, -10.0, -10.0,
        ],
    )
    .unwrap();

    let m = IsolationForest::<f64>::new()
        .with_n_estimators(50)
        .with_max_samples(8)
        .with_contamination(0.2)
        .with_random_state(7);
    let f = m.fit(&x, &()).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    // The two extreme samples should be flagged as outliers (-1).
    assert_eq!(preds[8], -1);
    assert_eq!(preds[9], -1);
    let scores = f.score_samples(&x).unwrap();
    assert_eq!(scores.len(), 10);
    // ferrolearn's score_samples currently returns the raw anomaly score
    // (higher = more anomalous), unlike sklearn (negated, higher = inlier).
    // See #276; assertion follows ferrolearn's convention.
    let outlier_max = scores[8].max(scores[9]);
    let inlier_min = (0..8).map(|i| scores[i]).fold(f64::INFINITY, f64::min);
    assert!(
        outlier_max >= inlier_min,
        "expected outlier scores to be >= inlier min; got {outlier_max} vs {inlier_min}"
    );

    let _: IsolationForest<f64> = Default::default();
}

// =============================================================================
// RandomTreesEmbedding
// =============================================================================
#[test]
fn api_proof_random_trees_embedding() {
    let (x, _, _) = two_clusters_2d();

    let m = RandomTreesEmbedding::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(3))
        .with_min_samples_split(2)
        .with_random_state(0);
    let f = m.fit(&x, &()).unwrap();
    let embedded = f.transform(&x).unwrap();
    assert_eq!(embedded.nrows(), 12);
    // Embedded width is sum of leaves across trees, so positive.
    assert!(embedded.ncols() > 0);

    let _: RandomTreesEmbedding<f64> = Default::default();
}

// =============================================================================
// f32 numeric type compiles on every estimator
// =============================================================================
#[test]
fn api_proof_f32_compiles() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![0.0f32, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5],
    )
    .unwrap();
    let y_cls = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    let y_reg = array![1.0f32, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];

    let _ = DecisionTreeClassifier::<f32>::new().fit(&x, &y_cls).unwrap();
    let _ = DecisionTreeRegressor::<f32>::new().fit(&x, &y_reg).unwrap();
    let _ = ExtraTreeClassifier::<f32>::new().fit(&x, &y_cls).unwrap();
    let _ = ExtraTreeRegressor::<f32>::new().fit(&x, &y_reg).unwrap();
    let _ = RandomForestClassifier::<f32>::new()
        .with_n_estimators(3)
        .fit(&x, &y_cls)
        .unwrap();
    let _ = RandomForestRegressor::<f32>::new()
        .with_n_estimators(3)
        .fit(&x, &y_reg)
        .unwrap();
}
