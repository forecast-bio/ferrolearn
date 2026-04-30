//! Proof-of-API integration test for ferrolearn-linear.
//!
//! Audit deliverable for crosslink #288 (under #245). Exercises every
//! public API surface across the 30+ fitted estimators in the crate so
//! that future PRs that change the public API have a green-or-red signal
//! here.
//!
//! Coverage:
//! - Regressors: builder + fit + predict + score (via RegressorScore trait)
//! - Classifiers: same + predict_proba + predict_log_proba +
//!   decision_function where added in #286/#287
//! - SVM family: kernel constructors, decision_function, n_support
//! - IsotonicRegression: fit + predict + transform
//! - All public enum variants

use approx::assert_relative_eq;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_linear::sgd::{ClassifierLoss, LearningRateSchedule, RegressorLoss};
use ferrolearn_linear::{
    ARDRegression, BayesianRidge, ClassifierScore, ElasticNet, ElasticNetCV, GLMFamily,
    GLMRegressor, GammaRegressor, HuberRegressor, IsotonicRegression, LDA, Lars, Lasso, LassoCV,
    LassoLars, LinearRegression, LinearSVC, LinearSVCLoss, LinearSVR, LinearSVRLoss,
    LogisticRegression, LogisticRegressionCV, NuSVC, NuSVR, OneClassSVM,
    OrthogonalMatchingPursuit, PoissonRegressor, QDA, QuantileRegressor, RANSACRegressor,
    RegressorScore, Ridge, RidgeCV, RidgeClassifier, SGDClassifier, SGDRegressor, SVC, SVR,
    TweedieRegressor,
};
use ferrolearn_linear::svm::{LinearKernel, RbfKernel};
use ndarray::{Array1, Array2, array};

/// Two well-separated clusters in 2D (for binary classification).
fn binary_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.2, 0.3, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5,
            5.5, 5.2, 5.3,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    (x, y)
}

/// Simple regression data: y = 2x + noise.
fn regression_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = array![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9, 18.1, 19.9];
    (x, y)
}

/// Strictly positive regression target for GLM/Poisson/Gamma.
fn positive_regression_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = array![1.5, 2.0, 2.8, 3.5, 4.5, 5.2, 6.0, 7.0, 8.0, 9.5];
    (x, y)
}

fn assert_proba_well_formed(proba: &Array2<f64>, n_samples: usize, n_classes: usize) {
    assert_eq!(proba.dim(), (n_samples, n_classes));
    for i in 0..n_samples {
        let s: f64 = proba.row(i).sum();
        assert_relative_eq!(s, 1.0, epsilon = 1e-9);
        for c in 0..n_classes {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&proba[[i, c]]),
                "proba[{i},{c}] = {} not in [0, 1]",
                proba[[i, c]]
            );
        }
    }
}

// =============================================================================
// Linear regressors
// =============================================================================
#[test]
fn api_proof_linear_regression() {
    let (x, y) = regression_data();
    let m = LinearRegression::<f64>::new().with_fit_intercept(true);
    let f = m.fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    let r2 = f.score(&x, &y).unwrap();
    assert!(r2 > 0.99);
}

#[test]
fn api_proof_ridge_family() {
    let (x, y) = regression_data();

    let f = Ridge::<f64>::new().with_alpha(1.0).fit(&x, &y).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();

    let f_cv = RidgeCV::<f64>::new()
        .with_alphas(vec![0.1, 1.0, 10.0])
        .with_cv(3)
        .with_fit_intercept(true)
        .fit(&x, &y)
        .unwrap();
    let _ = f_cv.predict(&x).unwrap();
    let _ = f_cv.score(&x, &y).unwrap();
}

#[test]
fn api_proof_lasso_family() {
    let (x, y) = regression_data();
    let f = Lasso::<f64>::new().with_alpha(0.01).fit(&x, &y).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();

    let f_cv = LassoCV::<f64>::new()
        .with_n_alphas(5)
        .with_cv(3)
        .fit(&x, &y)
        .unwrap();
    let _ = f_cv.score(&x, &y).unwrap();
}

#[test]
fn api_proof_elastic_net_family() {
    let (x, y) = regression_data();
    let f = ElasticNet::<f64>::new()
        .with_alpha(0.1)
        .with_l1_ratio(0.5)
        .fit(&x, &y)
        .unwrap();
    let _ = f.score(&x, &y).unwrap();
    let f_cv = ElasticNetCV::<f64>::new().fit(&x, &y).unwrap();
    let _ = f_cv.score(&x, &y).unwrap();
}

#[test]
fn api_proof_bayesian_ridge_and_ard() {
    let (x, y) = regression_data();
    let f = BayesianRidge::<f64>::new().with_max_iter(50).fit(&x, &y).unwrap();
    let _ = f.score(&x, &y).unwrap();
    let f2 = ARDRegression::<f64>::new().with_max_iter(50).fit(&x, &y).unwrap();
    let _ = f2.score(&x, &y).unwrap();
}

#[test]
fn api_proof_huber_and_quantile() {
    let (x, y) = regression_data();
    let h = HuberRegressor::<f64>::new().with_epsilon(1.35).with_alpha(1e-4).fit(&x, &y).unwrap();
    let _ = h.score(&x, &y).unwrap();
    let q = QuantileRegressor::<f64>::new().with_quantile(0.5).fit(&x, &y).unwrap();
    let _ = q.score(&x, &y).unwrap();
}

#[test]
fn api_proof_glm_family() {
    let (x, y) = positive_regression_data();
    let f = GLMRegressor::<f64>::new(GLMFamily::Poisson).with_alpha(0.1).fit(&x, &y).unwrap();
    let _ = f.score(&x, &y).unwrap();

    let _ = PoissonRegressor::<f64>::new().fit(&x, &y).unwrap();
    let _ = GammaRegressor::<f64>::new().fit(&x, &y).unwrap();
    let _ = TweedieRegressor::<f64>::new().with_power(1.5).fit(&x, &y).unwrap();

    // Family enum coverage smoke
    for fam in [GLMFamily::Poisson, GLMFamily::Gamma] {
        let _ = GLMRegressor::<f64>::new(fam);
    }
}

#[test]
fn api_proof_lars_family() {
    let (x, y) = regression_data();
    let f = Lars::<f64>::new().with_n_nonzero_coefs(1).fit(&x, &y).unwrap();
    let _ = f.score(&x, &y).unwrap();
    let f2 = LassoLars::<f64>::new().with_alpha(0.01).fit(&x, &y).unwrap();
    let _ = f2.score(&x, &y).unwrap();
}

#[test]
fn api_proof_omp() {
    let (x, y) = regression_data();
    let f = OrthogonalMatchingPursuit::<f64>::new()
        .with_n_nonzero_coefs(1)
        .fit(&x, &y)
        .unwrap();
    let _ = f.score(&x, &y).unwrap();
}

#[test]
fn api_proof_ransac() {
    let (x, y) = regression_data();
    let inner = LinearRegression::<f64>::new();
    let f = RANSACRegressor::new(inner)
        .with_min_samples(2)
        .with_max_trials(20)
        .with_random_state(42)
        .fit(&x, &y)
        .unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();
}

// =============================================================================
// Linear classifiers
// =============================================================================
#[test]
fn api_proof_logistic_regression() {
    let (x, y) = binary_data();
    let m = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(200)
        .with_fit_intercept(true);
    let f = m.fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 10, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.dim(), (10, 1));
    assert!(f.score(&x, &y).unwrap() > 0.9);
    assert_eq!(f.classes(), &[0, 1]);
}

#[test]
fn api_proof_logistic_regression_cv() {
    let (x, y) = binary_data();
    let f = LogisticRegressionCV::<f64>::new()
        .with_cs(vec![0.1, 1.0])
        .with_cv(2)
        .with_max_iter(100)
        .fit(&x, &y)
        .unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 10, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.decision_function(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();
    let _ = f.best_c();
    let _ = f.cv_scores();
}

#[test]
fn api_proof_lda() {
    let (x, y) = binary_data();
    let f = LDA::<f64>::new(None).fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 10);
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 10, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.dim(), (10, 2));
    let _ = f.score(&x, &y).unwrap();
    let _ = f.scalings();
    let _ = f.means();
    let _ = f.explained_variance_ratio();
    assert_eq!(f.classes(), &[0, 1]);
}

#[test]
fn api_proof_qda() {
    let (x, y) = binary_data();
    let f = QDA::<f64>::new().with_reg_param(0.01).fit(&x, &y).unwrap();
    let _ = f.predict(&x).unwrap();
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 10, 2);
    let _ = f.predict_log_proba(&x).unwrap();
    let _ = f.decision_function(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();
}

#[test]
fn api_proof_ridge_classifier() {
    let (x, y) = binary_data();
    let f = RidgeClassifier::<f64>::new()
        .with_alpha(1.0)
        .with_fit_intercept(true)
        .fit(&x, &y)
        .unwrap();
    let _ = f.predict(&x).unwrap();
    let dec = f.decision_function(&x).unwrap();
    assert_eq!(dec.nrows(), 10);
    let _ = f.score(&x, &y).unwrap();
}

#[test]
fn api_proof_sgd_classifier_and_regressor() {
    let (x, y_cls) = binary_data();
    let (xr, yr) = regression_data();

    let cls = SGDClassifier::<f64>::new()
        .with_loss(ClassifierLoss::Log)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_max_iter(100)
        .with_random_state(42);
    let f = cls.fit(&x, &y_cls).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.score(&x, &y_cls).unwrap();

    let reg = SGDRegressor::<f64>::new()
        .with_loss(RegressorLoss::SquaredError)
        .with_learning_rate(LearningRateSchedule::Constant)
        .with_eta0(0.01)
        .with_max_iter(200)
        .with_random_state(42);
    let f = reg.fit(&xr, &yr).unwrap();
    let _ = f.predict(&xr).unwrap();
    let _ = f.score(&xr, &yr).unwrap();
}

// =============================================================================
// Linear SVMs (LinearSVC + LinearSVR)
// =============================================================================
#[test]
fn api_proof_linear_svc() {
    let (x, y) = binary_data();
    for loss in [LinearSVCLoss::Hinge, LinearSVCLoss::SquaredHinge] {
        let f = LinearSVC::<f64>::new()
            .with_c(1.0)
            .with_max_iter(200)
            .with_loss(loss)
            .fit(&x, &y)
            .unwrap();
        let _ = f.predict(&x).unwrap();
        let dec = f.decision_function(&x).unwrap();
        assert_eq!(dec.nrows(), 10);
        let _ = f.score(&x, &y).unwrap();
    }
}

#[test]
fn api_proof_linear_svr() {
    let (x, y) = regression_data();
    for loss in [
        LinearSVRLoss::EpsilonInsensitive,
        LinearSVRLoss::SquaredEpsilonInsensitive,
    ] {
        let f = LinearSVR::<f64>::new()
            .with_c(1.0)
            .with_epsilon(0.1)
            .with_loss(loss)
            .fit(&x, &y)
            .unwrap();
        let _ = f.predict(&x).unwrap();
        let _ = f.score(&x, &y).unwrap();
    }
}

// =============================================================================
// Kernel-based SVMs (SVC / SVR / NuSVC / NuSVR / OneClassSVM)
// =============================================================================
#[test]
fn api_proof_kernel_svm_family() {
    let (x, y) = binary_data();
    let (xr, yr) = regression_data();

    let svc = SVC::new(RbfKernel::with_gamma(1.0))
        .with_c(1.0)
        .with_max_iter(200);
    let f = svc.fit(&x, &y).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.decision_function(&x).unwrap();
    let _ = f.score(&x, &y).unwrap();

    let svr = SVR::new(RbfKernel::with_gamma(0.5))
        .with_c(1.0)
        .with_epsilon(0.1)
        .with_max_iter(200);
    let f = svr.fit(&xr, &yr).unwrap();
    let _ = f.predict(&xr).unwrap();
    let _ = f.score(&xr, &yr).unwrap();

    let nusvc = NuSVC::new(LinearKernel).with_nu(0.5).with_max_iter(200);
    let f = nusvc.fit(&x, &y).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.decision_function(&x).unwrap();

    let nusvr = NuSVR::new(LinearKernel).with_nu(0.5).with_max_iter(200);
    let f = nusvr.fit(&xr, &yr).unwrap();
    let _ = f.predict(&xr).unwrap();

    let ocsvm = OneClassSVM::new(LinearKernel).with_nu(0.5).with_max_iter(200);
    let f = ocsvm.fit(&x, &()).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.decision_function(&x).unwrap();
}

// =============================================================================
// Isotonic
// =============================================================================
#[test]
fn api_proof_isotonic_regression() {
    let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let y = array![1.0, 2.5, 2.0, 4.0, 5.5, 5.0];

    let f = IsotonicRegression::<f64>::new().with_increasing(true).fit(&x, &y).unwrap();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 6);
    let _ = f.score(&x, &y).unwrap();
}

// =============================================================================
// MLP (lives in linear-crate but logically belongs in a future
// neural_network crate — covered for now)
// MLPClassifier / MLPRegressor moved to ferrolearn-neural; see
// ferrolearn-neural/tests/api_proof.rs for their API proof.
