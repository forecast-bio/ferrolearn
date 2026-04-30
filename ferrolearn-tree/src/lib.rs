//! # ferrolearn-tree
//!
//! Decision tree and ensemble tree models for the ferrolearn machine learning framework.
//!
//! This crate provides implementations of:
//!
//! - **[`DecisionTreeClassifier`]** / **[`DecisionTreeRegressor`]** — CART decision trees
//!   with configurable splitting criteria, depth limits, and minimum sample constraints.
//! - **[`RandomForestClassifier`]** / **[`RandomForestRegressor`]** — Bootstrap-aggregated
//!   ensembles of decision trees with random feature subsets, built in parallel via `rayon`.
//! - **[`GradientBoostingClassifier`]** / **[`GradientBoostingRegressor`]** — Gradient boosting
//!   ensembles that sequentially fit trees to the negative gradient of a loss function.
//! - **[`HistGradientBoostingClassifier`]** / **[`HistGradientBoostingRegressor`]** —
//!   Histogram-based gradient boosting with O(n_bins) split finding, subtraction trick,
//!   native NaN support, and optional best-first (leaf-wise) growth.
//! - **[`AdaBoostClassifier`]** — Adaptive Boosting using decision tree stumps with
//!   SAMME and SAMME.R algorithms.
//! - **[`ExtraTreeClassifier`]** / **[`ExtraTreeRegressor`]** — Extremely randomized
//!   trees where split thresholds are chosen randomly rather than via exhaustive search.
//! - **[`ExtraTreesClassifier`]** / **[`ExtraTreesRegressor`]** — Ensembles of
//!   extremely randomized trees with Rayon parallel fitting. No bootstrap by default.
//! - **[`IsolationForest`]** — Anomaly detection via random isolation trees.
//! - **[`VotingClassifier`]** / **[`VotingRegressor`]** — Ensembles of decision trees
//!   with varying hyperparameters, aggregated by majority vote or averaging.
//! - **[`RandomTreesEmbedding`]** — Unsupervised feature transformation via one-hot
//!   encoded leaf indices across an ensemble of randomly built trees.
//!
//! # Design
//!
//! Each model follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `DecisionTreeClassifier<F>`) holds hyperparameters
//!   and implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a new fitted type (e.g., `FittedDecisionTreeClassifier<F>`)
//!   that implements [`Predict`](ferrolearn_core::Predict).
//! - Calling `predict()` on an unfitted model is a compile-time error.
//!
//! # Pipeline Integration
//!
//! All models implement [`PipelineEstimator`](ferrolearn_core::pipeline::PipelineEstimator)
//! for `f64`, allowing them to be used as the final step in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Float Generics
//!
//! All models are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod adaboost;
pub mod adaboost_regressor;
pub mod bagging;
pub mod decision_tree;
pub mod extra_tree;
pub mod extra_trees_ensemble;
pub mod gradient_boosting;
pub mod hist_gradient_boosting;
pub mod isolation_forest;
pub mod random_forest;
pub mod random_trees_embedding;
pub mod voting;

// Re-export the main types at the crate root.
pub use adaboost::{AdaBoostAlgorithm, AdaBoostClassifier, FittedAdaBoostClassifier};
pub use adaboost_regressor::{AdaBoostLoss, AdaBoostRegressor, FittedAdaBoostRegressor};
pub use bagging::{
    BaggingClassifier, BaggingRegressor, FittedBaggingClassifier, FittedBaggingRegressor,
};
pub use decision_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor,
    FittedDecisionTreeClassifier, FittedDecisionTreeRegressor, Node, RegressionCriterion,
};
pub use extra_tree::{
    ExtraTreeClassifier, ExtraTreeRegressor, FittedExtraTreeClassifier, FittedExtraTreeRegressor,
};
pub use extra_trees_ensemble::{
    ExtraTreesClassifier, ExtraTreesRegressor, FittedExtraTreesClassifier,
    FittedExtraTreesRegressor,
};
pub use gradient_boosting::{
    ClassificationLoss, FittedGradientBoostingClassifier, FittedGradientBoostingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, RegressionLoss,
};
pub use hist_gradient_boosting::{
    FittedHistGradientBoostingClassifier, FittedHistGradientBoostingRegressor,
    HistClassificationLoss, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    HistNode, HistRegressionLoss,
};
pub use isolation_forest::{FittedIsolationForest, IsolationForest};
pub use random_forest::{
    FittedRandomForestClassifier, FittedRandomForestRegressor, MaxFeatures, RandomForestClassifier,
    RandomForestRegressor,
};
pub use random_trees_embedding::{FittedRandomTreesEmbedding, RandomTreesEmbedding};
pub use voting::{
    FittedVotingClassifier, FittedVotingRegressor, VotingClassifier, VotingRegressor,
};

use ndarray::{Array1, Array2};
use num_traits::Float;

/// Element-wise natural log of a probability matrix, used as the body of
/// every classifier `predict_log_proba` method in this crate. Clamps
/// values below `1e-300` so `ln(0)` never produces `-inf` / `NaN`.
pub(crate) fn log_proba<F: Float>(proba: &Array2<F>) -> Array2<F> {
    let eps = F::from(1e-300).unwrap();
    proba.mapv(|p| if p > eps { p.ln() } else { eps.ln() })
}

/// Mean accuracy: `(sum(predictions == targets)) / n`.
///
/// Used as the body of every classifier `score(&self, x, y)` method in
/// this crate to mirror sklearn's `ClassifierMixin.score`.
pub(crate) fn mean_accuracy<F: Float>(predictions: &Array1<usize>, targets: &Array1<usize>) -> F {
    let n = targets.len();
    if n == 0 {
        return F::zero();
    }
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    F::from(correct).unwrap() / F::from(n).unwrap()
}

/// R² coefficient of determination: `1 - SSres / SStot`.
///
/// Used as the body of every regressor `score(&self, x, y)` method in
/// this crate to mirror sklearn's `RegressorMixin.score`. Constant-y
/// returns `1.0` if predictions are also constant-perfect, else
/// `F::neg_infinity()` to flag the genuine miss.
pub(crate) fn r2_score<F: Float>(y_pred: &Array1<F>, y_true: &Array1<F>) -> F {
    let n = y_true.len();
    if n == 0 {
        return F::zero();
    }
    let mean = y_true.iter().copied().fold(F::zero(), |a, b| a + b) / F::from(n).unwrap();
    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    for i in 0..n {
        let r = y_true[i] - y_pred[i];
        let t = y_true[i] - mean;
        ss_res = ss_res + r * r;
        ss_tot = ss_tot + t * t;
    }
    if ss_tot == F::zero() {
        if ss_res == F::zero() {
            F::one()
        } else {
            F::neg_infinity()
        }
    } else {
        F::one() - ss_res / ss_tot
    }
}
