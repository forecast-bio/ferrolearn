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
//! - **[`AdaBoostClassifier`]** — Adaptive Boosting using decision tree stumps with
//!   SAMME and SAMME.R algorithms.
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
pub mod decision_tree;
pub mod gradient_boosting;
pub mod random_forest;

// Re-export the main types at the crate root.
pub use adaboost::{AdaBoostAlgorithm, AdaBoostClassifier, FittedAdaBoostClassifier};
pub use decision_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor,
    FittedDecisionTreeClassifier, FittedDecisionTreeRegressor, Node, RegressionCriterion,
};
pub use gradient_boosting::{
    ClassificationLoss, FittedGradientBoostingClassifier, FittedGradientBoostingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, RegressionLoss,
};
pub use random_forest::{
    FittedRandomForestClassifier, FittedRandomForestRegressor, MaxFeatures, RandomForestClassifier,
    RandomForestRegressor,
};
