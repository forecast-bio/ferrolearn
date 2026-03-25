//! # ferrolearn-linear
//!
//! Linear models for the ferrolearn machine learning framework.
//!
//! This crate provides implementations of the most common linear models
//! for both regression and classification tasks:
//!
//! - **[`LinearRegression`]** — Ordinary Least Squares via QR decomposition
//! - **[`Ridge`]** — L2-regularized regression via Cholesky decomposition
//! - **[`Lasso`]** — L1-regularized regression via coordinate descent
//! - **[`ElasticNet`]** — Combined L1/L2 regularization via coordinate descent
//! - **[`BayesianRidge`]** — Bayesian Ridge with automatic regularization tuning
//! - **[`HuberRegressor`]** — Robust regression via IRLS with Huber loss
//! - **[`LogisticRegression`]** — Binary and multiclass classification via L-BFGS
//! - **[`LogisticRegressionCV`]** — Logistic regression with cross-validated C
//! - **[`LinearSVC`]** — Linear Support Vector Classifier (primal coordinate descent)
//! - **[`LinearSVR`]** — Linear Support Vector Regressor (primal coordinate descent)
//! - **[`QDA`]** — Quadratic Discriminant Analysis
//! - **[`ARDRegression`]** — Automatic Relevance Determination (Bayesian, per-feature priors)
//! - **[`RidgeClassifier`]** — Ridge regression applied to classification
//!
//! All models implement the [`ferrolearn_core::Fit`] and [`ferrolearn_core::Predict`]
//! traits, and produce fitted types that implement [`ferrolearn_core::introspection::HasCoefficients`].
//!
//! # Design
//!
//! Each model follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `LinearRegression<F>`) holds hyperparameters
//!   and implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a new fitted type (e.g., `FittedLinearRegression<F>`)
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

pub mod ard;
pub mod bayesian_ridge;
pub mod elastic_net;
pub mod huber_regressor;
pub mod isotonic;
pub mod lasso;
pub mod lda;
mod linalg;
pub mod linear_regression;
pub mod linear_svc;
pub mod linear_svr;
pub mod logistic_regression;
pub mod logistic_regression_cv;
mod optim;
pub mod qda;
pub mod ransac;
pub mod ridge;
pub mod ridge_classifier;
pub mod sgd;
pub mod svm;

// Re-export the main types at the crate root.
pub use ard::{ARDRegression, FittedARDRegression};
pub use bayesian_ridge::{BayesianRidge, FittedBayesianRidge};
pub use elastic_net::{ElasticNet, FittedElasticNet};
pub use huber_regressor::{FittedHuberRegressor, HuberRegressor};
pub use isotonic::{FittedIsotonicRegression, IsotonicRegression};
pub use lasso::{FittedLasso, Lasso};
pub use lda::{FittedLDA, LDA};
pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use linear_svc::{FittedLinearSVC, LinearSVC};
pub use linear_svr::{FittedLinearSVR, LinearSVR};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use logistic_regression_cv::{FittedLogisticRegressionCV, LogisticRegressionCV};
pub use qda::{FittedQDA, QDA};
pub use ransac::{FittedRANSACRegressor, RANSACRegressor};
pub use ridge::{FittedRidge, Ridge};
pub use ridge_classifier::{FittedRidgeClassifier, RidgeClassifier};
pub use sgd::{FittedSGDClassifier, FittedSGDRegressor, SGDClassifier, SGDRegressor};
pub use svm::{
    FittedSVC, FittedSVR, Kernel, LinearKernel, PolynomialKernel, RbfKernel, SVC, SVR,
    SigmoidKernel,
};
