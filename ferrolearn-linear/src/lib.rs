//! # ferrolearn-linear
//!
//! Linear models for the ferrolearn machine learning framework.
//!
//! This crate provides implementations of the most common linear models
//! for both regression and classification tasks:
//!
//! - **[`LinearRegression`]** — Ordinary Least Squares via QR decomposition
//! - **[`Ridge`]** — L2-regularized regression via Cholesky decomposition
//! - **[`RidgeCV`]** — Ridge with built-in cross-validated alpha selection
//! - **[`Lasso`]** — L1-regularized regression via coordinate descent
//! - **[`LassoCV`]** — Lasso with built-in cross-validated alpha selection
//! - **[`ElasticNet`]** — Combined L1/L2 regularization via coordinate descent
//! - **[`ElasticNetCV`]** — ElasticNet with cross-validated (alpha, l1_ratio) selection
//! - **[`BayesianRidge`]** — Bayesian Ridge with automatic regularization tuning
//! - **[`HuberRegressor`]** — Robust regression via IRLS with Huber loss
//! - **[`LogisticRegression`]** — Binary and multiclass classification via L-BFGS
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

pub mod bayesian_ridge;
pub mod elastic_net;
pub mod elastic_net_cv;
pub mod huber_regressor;
pub mod isotonic;
pub mod lasso;
pub mod lasso_cv;
pub mod lda;
mod linalg;
pub mod linear_regression;
pub mod logistic_regression;
pub mod mlp;
mod optim;
pub mod ransac;
pub mod ridge;
pub mod ridge_cv;
pub mod sgd;
pub mod svm;

// Re-export the main types at the crate root.
pub use bayesian_ridge::{BayesianRidge, FittedBayesianRidge};
pub use elastic_net::{ElasticNet, FittedElasticNet};
pub use elastic_net_cv::{ElasticNetCV, FittedElasticNetCV};
pub use huber_regressor::{FittedHuberRegressor, HuberRegressor};
pub use isotonic::{FittedIsotonicRegression, IsotonicRegression};
pub use lasso::{FittedLasso, Lasso};
pub use lasso_cv::{FittedLassoCV, LassoCV};
pub use lda::{FittedLDA, LDA};
pub use linear_regression::{FittedLinearRegression, LinearRegression};
pub use logistic_regression::{FittedLogisticRegression, LogisticRegression};
pub use mlp::{
    Activation, FittedMLPClassifier, FittedMLPRegressor, MLPClassifier, MLPRegressor, Solver,
};
pub use ransac::{FittedRANSACRegressor, RANSACRegressor};
pub use ridge::{FittedRidge, Ridge};
pub use ridge_cv::{FittedRidgeCV, RidgeCV};
pub use sgd::{FittedSGDClassifier, FittedSGDRegressor, SGDClassifier, SGDRegressor};
pub use svm::{
    FittedSVC, FittedSVR, Kernel, LinearKernel, PolynomialKernel, RbfKernel, SVC, SVR,
    SigmoidKernel,
};
