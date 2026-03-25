//! # ferrolearn-kernel
//!
//! Multivariate kernel regression with automatic bandwidth selection,
//! heteroscedasticity diagnostics, and robust confidence intervals.
//!
//! This crate provides:
//!
//! - **[`NadarayaWatson`]** — Local constant (order 0) kernel regression.
//! - **[`LocalPolynomialRegression`]** — Local polynomial regression (orders 0–3+)
//!   with Tikhonov regularization.
//! - **[`CrossValidatedBandwidth`]** — Automatic bandwidth selection via LOO or
//!   k-fold cross-validation with O(n) hat matrix shortcut for NW.
//! - **Kernel functions** — Gaussian, Epanechnikov, Tricube, Biweight, Triweight,
//!   Uniform, Cosine, plus custom kernels via the [`Kernel`] trait.
//! - **Diagnostics** — Goodness of fit, heteroscedasticity tests (White,
//!   Breusch-Pagan, Goldfeld-Quandt, Dette-Munk-Wagner), residual analysis.
//! - **Confidence intervals** — Wild bootstrap CI with bias corrections,
//!   Fan-Yao variance estimation, conformal calibration.
//!
//! # Design
//!
//! Each estimator follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `NadarayaWatson<F>`) holds hyperparameters and
//!   implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` returns a fitted type (e.g., `FittedNadarayaWatson<F>`)
//!   that implements [`Predict`](ferrolearn_core::Predict).
//! - Calling `predict()` on an unfitted model is a compile-time error.

pub mod bandwidth;
pub mod confidence;
pub mod diagnostics;
pub mod hat_matrix;
pub mod kernel_ridge;
pub mod kernels;
pub mod local_polynomial;
pub mod nadaraya_watson;
pub mod nystroem;
pub mod rbf_sampler;
pub mod weights;

pub use bandwidth::{CrossValidatedBandwidth, CvStrategy, scott_bandwidth, silverman_bandwidth};
pub use confidence::{
    BiasCorrection, ConfidenceIntervalResult, ConformalResult, VarianceFunctionResult,
    conformal_calibrate_ci, fan_yao_variance_estimation, wild_bootstrap_confidence_intervals,
};
pub use diagnostics::{
    GoodnessOfFit, HeteroscedasticityTest, HeteroscedasticityTestResult, ResidualDiagnosticsResult,
    heteroscedasticity_test, residual_diagnostics,
};
pub use kernel_ridge::{FittedKernelRidge, KernelRidge};
pub use kernels::{
    BiweightKernel, CosineKernel, DynKernel, EpanechnikovKernel, GaussianKernel, Kernel,
    TricubeKernel, TriweightKernel, UniformKernel,
};
pub use local_polynomial::{FittedLocalPolynomialRegression, LocalPolynomialRegression};
pub use nadaraya_watson::{FittedNadarayaWatson, NadarayaWatson};
pub use nystroem::{FittedNystroem, KernelType, Nystroem};
pub use rbf_sampler::{FittedRBFSampler, RBFSampler};
