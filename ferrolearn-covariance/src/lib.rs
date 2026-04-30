//! # ferrolearn-covariance
//!
//! Covariance estimators for the ferrolearn machine learning framework — a
//! scikit-learn equivalent for Rust.
//!
//! All estimators follow the ferrolearn [`Fit`](ferrolearn_core::Fit)
//! trait pattern: the unfitted struct holds hyperparameters; calling
//! `fit()` returns a fitted struct that stores the learned covariance,
//! location (mean), and precision (inverse covariance).
//!
//! # Estimators
//!
//! - [`EmpiricalCovariance`] — maximum-likelihood covariance.
//! - [`ShrunkCovariance`] — fixed shrinkage toward a scaled identity.
//! - [`LedoitWolf`] — optimal Ledoit–Wolf shrinkage (data-driven).
//! - [`OAS`] — Oracle Approximating Shrinkage.
//! - [`MinCovDet`] — Minimum Covariance Determinant via FAST-MCD.
//! - [`EllipticEnvelope`] — outlier detection on top of MCD.
//! - [`GraphicalLasso`] — sparse inverse-covariance via L1-penalised log-likelihood.
//! - [`GraphicalLassoCV`] — cross-validated [`GraphicalLasso`].
//!
//! # Function-style helpers (sklearn parity)
//!
//! - [`empirical_covariance`] — one-shot empirical covariance.
//! - [`shrunk_covariance`] — apply fixed shrinkage to a precomputed cov.
//! - [`ledoit_wolf`] — return `(cov, shrinkage)`.
//! - [`ledoit_wolf_shrinkage`] — return only the shrinkage coefficient.
//! - [`oas`] — return `(cov, shrinkage)`.
//! - [`log_likelihood`] — Gaussian log-likelihood under a covariance.
//! - [`graphical_lasso`] — one-shot graphical-lasso fit.
//! - [`fast_mcd`] — robust MCD location/cov estimate.

pub mod covariance;
pub mod graphical_lasso;
pub mod helpers;

pub use covariance::{
    EllipticEnvelope, EmpiricalCovariance, FittedCovariance, FittedEllipticEnvelope,
    FittedLedoitWolf, FittedMinCovDet, FittedOAS, LedoitWolf, MinCovDet, OAS, ShrunkCovariance,
};
pub use graphical_lasso::{
    FittedGraphicalLasso, FittedGraphicalLassoCV, GraphicalLasso, GraphicalLassoCV, graphical_lasso,
};
pub use helpers::{
    empirical_covariance, fast_mcd, ledoit_wolf, ledoit_wolf_shrinkage, log_likelihood, oas,
    shrunk_covariance,
};
