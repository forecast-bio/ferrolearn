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
//! - **[`GaussianProcessRegressor`]** — Bayesian nonparametric regression with
//!   predictive mean and variance via Cholesky decomposition.
//! - **[`GaussianProcessClassifier`]** — Probabilistic classification via Laplace
//!   approximation (binary and multi-class one-vs-rest), with Rasmussen &
//!   Williams Algorithm 3.2 predictions and `log_marginal_likelihood()` for
//!   hyperparameter selection.
//! - **GP Kernels** — RBF, Matern (0.5/1.5/2.5), Constant, White, DotProduct,
//!   plus Sum and Product kernel composition via the [`GPKernel`](gp_kernels::GPKernel) trait.
//! - **[`KernelRidge`]** — Kernel ridge regression in dual form
//!   `(K + αI) c = y` with RBF / Polynomial / Linear / Sigmoid / Laplacian
//!   / Cosine kernels.
//! - **[`Nystroem`]** — Low-rank kernel approximation via the Nyström method,
//!   producing a dense feature embedding usable with linear models.
//! - **[`RBFSampler`]** — Random Fourier features (Rahimi & Recht 2007) for
//!   approximating the RBF kernel with a randomized cosine feature map.
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
pub mod gaussian_process;
pub mod gp_classifier;
pub mod gp_kernels;
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
pub use gaussian_process::{FittedGaussianProcessRegressor, GaussianProcessRegressor};
pub use gp_classifier::{FittedGaussianProcessClassifier, GaussianProcessClassifier};
pub use gp_kernels::{
    ConstantKernel, DotProductKernel, GPKernel, MaternKernel, ProductKernel, RBFKernel, SumKernel,
    WhiteKernel,
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

use ndarray::{Array1, Array2};
use num_traits::Float;

/// Mean accuracy: `(sum(predictions == targets)) / n`. Mirrors sklearn
/// `ClassifierMixin.score`.
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

/// R² coefficient of determination. Mirrors sklearn
/// `RegressorMixin.score`. Constant-y returns `1.0` if perfect, else
/// `F::neg_infinity()`.
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

/// Element-wise log of a probability matrix, used as the body of every
/// classifier `predict_log_proba` method. Clamps below `1e-300` to avoid
/// `-inf` / `NaN`.
pub(crate) fn log_proba<F: Float>(proba: &Array2<F>) -> Array2<F> {
    let eps = F::from(1e-300).unwrap();
    proba.mapv(|p| if p > eps { p.ln() } else { eps.ln() })
}
