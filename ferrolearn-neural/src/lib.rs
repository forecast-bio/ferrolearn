//! # ferrolearn-neural
//!
//! Neural network estimators for the ferrolearn machine learning framework —
//! a scikit-learn equivalent for Rust.
//!
//! - [`MLPClassifier`] / [`MLPRegressor`] — feedforward multi-layer
//!   perceptrons trained via mini-batch gradient descent (SGD with momentum
//!   or Adam).
//! - [`BernoulliRBM`] — Bernoulli Restricted Boltzmann Machine trained via
//!   one-step Contrastive Divergence (CD-1).

use ndarray::Array2;
use num_traits::Float;

pub mod mlp;
pub mod rbm;

pub use mlp::{
    Activation, FittedMLPClassifier, FittedMLPRegressor, MLPClassifier, MLPRegressor, Solver,
};
pub use rbm::{BernoulliRBM, FittedBernoulliRBM};

/// Element-wise log of a probability matrix, used as the body of every
/// classifier `predict_log_proba` method in this crate. Clamps values
/// below `1e-300` to avoid `-inf` / `NaN`.
pub(crate) fn log_proba<F: Float>(proba: &Array2<F>) -> Array2<F> {
    let eps = F::from(1e-300).unwrap();
    proba.mapv(|p| if p > eps { p.ln() } else { eps.ln() })
}
