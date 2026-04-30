//! # ferrolearn-bayes
//!
//! Bayesian methods for the ferrolearn machine learning framework.
//!
//! Two families of tools live here:
//!
//! 1. **Naive Bayes classifiers** — five variants for classification.
//! 2. **Conjugate priors** — closed-form posterior updates for parameter
//!    estimation. See [`conjugate`].
//!
//! This crate provides five Naive Bayes variants:
//!
//! - **[`GaussianNB`]** — Assumes Gaussian-distributed features. Suitable for
//!   continuous data.
//! - **[`MultinomialNB`]** — For discrete count data (e.g., word counts).
//!   Features must be non-negative.
//! - **[`BernoulliNB`]** — For binary/boolean features. Optional binarization
//!   threshold.
//! - **[`CategoricalNB`]** — For categorical features where each column takes
//!   on one of several discrete values. Laplace-smoothed.
//! - **[`ComplementNB`]** — A Multinomial NB variant that uses complement-class
//!   statistics; better suited for imbalanced datasets.
//!
//! # Design
//!
//! Each classifier follows the compile-time safety pattern:
//!
//! - The unfitted struct (e.g., `GaussianNB<F>`) holds hyperparameters and
//!   implements [`Fit`](ferrolearn_core::Fit).
//! - Calling `fit()` produces a fitted type (e.g., `FittedGaussianNB<F>`) that
//!   implements [`Predict`](ferrolearn_core::Predict) and
//!   [`HasClasses`](ferrolearn_core::introspection::HasClasses).
//! - The fitted types also expose a `predict_proba` method returning class
//!   probabilities as `Array2<F>`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::GaussianNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 2.0, 1.5, 2.5, 1.2, 1.8, 6.0, 7.0, 5.8, 6.5, 6.2, 7.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = GaussianNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

pub mod bernoulli;
pub mod categorical;
pub mod complement;
pub mod conjugate;
pub mod gaussian;
pub mod multinomial;

// Re-export all public types at the crate root.
pub use bernoulli::{BernoulliNB, FittedBernoulliNB};
pub use categorical::{CategoricalNB, FittedCategoricalNB};
pub use complement::{ComplementNB, FittedComplementNB};
pub use gaussian::{FittedGaussianNB, GaussianNB};
pub use multinomial::{FittedMultinomialNB, MultinomialNB};

use ndarray::Array2;
use num_traits::Float;

/// Smoothing-floor used when `force_alpha = false` to mirror scikit-learn's
/// legacy "raise alpha to 1e-10" behavior. When `force_alpha = true`
/// (the default), the user-supplied alpha is returned as-is, even if zero.
pub(crate) fn clamp_alpha<F: Float>(alpha: F, force_alpha: bool) -> F {
    if force_alpha {
        alpha
    } else {
        let floor = F::from(1e-10).unwrap();
        if alpha < floor { floor } else { alpha }
    }
}

/// Numerically stable row-wise log-softmax: returns `jll - logsumexp(jll, axis=1)`.
///
/// Used by every Fitted*NB to convert joint log-likelihoods into log
/// probabilities. The subtraction-of-row-max trick keeps the exponentials
/// bounded by 1, avoiding overflow.
pub(crate) fn log_softmax_rows<F: Float>(jll: &Array2<F>) -> Array2<F> {
    let n_samples = jll.nrows();
    let n_classes = jll.ncols();
    let mut log_proba = Array2::<F>::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        let max_score = jll
            .row(i)
            .iter()
            .fold(F::neg_infinity(), |a, &b| a.max(b));
        let mut sum_exp = F::zero();
        for ci in 0..n_classes {
            sum_exp = sum_exp + (jll[[i, ci]] - max_score).exp();
        }
        let log_norm = max_score + sum_exp.ln();
        for ci in 0..n_classes {
            log_proba[[i, ci]] = jll[[i, ci]] - log_norm;
        }
    }
    log_proba
}
