//! # ferrolearn-bayes
//!
//! Naive Bayes classifiers for the ferrolearn machine learning framework.
//!
//! This crate provides four Naive Bayes variants:
//!
//! - **[`GaussianNB`]** — Assumes Gaussian-distributed features. Suitable for
//!   continuous data.
//! - **[`MultinomialNB`]** — For discrete count data (e.g., word counts).
//!   Features must be non-negative.
//! - **[`BernoulliNB`]** — For binary/boolean features. Optional binarization
//!   threshold.
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
pub mod complement;
pub mod gaussian;
pub mod multinomial;

// Re-export all public types at the crate root.
pub use bernoulli::{BernoulliNB, FittedBernoulliNB};
pub use complement::{ComplementNB, FittedComplementNB};
pub use gaussian::{FittedGaussianNB, GaussianNB};
pub use multinomial::{FittedMultinomialNB, MultinomialNB};
