//! # ferrolearn-core
//!
//! Core traits, error types, dataset abstractions, and pipeline infrastructure
//! for the ferrolearn machine learning framework.
//!
//! This crate defines the foundational abstractions that all other ferrolearn
//! crates depend on:
//!
//! - **[`Fit`]**, **[`Predict`]**, **[`Transform`]**, **[`FitTransform`]** —
//!   the core ML traits with compile-time enforcement that `predict()` cannot
//!   be called on an unfitted model.
//! - **[`FerroError`]** — the unified error type with rich diagnostic context.
//! - **[`Dataset`]** — a trait for querying tabular data shape, with
//!   implementations for `ndarray::Array2<f32>` and `ndarray::Array2<f64>`.
//! - **[`pipeline::Pipeline`]** — a dynamic-dispatch pipeline that composes
//!   transformers and a final estimator.
//! - **Introspection traits** — [`HasCoefficients`], [`HasFeatureImportances`],
//!   [`HasClasses`] for inspecting fitted model internals.
//!
//! # Design Principles
//!
//! ## Compile-Time Safety (AC-3)
//!
//! The unfitted configuration struct (e.g., `LinearRegression`) implements
//! [`Fit`] but **not** [`Predict`]. Calling `fit()` returns a *new fitted
//! type* (e.g., `FittedLinearRegression`) that implements [`Predict`].
//! Attempting to call `predict()` on an unfitted model is a compile error.
//!
//! ## Float Generics (REQ-15)
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`.
//!
//! ## Error Handling
//!
//! All public functions return `Result<T, FerroError>`. Library code never panics.

pub mod dataset;
pub mod error;
pub mod introspection;
pub mod pipeline;
pub mod traits;
pub mod typed_pipeline;

// Re-export the most commonly used items at the crate root.
pub use dataset::Dataset;
pub use error::{FerroError, FerroResult};
pub use introspection::{HasClasses, HasCoefficients, HasFeatureImportances};
pub use traits::{Fit, FitTransform, Predict, Transform};
