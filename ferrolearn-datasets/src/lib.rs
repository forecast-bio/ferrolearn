//! # ferrolearn-datasets
//!
//! Built-in datasets and synthetic data generators for the ferrolearn machine
//! learning framework.
//!
//! This crate provides:
//!
//! - **[`toy`]** — classic datasets embedded at compile time:
//!   [`load_iris`], [`load_wine`], [`load_breast_cancer`], [`load_diabetes`],
//!   [`load_digits`].
//! - **[`generators`]** — synthetic dataset generators:
//!   [`make_classification`], [`make_regression`], [`make_blobs`],
//!   [`make_moons`], [`make_circles`].
//!
//! All functions are generic over `F: num_traits::Float` and return
//! `Result<T, ferrolearn_core::FerroError>`.

pub mod generators;
pub mod toy;

// Re-export toy loaders at the crate root.
pub use toy::{load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine};

// Re-export synthetic generators at the crate root.
pub use generators::{make_blobs, make_circles, make_classification, make_moons, make_regression};
