//! # ferrolearn-datasets
//!
//! Built-in datasets and synthetic data generators for the ferrolearn machine
//! learning framework.
//!
//! This crate provides:
//!
//! - **[`toy`]** — classic datasets embedded at compile time:
//!   [`load_iris`], [`load_wine`], [`load_breast_cancer`], [`load_diabetes`],
//!   [`load_digits`], [`load_linnerud`], [`load_olivetti_faces`].
//! - **[`generators`]** — synthetic dataset generators:
//!   [`make_classification`], [`make_regression`], [`make_blobs`],
//!   [`make_moons`], [`make_circles`], [`make_swiss_roll`], [`make_s_curve`].
//!
//! All functions are generic over `F: num_traits::Float` and return
//! `Result<T, ferrolearn_core::FerroError>`.

pub mod generators;
pub mod svmlight;
pub mod toy;

// Re-export svmlight + load_files at the crate root.
pub use svmlight::{
    dump_svmlight_file, load_files, load_svmlight_file, load_svmlight_files, load_svmlight_str,
};

// Re-export toy loaders at the crate root.
pub use toy::{
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_linnerud, load_olivetti_faces,
    load_wine,
};

// Re-export synthetic generators at the crate root.
pub use generators::{
    make_blobs, make_circles, make_classification, make_friedman1, make_friedman2, make_friedman3,
    make_gaussian_quantiles, make_hastie_10_2, make_low_rank_matrix, make_moons,
    make_multilabel_classification, make_regression, make_s_curve, make_sparse_spd_matrix,
    make_sparse_uncorrelated, make_spd_matrix, make_swiss_roll,
};
