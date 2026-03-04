//! # ferrolearn
//!
//! A scikit-learn equivalent for Rust.
//!
//! This is the top-level re-export crate that provides a unified API
//! for all ferrolearn functionality. Individual sub-crates can also
//! be used directly for finer-grained dependency control.

pub use ferrolearn_bayes as bayes;
pub use ferrolearn_cluster as cluster;
pub use ferrolearn_core as core;
pub use ferrolearn_datasets as datasets;
pub use ferrolearn_decomp as decomp;
pub use ferrolearn_io as io;
pub use ferrolearn_linear as linear;
pub use ferrolearn_metrics as metrics;
pub use ferrolearn_model_sel as model_selection;
pub use ferrolearn_neighbors as neighbors;
pub use ferrolearn_preprocess as preprocess;
pub use ferrolearn_sparse as sparse;
pub use ferrolearn_tree as tree;

// Also re-export the most common items at the top level.
pub use ferrolearn_core::pipeline::Pipeline;
pub use ferrolearn_core::{
    Backend, Dataset, DefaultBackend, FerroError, FerroResult, Fit, FitTransform, PartialFit,
    Predict, Transform,
};
