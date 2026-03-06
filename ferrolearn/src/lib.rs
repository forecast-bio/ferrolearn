//! # ferrolearn
//!
//! A scikit-learn equivalent for Rust.
//!
//! This is the top-level re-export crate that provides a unified API
//! for all ferrolearn functionality. Individual sub-crates can also
//! be used directly for finer-grained dependency control.

#[doc(inline)]
pub use ferrolearn_bayes as bayes;
#[doc(inline)]
pub use ferrolearn_cluster as cluster;
#[doc(inline)]
pub use ferrolearn_core as core;
#[doc(inline)]
pub use ferrolearn_datasets as datasets;
#[doc(inline)]
pub use ferrolearn_decomp as decomp;
#[doc(inline)]
pub use ferrolearn_io as io;
#[doc(inline)]
pub use ferrolearn_linear as linear;
#[doc(inline)]
pub use ferrolearn_metrics as metrics;
#[doc(inline)]
pub use ferrolearn_model_sel as model_selection;
#[doc(inline)]
pub use ferrolearn_neighbors as neighbors;
#[doc(inline)]
pub use ferrolearn_preprocess as preprocess;
#[doc(inline)]
pub use ferrolearn_sparse as sparse;
#[doc(inline)]
pub use ferrolearn_tree as tree;

// Also re-export the most common items at the top level.
#[doc(inline)]
pub use ferrolearn_core::pipeline::Pipeline;
#[doc(inline)]
pub use ferrolearn_core::{
    Backend, Dataset, DefaultBackend, FerroError, FerroResult, Fit, FitTransform, PartialFit,
    Predict, Transform,
};

/// Convenience prelude that re-exports the most commonly used traits and types.
///
/// ```rust
/// use ferrolearn::prelude::*;
/// ```
pub mod prelude {
    pub use ferrolearn_core::introspection::{HasClasses, HasCoefficients, HasFeatureImportances};
    pub use ferrolearn_core::pipeline::Pipeline;
    pub use ferrolearn_core::streaming::StreamingFitter;
    pub use ferrolearn_core::{
        Backend, Dataset, DefaultBackend, FerroError, FerroResult, Fit, FitTransform, PartialFit,
        Predict, Transform,
    };
}
