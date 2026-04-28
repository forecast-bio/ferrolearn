//! # ferrolearn-decomp
//!
//! Dimensionality reduction and matrix decomposition for the ferrolearn
//! machine learning framework.
//!
//! This crate provides PCA, TruncatedSVD, NMF, Kernel PCA, and manifold
//! learning methods that follow the ferrolearn `Fit`/`Transform` trait
//! pattern.
//!
//! ## Algorithms
//!
//! - [`PCA`] — Principal Component Analysis. Centres data and projects onto
//!   the directions of maximum variance.
//! - [`TruncatedSVD`] — Truncated Singular Value Decomposition using the
//!   randomized algorithm. Does **not** centre data, making it suitable for
//!   sparse inputs.
//! - [`NMF`] — Non-negative Matrix Factorization. Decomposes a non-negative
//!   matrix `X` into `W * H` where both factors are non-negative.
//! - [`KernelPCA`] — Kernel PCA. Non-linear dimensionality reduction via
//!   a kernel-induced feature space.
//! - [`MDS`] — Classical Multidimensional Scaling. Embeds data preserving
//!   pairwise distances.
//! - [`Isomap`] — Isometric Mapping. Non-linear dimensionality reduction
//!   via geodesic distances on a kNN graph.
//! - [`SpectralEmbedding`] — Laplacian Eigenmaps. Non-linear dimensionality
//!   reduction via the normalised graph Laplacian.
//! - [`LLE`] — Locally Linear Embedding. Non-linear dimensionality reduction
//!   preserving local reconstruction weights.
//! - [`Tsne`] — t-distributed Stochastic Neighbor Embedding. Non-linear
//!   dimensionality reduction using Barnes-Hut approximation.
//! - [`Umap`] — Uniform Manifold Approximation and Projection. Fast non-linear
//!   dimensionality reduction based on topological data analysis.
//! - [`LatentDirichletAllocation`] — Latent Dirichlet Allocation topic model.
//!   Discovers latent topics in document-term matrices.
//! - [`DictionaryLearning`] — Sparse coding with a learned dictionary.
//!
//! ## Pipeline Integration
//!
//! `PCA<f64>`, `TruncatedSVD<f64>`, `NMF<f64>`, and `KernelPCA<f64>` all
//! implement
//! [`PipelineTransformer`](ferrolearn_core::pipeline::PipelineTransformer)
//! so they can be used as transformer steps in a
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::PCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let pca = PCA::<f64>::new(1);
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let fitted = pca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

pub mod covariance;
pub mod cross_decomposition;
pub mod dictionary_learning;
pub mod factor_analysis;
pub mod fast_ica;
pub mod incremental_pca;
pub mod isomap;
pub mod kernel_pca;
pub mod lda_topic;
pub mod lle;
pub mod mds;
pub mod nmf;
pub mod pca;
pub mod spectral_embedding;
pub mod truncated_svd;
pub mod tsne;
pub mod umap;

// Re-exports
pub use covariance::{
    EllipticEnvelope, EmpiricalCovariance, FittedCovariance, FittedEllipticEnvelope,
    FittedLedoitWolf, FittedMinCovDet, FittedOAS, LedoitWolf, MinCovDet, OAS, ShrunkCovariance,
};
pub use cross_decomposition::{
    CCA, FittedCCA, FittedPLSCanonical, FittedPLSRegression, FittedPLSSVD, PLSCanonical,
    PLSRegression, PLSSVD,
};
pub use dictionary_learning::{
    DictFitAlgorithm, DictTransformAlgorithm, DictionaryLearning, FittedDictionaryLearning,
};
pub use factor_analysis::{FactorAnalysis, FittedFactorAnalysis};
pub use fast_ica::{Algorithm, FastICA, FittedFastICA, NonLinearity};
pub use incremental_pca::{FittedIncrementalPCA, IncrementalPCA};
pub use isomap::{FittedIsomap, Isomap};
pub use kernel_pca::{FittedKernelPCA, Kernel, KernelPCA};
pub use lda_topic::{
    FittedLatentDirichletAllocation, LatentDirichletAllocation, LdaLearningMethod,
};
pub use lle::{FittedLLE, LLE};
pub use mds::{Dissimilarity, FittedMDS, MDS};
pub use nmf::{FittedNMF, NMF, NMFInit, NMFSolver};
pub use pca::{FittedPCA, PCA};
pub use spectral_embedding::{Affinity, FittedSpectralEmbedding, SpectralEmbedding};
pub use truncated_svd::{FittedTruncatedSVD, TruncatedSVD};
pub use tsne::{FittedTsne, Tsne};
pub use umap::{FittedUmap, Umap, UmapMetric};
