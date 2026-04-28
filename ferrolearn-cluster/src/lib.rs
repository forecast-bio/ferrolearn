//! # ferrolearn-cluster
//!
//! Clustering algorithms for the ferrolearn machine learning framework.
//!
//! This crate provides unsupervised clustering methods:
//!
//! - **[`KMeans`]** — K-Means clustering with k-Means++ initialization
//!   and parallelized assignment via Rayon (REQ-6).
//! - **[`DBSCAN`]** — Density-Based Spatial Clustering of Applications
//!   with Noise (REQ-7).
//! - **[`GaussianMixture`]** — Gaussian Mixture Models via the
//!   Expectation-Maximisation algorithm, with four covariance types
//!   (full, tied, diagonal, spherical) (REQ-3).
//! - **[`AgglomerativeClustering`]** — Bottom-up hierarchical clustering
//!   with Ward, Complete, Average, and Single linkage (REQ-5).
//! - **[`MeanShift`]** — Non-parametric mode-seeking clustering (REQ-23).
//! - **[`SpectralClustering`]** — Graph Laplacian eigenmap clustering (REQ-23).
//! - **[`OPTICS`]** — Ordering Points To Identify the Clustering Structure (REQ-23).
//! - **[`Hdbscan`]** — Hierarchical DBSCAN with automatic cluster detection.
//! - **[`Birch`]** — Balanced Iterative Reducing and Clustering using Hierarchies.
//! - **[`LabelPropagation`]** — Semi-supervised label propagation through a similarity graph.
//! - **[`LabelSpreading`]** — Semi-supervised label spreading via normalized graph Laplacian.
//! - **[`AffinityPropagation`]** — Exemplar-based clustering via message passing,
//!   automatically determines number of clusters.
//! - **[`BisectingKMeans`]** — Divisive hierarchical clustering that recursively
//!   bisects the largest cluster.
//! - **[`BayesianGaussianMixture`]** — Variational Bayesian GMM with
//!   automatic component pruning via Dirichlet Process or Dirichlet
//!   Distribution priors.
//! - **[`FeatureAgglomeration`]** — Hierarchical clustering of features
//!   (columns) with pooling-based dimensionality reduction.
//!
//! # Design
//!
//! All algorithms follow the compile-time safety pattern:
//!
//! - The unfitted config struct implements [`Fit`](ferrolearn_core::Fit)
//!   with `Y = ()` (unsupervised).
//! - [`KMeans`] produces [`FittedKMeans`], which implements
//!   [`Predict`](ferrolearn_core::Predict) (assign to nearest centroid)
//!   and [`Transform`](ferrolearn_core::Transform) (distance to each centroid).
//! - [`DBSCAN`] produces [`FittedDBSCAN`], which only stores labels and
//!   core sample indices — it does **not** implement `Predict`.
//! - [`GaussianMixture`] produces [`FittedGaussianMixture`], which
//!   implements [`Predict`](ferrolearn_core::Predict) (hard assignment) and
//!   [`Transform`](ferrolearn_core::Transform) (soft responsibilities).
//! - [`AgglomerativeClustering`] produces [`FittedAgglomerativeClustering`],
//!   which stores labels and the merge tree — it does **not** implement
//!   `Predict`.
//! - [`MeanShift`] produces [`FittedMeanShift`], which implements
//!   [`Predict`](ferrolearn_core::Predict) (assign to nearest center).
//! - [`SpectralClustering`] produces [`FittedSpectralClustering`], which stores
//!   labels — it does **not** implement `Predict`.
//! - [`OPTICS`] produces [`FittedOPTICS`], which stores the reachability
//!   ordering and distances — it does **not** implement `Predict`.
//! - [`Hdbscan`] produces [`FittedHdbscan`], which stores labels and
//!   probabilities — it does **not** implement `Predict`.
//! - [`Birch`] produces [`FittedBirch`], which stores labels and subcluster
//!   centers — it does **not** implement `Predict`.
//! - [`LabelPropagation`] produces [`FittedLabelPropagation`], which implements
//!   [`Predict`](ferrolearn_core::Predict) for new data via nearest-neighbor lookup.
//! - [`LabelSpreading`] produces [`FittedLabelSpreading`], which implements
//!   [`Predict`](ferrolearn_core::Predict) for new data via nearest-neighbor lookup.
//! - [`AffinityPropagation`] produces [`FittedAffinityPropagation`], which stores
//!   exemplar indices and labels — it does **not** implement `Predict`.
//! - [`BisectingKMeans`] produces [`FittedBisectingKMeans`], which implements
//!   [`Predict`](ferrolearn_core::Predict) (assign to nearest center).
//! - [`BayesianGaussianMixture`] produces [`FittedBayesianGaussianMixture`],
//!   which implements [`Predict`](ferrolearn_core::Predict) (hard assignment).
//! - [`FeatureAgglomeration`] produces [`FittedFeatureAgglomeration`], which
//!   implements [`Transform`](ferrolearn_core::Transform) (pool features).
//!
//! # Float Generics
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod affinity_propagation;
pub mod agglomerative;
pub mod bayesian_gmm;
pub mod birch;
pub mod bisecting_kmeans;
pub mod dbscan;
pub mod feature_agglomeration;
pub mod gmm;
pub mod hdbscan;
pub mod kmeans;
pub mod label_propagation;
pub mod label_spreading;
pub mod mean_shift;
pub mod mini_batch_kmeans;
pub mod optics;
pub mod spectral;

// Re-export the main types at the crate root.
pub use affinity_propagation::{AffinityPropagation, FittedAffinityPropagation};
pub use agglomerative::{AgglomerativeClustering, FittedAgglomerativeClustering, Linkage};
pub use bayesian_gmm::{
    BayesianCovType, BayesianGaussianMixture, FittedBayesianGaussianMixture, WeightPriorType,
};
pub use birch::{Birch, FittedBirch};
pub use bisecting_kmeans::{BisectingKMeans, BisectingStrategy, FittedBisectingKMeans};
pub use dbscan::{DBSCAN, FittedDBSCAN};
pub use feature_agglomeration::{
    AgglomerativeLinkage, FeatureAgglomeration, FittedFeatureAgglomeration, PoolingFunc,
};
pub use gmm::{CovarianceType, FittedGaussianMixture, GaussianMixture};
pub use hdbscan::{FittedHdbscan, Hdbscan};
pub use kmeans::{FittedKMeans, KMeans};
pub use label_propagation::{FittedLabelPropagation, LabelPropagation, LabelPropagationKernel};
pub use label_spreading::{FittedLabelSpreading, LabelSpreading, LabelSpreadingKernel};
pub use mean_shift::{FittedMeanShift, MeanShift};
pub use mini_batch_kmeans::{FittedMiniBatchKMeans, MiniBatchKMeans, MiniBatchKMeansInit};
pub use optics::{FittedOPTICS, OPTICS};
pub use spectral::{FittedSpectralClustering, SpectralClustering};
