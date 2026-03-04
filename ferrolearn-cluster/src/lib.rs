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
//!
//! # Float Generics
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod agglomerative;
pub mod dbscan;
pub mod gmm;
pub mod kmeans;

// Re-export the main types at the crate root.
pub use agglomerative::{AgglomerativeClustering, FittedAgglomerativeClustering, Linkage};
pub use dbscan::{DBSCAN, FittedDBSCAN};
pub use gmm::{CovarianceType, FittedGaussianMixture, GaussianMixture};
pub use kmeans::{FittedKMeans, KMeans};
