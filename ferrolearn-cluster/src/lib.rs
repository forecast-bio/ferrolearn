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
//!
//! # Design
//!
//! Both algorithms follow the compile-time safety pattern:
//!
//! - The unfitted config struct implements [`Fit`](ferrolearn_core::Fit)
//!   with `Y = ()` (unsupervised).
//! - [`KMeans`] produces [`FittedKMeans`], which implements
//!   [`Predict`](ferrolearn_core::Predict) (assign to nearest centroid)
//!   and [`Transform`](ferrolearn_core::Transform) (distance to each centroid).
//! - [`DBSCAN`] produces [`FittedDBSCAN`], which only stores labels and
//!   core sample indices — it does **not** implement `Predict`.
//!
//! # Float Generics
//!
//! All algorithms are generic over `F: num_traits::Float + Send + Sync + 'static`,
//! supporting both `f32` and `f64`.

pub mod dbscan;
pub mod kmeans;

// Re-export the main types at the crate root.
pub use dbscan::{DBSCAN, FittedDBSCAN};
pub use kmeans::{FittedKMeans, KMeans};
