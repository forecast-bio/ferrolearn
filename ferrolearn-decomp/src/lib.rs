//! # ferrolearn-decomp
//!
//! Dimensionality reduction and matrix decomposition for the ferrolearn
//! machine learning framework.
//!
//! This crate provides PCA, TruncatedSVD, NMF, Kernel PCA, and other
//! decomposition methods that follow the ferrolearn `Fit`/`Transform`
//! trait pattern.
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

pub mod kernel_pca;
pub mod nmf;
pub mod pca;
pub mod truncated_svd;

// Re-exports
pub use kernel_pca::{FittedKernelPCA, Kernel, KernelPCA};
pub use nmf::{FittedNMF, NMF, NMFInit, NMFSolver};
pub use pca::{FittedPCA, PCA};
pub use truncated_svd::{FittedTruncatedSVD, TruncatedSVD};
