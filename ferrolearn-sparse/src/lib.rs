//! # ferrolearn-sparse
//!
//! Sparse matrix types for the ferrolearn machine learning framework.
//!
//! This crate provides three sparse matrix formats:
//!
//! - [`CsrMatrix<T>`] — Compressed Sparse Row, backed by [`sprs::CsMat<T>`] in CSR storage.
//!   Efficient for row slicing and matrix-vector products. Implements the
//!   [`ferrolearn_core::Dataset`] trait when `T: Float + Send + Sync + 'static`.
//! - [`CscMatrix<T>`] — Compressed Sparse Column, backed by [`sprs::CsMat<T>`] in CSC storage.
//!   Efficient for column slicing and transpose products.
//! - [`CooMatrix<T>`] — Coordinate (triplet) format, backed by [`sprs::TriMat<T>`].
//!   Convenient for incremental construction before converting to CSR/CSC.
//!
//! All three types support conversion between formats, conversion to/from dense
//! [`ndarray::Array2<T>`], slicing, scalar multiplication, element-wise addition,
//! and matrix-vector multiplication.
//!
//! # Quick Start
//!
//! ```
//! use ferrolearn_sparse::{CooMatrix, CsrMatrix};
//!
//! // Build in COO format, then convert.
//! let mut coo = CooMatrix::new(3, 3);
//! coo.push(0, 0, 1.0_f64);
//! coo.push(1, 2, 4.0);
//! coo.push(2, 1, 7.0);
//!
//! let csr = CsrMatrix::from_coo(&coo).unwrap();
//! let dense = csr.to_dense();
//! assert_eq!(dense[[0, 0]], 1.0);
//! assert_eq!(dense[[1, 2]], 4.0);
//! assert_eq!(dense[[2, 1]], 7.0);
//! ```

pub mod coo;
pub mod csc;
pub mod csr;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
