//! Pluggable backend trait for linear algebra operations.
//!
//! This module defines the [`Backend`] trait, which abstracts core linear algebra
//! operations (matrix multiply, SVD, QR, Cholesky, eigendecomposition, etc.).
//! Algorithms that need these operations can be generic over `Backend`, allowing
//! the implementation to be swapped at compile time (e.g., pure-Rust `faer` vs
//! system BLAS/LAPACK).
//!
//! The default backend is [`NdarrayFaerBackend`](crate::backend_faer::NdarrayFaerBackend),
//! which delegates to the `faer` crate for high-performance decompositions and
//! uses `ndarray::dot` for general matrix multiply.
//!
//! # Design
//!
//! All methods on `Backend` are associated functions (no `&self`). The backend
//! is a zero-sized type used as a type parameter, not as an instance:
//!
//! ```ignore
//! fn my_algorithm<B: Backend>(data: &Array2<f64>) -> FerroResult<Array1<f64>> {
//!     let (u, s, vt) = B::svd(data)?;
//!     // ...
//! }
//! ```

use crate::error::FerroResult;
use ndarray::{Array1, Array2};

/// Trait abstracting core linear algebra operations.
///
/// Algorithms that need matrix operations (SVD, QR, eigendecomposition, etc.)
/// can be generic over this trait, allowing the backend to be swapped
/// (e.g., pure-Rust faer vs system BLAS/LAPACK).
///
/// All methods are associated functions operating on `ndarray` arrays.
/// The implementing type is typically a zero-sized struct used solely as
/// a type parameter.
pub trait Backend: Send + Sync + 'static {
    /// General matrix multiply: `C = A * B`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the inner dimensions of `A`
    /// and `B` do not match (i.e., `A.ncols() != B.nrows()`).
    fn gemm(a: &Array2<f64>, b: &Array2<f64>) -> FerroResult<Array2<f64>>;

    /// Singular Value Decomposition: `A = U * diag(S) * Vt`.
    ///
    /// Returns `(U, S, Vt)` where:
    /// - `U` is an `(m, m)` orthogonal matrix,
    /// - `S` is a vector of `min(m, n)` non-negative singular values in
    ///   non-increasing order,
    /// - `Vt` is an `(n, n)` orthogonal matrix (the transpose of `V`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::NumericalInstability`] if the SVD fails to converge.
    fn svd(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array1<f64>, Array2<f64>)>;

    /// QR decomposition: `A = Q * R`.
    ///
    /// Returns `(Q, R)` where:
    /// - `Q` is an `(m, m)` orthogonal matrix,
    /// - `R` is an `(m, n)` upper trapezoidal matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::NumericalInstability`] if the decomposition fails.
    fn qr(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array2<f64>)>;

    /// Cholesky decomposition: `A = L * L^T` (lower triangular).
    ///
    /// The input matrix must be symmetric and positive definite.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::NumericalInstability`] if the matrix is not
    /// positive definite.
    fn cholesky(a: &Array2<f64>) -> FerroResult<Array2<f64>>;

    /// Solve linear system: `A * x = b`.
    ///
    /// Uses LU decomposition with partial pivoting for general square systems.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `A` is not square or if
    /// `b.len() != A.nrows()`.
    /// Returns [`FerroError::NumericalInstability`] if `A` is singular.
    fn solve(a: &Array2<f64>, b: &Array1<f64>) -> FerroResult<Array1<f64>>;

    /// Symmetric eigendecomposition: `A = V * diag(eigenvalues) * V^T`.
    ///
    /// The input matrix must be symmetric. Returns `(eigenvalues, V)` where:
    /// - `eigenvalues` is a vector of eigenvalues in non-decreasing order,
    /// - `V` is an `(n, n)` orthogonal matrix whose columns are the
    ///   corresponding eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `A` is not square.
    /// Returns [`FerroError::NumericalInstability`] if the decomposition fails
    /// to converge.
    fn eigh(a: &Array2<f64>) -> FerroResult<(Array1<f64>, Array2<f64>)>;

    /// Matrix determinant.
    ///
    /// Computed via LU decomposition.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `A` is not square.
    fn det(a: &Array2<f64>) -> FerroResult<f64>;

    /// Matrix inverse.
    ///
    /// Computed via LU decomposition.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `A` is not square.
    /// Returns [`FerroError::NumericalInstability`] if `A` is singular.
    fn inv(a: &Array2<f64>) -> FerroResult<Array2<f64>>;
}
