//! Default backend implementation using `faer` for linear algebra.
//!
//! [`NdarrayFaerBackend`] implements the [`Backend`](crate::backend::Backend)
//! trait by converting between `ndarray::Array2<f64>` and `faer::Mat<f64>`,
//! then delegating to `faer`'s high-performance decomposition routines.
//!
//! - **gemm**: uses `ndarray`'s `dot` (which may use optimized BLAS internally).
//! - **svd**: delegates to `faer::linalg::solvers::Svd`.
//! - **qr**: delegates to `faer::linalg::solvers::Qr`.
//! - **cholesky**: delegates to `faer::linalg::solvers::Llt`.
//! - **solve**: uses LU decomposition via `faer::linalg::solvers::PartialPivLu`.
//! - **eigh**: delegates to `faer::linalg::solvers::SelfAdjointEigen`.
//! - **det**: computed via `faer::MatRef::determinant`.
//! - **inv**: computed via LU decomposition with `DenseSolveCore::inverse`.

use crate::backend::Backend;
use crate::error::{FerroError, FerroResult};
use ndarray::{Array1, Array2};

/// Convert an `ndarray::Array2<f64>` to a `faer::Mat<f64>`.
fn ndarray_to_faer(a: &Array2<f64>) -> faer::Mat<f64> {
    let (nrows, ncols) = a.dim();
    faer::Mat::from_fn(nrows, ncols, |i, j| a[[i, j]])
}

/// Convert a `faer::Mat<f64>` to an `ndarray::Array2<f64>`.
fn faer_to_ndarray(m: &faer::Mat<f64>) -> Array2<f64> {
    let (nrows, ncols) = m.shape();
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

/// Convert a `faer::MatRef<'_, f64>` to an `ndarray::Array2<f64>`.
fn faer_ref_to_ndarray(m: faer::MatRef<'_, f64>) -> Array2<f64> {
    let (nrows, ncols) = m.shape();
    Array2::from_shape_fn((nrows, ncols), |(i, j)| m[(i, j)])
}

/// Convert a `faer::DiagRef<'_, f64>` to an `ndarray::Array1<f64>`.
fn faer_diag_to_ndarray(d: faer::diag::DiagRef<'_, f64>) -> Array1<f64> {
    let vals: Vec<f64> = d.column_vector().iter().copied().collect();
    Array1::from_vec(vals)
}

/// The default backend using the `faer` crate for linear algebra operations.
///
/// This is a zero-sized type intended for use as a type parameter on
/// algorithms that are generic over [`Backend`]:
///
/// ```ignore
/// fn my_algorithm<B: Backend>(data: &Array2<f64>) -> FerroResult<Array1<f64>> {
///     let (u, s, vt) = B::svd(data)?;
///     // ...
/// }
///
/// // Use the default backend:
/// my_algorithm::<NdarrayFaerBackend>(&data)?;
/// ```
pub struct NdarrayFaerBackend;

impl Backend for NdarrayFaerBackend {
    fn gemm(a: &Array2<f64>, b: &Array2<f64>) -> FerroResult<Array2<f64>> {
        if a.ncols() != b.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![a.nrows(), a.ncols()],
                actual: vec![b.nrows(), b.ncols()],
                context: format!(
                    "gemm: A is {}x{} but B is {}x{} (inner dimensions {} != {})",
                    a.nrows(),
                    a.ncols(),
                    b.nrows(),
                    b.ncols(),
                    a.ncols(),
                    b.nrows()
                ),
            });
        }
        Ok(a.dot(b))
    }

    fn svd(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let mat = ndarray_to_faer(a);
        let decomp = mat.svd().map_err(|e| FerroError::NumericalInstability {
            message: format!("SVD failed to converge: {e:?}"),
        })?;

        let u = faer_ref_to_ndarray(decomp.U());
        let s = faer_diag_to_ndarray(decomp.S());
        // faer returns V (not V^T), so we transpose it
        let vt = faer_ref_to_ndarray(decomp.V().transpose());

        Ok((u, s, vt))
    }

    fn qr(a: &Array2<f64>) -> FerroResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = a.dim();
        let mat = ndarray_to_faer(a);
        let decomp = mat.qr();

        let q = faer_to_ndarray(&decomp.compute_Q());

        // faer stores R as min(m,n) x n. We need the full (m x n) upper
        // trapezoidal R with zero rows below.
        let r_compact = decomp.R();
        let r_rows = r_compact.nrows();
        let mut r = Array2::<f64>::zeros((m, n));
        for i in 0..r_rows {
            for j in 0..n {
                r[[i, j]] = r_compact[(i, j)];
            }
        }

        Ok((q, r))
    }

    fn cholesky(a: &Array2<f64>) -> FerroResult<Array2<f64>> {
        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "cholesky: matrix must be square".into(),
            });
        }

        let mat = ndarray_to_faer(a);
        let decomp = mat
            .llt(faer::Side::Lower)
            .map_err(|e| FerroError::NumericalInstability {
                message: format!(
                    "Cholesky decomposition failed (matrix not positive definite): {e:?}"
                ),
            })?;

        Ok(faer_ref_to_ndarray(decomp.L()))
    }

    fn solve(a: &Array2<f64>, b: &Array1<f64>) -> FerroResult<Array1<f64>> {
        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "solve: coefficient matrix must be square".into(),
            });
        }
        if b.len() != nrows {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows],
                actual: vec![b.len()],
                context: format!("solve: b has length {} but A has {} rows", b.len(), nrows),
            });
        }

        use faer::linalg::solvers::Solve;

        let mat = ndarray_to_faer(a);
        let rhs = faer::Mat::from_fn(nrows, 1, |i, _| b[i]);
        let lu = mat.partial_piv_lu();
        let result = lu.solve(rhs.as_ref());

        Ok(Array1::from_shape_fn(nrows, |i| result[(i, 0)]))
    }

    fn eigh(a: &Array2<f64>) -> FerroResult<(Array1<f64>, Array2<f64>)> {
        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "eigh: matrix must be square".into(),
            });
        }

        let mat = ndarray_to_faer(a);
        let decomp = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
            FerroError::NumericalInstability {
                message: format!("Symmetric eigendecomposition failed to converge: {e:?}"),
            }
        })?;

        let eigenvalues = faer_diag_to_ndarray(decomp.S());
        let eigenvectors = faer_ref_to_ndarray(decomp.U());

        Ok((eigenvalues, eigenvectors))
    }

    fn det(a: &Array2<f64>) -> FerroResult<f64> {
        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "det: matrix must be square".into(),
            });
        }

        let mat = ndarray_to_faer(a);
        Ok(mat.as_ref().determinant())
    }

    fn inv(a: &Array2<f64>) -> FerroResult<Array2<f64>> {
        let (nrows, ncols) = a.dim();
        if nrows != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, nrows],
                actual: vec![nrows, ncols],
                context: "inv: matrix must be square".into(),
            });
        }

        use faer::linalg::solvers::DenseSolveCore;

        let mat = ndarray_to_faer(a);
        let lu = mat.partial_piv_lu();
        let inv_mat = lu.inverse();

        Ok(faer_to_ndarray(&inv_mat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Helper: shorthand alias for the backend
    // -----------------------------------------------------------------------
    type B = NdarrayFaerBackend;

    /// Assert that two 2D arrays are element-wise approximately equal.
    fn assert_mat_eq(actual: &Array2<f64>, expected: &Array2<f64>, eps: f64) {
        assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
        for ((i, j), &val) in actual.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = eps,);
        }
    }

    /// Assert that two 1D arrays are element-wise approximately equal.
    fn assert_vec_eq(actual: &Array1<f64>, expected: &Array1<f64>, eps: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, &val) in actual.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = eps);
        }
    }

    // -----------------------------------------------------------------------
    // gemm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gemm_identity() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let c = B::gemm(&a, &eye).unwrap();
        assert_mat_eq(&c, &a, 1e-12);
    }

    #[test]
    fn test_gemm_known_result() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = B::gemm(&a, &b).unwrap();
        let expected = array![[19.0, 22.0], [43.0, 50.0]];
        assert_mat_eq(&c, &expected, 1e-12);
    }

    #[test]
    fn test_gemm_rectangular() {
        // (2x3) * (3x2) = (2x2)
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = B::gemm(&a, &b).unwrap();
        let expected = array![[58.0, 64.0], [139.0, 154.0]];
        assert_mat_eq(&c, &expected, 1e-12);
    }

    #[test]
    fn test_gemm_shape_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = B::gemm(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // svd tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_svd_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (u, s, vt) = B::svd(&eye).unwrap();
        // Singular values of identity are [1, 1]
        for &val in s.iter() {
            assert_relative_eq!(val, 1.0, epsilon = 1e-12);
        }
        // U * diag(S) * Vt should reconstruct the original
        let reconstructed = reconstruct_svd(&u, &s, &vt);
        assert_mat_eq(&reconstructed, &eye, 1e-12);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (u, s, vt) = B::svd(&a).unwrap();
        let reconstructed = reconstruct_svd(&u, &s, &vt);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_svd_singular_values_descending() {
        let a = array![[3.0, 1.0], [1.0, 3.0]];
        let (_, s, _) = B::svd(&a).unwrap();
        assert!(s[0] >= s[1], "singular values should be non-increasing");
    }

    // -----------------------------------------------------------------------
    // qr tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_qr_reconstruction() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (q, r) = B::qr(&a).unwrap();
        let reconstructed = q.dot(&r);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_qr_orthogonality() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let (q, _) = B::qr(&a).unwrap();
        // Q^T * Q should be identity
        let qtq = q.t().dot(&q);
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        assert_mat_eq(&qtq, &eye, 1e-10);
    }

    #[test]
    fn test_qr_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (q, r) = B::qr(&eye).unwrap();
        let reconstructed = q.dot(&r);
        assert_mat_eq(&reconstructed, &eye, 1e-12);
    }

    // -----------------------------------------------------------------------
    // cholesky tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cholesky_known() {
        // A = [[4, 2], [2, 3]], L = [[2, 0], [1, sqrt(2)]]
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let l = B::cholesky(&a).unwrap();
        let reconstructed = l.dot(&l.t());
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let l = B::cholesky(&eye).unwrap();
        assert_mat_eq(&l, &eye, 1e-12);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite
        let a = array![[-1.0, 0.0], [0.0, -1.0]];
        let result = B::cholesky(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // solve tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_simple() {
        // [[2, 1], [1, 3]] * x = [5, 7] => x = [8/5, 9/5] = [1.6, 1.8]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 7.0];
        let x = B::solve(&a, &b).unwrap();
        assert_relative_eq!(x[0], 1.6, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.8, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![3.0, 7.0];
        let x = B::solve(&eye, &b).unwrap();
        assert_vec_eq(&x, &b, 1e-12);
    }

    #[test]
    fn test_solve_3x3() {
        // A = [[1,2,3],[0,1,4],[5,6,0]], b = [1,2,3]
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let b = array![1.0, 2.0, 3.0];
        let x = B::solve(&a, &b).unwrap();
        // Verify: A * x = b
        let ax = a.dot(&x);
        assert_vec_eq(&ax, &b, 1e-10);
    }

    #[test]
    fn test_solve_shape_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // Wrong size
        let result = B::solve(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // eigh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigh_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let (eigenvalues, eigenvectors) = B::eigh(&eye).unwrap();
        // All eigenvalues should be 1
        for &val in eigenvalues.iter() {
            assert_relative_eq!(val, 1.0, epsilon = 1e-12);
        }
        // V * V^T should be identity
        let vvt = eigenvectors.dot(&eigenvectors.t());
        assert_mat_eq(&vvt, &eye, 1e-12);
    }

    #[test]
    fn test_eigh_symmetric() {
        // Symmetric: [[2, 1], [1, 2]], eigenvalues = {1, 3}
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (eigenvalues, eigenvectors) = B::eigh(&a).unwrap();
        // faer returns eigenvalues sorted in non-decreasing order
        assert_relative_eq!(eigenvalues[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvalues[1], 3.0, epsilon = 1e-10);

        // Reconstruct: A = V * diag(eigenvalues) * V^T
        let reconstructed = reconstruct_eigh(&eigenvalues, &eigenvectors);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_eigh_eigenvalues_sorted() {
        let a = array![[5.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let (eigenvalues, _) = B::eigh(&a).unwrap();
        for i in 1..eigenvalues.len() {
            assert!(
                eigenvalues[i] >= eigenvalues[i - 1],
                "eigenvalues should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_eigh_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::eigh(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // det tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_det_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let d = B::det(&eye).unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_det_known() {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = B::det(&a).unwrap();
        assert_relative_eq!(d, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_det_singular() {
        // Singular matrix: det = 0
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let d = B::det(&a).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        // det([[1,2,3],[0,1,4],[5,6,0]]) = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24+40-15 = 1
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let d = B::det(&a).unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_det_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::det(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // inv tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inv_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let inv = B::inv(&eye).unwrap();
        assert_mat_eq(&inv, &eye, 1e-12);
    }

    #[test]
    fn test_inv_known() {
        // inv([[1,2],[3,4]]) = 1/(-2) * [[4,-2],[-3,1]] = [[-2,1],[1.5,-0.5]]
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let inv = B::inv(&a).unwrap();
        let expected = array![[-2.0, 1.0], [1.5, -0.5]];
        assert_mat_eq(&inv, &expected, 1e-10);
    }

    #[test]
    fn test_inv_roundtrip() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = B::inv(&a).unwrap();
        let product = a.dot(&inv);
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        assert_mat_eq(&product, &eye, 1e-10);
    }

    #[test]
    fn test_inv_3x3() {
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let inv = B::inv(&a).unwrap();
        let product = a.dot(&inv);
        let eye = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { 1.0 } else { 0.0 });
        assert_mat_eq(&product, &eye, 1e-10);
    }

    #[test]
    fn test_inv_not_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::inv(&a);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Cross-operation consistency tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_via_inv() {
        // solve(A, b) should give same result as inv(A) * b
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 7.0];
        let x_solve = B::solve(&a, &b).unwrap();
        let a_inv = B::inv(&a).unwrap();
        let x_inv = a_inv.dot(&b);
        assert_vec_eq(&x_solve, &x_inv, 1e-10);
    }

    #[test]
    fn test_det_via_eigh() {
        // For a symmetric matrix, det = product of eigenvalues
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let det_direct = B::det(&a).unwrap();
        let (eigenvalues, _) = B::eigh(&a).unwrap();
        let det_from_eig: f64 = eigenvalues.iter().product();
        assert_relative_eq!(det_direct, det_from_eig, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // Additional tests for completeness (20+ total)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gemm_single_element() {
        let a = array![[3.0]];
        let b = array![[4.0]];
        let c = B::gemm(&a, &b).unwrap();
        assert_relative_eq!(c[[0, 0]], 12.0, epsilon = 1e-12);
    }

    #[test]
    fn test_svd_diagonal() {
        let a = array![[3.0, 0.0], [0.0, 5.0]];
        let (_, s, _) = B::svd(&a).unwrap();
        // Singular values should be 5, 3 (descending)
        assert_relative_eq!(s[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(s[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_3x3() {
        // A = X^T * X guarantees positive definite
        let x = array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        let a = x.t().dot(&x);
        let l = B::cholesky(&a).unwrap();
        let reconstructed = l.dot(&l.t());
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_eigh_reconstruction_3x3() {
        let a = array![[5.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let (eigenvalues, eigenvectors) = B::eigh(&a).unwrap();
        let reconstructed = reconstruct_eigh(&eigenvalues, &eigenvectors);
        assert_mat_eq(&reconstructed, &a, 1e-10);
    }

    #[test]
    fn test_backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<NdarrayFaerBackend>();
    }

    #[test]
    fn test_cholesky_non_square_error() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = B::cholesky(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_non_square_error() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![1.0, 2.0];
        let result = B::solve(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Helpers for reconstruction
    // -----------------------------------------------------------------------

    /// Reconstruct a matrix from its SVD: A = U * diag(S) * Vt.
    fn reconstruct_svd(u: &Array2<f64>, s: &Array1<f64>, vt: &Array2<f64>) -> Array2<f64> {
        let m = u.nrows();
        let n = vt.ncols();
        let k = s.len();
        let mut result = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += u[[i, l]] * s[l] * vt[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }
        result
    }

    /// Reconstruct a matrix from symmetric eigendecomposition: A = V * diag(eigenvalues) * V^T.
    fn reconstruct_eigh(eigenvalues: &Array1<f64>, v: &Array2<f64>) -> Array2<f64> {
        let n = eigenvalues.len();
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += v[[i, k]] * eigenvalues[k] * v[[j, k]];
                }
                result[[i, j]] = sum;
            }
        }
        result
    }
}
