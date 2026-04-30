//! Convenience constructors for common sparse-matrix patterns.
//!
//! These mirror the most-used scipy.sparse helpers:
//!
//! - [`eye`] — sparse identity matrix.
//! - [`diags`] — sparse diagonal matrix.
//! - [`hstack`] — horizontal concatenation of CSR matrices.
//! - [`vstack`] — vertical concatenation of CSR matrices.

use ferrolearn_core::FerroError;
use num_traits::One;
use std::ops::Add;

use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;

/// Build an `n x n` sparse identity matrix.
pub fn eye<T>(n: usize) -> Result<CsrMatrix<T>, FerroError>
where
    T: Clone + One + Add<Output = T> + 'static,
{
    let mut coo = CooMatrix::<T>::with_capacity(n, n, n);
    for i in 0..n {
        coo.push(i, i, T::one())
            .map_err(|e| FerroError::InvalidParameter {
                name: "eye".into(),
                reason: format!("push failed at ({i}, {i}): {e}"),
            })?;
    }
    CsrMatrix::from_coo(&coo)
}

/// Build a sparse `n x n` matrix from a single diagonal vector at `offset`.
///
/// `offset == 0` puts `values` on the main diagonal; `offset > 0` shifts to
/// a super-diagonal; `offset < 0` shifts to a sub-diagonal.
pub fn diags<T>(values: &[T], offset: isize, n: usize) -> Result<CsrMatrix<T>, FerroError>
where
    T: Clone + Add<Output = T> + 'static,
{
    let mut coo = CooMatrix::<T>::with_capacity(n, n, values.len());
    for (k, v) in values.iter().enumerate() {
        let (i, j) = if offset >= 0 {
            (k, k + offset as usize)
        } else {
            (k + (-offset) as usize, k)
        };
        if i < n && j < n {
            coo.push(i, j, v.clone())
                .map_err(|e| FerroError::InvalidParameter {
                    name: "diags".into(),
                    reason: format!("push failed at ({i}, {j}): {e}"),
                })?;
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Horizontally concatenate CSR matrices.
///
/// All matrices must have the same number of rows.
pub fn hstack<T>(matrices: &[&CsrMatrix<T>]) -> Result<CsrMatrix<T>, FerroError>
where
    T: Clone + Add<Output = T> + 'static,
{
    if matrices.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "matrices".into(),
            reason: "hstack: at least one matrix required".into(),
        });
    }
    let n_rows = matrices[0].n_rows();
    for (idx, m) in matrices.iter().enumerate() {
        if m.n_rows() != n_rows {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_rows],
                actual: vec![m.n_rows()],
                context: format!("hstack: matrix {idx} has {} rows", m.n_rows()),
            });
        }
    }
    let total_cols: usize = matrices.iter().map(|m| m.n_cols()).sum();
    let mut coo = CooMatrix::<T>::new(n_rows, total_cols);
    let mut col_offset = 0usize;
    for m in matrices {
        for (val, (r, c)) in m.inner().iter() {
            coo.push(r, c + col_offset, val.clone())
                .map_err(|e| FerroError::InvalidParameter {
                    name: "hstack".into(),
                    reason: format!("push failed: {e}"),
                })?;
        }
        col_offset += m.n_cols();
    }
    CsrMatrix::from_coo(&coo)
}

/// Vertically concatenate CSR matrices.
///
/// All matrices must have the same number of columns.
pub fn vstack<T>(matrices: &[&CsrMatrix<T>]) -> Result<CsrMatrix<T>, FerroError>
where
    T: Clone + Add<Output = T> + 'static,
{
    if matrices.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "matrices".into(),
            reason: "vstack: at least one matrix required".into(),
        });
    }
    let n_cols = matrices[0].n_cols();
    for (idx, m) in matrices.iter().enumerate() {
        if m.n_cols() != n_cols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_cols],
                actual: vec![m.n_cols()],
                context: format!("vstack: matrix {idx} has {} cols", m.n_cols()),
            });
        }
    }
    let total_rows: usize = matrices.iter().map(|m| m.n_rows()).sum();
    let mut coo = CooMatrix::<T>::new(total_rows, n_cols);
    let mut row_offset = 0usize;
    for m in matrices {
        for (val, (r, c)) in m.inner().iter() {
            coo.push(r + row_offset, c, val.clone())
                .map_err(|e| FerroError::InvalidParameter {
                    name: "vstack".into(),
                    reason: format!("push failed: {e}"),
                })?;
        }
        row_offset += m.n_rows();
    }
    CsrMatrix::from_coo(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_basic() {
        let m: CsrMatrix<f64> = eye(3).unwrap();
        let dense = m.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                assert!((dense[[i, j]] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_diags_main_diagonal() {
        let m: CsrMatrix<f64> = diags(&[1.0, 2.0, 3.0], 0, 3).unwrap();
        let d = m.to_dense();
        assert!((d[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((d[[1, 1]] - 2.0).abs() < 1e-12);
        assert!((d[[2, 2]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_diags_super_diagonal() {
        let m: CsrMatrix<f64> = diags(&[1.0, 2.0], 1, 3).unwrap();
        let d = m.to_dense();
        assert!((d[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((d[[1, 2]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_hstack_basic() {
        let a: CsrMatrix<f64> = eye(2).unwrap();
        let b: CsrMatrix<f64> = diags(&[5.0, 5.0], 0, 2).unwrap();
        let h = hstack(&[&a, &b]).unwrap();
        assert_eq!(h.n_rows(), 2);
        assert_eq!(h.n_cols(), 4);
        let d = h.to_dense();
        assert!((d[[0, 2]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_vstack_basic() {
        let a: CsrMatrix<f64> = eye(2).unwrap();
        let b: CsrMatrix<f64> = diags(&[5.0, 5.0], 0, 2).unwrap();
        let v = vstack(&[&a, &b]).unwrap();
        assert_eq!(v.n_rows(), 4);
        assert_eq!(v.n_cols(), 2);
        let d = v.to_dense();
        assert!((d[[2, 0]] - 5.0).abs() < 1e-12);
    }
}
