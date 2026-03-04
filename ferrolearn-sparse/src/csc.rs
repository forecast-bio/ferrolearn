//! Compressed Sparse Column (CSC) matrix format.
//!
//! [`CscMatrix<T>`] is a newtype wrapper around [`sprs::CsMat<T>`] in CSC
//! storage. CSC matrices are efficient for column-wise operations and are the
//! natural choice when algorithms need to iterate over columns.

use std::ops::{Add, AddAssign, Mul, MulAssign};

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Zero;
use sprs::CsMat;

use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;

/// Compressed Sparse Column (CSC) sparse matrix.
///
/// Stores non-zero entries in column-major order using three arrays: `indptr`
/// (column pointer array of length `n_cols + 1`), `indices` (row indices of
/// each non-zero), and `data` (values of each non-zero).
///
/// # Type Parameter
///
/// `T` — the scalar element type.
#[derive(Debug, Clone)]
pub struct CscMatrix<T> {
    inner: CsMat<T>,
}

impl<T> CscMatrix<T>
where
    T: Clone,
{
    /// Construct a CSC matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    /// * `indptr` — column pointer array of length `n_cols + 1`.
    /// * `indices` — row index of each non-zero entry.
    /// * `data` — value of each non-zero entry.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if the data is structurally
    /// invalid (wrong lengths, out-of-bound indices, unsorted inner indices).
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, FerroError> {
        CsMat::try_new_csc((n_rows, n_cols), indptr, indices, data)
            .map(|inner| Self { inner })
            .map_err(|(_, _, _, err)| FerroError::InvalidParameter {
                name: "CscMatrix raw components".into(),
                reason: err.to_string(),
            })
    }

    /// Build a [`CscMatrix`] from a pre-validated [`sprs::CsMat<T>`] in CSC storage.
    ///
    /// This is used internally for format conversions.
    pub(crate) fn from_inner(inner: CsMat<T>) -> Self {
        debug_assert!(inner.is_csc(), "inner matrix must be in CSC storage");
        Self { inner }
    }

    /// Returns the number of rows.
    pub fn n_rows(&self) -> usize {
        self.inner.rows()
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.inner.cols()
    }

    /// Returns the number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Returns a reference to the underlying [`sprs::CsMat<T>`].
    pub fn inner(&self) -> &CsMat<T> {
        &self.inner
    }

    /// Consume this matrix and return the underlying [`sprs::CsMat<T>`].
    pub fn into_inner(self) -> CsMat<T> {
        self.inner
    }

    /// Construct a [`CscMatrix`] from a [`CooMatrix`] by converting to CSC.
    ///
    /// Duplicate entries at the same position are summed.
    ///
    /// # Errors
    ///
    /// This conversion is always successful for structurally valid inputs.
    pub fn from_coo(coo: &CooMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Add<Output = T> + 'static,
    {
        let inner: CsMat<T> = coo.inner().to_csc();
        Ok(Self { inner })
    }

    /// Construct a [`CscMatrix`] from a [`CsrMatrix`].
    ///
    /// # Errors
    ///
    /// This conversion is always successful.
    pub fn from_csr(csr: &CsrMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Default + 'static,
    {
        Ok(csr.to_csc())
    }

    /// Convert to [`CsrMatrix`].
    pub fn to_csr(&self) -> CsrMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        // from_csc is infallible for a valid CscMatrix
        CsrMatrix::from_csc(self).unwrap()
    }

    /// Convert to [`CooMatrix`].
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.n_rows(), self.n_cols(), self.nnz());
        for (val, (r, c)) in self.inner.iter() {
            // indices come from a valid matrix, so push is infallible here
            let _ = coo.push(r, c, val.clone());
        }
        coo
    }

    /// Convert this sparse matrix to a dense [`Array2<T>`].
    pub fn to_dense(&self) -> Array2<T>
    where
        T: Clone + Zero + 'static,
    {
        self.inner.to_dense()
    }

    /// Construct a [`CscMatrix`] from a dense [`Array2<T>`], dropping entries
    /// whose absolute value is less than or equal to `epsilon`.
    pub fn from_dense(dense: &ArrayView2<'_, T>, epsilon: T) -> Self
    where
        T: Copy + Zero + PartialOrd + num_traits::Signed + 'static,
    {
        let inner = CsMat::csc_from_dense(dense.view(), epsilon);
        Self { inner }
    }

    /// Return a new CSC matrix containing only the columns in `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `start > end` or
    /// `end > n_cols()`.
    pub fn col_slice(&self, start: usize, end: usize) -> Result<CscMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if start > end {
            return Err(FerroError::InvalidParameter {
                name: "col_slice range".into(),
                reason: format!("start ({start}) must be <= end ({end})"),
            });
        }
        if end > self.n_cols() {
            return Err(FerroError::InvalidParameter {
                name: "col_slice range".into(),
                reason: format!("end ({end}) exceeds n_cols ({})", self.n_cols()),
            });
        }
        let view = self.inner.slice_outer(start..end);
        Ok(Self {
            inner: view.to_owned(),
        })
    }

    /// Scalar multiplication in-place: multiplies every non-zero by `scalar`.
    ///
    /// Requires `T: for<'r> MulAssign<&'r T>`, which is satisfied by all
    /// primitive numeric types.
    pub fn scale(&mut self, scalar: T)
    where
        for<'r> T: MulAssign<&'r T>,
    {
        self.inner.scale(scalar);
    }

    /// Scalar multiplication returning a new matrix.
    pub fn mul_scalar(&self, scalar: T) -> CscMatrix<T>
    where
        T: Copy + Mul<Output = T> + Zero + 'static,
    {
        let new_inner = self.inner.map(|&v| v * scalar);
        Self { inner: new_inner }
    }

    /// Element-wise addition of two CSC matrices with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn add(&self, rhs: &CscMatrix<T>) -> Result<CscMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: Add<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CscMatrix::add".into(),
            });
        }
        let result = &self.inner + &rhs.inner;
        Ok(Self { inner: result })
    }

    /// Sparse matrix-dense vector product: computes `self * rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `rhs.len() != n_cols()`.
    pub fn mul_vec(&self, rhs: &Array1<T>) -> Result<Array1<T>, FerroError>
    where
        T: Clone + Zero + 'static,
        for<'r> &'r T: Mul<Output = T>,
        T: AddAssign,
    {
        if rhs.len() != self.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_cols()],
                actual: vec![rhs.len()],
                context: "CscMatrix::mul_vec".into(),
            });
        }
        let result = &self.inner * rhs;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn sample_csc() -> CscMatrix<f64> {
        // 3x3 sparse matrix (same logical matrix as CsrMatrix tests):
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        //
        // CSC indptr is per-column:
        //   col 0: rows 0, 2  → values 1, 4
        //   col 1: row 1      → value 3
        //   col 2: rows 0, 2  → values 2, 5
        CscMatrix::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 4.0, 3.0, 2.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_new_valid() {
        let m = sample_csc();
        assert_eq!(m.n_rows(), 3);
        assert_eq!(m.n_cols(), 3);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn test_to_dense() {
        let m = sample_csc();
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[0, 2]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
        assert_abs_diff_eq!(d[[2, 0]], 4.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_from_dense() {
        let dense = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let m = CscMatrix::from_dense(&dense.view(), 0.0);
        assert_eq!(m.nnz(), 2);
        let back = m.to_dense();
        assert_abs_diff_eq!(back[[0, 0]], 1.0);
        assert_abs_diff_eq!(back[[1, 1]], 2.0);
    }

    #[test]
    fn test_from_coo_roundtrip() {
        let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 2, 4.0).unwrap();
        coo.push(2, 1, 7.0).unwrap();
        let csc = CscMatrix::from_coo(&coo).unwrap();
        let dense = csc.to_dense();
        assert_abs_diff_eq!(dense[[0, 0]], 1.0);
        assert_abs_diff_eq!(dense[[1, 2]], 4.0);
        assert_abs_diff_eq!(dense[[2, 1]], 7.0);
    }

    #[test]
    fn test_csc_csr_roundtrip() {
        let csc = sample_csc();
        let csr = csc.to_csr();
        let back = CscMatrix::from_csr(&csr).unwrap();
        assert_eq!(back.to_dense(), csc.to_dense());
    }

    #[test]
    fn test_col_slice() {
        let m = sample_csc();
        let sliced = m.col_slice(0, 2).unwrap();
        assert_eq!(sliced.n_rows(), 3);
        assert_eq!(sliced.n_cols(), 2);
        let d = sliced.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
    }

    #[test]
    fn test_col_slice_empty() {
        let m = sample_csc();
        let sliced = m.col_slice(1, 1).unwrap();
        assert_eq!(sliced.n_cols(), 0);
    }

    #[test]
    fn test_col_slice_invalid() {
        let m = sample_csc();
        assert!(m.col_slice(2, 1).is_err());
        assert!(m.col_slice(0, 4).is_err());
    }

    #[test]
    fn test_mul_scalar() {
        let m = sample_csc();
        let m2 = m.mul_scalar(2.0);
        let d = m2.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_scale_in_place() {
        let mut m = sample_csc();
        m.scale(3.0);
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 3.0);
        assert_abs_diff_eq!(d[[2, 2]], 15.0);
    }

    #[test]
    fn test_add() {
        let m = sample_csc();
        let sum = m.add(&m).unwrap();
        let d = sum.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let m1 = sample_csc();
        let m2 = CscMatrix::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert!(m1.add(&m2).is_err());
    }

    #[test]
    fn test_mul_vec() {
        let m = sample_csc();
        let v = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let result = m.mul_vec(&v).unwrap();
        assert_abs_diff_eq!(result[0], 7.0);
        assert_abs_diff_eq!(result[1], 6.0);
        assert_abs_diff_eq!(result[2], 19.0);
    }

    #[test]
    fn test_mul_vec_shape_mismatch() {
        let m = sample_csc();
        let v = Array1::from(vec![1.0_f64, 2.0]);
        assert!(m.mul_vec(&v).is_err());
    }
}
