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
        for (val, (r, c)) in &self.inner {
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

/// Kani proof harnesses for CscMatrix structural invariants.
///
/// These proofs verify that after construction via `new()`, `from_coo()`, and
/// `add()`, the underlying CSC representation satisfies all structural
/// invariants:
///
/// - `indptr.len() == n_cols + 1`
/// - `indptr` is monotonically non-decreasing
/// - All row indices are less than `n_rows`
/// - `indices.len() == data.len()`
///
/// All proofs use small symbolic bounds (at most 3 rows/cols) because sparse
/// matrix verification is computationally expensive for Kani.
#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::coo::CooMatrix;

    /// Maximum dimension for symbolic exploration.
    const MAX_DIM: usize = 3;

    /// Helper: assert all CSC structural invariants on the inner `CsMat`.
    fn assert_csc_invariants<T>(m: &CscMatrix<T>) {
        let inner = m.inner();

        // Invariant 1: indptr length == n_cols + 1
        let indptr = inner.indptr();
        let indptr_raw = indptr.raw_storage();
        assert!(indptr_raw.len() == m.n_cols() + 1);

        // Invariant 2: indptr is monotonically non-decreasing
        for i in 0..m.n_cols() {
            assert!(indptr_raw[i] <= indptr_raw[i + 1]);
        }

        // Invariant 3: all row indices < n_rows
        let indices = inner.indices();
        for &row_idx in indices {
            assert!(row_idx < m.n_rows());
        }

        // Invariant 4: indices.len() == data.len()
        assert!(inner.indices().len() == inner.data().len());
    }

    /// Verify `indptr.len() == n_cols + 1` after `new()` with a symbolic
    /// empty matrix of arbitrary dimensions.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indptr_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build a valid empty CSC matrix
        let indptr = vec![0usize; n_cols + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr();
            assert!(inner_indptr.raw_storage().len() == n_cols + 1);
        }
    }

    /// Verify indptr monotonicity after `new()` with a symbolic single-entry
    /// matrix.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indptr_monotonic() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Place a single non-zero at a symbolic valid position
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        // Build indptr for a single entry in column `col`
        let mut indptr = vec![0usize; n_cols + 1];
        for i in (col + 1)..=n_cols {
            indptr[i] = 1;
        }
        let indices = vec![row];
        let data = vec![42i32];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr().raw_storage().to_vec();
            for i in 0..m.n_cols() {
                assert!(inner_indptr[i] <= inner_indptr[i + 1]);
            }
        }
    }

    /// Verify all row indices < n_rows after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_row_indices_in_bounds() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        let mut indptr = vec![0usize; n_cols + 1];
        for i in (col + 1)..=n_cols {
            indptr[i] = 1;
        }
        let indices = vec![row];
        let data = vec![1i32];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            for &r in m.inner().indices() {
                assert!(r < m.n_rows());
            }
        }
    }

    /// Verify `indices.len() == data.len()` after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_indices_data_same_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_cols + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CscMatrix::new(n_rows, n_cols, indptr, indices, data) {
            assert!(m.inner().indices().len() == m.inner().data().len());
        }
    }

    /// Verify that `new()` rejects inputs where indices and data have
    /// mismatched lengths.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_new_rejects_mismatched_lengths() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // indices has 1 element, data has 0 — must fail
        let indptr = vec![0usize; n_cols + 1];
        let indices = vec![0usize];
        let data: Vec<i32> = vec![];

        let result = CscMatrix::new(n_rows, n_cols, indptr, indices, data);
        assert!(result.is_err());
    }

    /// Verify all structural invariants after `from_coo()` with symbolic
    /// entries.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_from_coo_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let mut coo = CooMatrix::<i32>::new(n_rows, n_cols);

        // Insert a symbolic number of entries (0 or 1)
        let do_insert: bool = kani::any();
        if do_insert {
            let row: usize = kani::any();
            let col: usize = kani::any();
            kani::assume(row < n_rows);
            kani::assume(col < n_cols);
            let _ = coo.push(row, col, 1i32);
        }

        if let Ok(csc) = CscMatrix::from_coo(&coo) {
            assert_csc_invariants(&csc);
            assert!(csc.n_rows() == n_rows);
            assert!(csc.n_cols() == n_cols);
        }
    }

    /// Verify that `add()` preserves shape and structural invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_add_preserves_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build two valid empty CSC matrices of the same shape
        let indptr = vec![0usize; n_cols + 1];
        let a = CscMatrix::<i32>::new(n_rows, n_cols, indptr.clone(), vec![], vec![]);
        let b = CscMatrix::<i32>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                // Shape is preserved
                assert!(sum.n_rows() == n_rows);
                assert!(sum.n_cols() == n_cols);
                // Structural invariants hold
                assert_csc_invariants(&sum);
            }
        }
    }

    /// Verify that `add()` with non-empty matrices preserves invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_add_nonempty_preserves_invariants() {
        // Fixed 2x2 matrices with one entry each in different columns
        let a = CscMatrix::<i32>::new(2, 2, vec![0, 1, 1], vec![0], vec![1]);
        let b = CscMatrix::<i32>::new(2, 2, vec![0, 0, 1], vec![1], vec![2]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                assert!(sum.n_rows() == 2);
                assert!(sum.n_cols() == 2);
                assert_csc_invariants(&sum);
            }
        }
    }

    /// Verify `mul_vec()` output has correct dimension and does not panic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_output_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Empty matrix for tractable verification
        let indptr = vec![0usize; n_cols + 1];
        let m = CscMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let Ok(m) = m {
            let v = Array1::<f64>::zeros(n_cols);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == n_rows);
            }
        }
    }

    /// Verify `mul_vec()` rejects vectors of wrong dimension.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_rejects_wrong_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_cols + 1];
        let m = CscMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let Ok(m) = m {
            let wrong_len: usize = kani::any();
            kani::assume(wrong_len <= MAX_DIM);
            kani::assume(wrong_len != n_cols);
            let v = Array1::<f64>::zeros(wrong_len);
            let result = m.mul_vec(&v);
            assert!(result.is_err());
        }
    }

    /// Verify `mul_vec()` with a non-empty matrix produces the correct
    /// output dimension and does not trigger any out-of-bounds access.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csc_mul_vec_nonempty_no_oob() {
        // 2x3 CSC matrix with entries at (0,1) and (1,2)
        // Column 0: empty, Column 1: row 0, Column 2: row 1
        let m = CscMatrix::<f64>::new(2, 3, vec![0, 0, 1, 2], vec![0, 1], vec![3.0, 4.0]);
        if let Ok(m) = m {
            let v = Array1::from(vec![1.0, 2.0, 3.0]);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == 2);
            }
        }
    }
}
