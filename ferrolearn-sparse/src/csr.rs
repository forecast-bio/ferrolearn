//! Compressed Sparse Row (CSR) matrix format.
//!
//! [`CsrMatrix<T>`] is a newtype wrapper around [`sprs::CsMat<T>`] in CSR
//! storage. CSR matrices are efficient for row-wise operations, matrix-vector
//! products, and row slicing.

use std::ops::{Add, AddAssign, Mul, MulAssign};

use ferrolearn_core::{Dataset, FerroError};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, Zero};
use sprs::CsMat;

use crate::coo::CooMatrix;
use crate::csc::CscMatrix;

/// Compressed Sparse Row (CSR) sparse matrix.
///
/// Stores non-zero entries in row-major order using three arrays: `indptr`
/// (row pointer array of length `n_rows + 1`), `indices` (column indices of
/// each non-zero), and `data` (values of each non-zero).
///
/// # Type Parameter
///
/// `T` — the scalar element type. No bounds are required for basic structural
/// methods; arithmetic methods impose their own bounds.
///
/// # Dataset Trait
///
/// Implements [`ferrolearn_core::Dataset`] when `T: Float + Send + Sync + 'static`,
/// reporting `n_samples() == n_rows()`, `n_features() == n_cols()`, and
/// `is_sparse() == true`.
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    inner: CsMat<T>,
}

impl<T> CsrMatrix<T>
where
    T: Clone,
{
    /// Construct a CSR matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `n_rows` — number of rows.
    /// * `n_cols` — number of columns.
    /// * `indptr` — row pointer array of length `n_rows + 1`.
    /// * `indices` — column index of each non-zero entry.
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
        CsMat::try_new((n_rows, n_cols), indptr, indices, data)
            .map(|inner| Self { inner })
            .map_err(|(_, _, _, err)| FerroError::InvalidParameter {
                name: "CsrMatrix raw components".into(),
                reason: err.to_string(),
            })
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

    /// Construct a [`CsrMatrix`] from a [`CooMatrix`] by converting to CSR.
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
        let inner: CsMat<T> = coo.inner().to_csr();
        Ok(Self { inner })
    }

    /// Construct a [`CsrMatrix`] from a [`CscMatrix`].
    ///
    /// # Errors
    ///
    /// This conversion is always successful.
    pub fn from_csc(csc: &CscMatrix<T>) -> Result<Self, FerroError>
    where
        T: Clone + Default + 'static,
    {
        let inner = csc.inner().to_csr();
        Ok(Self { inner })
    }

    /// Convert to [`CscMatrix`].
    pub fn to_csc(&self) -> CscMatrix<T>
    where
        T: Clone + Default + 'static,
    {
        CscMatrix::from_inner(self.inner.to_csc())
    }

    /// Convert to [`CooMatrix`].
    ///
    /// Each non-zero becomes one triplet entry.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.n_rows(), self.n_cols(), self.nnz());
        for (val, (r, c)) in &self.inner {
            // indices come from a valid matrix so push is infallible here
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

    /// Construct a [`CsrMatrix`] from a dense [`Array2<T>`], dropping entries
    /// whose absolute value is less than or equal to `epsilon`.
    ///
    /// Entries `v` where `|v| <= epsilon` are treated as structural zeros.
    /// For integer types, pass `epsilon = 0`.
    pub fn from_dense(dense: &ArrayView2<'_, T>, epsilon: T) -> Self
    where
        T: Copy + Zero + PartialOrd + num_traits::Signed + 'static,
    {
        let inner = CsMat::csr_from_dense(dense.view(), epsilon);
        Self { inner }
    }

    /// Return a new CSR matrix containing only the rows in `start..end`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `start > end` or
    /// `end > n_rows()`.
    pub fn row_slice(&self, start: usize, end: usize) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Clone + Default + 'static,
    {
        if start > end {
            return Err(FerroError::InvalidParameter {
                name: "row_slice range".into(),
                reason: format!("start ({start}) must be <= end ({end})"),
            });
        }
        if end > self.n_rows() {
            return Err(FerroError::InvalidParameter {
                name: "row_slice range".into(),
                reason: format!("end ({end}) exceeds n_rows ({})", self.n_rows()),
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
    pub fn mul_scalar(&self, scalar: T) -> CsrMatrix<T>
    where
        T: Copy + Mul<Output = T> + Zero + 'static,
    {
        let new_inner = self.inner.map(|&v| v * scalar);
        Self { inner: new_inner }
    }

    /// Element-wise addition of two CSR matrices with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the matrices have different shapes.
    pub fn add(&self, rhs: &CsrMatrix<T>) -> Result<CsrMatrix<T>, FerroError>
    where
        T: Zero + Default + Clone + 'static,
        for<'r> &'r T: Add<&'r T, Output = T>,
    {
        if self.n_rows() != rhs.n_rows() || self.n_cols() != rhs.n_cols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_rows(), self.n_cols()],
                actual: vec![rhs.n_rows(), rhs.n_cols()],
                context: "CsrMatrix::add".into(),
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
                context: "CsrMatrix::mul_vec".into(),
            });
        }
        let result = &self.inner * rhs;
        Ok(result)
    }
}

impl<T> CsrMatrix<T>
where
    T: Float + Send + Sync + num_traits::Signed + 'static,
{
    /// Construct a [`CsrMatrix`] from a dense [`Array2<T>`], treating entries
    /// with absolute value at or below `T::epsilon()` as structural zeros.
    pub fn from_dense_float(dense: &ArrayView2<'_, T>) -> Self {
        CsrMatrix::from_dense(dense, T::epsilon())
    }
}

/// Implements [`Dataset`] so that `CsrMatrix<F>` can be passed to any
/// ferrolearn algorithm that accepts a dataset.
///
/// - `n_samples()` — number of rows (one sample per row).
/// - `n_features()` — number of columns (one feature per column).
/// - `is_sparse()` — always `true`.
impl<F> Dataset for CsrMatrix<F>
where
    F: Float + Send + Sync + 'static,
{
    fn n_samples(&self) -> usize {
        self.n_rows()
    }

    fn n_features(&self) -> usize {
        self.n_cols()
    }

    fn is_sparse(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn sample_csr() -> CsrMatrix<f64> {
        // 3x3 sparse matrix:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        CsrMatrix::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn test_new_valid() {
        let m = sample_csr();
        assert_eq!(m.n_rows(), 3);
        assert_eq!(m.n_cols(), 3);
        assert_eq!(m.nnz(), 5);
    }

    #[test]
    fn test_new_invalid() {
        // Wrong indptr length (needs n_rows+1 = 3, not 2)
        let res = CsrMatrix::<f64>::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(res.is_err());
    }

    #[test]
    fn test_to_dense() {
        let m = sample_csr();
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[0, 1]], 0.0);
        assert_abs_diff_eq!(d[[0, 2]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
        assert_abs_diff_eq!(d[[2, 0]], 4.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_from_dense() {
        let dense = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let m = CsrMatrix::from_dense(&dense.view(), 0.0);
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
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let dense = csr.to_dense();
        assert_abs_diff_eq!(dense[[0, 0]], 1.0);
        assert_abs_diff_eq!(dense[[1, 2]], 4.0);
        assert_abs_diff_eq!(dense[[2, 1]], 7.0);
        assert_abs_diff_eq!(dense[[0, 1]], 0.0);
    }

    #[test]
    fn test_to_coo_roundtrip() {
        let csr = sample_csr();
        let coo = csr.to_coo();
        let back = CsrMatrix::from_coo(&coo).unwrap();
        let d = back.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[2, 2]], 5.0);
    }

    #[test]
    fn test_csr_csc_roundtrip() {
        let csr = sample_csr();
        let csc = csr.to_csc();
        let back = CsrMatrix::from_csc(&csc).unwrap();
        assert_eq!(back.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_row_slice() {
        let m = sample_csr();
        let sliced = m.row_slice(0, 2).unwrap();
        assert_eq!(sliced.n_rows(), 2);
        assert_eq!(sliced.n_cols(), 3);
        let d = sliced.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 1.0);
        assert_abs_diff_eq!(d[[1, 1]], 3.0);
    }

    #[test]
    fn test_row_slice_empty() {
        let m = sample_csr();
        let sliced = m.row_slice(1, 1).unwrap();
        assert_eq!(sliced.n_rows(), 0);
    }

    #[test]
    fn test_row_slice_invalid() {
        let m = sample_csr();
        assert!(m.row_slice(2, 1).is_err());
        assert!(m.row_slice(0, 4).is_err());
    }

    #[test]
    fn test_mul_scalar() {
        let m = sample_csr();
        let m2 = m.mul_scalar(2.0);
        let d = m2.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_scale_in_place() {
        let mut m = sample_csr();
        m.scale(3.0);
        let d = m.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 3.0);
        assert_abs_diff_eq!(d[[2, 2]], 15.0);
    }

    #[test]
    fn test_add() {
        let m = sample_csr();
        let sum = m.add(&m).unwrap();
        let d = sum.to_dense();
        assert_abs_diff_eq!(d[[0, 0]], 2.0);
        assert_abs_diff_eq!(d[[1, 1]], 6.0);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let m1 = sample_csr();
        let m2 = CsrMatrix::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
        assert!(m1.add(&m2).is_err());
    }

    #[test]
    fn test_mul_vec() {
        let m = sample_csr();
        // [1 0 2]   [1]   [7]
        // [0 3 0] * [2] = [6]
        // [4 0 5]   [3]   [19]
        let v = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let result = m.mul_vec(&v).unwrap();
        assert_abs_diff_eq!(result[0], 7.0);
        assert_abs_diff_eq!(result[1], 6.0);
        assert_abs_diff_eq!(result[2], 19.0);
    }

    #[test]
    fn test_mul_vec_shape_mismatch() {
        let m = sample_csr();
        let v = Array1::from(vec![1.0_f64, 2.0]);
        assert!(m.mul_vec(&v).is_err());
    }

    #[test]
    fn test_dataset_trait() {
        let m = sample_csr();
        assert_eq!(m.n_samples(), 3);
        assert_eq!(m.n_features(), 3);
        assert!(m.is_sparse());
    }

    #[test]
    fn test_dataset_trait_object() {
        use ferrolearn_core::Dataset;
        let m: CsrMatrix<f64> = sample_csr();
        let d: &dyn Dataset = &m;
        assert_eq!(d.n_samples(), 3);
        assert!(d.is_sparse());
    }

    #[test]
    fn test_from_dense_float() {
        let dense = array![[1.0_f64, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let csr = CsrMatrix::from_dense_float(&dense.view());
        assert_eq!(csr.nnz(), 2);
        let back = csr.to_dense();
        assert_abs_diff_eq!(back[[0, 0]], 1.0);
        assert_abs_diff_eq!(back[[1, 2]], 2.0);
    }
}

/// Kani proof harnesses for CsrMatrix structural invariants.
///
/// These proofs verify that after construction via `new()`, `from_coo()`, and
/// `add()`, the underlying CSR representation satisfies all structural
/// invariants:
///
/// - `indptr.len() == n_rows + 1`
/// - `indptr` is monotonically non-decreasing
/// - All column indices are less than `n_cols`
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
    /// Maximum number of non-zero entries for symbolic exploration.
    const MAX_NNZ: usize = 4;

    /// Helper: assert all CSR structural invariants on the inner `CsMat`.
    fn assert_csr_invariants<T>(m: &CsrMatrix<T>) {
        let inner = m.inner();

        // Invariant 1: indptr length == n_rows + 1
        let indptr = inner.indptr();
        let indptr_raw = indptr.raw_storage();
        assert!(indptr_raw.len() == m.n_rows() + 1);

        // Invariant 2: indptr is monotonically non-decreasing
        for i in 0..m.n_rows() {
            assert!(indptr_raw[i] <= indptr_raw[i + 1]);
        }

        // Invariant 3: all column indices < n_cols
        let indices = inner.indices();
        for &col_idx in indices {
            assert!(col_idx < m.n_cols());
        }

        // Invariant 4: indices.len() == data.len()
        assert!(inner.indices().len() == inner.data().len());
    }

    /// Verify `indptr.len() == n_rows + 1` after `new()` with a symbolic
    /// empty matrix of arbitrary dimensions.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indptr_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build a valid empty CSR matrix
        let indptr = vec![0usize; n_rows + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr();
            assert!(inner_indptr.raw_storage().len() == n_rows + 1);
        }
    }

    /// Verify indptr monotonicity after `new()` with a symbolic single-entry
    /// matrix.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indptr_monotonic() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Place a single non-zero at a symbolic valid position
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        // Build indptr for a single entry at (row, col)
        let mut indptr = vec![0usize; n_rows + 1];
        for i in (row + 1)..=n_rows {
            indptr[i] = 1;
        }
        let indices = vec![col];
        let data = vec![42i32];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            let inner_indptr = m.inner().indptr().raw_storage().to_vec();
            for i in 0..m.n_rows() {
                assert!(inner_indptr[i] <= inner_indptr[i + 1]);
            }
        }
    }

    /// Verify all column indices < n_cols after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_column_indices_in_bounds() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let col: usize = kani::any();
        let row: usize = kani::any();
        kani::assume(row < n_rows);
        kani::assume(col < n_cols);

        let mut indptr = vec![0usize; n_rows + 1];
        for i in (row + 1)..=n_rows {
            indptr[i] = 1;
        }
        let indices = vec![col];
        let data = vec![1i32];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            for &c in m.inner().indices() {
                assert!(c < m.n_cols());
            }
        }
    }

    /// Verify `indices.len() == data.len()` after `new()`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_indices_data_same_length() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Try empty matrix
        let indptr = vec![0usize; n_rows + 1];
        let indices: Vec<usize> = vec![];
        let data: Vec<i32> = vec![];

        if let Ok(m) = CsrMatrix::new(n_rows, n_cols, indptr, indices, data) {
            assert!(m.inner().indices().len() == m.inner().data().len());
        }
    }

    /// Verify that `new()` rejects inputs where indices and data have
    /// mismatched lengths.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_new_rejects_mismatched_lengths() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // indices has 1 element, data has 0 — must fail
        let indptr = vec![0usize; n_rows + 1];
        let indices = vec![0usize];
        let data: Vec<i32> = vec![];

        let result = CsrMatrix::new(n_rows, n_cols, indptr, indices, data);
        assert!(result.is_err());
    }

    /// Verify all structural invariants after `from_coo()` with symbolic
    /// entries.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_from_coo_invariants() {
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

        if let Ok(csr) = CsrMatrix::from_coo(&coo) {
            assert_csr_invariants(&csr);
            assert!(csr.n_rows() == n_rows);
            assert!(csr.n_cols() == n_cols);
        }
    }

    /// Verify that `add()` preserves shape and structural invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_add_preserves_invariants() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Build two valid empty CSR matrices of the same shape
        let indptr = vec![0usize; n_rows + 1];
        let a = CsrMatrix::<i32>::new(n_rows, n_cols, indptr.clone(), vec![], vec![]);
        let b = CsrMatrix::<i32>::new(n_rows, n_cols, indptr, vec![], vec![]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                // Shape is preserved
                assert!(sum.n_rows() == n_rows);
                assert!(sum.n_cols() == n_cols);
                // Structural invariants hold
                assert_csr_invariants(&sum);
            }
        }
    }

    /// Verify that `add()` with non-empty matrices preserves invariants.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_add_nonempty_preserves_invariants() {
        // Fixed 2x2 matrices with one entry each in different positions
        let a = CsrMatrix::<i32>::new(2, 2, vec![0, 1, 1], vec![0], vec![1]);
        let b = CsrMatrix::<i32>::new(2, 2, vec![0, 0, 1], vec![1], vec![2]);

        if let (Ok(a), Ok(b)) = (a, b) {
            if let Ok(sum) = a.add(&b) {
                assert!(sum.n_rows() == 2);
                assert!(sum.n_cols() == 2);
                assert_csr_invariants(&sum);
            }
        }
    }

    /// Verify `mul_vec()` output has correct dimension and does not panic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn csr_mul_vec_output_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        // Empty matrix for tractable verification
        let indptr = vec![0usize; n_rows + 1];
        let m = CsrMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

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
    fn csr_mul_vec_rejects_wrong_dimension() {
        let n_rows: usize = kani::any();
        let n_cols: usize = kani::any();
        kani::assume(n_rows > 0 && n_rows <= MAX_DIM);
        kani::assume(n_cols > 0 && n_cols <= MAX_DIM);

        let indptr = vec![0usize; n_rows + 1];
        let m = CsrMatrix::<f64>::new(n_rows, n_cols, indptr, vec![], vec![]);

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
    fn csr_mul_vec_nonempty_no_oob() {
        // 2x3 matrix with entries at (0,1) and (1,2)
        let m = CsrMatrix::<f64>::new(2, 3, vec![0, 1, 2], vec![1, 2], vec![3.0, 4.0]);
        if let Ok(m) = m {
            let v = Array1::from(vec![1.0, 2.0, 3.0]);
            if let Ok(result) = m.mul_vec(&v) {
                assert!(result.len() == 2);
            }
        }
    }
}
