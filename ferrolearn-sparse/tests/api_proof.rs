//! Proof-of-API integration test for ferrolearn-sparse.
//!
//! Audit deliverable for crosslink #323 (under #251). Exercises every
//! public matrix type (CSR / CSC / COO) plus the new helpers (eye, diags,
//! hstack, vstack) added in #319.

use ferrolearn_sparse::{CooMatrix, CscMatrix, CsrMatrix, diags, eye, hstack, vstack};
use ndarray::array;

#[test]
fn api_proof_coo_csr_csc_round_trip() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(1, 2, 4.0).unwrap();
    coo.push(2, 1, 7.0).unwrap();

    let csr = CsrMatrix::from_coo(&coo).unwrap();
    let csc: CscMatrix<f64> = csr.to_csc();
    let back: CsrMatrix<f64> = CsrMatrix::from_csc(&csc).unwrap();

    let dense = back.to_dense();
    assert_eq!(dense[[0, 0]], 1.0);
    assert_eq!(dense[[1, 2]], 4.0);
    assert_eq!(dense[[2, 1]], 7.0);
    assert_eq!(csr.n_rows(), 3);
    assert_eq!(csr.n_cols(), 3);
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csc.n_rows(), 3);
    assert_eq!(csc.n_cols(), 3);
    assert_eq!(csc.nnz(), 3);
}

#[test]
fn api_proof_csr_from_dense() {
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]];
    let csr = CsrMatrix::<f64>::from_dense(&dense.view(), 0.0);
    assert_eq!(csr.nnz(), 4);
    let back = csr.to_dense();
    assert_eq!(back, dense);
}

#[test]
fn api_proof_helpers_eye_diags_stack() {
    let id: CsrMatrix<f64> = eye(3).unwrap();
    assert_eq!(id.nnz(), 3);
    let d = id.to_dense();
    for i in 0..3 {
        for j in 0..3 {
            assert!((d[[i, j]] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }

    // diagonals — main, super, sub
    let main: CsrMatrix<f64> = diags(&[1.0, 2.0, 3.0], 0, 3).unwrap();
    let super_d: CsrMatrix<f64> = diags(&[1.0, 2.0], 1, 3).unwrap();
    let sub_d: CsrMatrix<f64> = diags(&[1.0, 2.0], -1, 3).unwrap();
    assert_eq!(main.nnz(), 3);
    assert_eq!(super_d.nnz(), 2);
    assert_eq!(sub_d.nnz(), 2);

    // Stack
    let h = hstack(&[&id, &main]).unwrap();
    assert_eq!((h.n_rows(), h.n_cols()), (3, 6));
    let v = vstack(&[&id, &main]).unwrap();
    assert_eq!((v.n_rows(), v.n_cols()), (6, 3));
}

#[test]
fn api_proof_csr_row_slice() {
    let dense = array![[1.0_f64, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]];
    let csr = CsrMatrix::<f64>::from_dense(&dense.view(), 0.0);
    let sliced = csr.row_slice(1, 3).unwrap();
    assert_eq!(sliced.n_rows(), 2);
    assert_eq!(sliced.n_cols(), 2);
}
