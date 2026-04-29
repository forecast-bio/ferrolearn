//! Lanczos eigensolver for large symmetric matrices.
//!
//! This module provides an equivalent to SciPy's `scipy.sparse.linalg.eigsh`:
//! a Lanczos iteration with implicit restarts (Sorensen's Implicitly Restarted
//! Lanczos Method — IRLM) for computing a subset of eigenvalues and
//! eigenvectors of a large, symmetric matrix without forming it densely.
//!
//! The solver accepts either:
//! - A **closure** `matvec` that computes the matrix-vector product `A * v`,
//!   allowing implicit, sparse, or dense representations.
//! - A **sparse CSR matrix** (`sprs::CsMat<f64>`) via the convenience
//!   [`eigsh`] function.
//!
//! # Algorithm
//!
//! 1. **Lanczos iteration** builds an orthonormal Krylov basis `V` and a
//!    symmetric tridiagonal matrix `T` such that `A V ≈ V T`.
//!    Full reorthogonalisation (modified Gram-Schmidt) is applied at every
//!    step to combat loss of orthogonality.
//! 2. **Tridiagonal eigendecomposition** of `T` is performed using the
//!    symmetric QR algorithm with Wilkinson shifts.
//! 3. **Implicit restart** (thick restart): after eigendecomposing `T`,
//!    only the `k` wanted Ritz values are retained and the Lanczos
//!    factorisation is compressed, then extended again. This is repeated
//!    until convergence or the iteration budget is exhausted.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_numerical::sparse_eig::{eigsh, WhichEigenvalues};
//! use sprs::CsMat;
//!
//! // Build a 4×4 diagonal matrix with eigenvalues 1, 2, 3, 4.
//! let diag: CsMat<f64> = CsMat::new(
//!     (4, 4),
//!     vec![0, 1, 2, 3, 4],
//!     vec![0, 1, 2, 3],
//!     vec![1.0, 2.0, 3.0, 4.0],
//! );
//! let result = eigsh(&diag, 2, WhichEigenvalues::LargestAlgebraic).unwrap();
//! assert_eq!(result.eigenvalues.len(), 2);
//! ```

use ndarray::{Array1, Array2, s};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which end of the spectrum to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichEigenvalues {
    /// Largest eigenvalues (largest algebraic value).
    LargestAlgebraic,
    /// Smallest eigenvalues (smallest algebraic value).
    SmallestAlgebraic,
    /// Largest magnitude eigenvalues.
    LargestMagnitude,
    /// Smallest magnitude eigenvalues.
    SmallestMagnitude,
}

/// Result of an eigendecomposition.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues, sorted according to the `which` parameter.
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns of the matrix.
    pub eigenvectors: Array2<f64>,
}

/// Configuration for the Lanczos eigensolver.
///
/// Use the builder-style methods ([`LanczosSolver::with_which`], etc.) to
/// customise convergence parameters.
#[derive(Debug, Clone)]
pub struct LanczosSolver {
    /// Number of eigenvalues to compute.
    pub k: usize,
    /// Which end of the spectrum to target.
    pub which: WhichEigenvalues,
    /// Maximum number of restart iterations (default 300).
    pub max_iter: usize,
    /// Convergence tolerance for the Ritz residual (default 1e-10).
    pub tol: f64,
    /// Number of Lanczos vectors to maintain. `None` selects a sensible
    /// default: `min(n, max(2k+1, 20))`.
    pub ncv: Option<usize>,
}

impl LanczosSolver {
    /// Create a new Lanczos solver to compute `k` eigenvalues.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            which: WhichEigenvalues::LargestAlgebraic,
            max_iter: 300,
            tol: 1e-10,
            ncv: None,
        }
    }

    /// Set which eigenvalues to target.
    #[must_use]
    pub fn with_which(mut self, which: WhichEigenvalues) -> Self {
        self.which = which;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of restart iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of Lanczos vectors to maintain.
    #[must_use]
    pub fn with_ncv(mut self, ncv: usize) -> Self {
        self.ncv = Some(ncv);
        self
    }

    /// Solve using a matrix-vector product closure.
    ///
    /// `n` is the dimension of the square matrix. The closure `matvec`
    /// must compute `A * v` for any vector `v` of length `n`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `k` is zero or exceeds `n`.
    /// - The solver fails to converge within `max_iter` restarts.
    pub fn solve<F>(&self, n: usize, matvec: F) -> Result<EigenResult, String>
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        self.solve_impl(n, matvec)
    }

    /// Convenience: solve for a sparse CSR matrix.
    ///
    /// The matrix must be square and symmetric (only the structure is
    /// validated; symmetry of values is the caller's responsibility).
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or if the solver
    /// fails to converge.
    pub fn solve_sparse(&self, mat: &sprs::CsMat<f64>) -> Result<EigenResult, String> {
        let (rows, cols) = mat.shape();
        if rows != cols {
            return Err(format!("Matrix must be square, got shape ({rows}, {cols})"));
        }
        let n = rows;
        let mat_csr = mat.to_csr();
        self.solve_impl(n, |v: &Array1<f64>| sparse_matvec(&mat_csr, v))
    }

    // -----------------------------------------------------------------------
    // Core implementation
    // -----------------------------------------------------------------------

    /// Core Lanczos iteration with implicit restarts.
    fn solve_impl<F>(&self, n: usize, mut matvec: F) -> Result<EigenResult, String>
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        let k = self.k;
        if k == 0 {
            return Err("k must be at least 1".to_string());
        }
        if k > n {
            return Err(format!(
                "k ({k}) must not exceed the matrix dimension n ({n})"
            ));
        }

        // Special case: k == n, just build full Lanczos factorisation.
        let ncv = if k == n {
            n
        } else {
            self.ncv.unwrap_or_else(|| n.min((2 * k + 1).max(20)))
        };
        let ncv = ncv.min(n);

        if ncv < k + 1 && k < n {
            return Err(format!("ncv ({ncv}) must be at least k+1 ({})", k + 1));
        }

        // Lanczos vectors (columns of V), tridiagonal entries.
        let mut v_mat = Array2::<f64>::zeros((n, ncv));
        let mut alpha = Array1::<f64>::zeros(ncv); // diagonal of T
        let mut beta = Array1::<f64>::zeros(ncv); // sub-diagonal of T

        // Initial random vector (deterministic seed for reproducibility).
        let mut v0 = Array1::<f64>::zeros(n);
        // Simple deterministic initialisation.
        for i in 0..n {
            v0[i] = ((i as f64 + 1.0) * std::f64::consts::FRAC_1_SQRT_2).sin();
        }
        let norm0 = l2_norm(&v0);
        if norm0 == 0.0 {
            v0[0] = 1.0;
        } else {
            v0 /= norm0;
        }
        v_mat.column_mut(0).assign(&v0);

        // Build initial Lanczos factorisation of size ncv.
        let mut f = build_lanczos(
            &mut matvec,
            &mut v_mat,
            &mut alpha,
            &mut beta,
            0, // start index
            ncv,
            n,
        );

        // Implicit restart loop.
        for _iter in 0..self.max_iter {
            // Eigendecompose the tridiagonal matrix T of size ncv.
            let m = ncv;
            let (eigenvalues, eigvecs_t) = tridiag_qr_eigen(
                &alpha.slice(s![..m]).to_owned(),
                &beta.slice(s![1..m]).to_owned(),
            );

            // Select the k wanted Ritz values.
            let wanted_indices = select_indices(&eigenvalues, k, self.which);

            // Check convergence: the Ritz residual for eigenvalue i is
            // |beta_{m}| * |last component of the i-th eigenvector of T|.
            let beta_m = l2_norm(&f);
            let mut converged = true;
            for &idx in &wanted_indices {
                let last_comp = eigvecs_t[[m - 1, idx]].abs();
                let ritz_residual = beta_m * last_comp;
                if ritz_residual > self.tol * eigenvalues[idx].abs().max(1.0) {
                    converged = false;
                    break;
                }
            }

            if converged || k == ncv {
                // Extract eigenpairs.
                let mut result_eigenvalues = Array1::<f64>::zeros(k);
                let mut result_eigenvectors = Array2::<f64>::zeros((n, k));
                for (out_col, &idx) in wanted_indices.iter().enumerate() {
                    result_eigenvalues[out_col] = eigenvalues[idx];
                    // Ritz vector = V * (eigenvector of T)
                    let y = eigvecs_t.column(idx);
                    let v_block = v_mat.slice(s![.., ..m]);
                    let ritz_vec = v_block.dot(&y);
                    result_eigenvectors.column_mut(out_col).assign(&ritz_vec);
                }
                return Ok(EigenResult {
                    eigenvalues: result_eigenvalues,
                    eigenvectors: result_eigenvectors,
                });
            }

            // ---------------------------------------------------------------
            // Implicit restart (Sorensen thick restart)
            // ---------------------------------------------------------------
            // We keep the k wanted Ritz vectors and restart from them.

            // Compute the full Ritz vectors for the k wanted ones.
            let v_block = v_mat.slice(s![.., ..m]).to_owned();
            let mut q_k = Array2::<f64>::zeros((n, k));
            for (j, &idx) in wanted_indices.iter().enumerate() {
                let y = eigvecs_t.column(idx);
                let ritz_vec = v_block.dot(&y);
                q_k.column_mut(j).assign(&ritz_vec);
            }

            // Reorthogonalise the k Ritz vectors (they should already be
            // approximately orthonormal, but numerical drift accumulates).
            gram_schmidt_columns(&mut q_k);

            // Place the k Ritz vectors as the first k columns of V.
            for j in 0..k {
                v_mat.column_mut(j).assign(&q_k.column(j));
            }

            // Rebuild the tridiagonal entries for the first k columns
            // using the Ritz values on the diagonal and the coupling
            // coefficients on the sub-diagonal.
            for (j, &idx) in wanted_indices.iter().enumerate() {
                alpha[j] = eigenvalues[idx];
            }
            // The sub-diagonal coupling from block k-1 to k is:
            //   beta[k] = beta_m * eigvecs_t[m-1, wanted_k-1]
            // but for the thick restart we zero out sub-diags 1..k-1
            // and set beta[k] to the residual norm.
            for j in 0..k {
                beta[j] = 0.0;
            }

            // The residual vector for the restart:
            // f_k = beta_m * (sum over unwanted shifts of q_{unwanted}) ...
            // In the thick restart, f is already available; we orthogonalise
            // it against the new basis.
            let mut r = f.clone();
            // Add contributions from the last components of wanted eigenvectors.
            // r = beta_m * v_m (the continuation vector) — we already have f = beta_m * v_m
            // projected: r -= sum_j <r, q_j> * q_j
            for j in 0..k {
                let qj = v_mat.column(j).to_owned();
                let proj = r.dot(&qj);
                r = r - proj * &qj;
            }
            let r_norm = l2_norm(&r);

            if r_norm < 1e-15 {
                // The residual is effectively zero — we have an invariant
                // subspace. Restart with a random perturbation.
                let seed = (_iter as f64 + 1.0) * std::f64::consts::E;
                for i in 0..n {
                    r[i] = ((i as f64 * seed + 0.5) * 7.31 + seed).sin()
                        + ((i as f64 + 1.0) * (seed + 3.7)).cos();
                }
                // Double Gram-Schmidt for robust orthogonality.
                for _pass in 0..2 {
                    for j in 0..k {
                        let qj = v_mat.column(j).to_owned();
                        let proj = r.dot(&qj);
                        r = r - proj * &qj;
                    }
                }
                let rn = l2_norm(&r);
                if rn > 1e-15 {
                    r /= rn;
                }
                beta[k] = 0.0;
            } else {
                r /= r_norm;
                beta[k] = r_norm;
            }

            v_mat.column_mut(k).assign(&r);

            // Continue building the Lanczos factorisation from index k+1.
            f = build_lanczos(
                &mut matvec,
                &mut v_mat,
                &mut alpha,
                &mut beta,
                k, // start from column k (v_mat[k] is already set)
                ncv,
                n,
            );
        }

        Err(format!(
            "Lanczos solver did not converge within {} iterations",
            self.max_iter
        ))
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Compute `k` eigenvalues/eigenvectors of a symmetric sparse matrix.
///
/// This is a thin wrapper around [`LanczosSolver`].
///
/// # Errors
///
/// Returns an error if the matrix is not square or the solver fails to
/// converge.
pub fn eigsh(
    mat: &sprs::CsMat<f64>,
    k: usize,
    which: WhichEigenvalues,
) -> Result<EigenResult, String> {
    LanczosSolver::new(k).with_which(which).solve_sparse(mat)
}

// ---------------------------------------------------------------------------
// Lanczos building block
// ---------------------------------------------------------------------------

/// Extend a partial Lanczos factorisation from column `start` to column
/// `end - 1`. The vector `v_mat[:, start]` must already be filled with a
/// normalised vector, and `beta[start]` must contain the coupling
/// coefficient from the previous step (or 0 if `start == 0`).
///
/// Returns the unnormalised residual vector `f = beta_{end} * v_{end}`.
fn build_lanczos<F>(
    matvec: &mut F,
    v_mat: &mut Array2<f64>,
    alpha: &mut Array1<f64>,
    beta: &mut Array1<f64>,
    start: usize,
    end: usize,
    n: usize,
) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let mut f = Array1::<f64>::zeros(n);

    for j in start..end {
        let vj = v_mat.column(j).to_owned();
        let mut w = matvec(&vj);

        // Subtract contribution from previous vector.
        if j > 0 {
            let bj = beta[j];
            let v_prev = v_mat.column(j - 1).to_owned();
            w = w - bj * &v_prev;
        }

        // Diagonal entry.
        let a = w.dot(&vj);
        alpha[j] = a;
        w = w - a * &vj;

        // Full reorthogonalisation (modified Gram-Schmidt) against all
        // previous Lanczos vectors to maintain orthogonality.
        for i in 0..=j {
            let vi = v_mat.column(i).to_owned();
            let proj = w.dot(&vi);
            w = w - proj * &vi;
        }
        // Second pass of MGS for numerical stability.
        for i in 0..=j {
            let vi = v_mat.column(i).to_owned();
            let proj = w.dot(&vi);
            w = w - proj * &vi;
        }

        let b = l2_norm(&w);

        if j + 1 < end {
            if b < 1e-15 {
                // Invariant subspace found — generate a new random vector
                // orthogonal to the current basis.
                let mut new_v = Array1::<f64>::zeros(n);
                // Use a hash-like seed that varies strongly with j to
                // avoid near-linear-dependence between restarts.
                let seed = (j as f64 + 1.0) * std::f64::consts::E;
                for i in 0..n {
                    new_v[i] = ((i as f64 * seed + 0.5) * 7.31 + seed).sin()
                        + ((i as f64 + 1.0) * (seed + 3.7)).cos();
                }
                // Double Gram-Schmidt to ensure robust orthogonality.
                for _pass in 0..2 {
                    for i in 0..=j {
                        let vi = v_mat.column(i).to_owned();
                        let proj = new_v.dot(&vi);
                        new_v = new_v - proj * &vi;
                    }
                }
                let nn = l2_norm(&new_v);
                if nn > 1e-15 {
                    new_v /= nn;
                }
                beta[j + 1] = 0.0;
                v_mat.column_mut(j + 1).assign(&new_v);
            } else {
                w /= b;
                beta[j + 1] = b;
                v_mat.column_mut(j + 1).assign(&w);
            }
        } else {
            // This is the residual that sticks out beyond the factorisation.
            f = w; // unnormalised
        }
    }

    f
}

// ---------------------------------------------------------------------------
// Sparse matrix-vector product
// ---------------------------------------------------------------------------

/// Compute `y = A * x` for a sparse CSR matrix `A`.
fn sparse_matvec(mat: &sprs::CsMat<f64>, x: &Array1<f64>) -> Array1<f64> {
    let (rows, _cols) = mat.shape();
    let mut y = Array1::<f64>::zeros(rows);
    for (row_idx, row_vec) in mat.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row_vec.iter() {
            sum += val * x[col_idx];
        }
        y[row_idx] = sum;
    }
    y
}

// ---------------------------------------------------------------------------
// Tridiagonal symmetric QR eigendecomposition
// ---------------------------------------------------------------------------

/// Eigendecompose a symmetric tridiagonal matrix defined by diagonal `d`
/// and sub-diagonal `e` using the implicit symmetric QR algorithm with
/// Wilkinson shifts (Golub & Van Loan, Algorithm 8.3.3).
///
/// `d` has length `n` (diagonal), `e` has length `n-1` (sub-diagonal,
/// where `e[i]` connects `d[i]` and `d[i+1]`).
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored as
/// columns. The eigenvalues are returned in ascending algebraic order.
fn tridiag_qr_eigen(d: &Array1<f64>, e: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = d.len();
    assert_eq!(e.len() + 1, n, "sub-diagonal length must be n-1");

    if n == 0 {
        return (Array1::zeros(0), Array2::zeros((0, 0)));
    }
    if n == 1 {
        return (d.clone(), Array2::eye(1));
    }

    // Working copies: diag[0..n] and sub[0..n-1].
    let mut diag = d.to_vec();
    let mut sub = e.to_vec();

    // Eigenvector accumulator (start with identity).
    let mut z = Array2::<f64>::eye(n);

    let max_total_iter = 30 * n * n;
    let mut total_iter = 0;

    // `end` is one past the last active index; we deflate from the bottom.
    let mut end = n;

    while end > 1 && total_iter < max_total_iter {
        // Check for convergence of sub[end-2] (connecting diag[end-2]
        // and diag[end-1]).
        let threshold = 1e-14 * (diag[end - 2].abs() + diag[end - 1].abs()).max(f64::MIN_POSITIVE);
        if sub[end - 2].abs() <= threshold {
            sub[end - 2] = 0.0;
            end -= 1;
            continue;
        }

        // Find the start of the unreduced block: the largest `start`
        // such that sub[start..end-1] are all non-negligible.
        let mut start = end - 2;
        while start > 0 {
            let thr = 1e-14 * (diag[start - 1].abs() + diag[start].abs()).max(f64::MIN_POSITIVE);
            if sub[start - 1].abs() <= thr {
                sub[start - 1] = 0.0;
                break;
            }
            start -= 1;
        }

        // Wilkinson shift from the trailing 2×2 block.
        let sigma = wilkinson_shift(diag[end - 2], diag[end - 1], sub[end - 2]);

        // Implicit QR step on diag[start..end], sub[start..end-1].
        implicit_qr_step(&mut diag, &mut sub, &mut z, start, end, sigma);
        total_iter += 1;
    }

    // Sort eigenvalues in ascending order and reorder eigenvectors.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap());

    let sorted_evals = Array1::from_iter(indices.iter().map(|&i| diag[i]));
    let mut sorted_evecs = Array2::<f64>::zeros((n, n));
    for (out_col, &idx) in indices.iter().enumerate() {
        sorted_evecs.column_mut(out_col).assign(&z.column(idx));
    }

    (sorted_evals, sorted_evecs)
}

/// Compute the Wilkinson shift for a 2×2 trailing block:
///
/// ```text
/// [ a   b ]
/// [ b   c ]
/// ```
///
/// Returns the eigenvalue of the 2×2 block closer to `c`.
fn wilkinson_shift(a: f64, c: f64, b: f64) -> f64 {
    let delta = (a - c) / 2.0;
    if delta == 0.0 {
        c - b.abs()
    } else {
        c - b * b / (delta + delta.signum() * (delta * delta + b * b).sqrt())
    }
}

/// Perform one implicit symmetric QR step with shift `sigma` on the
/// tridiagonal matrix stored as `diag[start..end]` (diagonal) and
/// `sub[start..end-1]` (sub-diagonal). Accumulates Givens rotations
/// into the eigenvector matrix `z`.
///
/// This follows the standard bulge-chasing scheme (Golub & Van Loan,
/// sec 8.3.5): form the first Givens rotation from the shifted leading
/// element, then chase the resulting bulge down the tridiagonal until it
/// falls off the bottom.
fn implicit_qr_step(
    diag: &mut [f64],
    sub: &mut [f64],
    z: &mut Array2<f64>,
    start: usize,
    end: usize, // exclusive: active block is diag[start..end]
    sigma: f64,
) {
    // The implicit shift: we want to chase the bulge created by the first
    // Givens rotation G(start, start+1) that would zero the (start+1, start)
    // entry of (T - sigma I).
    let m = end - 1; // last index in block

    let mut bulge; // the bulge element that we chase

    // First rotation: zero out the (start, start+1) entry of T - sigma I.
    {
        let x = diag[start] - sigma;
        let y = sub[start];
        let (c, s) = givens_rot(x, y);

        // Apply G^T * T * G for rows/cols start, start+1.
        apply_givens_to_tridiag(diag, sub, start, c, s);

        // The rotation creates a bulge at position (start, start+2) if it
        // exists — stored in `bulge`.
        if start + 2 <= m {
            bulge = s * sub[start + 1];
            sub[start + 1] *= c;
        } else {
            bulge = 0.0;
        }

        // Accumulate into eigenvector matrix.
        apply_givens_to_columns(z, start, start + 1, c, s);
    }

    // Chase the bulge down.
    for k in start + 1..m {
        let (c, s) = givens_rot(sub[k - 1], bulge);

        // sub[k-1] absorbs the bulge.
        sub[k - 1] = c * sub[k - 1] + s * bulge;

        // Apply G^T * T * G for rows/cols k, k+1.
        apply_givens_to_tridiag(diag, sub, k, c, s);

        // New bulge appears at (k, k+2) if k+2 <= m.
        if k + 2 <= m {
            bulge = s * sub[k + 1];
            sub[k + 1] *= c;
        } else {
            bulge = 0.0;
        }

        apply_givens_to_columns(z, k, k + 1, c, s);
    }
}

/// Apply a Givens rotation to the 2×2 block of the tridiagonal matrix
/// at positions (k, k+1). This computes `G^T * T_block * G` where
/// `G = [[c, s], [-s, c]]` and the 2×2 block is:
///
/// ```text
/// [ diag[k]    sub[k] ]
/// [ sub[k]   diag[k+1]]
/// ```
fn apply_givens_to_tridiag(diag: &mut [f64], sub: &mut [f64], k: usize, c: f64, s: f64) {
    let d0 = diag[k];
    let d1 = diag[k + 1];
    let e = sub[k];

    // G^T * [[d0, e], [e, d1]] * G
    diag[k] = c * c * d0 + 2.0 * c * s * e + s * s * d1;
    diag[k + 1] = s * s * d0 - 2.0 * c * s * e + c * c * d1;
    sub[k] = c * s * (d1 - d0) + (c * c - s * s) * e;
}

/// Apply a Givens rotation to columns `i` and `j` of matrix `z`:
/// `z[:, i], z[:, j] <- z[:, i]*c + z[:, j]*s,  -z[:, i]*s + z[:, j]*c`
fn apply_givens_to_columns(z: &mut Array2<f64>, i: usize, j: usize, c: f64, s: f64) {
    let nrows = z.nrows();
    for row in 0..nrows {
        let a = z[[row, i]];
        let b = z[[row, j]];
        z[[row, i]] = c * a + s * b;
        z[[row, j]] = -s * a + c * b;
    }
}

/// Compute a Givens rotation `(c, s)` such that:
///
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// where `r = sqrt(a^2 + b^2)`.
fn givens_rot(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if a == 0.0 {
        (0.0, b.signum())
    } else {
        let r = a.hypot(b);
        (a / r, b / r)
    }
}

// ---------------------------------------------------------------------------
// Eigenvalue selection
// ---------------------------------------------------------------------------

/// Select `k` indices from `eigenvalues` according to `which`.
fn select_indices(eigenvalues: &Array1<f64>, k: usize, which: WhichEigenvalues) -> Vec<usize> {
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();

    match which {
        WhichEigenvalues::LargestAlgebraic => {
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .partial_cmp(&eigenvalues[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestAlgebraic => {
            indices.sort_by(|&a, &b| {
                eigenvalues[a]
                    .partial_cmp(&eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::LargestMagnitude => {
            indices.sort_by(|&a, &b| {
                eigenvalues[b]
                    .abs()
                    .partial_cmp(&eigenvalues[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        WhichEigenvalues::SmallestMagnitude => {
            indices.sort_by(|&a, &b| {
                eigenvalues[a]
                    .abs()
                    .partial_cmp(&eigenvalues[b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    indices.truncate(k);
    indices
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// Euclidean (L2) norm of a vector.
fn l2_norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// In-place modified Gram-Schmidt orthonormalisation of columns of `a`.
fn gram_schmidt_columns(a: &mut Array2<f64>) {
    let ncols = a.ncols();
    for j in 0..ncols {
        // Orthogonalise against all previous columns.
        for i in 0..j {
            let qi = a.column(i).to_owned();
            let proj = a.column(j).dot(&qi);
            let col = a.column(j).to_owned();
            a.column_mut(j).assign(&(col - proj * &qi));
        }
        // Normalise.
        let col = a.column(j).to_owned();
        let norm = l2_norm(&col);
        if norm > 1e-15 {
            a.column_mut(j).assign(&(col / norm));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    /// Build a sparse diagonal matrix from a slice of diagonal values.
    fn sparse_diag(values: &[f64]) -> sprs::CsMat<f64> {
        let n = values.len();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);
        indptr.push(0);
        for (i, &v) in values.iter().enumerate() {
            indices.push(i);
            data.push(v);
            indptr.push(i + 1);
        }
        sprs::CsMat::new((n, n), indptr, indices, data)
    }

    /// Build a sparse matrix from a dense `Array2`.
    fn dense_to_sparse(a: &Array2<f64>) -> sprs::CsMat<f64> {
        let (rows, cols) = (a.nrows(), a.ncols());
        let mut indptr = Vec::with_capacity(rows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..rows {
            for j in 0..cols {
                let v = a[[i, j]];
                if v.abs() > 1e-15 {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        sprs::CsMat::new((rows, cols), indptr, indices, data)
    }

    #[test]
    fn tridiag_qr_identity() {
        // Direct test of tridiag_qr_eigen on a 5×5 identity (diag=1, off=0).
        let d = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let e = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let (evals, _evecs) = super::tridiag_qr_eigen(&d, &e);
        for &ev in &evals {
            assert_abs_diff_eq!(ev, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn tridiag_qr_simple() {
        // 3×3 tridiagonal: diag=[2,2,2], off=[1,1]
        // Eigenvalues: 2 + 2*cos(k*pi/4) for k=1,2,3
        // = 2+sqrt(2), 2, 2-sqrt(2)
        let d = Array1::from_vec(vec![2.0, 2.0, 2.0]);
        let e = Array1::from_vec(vec![1.0, 1.0]);
        let (evals, evecs) = super::tridiag_qr_eigen(&d, &e);

        let mut sorted: Vec<f64> = evals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sq2 = std::f64::consts::SQRT_2;
        assert_abs_diff_eq!(sorted[0], 2.0 - sq2, epsilon = 1e-10);
        assert_abs_diff_eq!(sorted[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sorted[2], 2.0 + sq2, epsilon = 1e-10);

        // Verify eigenvectors: Q^T Q = I
        let qt_q = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qt_q[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn identity_eigenvalues() {
        // 10×10 identity matrix: all eigenvalues should be 1.
        let n = 10;
        let mat = sparse_diag(&vec![1.0; n]);
        let result = eigsh(&mat, n, WhichEigenvalues::LargestAlgebraic).unwrap();
        assert_eq!(result.eigenvalues.len(), n);
        for &ev in &result.eigenvalues {
            assert_abs_diff_eq!(ev, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn diagonal_matrix_top_k() {
        // Diagonal matrix with eigenvalues 1..=10, verify top-3.
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        let mat = sparse_diag(&values);
        let result = eigsh(&mat, 3, WhichEigenvalues::LargestAlgebraic).unwrap();
        let mut evals: Vec<f64> = result.eigenvalues.to_vec();
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_abs_diff_eq!(evals[0], 10.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[1], 9.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[2], 8.0, epsilon = 1e-8);
    }

    #[test]
    fn diagonal_matrix_bottom_k() {
        // Same diagonal matrix, verify bottom-3.
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        let mat = sparse_diag(&values);
        let result = eigsh(&mat, 3, WhichEigenvalues::SmallestAlgebraic).unwrap();
        let mut evals: Vec<f64> = result.eigenvalues.to_vec();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(evals[0], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[1], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[2], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn symmetric_dense_matches_exact() {
        // A small 5×5 symmetric matrix with known eigenvalues.
        //
        //     [2  1  0  0  0]
        //     [1  2  1  0  0]
        // A = [0  1  2  1  0]
        //     [0  0  1  2  1]
        //     [0  0  0  1  2]
        //
        // Eigenvalues: 2 + 2*cos(k*pi/6) for k = 1..5
        let n = 5;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 2.0;
            if i + 1 < n {
                a[[i, i + 1]] = 1.0;
                a[[i + 1, i]] = 1.0;
            }
        }
        let mat = dense_to_sparse(&a);

        // Analytical eigenvalues for this tridiagonal Toeplitz matrix:
        // lambda_k = 2 + 2*cos(k*pi/(n+1)) for k = 1..n
        let mut expected: Vec<f64> = (1..=n)
            .map(|k| 2.0 + 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos())
            .collect();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let result = eigsh(&mat, 5, WhichEigenvalues::LargestAlgebraic).unwrap();
        let mut computed: Vec<f64> = result.eigenvalues.to_vec();
        computed.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for (c, e) in computed.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(c, e, epsilon = 1e-8);
        }
    }

    #[test]
    fn sparse_tridiagonal() {
        // Classic 1,-2,1 tridiagonal matrix of size 20.
        // Eigenvalues: -2 + 2*cos(k*pi/(n+1)) = -4*sin^2(k*pi/(2(n+1)))
        // for k = 1..n.
        let n = 20;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = -2.0;
            if i + 1 < n {
                a[[i, i + 1]] = 1.0;
                a[[i + 1, i]] = 1.0;
            }
        }
        let mat = dense_to_sparse(&a);

        // Compute top-5 eigenvalues (least negative).
        let result = eigsh(&mat, 5, WhichEigenvalues::LargestAlgebraic).unwrap();
        let mut computed: Vec<f64> = result.eigenvalues.to_vec();
        computed.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Analytical eigenvalues sorted descending.
        let mut expected: Vec<f64> = (1..=n)
            .map(|k| {
                -4.0 * (k as f64 * std::f64::consts::PI / (2.0 * (n as f64 + 1.0)))
                    .sin()
                    .powi(2)
            })
            .collect();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for i in 0..5 {
            assert_abs_diff_eq!(computed[i], expected[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn eigenvector_orthogonality() {
        // Verify that returned eigenvectors are orthonormal.
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        let mat = sparse_diag(&values);
        let result = eigsh(&mat, 5, WhichEigenvalues::LargestAlgebraic).unwrap();

        let k = result.eigenvectors.ncols();
        for i in 0..k {
            let vi = result.eigenvectors.column(i);
            // Check unit norm.
            let norm = vi.dot(&vi).sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
            // Check orthogonality with other eigenvectors.
            for j in (i + 1)..k {
                let vj = result.eigenvectors.column(j);
                let dot = vi.dot(&vj).abs();
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn matvec_closure_api() {
        // Verify the closure-based API works with a 6×6 diagonal matrix.
        let diag_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let n = diag_values.len();

        let solver = LanczosSolver::new(3).with_which(WhichEigenvalues::LargestAlgebraic);
        let result = solver
            .solve(n, |v: &Array1<f64>| {
                let mut out = Array1::<f64>::zeros(n);
                for i in 0..n {
                    out[i] = diag_values[i] * v[i];
                }
                out
            })
            .unwrap();

        let mut evals: Vec<f64> = result.eigenvalues.to_vec();
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_abs_diff_eq!(evals[0], 60.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[1], 50.0, epsilon = 1e-8);
        assert_abs_diff_eq!(evals[2], 40.0, epsilon = 1e-8);
    }
}
