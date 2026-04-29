//! Internal linear algebra utilities.
//!
//! This module provides helper functions for solving linear systems
//! using QR and Cholesky decompositions. The implementations convert
//! between `ndarray` arrays and `faer` matrices for computation.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Convert an `ndarray::Array2<F>` to a `faer::Mat<f64>`.
///
/// This is used internally when `F` is `f64` (the common case).
fn ndarray_to_faer_f64(a: &Array2<f64>) -> faer::Mat<f64> {
    let (nrows, ncols) = a.dim();
    faer::Mat::from_fn(nrows, ncols, |i, j| a[[i, j]])
}

/// Convert a `faer::Mat<f64>` column vector to an `ndarray::Array1<f64>`.
fn faer_col_to_ndarray_f64(col: &faer::Mat<f64>) -> Array1<f64> {
    let n = col.nrows();
    Array1::from_shape_fn(n, |i| col[(i, 0)])
}

/// Solve the least squares problem `X @ w = y` for `w`.
///
/// Uses QR decomposition for numerical stability. For `f64` data, this
/// delegates to `faer`'s highly optimized QR solver. For other float
/// types, a pure ndarray normal-equation solver is used.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the system is singular
/// or numerically ill-conditioned.
pub(crate) fn solve_lstsq<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let (n_samples, n_features) = x.dim();

    if n_samples < n_features {
        return Err(FerroError::InsufficientSamples {
            required: n_features,
            actual: n_samples,
            context: "need at least as many samples as features for least squares".into(),
        });
    }

    // Use faer QR decomposition for f64 (higher numerical accuracy).
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        // Convert to f64 arrays, solve with faer, convert back.
        let x_f64 = x.mapv(|v| v.to_f64().unwrap());
        let y_f64 = y.mapv(|v| v.to_f64().unwrap());
        let result = solve_lstsq_faer(&x_f64, &y_f64)?;
        return Ok(result.mapv(|v| F::from(v).unwrap()));
    }

    // Fallback for f32 and other float types: normal equations.
    solve_normal_equations(x, y)
}

/// Solve `X @ w = y` via the normal equations: `(X^T X) w = X^T y`.
///
/// Uses Cholesky decomposition of `X^T X` for efficiency. Falls back
/// to a direct solver if Cholesky fails (system may be ill-conditioned).
pub(crate) fn solve_normal_equations<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    let n = xtx.nrows();

    // Try Cholesky decomposition (X^T X should be positive semi-definite).
    match cholesky_solve(&xtx, &xty) {
        Ok(w) => Ok(w),
        Err(_) => {
            // Fallback: use LU-style Gaussian elimination.
            gaussian_solve(n, &xtx, &xty)
        }
    }
}

/// Solve a symmetric positive-definite system `A @ x = b` via Cholesky.
fn cholesky_solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError> {
    let n = a.nrows();

    // Compute lower triangular L such that A = L @ L^T.
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "matrix is not positive definite".into(),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L @ z = b
    let mut z = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * z[j];
        }
        z[i] = sum / l[[i, i]];
    }

    // Backward substitution: L^T @ x = z
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Solve `A @ x = b` via Gaussian elimination with partial pivoting.
fn gaussian_solve<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    // Augmented matrix [A | b].
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix encountered during Gaussian elimination".into(),
            });
        }

        // Swap rows.
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below.
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    // Back substitution.
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or_else(F::epsilon) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot during back substitution".into(),
            });
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Solve `(X^T X + alpha * I) @ w = X^T y` (Ridge regression).
///
/// Uses Cholesky decomposition since `X^T X + alpha * I` is guaranteed
/// to be positive definite for `alpha > 0`.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the regularized system
/// is somehow singular (should not happen for `alpha > 0`).
pub(crate) fn solve_ridge<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    alpha: F,
) -> Result<Array1<F>, FerroError> {
    let xt = x.t();
    let mut xtx = xt.dot(x);
    let xty = xt.dot(y);
    let n = xtx.nrows();

    // Add regularization: X^T X + alpha * I
    for i in 0..n {
        xtx[[i, i]] = xtx[[i, i]] + alpha;
    }

    cholesky_solve(&xtx, &xty).or_else(|_| gaussian_solve(n, &xtx, &xty))
}

/// Solve `X^T X w = X^T y` using faer QR decomposition (f64 only).
///
/// This provides the highest numerical accuracy for f64 data.
pub(crate) fn solve_lstsq_faer(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, FerroError> {
    use faer::linalg::solvers::SolveLstsq;

    let a = ndarray_to_faer_f64(x);
    let (n_samples, _n_features) = x.dim();
    let rhs = faer::Mat::from_fn(n_samples, 1, |i, _| y[i]);

    let qr = a.qr();
    let result = qr.solve_lstsq(rhs.as_ref());
    Ok(faer_col_to_ndarray_f64(&result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_lstsq_simple() {
        // 2x = 4 -> x = 2
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_lstsq_multi() {
        // y = x1 + 2*x2
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let w = solve_lstsq(&x, &y).unwrap();
        assert_relative_eq!(w[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(w[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_ridge() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_ridge(&x, &y, 0.0).unwrap();
        assert_relative_eq!(w[0], 2.0, epsilon = 1e-10);

        // With regularization, coefficients should shrink.
        let w_reg = solve_ridge(&x, &y, 10.0).unwrap();
        assert!(w_reg[0].abs() < w[0].abs());
    }

    #[test]
    fn test_solve_lstsq_faer() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let w = solve_lstsq_faer(&x, &y).unwrap();
        assert_relative_eq!(w[0], 2.0, epsilon = 1e-10);
    }
}
