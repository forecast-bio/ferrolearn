//! Hat matrix computation and LOOCV shortcut.
//!
//! For Nadaraya-Watson regression, the smoothing matrix `H` satisfies `ŷ = Hy`.
//! The diagonal `H_ii` gives leverage values. The O(n) LOOCV shortcut uses:
//!
//! `ŷ_{-i} = (ŷ_i - H_ii * y_i) / (1 - H_ii)`

use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::kernels::Kernel;
use crate::weights::compute_kernel_weights;

/// Compute the full hat matrix for Nadaraya-Watson regression.
///
/// `H_ij = w_ij / Σ_k w_ik` where `w_ij` are the kernel weights.
///
/// Returns a matrix of shape `(n, n)`.
pub fn compute_hat_matrix<F, K>(x: &Array2<F>, bandwidth: &Array1<F>, kernel: &K) -> Array2<F>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let raw_weights = compute_kernel_weights(x, x, bandwidth, kernel);
    let n = x.nrows();
    let mut hat = Array2::zeros((n, n));

    for i in 0..n {
        let row = raw_weights.row(i);
        let sum: F = row.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            for j in 0..n {
                hat[[i, j]] = row[j] / sum;
            }
        } else {
            let uniform = F::one() / F::from(n).unwrap();
            hat.row_mut(i).fill(uniform);
        }
    }

    hat
}

/// Extract the hat matrix diagonal (leverage values).
pub fn hat_matrix_diagonal<F, K>(x: &Array2<F>, bandwidth: &Array1<F>, kernel: &K) -> Array1<F>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let raw_weights = compute_kernel_weights(x, x, bandwidth, kernel);
    let n = x.nrows();
    let mut diag = Array1::zeros(n);

    for i in 0..n {
        let row = raw_weights.row(i);
        let sum: F = row.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            diag[i] = row[i] / sum;
        } else {
            diag[i] = F::one() / F::from(n).unwrap();
        }
    }

    diag
}

/// Effective degrees of freedom: `trace(H)`.
pub fn effective_df<F, K>(x: &Array2<F>, bandwidth: &Array1<F>, kernel: &K) -> F
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let diag = hat_matrix_diagonal(x, bandwidth, kernel);
    diag.sum()
}

/// O(n) LOOCV error using the hat matrix diagonal shortcut.
///
/// For Nadaraya-Watson, the LOOCV prediction at point i is:
///
/// `ŷ_{-i} = (ŷ_i - H_ii * y_i) / (1 - H_ii)`
///
/// Returns the mean squared LOOCV error.
pub fn loocv_hat_matrix_shortcut<F, K>(
    x: &Array2<F>,
    y: &Array1<F>,
    bandwidth: &Array1<F>,
    kernel: &K,
) -> F
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let hat = compute_hat_matrix(x, bandwidth, kernel);
    let n = x.nrows();
    let n_f = F::from(n).unwrap();

    // ŷ = H * y
    let y_hat = hat.dot(y);

    // Diagonal
    let h_diag: Array1<F> = (0..n).map(|i| hat[[i, i]]).collect();

    let eps = F::from(1e-10).unwrap();
    let mut sse = F::zero();

    for i in 0..n {
        let denom = F::one() - h_diag[i];
        let denom_safe = if denom.abs() > eps { denom } else { eps };
        let y_loo = (y_hat[i] - h_diag[i] * y[i]) / denom_safe;
        let err = y[i] - y_loo;
        sse = sse + err * err;
    }

    sse / n_f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::GaussianKernel;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn hat_matrix_rows_sum_to_one() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let bw = array![1.0f64];
        let h = compute_hat_matrix(&x, &bw, &GaussianKernel);
        for row in h.rows() {
            let sum: f64 = row.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn hat_matrix_diagonal_bounded() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let bw = array![1.0f64];
        let diag = hat_matrix_diagonal(&x, &bw, &GaussianKernel);
        for &h_ii in &diag {
            assert!(h_ii >= 0.0, "H_ii should be non-negative");
            assert!(h_ii <= 1.0, "H_ii should be <= 1");
        }
    }

    #[test]
    fn effective_df_bounded() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(f64::from).collect()).unwrap();
        let bw = array![1.0f64];
        let df = effective_df(&x, &bw, &GaussianKernel);
        assert!(df >= 1.0, "Effective DF should be >= 1");
        assert!(df <= 10.0, "Effective DF should be <= n");
    }

    #[test]
    fn loocv_shortcut_matches_naive() {
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let x = Array2::from_shape_vec((n, 1), x_data.clone()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let bw = array![1.0f64];

        // Shortcut LOOCV
        let mse_shortcut = loocv_hat_matrix_shortcut(&x, &y, &bw, &GaussianKernel);

        // Naive LOOCV (refit for each held-out point)
        let mut sse_naive = 0.0;
        for i in 0..n {
            let mut x_train_vec = Vec::with_capacity(n - 1);
            let mut y_train_vec = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j != i {
                    x_train_vec.push(x_data[j]);
                    y_train_vec.push(y[j]);
                }
            }
            let x_train = Array2::from_shape_vec((n - 1, 1), x_train_vec).unwrap();
            let y_train = Array1::from_vec(y_train_vec);
            let x_test = Array2::from_shape_vec((1, 1), vec![x_data[i]]).unwrap();

            let w = crate::weights::compute_kernel_weights(&x_test, &x_train, &bw, &GaussianKernel);
            let pred = crate::weights::nw_predict_from_weights(&w, &y_train);
            let err = y[i] - pred[0];
            sse_naive += err * err;
        }
        let mse_naive = sse_naive / n as f64;

        assert_abs_diff_eq!(mse_shortcut, mse_naive, epsilon = 1e-10);
    }

    #[test]
    fn loocv_small_bandwidth_high_error() {
        // Very small bandwidth => overfitting => high LOOCV error
        let x = Array2::from_shape_vec((10, 1), (0..10).map(f64::from).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let bw_small = array![0.01f64];
        let bw_good = array![1.5f64];

        let mse_small = loocv_hat_matrix_shortcut(&x, &y, &bw_small, &GaussianKernel);
        let mse_good = loocv_hat_matrix_shortcut(&x, &y, &bw_good, &GaussianKernel);

        // Small bandwidth should have higher LOOCV error (overfitting)
        assert!(
            mse_small > mse_good,
            "Small bw MSE {mse_small} should exceed good bw MSE {mse_good}"
        );
    }

    #[test]
    fn df_increases_with_smaller_bandwidth() {
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| f64::from(i) * 0.5).collect()).unwrap();
        let df_small_bw = effective_df(&x, &array![0.5f64], &GaussianKernel);
        let df_large_bw = effective_df(&x, &array![5.0f64], &GaussianKernel);

        assert!(
            df_small_bw > df_large_bw,
            "Smaller bw should give higher DF: {df_small_bw} vs {df_large_bw}"
        );
    }
}
