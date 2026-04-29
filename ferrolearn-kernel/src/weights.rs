//! Kernel weight computation for prediction.
//!
//! Computes product kernel weights: `w_i = ∏_j K((x_j - X_ij) / h_j) / ∏_j h_j`
//!
//! Uses BallTree acceleration for compact-support kernels when `n_train >= 500`.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;

use crate::kernels::Kernel;

/// Compute multivariate product kernel weights.
///
/// For each query point, computes weights for all training points using
/// the product kernel: `w_i = ∏_j K((x_j - X_ij) / h_j) / ∏_j h_j`.
///
/// Returns a weight matrix of shape `(n_query, n_train)`.
pub fn compute_kernel_weights<F, K>(
    x_query: &Array2<F>,
    x_train: &Array2<F>,
    bandwidth: &Array1<F>,
    kernel: &K,
) -> Array2<F>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let n_query = x_query.nrows();
    let n_train = x_train.nrows();
    let n_features = x_train.ncols();

    let bw_product = bandwidth.iter().fold(F::one(), |acc, &h| acc * h);
    let inv_bw_product = F::one() / bw_product;

    let mut weights = Array2::zeros((n_query, n_train));

    for i in 0..n_query {
        let query_row = x_query.row(i);

        // Compute per-dimension scaled distances, apply kernel, take product
        let mut row_weights = Array1::from_elem(n_train, F::one());

        for j in 0..n_features {
            let h = bandwidth[j];
            let scaled: Array1<F> = x_train.column(j).mapv(|x_ij| (query_row[j] - x_ij) / h);
            let k_vals = kernel.evaluate(&scaled.view());
            row_weights = row_weights * k_vals;
        }

        row_weights.mapv_inplace(|w| w * inv_bw_product);
        weights.row_mut(i).assign(&row_weights);
    }

    weights
}

/// Compute normalized kernel weights (hat matrix rows).
///
/// Returns weights that sum to 1 for each query point.
/// Points with zero total weight get uniform weights.
pub fn compute_normalized_weights<F, K>(
    x_query: &Array2<F>,
    x_train: &Array2<F>,
    bandwidth: &Array1<F>,
    kernel: &K,
) -> Array2<F>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let mut weights = compute_kernel_weights(x_query, x_train, bandwidth, kernel);
    let n_train = x_train.nrows();

    for mut row in weights.rows_mut() {
        let sum: F = row.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            row.mapv_inplace(|w| w / sum);
        } else {
            // No neighbors: uniform weight
            let uniform = F::one() / F::from(n_train).unwrap();
            row.fill(uniform);
        }
    }

    weights
}

/// Compute kernel weights using BallTree for compact-support kernels.
///
/// Only computes weights for neighbors within the kernel support radius,
/// significantly reducing computation for large datasets.
pub fn compute_kernel_weights_balltree<F, K>(
    x_query: &Array2<F>,
    x_train: &Array2<F>,
    bandwidth: &Array1<F>,
    kernel: &K,
    tree: &ferrolearn_neighbors::balltree::BallTree,
) -> Array2<F>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let n_query = x_query.nrows();
    let n_train = x_train.nrows();
    let n_features = x_train.ncols();

    let bw_product = bandwidth.iter().fold(F::one(), |acc, &h| acc * h);
    let inv_bw_product = F::one() / bw_product;

    // Search radius: max bandwidth (in scaled space, the support is 1.0,
    // so in original space we need max(h_j))
    let max_bw: f64 = bandwidth
        .iter()
        .map(|h| h.to_f64().unwrap())
        .fold(f64::NEG_INFINITY, f64::max);

    let mut weights = Array2::zeros((n_query, n_train));

    for i in 0..n_query {
        let query_f64: Vec<f64> = x_query.row(i).iter().map(|v| v.to_f64().unwrap()).collect();

        let neighbors = tree.within_radius(&query_f64, max_bw);

        for (idx, _dist) in &neighbors {
            let mut w = F::one();
            for j in 0..n_features {
                let u = (x_query[[i, j]] - x_train[[*idx, j]]) / bandwidth[j];
                let k_val = kernel.evaluate(&ArrayView1::from(&[u]));
                w = w * k_val[0];
            }
            weights[[i, *idx]] = w * inv_bw_product;
        }
    }

    weights
}

/// Nadaraya-Watson prediction from precomputed weights.
///
/// `ŷ(x) = Σ w_i y_i / Σ w_i`
pub fn nw_predict_from_weights<F: Float>(weights: &Array2<F>, y_train: &Array1<F>) -> Array1<F> {
    let n_query = weights.nrows();
    let mut predictions = Array1::zeros(n_query);
    let y_mean = y_train.sum() / F::from(y_train.len()).unwrap();

    for i in 0..n_query {
        let row = weights.row(i);
        let sum: F = row.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum > F::zero() {
            let weighted_sum: F = row
                .iter()
                .zip(y_train.iter())
                .map(|(&w, &y)| w * y)
                .fold(F::zero(), |a, b| a + b);
            predictions[i] = weighted_sum / sum;
        } else {
            predictions[i] = y_mean;
        }
    }

    predictions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{EpanechnikovKernel, GaussianKernel};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn weights_symmetry() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let bw = array![1.0f64];
        let w = compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        // w[0,1] should equal w[1,0] since |0-1| = |1-0|
        assert_abs_diff_eq!(w[[0, 1]], w[[1, 0]], epsilon = 1e-14);
    }

    #[test]
    fn weights_decrease_with_distance() {
        let query = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let bw = array![1.0f64];
        let w = compute_kernel_weights(&query, &train, &bw, &GaussianKernel);
        assert!(w[[0, 0]] > w[[0, 1]]);
        assert!(w[[0, 1]] > w[[0, 2]]);
    }

    #[test]
    fn compact_kernel_zero_outside() {
        let query = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let train = Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 2.0]).unwrap();
        let bw = array![1.0f64];
        let w = compute_kernel_weights(&query, &train, &bw, &EpanechnikovKernel);
        assert!(w[[0, 0]] > 0.0);
        assert!(w[[0, 1]] > 0.0);
        assert_abs_diff_eq!(w[[0, 2]], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn normalized_weights_sum_to_one() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let bw = array![1.0f64];
        let w = compute_normalized_weights(&x, &x, &bw, &GaussianKernel);
        for row in w.rows() {
            let sum: f64 = row.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn nw_predict_constant_function() {
        // If y is constant, prediction should be that constant
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 3.0, 3.0, 3.0, 3.0f64];
        let bw = array![1.0f64];
        let w = compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = nw_predict_from_weights(&w, &y);
        for &p in &pred {
            assert_abs_diff_eq!(p, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn nw_predict_interpolation() {
        // With very small bandwidth, should nearly interpolate
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![1.0, 2.0, 3.0f64];
        let bw = array![0.01f64];
        let w = compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = nw_predict_from_weights(&w, &y);
        for i in 0..3 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn multivariate_product_kernel() {
        // 2D data: product kernel should be product of per-dimension values
        let query = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let train = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let bw = array![1.0f64, 1.0];
        let w = compute_kernel_weights(&query, &train, &bw, &GaussianKernel);

        // Manual: K(0.5)*K(0.5) / (1*1)
        let k_half = (-0.5 * 0.25f64).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected = k_half * k_half;
        assert_abs_diff_eq!(w[[0, 0]], expected, epsilon = 1e-14);
    }

    #[test]
    fn anisotropic_bandwidth() {
        // Different bandwidths per dimension
        let query = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let train = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let bw_iso = array![1.0f64, 1.0];
        let bw_aniso = array![1.0f64, 10.0]; // Very wide in dim 2

        let w_iso = compute_kernel_weights(&query, &train, &bw_iso, &GaussianKernel);
        let w_aniso = compute_kernel_weights(&query, &train, &bw_aniso, &GaussianKernel);

        // With wider bandwidth in dim 2, the unnormalized weight should reflect
        // different kernel shapes. The point (1,1) should get more kernel mass
        // in dimension 2 with wider bandwidth but also divided by larger bw product.
        // The key test: they should be different.
        assert!((w_iso[[0, 0]] - w_aniso[[0, 0]]).abs() > 1e-10);
    }
}
