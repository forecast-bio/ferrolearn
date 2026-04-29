//! Property-based tests for kernel regression invariants.
//!
//! Verifies that structural invariants hold across randomly generated data.

use ndarray::{Array1, Array2};
use proptest::prelude::*;

use ferrolearn_kernel::kernels::GaussianKernel;
use ferrolearn_kernel::weights;

/// Generate a random 1D dataset with n points in [0, range].
fn arb_dataset_1d(
    n_range: std::ops::Range<usize>,
    x_range: f64,
) -> impl Strategy<Value = (Array2<f64>, Array1<f64>)> {
    n_range.prop_flat_map(move |n| {
        (
            proptest::collection::vec(0.0..x_range, n),
            proptest::collection::vec(-10.0..10.0f64, n),
        )
            .prop_map(move |(x_vec, y_vec)| {
                let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
                let y = Array1::from_vec(y_vec);
                (x, y)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Invariant 1: Row permutation invariance.
    ///
    /// Predictions should not depend on the ordering of training data.
    #[test]
    fn symmetry_permutation_invariant(
        (x, y) in arb_dataset_1d(10..30, 5.0),
        _seed in 0u64..1000,
    ) {
        let bw = ndarray::array![0.5f64];
        let w1 = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred1 = weights::nw_predict_from_weights(&w1, &y);

        // Reverse the training data
        let n = x.nrows();
        let mut x_rev_data: Vec<f64> = Vec::with_capacity(n);
        let mut y_rev_data: Vec<f64> = Vec::with_capacity(n);
        for i in (0..n).rev() {
            x_rev_data.push(x[[i, 0]]);
            y_rev_data.push(y[i]);
        }
        let x_rev = Array2::from_shape_vec((n, 1), x_rev_data).unwrap();
        let y_rev = Array1::from_vec(y_rev_data);

        let w2 = weights::compute_kernel_weights(&x, &x_rev, &bw, &GaussianKernel);
        let pred2 = weights::nw_predict_from_weights(&w2, &y_rev);

        for i in 0..n {
            let diff = (pred1[i] - pred2[i]).abs();
            prop_assert!(
                diff < 1e-10,
                "Permutation changed prediction at {i}: {:.8} vs {:.8}",
                pred1[i], pred2[i]
            );
        }
    }

    /// Invariant 2: Small bandwidth → interpolation.
    ///
    /// With very small bandwidth and distinct x values, NW should nearly
    /// interpolate training data.
    #[test]
    fn interpolation_small_bandwidth(
        n in 5usize..20,
        y_vals in proptest::collection::vec(-10.0..10.0f64, 5..20),
    ) {
        let n = n.min(y_vals.len());
        // Use evenly spaced distinct x values
        let x_vec: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
        let y = Array1::from_vec(y_vals[..n].to_vec());

        let bw = ndarray::array![0.001f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        for i in 0..n {
            let diff = (pred[i] - y[i]).abs();
            prop_assert!(
                diff < 0.1,
                "Small bandwidth should interpolate: pred={:.6}, y={:.6} at i={i}",
                pred[i], y[i]
            );
        }
    }

    /// Invariant 3: Large bandwidth → mean convergence.
    ///
    /// With very large bandwidth, NW should predict close to the mean of y.
    #[test]
    fn large_bandwidth_converges_to_mean(
        (x, y) in arb_dataset_1d(10..30, 5.0),
    ) {
        let bw = ndarray::array![1000.0f64];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let y_mean = y.mean().unwrap();

        for i in 0..x.nrows() {
            let diff = (pred[i] - y_mean).abs();
            prop_assert!(
                diff < 0.1,
                "Large bandwidth should predict near mean: pred={:.6}, mean={:.6} at i={i}",
                pred[i], y_mean
            );
        }
    }

    /// Invariant 4: Non-negativity of weights.
    ///
    /// All kernel weights must be >= 0.
    #[test]
    fn weights_nonnegative(
        (x, _y) in arb_dataset_1d(5..20, 5.0),
        bw_val in 0.01..10.0f64,
    ) {
        let bw = ndarray::array![bw_val];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);

        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                prop_assert!(
                    w[[i, j]] >= 0.0,
                    "Weight [{i},{j}] = {} is negative", w[[i, j]]
                );
            }
        }
    }

    /// Invariant 5: NW predictions bounded by [min(y), max(y)].
    ///
    /// Since NW is a convex combination of y values, predictions
    /// must stay within the range of training targets.
    #[test]
    fn nw_prediction_bounded(
        (x, y) in arb_dataset_1d(5..30, 5.0),
        bw_val in 0.1..5.0f64,
    ) {
        let bw = ndarray::array![bw_val];
        let w = weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel);
        let pred = weights::nw_predict_from_weights(&w, &y);

        let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for i in 0..x.nrows() {
            prop_assert!(
                pred[i] >= y_min - 1e-10 && pred[i] <= y_max + 1e-10,
                "Prediction {:.6} outside [{:.6}, {:.6}] at i={i}",
                pred[i], y_min, y_max
            );
        }
    }

    /// Invariant 7: Bandwidth monotonicity for effective DF.
    ///
    /// Smaller bandwidth → higher effective DF (more complex fit).
    #[test]
    fn bandwidth_monotonicity_df(
        (x, _y) in arb_dataset_1d(10..30, 5.0),
        bw_small in 0.1..1.0f64,
    ) {
        let bw_large = bw_small * 5.0;
        let bw_s = ndarray::array![bw_small];
        let bw_l = ndarray::array![bw_large];

        let df_small = ferrolearn_kernel::hat_matrix::effective_df(&x, &bw_s, &GaussianKernel);
        let df_large = ferrolearn_kernel::hat_matrix::effective_df(&x, &bw_l, &GaussianKernel);

        prop_assert!(
            df_small >= df_large - 1e-10,
            "Smaller bw should give >= DF: small_bw={bw_small:.3} df={df_small:.3}, large_bw={bw_large:.3} df={df_large:.3}"
        );
    }

    /// Invariant 10: Hat matrix trace bounded by [1, n].
    #[test]
    fn hat_matrix_trace_bounded(
        (x, _y) in arb_dataset_1d(5..25, 5.0),
        bw_val in 0.1..5.0f64,
    ) {
        let bw = ndarray::array![bw_val];
        let df = ferrolearn_kernel::hat_matrix::effective_df(&x, &bw, &GaussianKernel);
        let n = x.nrows() as f64;

        prop_assert!(
            df >= 0.9 && df <= n + 0.1,
            "Effective DF {df:.3} outside [1, {n}] for bw={bw_val:.3}"
        );
    }
}
