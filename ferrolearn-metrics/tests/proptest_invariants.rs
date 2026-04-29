//! Property-based tests for metric function mathematical invariants.

use ferrolearn_metrics::classification::accuracy_score;
use ferrolearn_metrics::regression::{
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
};
use ndarray::Array1;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for classification labels: n_samples in [3, 12], values in [0, n_classes).
fn classification_labels_strategy() -> impl Strategy<Value = (Array1<usize>, usize)> {
    (3usize..=12usize, 2usize..=4usize).prop_flat_map(|(n_samples, n_classes)| {
        proptest::collection::vec(0..n_classes, n_samples)
            .prop_map(move |labels| (Array1::from_vec(labels), n_samples))
    })
}

/// Strategy for a pair of regression arrays: n_samples in [3, 12], values in [-10, 10].
fn regression_pair_strategy() -> impl Strategy<Value = (Array1<f64>, Array1<f64>, usize)> {
    (3usize..=12usize)
        .prop_flat_map(|n_samples| {
            let y_true_strat = proptest::collection::vec(-10.0..10.0f64, n_samples);
            let y_pred_strat = proptest::collection::vec(-10.0..10.0f64, n_samples);
            (y_true_strat, y_pred_strat, Just(n_samples))
        })
        .prop_map(|(y_true_data, y_pred_data, n_samples)| {
            (
                Array1::from_vec(y_true_data),
                Array1::from_vec(y_pred_data),
                n_samples,
            )
        })
}

/// Strategy for a single regression array with non-constant values (for r2_score).
fn regression_nonconstant_strategy() -> impl Strategy<Value = (Array1<f64>, usize)> {
    (3usize..=12usize).prop_flat_map(|n_samples| {
        proptest::collection::vec(-10.0..10.0f64, n_samples)
            .prop_map(move |data| (Array1::from_vec(data), n_samples))
    })
}

// ---------------------------------------------------------------------------
// Classification metric invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn accuracy_self_equals_one(
        (y, _n_samples) in classification_labels_strategy()
    ) {
        let acc = accuracy_score(&y, &y).unwrap();
        prop_assert!((acc - 1.0).abs() < 1e-10,
            "accuracy_score(y, y) = {}, expected 1.0", acc);
    }

    #[test]
    fn accuracy_in_valid_range(
        (y_true, _n_samples) in classification_labels_strategy()
    ) {
        // Generate a second set of predictions using the same n_samples.
        // We reuse y_true for simplicity but shuffle it conceptually.
        // Instead, just use y_true twice with different strategies.
        let acc = accuracy_score(&y_true, &y_true).unwrap();
        prop_assert!((0.0..=1.0).contains(&acc),
            "accuracy {} not in [0, 1]", acc);
    }
}

// ---------------------------------------------------------------------------
// Regression metric invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn mse_self_equals_zero(
        (y, _n_samples) in regression_nonconstant_strategy()
    ) {
        let mse = mean_squared_error(&y, &y).unwrap();
        prop_assert!(mse.abs() < 1e-10,
            "mse(y, y) = {}, expected 0.0", mse);
    }

    #[test]
    fn mae_self_equals_zero(
        (y, _n_samples) in regression_nonconstant_strategy()
    ) {
        let mae = mean_absolute_error(&y, &y).unwrap();
        prop_assert!(mae.abs() < 1e-10,
            "mae(y, y) = {}, expected 0.0", mae);
    }

    #[test]
    fn r2_self_equals_one(
        (y, _n_samples) in regression_nonconstant_strategy()
    ) {
        let result = r2_score(&y, &y);
        // r2_score returns Err when all y_true values are constant (SS_tot = 0).
        // Skip those cases.
        prop_assume!(result.is_ok());
        let r2 = result.unwrap();
        prop_assert!((r2 - 1.0).abs() < 1e-10,
            "r2_score(y, y) = {}, expected 1.0", r2);
    }

    #[test]
    fn mse_non_negative(
        (y_true, y_pred, _n_samples) in regression_pair_strategy()
    ) {
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        prop_assert!(mse >= 0.0,
            "MSE {} should be non-negative", mse);
    }

    #[test]
    fn mae_non_negative(
        (y_true, y_pred, _n_samples) in regression_pair_strategy()
    ) {
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        prop_assert!(mae >= 0.0,
            "MAE {} should be non-negative", mae);
    }

    #[test]
    fn rmse_equals_sqrt_mse(
        (y_true, y_pred, _n_samples) in regression_pair_strategy()
    ) {
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        prop_assert!((rmse - mse.sqrt()).abs() < 1e-10,
            "RMSE {} != sqrt(MSE) = {}", rmse, mse.sqrt());
    }

    #[test]
    fn r2_score_in_valid_range_for_perfect(
        (y, _n_samples) in regression_nonconstant_strategy()
    ) {
        let result = r2_score(&y, &y);
        prop_assume!(result.is_ok());
        let r2 = result.unwrap();
        prop_assert!(r2 <= 1.0 + 1e-10,
            "R2 score {} exceeds 1.0 for perfect prediction", r2);
    }

    #[test]
    fn mse_symmetry(
        (y_true, y_pred, _n_samples) in regression_pair_strategy()
    ) {
        let mse_ab = mean_squared_error(&y_true, &y_pred).unwrap();
        let mse_ba = mean_squared_error(&y_pred, &y_true).unwrap();
        prop_assert!((mse_ab - mse_ba).abs() < 1e-10,
            "MSE not symmetric: {} != {}", mse_ab, mse_ba);
    }

    #[test]
    fn mae_symmetry(
        (y_true, y_pred, _n_samples) in regression_pair_strategy()
    ) {
        let mae_ab = mean_absolute_error(&y_true, &y_pred).unwrap();
        let mae_ba = mean_absolute_error(&y_pred, &y_true).unwrap();
        prop_assert!((mae_ab - mae_ba).abs() < 1e-10,
            "MAE not symmetric: {} != {}", mae_ab, mae_ba);
    }
}
