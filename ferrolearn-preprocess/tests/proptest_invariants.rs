//! Property-based tests for StandardScaler mathematical invariants.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::StandardScaler;
use ndarray::Array2;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for generating a random data matrix suitable for StandardScaler.
/// At least 3 rows (to have non-trivial statistics) and 2-4 columns.
fn data_matrix_strategy() -> impl Strategy<Value = (Array2<f64>, usize, usize)> {
    (3usize..=10usize, 2usize..=4usize).prop_flat_map(|(n_rows, n_cols)| {
        proptest::collection::vec(-10.0..10.0f64, n_rows * n_cols).prop_map(move |data| {
            (
                Array2::from_shape_vec((n_rows, n_cols), data).unwrap(),
                n_rows,
                n_cols,
            )
        })
    })
}

// ---------------------------------------------------------------------------
// StandardScaler invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn standard_scaler_inverse_transform_roundtrip(
        (x, _n_rows, _n_cols) in data_matrix_strategy()
    ) {
        let scaler = StandardScaler::<f64>::new();
        let result = scaler.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();

        let scaled = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&scaled).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            prop_assert!((a - b).abs() < 1e-8,
                "Roundtrip failed: original {} != recovered {}", a, b);
        }
    }

    #[test]
    fn standard_scaler_transformed_mean_near_zero(
        (x, n_rows, n_cols) in data_matrix_strategy()
    ) {
        let scaler = StandardScaler::<f64>::new();
        let result = scaler.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let scaled = fitted.transform(&x).unwrap();

        for j in 0..n_cols {
            // Check if the column has non-zero std
            if fitted.std()[j] == 0.0 {
                continue; // Zero-variance columns are left unchanged
            }
            let col_mean: f64 = scaled.column(j).sum() / n_rows as f64;
            prop_assert!(col_mean.abs() < 1e-8,
                "Column {} mean {} should be near 0", j, col_mean);
        }
    }

    #[test]
    fn standard_scaler_transformed_std_near_one(
        (x, n_rows, n_cols) in data_matrix_strategy()
    ) {
        let scaler = StandardScaler::<f64>::new();
        let result = scaler.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let scaled = fitted.transform(&x).unwrap();

        for j in 0..n_cols {
            if fitted.std()[j] == 0.0 {
                continue; // Zero-variance columns are left unchanged
            }
            let col_mean: f64 = scaled.column(j).sum() / n_rows as f64;
            let variance: f64 = scaled.column(j)
                .iter()
                .map(|&v| (v - col_mean).powi(2))
                .sum::<f64>() / n_rows as f64;
            let std_dev = variance.sqrt();
            prop_assert!((std_dev - 1.0).abs() < 1e-6,
                "Column {} std {} should be near 1.0", j, std_dev);
        }
    }

    #[test]
    fn standard_scaler_transform_preserves_shape(
        (x, n_rows, n_cols) in data_matrix_strategy()
    ) {
        let scaler = StandardScaler::<f64>::new();
        let result = scaler.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let scaled = fitted.transform(&x).unwrap();

        prop_assert_eq!(scaled.nrows(), n_rows);
        prop_assert_eq!(scaled.ncols(), n_cols);
    }
}
