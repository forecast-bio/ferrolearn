//! Oracle tests that compare ferrolearn preprocessing transformers against
//! scikit-learn reference outputs stored in `fixtures/*.json`.

use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};

/// Helper: parse a JSON nested array into an `Array2<f64>`.
fn json_to_array2(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

/// Helper: parse a JSON flat array into an `Array1<f64>`.
fn json_to_array1(value: &serde_json::Value) -> Array1<f64> {
    let vec: Vec<f64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Array1::from_vec(vec)
}

/// Helper: assert two f64 slices are element-wise equal within `epsilon`,
/// with an informative panic message on failure.
fn assert_array_close(actual: &[f64], expected: &[f64], epsilon: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let denom = e.abs().max(1.0);
        assert!(
            diff / denom <= epsilon,
            "{label}[{i}]: actual={a}, expected={e}, rel_diff={rel}",
            rel = diff / denom,
        );
    }
}

// ---------------------------------------------------------------------------
// StandardScaler oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_standard_scaler_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/standard_scaler.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    let expected_mean = json_to_array1(&fixture["expected"]["mean"]);
    let expected_std = json_to_array1(&fixture["expected"]["std"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let scaler = ferrolearn_preprocess::StandardScaler::<f64>::new();
    let fitted = scaler.fit(&x, &()).unwrap();

    // Compare learned mean.
    assert_array_close(
        fitted.mean().as_slice().unwrap(),
        expected_mean.as_slice().unwrap(),
        1e-10,
        "StandardScaler mean",
    );

    // Compare learned std.
    assert_array_close(
        fitted.std().as_slice().unwrap(),
        expected_std.as_slice().unwrap(),
        1e-10,
        "StandardScaler std",
    );

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "StandardScaler transformed",
    );
}

// ---------------------------------------------------------------------------
// MinMaxScaler oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_minmax_scaler_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/minmax_scaler.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    let expected_data_min = json_to_array1(&fixture["expected"]["data_min"]);
    let expected_data_max = json_to_array1(&fixture["expected"]["data_max"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let scaler = ferrolearn_preprocess::MinMaxScaler::<f64>::new();
    let fitted = scaler.fit(&x, &()).unwrap();

    // Compare learned data_min.
    assert_array_close(
        fitted.data_min().as_slice().unwrap(),
        expected_data_min.as_slice().unwrap(),
        1e-10,
        "MinMaxScaler data_min",
    );

    // Compare learned data_max.
    assert_array_close(
        fitted.data_max().as_slice().unwrap(),
        expected_data_max.as_slice().unwrap(),
        1e-10,
        "MinMaxScaler data_max",
    );

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "MinMaxScaler transformed",
    );
}

// ---------------------------------------------------------------------------
// RobustScaler oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_robust_scaler_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/robust_scaler.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    let expected_center = json_to_array1(&fixture["expected"]["center"]);
    let expected_scale = json_to_array1(&fixture["expected"]["scale"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let scaler = ferrolearn_preprocess::RobustScaler::<f64>::new();
    let fitted = scaler.fit(&x, &()).unwrap();

    // Compare learned center (median).
    assert_array_close(
        fitted.median().as_slice().unwrap(),
        expected_center.as_slice().unwrap(),
        1e-10,
        "RobustScaler center",
    );

    // Compare learned scale (IQR).
    assert_array_close(
        fitted.iqr().as_slice().unwrap(),
        expected_scale.as_slice().unwrap(),
        1e-10,
        "RobustScaler scale",
    );

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "RobustScaler transformed",
    );
}
