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

// ---------------------------------------------------------------------------
// Helper: parse a JSON nested array into an `Array2<f64>`, treating null as NaN.
// ---------------------------------------------------------------------------

fn json_to_array2_nan(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    if v.is_null() {
                        f64::NAN
                    } else {
                        v.as_f64().unwrap()
                    }
                })
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

/// Helper: parse a JSON nested array into an `Array2<usize>`.
fn json_to_array2_usize(value: &serde_json::Value) -> Array2<usize> {
    let rows: Vec<Vec<usize>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<usize> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

// ---------------------------------------------------------------------------
// MaxAbsScaler oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_max_abs_scaler_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/max_abs_scaler.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    let expected_max_abs = json_to_array1(&fixture["expected"]["max_abs"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let scaler = ferrolearn_preprocess::MaxAbsScaler::<f64>::new();
    let fitted = scaler.fit(&x, &()).unwrap();

    // Compare learned max_abs.
    assert_array_close(
        fitted.max_abs().as_slice().unwrap(),
        expected_max_abs.as_slice().unwrap(),
        1e-10,
        "MaxAbsScaler max_abs",
    );

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "MaxAbsScaler transformed",
    );
}

// ---------------------------------------------------------------------------
// Normalizer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_normalizer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/normalizer.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    // Normalizer is stateless; use L2 norm as specified in the fixture.
    let normalizer = ferrolearn_preprocess::Normalizer::<f64>::new(
        ferrolearn_preprocess::normalizer::NormType::L2,
    );

    // Compare transformed output.
    let transformed = normalizer.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "Normalizer transformed",
    );
}

// ---------------------------------------------------------------------------
// Binarizer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_binarizer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/binarizer.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let threshold = fixture["params"]["threshold"].as_f64().unwrap();
    let binarizer = ferrolearn_preprocess::Binarizer::<f64>::new(threshold);

    // Compare transformed output — exact match since output is 0.0 or 1.0.
    let transformed = binarizer.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_eq!(
        actual_flat.len(),
        expected_flat.len(),
        "Binarizer: length mismatch"
    );
    for (i, (&a, &e)) in actual_flat.iter().zip(expected_flat.iter()).enumerate() {
        assert!(a == e, "Binarizer[{i}]: actual={a}, expected={e}",);
    }
}

// ---------------------------------------------------------------------------
// PolynomialFeatures oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_polynomial_features_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/polynomial_features.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let expected_n_output = fixture["expected"]["n_output_features"].as_u64().unwrap() as usize;
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let degree = fixture["params"]["degree"].as_u64().unwrap() as usize;
    let interaction_only = fixture["params"]["interaction_only"].as_bool().unwrap();
    let include_bias = fixture["params"]["include_bias"].as_bool().unwrap();
    let poly = ferrolearn_preprocess::PolynomialFeatures::<f64>::new(
        degree,
        interaction_only,
        include_bias,
    )
    .unwrap();

    // Compare transformed output.
    let transformed = poly.transform(&x).unwrap();

    // Check shape.
    assert_eq!(
        transformed.ncols(),
        expected_n_output,
        "PolynomialFeatures: n_output_features mismatch"
    );
    assert_eq!(
        transformed.shape(),
        expected_transformed.shape(),
        "PolynomialFeatures: shape mismatch"
    );

    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "PolynomialFeatures transformed",
    );
}

// ---------------------------------------------------------------------------
// OneHotEncoder oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_one_hot_encoder_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/one_hot_encoder.json")).unwrap();

    let x = json_to_array2_usize(&fixture["input"]["X"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let enc = ferrolearn_preprocess::OneHotEncoder::<f64>::new();
    let fitted = enc.fit(&x, &()).unwrap();

    // Compare transformed output — exact match for binary encoding.
    let transformed = fitted.transform(&x).unwrap();
    assert_eq!(
        transformed.shape(),
        expected_transformed.shape(),
        "OneHotEncoder: shape mismatch"
    );

    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    for (i, (&a, &e)) in actual_flat.iter().zip(expected_flat.iter()).enumerate() {
        assert!(a == e, "OneHotEncoder[{i}]: actual={a}, expected={e}",);
    }
}

// ---------------------------------------------------------------------------
// LabelEncoder oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_label_encoder_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/label_encoder.json")).unwrap();

    let labels_raw: Vec<String> = fixture["input"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let labels = Array1::from_vec(labels_raw);

    let expected_classes: Vec<String> = fixture["expected"]["classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let expected_transformed: Vec<usize> = fixture["expected"]["transformed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let enc = ferrolearn_preprocess::LabelEncoder::new();
    let fitted = enc.fit(&labels, &()).unwrap();

    // Compare learned classes — exact match.
    assert_eq!(
        fitted.classes(),
        expected_classes.as_slice(),
        "LabelEncoder: classes mismatch"
    );

    // Compare transformed output — exact match.
    let transformed = fitted.transform(&labels).unwrap();
    let actual: Vec<usize> = transformed.iter().copied().collect();
    assert_eq!(
        actual, expected_transformed,
        "LabelEncoder: transformed mismatch"
    );
}

// ---------------------------------------------------------------------------
// QuantileTransformer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_quantile_transformer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/quantile_transformer.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let n_quantiles = fixture["params"]["n_quantiles"].as_u64().unwrap() as usize;
    let qt = ferrolearn_preprocess::QuantileTransformer::<f64>::new(
        n_quantiles,
        ferrolearn_preprocess::OutputDistribution::Uniform,
        0,
    );
    let fitted = qt.fit(&x, &()).unwrap();

    // Compare transformed output with tolerance for quantile interpolation differences.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "QuantileTransformer transformed",
    );
}

// ---------------------------------------------------------------------------
// KBinsDiscretizer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_kbins_discretizer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/kbins_discretizer.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let n_bins = fixture["params"]["n_bins"].as_u64().unwrap() as usize;
    let disc = ferrolearn_preprocess::KBinsDiscretizer::<f64>::new(
        n_bins,
        ferrolearn_preprocess::BinEncoding::Ordinal,
        ferrolearn_preprocess::BinStrategy::Uniform,
    );
    let fitted = disc.fit(&x, &()).unwrap();

    // Compare bin_edges — verify they exist and have correct count per feature.
    let expected_bin_edges = fixture["expected"]["bin_edges"].as_array().unwrap();
    let actual_bin_edges = fitted.bin_edges();
    assert_eq!(
        actual_bin_edges.len(),
        expected_bin_edges.len(),
        "KBinsDiscretizer: number of feature edge vectors mismatch"
    );
    for (j, (actual_edges, expected_edges)) in actual_bin_edges
        .iter()
        .zip(expected_bin_edges.iter())
        .enumerate()
    {
        let exp_vec: Vec<f64> = expected_edges
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(
            actual_edges.len(),
            exp_vec.len(),
            "KBinsDiscretizer feature {j}: edge count mismatch"
        );
        let actual_slice: Vec<f64> = actual_edges.to_vec();
        assert_array_close(
            &actual_slice,
            &exp_vec,
            1e-10,
            &format!("KBinsDiscretizer bin_edges[{j}]"),
        );
    }

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "KBinsDiscretizer transformed",
    );
}

// ---------------------------------------------------------------------------
// SimpleImputer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_simple_imputer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/simple_imputer.json")).unwrap();

    // Parse input X with null → NaN handling.
    let x = json_to_array2_nan(&fixture["input"]["X"]);

    let expected_statistics = json_to_array1(&fixture["expected"]["statistics"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let imputer = ferrolearn_preprocess::SimpleImputer::<f64>::new(
        ferrolearn_preprocess::ImputeStrategy::Mean,
    );
    let fitted = imputer.fit(&x, &()).unwrap();

    // Compare learned statistics (column means).
    assert_array_close(
        fitted.fill_values().as_slice().unwrap(),
        expected_statistics.as_slice().unwrap(),
        1e-10,
        "SimpleImputer statistics",
    );

    // Compare transformed output (NaN filled).
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-10,
        "SimpleImputer transformed",
    );
}

// ---------------------------------------------------------------------------
// PowerTransformer oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_power_transformer_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/power_transformer.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    let expected_lambdas = json_to_array1(&fixture["expected"]["lambdas"]);
    let expected_transformed = json_to_array2(&fixture["expected"]["transformed"]);

    let pt = ferrolearn_preprocess::PowerTransformer::<f64>::new();
    let fitted = pt.fit(&x, &()).unwrap();

    // Compare learned lambdas — should match sklearn's Brent optimizer closely.
    assert_array_close(
        fitted.lambdas().as_slice().unwrap(),
        expected_lambdas.as_slice().unwrap(),
        1e-2,
        "PowerTransformer lambdas",
    );

    // Compare transformed output.
    let transformed = fitted.transform(&x).unwrap();
    let actual_flat: Vec<f64> = transformed.iter().copied().collect();
    let expected_flat: Vec<f64> = expected_transformed.iter().copied().collect();
    assert_array_close(
        &actual_flat,
        &expected_flat,
        1e-2,
        "PowerTransformer transformed",
    );
}
