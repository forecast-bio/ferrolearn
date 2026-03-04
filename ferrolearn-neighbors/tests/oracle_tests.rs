//! Oracle tests comparing ferrolearn KNN models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2};

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

fn json_to_array1_f64(value: &serde_json::Value) -> Array1<f64> {
    let vec: Vec<f64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Array1::from_vec(vec)
}

fn json_to_labels(value: &serde_json::Value) -> Array1<usize> {
    let vec: Vec<usize> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Array1::from_vec(vec)
}

// ---------------------------------------------------------------------------
// KNeighbors Classifier
// ---------------------------------------------------------------------------

#[test]
fn test_kneighbors_classifier_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/kneighbors_classifier.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_neighbors::KNeighborsClassifier::<f64>::new().with_n_neighbors(5);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // KNN is deterministic — predictions should match sklearn exactly
    // (or nearly so, with possible tie-breaking differences).
    let n_match: usize = preds
        .iter()
        .zip(expected_preds.iter())
        .filter(|(a, b)| a == b)
        .count();
    let match_rate = n_match as f64 / expected_preds.len() as f64;

    assert!(
        match_rate >= 0.95,
        "KNeighborsClassifier prediction match rate {match_rate:.4} < 0.95 vs sklearn"
    );

    // Accuracy should be very close to sklearn's.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 0.05,
        "KNeighborsClassifier accuracy {accuracy:.4} too far from sklearn {sklearn_accuracy:.4}"
    );
}

// ---------------------------------------------------------------------------
// KNeighbors Regressor
// ---------------------------------------------------------------------------

#[test]
fn test_kneighbors_regressor_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/kneighbors_regressor.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1_f64(&fixture["input"]["y"]);

    let expected_preds = json_to_array1_f64(&fixture["expected"]["predictions"]);
    let sklearn_r2 = fixture["expected"]["r2"].as_f64().unwrap();

    let model = ferrolearn_neighbors::KNeighborsRegressor::<f64>::new().with_n_neighbors(5);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // KNN regression is deterministic — predictions should match closely.
    let max_diff: f64 = preds
        .iter()
        .zip(expected_preds.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1.0,
        "KNeighborsRegressor max prediction diff {max_diff:.4} >= 1.0 vs sklearn"
    );

    // R² should be close to sklearn's.
    let y_mean = y.mean().unwrap();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    assert!(
        (r2 - sklearn_r2).abs() < 0.1,
        "KNeighborsRegressor R² {r2:.4} too far from sklearn {sklearn_r2:.4}"
    );
}
