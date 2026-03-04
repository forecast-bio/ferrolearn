//! Oracle tests that compare ferrolearn metrics against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use approx::assert_relative_eq;
use ndarray::Array1;

// ---------------------------------------------------------------------------
// Classification metrics oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_classification_metrics_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/classification_metrics.json")).unwrap();

    // Parse y_true and y_pred as usize arrays.
    let y_true_vec: Vec<usize> = fixture["input"]["y_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y_pred_vec: Vec<usize> = fixture["input"]["y_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y_true = Array1::from_vec(y_true_vec);
    let y_pred = Array1::from_vec(y_pred_vec);

    let expected_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();
    let expected_precision = fixture["expected"]["precision"].as_f64().unwrap();
    let expected_recall = fixture["expected"]["recall"].as_f64().unwrap();
    let expected_f1 = fixture["expected"]["f1"].as_f64().unwrap();

    // Parse expected confusion matrix.
    let expected_cm_rows: Vec<Vec<usize>> = fixture["expected"]["confusion_matrix"]
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

    // Compare accuracy.
    let accuracy = ferrolearn_metrics::accuracy_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(accuracy, expected_accuracy, epsilon = 1e-10);

    // Compare precision (binary average).
    let precision =
        ferrolearn_metrics::precision_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
            .unwrap();
    assert_relative_eq!(precision, expected_precision, epsilon = 1e-10);

    // Compare recall (binary average).
    let recall =
        ferrolearn_metrics::recall_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
            .unwrap();
    assert_relative_eq!(recall, expected_recall, epsilon = 1e-10);

    // Compare F1 (binary average).
    let f1 = ferrolearn_metrics::f1_score(&y_true, &y_pred, ferrolearn_metrics::Average::Binary)
        .unwrap();
    assert_relative_eq!(f1, expected_f1, epsilon = 1e-10);

    // Compare confusion matrix.
    let cm = ferrolearn_metrics::confusion_matrix(&y_true, &y_pred).unwrap();
    for (i, expected_row) in expected_cm_rows.iter().enumerate() {
        for (j, &expected_val) in expected_row.iter().enumerate() {
            assert_eq!(
                cm[[i, j]],
                expected_val,
                "confusion_matrix[{i},{j}]: actual={}, expected={expected_val}",
                cm[[i, j]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Regression metrics oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_regression_metrics_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/regression_metrics.json")).unwrap();

    // Parse y_true and y_pred as f64 arrays.
    let y_true_vec: Vec<f64> = fixture["input"]["y_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let y_pred_vec: Vec<f64> = fixture["input"]["y_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let y_true = Array1::from_vec(y_true_vec);
    let y_pred = Array1::from_vec(y_pred_vec);

    let expected_mae = fixture["expected"]["mae"].as_f64().unwrap();
    let expected_mse = fixture["expected"]["mse"].as_f64().unwrap();
    let expected_rmse = fixture["expected"]["rmse"].as_f64().unwrap();
    let expected_r2 = fixture["expected"]["r2"].as_f64().unwrap();

    // Compare MAE.
    let mae: f64 = ferrolearn_metrics::mean_absolute_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(mae, expected_mae, epsilon = 1e-10);

    // Compare MSE.
    let mse: f64 = ferrolearn_metrics::mean_squared_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(mse, expected_mse, epsilon = 1e-10);

    // Compare RMSE.
    let rmse: f64 = ferrolearn_metrics::root_mean_squared_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(rmse, expected_rmse, epsilon = 1e-10);

    // Compare R2.
    let r2: f64 = ferrolearn_metrics::r2_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(r2, expected_r2, epsilon = 1e-10);
}
