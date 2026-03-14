//! Oracle tests that compare ferrolearn metrics against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};

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

// ---------------------------------------------------------------------------
// ROC AUC oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_roc_auc_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/roc_auc.json")).unwrap();

    // Parse y_true as usize array and y_score as f64 array.
    let y_true_vec: Vec<usize> = fixture["input"]["y_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y_score_vec: Vec<f64> = fixture["input"]["y_score"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let y_true = Array1::from_vec(y_true_vec);
    let y_score = Array1::from_vec(y_score_vec);

    let expected_auc = fixture["expected"]["auc"].as_f64().unwrap();

    let auc = ferrolearn_metrics::roc_auc_score(&y_true, &y_score).unwrap();
    assert_relative_eq!(auc, expected_auc, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Log loss oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_log_loss_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/log_loss.json")).unwrap();

    // Parse y_true as usize array.
    let y_true_vec: Vec<usize> = fixture["input"]["y_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y_true = Array1::from_vec(y_true_vec);

    // Parse y_prob as a 2D array (n_samples x n_classes).
    let y_prob_rows: Vec<Vec<f64>> = fixture["input"]["y_prob"]
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
    let n_rows = y_prob_rows.len();
    let n_cols = y_prob_rows[0].len();
    let flat: Vec<f64> = y_prob_rows.into_iter().flatten().collect();
    let y_prob = Array2::from_shape_vec((n_rows, n_cols), flat).unwrap();

    let expected_loss = fixture["expected"]["loss"].as_f64().unwrap();

    let loss = ferrolearn_metrics::log_loss(&y_true, &y_prob).unwrap();
    assert_relative_eq!(loss, expected_loss, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// Clustering metrics oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_clustering_metrics_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/clustering_metrics.json")).unwrap();

    // Parse X as Array2<f64>.
    let x_rows: Vec<Vec<f64>> = fixture["input"]["X"]
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
    let n_samples = x_rows.len();
    let n_features = x_rows[0].len();
    let flat_x: Vec<f64> = x_rows.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), flat_x).unwrap();

    // Parse labels_true and labels_pred as isize arrays.
    let labels_true_vec: Vec<isize> = fixture["input"]["labels_true"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_pred_vec: Vec<isize> = fixture["input"]["labels_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as isize)
        .collect();
    let labels_true = Array1::from_vec(labels_true_vec);
    let labels_pred = Array1::from_vec(labels_pred_vec);

    let expected_silhouette = fixture["expected"]["silhouette"].as_f64().unwrap();
    let expected_ari = fixture["expected"]["adjusted_rand"].as_f64().unwrap();
    let expected_ami = fixture["expected"]["adjusted_mutual_info"]
        .as_f64()
        .unwrap();
    let expected_db = fixture["expected"]["davies_bouldin"].as_f64().unwrap();

    // Compare silhouette score.
    let sil: f64 = ferrolearn_metrics::silhouette_score(&x, &labels_pred).unwrap();
    assert_relative_eq!(sil, expected_silhouette, epsilon = 1e-6);

    // Compare adjusted Rand index.
    let ari = ferrolearn_metrics::adjusted_rand_score(&labels_true, &labels_pred).unwrap();
    assert_relative_eq!(ari, expected_ari, epsilon = 1e-6);

    // Compare adjusted mutual information.
    let ami = ferrolearn_metrics::adjusted_mutual_info(&labels_true, &labels_pred).unwrap();
    assert_relative_eq!(ami, expected_ami, epsilon = 1e-6);

    // Compare Davies-Bouldin score.
    let db: f64 = ferrolearn_metrics::davies_bouldin_score(&x, &labels_pred).unwrap();
    assert_relative_eq!(db, expected_db, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// Extended regression metrics oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_regression_metrics_extended_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/regression_metrics_extended.json"
    ))
    .unwrap();

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

    // sklearn returns MAPE as a fraction; ferrolearn returns it as a percentage
    // (multiplied by 100). Convert the expected value accordingly.
    let expected_mape_fraction = fixture["expected"]["mape"].as_f64().unwrap();
    let expected_mape_percent = expected_mape_fraction * 100.0;
    let expected_evs = fixture["expected"]["explained_variance"].as_f64().unwrap();

    // Compare MAPE (ferrolearn returns percentage, sklearn returns fraction).
    let mape: f64 = ferrolearn_metrics::mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    assert_relative_eq!(mape, expected_mape_percent, epsilon = 1e-10);

    // Compare explained variance score.
    let evs: f64 = ferrolearn_metrics::explained_variance_score(&y_true, &y_pred).unwrap();
    assert_relative_eq!(evs, expected_evs, epsilon = 1e-10);
}
