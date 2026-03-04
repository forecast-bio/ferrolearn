//! Oracle tests comparing ferrolearn tree-based models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use ferrolearn_core::introspection::HasFeatureImportances;
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2};

/// Parse a JSON nested array into an `Array2<f64>`.
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

/// Parse a JSON flat array of floats into `Array1<f64>`.
fn json_to_array1_f64(value: &serde_json::Value) -> Array1<f64> {
    let vec: Vec<f64> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    Array1::from_vec(vec)
}

/// Parse a JSON flat array of ints into `Array1<usize>`.
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
// Decision Tree Classifier
// ---------------------------------------------------------------------------

#[test]
fn test_decision_tree_classifier_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/decision_tree_classifier.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_tree::DecisionTreeClassifier::<f64>::new().with_max_depth(Some(3));
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Compute accuracy.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;

    // Our implementation may produce a slightly different tree, but accuracy
    // on training data with max_depth=3 should be comparable to sklearn's.
    assert!(
        accuracy >= sklearn_accuracy - 0.05,
        "DecisionTreeClassifier accuracy {accuracy:.4} too far below sklearn {sklearn_accuracy:.4}"
    );

    // Feature importances should sum to ~1.0.
    let importances = fitted.feature_importances();
    let imp_sum: f64 = importances.iter().sum();
    assert!(
        (imp_sum - 1.0).abs() < 0.01,
        "Feature importances sum to {imp_sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// Decision Tree Regressor
// ---------------------------------------------------------------------------

#[test]
fn test_decision_tree_regressor_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/decision_tree_regressor.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1_f64(&fixture["input"]["y"]);

    let sklearn_r2 = fixture["expected"]["r2"].as_f64().unwrap();

    let model = ferrolearn_tree::DecisionTreeRegressor::<f64>::new().with_max_depth(Some(4));
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // Compute R².
    let y_mean = y.mean().unwrap();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    // Training R² with depth=4 should be high for both implementations.
    assert!(
        r2 >= sklearn_r2 - 0.1,
        "DecisionTreeRegressor R² {r2:.4} too far below sklearn {sklearn_r2:.4}"
    );

    // Feature importances should sum to ~1.0.
    let importances = fitted.feature_importances();
    let imp_sum: f64 = importances.iter().sum();
    assert!(
        (imp_sum - 1.0).abs() < 0.01,
        "Feature importances sum to {imp_sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// Random Forest Classifier
// ---------------------------------------------------------------------------

#[test]
fn test_random_forest_classifier_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/random_forest_classifier.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_tree::RandomForestClassifier::<f64>::new()
        .with_n_estimators(20)
        .with_max_depth(Some(4))
        .with_random_state(42);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;

    // Random forest on training data should achieve high accuracy.
    assert!(
        accuracy >= 0.90,
        "RandomForestClassifier accuracy {accuracy:.4} < 0.90 (sklearn: {sklearn_accuracy:.4})"
    );

    let importances = fitted.feature_importances();
    let imp_sum: f64 = importances.iter().sum();
    assert!(
        (imp_sum - 1.0).abs() < 0.01,
        "Feature importances sum to {imp_sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// Random Forest Regressor
// ---------------------------------------------------------------------------

#[test]
fn test_random_forest_regressor_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/random_forest_regressor.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1_f64(&fixture["input"]["y"]);

    let sklearn_r2 = fixture["expected"]["r2"].as_f64().unwrap();

    let model = ferrolearn_tree::RandomForestRegressor::<f64>::new()
        .with_n_estimators(20)
        .with_max_depth(Some(4))
        .with_random_state(42);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let y_mean = y.mean().unwrap();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    // Training R² should be reasonably high.
    assert!(
        r2 >= 0.70,
        "RandomForestRegressor R² {r2:.4} < 0.70 (sklearn: {sklearn_r2:.4})"
    );

    let importances = fitted.feature_importances();
    let imp_sum: f64 = importances.iter().sum();
    assert!(
        (imp_sum - 1.0).abs() < 0.01,
        "Feature importances sum to {imp_sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// Gradient Boosting Classifier
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_boosting_classifier_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/gradient_boosting_classifier.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_tree::GradientBoostingClassifier::<f64>::new()
        .with_n_estimators(50)
        .with_max_depth(Some(2))
        .with_learning_rate(0.1)
        .with_random_state(42);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;

    // GBM on training data should achieve high accuracy.
    assert!(
        accuracy >= 0.90,
        "GradientBoostingClassifier accuracy {accuracy:.4} < 0.90 (sklearn: {sklearn_accuracy:.4})"
    );

    let importances = fitted.feature_importances();
    let imp_sum: f64 = importances.iter().sum();
    assert!(
        (imp_sum - 1.0).abs() < 0.01,
        "Feature importances sum to {imp_sum}, expected ~1.0"
    );
}

// ---------------------------------------------------------------------------
// Gradient Boosting Regressor
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_boosting_regressor_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../fixtures/gradient_boosting_regressor.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1_f64(&fixture["input"]["y"]);

    let sklearn_r2 = fixture["expected"]["r2"].as_f64().unwrap();

    let model = ferrolearn_tree::GradientBoostingRegressor::<f64>::new()
        .with_n_estimators(50)
        .with_max_depth(Some(2))
        .with_learning_rate(0.1)
        .with_random_state(42);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let y_mean = y.mean().unwrap();
    let ss_res: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|t| (t - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    assert!(
        r2 >= 0.50,
        "GradientBoostingRegressor R² {r2:.4} < 0.50 (sklearn: {sklearn_r2:.4})"
    );
}

// ---------------------------------------------------------------------------
// AdaBoost Classifier
// ---------------------------------------------------------------------------

#[test]
fn test_adaboost_classifier_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/adaboost_classifier.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_tree::AdaBoostClassifier::<f64>::new()
        .with_n_estimators(50)
        .with_learning_rate(1.0)
        .with_random_state(42);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;

    // AdaBoost on training data should achieve good accuracy on iris.
    assert!(
        accuracy >= 0.85,
        "AdaBoostClassifier accuracy {accuracy:.4} < 0.85 (sklearn: {sklearn_accuracy:.4})"
    );
}
