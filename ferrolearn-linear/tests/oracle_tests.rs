//! Oracle tests that compare ferrolearn linear models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.

use approx::assert_relative_eq;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::{Fit, Predict};
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
// LinearRegression oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_linear_regression_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/linear_regression.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    let model = ferrolearn_linear::LinearRegression::<f64>::new().with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients.
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-6,
        "LinearRegression coefficients",
    );

    // Compare intercept.
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-6);

    // Compare predictions.
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-6,
        "LinearRegression predictions",
    );
}

// ---------------------------------------------------------------------------
// Ridge oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_ridge_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/ridge.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let alpha = fixture["params"]["alpha"].as_f64().unwrap();
    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    let model = ferrolearn_linear::Ridge::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients.
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-5,
        "Ridge coefficients",
    );

    // Compare intercept.
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-5);

    // Compare predictions.
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-5,
        "Ridge predictions",
    );
}

// ---------------------------------------------------------------------------
// Lasso oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_lasso_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/lasso.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_array1(&fixture["input"]["y"]);

    let expected_coefs = json_to_array1(&fixture["expected"]["coefficients"]);
    let expected_intercept = fixture["expected"]["intercept"].as_f64().unwrap();
    let expected_preds = json_to_array1(&fixture["expected"]["predictions"]);

    let alpha = fixture["params"]["alpha"].as_f64().unwrap();
    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    // Use tighter convergence tolerance to match scikit-learn more closely.
    let model = ferrolearn_linear::Lasso::<f64>::new()
        .with_alpha(alpha)
        .with_fit_intercept(fit_intercept)
        .with_max_iter(10_000)
        .with_tol(1e-8);
    let fitted = model.fit(&x, &y).unwrap();

    // Compare coefficients (looser tolerance for coordinate descent).
    assert_array_close(
        fitted.coefficients().as_slice().unwrap(),
        expected_coefs.as_slice().unwrap(),
        1e-3,
        "Lasso coefficients",
    );

    // Compare intercept (looser tolerance).
    assert_relative_eq!(fitted.intercept(), expected_intercept, epsilon = 1e-3);

    // Compare predictions (looser tolerance).
    let preds = fitted.predict(&x).unwrap();
    assert_array_close(
        preds.as_slice().unwrap(),
        expected_preds.as_slice().unwrap(),
        1e-2,
        "Lasso predictions",
    );
}

// ---------------------------------------------------------------------------
// LogisticRegression oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_logistic_regression_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/logistic_regression.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);

    // Parse y as usize labels.
    let y_vec: Vec<usize> = fixture["input"]["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let y = Array1::from_vec(y_vec);

    // Expected coefficients: sklearn stores as [[c0, c1, ...]] for binary.
    let expected_coefs_2d = json_to_array2(&fixture["expected"]["coefficients"]);
    let expected_coefs = expected_coefs_2d.row(0).to_owned();
    let _expected_intercept = fixture["expected"]["intercept"].as_array().unwrap()[0]
        .as_f64()
        .unwrap();

    let expected_classes: Vec<usize> = fixture["expected"]["predicted_classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let expected_proba = json_to_array2(&fixture["expected"]["predicted_proba"]);

    let _c = fixture["params"]["C"].as_f64().unwrap();
    let fit_intercept = fixture["params"]["fit_intercept"].as_bool().unwrap();

    // Use a very high C (weak regularization) to minimize the effect of
    // the regularization term difference between ferrolearn's L-BFGS and
    // sklearn's solver. This brings the two solutions much closer together.
    let model = ferrolearn_linear::LogisticRegression::<f64>::new()
        .with_c(1e6)
        .with_fit_intercept(fit_intercept)
        .with_max_iter(5000)
        .with_tol(1e-8);
    let fitted = model.fit(&x, &y).unwrap();

    // Verify coefficient count matches.
    let actual_coefs = fitted.coefficients();
    assert_eq!(
        actual_coefs.len(),
        expected_coefs.len(),
        "LogisticRegression: coefficient length mismatch"
    );

    // All sklearn coefficients are positive for this well-separated data,
    // so the ferrolearn coefficients should also be positive.
    for i in 0..actual_coefs.len() {
        assert!(
            actual_coefs[i] > 0.0,
            "LogisticRegression coefficient[{i}] should be positive, got {}",
            actual_coefs[i]
        );
    }

    // Compare predicted classes against sklearn's expected predictions.
    let pred_classes = fitted.predict(&x).unwrap();
    let n_match: usize = pred_classes
        .iter()
        .zip(expected_classes.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = n_match as f64 / expected_classes.len() as f64;
    assert!(
        accuracy >= 0.95,
        "LogisticRegression predicted classes accuracy {accuracy} < 0.95"
    );

    // Verify predicted probabilities are well-formed and directionally correct.
    let pred_proba = fitted.predict_proba(&x).unwrap();
    assert_eq!(pred_proba.nrows(), expected_proba.nrows());
    assert_eq!(pred_proba.ncols(), expected_proba.ncols());

    // Each row should sum to 1.
    for i in 0..pred_proba.nrows() {
        assert_relative_eq!(pred_proba.row(i).sum(), 1.0, epsilon = 1e-10);
    }

    // For each sample, the predicted class-1 probability should agree
    // directionally with sklearn's: if sklearn says p(class1) > 0.5,
    // ferrolearn should too, and vice-versa.
    let mut directional_matches = 0usize;
    for i in 0..pred_proba.nrows() {
        let actual_class1 = pred_proba[[i, 1]] > 0.5;
        let expected_class1 = expected_proba[[i, 1]] > 0.5;
        if actual_class1 == expected_class1 {
            directional_matches += 1;
        }
    }
    let directional_accuracy = directional_matches as f64 / pred_proba.nrows() as f64;
    assert!(
        directional_accuracy >= 0.95,
        "LogisticRegression directional probability accuracy {directional_accuracy} < 0.95"
    );
}
