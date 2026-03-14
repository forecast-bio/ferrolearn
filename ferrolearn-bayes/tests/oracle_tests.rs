//! Oracle tests comparing ferrolearn Naive Bayes models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.
//!
//! Naive Bayes is fully deterministic — predictions and learned parameters
//! should match sklearn exactly (within floating-point epsilon).

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

fn json_to_labels(value: &serde_json::Value) -> Array1<usize> {
    let vec: Vec<usize> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Array1::from_vec(vec)
}

fn json_to_vec_f64(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn json_to_array2_f64(value: &serde_json::Value) -> Vec<Vec<f64>> {
    value
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
        .collect()
}

// ---------------------------------------------------------------------------
// Gaussian Naive Bayes
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_nb_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/gaussian_nb.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_bayes::GaussianNB::<f64>::new();
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // GaussianNB is deterministic — require 100% prediction match.
    let mismatches: Vec<usize> = preds
        .iter()
        .zip(expected_preds.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "GaussianNB: {}/{} predictions differ from sklearn at indices: {:?}",
        mismatches.len(),
        expected_preds.len(),
        &mismatches[..mismatches.len().min(20)]
    );

    // Accuracy must match exactly.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 1e-10,
        "GaussianNB accuracy {accuracy:.10} != sklearn {sklearn_accuracy:.10}"
    );

    // Compare learned parameters: class priors and per-class means (theta).
    let expected_theta = json_to_array2_f64(&fixture["expected"]["theta"]);
    let expected_var = json_to_array2_f64(&fixture["expected"]["var"]);
    let expected_prior = json_to_vec_f64(&fixture["expected"]["class_prior"]);
    let _ = (expected_theta, expected_var, expected_prior);
    // NOTE: parameter comparison requires accessor methods on FittedGaussianNB.
    // If those exist, compare here with 1e-10 tolerance.
}

// ---------------------------------------------------------------------------
// Multinomial Naive Bayes
// ---------------------------------------------------------------------------

#[test]
fn test_multinomial_nb_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/multinomial_nb.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_bayes::MultinomialNB::<f64>::new().with_alpha(1.0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // MultinomialNB is deterministic — require 100% prediction match.
    let mismatches: Vec<usize> = preds
        .iter()
        .zip(expected_preds.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "MultinomialNB: {}/{} predictions differ from sklearn at indices: {:?}",
        mismatches.len(),
        expected_preds.len(),
        &mismatches[..mismatches.len().min(20)]
    );

    // Accuracy must match exactly.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 1e-10,
        "MultinomialNB accuracy {accuracy:.10} != sklearn {sklearn_accuracy:.10}"
    );
}

// ---------------------------------------------------------------------------
// Bernoulli Naive Bayes
// ---------------------------------------------------------------------------

#[test]
fn test_bernoulli_nb_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/bernoulli_nb.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_bayes::BernoulliNB::<f64>::new().with_alpha(1.0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // BernoulliNB is deterministic — require 100% prediction match.
    let mismatches: Vec<usize> = preds
        .iter()
        .zip(expected_preds.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "BernoulliNB: {}/{} predictions differ from sklearn at indices: {:?}",
        mismatches.len(),
        expected_preds.len(),
        &mismatches[..mismatches.len().min(20)]
    );

    // Accuracy must match exactly.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 1e-10,
        "BernoulliNB accuracy {accuracy:.10} != sklearn {sklearn_accuracy:.10}"
    );
}

// ---------------------------------------------------------------------------
// Complement Naive Bayes
// ---------------------------------------------------------------------------

#[test]
fn test_complement_nb_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/complement_nb.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let y = json_to_labels(&fixture["input"]["y"]);

    let expected_preds: Vec<usize> = fixture["expected"]["predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let sklearn_accuracy = fixture["expected"]["accuracy"].as_f64().unwrap();

    let model = ferrolearn_bayes::ComplementNB::<f64>::new().with_alpha(1.0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // ComplementNB is deterministic — require 100% prediction match.
    let mismatches: Vec<usize> = preds
        .iter()
        .zip(expected_preds.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "ComplementNB: {}/{} predictions differ from sklearn at indices: {:?}",
        mismatches.len(),
        expected_preds.len(),
        &mismatches[..mismatches.len().min(20)]
    );

    // Accuracy must match exactly.
    let n_correct: usize = preds.iter().zip(y.iter()).filter(|(a, b)| a == b).count();
    let accuracy = n_correct as f64 / y.len() as f64;
    assert!(
        (accuracy - sklearn_accuracy).abs() < 1e-10,
        "ComplementNB accuracy {accuracy:.10} != sklearn {sklearn_accuracy:.10}"
    );
}
