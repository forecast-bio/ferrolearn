//! Property-based tests for Gaussian Naive Bayes mathematical invariants.

use ferrolearn_bayes::GaussianNB;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for classification data with at least 2 classes.
fn classification_data_strategy() -> impl Strategy<Value = (Array2<f64>, Array1<usize>)> {
    (6usize..=12usize, 2usize..=4usize, 2usize..=3usize)
        .prop_flat_map(|(n_samples, n_feat, n_classes)| {
            let x_strat = proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat);
            (x_strat, Just(n_samples), Just(n_feat), Just(n_classes))
        })
        .prop_map(|(x_data, n_samples, n_feat, n_classes)| {
            let x = Array2::from_shape_vec((n_samples, n_feat), x_data).unwrap();
            let labels: Vec<usize> = (0..n_samples).map(|i| i % n_classes).collect();
            let y = Array1::from_vec(labels);
            (x, y)
        })
}

// ---------------------------------------------------------------------------
// GaussianNB invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn gaussian_nb_predict_proba_rows_sum_to_one(
        (x, y) in classification_data_strategy()
    ) {
        let model = GaussianNB::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            prop_assert!((row_sum - 1.0).abs() < 1e-10,
                "Row {} sum {} != 1.0", i, row_sum);
        }
    }

    #[test]
    fn gaussian_nb_predict_proba_values_in_01(
        (x, y) in classification_data_strategy()
    ) {
        let model = GaussianNB::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for &val in &proba {
            prop_assert!((0.0..=1.0 + 1e-10).contains(&val),
                "Probability {} outside [0, 1]", val);
        }
    }

    #[test]
    fn gaussian_nb_predict_labels_subset_of_training_labels(
        (x, y) in classification_data_strategy()
    ) {
        let model = GaussianNB::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        let classes = fitted.classes();

        for &p in &preds {
            prop_assert!(classes.contains(&p),
                "Predicted label {} not in training classes {:?}", p, classes);
        }
    }

    #[test]
    fn gaussian_nb_predict_output_length_matches_input(
        (x, y) in classification_data_strategy()
    ) {
        let model = GaussianNB::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        prop_assert_eq!(preds.len(), x.nrows(),
            "predictions length {} != input rows {}", preds.len(), x.nrows());
    }

    #[test]
    fn gaussian_nb_predict_matches_argmax_proba(
        (x, y) in classification_data_strategy()
    ) {
        let model = GaussianNB::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        let classes = fitted.classes();

        for i in 0..proba.nrows() {
            let mut best_idx = 0;
            let mut best_val = proba[[i, 0]];
            for c in 1..proba.ncols() {
                if proba[[i, c]] > best_val {
                    best_val = proba[[i, c]];
                    best_idx = c;
                }
            }
            prop_assert_eq!(preds[i], classes[best_idx],
                "predict[{}]={} != argmax(predict_proba)={}", i, preds[i], classes[best_idx]);
        }
    }
}
