//! Property-based tests for decision tree mathematical invariants.

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::DecisionTreeClassifier;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for classification data with at least 2 classes.
fn classification_data_strategy(
) -> impl Strategy<Value = (Array2<f64>, Array1<usize>, usize)> {
    (6usize..=12usize, 2usize..=3usize, 2usize..=3usize).prop_flat_map(
        |(n_samples, n_feat, n_classes)| {
            let x_strat = proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat);
            (x_strat, Just(n_samples), Just(n_feat), Just(n_classes))
        },
    )
    .prop_map(|(x_data, n_samples, n_feat, n_classes)| {
        let x = Array2::from_shape_vec((n_samples, n_feat), x_data).unwrap();
        // Distribute labels evenly across classes
        let labels: Vec<usize> = (0..n_samples).map(|i| i % n_classes).collect();
        let y = Array1::from_vec(labels);
        (x, y, n_classes)
    })
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn decision_tree_predict_labels_subset_of_training_labels(
        (x, y, _n_classes) in classification_data_strategy()
    ) {
        let model = DecisionTreeClassifier::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        let classes = fitted.classes();

        for &p in preds.iter() {
            prop_assert!(classes.contains(&p),
                "Predicted label {} not in training classes {:?}", p, classes);
        }
    }

    #[test]
    fn decision_tree_max_depth_1_at_most_2_predictions(
        (x, y, _n_classes) in classification_data_strategy()
    ) {
        let model = DecisionTreeClassifier::<f64>::new().with_max_depth(Some(1));
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        let mut unique_preds: Vec<usize> = preds.to_vec();
        unique_preds.sort_unstable();
        unique_preds.dedup();

        prop_assert!(unique_preds.len() <= 2,
            "max_depth=1 should produce at most 2 distinct predictions, got {}",
            unique_preds.len());
    }

    #[test]
    fn decision_tree_predict_proba_rows_sum_to_one(
        (x, y, _n_classes) in classification_data_strategy()
    ) {
        let model = DecisionTreeClassifier::<f64>::new();
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
    fn decision_tree_predict_proba_values_in_01(
        (x, y, _n_classes) in classification_data_strategy()
    ) {
        let model = DecisionTreeClassifier::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for &val in proba.iter() {
            prop_assert!(val >= 0.0 && val <= 1.0 + 1e-10,
                "Probability {} outside [0, 1]", val);
        }
    }

    #[test]
    fn decision_tree_predict_output_length_matches_input(
        (x, y, _n_classes) in classification_data_strategy()
    ) {
        let model = DecisionTreeClassifier::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        prop_assert_eq!(preds.len(), x.nrows(),
            "predictions length {} != input rows {}", preds.len(), x.nrows());
    }
}
