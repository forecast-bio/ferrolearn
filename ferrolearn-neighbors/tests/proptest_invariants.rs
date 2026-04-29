//! Property-based tests for KNN mathematical invariants.

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neighbors::KNeighborsClassifier;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for classification data with at least 2 classes and enough samples for k=3.
fn classification_data_strategy() -> impl Strategy<Value = (Array2<f64>, Array1<usize>)> {
    (6usize..=12usize, 2usize..=3usize, 2usize..=3usize)
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
// KNeighborsClassifier invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn knn_predict_labels_subset_of_training_labels(
        (x, y) in classification_data_strategy()
    ) {
        let model = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
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
    fn knn_k1_recovers_training_labels(
        (x, y) in classification_data_strategy()
    ) {
        let model = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..preds.len() {
            prop_assert_eq!(preds[i], y[i],
                "k=1 should recover training label: preds[{}]={} != y[{}]={}", i, preds[i], i, y[i]);
        }
    }

    #[test]
    fn knn_predict_output_length_matches_input(
        (x, y) in classification_data_strategy()
    ) {
        let model = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        prop_assert_eq!(preds.len(), x.nrows(),
            "predictions length {} != input rows {}", preds.len(), x.nrows());
    }
}
