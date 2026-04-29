//! Property-based tests for KMeans mathematical invariants.

use ferrolearn_cluster::KMeans;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::Array2;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for KMeans data: n_clusters in [2, 4], n_samples >= n_clusters,
/// n_features in [2, 4].
fn kmeans_data_strategy() -> impl Strategy<Value = (Array2<f64>, usize, usize, usize)> {
    (2usize..=4usize, 2usize..=3usize)
        .prop_flat_map(|(n_clusters, n_feat)| {
            let min_samples = n_clusters;
            let max_samples = n_clusters + 8;
            (min_samples..=max_samples, Just(n_clusters), Just(n_feat))
        })
        .prop_flat_map(|(n_samples, n_clusters, n_feat)| {
            proptest::collection::vec(-10.0..10.0f64, n_samples * n_feat).prop_map(move |data| {
                (
                    Array2::from_shape_vec((n_samples, n_feat), data).unwrap(),
                    n_samples,
                    n_feat,
                    n_clusters,
                )
            })
        })
}

// ---------------------------------------------------------------------------
// KMeans invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn kmeans_labels_in_valid_range(
        (x, _n_samples, _n_feat, n_clusters) in kmeans_data_strategy()
    ) {
        let model = KMeans::<f64>::new(n_clusters)
            .with_random_state(42)
            .with_n_init(2)
            .with_max_iter(50);
        let result = model.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let labels = fitted.labels();

        for &label in labels {
            prop_assert!(label < n_clusters,
                "Label {} >= n_clusters {}", label, n_clusters);
        }
    }

    #[test]
    fn kmeans_cluster_centers_shape(
        (x, _n_samples, n_feat, n_clusters) in kmeans_data_strategy()
    ) {
        let model = KMeans::<f64>::new(n_clusters)
            .with_random_state(42)
            .with_n_init(2)
            .with_max_iter(50);
        let result = model.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let centers = fitted.cluster_centers();

        prop_assert_eq!(centers.nrows(), n_clusters,
            "cluster_centers_ has {} rows, expected {}", centers.nrows(), n_clusters);
        prop_assert_eq!(centers.ncols(), n_feat,
            "cluster_centers_ has {} cols, expected {}", centers.ncols(), n_feat);
    }

    #[test]
    fn kmeans_inertia_non_negative(
        (x, _n_samples, _n_feat, n_clusters) in kmeans_data_strategy()
    ) {
        let model = KMeans::<f64>::new(n_clusters)
            .with_random_state(42)
            .with_n_init(2)
            .with_max_iter(50);
        let result = model.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();

        prop_assert!(fitted.inertia() >= 0.0,
            "inertia {} should be non-negative", fitted.inertia());
    }

    #[test]
    fn kmeans_predict_labels_in_valid_range(
        (x, _n_samples, _n_feat, n_clusters) in kmeans_data_strategy()
    ) {
        let model = KMeans::<f64>::new(n_clusters)
            .with_random_state(42)
            .with_n_init(2)
            .with_max_iter(50);
        let result = model.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        for &label in &preds {
            prop_assert!(label < n_clusters,
                "Predicted label {} >= n_clusters {}", label, n_clusters);
        }
    }

    #[test]
    fn kmeans_predict_output_length_matches_input(
        (x, n_samples, _n_feat, n_clusters) in kmeans_data_strategy()
    ) {
        let model = KMeans::<f64>::new(n_clusters)
            .with_random_state(42)
            .with_n_init(2)
            .with_max_iter(50);
        let result = model.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        prop_assert_eq!(preds.len(), n_samples,
            "predictions length {} != input rows {}", preds.len(), n_samples);
    }
}
