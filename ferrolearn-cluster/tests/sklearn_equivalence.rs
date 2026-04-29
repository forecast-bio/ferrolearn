//! Statistical equivalence tests comparing ferrolearn clustering models against
//! scikit-learn reference values stored in `tests/fixtures/sklearn_reference/`.

use ferrolearn_core::Fit;
use ndarray::Array2;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_kmeans() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/kmeans.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X_train"]);
    let sklearn_inertia = fixture["inertia"].as_f64().unwrap();
    let sklearn_labels: Vec<i64> = fixture["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_centers = json_to_array2(&fixture["cluster_centers"]);

    let model = ferrolearn_cluster::KMeans::<f64>::new(3)
        .with_random_state(42)
        .with_n_init(10);
    let fitted = model.fit(&x, &()).unwrap();

    // Verify cluster count.
    let centers = fitted.cluster_centers();
    assert_eq!(centers.nrows(), 3, "Expected 3 cluster centers");
    assert_eq!(centers.ncols(), 5, "Expected 5-dimensional centers");

    // Labels should assign all samples.
    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    // All labels in {0, 1, 2}.
    for &l in labels {
        assert!(l < 3, "Label {l} out of range [0, 3)");
    }

    // Inertia should be in the same ballpark as sklearn's.
    let inertia = fitted.inertia();
    let ratio = inertia / sklearn_inertia;
    assert!(
        (0.5..2.0).contains(&ratio),
        "KMeans inertia {inertia:.2} too far from sklearn {sklearn_inertia:.2} (ratio: {ratio:.2})"
    );

    // Each sklearn centroid should have a nearby ferrolearn centroid
    // (labels may be permuted).
    for sklearn_row in 0..sklearn_centers.nrows() {
        let sk_center = sklearn_centers.row(sklearn_row);
        let min_dist: f64 = (0..centers.nrows())
            .map(|i| {
                let fl_center = centers.row(i);
                let d: f64 = sk_center
                    .iter()
                    .zip(fl_center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                d.sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist < 3.0,
            "Sklearn centroid {sklearn_row} has no nearby ferrolearn centroid (min_dist={min_dist:.2})"
        );
    }

    // Pairwise label agreement (up to permutation).
    let mut agree = 0usize;
    let mut total = 0usize;
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len().min(i + 10) {
            let same_sklearn = sklearn_labels[i] == sklearn_labels[j];
            let same_ferro = labels[i] == labels[j];
            if same_sklearn == same_ferro {
                agree += 1;
            }
            total += 1;
        }
    }
    let pairwise_agreement = agree as f64 / total as f64;
    assert!(
        pairwise_agreement >= 0.90,
        "KMeans pairwise label agreement {pairwise_agreement:.4} < 0.90"
    );
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_dbscan() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/dbscan.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X_train"]);
    let sklearn_n_clusters = fixture["n_clusters"].as_u64().unwrap() as usize;
    let sklearn_n_noise = fixture["n_noise"].as_u64().unwrap() as usize;

    let model = ferrolearn_cluster::DBSCAN::<f64>::new(3.0).with_min_samples(5);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    // DBSCAN is deterministic -- cluster count and noise count should match.
    let n_clusters = fitted.n_clusters();
    let n_noise: usize = labels.iter().filter(|&&l| l == -1).count();

    assert_eq!(
        n_clusters, sklearn_n_clusters,
        "DBSCAN n_clusters: got {n_clusters}, sklearn {sklearn_n_clusters}"
    );
    assert_eq!(
        n_noise, sklearn_n_noise,
        "DBSCAN n_noise: got {n_noise}, sklearn {sklearn_n_noise}"
    );

    // Core sample count should match.
    let sklearn_core_count = fixture["core_sample_indices"].as_array().unwrap().len();
    let core_count = fitted.core_sample_indices().len();
    assert_eq!(
        core_count, sklearn_core_count,
        "DBSCAN core sample count: got {core_count}, sklearn {sklearn_core_count}"
    );
}

// ---------------------------------------------------------------------------
// Agglomerative Clustering
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_agglomerative() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/agglomerative_clustering.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X_train"]);
    let sklearn_labels: Vec<i64> = fixture["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let model = ferrolearn_cluster::AgglomerativeClustering::<f64>::new(3)
        .with_linkage(ferrolearn_cluster::Linkage::Ward);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());
    assert_eq!(fitted.n_clusters(), 3);

    for &l in labels {
        assert!(l < 3, "Label {l} out of range [0, 3)");
    }

    // Ward linkage is deterministic -- labels should agree up to permutation.
    let mut agree = 0usize;
    let mut total = 0usize;
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len().min(i + 10) {
            let same_sklearn = sklearn_labels[i] == sklearn_labels[j];
            let same_ferro = labels[i] == labels[j];
            if same_sklearn == same_ferro {
                agree += 1;
            }
            total += 1;
        }
    }
    let pairwise_agreement = agree as f64 / total as f64;
    assert!(
        pairwise_agreement >= 0.90,
        "AgglomerativeClustering pairwise label agreement {pairwise_agreement:.4} < 0.90"
    );
}
