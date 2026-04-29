//! Oracle tests comparing ferrolearn clustering models against scikit-learn
//! reference outputs stored in `fixtures/*.json`.
//!
//! Deterministic algorithms (KMeans with seed, DBSCAN, Agglomerative) should
//! match sklearn exactly or near-exactly. Stochastic algorithms get slightly
//! looser tolerances but still strict enough to catch real bugs.

use ferrolearn_core::{Fit, Predict};
use ndarray::Array2;

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

/// Compute pairwise label agreement (up to permutation) between two label sets.
fn pairwise_agreement(labels_a: &[usize], labels_b: &[i64]) -> f64 {
    let mut agree = 0usize;
    let mut total = 0usize;
    for i in 0..labels_a.len() {
        for j in (i + 1)..labels_a.len().min(i + 10) {
            let same_a = labels_a[i] == labels_a[j];
            let same_b = labels_b[i] == labels_b[j];
            if same_a == same_b {
                agree += 1;
            }
            total += 1;
        }
    }
    agree as f64 / total as f64
}

/// Find the minimum distance from one centroid to any centroid in a set.
fn min_centroid_dist(target: ndarray::ArrayView1<f64>, centers: &Array2<f64>) -> f64 {
    (0..centers.nrows())
        .map(|i| {
            let c = centers.row(i);
            target
                .iter()
                .zip(c.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// KMeans — deterministic with fixed seed, should match sklearn tightly
// ---------------------------------------------------------------------------

#[test]
fn test_kmeans_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/kmeans.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_inertia = fixture["expected"]["inertia"].as_f64().unwrap();
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_centers = json_to_array2(&fixture["expected"]["cluster_centers"]);

    let model = ferrolearn_cluster::KMeans::<f64>::new(3)
        .with_random_state(42)
        .with_n_init(10);
    let fitted = model.fit(&x, &()).unwrap();

    let centers = fitted.cluster_centers();
    assert_eq!(centers.nrows(), 3);
    assert_eq!(centers.ncols(), 2);

    let labels = fitted.labels();
    assert_eq!(labels.len(), 150);

    // Inertia: KMeans with same seed should converge to same solution.
    // Tight tolerance: within 5%.
    let inertia = fitted.inertia();
    let ratio = inertia / sklearn_inertia;
    assert!(
        (0.95..1.05).contains(&ratio),
        "KMeans inertia {inertia:.4} vs sklearn {sklearn_inertia:.4} (ratio: {ratio:.4})"
    );

    // Centroids should be very close (< 0.5 Euclidean).
    for sk_row in 0..sklearn_centers.nrows() {
        let dist = min_centroid_dist(sklearn_centers.row(sk_row), centers);
        assert!(
            dist < 0.5,
            "KMeans centroid {sk_row}: min dist to ferrolearn = {dist:.4} (expected < 0.5)"
        );
    }

    // Pairwise label agreement: deterministic, expect near-perfect.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.99,
        "KMeans pairwise label agreement {agreement:.4} < 0.99"
    );
}

// ---------------------------------------------------------------------------
// DBSCAN — fully deterministic, should match sklearn exactly
// ---------------------------------------------------------------------------

#[test]
fn test_dbscan_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/dbscan.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;
    let sklearn_n_noise = fixture["expected"]["n_noise"].as_u64().unwrap() as usize;

    let model = ferrolearn_cluster::DBSCAN::<f64>::new(1.5).with_min_samples(5);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    let n_clusters = fitted.n_clusters();
    let n_noise: usize = labels.iter().filter(|&&l| l == -1).count();

    // DBSCAN is deterministic — exact match required.
    assert_eq!(
        n_clusters, sklearn_n_clusters,
        "DBSCAN n_clusters: got {n_clusters}, sklearn {sklearn_n_clusters}"
    );
    assert_eq!(
        n_noise, sklearn_n_noise,
        "DBSCAN n_noise: got {n_noise}, sklearn {sklearn_n_noise}"
    );

    let sklearn_core_count = fixture["expected"]["core_sample_indices"]
        .as_array()
        .unwrap()
        .len();
    let core_count = fitted.core_sample_indices().len();
    assert_eq!(
        core_count, sklearn_core_count,
        "DBSCAN core sample count: got {core_count}, sklearn {sklearn_core_count}"
    );
}

// ---------------------------------------------------------------------------
// Agglomerative — deterministic (Ward linkage), should match near-exactly
// ---------------------------------------------------------------------------

#[test]
fn test_agglomerative_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/agglomerative_clustering.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
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

    // Ward linkage is deterministic — expect near-perfect agreement.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.99,
        "AgglomerativeClustering pairwise agreement {agreement:.4} < 0.99"
    );
}

// ---------------------------------------------------------------------------
// MiniBatchKMeans — stochastic (mini-batch sampling), moderate tolerance
// ---------------------------------------------------------------------------

#[test]
fn test_mini_batch_kmeans_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/mini_batch_kmeans.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_inertia = fixture["expected"]["inertia"].as_f64().unwrap();
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_centers = json_to_array2(&fixture["expected"]["cluster_centers"]);

    let model = ferrolearn_cluster::MiniBatchKMeans::<f64>::new(3)
        .with_random_state(42)
        .with_batch_size(50)
        .with_n_init(5);
    let fitted = model.fit(&x, &()).unwrap();

    let centers = fitted.cluster_centers();
    assert_eq!(centers.nrows(), 3);

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    // Inertia within 10% (mini-batch is approximate).
    let inertia = fitted.inertia();
    let ratio = inertia / sklearn_inertia;
    assert!(
        (0.90..1.10).contains(&ratio),
        "MiniBatchKMeans inertia {inertia:.4} vs sklearn {sklearn_inertia:.4} (ratio: {ratio:.4})"
    );

    // Centroids should be close (< 1.0 Euclidean).
    for sk_row in 0..sklearn_centers.nrows() {
        let dist = min_centroid_dist(sklearn_centers.row(sk_row), centers);
        assert!(
            dist < 1.0,
            "MiniBatchKMeans centroid {sk_row}: min dist = {dist:.4} (expected < 1.0)"
        );
    }

    // Pairwise agreement: well-separated blobs, expect >= 95%.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.95,
        "MiniBatchKMeans pairwise agreement {agreement:.4} < 0.95"
    );
}

// ---------------------------------------------------------------------------
// MeanShift — deterministic given bandwidth, should match exactly
// ---------------------------------------------------------------------------

#[test]
fn test_mean_shift_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/mean_shift.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_centers = json_to_array2(&fixture["expected"]["cluster_centers"]);

    let model = ferrolearn_cluster::MeanShift::<f64>::new().with_bandwidth(2.0);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    let n_clusters = fitted.n_clusters();

    // MeanShift is deterministic — exact n_clusters match.
    assert_eq!(
        n_clusters, sklearn_n_clusters,
        "MeanShift n_clusters: got {n_clusters}, sklearn {sklearn_n_clusters}"
    );

    // Centroids should be very close (< 0.5).
    let centers = fitted.cluster_centers();
    for sk_row in 0..sklearn_centers.nrows() {
        let dist = min_centroid_dist(sklearn_centers.row(sk_row), centers);
        assert!(
            dist < 0.5,
            "MeanShift centroid {sk_row}: min dist = {dist:.4} (expected < 0.5)"
        );
    }

    // Pairwise agreement: deterministic, expect near-perfect.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.99,
        "MeanShift pairwise agreement {agreement:.4} < 0.99"
    );
}

// ---------------------------------------------------------------------------
// GaussianMixture — stochastic (EM), moderate tolerance
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_mixture_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/gaussian_mixture.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;
    let sklearn_means = json_to_array2(&fixture["expected"]["means"]);

    let model = ferrolearn_cluster::GaussianMixture::<f64>::new(3)
        .with_random_state(42)
        .with_n_init(3);
    let fitted = model.fit(&x, &()).unwrap();

    assert_eq!(fitted.means().nrows(), sklearn_n_clusters);

    let labels = fitted.predict(&x).unwrap();

    // Component means should be close (< 1.0).
    let means = fitted.means();
    for sk_row in 0..sklearn_means.nrows() {
        let dist = min_centroid_dist(sklearn_means.row(sk_row), means);
        assert!(
            dist < 1.0,
            "GMM mean {sk_row}: min dist = {dist:.4} (expected < 1.0)"
        );
    }

    // Pairwise agreement: well-separated blobs, expect >= 95%.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.95,
        "GaussianMixture pairwise agreement {agreement:.4} < 0.95"
    );
}

// ---------------------------------------------------------------------------
// OPTICS — deterministic, should match sklearn's cluster extraction
// ---------------------------------------------------------------------------

#[test]
fn test_optics_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/optics.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;
    let sklearn_n_noise = fixture["expected"]["n_noise"].as_u64().unwrap() as usize;

    let model = ferrolearn_cluster::OPTICS::<f64>::new(5).with_max_eps(10.0);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    // Ordering should be a valid permutation.
    let ordering = fitted.ordering();
    assert_eq!(ordering.len(), x.nrows());
    let mut sorted = ordering.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        x.nrows(),
        "OPTICS ordering is not a valid permutation"
    );

    // Reachability distances should be available.
    let reachability = fitted.reachability();
    assert_eq!(reachability.len(), x.nrows());

    let n_clusters = fitted.n_clusters();
    let n_noise: usize = labels.iter().filter(|&&l| l == -1).count();

    // OPTICS is deterministic — n_clusters should match exactly.
    assert_eq!(
        n_clusters, sklearn_n_clusters,
        "OPTICS n_clusters: got {n_clusters}, sklearn {sklearn_n_clusters}"
    );

    // Noise count may differ by a few points due to predecessor correction
    // boundary details (Algorithm 2 of Schubert & Gertz 2018).
    let noise_diff = (n_noise as isize - sklearn_n_noise as isize).unsigned_abs();
    assert!(
        noise_diff <= 10,
        "OPTICS n_noise: got {n_noise}, sklearn {sklearn_n_noise} (diff {noise_diff} > 10)"
    );
}

// ---------------------------------------------------------------------------
// Birch — deterministic, should match sklearn closely
// ---------------------------------------------------------------------------

#[test]
fn test_birch_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/birch.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;

    let model = ferrolearn_cluster::Birch::<f64>::new()
        .with_threshold(0.5)
        .with_n_clusters(3);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());
    assert_eq!(fitted.n_clusters(), sklearn_n_clusters);

    // Birch is deterministic — expect near-perfect agreement.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.99,
        "Birch pairwise agreement {agreement:.4} < 0.99"
    );
}

// ---------------------------------------------------------------------------
// SpectralClustering — eigensolver may vary, but well-separated data
// should yield near-perfect results
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_clustering_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/spectral_clustering.json")).unwrap();

    let x = json_to_array2(&fixture["input"]["X"]);
    let sklearn_labels: Vec<i64> = fixture["expected"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let sklearn_n_clusters = fixture["expected"]["n_clusters"].as_u64().unwrap() as usize;

    let model = ferrolearn_cluster::SpectralClustering::<f64>::new(3).with_random_state(42);
    let fitted = model.fit(&x, &()).unwrap();

    let labels = fitted.labels();
    assert_eq!(labels.len(), x.nrows());

    let mut unique = labels.to_vec();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), sklearn_n_clusters);

    // Well-separated blobs — even with eigensolver differences, expect >= 95%.
    let agreement = pairwise_agreement(labels.as_slice().unwrap(), &sklearn_labels);
    assert!(
        agreement >= 0.95,
        "SpectralClustering pairwise agreement {agreement:.4} < 0.95"
    );
}
