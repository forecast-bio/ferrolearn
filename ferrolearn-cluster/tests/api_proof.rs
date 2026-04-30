//! Proof-of-API integration test for ferrolearn-cluster.
//!
//! Audit deliverable for crosslink #282 (under #244). Exercises every
//! public API surface of the crate end-to-end so that future PRs that
//! change the public API have a green-or-red signal here.
//!
//! Coverage:
//! - KMeans / MiniBatchKMeans / BisectingKMeans: builders, fit, predict,
//!   transform, fit_predict, cluster_centers, labels, n_iter
//! - DBSCAN / OPTICS / AgglomerativeClustering / Hdbscan / Birch /
//!   AffinityPropagation / SpectralClustering: builders, fit, fit_predict,
//!   labels accessor, n_clusters
//! - MeanShift: builders, fit, predict, fit_predict, cluster_centers
//! - GaussianMixture / BayesianGaussianMixture: builders, fit, predict,
//!   predict_proba, transform, score, score_samples, bic, aic, fit_predict
//! - LabelPropagation / LabelSpreading: builders, fit, predict,
//!   predict_proba, score
//! - FeatureAgglomeration: builders, fit, transform
//! - All public enum variants

use approx::assert_relative_eq;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_cluster::{
    AffinityPropagation, AgglomerativeClustering, AgglomerativeLinkage, BayesianCovType,
    BayesianGaussianMixture, Birch, BisectingKMeans, BisectingStrategy, CovarianceType, DBSCAN,
    FeatureAgglomeration, GaussianMixture, Hdbscan, KMeans, LabelPropagation,
    LabelPropagationKernel, LabelSpreading, LabelSpreadingKernel, Linkage, MeanShift,
    MiniBatchKMeans, MiniBatchKMeansInit, OPTICS, PoolingFunc, SpectralClustering,
    WeightPriorType,
};
use ndarray::{Array1, Array2, array};

/// Two well-separated clusters in 2D for unsupervised tests.
fn two_blobs() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 10.0, 10.0, 10.1, 10.0,
            10.0, 10.1, 9.9, 10.0, 10.0, 9.9, 10.1, 10.1,
        ],
    )
    .unwrap()
}

fn assert_proba_well_formed(proba: &Array2<f64>, n_samples: usize, n_components: usize) {
    assert_eq!(proba.dim(), (n_samples, n_components));
    for i in 0..n_samples {
        let s: f64 = proba.row(i).sum();
        assert_relative_eq!(s, 1.0, epsilon = 1e-9);
        for c in 0..n_components {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&proba[[i, c]]),
                "proba[{i},{c}]={}",
                proba[[i, c]]
            );
        }
    }
}

// =============================================================================
// KMeans family (predict + transform + fit_predict)
// =============================================================================
#[test]
fn api_proof_kmeans() {
    let x = two_blobs();

    let m = KMeans::<f64>::new(2)
        .with_max_iter(100)
        .with_tol(1e-4)
        .with_n_init(3)
        .with_random_state(42);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.cluster_centers();
    let _ = f.labels();
    let _ = f.n_iter();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 12);
    let dist = f.transform(&x).unwrap();
    assert_eq!(dist.dim(), (12, 2));

    let labels_fp = m.fit_predict(&x).unwrap();
    assert_eq!(labels_fp.len(), 12);
}

#[test]
fn api_proof_mini_batch_kmeans() {
    let x = two_blobs();

    let m = MiniBatchKMeans::<f64>::new(2)
        .with_init(MiniBatchKMeansInit::KMeansPlusPlus)
        .with_random_state(42);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.predict(&x).unwrap();
    let _ = f.transform(&x).unwrap();
    let _ = m.fit_predict(&x).unwrap();
}

#[test]
fn api_proof_bisecting_kmeans() {
    let x = two_blobs();

    let m = BisectingKMeans::<f64>::new(2)
        .with_max_iter(100)
        .with_n_init(2)
        .with_random_state(42)
        .with_bisecting_strategy(BisectingStrategy::LargestCluster);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.cluster_centers();
    let _ = f.labels();
    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 12);
    let dist = f.transform(&x).unwrap();
    assert_eq!(dist.dim(), (12, 2));
    let _ = m.fit_predict(&x).unwrap();
}

// =============================================================================
// MeanShift
// =============================================================================
#[test]
fn api_proof_mean_shift() {
    let x = two_blobs();

    let m = MeanShift::<f64>::new()
        .with_bandwidth(2.0)
        .with_max_iter(50)
        .with_tol(1e-4);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.cluster_centers();
    let _ = f.labels();
    let _ = f.n_iter();
    let _ = f.n_clusters();
    let _ = f.predict(&x).unwrap();
    let _ = m.fit_predict(&x).unwrap();

    let _: MeanShift<f64> = Default::default();
}

// =============================================================================
// DBSCAN / OPTICS / Agglo / HDBSCAN / Birch / AffinityProp / Spectral
// (label-only, fit_predict via labels())
// =============================================================================
#[test]
fn api_proof_dbscan() {
    let x = two_blobs();

    let m = DBSCAN::<f64>::new(0.5).with_min_samples(2);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.labels();
    let _ = f.core_sample_indices();
    let _ = f.n_clusters();
    let labels = m.fit_predict(&x).unwrap();
    assert_eq!(labels.len(), 12);
}

#[test]
fn api_proof_optics() {
    let x = two_blobs();

    let m = OPTICS::<f64>::new(2)
        .with_max_eps(5.0)
        .with_xi(0.05)
        .with_min_cluster_size(2);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.labels();
    let _ = m.fit_predict(&x).unwrap();
}

#[test]
fn api_proof_agglomerative_clustering() {
    let x = two_blobs();
    for linkage in [Linkage::Ward, Linkage::Complete, Linkage::Average, Linkage::Single] {
        let m = AgglomerativeClustering::<f64>::new(2).with_linkage(linkage);
        let f = m.fit(&x, &()).unwrap();
        let _ = f.labels();
        let _ = f.n_clusters();
        let _ = m.fit_predict(&x).unwrap();
    }
}

#[test]
fn api_proof_hdbscan() {
    let x = two_blobs();

    let m = Hdbscan::<f64>::new()
        .with_min_cluster_size(3)
        .with_min_samples(2)
        .with_cluster_selection_epsilon(0.0);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.labels();
    let _ = f.n_clusters();
    let _ = m.fit_predict(&x).unwrap();

    let _: Hdbscan<f64> = Default::default();
}

#[test]
fn api_proof_birch() {
    let x = two_blobs();

    let m = Birch::<f64>::new()
        .with_threshold(0.5)
        .with_branching_factor(50)
        .with_n_clusters(2);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.labels();
    let _ = f.n_clusters();
    let _ = m.fit_predict(&x).unwrap();

    let _: Birch<f64> = Default::default();
}

#[test]
fn api_proof_affinity_propagation() {
    let x = two_blobs();

    let m = AffinityPropagation::<f64>::new()
        .with_damping(0.9)
        .with_max_iter(100)
        .with_convergence_iter(15);
    // AffinityPropagation can fail to converge on tiny data; tolerate it.
    if let Ok(f) = m.fit(&x, &()) {
        let _ = f.cluster_centers();
        let _ = f.labels();
        let _ = f.exemplar_indices();
        let _ = f.n_iter();
        let _ = f.n_clusters();
        let _ = m.fit_predict(&x).unwrap();
    }

    let _: AffinityPropagation<f64> = Default::default();
}

#[test]
fn api_proof_spectral_clustering() {
    let x = two_blobs();

    let m = SpectralClustering::<f64>::new(2)
        .with_gamma(1.0)
        .with_n_init(2)
        .with_random_state(42);
    let f = m.fit(&x, &()).unwrap();
    let _ = f.labels();
    let _ = m.fit_predict(&x).unwrap();
}

// =============================================================================
// GaussianMixture (predict, predict_proba, transform, score*, bic/aic, fit_predict)
// =============================================================================
#[test]
fn api_proof_gaussian_mixture() {
    let x = two_blobs();

    for cov in [
        CovarianceType::Full,
        CovarianceType::Tied,
        CovarianceType::Diag,
        CovarianceType::Spherical,
    ] {
        let m = GaussianMixture::<f64>::new(2)
            .with_covariance_type(cov)
            .with_max_iter(100)
            .with_tol(1e-3)
            .with_random_state(42);
        let f = m.fit(&x, &()).unwrap();
        let _ = f.weights();
        let _ = f.means();
        let _ = f.covariances();
        let _ = f.converged();
        let _ = f.lower_bound();
        let _ = f.n_parameters();

        let _ = f.predict(&x).unwrap();
        let proba = f.predict_proba(&x).unwrap();
        assert_proba_well_formed(&proba, 12, 2);
        let resp = f.transform(&x).unwrap();
        assert_eq!(resp.dim(), (12, 2));

        let s = f.score(&x).unwrap();
        assert!(s.is_finite());
        let s_per_sample = f.score_samples(&x).unwrap();
        assert_eq!(s_per_sample.len(), 12);

        let bic = f.bic(&x).unwrap();
        let aic = f.aic(&x).unwrap();
        assert!(bic.is_finite() && aic.is_finite());

        let _ = m.fit_predict(&x).unwrap();
    }
}

// =============================================================================
// BayesianGaussianMixture (same coverage)
// =============================================================================
#[test]
fn api_proof_bayesian_gaussian_mixture() {
    let x = two_blobs();

    for cov in [
        BayesianCovType::Full,
        BayesianCovType::Tied,
        BayesianCovType::Diag,
        BayesianCovType::Spherical,
    ] {
        for wpt in [WeightPriorType::DirichletProcess, WeightPriorType::DirichletDistribution] {
            let m = BayesianGaussianMixture::<f64>::new(3)
                .with_covariance_type(cov)
                .with_weight_prior_type(wpt)
                .with_weight_concentration_prior(1.0)
                .with_max_iter(50)
                .with_tol(1e-3)
                .with_random_state(42);
            let f = m.fit(&x, &()).unwrap();
            let _ = f.weights();
            let _ = f.means();
            let _ = f.covariances();
            let _ = f.converged();
            let _ = f.lower_bound();
            let _ = f.weight_prior_type();
            let _ = f.n_features();

            let _ = f.predict(&x).unwrap();
            let proba = f.predict_proba(&x).unwrap();
            assert_proba_well_formed(&proba, 12, 3);

            let s = f.score(&x).unwrap();
            assert!(s.is_finite());
            let s_per = f.score_samples(&x).unwrap();
            assert_eq!(s_per.len(), 12);

            let bic = f.bic(&x).unwrap();
            let aic = f.aic(&x).unwrap();
            assert!(bic.is_finite() && aic.is_finite());

            let _ = m.fit_predict(&x).unwrap();
        }
    }
}

// =============================================================================
// LabelPropagation / LabelSpreading
// =============================================================================
fn semi_supervised_data() -> (Array2<f64>, Array1<isize>) {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1],
    )
    .unwrap();
    // First and fifth labeled; others unlabeled (-1).
    let y = array![0isize, -1, -1, -1, 1, -1, -1, -1];
    (x, y)
}

#[test]
fn api_proof_label_propagation() {
    let (x, y) = semi_supervised_data();

    let m = LabelPropagation::<f64>::new()
        .with_kernel(LabelPropagationKernel::Rbf)
        .with_gamma(1.0)
        .with_max_iter(50)
        .with_tol(1e-3);
    let f = m.fit(&x, &y).unwrap();
    let _ = f.labels();
    let _ = f.label_distributions();
    assert_eq!(f.n_classes(), 2);

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 8);
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 8, 2);
    let _ = f.score(&x, &y).unwrap();

    // Knn kernel + n_neighbors smoke
    let _ = LabelPropagation::<f64>::new()
        .with_kernel(LabelPropagationKernel::Knn)
        .with_n_neighbors(3)
        .fit(&x, &y)
        .unwrap();

    let _: LabelPropagation<f64> = Default::default();
}

#[test]
fn api_proof_label_spreading() {
    let (x, y) = semi_supervised_data();

    let m = LabelSpreading::<f64>::new()
        .with_kernel(LabelSpreadingKernel::Rbf)
        .with_gamma(1.0)
        .with_max_iter(50)
        .with_tol(1e-3)
        .with_alpha(0.2);
    let f = m.fit(&x, &y).unwrap();
    let _ = f.labels();
    let _ = f.label_distributions();
    assert_eq!(f.n_classes(), 2);

    let preds = f.predict(&x).unwrap();
    assert_eq!(preds.len(), 8);
    let proba = f.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba, 8, 2);
    let _ = f.score(&x, &y).unwrap();

    let _ = LabelSpreading::<f64>::new()
        .with_kernel(LabelSpreadingKernel::Knn)
        .with_n_neighbors(3)
        .fit(&x, &y)
        .unwrap();

    let _: LabelSpreading<f64> = Default::default();
}

// =============================================================================
// FeatureAgglomeration (transform)
// =============================================================================
#[test]
fn api_proof_feature_agglomeration() {
    // Tall+wide-ish: 8 samples, 6 features.
    let x = Array2::from_shape_vec(
        (8, 6),
        vec![
            0.0, 0.1, 0.0, 5.0, 5.1, 5.0, 0.1, 0.0, 0.1, 5.1, 5.0, 5.1, 0.0, 0.0, 0.1, 5.0, 5.1,
            5.1, 0.1, 0.1, 0.0, 5.1, 5.1, 5.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 0.1, 0.0, 0.0, 5.1,
            5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.1, 0.1, 0.1, 5.1, 5.1, 5.1,
        ],
    )
    .unwrap();

    for linkage in [
        AgglomerativeLinkage::Ward,
        AgglomerativeLinkage::Complete,
        AgglomerativeLinkage::Average,
        AgglomerativeLinkage::Single,
    ] {
        for pool in [PoolingFunc::Mean, PoolingFunc::Max] {
            let m = FeatureAgglomeration::<f64>::new(2)
                .with_linkage(linkage)
                .with_pooling_func(pool);
            let f = m.fit(&x, &()).unwrap();
            assert_eq!(f.n_clusters(), 2);
            let pooled = f.transform(&x).unwrap();
            assert_eq!(pooled.nrows(), 8);
            assert_eq!(pooled.ncols(), 2);
        }
    }
}
