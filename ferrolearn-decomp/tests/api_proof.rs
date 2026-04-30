//! Proof-of-API integration test for ferrolearn-decomp.
//!
//! Audit deliverable for crosslink #298 (under #246). Exercises every
//! public estimator in the crate end-to-end so future PRs that change the
//! public API have a green-or-red signal here.

use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_decomp::{
    Affinity, Algorithm, CCA, DictFitAlgorithm, DictTransformAlgorithm, DictionaryLearning,
    Dissimilarity, FactorAnalysis, FastICA, IncrementalPCA, Isomap, Kernel, KernelPCA, LLE,
    LatentDirichletAllocation, LdaLearningMethod, MDS, MiniBatchNMF, MiniBatchNMFInit, NMF,
    NMFInit, NMFSolver, NonLinearity, PCA, PLSCanonical, PLSRegression, PLSSVD, SparsePCA,
    SpectralEmbedding, TruncatedSVD, Tsne, Umap, UmapMetric,
};
use ndarray::Array2;

/// Wide-enough 2D point cloud (12×4) — used by most decomposers.
fn small_2d_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (12, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1, 1.2, 2.2, 3.2, 4.2, 1.3, 2.3, 3.3, 4.3, 1.4,
            2.4, 3.4, 4.4, 1.5, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0, 5.1, 6.1, 7.1, 8.1, 5.2, 6.2,
            7.2, 8.2, 5.3, 6.3, 7.3, 8.3, 5.4, 6.4, 7.4, 8.4, 5.5, 6.5, 7.5, 8.5,
        ],
    )
    .unwrap()
}

/// Non-negative count-style data for NMF / LDA.
fn count_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 5),
        vec![
            1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 1.0, 4.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            3.0, 5.0, 1.0, 2.0, 1.0, 0.0, 4.0, 3.0, 0.0, 5.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 2.0,
            0.0, 1.0, 2.0, 3.0, 1.0, 0.0,
        ],
    )
    .unwrap()
}

// =============================================================================
// PCA family
// =============================================================================
#[test]
fn api_proof_pca() {
    let x = small_2d_data();
    let f = PCA::<f64>::new(2).fit(&x, &()).unwrap();
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
    let recon = f.inverse_transform(&z).unwrap();
    assert_eq!(recon.dim(), x.dim());
}

#[test]
fn api_proof_incremental_pca() {
    let x = small_2d_data();
    let f = IncrementalPCA::<f64>::new(2).with_batch_size(4).fit(&x, &()).unwrap();
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
    let recon = f.inverse_transform(&z).unwrap();
    assert_eq!(recon.dim(), x.dim());
    let _ = f.components();
    let _ = f.explained_variance();
    let _ = f.explained_variance_ratio();
    let _ = f.mean();
    let _ = f.singular_values();
    let _ = f.n_samples_seen();
}

#[test]
fn api_proof_kernel_pca() {
    let x = small_2d_data();
    for kernel in [Kernel::Linear, Kernel::RBF, Kernel::Polynomial, Kernel::Sigmoid] {
        let f = KernelPCA::<f64>::new(2)
            .with_kernel(kernel)
            .with_gamma(1.0)
            .with_degree(2)
            .with_coef0(0.0)
            .fit(&x, &())
            .unwrap();
        let z = f.transform(&x).unwrap();
        assert_eq!(z.dim(), (12, 2));
    }
}

#[test]
fn api_proof_sparse_pca() {
    let x = small_2d_data();
    let f = SparsePCA::<f64>::new(2)
        .with_alpha(0.1)
        .with_max_iter(50)
        .with_tol(1e-3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
}

// =============================================================================
// TruncatedSVD
// =============================================================================
#[test]
fn api_proof_truncated_svd() {
    let x = small_2d_data();
    let f = TruncatedSVD::<f64>::new(2).with_random_state(0).fit(&x, &()).unwrap();
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
    let recon = f.inverse_transform(&z).unwrap();
    assert_eq!(recon.dim(), x.dim());
    let _ = f.components();
    let _ = f.singular_values();
    let _ = f.explained_variance();
    let _ = f.explained_variance_ratio();
}

// =============================================================================
// NMF + MiniBatchNMF
// =============================================================================
#[test]
fn api_proof_nmf() {
    let x = count_data();
    for solver in [NMFSolver::CoordinateDescent, NMFSolver::MultiplicativeUpdate] {
        for init in [NMFInit::Random, NMFInit::Nndsvd] {
            let f = NMF::<f64>::new(2)
                .with_max_iter(100)
                .with_tol(1e-4)
                .with_solver(solver)
                .with_init(init)
                .with_random_state(0)
                .fit(&x, &())
                .unwrap();
            let w = f.transform(&x).unwrap();
            assert_eq!(w.dim(), (8, 2));
            let recon = f.inverse_transform(&w).unwrap();
            assert_eq!(recon.dim(), x.dim());
            let _ = f.components();
            let _ = f.reconstruction_err();
            let _ = f.n_iter();
        }
    }
}

#[test]
fn api_proof_minibatch_nmf() {
    let x = count_data();
    for init in [MiniBatchNMFInit::Random, MiniBatchNMFInit::Nndsvd] {
        let f = MiniBatchNMF::<f64>::new(2)
            .with_max_iter(50)
            .with_batch_size(4)
            .with_tol(1e-4)
            .with_random_state(0)
            .with_init(init)
            .fit(&x, &())
            .unwrap();
        let w = f.transform(&x).unwrap();
        assert_eq!(w.dim(), (8, 2));
    }
}

// =============================================================================
// FastICA
// =============================================================================
#[test]
fn api_proof_fast_ica() {
    let x = small_2d_data();
    for algo in [Algorithm::Parallel, Algorithm::Deflation] {
        for fun in [NonLinearity::LogCosh, NonLinearity::Exp, NonLinearity::Cube] {
            let f = FastICA::<f64>::new(2)
                .with_algorithm(algo)
                .with_fun(fun)
                .with_max_iter(100)
                .with_tol(1e-4)
                .with_random_state(0)
                .fit(&x, &())
                .unwrap();
            let z = f.transform(&x).unwrap();
            assert_eq!(z.dim(), (12, 2));
        }
    }
}

// =============================================================================
// FactorAnalysis
// =============================================================================
#[test]
fn api_proof_factor_analysis() {
    let x = small_2d_data();
    let f = FactorAnalysis::<f64>::new(2)
        .with_max_iter(50)
        .with_tol(1e-3)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
    let recon = f.inverse_transform(&z).unwrap();
    assert_eq!(recon.dim(), x.dim());
    let _ = f.components();
    let _ = f.noise_variance();
    let _ = f.mean();
    let _ = f.n_iter();
    let _ = f.log_likelihood();
}

// =============================================================================
// DictionaryLearning
// =============================================================================
#[test]
fn api_proof_dictionary_learning() {
    let x = small_2d_data();
    for tx_algo in [DictTransformAlgorithm::Omp, DictTransformAlgorithm::LassoCd] {
        let f = DictionaryLearning::new(2)
            .with_alpha(0.1)
            .with_max_iter(20)
            .with_tol(1e-3)
            .with_fit_algorithm(DictFitAlgorithm::CoordinateDescent)
            .with_transform_algorithm(tx_algo)
            .with_transform_n_nonzero_coefs(2)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        let _ = f.transform(&x).unwrap();
    }
}

// =============================================================================
// LatentDirichletAllocation
// =============================================================================
#[test]
fn api_proof_latent_dirichlet_allocation() {
    let x = count_data();
    for method in [LdaLearningMethod::Batch, LdaLearningMethod::Online] {
        let f = LatentDirichletAllocation::new(3)
            .with_max_iter(20)
            .with_learning_method(method)
            .with_learning_offset(10.0)
            .with_learning_decay(0.7)
            .with_doc_topic_prior(0.1)
            .with_topic_word_prior(0.1)
            .with_max_doc_update_iter(50)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        let _ = f.transform(&x).unwrap();
    }
}

// =============================================================================
// Manifold: MDS / Isomap / LLE / SpectralEmbedding / TSNE / Umap
//
// These are f64-only and most don't expose Transform — they're fit-time
// embedders accessed via .embedding().
// =============================================================================
#[test]
fn api_proof_mds() {
    let x = small_2d_data();
    let f = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Euclidean)
        .fit(&x, &())
        .unwrap();
    let emb = f.embedding();
    assert_eq!(emb.nrows(), 12);
    assert_eq!(emb.ncols(), 2);

    // Precomputed dissimilarity path: feed a square distance matrix.
    let dist = Array2::<f64>::zeros((6, 6));
    let _ = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .fit(&dist, &())
        .unwrap();
}

#[test]
fn api_proof_isomap() {
    let x = small_2d_data();
    // Two well-separated clusters need enough neighbors to connect.
    let f = Isomap::new(2).with_n_neighbors(7).fit(&x, &()).unwrap();
    let emb = f.embedding();
    assert_eq!(emb.dim(), (12, 2));
    // Isomap also implements Transform<Array2<f64>>.
    let z = f.transform(&x).unwrap();
    assert_eq!(z.dim(), (12, 2));
}

#[test]
fn api_proof_lle() {
    let x = small_2d_data();
    let f = LLE::new(2)
        .with_n_neighbors(3)
        .with_reg(1e-3)
        .fit(&x, &())
        .unwrap();
    let emb = f.embedding();
    assert_eq!(emb.dim(), (12, 2));
}

#[test]
fn api_proof_spectral_embedding() {
    let x = small_2d_data();
    for affinity in [
        Affinity::RBF { gamma: 1.0 },
        Affinity::NearestNeighbors { n_neighbors: 3 },
    ] {
        let f = SpectralEmbedding::new(2)
            .with_affinity(affinity)
            .fit(&x, &())
            .unwrap();
        let emb = f.embedding();
        assert_eq!(emb.dim(), (12, 2));
    }
}

#[test]
fn api_proof_tsne() {
    let x = small_2d_data();
    let f = Tsne::new()
        .with_n_components(2)
        .with_perplexity(2.0)
        .with_learning_rate(50.0)
        .with_n_iter(100)
        .with_early_exaggeration(4.0)
        .with_theta(0.5)
        .with_random_state(0)
        .fit(&x, &())
        .unwrap();
    let emb = f.embedding();
    assert_eq!(emb.dim(), (12, 2));
}

#[test]
fn api_proof_umap() {
    let x = small_2d_data();
    for metric in [UmapMetric::Euclidean, UmapMetric::Cosine] {
        let f = Umap::new()
            .with_n_components(2)
            .with_n_neighbors(3)
            .with_min_dist(0.1)
            .with_spread(1.0)
            .with_learning_rate(1.0)
            .with_n_epochs(50)
            .with_metric(metric)
            .with_negative_sample_rate(5)
            .with_random_state(0)
            .fit(&x, &())
            .unwrap();
        let emb = f.embedding();
        assert_eq!(emb.dim(), (12, 2));
        // Umap also implements Transform<Array2<f64>>.
        let z = f.transform(&x).unwrap();
        assert_eq!(z.dim(), (12, 2));
    }
}

// =============================================================================
// Cross-decomposition
// =============================================================================
#[test]
fn api_proof_cross_decomposition() {
    let x = small_2d_data();
    let y = Array2::from_shape_vec(
        (12, 2),
        vec![
            1.0, 0.0, 1.1, 0.1, 1.2, 0.2, 1.3, 0.3, 1.4, 0.4, 1.5, 0.5, 5.0, 5.0, 5.1, 5.1, 5.2,
            5.2, 5.3, 5.3, 5.4, 5.4, 5.5, 5.5,
        ],
    )
    .unwrap();

    let pls_reg = PLSRegression::<f64>::new(2)
        .with_max_iter(100)
        .with_tol(1e-6)
        .with_scale(true)
        .fit(&x, &y)
        .unwrap();
    let preds = pls_reg.predict(&x).unwrap();
    assert_eq!(preds.dim(), (12, 2));
    let z = pls_reg.transform(&x).unwrap();
    assert_eq!(z.nrows(), 12);

    let pls_can = PLSCanonical::<f64>::new(2)
        .with_max_iter(100)
        .with_tol(1e-6)
        .with_scale(true)
        .fit(&x, &y)
        .unwrap();
    let _ = pls_can.transform(&x).unwrap();
    let _ = pls_can.transform_y(&y).unwrap();

    let pls_svd = PLSSVD::<f64>::new(2).with_scale(true).fit(&x, &y).unwrap();
    let _ = pls_svd.transform(&x).unwrap();

    let cca = CCA::<f64>::new(2)
        .with_max_iter(100)
        .with_tol(1e-6)
        .with_scale(true)
        .fit(&x, &y)
        .unwrap();
    let _ = cca.transform(&x).unwrap();
    let _ = cca.transform_y(&y).unwrap();
}

// Covariance estimators moved to ferrolearn-covariance; see
// ferrolearn-covariance/tests/api_proof.rs for their API proof.
