//! Proof-of-API integration test for ferrolearn-metrics.
//!
//! Audit deliverable for crosslink #309 (under #248). Exercises every
//! public function and type after the orphan wiring (#302), the 13 added
//! classification metrics (#303), the 4 added clustering metrics (#304),
//! and the 3 added pairwise items (#305). Every call uses
//! verified-from-source signatures.

use ferrolearn_metrics::{
    // Classification
    Average,
    // Pairwise
    Metric,
    // Clustering
    NmiMethod,
    PairwiseKernel,
    // Scorer
    Scorer,
    accuracy_score,
    adjusted_mutual_info,
    adjusted_rand_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calibration_curve,
    calinski_harabasz_score,
    chebyshev_distances,
    classification_report,
    cohen_kappa_score,
    completeness_score,
    confusion_matrix,
    contingency_matrix,
    cosine_distances,
    davies_bouldin_score,
    // Ranking
    dcg_score,
    det_curve,
    euclidean_distances,
    // Regression
    explained_variance_score,
    f1_score,
    fbeta_score,
    fowlkes_mallows_score,
    hamming_loss,
    hinge_loss,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    jaccard_score,
    log_loss,
    make_scorer,
    manhattan_distances,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_pinball_loss,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    multilabel_confusion_matrix,
    mutual_info_score,
    nan_euclidean_distances,
    ndcg_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    pairwise_distances,
    pairwise_distances_argmin,
    pairwise_distances_argmin_min,
    pairwise_kernels,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    rand_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
    root_mean_squared_log_error,
    silhouette_samples,
    silhouette_score,
    top_k_accuracy_score,
    v_measure_score,
    zero_one_loss,
};
use ndarray::{Array1, Array2, array};

// =============================================================================
// Classification
// =============================================================================
#[test]
fn api_proof_classification_basic() {
    let y_true = array![0usize, 1, 1, 0, 1, 2, 2, 0];
    let y_pred = array![0usize, 1, 0, 0, 1, 2, 1, 0];

    let _ = accuracy_score(&y_true, &y_pred).unwrap();
    let _ = hamming_loss(&y_true, &y_pred).unwrap();
    let _ = zero_one_loss(&y_true, &y_pred, true).unwrap();
    let _ = zero_one_loss(&y_true, &y_pred, false).unwrap();

    for avg in [Average::Macro, Average::Micro, Average::Weighted] {
        let _ = precision_score(&y_true, &y_pred, avg).unwrap();
        let _ = recall_score(&y_true, &y_pred, avg).unwrap();
        let _ = f1_score(&y_true, &y_pred, avg).unwrap();
        let _ = fbeta_score(&y_true, &y_pred, 0.5, avg).unwrap();
        let _ = jaccard_score(&y_true, &y_pred, avg).unwrap();
        let _ = precision_recall_fscore_support(&y_true, &y_pred, 1.0, avg).unwrap();
    }

    let _ = balanced_accuracy_score(&y_true, &y_pred, false).unwrap();
    let _ = balanced_accuracy_score(&y_true, &y_pred, true).unwrap();
    let _ = matthews_corrcoef(&y_true, &y_pred).unwrap();
    let _ = cohen_kappa_score(&y_true, &y_pred).unwrap();

    let cm = confusion_matrix(&y_true, &y_pred).unwrap();
    assert_eq!(cm.nrows(), 3);

    let mlcm = multilabel_confusion_matrix(&y_true, &y_pred).unwrap();
    assert_eq!(mlcm.shape(), &[3, 2, 2]);

    let report = classification_report(&y_true, &y_pred).unwrap();
    assert!(report.contains("accuracy"));
}

#[test]
fn api_proof_classification_binary_scores() {
    let y_true = array![0usize, 0, 1, 1, 1, 0];
    let y_score = array![0.1_f64, 0.4, 0.35, 0.8, 0.9, 0.2];
    let y_prob = y_score.clone();
    let y_dec = array![-1.0_f64, 0.5, 0.2, 1.5, 2.0, -0.5];

    let _ = roc_auc_score(&y_true, &y_score).unwrap();
    let (_fpr, _tpr, _t) = roc_curve(&y_true, &y_score).unwrap();
    let (_p, _r, _t2) = precision_recall_curve(&y_true, &y_score).unwrap();
    let _ = average_precision_score(&y_true, &y_score).unwrap();
    let (_fpr2, _fnr, _t3) = det_curve(&y_true, &y_score).unwrap();
    let _ = brier_score_loss(&y_true, &y_prob).unwrap();
    let _ = hinge_loss(&y_true, &y_dec).unwrap();

    // calibration curve
    let (_frac_pos, _mean_pred) = calibration_curve(&y_true, &y_prob, 3).unwrap();

    // auc on arbitrary curve
    let xs = array![0.0_f64, 0.5, 1.0];
    let ys = array![0.0_f64, 0.5, 1.0];
    let _ = auc(&xs, &ys).unwrap();

    // log_loss + top_k_accuracy
    let y_lab = array![0usize, 1, 1];
    let proba = Array2::<f64>::from_shape_vec((3, 2), vec![0.9, 0.1, 0.2, 0.8, 0.4, 0.6]).unwrap();
    let _ = log_loss(&y_lab, &proba).unwrap();
    let _ = top_k_accuracy_score(&y_lab, &proba, 1).unwrap();
}

// =============================================================================
// Regression
// =============================================================================
#[test]
fn api_proof_regression() {
    let y_true: Array1<f64> = array![3.0, -0.5, 2.0, 7.0, 5.0];
    let y_pred: Array1<f64> = array![2.5, 0.0, 2.1, 7.8, 5.2];

    let _ = mean_absolute_error(&y_true, &y_pred).unwrap();
    let _ = mean_squared_error(&y_true, &y_pred).unwrap();
    let _ = root_mean_squared_error(&y_true, &y_pred).unwrap();
    let _ = r2_score(&y_true, &y_pred).unwrap();
    let _ = mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
    let _ = explained_variance_score(&y_true, &y_pred).unwrap();
    let _ = median_absolute_error(&y_true, &y_pred).unwrap();
    let _ = max_error(&y_true, &y_pred).unwrap();
    let _ = mean_pinball_loss(&y_true, &y_pred, 0.5).unwrap();

    // log/poisson/gamma/tweedie need positive values
    let yp_pos: Array1<f64> = array![1.0, 1.5, 2.0, 3.0, 2.5];
    let yt_pos: Array1<f64> = array![1.2, 1.4, 2.1, 2.9, 2.6];
    let _ = mean_squared_log_error(&yt_pos, &yp_pos).unwrap();
    let _ = root_mean_squared_log_error(&yt_pos, &yp_pos).unwrap();
    let _ = mean_poisson_deviance(&yt_pos, &yp_pos).unwrap();
    let _ = mean_gamma_deviance(&yt_pos, &yp_pos).unwrap();
    let _ = mean_tweedie_deviance(&yt_pos, &yp_pos, 1.5).unwrap();
}

// =============================================================================
// Clustering
// =============================================================================
#[test]
fn api_proof_clustering_label_only() {
    let labels_true = array![0isize, 0, 1, 1, 2, 2];
    let labels_pred = array![0isize, 0, 1, 2, 2, 2];

    let _ = adjusted_rand_score(&labels_true, &labels_pred).unwrap();
    let _ = adjusted_mutual_info(&labels_true, &labels_pred).unwrap();
    let _ = mutual_info_score(&labels_true, &labels_pred).unwrap();
    let _ = homogeneity_score(&labels_true, &labels_pred).unwrap();
    let _ = completeness_score(&labels_true, &labels_pred).unwrap();
    let _ = v_measure_score(&labels_true, &labels_pred).unwrap();
    let (_h, _c, _v) = homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();
    let _ = rand_score(&labels_true, &labels_pred).unwrap();
    let _ = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
    for method in [
        NmiMethod::Arithmetic,
        NmiMethod::Geometric,
        NmiMethod::Min,
        NmiMethod::Max,
    ] {
        let _ = normalized_mutual_info_score(&labels_true, &labels_pred, method).unwrap();
    }

    let cm = contingency_matrix(&labels_true, &labels_pred).unwrap();
    assert_eq!(cm.nrows(), 3);
    let pcm = pair_confusion_matrix(&labels_true, &labels_pred).unwrap();
    assert_eq!(pcm.shape(), &[2, 2]);
}

#[test]
fn api_proof_clustering_with_features() {
    let x: Array2<f64> = ndarray::array![
        [0.0, 0.0],
        [0.1, 0.2],
        [10.0, 10.0],
        [10.1, 10.2],
        [5.0, 5.0],
        [5.2, 4.9],
    ];
    let labels = array![0isize, 0, 1, 1, 2, 2];

    let _ = silhouette_score::<f64>(&x, &labels).unwrap();
    let _ = silhouette_samples::<f64>(&x, &labels).unwrap();
    let _ = davies_bouldin_score::<f64>(&x, &labels).unwrap();
    let _ = calinski_harabasz_score::<f64>(&x, &labels).unwrap();
}

// =============================================================================
// Pairwise
// =============================================================================
#[test]
fn api_proof_pairwise() {
    let x: Array2<f64> = ndarray::array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let y: Array2<f64> = ndarray::array![[0.0, 1.0], [3.0, 3.0]];

    let _ = euclidean_distances::<f64>(&x, &y).unwrap();
    let _ = manhattan_distances::<f64>(&x, &y).unwrap();
    let _ = cosine_distances::<f64>(&x, &y).unwrap();
    let _ = chebyshev_distances::<f64>(&x, &y).unwrap();
    for m in [
        Metric::Euclidean,
        Metric::Manhattan,
        Metric::Cosine,
        Metric::Chebyshev,
    ] {
        let _ = pairwise_distances::<f64>(&x, &y, m).unwrap();
    }

    let mut x_nan = x.clone();
    x_nan[[0, 0]] = f64::NAN;
    let _ = nan_euclidean_distances::<f64>(&x_nan, &y).unwrap();

    let idx = pairwise_distances_argmin::<f64>(&x, &y, Metric::Euclidean).unwrap();
    assert_eq!(idx.len(), 3);
    let (idx2, mins) = pairwise_distances_argmin_min::<f64>(&x, &y, Metric::Euclidean).unwrap();
    assert_eq!(idx2.len(), 3);
    assert_eq!(mins.len(), 3);

    for kernel in [
        PairwiseKernel::Linear,
        PairwiseKernel::Polynomial {
            degree: 3,
            gamma: 1.0,
            coef0: 1.0,
        },
        PairwiseKernel::Rbf { gamma: 0.5 },
        PairwiseKernel::Sigmoid {
            gamma: 0.5,
            coef0: 0.0,
        },
        PairwiseKernel::Laplacian { gamma: 0.5 },
    ] {
        let _ = pairwise_kernels::<f64>(&x, &y, kernel).unwrap();
    }
}

// =============================================================================
// Ranking
// =============================================================================
#[test]
fn api_proof_ranking() {
    let y_true: Array1<f64> = array![3.0, 2.0, 3.0, 0.0, 1.0];
    let y_score: Array1<f64> = array![0.9, 0.5, 0.8, 0.4, 0.2];
    let _ = dcg_score::<f64>(&y_true, &y_score, None).unwrap();
    let _ = ndcg_score::<f64>(&y_true, &y_score, None).unwrap();
    let _ = dcg_score::<f64>(&y_true, &y_score, Some(3)).unwrap();
    let _ = ndcg_score::<f64>(&y_true, &y_score, Some(3)).unwrap();
}

// =============================================================================
// Scorer
// =============================================================================
#[test]
fn api_proof_scorer() {
    let scorer: Scorer<f64> = make_scorer(mean_absolute_error, false, "neg_mae");
    assert!(!scorer.greater_is_better);
    assert_eq!(scorer.name, "neg_mae");
    let y_true: Array1<f64> = array![1.0, 2.0, 3.0];
    let y_pred: Array1<f64> = array![1.0, 2.0, 3.0];
    let _ = scorer.score(&y_true, &y_pred).unwrap();
    let _ = scorer.sign();
}
