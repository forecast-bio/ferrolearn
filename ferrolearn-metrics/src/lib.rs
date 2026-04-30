//! # ferrolearn-metrics
//!
//! Classification and regression evaluation metrics for the ferrolearn machine
//! learning framework — a scikit-learn equivalent for Rust.
//!
//! ## Classification Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`classification::accuracy_score`] | Fraction of correctly classified samples |
//! | [`classification::precision_score`] | Positive predictive value (TP / (TP + FP)) |
//! | [`classification::recall_score`] | Sensitivity (TP / (TP + FN)) |
//! | [`classification::f1_score`] | Harmonic mean of precision and recall |
//! | [`classification::roc_auc_score`] | Area under the ROC curve (binary) |
//! | [`classification::roc_curve`] | Compute ROC curve (FPR, TPR, thresholds) |
//! | [`classification::precision_recall_curve`] | Compute precision-recall curve |
//! | [`classification::auc`] | Area under an arbitrary curve (trapezoidal rule) |
//! | [`classification::average_precision_score`] | Average precision from PR curve |
//! | [`classification::confusion_matrix`] | Matrix of true vs. predicted counts |
//! | [`classification::log_loss`] | Cross-entropy loss for probabilistic classifiers |
//! | [`classification::top_k_accuracy_score`] | Fraction of samples with true label in top-k predictions |
//! | [`classification::calibration_curve`] | Calibration curve (reliability diagram) data |
//!
//! ## Ranking Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`ranking::dcg_score`] | Discounted Cumulative Gain |
//! | [`ranking::ndcg_score`] | Normalized Discounted Cumulative Gain |
//!
//! ## Regression Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`regression::mean_absolute_error`] | Mean of absolute residuals |
//! | [`regression::mean_squared_error`] | Mean of squared residuals |
//! | [`regression::root_mean_squared_error`] | Square root of MSE |
//! | [`regression::r2_score`] | Coefficient of determination |
//! | [`regression::mean_absolute_percentage_error`] | Mean absolute percentage error |
//! | [`regression::explained_variance_score`] | Fraction of variance explained |
//!
//! ## Pairwise Distances
//!
//! | Function | Description |
//! |---|---|
//! | [`pairwise::pairwise_distances`] | Distance matrix with selectable metric |
//! | [`pairwise::euclidean_distances`] | Optimised L2 pairwise distances |
//! | [`pairwise::manhattan_distances`] | L1 pairwise distances |
//! | [`pairwise::cosine_distances`] | `1 - cosine_similarity` pairwise distances |
//! | [`pairwise::chebyshev_distances`] | L-infinity pairwise distances |
//!
//! ## Example
//!
//! ```rust
//! use ferrolearn_metrics::classification::{accuracy_score, f1_score, Average};
//! use ferrolearn_metrics::regression::r2_score;
//! use ndarray::array;
//!
//! // Classification
//! let y_true = array![0usize, 1, 2, 1, 0];
//! let y_pred = array![0usize, 1, 2, 0, 0];
//! let acc = accuracy_score(&y_true, &y_pred).unwrap();
//! assert!((acc - 0.8).abs() < 1e-10);
//!
//! // Regression
//! let y_true_reg = array![1.0_f64, 2.0, 3.0];
//! let y_pred_reg = array![1.0_f64, 2.0, 3.0];
//! let r2 = r2_score(&y_true_reg, &y_pred_reg).unwrap();
//! assert!((r2 - 1.0).abs() < 1e-10);
//! ```

pub mod classification;
pub mod clustering;
pub mod pairwise;
pub mod ranking;
pub mod regression;
pub mod scorer;

// Flat re-exports for convenient access.
pub use classification::{
    Average, accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    brier_score_loss, calibration_curve, classification_report, cohen_kappa_score,
    confusion_matrix, d2_brier_score, d2_log_loss_score, det_curve, f1_score, fbeta_score,
    hamming_loss, hinge_loss, jaccard_score, log_loss, matthews_corrcoef,
    multilabel_confusion_matrix, precision_recall_curve, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve, top_k_accuracy_score, zero_one_loss,
};
pub use clustering::{
    NmiMethod, adjusted_mutual_info, adjusted_rand_score, calinski_harabasz_score,
    completeness_score, contingency_matrix, davies_bouldin_score, fowlkes_mallows_score,
    homogeneity_completeness_v_measure, homogeneity_score, mutual_info_score,
    normalized_mutual_info_score, pair_confusion_matrix, rand_score, silhouette_samples,
    silhouette_score, v_measure_score,
};
pub use pairwise::{
    DistanceMetric, Metric, PairwiseKernel, chebyshev_distances, cosine_distances,
    euclidean_distances, manhattan_distances, nan_euclidean_distances, pairwise_distances,
    pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_kernels,
};
pub use ranking::{
    coverage_error, dcg_score, label_ranking_average_precision_score, label_ranking_loss,
    ndcg_score,
};
pub use regression::{
    d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, explained_variance_score,
    max_error, mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance,
    mean_pinball_loss, mean_poisson_deviance, mean_squared_error, mean_squared_log_error,
    mean_tweedie_deviance, median_absolute_error, r2_score, root_mean_squared_error,
    root_mean_squared_log_error,
};
pub use scorer::{
    BUILTIN_SCORER_NAMES, Scorer, ScoringInput, check_scoring, get_scorer, get_scorer_names,
    make_scorer,
};
