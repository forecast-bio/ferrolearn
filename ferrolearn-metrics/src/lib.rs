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
//! | [`classification::precision_recall_fscore_support`] | Per-class precision, recall, F-score, and support |
//! | [`classification::multilabel_confusion_matrix`] | Per-class 2x2 confusion matrices |
//! | [`classification::det_curve`] | Detection Error Tradeoff curve (FNR vs FPR) |
//! | [`classification::roc_auc_score`] | Area under the ROC curve (binary) |
//! | [`classification::roc_curve`] | Compute ROC curve (FPR, TPR, thresholds) |
//! | [`classification::precision_recall_curve`] | Compute precision-recall curve |
//! | [`classification::auc`] | Area under an arbitrary curve (trapezoidal rule) |
//! | [`classification::average_precision_score`] | Average precision from PR curve |
//! | [`classification::confusion_matrix`] | Matrix of true vs. predicted counts |
//! | [`classification::log_loss`] | Cross-entropy loss for probabilistic classifiers |
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
//! | [`regression::mean_pinball_loss`] | Quantile (pinball) loss |
//! | [`regression::mean_poisson_deviance`] | Mean Poisson deviance |
//! | [`regression::mean_gamma_deviance`] | Mean Gamma deviance |
//! | [`regression::mean_tweedie_deviance`] | Mean Tweedie deviance (generalised) |
//!
//! ## Clustering Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`clustering::silhouette_score`] | Mean silhouette coefficient |
//! | [`clustering::adjusted_rand_score`] | Adjusted Rand Index |
//! | [`clustering::rand_score`] | Unadjusted Rand Index |
//! | [`clustering::adjusted_mutual_info`] | Adjusted Mutual Information |
//! | [`clustering::normalized_mutual_info_score`] | Normalized Mutual Information |
//! | [`clustering::fowlkes_mallows_score`] | Fowlkes-Mallows Index |
//! | [`clustering::davies_bouldin_score`] | Davies-Bouldin Index |
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
//! | [`pairwise::nan_euclidean_distances`] | L2 distances with NaN-feature handling |
//!
//! ## Scorer Utility
//!
//! | Item | Description |
//! |---|---|
//! | [`scorer::Scorer`] | Wraps a scoring function with optimisation metadata |
//! | [`scorer::make_scorer`] | Create a `Scorer` from a function pointer |
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
pub mod regression;
pub mod scorer;

// Flat re-exports for convenient access.
pub use classification::{
    Average, accuracy_score, auc, average_precision_score, confusion_matrix, det_curve, f1_score,
    log_loss, multilabel_confusion_matrix, precision_recall_curve, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve,
};
pub use clustering::{
    NmiMethod, adjusted_mutual_info, adjusted_rand_score, davies_bouldin_score,
    fowlkes_mallows_score, normalized_mutual_info_score, rand_score, silhouette_score,
};
pub use pairwise::{
    Metric, chebyshev_distances, cosine_distances, euclidean_distances, manhattan_distances,
    nan_euclidean_distances, pairwise_distances,
};
pub use regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance, mean_squared_error,
    mean_tweedie_deviance, r2_score, root_mean_squared_error,
};
pub use scorer::{Scorer, make_scorer};
