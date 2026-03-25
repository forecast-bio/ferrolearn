//! # ferrolearn-metrics
//!
//! Classification, regression, and clustering evaluation metrics for the
//! ferrolearn machine learning framework — a scikit-learn equivalent for Rust.
//!
//! ## Classification Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`classification::accuracy_score`] | Fraction of correctly classified samples |
//! | [`classification::balanced_accuracy_score`] | Mean of per-class recall |
//! | [`classification::precision_score`] | Positive predictive value (TP / (TP + FP)) |
//! | [`classification::recall_score`] | Sensitivity (TP / (TP + FN)) |
//! | [`classification::f1_score`] | Harmonic mean of precision and recall |
//! | [`classification::fbeta_score`] | Weighted harmonic mean of precision and recall |
//! | [`classification::matthews_corrcoef`] | Matthew's correlation coefficient |
//! | [`classification::cohen_kappa_score`] | Inter-rater agreement corrected for chance |
//! | [`classification::roc_auc_score`] | Area under the ROC curve (binary) |
//! | [`classification::confusion_matrix`] | Matrix of true vs. predicted counts |
//! | [`classification::log_loss`] | Cross-entropy loss for probabilistic classifiers |
//! | [`classification::brier_score_loss`] | Mean squared error of probability estimates |
//! | [`classification::hinge_loss`] | Average hinge loss for SVM-style classifiers |
//! | [`classification::hamming_loss`] | Fraction of labels that differ |
//! | [`classification::jaccard_score`] | Intersection over union |
//! | [`classification::zero_one_loss`] | Fraction or count of misclassifications |
//! | [`classification::classification_report`] | Formatted text report of per-class metrics |
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
//! | [`regression::median_absolute_error`] | Median of absolute residuals |
//! | [`regression::max_error`] | Maximum absolute error |
//! | [`regression::mean_squared_log_error`] | MSE on log-transformed values |
//! | [`regression::root_mean_squared_log_error`] | Square root of MSLE |
//!
//! ## Clustering Metrics
//!
//! | Function | Description |
//! |---|---|
//! | [`clustering::silhouette_score`] | Mean silhouette coefficient |
//! | [`clustering::silhouette_samples`] | Per-sample silhouette coefficients |
//! | [`clustering::adjusted_rand_score`] | Adjusted Rand Index |
//! | [`clustering::adjusted_mutual_info`] | Adjusted Mutual Information |
//! | [`clustering::davies_bouldin_score`] | Davies-Bouldin index |
//! | [`clustering::calinski_harabasz_score`] | Calinski-Harabasz (Variance Ratio) |
//! | [`clustering::homogeneity_score`] | Cluster homogeneity |
//! | [`clustering::completeness_score`] | Cluster completeness |
//! | [`clustering::v_measure_score`] | V-measure (harmonic mean of h & c) |
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
pub mod regression;

// Flat re-exports for convenient access.
pub use classification::{
    Average, accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, hamming_loss, hinge_loss,
    jaccard_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    zero_one_loss,
};
pub use clustering::{
    adjusted_mutual_info, adjusted_rand_score, calinski_harabasz_score, completeness_score,
    davies_bouldin_score, homogeneity_score, silhouette_samples, silhouette_score, v_measure_score,
};
pub use regression::{
    explained_variance_score, max_error, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score,
    root_mean_squared_error, root_mean_squared_log_error,
};
