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
pub mod regression;

// Flat re-exports for convenient access.
pub use classification::{
    Average, accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
};
pub use regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score, root_mean_squared_error,
};
