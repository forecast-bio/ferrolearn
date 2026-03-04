//! # ferrolearn-model-sel
//!
//! Model selection utilities for the ferrolearn machine learning framework.
//!
//! This crate provides cross-validation, data splitting, hyperparameter search,
//! and related model selection tools:
//!
//! - [`train_test_split`] — shuffle and split data into train/test sets.
//! - [`KFold`] — k-fold cross-validation splitter.
//! - [`StratifiedKFold`] — stratified k-fold that preserves class balance.
//! - [`cross_val_score`] — evaluate a pipeline using cross-validation.
//! - [`GridSearchCV`] — exhaustive hyperparameter search over a parameter grid.
//! - [`RandomizedSearchCV`] — randomized hyperparameter search over distributions.
//! - [`TimeSeriesSplit`] — time-series aware cross-validation splitter.
//! - [`HalvingGridSearchCV`] — successive-halving hyperparameter search.
//! - [`param_grid!`] — macro for building Cartesian-product parameter grids.
//! - [`ParamValue`] / [`ParamSet`] — hyperparameter value and set types.
//! - [`distributions`] — sampling distributions for [`RandomizedSearchCV`].
//! - [`CalibratedClassifierCV`] — probability calibration via cross-validation.
//! - [`SelfTrainingClassifier`] — semi-supervised self-training meta-estimator.
//!
//! # Quick Start
//!
//! ```rust
//! use ferrolearn_model_sel::{train_test_split, KFold};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::<f64>::zeros((20, 3));
//! let y = Array1::<f64>::zeros(20);
//!
//! let (x_train, x_test, y_train, y_test) =
//!     train_test_split(&x, &y, 0.2, Some(42)).unwrap();
//! assert_eq!(x_train.nrows(), 16);
//! assert_eq!(x_test.nrows(), 4);
//!
//! let kf = KFold::new(5);
//! let folds = kf.split(20);
//! assert_eq!(folds.len(), 5);
//! ```

pub mod calibration;
pub mod cross_validation;
pub mod distributions;
pub mod grid_search;
pub mod halving_grid_search;
pub mod param_grid;
pub mod random_search;
pub mod self_training;
pub mod split;
pub mod time_series_split;

pub use calibration::{CalibratedClassifierCV, CalibrationMethod, FittedCalibratedClassifierCV};
pub use cross_validation::{CrossValidator, KFold, StratifiedKFold, cross_val_score};
pub use grid_search::{CvResults, GridSearchCV};
pub use halving_grid_search::HalvingGridSearchCV;
pub use param_grid::{ParamSet, ParamValue};
pub use random_search::RandomizedSearchCV;
pub use self_training::{FittedSelfTrainingClassifier, SelfTrainingClassifier, UNLABELED};
pub use split::train_test_split;
pub use time_series_split::TimeSeriesSplit;
