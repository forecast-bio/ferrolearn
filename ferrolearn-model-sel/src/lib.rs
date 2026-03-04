//! # ferrolearn-model-sel
//!
//! Model selection utilities for the ferrolearn machine learning framework.
//!
//! This crate provides cross-validation, data splitting, and related model
//! selection tools:
//!
//! - [`train_test_split`] — shuffle and split data into train/test sets.
//! - [`KFold`] — k-fold cross-validation splitter.
//! - [`StratifiedKFold`] — stratified k-fold that preserves class balance.
//! - [`cross_val_score`] — evaluate a pipeline using cross-validation.
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

pub mod cross_validation;
pub mod split;

pub use cross_validation::{CrossValidator, KFold, StratifiedKFold, cross_val_score};
pub use split::train_test_split;
