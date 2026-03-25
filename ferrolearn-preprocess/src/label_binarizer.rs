//! One-vs-rest label binarizer.
//!
//! Transforms a vector of integer class labels into a binary indicator matrix.
//! For *K* classes the output has *K* columns (one-hot rows), except in the
//! binary case (*K* = 2) where a single column is produced.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::label_binarizer::LabelBinarizer;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let lb = LabelBinarizer::new();
//! let y = array![0_usize, 1, 2, 1];
//! let fitted = lb.fit(&y, &()).unwrap();
//! let mat = fitted.transform(&y).unwrap();
//! // 3 classes → (4, 3) indicator matrix
//! assert_eq!(mat.shape(), &[4, 3]);
//! assert_eq!(mat[[0, 0]], 1.0);
//! assert_eq!(mat[[0, 1]], 0.0);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// LabelBinarizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted one-vs-rest label binarizer.
///
/// Calling [`Fit::fit`] on an `Array1<usize>` discovers the sorted set of
/// unique class labels and returns a [`FittedLabelBinarizer`].
#[derive(Debug, Clone, Default)]
pub struct LabelBinarizer;

impl LabelBinarizer {
    /// Create a new `LabelBinarizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// FittedLabelBinarizer
// ---------------------------------------------------------------------------

/// A fitted label binarizer holding the discovered class set.
///
/// Created by calling [`Fit::fit`] on a [`LabelBinarizer`].
#[derive(Debug, Clone)]
pub struct FittedLabelBinarizer {
    /// Sorted unique class labels observed during fitting.
    classes: Vec<usize>,
}

impl FittedLabelBinarizer {
    /// Return the sorted class labels discovered during fitting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Return the number of unique classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Map a binary indicator matrix back to integer class labels.
    ///
    /// For each row the class with the largest value (argmax) is chosen.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does
    /// not match the expected output width (1 for binary, *K* for multiclass).
    pub fn inverse_transform(&self, y: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let k = self.classes.len();
        let expected_cols = if k == 2 { 1 } else { k };

        if y.ncols() != expected_cols {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.nrows(), expected_cols],
                actual: vec![y.nrows(), y.ncols()],
                context: "FittedLabelBinarizer::inverse_transform".into(),
            });
        }

        let n = y.nrows();
        let mut result = Array1::zeros(n);

        if k == 2 {
            // Single column: threshold at 0.5
            for i in 0..n {
                result[i] = if y[[i, 0]] >= 0.5 {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: argmax per row
            for i in 0..n {
                let row = y.row(i);
                let mut best_j = 0;
                let mut best_v = f64::NEG_INFINITY;
                for (j, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best_j = j;
                    }
                }
                result[i] = self.classes[best_j];
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array1<usize>, ()> for LabelBinarizer {
    type Fitted = FittedLabelBinarizer;
    type Error = FerroError;

    /// Fit the binarizer by discovering unique class labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input is empty.
    fn fit(&self, y: &Array1<usize>, _target: &()) -> Result<FittedLabelBinarizer, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LabelBinarizer::fit".into(),
            });
        }

        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        Ok(FittedLabelBinarizer { classes })
    }
}

impl Transform<Array1<usize>> for FittedLabelBinarizer {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform labels into a binary indicator matrix.
    ///
    /// - For *K* = 2 classes the output shape is `(n, 1)`.
    /// - For *K* > 2 classes the output shape is `(n, K)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if any label in `y` was not
    /// seen during fitting.
    fn transform(&self, y: &Array1<usize>) -> Result<Array2<f64>, FerroError> {
        let k = self.classes.len();
        let n = y.len();

        // Build a lookup: class_value → column index
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        if k == 2 {
            // Binary: single column, 1.0 for the second class
            let mut out = Array2::zeros((n, 1));
            for (i, &label) in y.iter().enumerate() {
                let idx = class_to_idx.get(&label).ok_or_else(|| {
                    FerroError::InvalidParameter {
                        name: "y".into(),
                        reason: format!("unknown label {label} not seen during fit"),
                    }
                })?;
                out[[i, 0]] = if *idx == 1 { 1.0 } else { 0.0 };
            }
            Ok(out)
        } else {
            // Multiclass: one-hot rows
            let mut out = Array2::zeros((n, k));
            for (i, &label) in y.iter().enumerate() {
                let &idx = class_to_idx.get(&label).ok_or_else(|| {
                    FerroError::InvalidParameter {
                        name: "y".into(),
                        reason: format!("unknown label {label} not seen during fit"),
                    }
                })?;
                out[[i, idx]] = 1.0;
            }
            Ok(out)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fit_discovers_sorted_classes() {
        let lb = LabelBinarizer::new();
        let y = array![2_usize, 0, 1, 2, 0];
        let fitted = lb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[0, 1, 2]);
    }

    #[test]
    fn test_fit_empty_input_error() {
        let lb = LabelBinarizer::new();
        let y: Array1<usize> = Array1::zeros(0);
        assert!(lb.fit(&y, &()).is_err());
    }

    #[test]
    fn test_binary_transform_single_column() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 0, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 1]);
        assert_eq!(mat[[0, 0]], 0.0); // class 0 → 0
        assert_eq!(mat[[1, 0]], 1.0); // class 1 → 1
        assert_eq!(mat[[2, 0]], 0.0);
        assert_eq!(mat[[3, 0]], 1.0);
    }

    #[test]
    fn test_multiclass_transform_indicator_matrix() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 3]);
        // Row 0: class 0 → [1, 0, 0]
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[0, 2]], 0.0);
        // Row 2: class 2 → [0, 0, 1]
        assert_eq!(mat[[2, 0]], 0.0);
        assert_eq!(mat[[2, 1]], 0.0);
        assert_eq!(mat[[2, 2]], 1.0);
    }

    #[test]
    fn test_inverse_transform_multiclass() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_inverse_transform_binary() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 0, 1];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_transform_unknown_label_error() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2];
        let fitted = lb.fit(&y, &()).unwrap();
        let y2 = array![0_usize, 3]; // 3 not in {0,1,2}
        assert!(fitted.transform(&y2).is_err());
    }

    #[test]
    fn test_inverse_transform_shape_mismatch() {
        let lb = LabelBinarizer::new();
        let y = array![0_usize, 1, 2];
        let fitted = lb.fit(&y, &()).unwrap();
        // 3 classes expects 3 columns, but we give 2
        let bad = Array2::<f64>::zeros((2, 2));
        assert!(fitted.inverse_transform(&bad).is_err());
    }

    #[test]
    fn test_single_class() {
        let lb = LabelBinarizer::new();
        let y = array![5_usize, 5, 5];
        let fitted = lb.fit(&y, &()).unwrap();
        assert_eq!(fitted.n_classes(), 1);
        // 1 class → 1 column (all zeros since it's the only class)
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[3, 1]);
    }

    #[test]
    fn test_non_contiguous_classes() {
        let lb = LabelBinarizer::new();
        let y = array![10_usize, 20, 30, 10];
        let fitted = lb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[10, 20, 30]);
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[4, 3]);
        assert_eq!(mat[[0, 0]], 1.0); // 10 → col 0
        assert_eq!(mat[[1, 1]], 1.0); // 20 → col 1
        assert_eq!(mat[[2, 2]], 1.0); // 30 → col 2
    }

    #[test]
    fn test_roundtrip_multiclass_non_contiguous() {
        let lb = LabelBinarizer::new();
        let y = array![10_usize, 20, 30, 20];
        let fitted = lb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }
}
