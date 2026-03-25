//! Multi-label binarizer.
//!
//! Transforms a list of label sets into a multi-hot binary indicator matrix.
//! Each sample can belong to zero or more classes simultaneously.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::multi_label_binarizer::MultiLabelBinarizer;
//! use ferrolearn_core::traits::{Fit, Transform};
//!
//! let mlb = MultiLabelBinarizer::new();
//! let y = vec![vec![0, 1], vec![1, 2], vec![0]];
//! let fitted = mlb.fit(&y, &()).unwrap();
//! let mat = fitted.transform(&y).unwrap();
//! // 3 classes → (3, 3) multi-hot matrix
//! assert_eq!(mat.shape(), &[3, 3]);
//! assert_eq!(mat[[0, 0]], 1.0); // sample 0 has label 0
//! assert_eq!(mat[[0, 1]], 1.0); // sample 0 has label 1
//! assert_eq!(mat[[0, 2]], 0.0); // sample 0 does NOT have label 2
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// MultiLabelBinarizer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted multi-label binarizer.
///
/// Calling [`Fit::fit`] on a `&[Vec<usize>]` discovers the sorted set of all
/// unique labels across all samples and returns a [`FittedMultiLabelBinarizer`].
#[derive(Debug, Clone, Default)]
pub struct MultiLabelBinarizer;

impl MultiLabelBinarizer {
    /// Create a new `MultiLabelBinarizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// FittedMultiLabelBinarizer
// ---------------------------------------------------------------------------

/// A fitted multi-label binarizer holding the discovered class set.
///
/// Created by calling [`Fit::fit`] on a [`MultiLabelBinarizer`].
#[derive(Debug, Clone)]
pub struct FittedMultiLabelBinarizer {
    /// Sorted unique class labels observed during fitting.
    classes: Vec<usize>,
}

impl FittedMultiLabelBinarizer {
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

    /// Map a multi-hot indicator matrix back to label sets.
    ///
    /// Each column value is thresholded at 0.5: values >= 0.5 are included.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does
    /// not match the number of classes.
    pub fn inverse_transform(&self, y: &Array2<f64>) -> Result<Vec<Vec<usize>>, FerroError> {
        let k = self.classes.len();
        if y.ncols() != k {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.nrows(), k],
                actual: vec![y.nrows(), y.ncols()],
                context: "FittedMultiLabelBinarizer::inverse_transform".into(),
            });
        }

        let n = y.nrows();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut labels = Vec::new();
            for (j, &cls) in self.classes.iter().enumerate() {
                if y[[i, j]] >= 0.5 {
                    labels.push(cls);
                }
            }
            result.push(labels);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Vec<Vec<usize>>, ()> for MultiLabelBinarizer {
    type Fitted = FittedMultiLabelBinarizer;
    type Error = FerroError;

    /// Fit the binarizer by discovering all unique labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input is empty.
    fn fit(
        &self,
        y: &Vec<Vec<usize>>,
        _target: &(),
    ) -> Result<FittedMultiLabelBinarizer, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MultiLabelBinarizer::fit".into(),
            });
        }

        let mut classes: Vec<usize> = y.iter().flatten().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        Ok(FittedMultiLabelBinarizer { classes })
    }
}

impl Transform<Vec<Vec<usize>>> for FittedMultiLabelBinarizer {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform label sets into a multi-hot indicator matrix.
    ///
    /// Each row has a `1.0` in every column corresponding to one of its labels
    /// and `0.0` elsewhere.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if any label was not seen
    /// during fitting.
    fn transform(&self, y: &Vec<Vec<usize>>) -> Result<Array2<f64>, FerroError> {
        let k = self.classes.len();
        let n = y.len();

        // Build lookup: class_value → column index
        let class_to_idx: std::collections::HashMap<usize, usize> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let mut out = Array2::zeros((n, k));

        for (i, labels) in y.iter().enumerate() {
            for &label in labels {
                let &idx = class_to_idx.get(&label).ok_or_else(|| {
                    FerroError::InvalidParameter {
                        name: "y".into(),
                        reason: format!("unknown label {label} not seen during fit"),
                    }
                })?;
                out[[i, idx]] = 1.0;
            }
        }

        Ok(out)
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
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![2, 0], vec![1]];
        let fitted = mlb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[0, 1, 2]);
    }

    #[test]
    fn test_fit_empty_input_error() {
        let mlb = MultiLabelBinarizer::new();
        let y: Vec<Vec<usize>> = vec![];
        assert!(mlb.fit(&y, &()).is_err());
    }

    #[test]
    fn test_transform_multi_hot() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[3, 3]);
        // Row 0: labels {0, 2} → [1, 0, 1]
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[0, 2]], 1.0);
        // Row 1: labels {1} → [0, 1, 0]
        assert_eq!(mat[[1, 0]], 0.0);
        assert_eq!(mat[[1, 1]], 1.0);
        assert_eq!(mat[[1, 2]], 0.0);
        // Row 2: labels {0, 1, 2} → [1, 1, 1]
        assert_eq!(mat[[2, 0]], 1.0);
        assert_eq!(mat[[2, 1]], 1.0);
        assert_eq!(mat[[2, 2]], 1.0);
    }

    #[test]
    fn test_transform_unknown_label_error() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let y2 = vec![vec![0, 5]]; // 5 not in {0, 1}
        assert!(fitted.transform(&y2).is_err());
    }

    #[test]
    fn test_inverse_transform_roundtrip() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 2], vec![1], vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_inverse_transform_shape_mismatch() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        // 3 classes expects 3 columns
        let bad = Array2::<f64>::zeros((2, 2));
        assert!(fitted.inverse_transform(&bad).is_err());
    }

    #[test]
    fn test_empty_label_set() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1], vec![]]; // second sample has no labels
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[2, 2]);
        // Row 1 should be all zeros
        assert_eq!(mat[[1, 0]], 0.0);
        assert_eq!(mat[[1, 1]], 0.0);
    }

    #[test]
    fn test_inverse_transform_empty_row() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1], vec![]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_non_contiguous_classes() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![10, 30], vec![20]];
        let fitted = mlb.fit(&y, &()).unwrap();
        assert_eq!(fitted.classes(), &[10, 20, 30]);
        let mat = fitted.transform(&y).unwrap();
        assert_eq!(mat.shape(), &[2, 3]);
        assert_eq!(mat[[0, 0]], 1.0); // 10
        assert_eq!(mat[[0, 1]], 0.0); // 20
        assert_eq!(mat[[0, 2]], 1.0); // 30
    }

    #[test]
    fn test_inverse_transform_non_contiguous_roundtrip() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![10, 30], vec![20]];
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, y);
    }

    #[test]
    fn test_duplicate_labels_in_input() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 0, 1]]; // duplicate 0
        let fitted = mlb.fit(&y, &()).unwrap();
        let mat = fitted.transform(&y).unwrap();
        // Still produces [1, 1] — duplicates don't cause double-counting
        assert_eq!(mat.shape(), &[1, 2]);
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 1.0);
    }

    #[test]
    fn test_inverse_threshold() {
        let mlb = MultiLabelBinarizer::new();
        let y = vec![vec![0, 1, 2]];
        let fitted = mlb.fit(&y, &()).unwrap();
        // Values below 0.5 → not included
        let mat = array![[0.4, 0.6, 0.5]];
        let recovered = fitted.inverse_transform(&mat).unwrap();
        assert_eq!(recovered, vec![vec![1, 2]]); // 0.4 < 0.5 so label 0 excluded
    }
}
