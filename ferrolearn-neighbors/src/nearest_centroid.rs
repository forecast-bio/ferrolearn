//! Nearest Centroid classifier.
//!
//! This module provides [`NearestCentroid`], which classifies samples by
//! computing the mean of each class during training and assigning new
//! samples to the class with the nearest centroid (Euclidean distance).
//!
//! An optional `shrink_threshold` parameter allows centroid shrinkage: each
//! class centroid is moved toward the overall centroid by subtracting a
//! threshold-dependent amount from the per-class offsets.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::nearest_centroid::NearestCentroid;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
//!     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = NearestCentroid::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// NearestCentroid
// ---------------------------------------------------------------------------

/// Nearest Centroid classifier.
///
/// Classifies samples by assigning them to the class with the nearest
/// centroid (mean) in Euclidean space.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct NearestCentroid<F> {
    /// Optional shrinkage threshold. If set, each class centroid is moved
    /// toward the overall centroid by this amount. Default: `None`.
    pub shrink_threshold: Option<F>,
}

impl<F: Float> NearestCentroid<F> {
    /// Create a new `NearestCentroid` with no shrinkage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shrink_threshold: None,
        }
    }

    /// Set the shrinkage threshold.
    ///
    /// When set, each per-class centroid deviation from the overall centroid
    /// is soft-thresholded (shifted toward zero), which can improve
    /// generalization and implicitly perform feature selection.
    #[must_use]
    pub fn with_shrink_threshold(mut self, threshold: F) -> Self {
        self.shrink_threshold = Some(threshold);
        self
    }
}

impl<F: Float> Default for NearestCentroid<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Nearest Centroid classifier.
///
/// Stores the class centroids and class labels computed during fitting.
#[derive(Debug, Clone)]
pub struct FittedNearestCentroid<F> {
    /// Per-class centroids, shape `(n_classes, n_features)`.
    centroids: Array2<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for NearestCentroid<F> {
    type Fitted = FittedNearestCentroid<F>;
    type Error = FerroError;

    /// Fit the Nearest Centroid classifier by computing class means.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if `shrink_threshold` is negative.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedNearestCentroid<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NearestCentroid requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if let Some(threshold) = self.shrink_threshold {
            if threshold < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "shrink_threshold".into(),
                    reason: "must be non-negative".into(),
                });
            }
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Compute per-class centroids.
        let mut centroids = Array2::<F>::zeros((n_classes, n_features));

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();
            let n_c_f = F::from(n_c).unwrap();

            for j in 0..n_features {
                let sum = class_indices
                    .iter()
                    .fold(F::zero(), |acc, &i| acc + x[[i, j]]);
                centroids[[ci, j]] = sum / n_c_f;
            }
        }

        // Apply shrinkage if requested.
        if let Some(threshold) = self.shrink_threshold {
            // Compute overall centroid.
            let mut overall = Array1::<F>::zeros(n_features);
            for j in 0..n_features {
                let sum = (0..n_samples).fold(F::zero(), |acc, i| acc + x[[i, j]]);
                overall[j] = sum / F::from(n_samples).unwrap();
            }

            // Compute within-class standard deviation per feature.
            let mut pooled_var = Array1::<F>::zeros(n_features);
            for (ci, &class_label) in classes.iter().enumerate() {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                    .collect();

                for j in 0..n_features {
                    let mean = centroids[[ci, j]];
                    let var_sum = class_indices
                        .iter()
                        .fold(F::zero(), |acc, &i| {
                            let d = x[[i, j]] - mean;
                            acc + d * d
                        });
                    pooled_var[j] = pooled_var[j] + var_sum;
                }
            }

            let denom = F::from(n_samples - n_classes).unwrap().max(F::one());
            for j in 0..n_features {
                pooled_var[j] = (pooled_var[j] / denom).sqrt();
                if pooled_var[j] < F::from(1e-10).unwrap() {
                    pooled_var[j] = F::one(); // Avoid division by zero.
                }
            }

            // Soft-threshold the deviation of each class centroid from the overall centroid.
            for ci in 0..n_classes {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &label)| {
                        if label == classes[ci] { Some(i) } else { None }
                    })
                    .collect();

                let n_c_f = F::from(class_indices.len()).unwrap();
                let m_k = (F::one() / n_c_f - F::one() / F::from(n_samples).unwrap()).sqrt();

                for j in 0..n_features {
                    let delta = (centroids[[ci, j]] - overall[j]) / (m_k * pooled_var[j]);
                    let sign = if delta > F::zero() {
                        F::one()
                    } else if delta < F::zero() {
                        -F::one()
                    } else {
                        F::zero()
                    };
                    let shrunk = (delta.abs() - threshold).max(F::zero()) * sign;
                    centroids[[ci, j]] = overall[j] + shrunk * m_k * pooled_var[j];
                }
            }
        }

        Ok(FittedNearestCentroid { centroids, classes })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedNearestCentroid<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels by finding the nearest centroid.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let n_features_fitted = self.centroids.ncols();

        if n_features != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![n_features],
                context: "number of features must match fitted NearestCentroid".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_dist = F::infinity();

            for ci in 0..n_classes {
                let dist: F = (0..n_features)
                    .map(|j| {
                        let d = x[[i, j]] - self.centroids[[ci, j]];
                        d * d
                    })
                    .fold(F::zero(), |a, b| a + b);

                if dist < best_dist {
                    best_dist = dist;
                    best_class = ci;
                }
            }

            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedNearestCentroid<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

impl<F: Float + Send + Sync + 'static> FittedNearestCentroid<F> {
    /// Get the class centroids.
    ///
    /// Returns an array of shape `(n_classes, n_features)`.
    #[must_use]
    pub fn centroids(&self) -> &Array2<F> {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_2class_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5,
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_nearest_centroid_fit_predict() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 8);
    }

    #[test]
    fn test_nearest_centroid_centroids() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        assert_eq!(centroids.nrows(), 2);
        assert_eq!(centroids.ncols(), 2);

        // Class 0: mean of (0,0), (0.5,0), (0,0.5), (0.5,0.5) = (0.25, 0.25)
        assert_relative_eq!(centroids[[0, 0]], 0.25, epsilon = 1e-10);
        assert_relative_eq!(centroids[[0, 1]], 0.25, epsilon = 1e-10);

        // Class 1: mean of (5,5), (5.5,5), (5,5.5), (5.5,5.5) = (5.25, 5.25)
        assert_relative_eq!(centroids[[1, 0]], 5.25, epsilon = 1e-10);
        assert_relative_eq!(centroids[[1, 1]], 5.25, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_centroid_has_classes() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_nearest_centroid_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
                5.0, 0.0, 5.5, 0.0, 5.0, 0.5,
                0.0, 5.0, 0.5, 5.0, 0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_nearest_centroid_with_shrinkage() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct with shrinkage, got {correct}");
    }

    #[test]
    fn test_nearest_centroid_shrinkage_high_threshold() {
        // With a very high threshold, centroids collapse to the overall mean.
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(1000.0);
        let fitted = model.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Both centroids should be very close to the overall mean.
        let overall_mean_0 = (0.0 + 0.5 + 0.0 + 0.5 + 5.0 + 5.5 + 5.0 + 5.5) / 8.0;
        assert_relative_eq!(centroids[[0, 0]], overall_mean_0, epsilon = 0.1);
        assert_relative_eq!(centroids[[1, 0]], overall_mean_0, epsilon = 0.1);
    }

    #[test]
    fn test_nearest_centroid_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = NearestCentroid::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nearest_centroid_shape_mismatch_predict() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_nearest_centroid_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = NearestCentroid::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nearest_centroid_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5]).unwrap();
        let y = array![5usize, 5, 5];
        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 5));
    }

    #[test]
    fn test_nearest_centroid_default() {
        let model = NearestCentroid::<f64>::default();
        assert!(model.shrink_threshold.is_none());
    }

    #[test]
    fn test_nearest_centroid_negative_shrink_threshold() {
        let (x, y) = make_2class_data();
        let model = NearestCentroid::<f64>::new().with_shrink_threshold(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nearest_centroid_noncontiguous_labels() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![10usize, 10, 10, 20, 20, 20];

        let model = NearestCentroid::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[10, 20]);

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 10);
        assert_eq!(preds[5], 20);
    }
}
