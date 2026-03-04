//! Gaussian Naive Bayes classifier.
//!
//! This module provides [`GaussianNB`], a Naive Bayes classifier that assumes
//! features are normally (Gaussian) distributed within each class. Each
//! feature's likelihood is modelled by the Gaussian density
//! `N(mu_ci, sigma_ci^2)` estimated from the training data.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::GaussianNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 2.0, 1.5, 1.8, 2.0, 2.5,
//!          6.0, 7.0, 6.5, 6.8, 7.0, 7.5],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = GaussianNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Gaussian Naive Bayes classifier.
///
/// Assumes features are Gaussian-distributed within each class.
/// Variance smoothing is applied to avoid numerical issues with zero variance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GaussianNB<F> {
    /// Variance smoothing parameter added to all variances.
    /// Prevents division by zero when a feature has near-zero variance.
    /// Default: `1e-9`.
    pub var_smoothing: F,
}

impl<F: Float> GaussianNB<F> {
    /// Create a new `GaussianNB` with default settings.
    ///
    /// Default: `var_smoothing = 1e-9`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            var_smoothing: F::from(1e-9).unwrap(),
        }
    }

    /// Set the variance smoothing parameter.
    #[must_use]
    pub fn with_var_smoothing(mut self, var_smoothing: F) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }
}

impl<F: Float> Default for GaussianNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Gaussian Naive Bayes classifier.
///
/// Stores the per-class prior, mean, and variance computed during fitting.
#[derive(Debug, Clone)]
pub struct FittedGaussianNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Per-class per-feature mean, shape `(n_classes, n_features)`.
    theta: Array2<F>,
    /// Per-class per-feature variance (smoothed), shape `(n_classes, n_features)`.
    sigma: Array2<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for GaussianNB<F> {
    type Fitted = FittedGaussianNB<F>;
    type Error = FerroError;

    /// Fit the Gaussian NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedGaussianNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "GaussianNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let mut theta = Array2::<F>::zeros((n_classes, n_features));
        let mut sigma = Array2::<F>::zeros((n_classes, n_features));
        let mut log_prior = Array1::<F>::zeros(n_classes);

        let n_f = F::from(n_samples).unwrap();

        for (ci, &class_label) in classes.iter().enumerate() {
            // Collect indices of samples belonging to this class.
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_mask.len();
            let n_c_f = F::from(n_c).unwrap();

            // Log prior: log(count_c / n_total).
            log_prior[ci] = (n_c_f / n_f).ln();

            // Compute per-feature mean.
            for j in 0..n_features {
                let mean = class_mask.iter().fold(F::zero(), |acc, &i| acc + x[[i, j]]) / n_c_f;
                theta[[ci, j]] = mean;

                // Compute variance (population variance).
                let var = if n_c > 1 {
                    class_mask.iter().fold(F::zero(), |acc, &i| {
                        let diff = x[[i, j]] - mean;
                        acc + diff * diff
                    }) / n_c_f
                } else {
                    F::zero()
                };

                sigma[[ci, j]] = var;
            }
        }

        // Apply variance smoothing: add var_smoothing * max(variance_all_features).
        // Following scikit-learn: epsilon = var_smoothing * max of all variances.
        let max_var = sigma
            .iter()
            .fold(F::zero(), |acc, &v| if v > acc { v } else { acc });
        let epsilon = self.var_smoothing * max_var.max(F::one());
        sigma.mapv_inplace(|v| v + epsilon);

        Ok(FittedGaussianNB {
            classes,
            log_prior,
            theta,
            sigma,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianNB<F> {
    /// Compute the joint log-likelihood for each class.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` containing the
    /// unnormalized log-posterior scores.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let two = F::from(2.0).unwrap();
        let pi = F::from(std::f64::consts::PI).unwrap();
        let log_two_pi = (two * pi).ln();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for ci in 0..n_classes {
            for i in 0..n_samples {
                let mut log_likelihood = self.log_prior[ci];
                for j in 0..n_features {
                    let mu = self.theta[[ci, j]];
                    let var = self.sigma[[ci, j]];
                    let diff = x[[i, j]] - mu;
                    // log N(x; mu, var) = -0.5 * (log(2*pi*var) + (x-mu)^2/var)
                    log_likelihood =
                        log_likelihood - (log_two_pi + var.ln()) / two - diff * diff / (two * var);
                }
                scores[[i, ci]] = log_likelihood;
            }
        }

        scores
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns an array of shape `(n_samples, n_classes)` where each row
    /// sums to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted GaussianNB".into(),
            });
        }

        let log_scores = self.joint_log_likelihood(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            // Numerically stable softmax: subtract row max before exp.
            let max_score = log_scores
                .row(i)
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b));

            let mut row_sum = F::zero();
            for ci in 0..n_classes {
                let p = (log_scores[[i, ci]] - max_score).exp();
                proba[[i, ci]] = p;
                row_sum = row_sum + p;
            }
            for ci in 0..n_classes {
                proba[[i, ci]] = proba[[i, ci]] / row_sum;
            }
        }

        Ok(proba)
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features_fitted = self.theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted GaussianNB".into(),
            });
        }

        let scores = self.joint_log_likelihood(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_score = scores[[i, 0]];
            for ci in 1..n_classes {
                if scores[[i, ci]] > best_score {
                    best_score = scores[[i, ci]];
                    best_class = ci;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedGaussianNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for f64.
impl PipelineEstimator for GaussianNB<f64> {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedGaussianNBPipeline(fitted)))
    }
}

struct FittedGaussianNBPipeline(FittedGaussianNB<f64>);

unsafe impl Send for FittedGaussianNBPipeline {}
unsafe impl Sync for FittedGaussianNBPipeline {}

impl FittedPipelineEstimator for FittedGaussianNBPipeline {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| v as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_2class_data() -> (Array2<f64>, Array1<usize>) {
        // Two well-separated Gaussian clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.2, 0.8, 0.9, 1.1, 1.1, 0.9, // class 0
                5.0, 5.0, 5.1, 4.9, 4.8, 5.2, 5.0, 4.8, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_gaussian_nb_fit_predict_2class() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // Should classify the training data correctly.
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 8);
    }

    #[test]
    fn test_gaussian_nb_predict_proba_sums_to_one() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gaussian_nb_predict_proba_ordering() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        // First 4 samples should have higher probability of class 0.
        for i in 0..4 {
            assert!(proba[[i, 0]] > proba[[i, 1]]);
        }
        // Last 4 samples should have higher probability of class 1.
        for i in 4..8 {
            assert!(proba[[i, 1]] > proba[[i, 0]]);
        }
    }

    #[test]
    fn test_gaussian_nb_has_classes() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_gaussian_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // class 0
                5.0, 0.0, 5.1, 0.0, 5.0, 0.1, // class 1
                0.0, 5.0, 0.1, 5.0, 0.0, 5.1, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_gaussian_nb_var_smoothing_effect() {
        // When all samples in a class have identical features (zero variance),
        // var_smoothing prevents division by zero.
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 5.0, 5.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model_default = GaussianNB::<f64>::new();
        let fitted = model_default.fit(&x, &y).unwrap();
        // Should not panic — var_smoothing handles zero variance.
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);

        // With higher smoothing the model still predicts.
        let model_high = GaussianNB::<f64>::new().with_var_smoothing(0.1);
        let fitted_high = model_high.fit(&x, &y).unwrap();
        let preds_high = fitted_high.predict(&x).unwrap();
        assert_eq!(preds_high.len(), 4);
    }

    #[test]
    fn test_gaussian_nb_shape_mismatch_fit() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0usize, 1, 0]; // Wrong length
        let model = GaussianNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gaussian_nb_shape_mismatch_predict() {
        let (x, y) = make_2class_data();
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_gaussian_nb_single_class() {
        // Single class — still fits, always predicts that class.
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.5, 2.5, 0.8, 1.8]).unwrap();
        let y = array![0usize, 0, 0];

        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_gaussian_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = GaussianNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gaussian_nb_default() {
        let model = GaussianNB::<f64>::default();
        assert_relative_eq!(model.var_smoothing, 1e-9, epsilon = 1e-15);
    }

    #[test]
    fn test_gaussian_nb_pipeline() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_gaussian_nb_unordered_classes() {
        // Classes are not 0..n, and not in order in y.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 1.2, 1.1, 5.0, 4.9, 5.1]).unwrap();
        let y = array![3usize, 3, 3, 7, 7, 7];
        let model = GaussianNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[3, 7]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds[0] == 3 || preds[0] == 7);
        assert_eq!(preds[0], 3);
        assert_eq!(preds[3], 7);
    }
}
