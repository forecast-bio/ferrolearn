//! Multinomial Naive Bayes classifier.
//!
//! This module provides [`MultinomialNB`], suitable for discrete count data
//! (e.g., word counts in text classification). Features must be non-negative.
//!
//! The log-likelihood for feature `j` in class `c` is:
//!
//! ```text
//! log theta_cj = log( (N_cj + alpha) / (N_c + alpha * n_features) )
//! ```
//!
//! where `N_cj` is the total count of feature `j` in class `c`, `N_c` is the
//! total count of all features in class `c`, and `alpha` is the Laplace (add-1)
//! smoothing parameter.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::MultinomialNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         3.0, 1.0, 0.0,
//!         2.0, 0.0, 1.0,
//!         4.0, 2.0, 0.0,
//!         0.0, 1.0, 4.0,
//!         1.0, 0.0, 3.0,
//!         0.0, 2.0, 5.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = MultinomialNB::<f64>::new();
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

/// Multinomial Naive Bayes classifier.
///
/// Suitable for discrete count data. Features must be non-negative.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct MultinomialNB<F> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    pub alpha: F,
}

impl<F: Float> MultinomialNB<F> {
    /// Create a new `MultinomialNB` with Laplace smoothing (`alpha = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self { alpha: F::one() }
    }

    /// Set the Laplace smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> Default for MultinomialNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Multinomial Naive Bayes classifier.
#[derive(Debug, Clone)]
pub struct FittedMultinomialNB<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, shape `(n_classes,)`.
    log_prior: Array1<F>,
    /// Log feature probabilities per class, shape `(n_classes, n_features)`.
    log_theta: Array2<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for MultinomialNB<F> {
    type Fitted = FittedMultinomialNB<F>;
    type Error = FerroError;

    /// Fit the Multinomial NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if any feature value is negative.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedMultinomialNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MultinomialNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Validate non-negative features.
        if x.iter().any(|&v| v < F::zero()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "MultinomialNB requires non-negative feature values".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_f = F::from(n_samples).unwrap();
        let n_feat_f = F::from(n_features).unwrap();

        let mut log_prior = Array1::<F>::zeros(n_classes);
        let mut log_theta = Array2::<F>::zeros((n_classes, n_features));

        for (ci, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_c = class_indices.len();
            let n_c_f = F::from(n_c).unwrap();
            log_prior[ci] = (n_c_f / n_f).ln();

            // Sum of feature counts for this class.
            let mut feature_counts = Array1::<F>::zeros(n_features);
            for &i in &class_indices {
                for j in 0..n_features {
                    feature_counts[j] = feature_counts[j] + x[[i, j]];
                }
            }

            // Total count across all features for this class.
            let total_count = feature_counts.sum();

            // Smoothed log probabilities.
            let denom = total_count + self.alpha * n_feat_f;
            for j in 0..n_features {
                log_theta[[ci, j]] = ((feature_counts[j] + self.alpha) / denom).ln();
            }
        }

        Ok(FittedMultinomialNB {
            classes,
            log_prior,
            log_theta,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedMultinomialNB<F> {
    /// Compute joint log-likelihood for each class.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let n_features = x.ncols();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.log_prior[ci];
                for j in 0..n_features {
                    score = score + x[[i, j]] * self.log_theta[[ci, j]];
                }
                scores[[i, ci]] = score;
            }
        }

        scores
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_fitted = self.log_theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted MultinomialNB".into(),
            });
        }

        let log_scores = self.joint_log_likelihood(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedMultinomialNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features_fitted = self.log_theta.ncols();
        if x.ncols() != n_features_fitted {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_features_fitted],
                actual: vec![x.ncols()],
                context: "number of features must match fitted MultinomialNB".into(),
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

impl<F: Float + Send + Sync + 'static> HasClasses for FittedMultinomialNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for f64.
impl PipelineEstimator for MultinomialNB<f64> {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedMultinomialNBPipeline(fitted)))
    }
}

struct FittedMultinomialNBPipeline(FittedMultinomialNB<f64>);

unsafe impl Send for FittedMultinomialNBPipeline {}
unsafe impl Sync for FittedMultinomialNBPipeline {}

impl FittedPipelineEstimator for FittedMultinomialNBPipeline {
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

    fn make_count_data() -> (Array2<f64>, Array1<usize>) {
        // Simple word-count like data: two classes
        // class 0 has many feature 0, class 1 has many feature 2.
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                5.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 0.0,
                2.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_multinomial_nb_fit_predict() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 6);
    }

    #[test]
    fn test_multinomial_nb_predict_proba_sums_to_one() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multinomial_nb_has_classes() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_multinomial_nb_alpha_smoothing_effect() {
        let (x, y) = make_count_data();

        // With alpha = 0, very sharp probabilities.
        let model_sharp = MultinomialNB::<f64>::new().with_alpha(0.0);
        let fitted_sharp = model_sharp.fit(&x, &y).unwrap();
        let proba_sharp = fitted_sharp.predict_proba(&x).unwrap();

        // With alpha = 100, very smoothed probabilities (closer to uniform).
        let model_smooth = MultinomialNB::<f64>::new().with_alpha(100.0);
        let fitted_smooth = model_smooth.fit(&x, &y).unwrap();
        let proba_smooth = fitted_smooth.predict_proba(&x).unwrap();

        // Smoothed probabilities for class 0 on class-0 samples should be less extreme.
        // i.e., max probability should be lower with high alpha.
        assert!(proba_smooth[[0, 0]] < proba_sharp[[0, 0]]);
    }

    #[test]
    fn test_multinomial_nb_negative_features_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -1.0, 3.0, 2.0, 1.0, 0.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = MultinomialNB::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "X"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_multinomial_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![1.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = MultinomialNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_multinomial_nb_shape_mismatch_predict() {
        let (x, y) = make_count_data();
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_multinomial_nb_single_class() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0])
            .unwrap();
        let y = array![2usize, 2, 2];
        let model = MultinomialNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[2]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 2));
    }

    #[test]
    fn test_multinomial_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = MultinomialNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_multinomial_nb_default() {
        let model = MultinomialNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }
}
