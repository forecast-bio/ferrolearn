//! Categorical Naive Bayes classifier.
//!
//! This module provides [`CategoricalNB`], suitable for features that take on
//! one of K discrete categorical values (encoded as non-negative integers cast
//! to floats). Each feature column may have a different number of categories.
//!
//! The log-likelihood for feature `j` in class `c` taking value `k` is:
//!
//! ```text
//! log P(x_j = k | c) = log( (N_cjk + alpha) / (N_c + alpha * K_j) )
//! ```
//!
//! where `N_cjk` is the count of feature `j` equal to `k` in class `c`,
//! `N_c` is the total number of samples in class `c`, `K_j` is the number
//! of distinct categories for feature `j`, and `alpha` is the Laplace
//! smoothing parameter.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::CategoricalNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         0.0, 1.0, 2.0,
//!         1.0, 0.0, 2.0,
//!         0.0, 1.0, 1.0,
//!         2.0, 0.0, 0.0,
//!         2.0, 1.0, 0.0,
//!         1.0, 0.0, 1.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = CategoricalNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::marker::PhantomData;

/// Categorical Naive Bayes classifier.
///
/// Suitable for features where each column takes on one of several discrete
/// categorical values (encoded as non-negative integers cast to floats).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct CategoricalNB<F: Float + Send + Sync + 'static> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    alpha: F,
    _marker: PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> CategoricalNB<F> {
    /// Create a new `CategoricalNB` with Laplace smoothing (`alpha = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            _marker: PhantomData,
        }
    }

    /// Set the Laplace smoothing parameter.
    ///
    /// # Panics
    ///
    /// Does not panic. Invalid alpha values are caught at `fit()` time.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for CategoricalNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Categorical Naive Bayes classifier.
///
/// Stores the per-class log prior and per-feature, per-class, per-category
/// log probabilities computed during fitting.
#[derive(Debug, Clone)]
pub struct FittedCategoricalNB<F: Float + Send + Sync + 'static> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, length `n_classes`.
    class_log_prior: Vec<F>,
    /// For each feature i and class c, store log P(x_i = k | c) for each category k.
    /// Indexed as `feature_log_prob[feature_idx][class_idx][category_idx]`.
    /// Categories are mapped from their integer value to a contiguous index via
    /// `categories[feature_idx]`.
    feature_log_prob: Vec<Vec<Vec<F>>>,
    /// For each feature, the sorted list of known category values.
    /// `categories[feature_idx]` is a `Vec<usize>` of category integer values.
    categories: Vec<Vec<usize>>,
    /// Number of features the model was fitted on.
    n_features: usize,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for CategoricalNB<F> {
    type Fitted = FittedCategoricalNB<F>;
    type Error = FerroError;

    /// Fit the Categorical NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if `alpha <= 0`.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedCategoricalNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "CategoricalNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "alpha must be positive for CategoricalNB".into(),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let n_f = F::from(n_samples).unwrap();

        // Compute log priors.
        let mut class_log_prior = Vec::with_capacity(n_classes);
        let mut class_indices: Vec<Vec<usize>> = Vec::with_capacity(n_classes);

        for &class_label in &classes {
            let indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();
            let n_c = F::from(indices.len()).unwrap();
            class_log_prior.push((n_c / n_f).ln());
            class_indices.push(indices);
        }

        // For each feature, discover categories and compute log probabilities.
        let mut feature_log_prob: Vec<Vec<Vec<F>>> = Vec::with_capacity(n_features);
        let mut categories_per_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Discover all categories for this feature across all samples.
            let mut cats: Vec<usize> = Vec::new();
            for i in 0..n_samples {
                let val = x[[i, j]].to_usize().unwrap_or(0);
                cats.push(val);
            }
            cats.sort_unstable();
            cats.dedup();
            let n_cats = cats.len();
            let n_cats_f = F::from(n_cats).unwrap();

            // For each class, compute log P(x_j = k | c) for each category k.
            let mut class_log_probs: Vec<Vec<F>> = Vec::with_capacity(n_classes);
            for (ci, _class_label) in classes.iter().enumerate() {
                let n_c = class_indices[ci].len();
                let n_c_f = F::from(n_c).unwrap();
                let denom = n_c_f + self.alpha * n_cats_f;

                let mut cat_log_probs: Vec<F> = Vec::with_capacity(n_cats);
                for &cat_val in &cats {
                    // Count how many samples in this class have this category value.
                    let count = class_indices[ci]
                        .iter()
                        .filter(|&&sample_idx| {
                            x[[sample_idx, j]].to_usize().unwrap_or(0) == cat_val
                        })
                        .count();
                    let count_f = F::from(count).unwrap();
                    let log_prob = ((count_f + self.alpha) / denom).ln();
                    cat_log_probs.push(log_prob);
                }
                class_log_probs.push(cat_log_probs);
            }

            feature_log_prob.push(class_log_probs);
            categories_per_feature.push(cats);
        }

        Ok(FittedCategoricalNB {
            classes,
            class_log_prior,
            feature_log_prob,
            categories: categories_per_feature,
            n_features,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedCategoricalNB<F> {
    /// Look up the log probability for a given feature, class, and category value.
    ///
    /// If the category value was not seen during training, returns a uniform
    /// log probability based on the number of known categories for that feature.
    fn log_prob_for(&self, feature_idx: usize, class_idx: usize, cat_value: usize) -> F {
        let cats = &self.categories[feature_idx];
        match cats.binary_search(&cat_value) {
            Ok(cat_idx) => self.feature_log_prob[feature_idx][class_idx][cat_idx],
            Err(_) => {
                // Unseen category: use uniform probability 1 / (n_known_cats + 1).
                // This gracefully degrades for unseen categories.
                let n_cats_plus_one = F::from(cats.len() + 1).unwrap();
                (F::one() / n_cats_plus_one).ln()
            }
        }
    }

    /// Compute joint log-likelihood for each class.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.class_log_prior[ci];
                for j in 0..self.n_features {
                    let cat_value = x[[i, j]].to_usize().unwrap_or(0);
                    score = score + self.log_prob_for(j, ci, cat_value);
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
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedCategoricalNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
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

impl<F: Float + Send + Sync + 'static> HasClasses for FittedCategoricalNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for CategoricalNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedCategoricalNBPipeline(fitted)))
    }
}

struct FittedCategoricalNBPipeline<F: Float + Send + Sync + 'static>(FittedCategoricalNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedCategoricalNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedCategoricalNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedCategoricalNBPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_categorical_data() -> (Array2<f64>, Array1<usize>) {
        // Categorical features: 3 features, each taking values in {0, 1, 2}.
        // Class 0 tends to have low values, class 1 tends to have high values.
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 1.0, // class 0
                1.0, 0.0, 0.0, // class 0
                0.0, 1.0, 0.0, // class 0
                0.0, 0.0, 0.0, // class 0
                2.0, 2.0, 1.0, // class 1
                2.0, 1.0, 2.0, // class 1
                1.0, 2.0, 2.0, // class 1
                2.0, 2.0, 2.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_categorical_nb_fit_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        // Should get most or all correct on training data.
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_categorical_nb_predict_proba_sums_to_one() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_has_classes() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_categorical_nb_alpha_smoothing_effect() {
        let (x, y) = make_categorical_data();

        // With small alpha, sharper probabilities.
        let model_sharp = CategoricalNB::<f64>::new().with_alpha(0.01);
        let fitted_sharp = model_sharp.fit(&x, &y).unwrap();
        let proba_sharp = fitted_sharp.predict_proba(&x).unwrap();

        // With large alpha, smoother probabilities (closer to uniform).
        let model_smooth = CategoricalNB::<f64>::new().with_alpha(100.0);
        let fitted_smooth = model_smooth.fit(&x, &y).unwrap();
        let proba_smooth = fitted_smooth.predict_proba(&x).unwrap();

        // Smoothed probabilities for the dominant class on a class-0 sample
        // should be less extreme.
        let sharp_max = proba_sharp[[0, 0]].max(proba_sharp[[0, 1]]);
        let smooth_max = proba_smooth[[0, 0]].max(proba_smooth[[0, 1]]);
        assert!(smooth_max < sharp_max);
    }

    #[test]
    fn test_categorical_nb_invalid_alpha_zero() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(0.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_categorical_nb_invalid_alpha_negative() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![0.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_categorical_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = array![2usize, 2, 2];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[2]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 2));
    }

    #[test]
    fn test_categorical_nb_default() {
        let model = CategoricalNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_categorical_nb_unseen_category() {
        // Fit on categories {0, 1}, then predict with a sample containing
        // category 5 (unseen). Should not panic, should return valid probabilities.
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Predict with unseen category 5 in feature 0.
        let x_new = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();
        assert_eq!(preds.len(), 1);

        let proba = fitted.predict_proba(&x_new).unwrap();
        assert_relative_eq!(proba.row(0).sum(), 1.0, epsilon = 1e-10);
        // Both probabilities should be between 0 and 1.
        assert!(proba[[0, 0]] > 0.0 && proba[[0, 0]] < 1.0);
        assert!(proba[[0, 1]] > 0.0 && proba[[0, 1]] < 1.0);
    }

    #[test]
    fn test_categorical_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_categorical_nb_pipeline() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_categorical_nb_predict_proba_ordering() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // First 4 samples are class 0 — class 0 probability should be higher.
        for i in 0..4 {
            assert!(
                proba[[i, 0]] > proba[[i, 1]],
                "sample {i}: P(c=0)={} should be > P(c=1)={}",
                proba[[i, 0]],
                proba[[i, 1]]
            );
        }
        // Last 4 samples are class 1 — class 1 probability should be higher.
        for i in 4..8 {
            assert!(
                proba[[i, 1]] > proba[[i, 0]],
                "sample {i}: P(c=1)={} should be > P(c=0)={}",
                proba[[i, 1]],
                proba[[i, 0]]
            );
        }
    }

    #[test]
    fn test_categorical_nb_f32() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = CategoricalNB::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);

        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            let sum: f32 = proba.row(i).sum();
            assert!((sum - 1.0f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_categorical_nb_unordered_classes() {
        // Classes are not 0..n, and not contiguous.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = array![5usize, 5, 5, 10, 10, 10];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5, 10]);

        let preds = fitted.predict(&x).unwrap();
        // First 3 should predict class 5.
        for i in 0..3 {
            assert_eq!(preds[i], 5);
        }
        // Last 3 should predict class 10.
        for i in 3..6 {
            assert_eq!(preds[i], 10);
        }
    }
}
