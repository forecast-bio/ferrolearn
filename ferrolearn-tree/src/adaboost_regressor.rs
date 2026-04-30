//! AdaBoost regressor.
//!
//! This module provides [`AdaBoostRegressor`], which implements the AdaBoost.R2
//! algorithm using decision tree stumps as base estimators. Three loss function
//! variants are supported: linear, square, and exponential.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::AdaBoostRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 1), vec![
//!     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
//! ]).unwrap();
//! let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let model = AdaBoostRegressor::<f64>::new()
//!     .with_n_estimators(50)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use crate::decision_tree::{self, Node, build_regression_tree_with_feature_subset};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasFeatureImportances;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// AdaBoostLoss
// ---------------------------------------------------------------------------

/// Loss function for AdaBoost.R2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaBoostLoss {
    /// Linear loss: `L_i = |y_i - y_pred_i| / max_error`.
    Linear,
    /// Square loss: `L_i = (|y_i - y_pred_i| / max_error)^2`.
    Square,
    /// Exponential loss: `L_i = 1 - exp(-|y_i - y_pred_i| / max_error)`.
    Exponential,
}

// ---------------------------------------------------------------------------
// AdaBoostRegressor
// ---------------------------------------------------------------------------

/// AdaBoost.R2 regressor using decision trees as base estimators.
///
/// Implements the AdaBoost.R2 algorithm (Drucker 1997), which iteratively
/// fits regression trees to reweighted training data. At each round, samples
/// with large errors receive higher weight, focusing subsequent estimators
/// on the hardest-to-predict examples.
///
/// The final prediction is a weighted median of the individual estimator
/// predictions.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct AdaBoostRegressor<F> {
    /// Number of boosting stages.
    pub n_estimators: usize,
    /// Learning rate (shrinkage). Lower values require more estimators.
    pub learning_rate: F,
    /// Maximum depth of each base decision tree (default 3).
    pub max_depth: Option<usize>,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Loss function for computing sample errors.
    pub loss: AdaBoostLoss,
}

impl<F: Float> AdaBoostRegressor<F> {
    /// Create a new `AdaBoostRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 50`, `learning_rate = 1.0`,
    /// `max_depth = Some(3)`, `random_state = None`,
    /// `loss = Linear`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: F::one(),
            max_depth: Some(3),
            random_state: None,
            loss: AdaBoostLoss::Linear,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: F) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum depth of each base decision tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: AdaBoostLoss) -> Self {
        self.loss = loss;
        self
    }
}

impl<F: Float> Default for AdaBoostRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedAdaBoostRegressor
// ---------------------------------------------------------------------------

/// A fitted AdaBoost.R2 regressor.
///
/// Stores the sequence of regression trees and their weights. Predictions
/// are made by weighted median of estimator predictions.
#[derive(Debug, Clone)]
pub struct FittedAdaBoostRegressor<F> {
    /// Sequence of fitted regression trees.
    estimators: Vec<Vec<Node<F>>>,
    /// Log-inverse confidence of each estimator: `ln(1 / beta_t)`.
    estimator_weights: Vec<F>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores aggregated across the boosted trees,
    /// weighted by `estimator_weights` (normalized to sum to 1).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F> for FittedAdaBoostRegressor<F> {
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> FittedAdaBoostRegressor<F> {
    /// Returns a reference to the individual tree node vectors.
    #[must_use]
    pub fn estimators(&self) -> &[Vec<Node<F>>] {
        &self.estimators
    }

    /// Returns the estimator weights (log-inverse confidence).
    #[must_use]
    pub fn estimator_weights(&self) -> &[F] {
        &self.estimator_weights
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// R² coefficient of determination on the given test data.
    /// Equivalent to sklearn's `RegressorMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<F>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::r2_score(&preds, y))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for AdaBoostRegressor<F> {
    type Fitted = FittedAdaBoostRegressor<F>;
    type Error = FerroError;

    /// Fit the AdaBoost.R2 regressor.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedAdaBoostRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "AdaBoostRegressor requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }

        let eps = F::from(1e-10).unwrap();
        let half = F::from(0.5).unwrap();
        let n_f = F::from(n_samples).unwrap();

        // Initialize sample weights uniformly.
        let mut weights = vec![F::one() / n_f; n_samples];

        let all_features: Vec<usize> = (0..n_features).collect();
        let tree_params = decision_tree::TreeParams {
            max_depth: self.max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        };

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Build weighted sample indices using systematic resampling.
            let indices = resample_weighted(&weights, n_samples);

            let tree = build_regression_tree_with_feature_subset(
                x,
                y,
                &indices,
                &all_features,
                &tree_params,
            );

            // Compute predictions on full training set.
            let mut preds = vec![F::zero(); n_samples];
            for (i, pred) in preds.iter_mut().enumerate() {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf { value, .. } = tree[leaf_idx] {
                    *pred = value;
                }
            }

            // Compute per-sample absolute errors and the maximum error.
            let mut abs_errors = vec![F::zero(); n_samples];
            let mut max_error = F::zero();
            for i in 0..n_samples {
                abs_errors[i] = (y[i] - preds[i]).abs();
                if abs_errors[i] > max_error {
                    max_error = abs_errors[i];
                }
            }

            // If max_error is zero, perfect fit; keep this estimator and stop.
            if max_error <= eps {
                estimators.push(tree);
                estimator_weights.push(F::one());
                break;
            }

            // Compute normalised loss for each sample.
            let losses: Vec<F> = abs_errors
                .iter()
                .map(|&e| {
                    let normalised = e / max_error;
                    match self.loss {
                        AdaBoostLoss::Linear => normalised,
                        AdaBoostLoss::Square => normalised * normalised,
                        AdaBoostLoss::Exponential => F::one() - (-normalised).exp(),
                    }
                })
                .collect();

            // Compute weighted average loss.
            let weight_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            let avg_loss = if weight_sum > F::zero() {
                weights
                    .iter()
                    .zip(losses.iter())
                    .map(|(&w, &l)| w * l)
                    .fold(F::zero(), |a, b| a + b)
                    / weight_sum
            } else {
                half
            };

            // If error is >= 0.5, stop (this estimator doesn't help).
            if avg_loss >= half {
                if estimators.is_empty() {
                    // Keep at least one estimator.
                    estimators.push(tree);
                    estimator_weights.push(F::one());
                }
                break;
            }

            // Compute beta = avg_loss / (1 - avg_loss).
            let beta = avg_loss / (F::one() - avg_loss).max(eps);

            // Estimator weight = ln(1 / beta) * learning_rate.
            let est_weight = (F::one() / beta.max(eps)).ln() * self.learning_rate;

            // Update sample weights: w_i *= beta^(1 - loss_i).
            for i in 0..n_samples {
                let exponent = F::one() - losses[i];
                weights[i] = weights[i] * beta.powf(exponent);
            }

            // Normalise weights.
            let new_sum: F = weights.iter().copied().fold(F::zero(), |a, b| a + b);
            if new_sum > F::zero() {
                for w in &mut weights {
                    *w = *w / new_sum;
                }
            }

            estimators.push(tree);
            estimator_weights.push(est_weight);
        }

        let feature_importances = decision_tree::aggregate_tree_importances(
            &estimators,
            None,
            Some(&estimator_weights),
            n_features,
        );

        Ok(FittedAdaBoostRegressor {
            estimators,
            estimator_weights,
            n_features,
            feature_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedAdaBoostRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values by weighted median of estimator predictions.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = x.row(i);

            // Collect (prediction, weight) pairs for this sample.
            let mut pred_weight: Vec<(F, F)> = self
                .estimators
                .iter()
                .zip(self.estimator_weights.iter())
                .map(|(tree_nodes, &w)| {
                    let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                    let val = if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                        value
                    } else {
                        F::zero()
                    };
                    (val, w)
                })
                .collect();

            predictions[i] = weighted_median(&mut pred_weight);
        }

        Ok(predictions)
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for AdaBoostRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedAdaBoostRegressor<F> {
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the weighted median of a set of `(value, weight)` pairs.
///
/// Sorts by value, then finds the value where cumulative weight reaches
/// half the total weight.
fn weighted_median<F: Float>(pairs: &mut [(F, F)]) -> F {
    if pairs.is_empty() {
        return F::zero();
    }
    if pairs.len() == 1 {
        return pairs[0].0;
    }

    // Sort by value.
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_weight: F = pairs.iter().map(|&(_, w)| w).fold(F::zero(), |a, b| a + b);
    let half = total_weight / F::from(2.0).unwrap();

    let mut cumulative = F::zero();
    for &(val, w) in pairs.iter() {
        cumulative = cumulative + w;
        if cumulative >= half {
            return val;
        }
    }

    // Fallback: last value.
    pairs.last().unwrap().0
}

/// Resample indices proportional to weights (weighted bootstrap).
///
/// Uses systematic resampling: the cumulative weight distribution
/// determines which original indices appear in the resampled set.
fn resample_weighted<F: Float>(weights: &[F], n: usize) -> Vec<usize> {
    if weights.is_empty() {
        return Vec::new();
    }

    // Build cumulative distribution.
    let mut cumsum = Vec::with_capacity(weights.len());
    let mut running = F::zero();
    for &w in weights {
        running = running + w;
        cumsum.push(running);
    }

    let total = running;
    if total <= F::zero() {
        return (0..n).collect();
    }

    let mut indices = Vec::with_capacity(n);
    let step = total / F::from(n).unwrap();
    let mut threshold = step / F::from(2.0).unwrap();
    let mut j = 0;

    for _ in 0..n {
        while j < cumsum.len() - 1 && cumsum[j] < threshold {
            j += 1;
        }
        indices.push(j);
        threshold = threshold + step;
    }

    indices
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaboost_regressor_simple() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        // AdaBoost.R2 should produce reasonable predictions.
        for i in 0..6 {
            assert!(
                (preds[i] - y[i]).abs() < 2.0,
                "pred[{i}] = {}, expected ~{}",
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_adaboost_regressor_reproducibility() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_adaboost_regressor_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0]);

        let model = AdaBoostRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = AdaBoostRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_regressor_zero_estimators() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        let model = AdaBoostRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_regressor_invalid_learning_rate() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);

        let model = AdaBoostRegressor::<f64>::new().with_learning_rate(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model = AdaBoostRegressor::<f64>::new().with_learning_rate(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_adaboost_regressor_predict_shape_mismatch() {
        let x_train = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y_train = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x_train, &y_train).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_adaboost_regressor_square_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_loss(AdaBoostLoss::Square)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_regressor_exponential_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_loss(AdaBoostLoss::Exponential)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_regressor_with_max_depth() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_max_depth(Some(1))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_adaboost_regressor_learning_rate() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_adaboost_regressor_default() {
        let model = AdaBoostRegressor::<f64>::default();
        assert_eq!(model.n_estimators, 50);
        assert_eq!(model.learning_rate, 1.0);
        assert_eq!(model.max_depth, Some(3));
        assert!(model.random_state.is_none());
        assert_eq!(model.loss, AdaBoostLoss::Linear);
    }

    #[test]
    fn test_adaboost_regressor_perfect_fit() {
        // With a single unique x, the tree should perfectly predict y.
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from(vec![5.0, 5.0, 5.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..3 {
            assert!(
                (preds[i] - 5.0).abs() < 1e-6,
                "pred[{i}] = {}, expected 5.0",
                preds[i]
            );
        }
    }

    #[test]
    fn test_adaboost_regressor_two_features() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0, 5.0, 3.0, 6.0, 3.0, 7.0, 4.0, 8.0, 4.0,
            ],
        )
        .unwrap();
        let y = Array1::from(vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]);

        let model = AdaBoostRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_weighted_median_basic() {
        let mut pairs = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        let median = weighted_median(&mut pairs);
        assert!((median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_median_unequal_weights() {
        // The value 3.0 has weight 5.0 (more than half of total 7.0).
        let mut pairs = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 5.0)];
        let median = weighted_median(&mut pairs);
        assert!((median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_median_single() {
        let mut pairs = vec![(42.0, 1.0)];
        let median = weighted_median(&mut pairs);
        assert!((median - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_median_empty() {
        let mut pairs: Vec<(f64, f64)> = vec![];
        let median = weighted_median(&mut pairs);
        assert!((median - 0.0).abs() < 1e-10);
    }
}
