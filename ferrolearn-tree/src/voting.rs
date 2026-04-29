//! Voting ensemble classifiers and regressors.
//!
//! This module provides [`VotingClassifier`] and [`VotingRegressor`], which
//! train multiple decision trees with different hyperparameter configurations
//! on the full dataset and aggregate their predictions (majority vote for
//! classification, averaging for regression).
//!
//! Unlike [`RandomForestClassifier`](crate::RandomForestClassifier), voting
//! ensembles do **not** use bootstrap sampling — each tree sees the entire
//! dataset. Diversity comes from varying the tree hyperparameters.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::VotingClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,  4.0, 4.0,
//!     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,  8.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let model = VotingClassifier::<f64>::new()
//!     .with_max_depths(vec![Some(2), Some(3), Some(5), None]);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};

use crate::decision_tree::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor,
    FittedDecisionTreeClassifier, FittedDecisionTreeRegressor,
};

// ---------------------------------------------------------------------------
// VotingClassifier
// ---------------------------------------------------------------------------

/// Voting ensemble classifier.
///
/// Trains multiple decision tree classifiers with different hyperparameter
/// configurations on the full dataset. Final predictions are made by majority
/// vote across all trees.
///
/// Diversity is introduced by varying `max_depth` across the ensemble members.
/// If no explicit depths are provided, a default set of depths is used.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingClassifier<F> {
    /// Maximum depth settings for each tree in the ensemble.
    /// Each entry produces one decision tree.
    pub max_depths: Vec<Option<usize>>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Splitting criterion for all trees.
    pub criterion: ClassificationCriterion,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> VotingClassifier<F> {
    /// Create a new `VotingClassifier` with default settings.
    ///
    /// Defaults: `max_depths = [Some(2), Some(4), Some(6), None]`,
    /// `min_samples_split = 2`, `min_samples_leaf = 1`, `criterion = Gini`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depths: vec![Some(2), Some(4), Some(6), None],
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: ClassificationCriterion::Gini,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum depth settings for each ensemble member.
    ///
    /// Each entry in the vector produces one decision tree.
    #[must_use]
    pub fn with_max_depths(mut self, max_depths: Vec<Option<usize>>) -> Self {
        self.max_depths = max_depths;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the splitting criterion for all trees.
    #[must_use]
    pub fn with_criterion(mut self, criterion: ClassificationCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}

impl<F: Float> Default for VotingClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedVotingClassifier
// ---------------------------------------------------------------------------

/// A fitted voting ensemble classifier.
///
/// Stores the individually fitted decision trees and aggregates their
/// predictions by majority vote.
#[derive(Debug, Clone)]
pub struct FittedVotingClassifier<F> {
    /// The fitted decision tree classifiers.
    trees: Vec<FittedDecisionTreeClassifier<F>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedVotingClassifier<F> {
    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for VotingClassifier<F> {
    type Fitted = FittedVotingClassifier<F>;
    type Error = FerroError;

    /// Fit the voting classifier by training each decision tree on the full dataset.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if configuration is invalid.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedVotingClassifier<F>, FerroError> {
        let n_samples = x.nrows();

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
                context: "VotingClassifier requires at least one sample".into(),
            });
        }
        if self.max_depths.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "max_depths".into(),
                reason: "must contain at least one entry".into(),
            });
        }

        // Determine unique classes from the full dataset.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        let mut trees = Vec::with_capacity(self.max_depths.len());
        for &max_depth in &self.max_depths {
            let tree = DecisionTreeClassifier::<F>::new()
                .with_max_depth(max_depth)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf)
                .with_criterion(self.criterion);
            let fitted = tree.fit(x, y)?;
            trees.push(fitted);
        }

        Ok(FittedVotingClassifier { trees, classes })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedVotingClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels by majority vote across all trees.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        // Collect predictions from all trees.
        let all_preds: Vec<Array1<usize>> = self
            .trees
            .iter()
            .map(|tree| tree.predict(x))
            .collect::<Result<Vec<_>, _>>()?;

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut votes = vec![0usize; n_classes];
            for tree_preds in &all_preds {
                let pred = tree_preds[i];
                if let Some(class_idx) = self.classes.iter().position(|&c| c == pred) {
                    votes[class_idx] += 1;
                }
            }
            let winner = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &count)| count)
                .map_or(0, |(idx, _)| idx);
            predictions[i] = self.classes[winner];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedVotingClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for VotingClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedVotingClassifierPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedVotingClassifier<F>`.
struct FittedVotingClassifierPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedVotingClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedVotingClassifierPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// VotingRegressor
// ---------------------------------------------------------------------------

/// Voting ensemble regressor.
///
/// Trains multiple decision tree regressors with different hyperparameter
/// configurations on the full dataset. Final predictions are the average
/// across all trees.
///
/// Diversity is introduced by varying `max_depth` across the ensemble members.
/// If no explicit depths are provided, a default set of depths is used.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_tree::VotingRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array1, Array2};
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,  2.0, 3.0,  3.0, 3.0,
///     5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
/// ]).unwrap();
/// let y = array![1.0, 2.0, 3.0, 5.0, 6.0, 7.0];
///
/// let model = VotingRegressor::<f64>::new()
///     .with_max_depths(vec![Some(2), Some(4), None]);
/// let fitted = model.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingRegressor<F> {
    /// Maximum depth settings for each tree in the ensemble.
    pub max_depths: Vec<Option<usize>>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> VotingRegressor<F> {
    /// Create a new `VotingRegressor` with default settings.
    ///
    /// Defaults: `max_depths = [Some(2), Some(4), Some(6), None]`,
    /// `min_samples_split = 2`, `min_samples_leaf = 1`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_depths: vec![Some(2), Some(4), Some(6), None],
            min_samples_split: 2,
            min_samples_leaf: 1,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum depth settings for each ensemble member.
    #[must_use]
    pub fn with_max_depths(mut self, max_depths: Vec<Option<usize>>) -> Self {
        self.max_depths = max_depths;
        self
    }

    /// Set the minimum number of samples required to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in a leaf node.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
}

impl<F: Float> Default for VotingRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedVotingRegressor
// ---------------------------------------------------------------------------

/// A fitted voting ensemble regressor.
///
/// Stores the individually fitted decision tree regressors and aggregates
/// their predictions by averaging.
#[derive(Debug, Clone)]
pub struct FittedVotingRegressor<F> {
    /// The fitted decision tree regressors.
    trees: Vec<FittedDecisionTreeRegressor<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedVotingRegressor<F> {
    /// Returns the number of trees in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for VotingRegressor<F> {
    type Fitted = FittedVotingRegressor<F>;
    type Error = FerroError;

    /// Fit the voting regressor by training each decision tree on the full dataset.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] if configuration is invalid.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedVotingRegressor<F>, FerroError> {
        let n_samples = x.nrows();

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
                context: "VotingRegressor requires at least one sample".into(),
            });
        }
        if self.max_depths.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "max_depths".into(),
                reason: "must contain at least one entry".into(),
            });
        }

        let mut trees = Vec::with_capacity(self.max_depths.len());
        for &max_depth in &self.max_depths {
            let tree = DecisionTreeRegressor::<F>::new()
                .with_max_depth(max_depth)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf);
            let fitted = tree.fit(x, y)?;
            trees.push(fitted);
        }

        Ok(FittedVotingRegressor { trees })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedVotingRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values by averaging across all trees.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_samples = x.nrows();
        let n_trees_f = F::from(self.trees.len()).unwrap();

        let all_preds: Vec<Array1<F>> = self
            .trees
            .iter()
            .map(|tree| tree.predict(x))
            .collect::<Result<Vec<_>, _>>()?;

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut sum = F::zero();
            for tree_preds in &all_preds {
                sum = sum + tree_preds[i];
            }
            predictions[i] = sum / n_trees_f;
        }

        Ok(predictions)
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for VotingRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedVotingRegressor<F> {
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_classification_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 5.0, 6.0, 7.0];
        (x, y)
    }

    // -- VotingClassifier tests --

    #[test]
    fn test_voting_classifier_default() {
        let model = VotingClassifier::<f64>::new();
        assert_eq!(model.max_depths.len(), 4);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
    }

    #[test]
    fn test_voting_classifier_builder() {
        let model = VotingClassifier::<f64>::new()
            .with_max_depths(vec![Some(1), Some(3)])
            .with_min_samples_split(5)
            .with_min_samples_leaf(2)
            .with_criterion(ClassificationCriterion::Entropy);
        assert_eq!(model.max_depths.len(), 2);
        assert_eq!(model.min_samples_split, 5);
        assert_eq!(model.min_samples_leaf, 2);
        assert_eq!(model.criterion, ClassificationCriterion::Entropy);
    }

    #[test]
    fn test_voting_classifier_fit_predict() {
        let (x, y) = make_classification_data();
        let model = VotingClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        // On training data with a clear separation, should get most right.
        for i in 0..4 {
            assert_eq!(preds[i], 0, "sample {i} should be class 0");
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1, "sample {i} should be class 1");
        }
    }

    #[test]
    fn test_voting_classifier_has_classes() {
        let (x, y) = make_classification_data();
        let model = VotingClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_voting_classifier_n_estimators() {
        let (x, y) = make_classification_data();
        let model = VotingClassifier::<f64>::new().with_max_depths(vec![Some(2), Some(4), None]);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 3);
    }

    #[test]
    fn test_voting_classifier_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let model = VotingClassifier::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_classifier_shape_mismatch_error() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array1::<usize>::zeros(3);
        let model = VotingClassifier::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_classifier_empty_depths_error() {
        let (x, y) = make_classification_data();
        let model = VotingClassifier::<f64>::new().with_max_depths(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 8.0, 8.0, 9.0, 8.0,
                8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = VotingClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 9);
        assert_eq!(fitted.n_classes(), 3);
    }

    // -- VotingRegressor tests --

    #[test]
    fn test_voting_regressor_default() {
        let model = VotingRegressor::<f64>::new();
        assert_eq!(model.max_depths.len(), 4);
        assert_eq!(model.min_samples_split, 2);
        assert_eq!(model.min_samples_leaf, 1);
    }

    #[test]
    fn test_voting_regressor_builder() {
        let model = VotingRegressor::<f64>::new()
            .with_max_depths(vec![Some(1), Some(5)])
            .with_min_samples_split(3)
            .with_min_samples_leaf(2);
        assert_eq!(model.max_depths.len(), 2);
        assert_eq!(model.min_samples_split, 3);
        assert_eq!(model.min_samples_leaf, 2);
    }

    #[test]
    fn test_voting_regressor_fit_predict() {
        let (x, y) = make_regression_data();
        let model = VotingRegressor::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        // The average of multiple trees should approximate the training targets
        // on the training data.
        for i in 0..6 {
            let err = (preds[i] - y[i]).abs();
            assert!(
                err < 3.0,
                "prediction {:.2} should be close to target {:.2}",
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_voting_regressor_n_estimators() {
        let (x, y) = make_regression_data();
        let model = VotingRegressor::<f64>::new().with_max_depths(vec![Some(2), None]);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_estimators(), 2);
    }

    #[test]
    fn test_voting_regressor_empty_data_error() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let model = VotingRegressor::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_regressor_shape_mismatch_error() {
        let x = Array2::<f64>::zeros((5, 2));
        let y = Array1::<f64>::zeros(3);
        let model = VotingRegressor::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_regressor_empty_depths_error() {
        let (x, y) = make_regression_data();
        let model = VotingRegressor::<f64>::new().with_max_depths(vec![]);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_voting_regressor_averaging() {
        // With a single tree (unlimited depth), predictions on training data
        // should exactly match the targets.
        let (x, y) = make_regression_data();
        let model = VotingRegressor::<f64>::new().with_max_depths(vec![None]);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..6 {
            assert!(
                (preds[i] - y[i]).abs() < 1e-10,
                "single unlimited tree should overfit training data"
            );
        }
    }

    #[test]
    fn test_voting_classifier_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];
        let model = VotingClassifier::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_voting_regressor_f32() {
        let x = Array2::<f32>::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![1.0_f32, 2.0, 3.0, 5.0, 6.0, 7.0];
        let model = VotingRegressor::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
