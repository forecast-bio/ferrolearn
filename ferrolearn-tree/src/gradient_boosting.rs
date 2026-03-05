//! Gradient boosting classifiers and regressors.
//!
//! This module provides [`GradientBoostingClassifier`] and [`GradientBoostingRegressor`],
//! which build ensembles of decision trees sequentially. Each tree fits the negative
//! gradient (pseudo-residuals) of the loss function, progressively reducing prediction error.
//!
//! # Regression Losses
//!
//! - **`LeastSquares`** (L2): mean squared error; pseudo-residuals are `y - F(x)`.
//! - **`Lad`** (L1): least absolute deviation; pseudo-residuals are `sign(y - F(x))`.
//! - **`Huber`**: a blend of L2 (for small residuals) and L1 (for large residuals),
//!   controlled by the `alpha` quantile parameter (default 0.9).
//!
//! # Classification Loss
//!
//! - **`LogLoss`**: binary and multiclass logistic loss. For binary classification a
//!   single model is trained on log-odds; for *K*-class problems *K* trees are built
//!   per boosting round (one-vs-rest in probability space via softmax).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_tree::GradientBoostingRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let x = Array2::from_shape_vec((8, 1), vec![
//!     1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//! ]).unwrap();
//! let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
//!
//! let model = GradientBoostingRegressor::<f64>::new()
//!     .with_n_estimators(50)
//!     .with_learning_rate(0.1)
//!     .with_random_state(42);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 8);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasFeatureImportances};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample as rand_sample_indices;

use crate::decision_tree::{
    self, Node, build_regression_tree_with_feature_subset, compute_feature_importances,
};

// ---------------------------------------------------------------------------
// Regression loss enum
// ---------------------------------------------------------------------------

/// Loss function for gradient boosting regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionLoss {
    /// Least squares (L2) loss.
    LeastSquares,
    /// Least absolute deviation (L1) loss.
    Lad,
    /// Huber loss: L2 for small residuals, L1 for large residuals.
    Huber,
}

/// Loss function for gradient boosting classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationLoss {
    /// Log-loss (logistic / cross-entropy) for binary and multiclass.
    LogLoss,
}

// ---------------------------------------------------------------------------
// GradientBoostingRegressor
// ---------------------------------------------------------------------------

/// Gradient boosting regressor.
///
/// Builds an additive model in a forward stage-wise fashion, fitting each
/// regression tree to the negative gradient of the loss function evaluated
/// on the current ensemble prediction.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor<F> {
    /// Number of boosting stages (trees).
    pub n_estimators: usize,
    /// Learning rate (shrinkage) applied to each tree's contribution.
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of samples to use for fitting each tree (stochastic boosting).
    pub subsample: f64,
    /// Loss function.
    pub loss: RegressionLoss,
    /// Alpha quantile for Huber loss (only used when `loss == Huber`).
    pub huber_alpha: f64,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> GradientBoostingRegressor<F> {
    /// Create a new `GradientBoostingRegressor` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `learning_rate = 0.1`,
    /// `max_depth = Some(3)`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `subsample = 1.0`,
    /// `loss = LeastSquares`, `huber_alpha = 0.9`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            loss: RegressionLoss::LeastSquares,
            huber_alpha: 0.9,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate (shrinkage).
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the minimum number of samples to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set the subsample ratio (fraction of training data per tree).
    #[must_use]
    pub fn with_subsample(mut self, ratio: f64) -> Self {
        self.subsample = ratio;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: RegressionLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the alpha quantile for Huber loss.
    #[must_use]
    pub fn with_huber_alpha(mut self, alpha: f64) -> Self {
        self.huber_alpha = alpha;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for GradientBoostingRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedGradientBoostingRegressor
// ---------------------------------------------------------------------------

/// A fitted gradient boosting regressor.
///
/// Stores the initial prediction (intercept) and the sequence of fitted trees.
/// Predictions are computed as `init + learning_rate * sum(tree_predictions)`.
#[derive(Debug, Clone)]
pub struct FittedGradientBoostingRegressor<F> {
    /// Initial prediction (mean of training targets for L2 loss, median for L1/Huber).
    init: F,
    /// Learning rate used during training.
    learning_rate: F,
    /// Sequence of fitted trees (one per boosting round).
    trees: Vec<Vec<Node<F>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for GradientBoostingRegressor<F> {
    type Fitted = FittedGradientBoostingRegressor<F>;
    type Error = FerroError;

    /// Fit the gradient boosting regressor.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedGradientBoostingRegressor<F>, FerroError> {
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
                context: "GradientBoostingRegressor requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "subsample".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let lr = F::from(self.learning_rate).unwrap();
        let params = decision_tree::TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        // Initial prediction.
        let init = match self.loss {
            RegressionLoss::LeastSquares => {
                let sum: F = y.iter().copied().fold(F::zero(), |a, b| a + b);
                sum / F::from(n_samples).unwrap()
            }
            RegressionLoss::Lad | RegressionLoss::Huber => median_f(y),
        };

        // Current predictions for each sample.
        let mut f_vals = Array1::from_elem(n_samples, init);

        let all_features: Vec<usize> = (0..n_features).collect();
        let subsample_size = ((self.subsample * n_samples as f64).ceil() as usize)
            .max(1)
            .min(n_samples);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use rand::RngCore;
                StdRng::seed_from_u64(rand::rng().next_u64())
            }
        };

        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute pseudo-residuals (negative gradient).
            let residuals = compute_regression_residuals(y, &f_vals, self.loss, self.huber_alpha);

            // Subsample indices.
            let sample_indices = if subsample_size < n_samples {
                rand_sample_indices(&mut rng, n_samples, subsample_size).into_vec()
            } else {
                (0..n_samples).collect()
            };

            // Build a regression tree on the pseudo-residuals.
            let tree = build_regression_tree_with_feature_subset(
                x,
                &residuals,
                &sample_indices,
                &all_features,
                &params,
            );

            // Update predictions.
            for i in 0..n_samples {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf { value, .. } = tree[leaf_idx] {
                    f_vals[i] = f_vals[i] + lr * value;
                }
            }

            trees.push(tree);
        }

        // Compute feature importances across all trees.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees {
            let tree_imp = compute_feature_importances(tree_nodes, n_features, n_samples);
            total_importances = total_importances + tree_imp;
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedGradientBoostingRegressor {
            init,
            learning_rate: lr,
            trees,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedGradientBoostingRegressor<F> {
    /// Returns the initial prediction (intercept) of the boosted model.
    #[must_use]
    pub fn init(&self) -> F {
        self.init
    }

    /// Returns the learning rate used during training.
    #[must_use]
    pub fn learning_rate(&self) -> F {
        self.learning_rate
    }

    /// Returns a reference to the sequence of fitted trees.
    #[must_use]
    pub fn trees(&self) -> &[Vec<Node<F>>] {
        &self.trees
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGradientBoostingRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values.
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
        let mut predictions = Array1::from_elem(n_samples, self.init);

        for i in 0..n_samples {
            let row = x.row(i);
            for tree_nodes in &self.trees {
                let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                    predictions[i] = predictions[i] + self.learning_rate * value;
                }
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedGradientBoostingRegressor<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for GradientBoostingRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedGradientBoostingRegressor<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// GradientBoostingClassifier
// ---------------------------------------------------------------------------

/// Gradient boosting classifier.
///
/// For binary classification a single model is trained on log-odds residuals.
/// For multiclass (*K* classes), *K* regression trees are built per boosting
/// round (one-vs-rest in probability space via softmax).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier<F> {
    /// Number of boosting stages.
    pub n_estimators: usize,
    /// Learning rate (shrinkage).
    pub learning_rate: f64,
    /// Maximum depth of each tree.
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf node.
    pub min_samples_leaf: usize,
    /// Fraction of samples to use for fitting each tree.
    pub subsample: f64,
    /// Classification loss function.
    pub loss: ClassificationLoss,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> GradientBoostingClassifier<F> {
    /// Create a new `GradientBoostingClassifier` with default settings.
    ///
    /// Defaults: `n_estimators = 100`, `learning_rate = 0.1`,
    /// `max_depth = Some(3)`, `min_samples_split = 2`,
    /// `min_samples_leaf = 1`, `subsample = 1.0`,
    /// `loss = LogLoss`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            loss: ClassificationLoss::LogLoss,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of boosting stages.
    #[must_use]
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the learning rate (shrinkage).
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn with_max_depth(mut self, d: Option<usize>) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the minimum number of samples to split a node.
    #[must_use]
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set the minimum number of samples in a leaf.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }

    /// Set the subsample ratio.
    #[must_use]
    pub fn with_subsample(mut self, ratio: f64) -> Self {
        self.subsample = ratio;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float> Default for GradientBoostingClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedGradientBoostingClassifier
// ---------------------------------------------------------------------------

/// A fitted gradient boosting classifier.
///
/// For binary classification, stores a single sequence of trees predicting log-odds.
/// For multiclass, stores `K` sequences of trees (one per class).
#[derive(Debug, Clone)]
pub struct FittedGradientBoostingClassifier<F> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Initial predictions per class (log-odds or log-prior).
    init: Vec<F>,
    /// Learning rate.
    learning_rate: F,
    /// Trees: for binary, `trees[0]` has all trees. For multiclass,
    /// `trees[k]` has trees for class k.
    trees: Vec<Vec<Vec<Node<F>>>>,
    /// Number of features.
    n_features: usize,
    /// Per-feature importance scores (normalised).
    feature_importances: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for GradientBoostingClassifier<F>
{
    type Fitted = FittedGradientBoostingClassifier<F>;
    type Error = FerroError;

    /// Fit the gradient boosting classifier.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    /// Returns [`FerroError::InvalidParameter`] for invalid hyperparameters.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedGradientBoostingClassifier<F>, FerroError> {
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
                context: "GradientBoostingClassifier requires at least one sample".into(),
            });
        }
        if self.n_estimators == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_estimators".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.learning_rate <= 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "learning_rate".into(),
                reason: "must be positive".into(),
            });
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "subsample".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "need at least 2 distinct classes".into(),
            });
        }

        let y_mapped: Vec<usize> = y
            .iter()
            .map(|&c| classes.iter().position(|&cl| cl == c).unwrap())
            .collect();

        let lr = F::from(self.learning_rate).unwrap();
        let params = decision_tree::TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
        };

        let all_features: Vec<usize> = (0..n_features).collect();
        let subsample_size = ((self.subsample * n_samples as f64).ceil() as usize)
            .max(1)
            .min(n_samples);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use rand::RngCore;
                StdRng::seed_from_u64(rand::rng().next_u64())
            }
        };

        if n_classes == 2 {
            // Binary classification: single model on log-odds.
            self.fit_binary(
                x,
                &y_mapped,
                n_samples,
                n_features,
                &classes,
                lr,
                &params,
                &all_features,
                subsample_size,
                &mut rng,
            )
        } else {
            // Multiclass: K trees per round.
            self.fit_multiclass(
                x,
                &y_mapped,
                n_samples,
                n_features,
                n_classes,
                &classes,
                lr,
                &params,
                &all_features,
                subsample_size,
                &mut rng,
            )
        }
    }
}

impl<F: Float + Send + Sync + 'static> GradientBoostingClassifier<F> {
    /// Fit binary classification (log-loss on log-odds).
    #[allow(clippy::too_many_arguments)]
    fn fit_binary(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        classes: &[usize],
        lr: F,
        params: &decision_tree::TreeParams,
        all_features: &[usize],
        subsample_size: usize,
        rng: &mut StdRng,
    ) -> Result<FittedGradientBoostingClassifier<F>, FerroError> {
        // Count positive class proportion for initial log-odds.
        let pos_count = y_mapped.iter().filter(|&&c| c == 1).count();
        let p = F::from(pos_count).unwrap() / F::from(n_samples).unwrap();
        let eps = F::from(1e-15).unwrap();
        let p_clipped = p.max(eps).min(F::one() - eps);
        let init_val = (p_clipped / (F::one() - p_clipped)).ln();

        let mut f_vals = Array1::from_elem(n_samples, init_val);
        let mut trees_seq: Vec<Vec<Node<F>>> = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Compute probabilities from current log-odds.
            let probs: Vec<F> = f_vals.iter().map(|&fv| sigmoid(fv)).collect();

            // Pseudo-residuals: y - p.
            let mut residuals = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let yi = F::from(y_mapped[i]).unwrap();
                residuals[i] = yi - probs[i];
            }

            // Subsample.
            let sample_indices = if subsample_size < n_samples {
                rand_sample_indices(rng, n_samples, subsample_size).into_vec()
            } else {
                (0..n_samples).collect()
            };

            // Build tree on residuals.
            let tree = build_regression_tree_with_feature_subset(
                x,
                &residuals,
                &sample_indices,
                all_features,
                params,
            );

            // Update f_vals.
            for i in 0..n_samples {
                let row = x.row(i);
                let leaf_idx = decision_tree::traverse(&tree, &row);
                if let Node::Leaf { value, .. } = tree[leaf_idx] {
                    f_vals[i] = f_vals[i] + lr * value;
                }
            }

            trees_seq.push(tree);
        }

        // Feature importances.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for tree_nodes in &trees_seq {
            let tree_imp = compute_feature_importances(tree_nodes, n_features, n_samples);
            total_importances = total_importances + tree_imp;
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedGradientBoostingClassifier {
            classes: classes.to_vec(),
            init: vec![init_val],
            learning_rate: lr,
            trees: vec![trees_seq],
            n_features,
            feature_importances: total_importances,
        })
    }

    /// Fit multiclass classification (K trees per round, softmax).
    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass(
        &self,
        x: &Array2<F>,
        y_mapped: &[usize],
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        classes: &[usize],
        lr: F,
        params: &decision_tree::TreeParams,
        all_features: &[usize],
        subsample_size: usize,
        rng: &mut StdRng,
    ) -> Result<FittedGradientBoostingClassifier<F>, FerroError> {
        // Initial log-prior for each class.
        let mut class_counts = vec![0usize; n_classes];
        for &c in y_mapped {
            class_counts[c] += 1;
        }
        let n_f = F::from(n_samples).unwrap();
        let eps = F::from(1e-15).unwrap();
        let init_vals: Vec<F> = class_counts
            .iter()
            .map(|&cnt| {
                let p = (F::from(cnt).unwrap() / n_f).max(eps);
                p.ln()
            })
            .collect();

        // f_vals[k][i] = current raw score for class k, sample i.
        let mut f_vals: Vec<Array1<F>> = init_vals
            .iter()
            .map(|&init| Array1::from_elem(n_samples, init))
            .collect();

        let mut trees_per_class: Vec<Vec<Vec<Node<F>>>> = (0..n_classes)
            .map(|_| Vec::with_capacity(self.n_estimators))
            .collect();

        for _ in 0..self.n_estimators {
            // Compute softmax probabilities.
            let probs = softmax_matrix(&f_vals, n_samples, n_classes);

            // Subsample.
            let sample_indices = if subsample_size < n_samples {
                rand_sample_indices(rng, n_samples, subsample_size).into_vec()
            } else {
                (0..n_samples).collect()
            };

            // For each class, compute residuals and fit a tree.
            for k in 0..n_classes {
                let mut residuals = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let yi_k = if y_mapped[i] == k {
                        F::one()
                    } else {
                        F::zero()
                    };
                    residuals[i] = yi_k - probs[k][i];
                }

                let tree = build_regression_tree_with_feature_subset(
                    x,
                    &residuals,
                    &sample_indices,
                    all_features,
                    params,
                );

                // Update f_vals for class k.
                for (i, fv) in f_vals[k].iter_mut().enumerate() {
                    let row = x.row(i);
                    let leaf_idx = decision_tree::traverse(&tree, &row);
                    if let Node::Leaf { value, .. } = tree[leaf_idx] {
                        *fv = *fv + lr * value;
                    }
                }

                trees_per_class[k].push(tree);
            }
        }

        // Feature importances aggregated across all classes and rounds.
        let mut total_importances = Array1::<F>::zeros(n_features);
        for class_trees in &trees_per_class {
            for tree_nodes in class_trees {
                let tree_imp = compute_feature_importances(tree_nodes, n_features, n_samples);
                total_importances = total_importances + tree_imp;
            }
        }
        let imp_sum: F = total_importances
            .iter()
            .copied()
            .fold(F::zero(), |a, b| a + b);
        if imp_sum > F::zero() {
            total_importances.mapv_inplace(|v| v / imp_sum);
        }

        Ok(FittedGradientBoostingClassifier {
            classes: classes.to_vec(),
            init: init_vals,
            learning_rate: lr,
            trees: trees_per_class,
            n_features,
            feature_importances: total_importances,
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedGradientBoostingClassifier<F> {
    /// Returns the initial predictions per class (log-odds or log-prior).
    #[must_use]
    pub fn init(&self) -> &[F] {
        &self.init
    }

    /// Returns the learning rate used during training.
    #[must_use]
    pub fn learning_rate(&self) -> F {
        self.learning_rate
    }

    /// Returns a reference to the tree ensemble.
    ///
    /// For binary classification, `trees()[0]` contains all trees.
    /// For multiclass, `trees()[k]` contains trees for class `k`.
    #[must_use]
    pub fn trees(&self) -> &[Vec<Vec<Node<F>>>] {
        &self.trees
    }

    /// Returns the number of features the model was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGradientBoostingClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels.
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
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        if n_classes == 2 {
            // Binary: single log-odds model.
            let init = self.init[0];
            let mut predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let row = x.row(i);
                let mut f_val = init;
                for tree_nodes in &self.trees[0] {
                    let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                    if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                        f_val = f_val + self.learning_rate * value;
                    }
                }
                let prob = sigmoid(f_val);
                let class_idx = if prob >= F::from(0.5).unwrap() { 1 } else { 0 };
                predictions[i] = self.classes[class_idx];
            }
            Ok(predictions)
        } else {
            // Multiclass: K models, argmax of softmax.
            let mut predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let row = x.row(i);
                let mut scores = Vec::with_capacity(n_classes);
                for k in 0..n_classes {
                    let mut f_val = self.init[k];
                    for tree_nodes in &self.trees[k] {
                        let leaf_idx = decision_tree::traverse(tree_nodes, &row);
                        if let Node::Leaf { value, .. } = tree_nodes[leaf_idx] {
                            f_val = f_val + self.learning_rate * value;
                        }
                    }
                    scores.push(f_val);
                }
                let best_k = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(k, _)| k)
                    .unwrap_or(0);
                predictions[i] = self.classes[best_k];
            }
            Ok(predictions)
        }
    }
}

impl<F: Float + Send + Sync + 'static> HasFeatureImportances<F>
    for FittedGradientBoostingClassifier<F>
{
    fn feature_importances(&self) -> &Array1<F> {
        &self.feature_importances
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedGradientBoostingClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for GradientBoostingClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedGbcPipelineAdapter(fitted)))
    }
}

/// Pipeline adapter for `FittedGradientBoostingClassifier<F>`.
struct FittedGbcPipelineAdapter<F: Float + Send + Sync + 'static>(
    FittedGradientBoostingClassifier<F>,
);

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedGbcPipelineAdapter<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

/// Compute softmax probabilities for each class across all samples.
///
/// Returns `probs[k][i]` = probability of class k for sample i.
fn softmax_matrix<F: Float>(
    f_vals: &[Array1<F>],
    n_samples: usize,
    n_classes: usize,
) -> Vec<Vec<F>> {
    let mut probs: Vec<Vec<F>> = vec![vec![F::zero(); n_samples]; n_classes];

    for i in 0..n_samples {
        // Find max for numerical stability.
        let max_val = (0..n_classes)
            .map(|k| f_vals[k][i])
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });

        let mut sum = F::zero();
        let mut exps = vec![F::zero(); n_classes];
        for k in 0..n_classes {
            exps[k] = (f_vals[k][i] - max_val).exp();
            sum = sum + exps[k];
        }

        let eps = F::from(1e-15).unwrap();
        if sum < eps {
            sum = eps;
        }

        for k in 0..n_classes {
            probs[k][i] = exps[k] / sum;
        }
    }

    probs
}

/// Compute the median of an Array1.
fn median_f<F: Float>(arr: &Array1<F>) -> F {
    let mut sorted: Vec<F> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / F::from(2.0).unwrap()
    }
}

/// Compute the quantile of a slice at level `alpha` (0..1).
fn quantile_f<F: Float>(vals: &[F], alpha: f64) -> F {
    if vals.is_empty() {
        return F::zero();
    }
    let mut sorted: Vec<F> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * alpha).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

/// Compute pseudo-residuals (negative gradient) for regression losses.
fn compute_regression_residuals<F: Float>(
    y: &Array1<F>,
    f_vals: &Array1<F>,
    loss: RegressionLoss,
    huber_alpha: f64,
) -> Array1<F> {
    let n = y.len();
    match loss {
        RegressionLoss::LeastSquares => {
            // negative gradient of 0.5*(y - f)^2 is (y - f)
            let mut residuals = Array1::zeros(n);
            for i in 0..n {
                residuals[i] = y[i] - f_vals[i];
            }
            residuals
        }
        RegressionLoss::Lad => {
            // negative gradient of |y - f| is sign(y - f)
            let mut residuals = Array1::zeros(n);
            for i in 0..n {
                let diff = y[i] - f_vals[i];
                residuals[i] = if diff > F::zero() {
                    F::one()
                } else if diff < F::zero() {
                    -F::one()
                } else {
                    F::zero()
                };
            }
            residuals
        }
        RegressionLoss::Huber => {
            // Compute residuals and delta from quantile.
            let raw_residuals: Vec<F> = (0..n).map(|i| (y[i] - f_vals[i]).abs()).collect();
            let delta = quantile_f(&raw_residuals, huber_alpha);

            let mut residuals = Array1::zeros(n);
            for i in 0..n {
                let diff = y[i] - f_vals[i];
                if diff.abs() <= delta {
                    residuals[i] = diff;
                } else if diff > F::zero() {
                    residuals[i] = delta;
                } else {
                    residuals[i] = -delta;
                }
            }
            residuals
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -- Regressor tests --

    #[test]
    fn test_gbr_simple_least_squares() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert!(preds[i] < 3.0, "Expected ~1.0, got {}", preds[i]);
        }
        for i in 4..8 {
            assert!(preds[i] > 3.0, "Expected ~5.0, got {}", preds[i]);
        }
    }

    #[test]
    fn test_gbr_lad_loss() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_loss(RegressionLoss::Lad)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        // LAD should still separate the two groups.
        for i in 0..4 {
            assert!(preds[i] < 3.5, "LAD expected <3.5, got {}", preds[i]);
        }
        for i in 4..8 {
            assert!(preds[i] > 2.5, "LAD expected >2.5, got {}", preds[i]);
        }
    }

    #[test]
    fn test_gbr_huber_loss() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_loss(RegressionLoss::Huber)
            .with_huber_alpha(0.9)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_gbr_reproducibility() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(123);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();

        for (p1, p2) in preds1.iter().zip(preds2.iter()) {
            assert_relative_eq!(*p1, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gbr_feature_importances() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 3);
        // First feature should be most important.
        assert!(importances[0] > importances[1]);
        assert!(importances[0] > importances[2]);
    }

    #[test]
    fn test_gbr_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = GradientBoostingRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbr_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_gbr_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);

        let model = GradientBoostingRegressor::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbr_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = GradientBoostingRegressor::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbr_invalid_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbr_invalid_subsample() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_subsample(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model2 = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(5)
            .with_subsample(1.5);
        assert!(model2.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbr_subsample() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(50)
            .with_subsample(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_gbr_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_gbr_f32_support() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);

        let model = GradientBoostingRegressor::<f32>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_gbr_max_depth() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(20)
            .with_max_depth(Some(1))
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_gbr_default_trait() {
        let model = GradientBoostingRegressor::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert!((model.learning_rate - 0.1).abs() < 1e-10);
    }

    // -- Classifier tests --

    #[test]
    fn test_gbc_binary_simple() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 8);
        for i in 0..4 {
            assert_eq!(preds[i], 0, "Expected 0 at index {}, got {}", i, preds[i]);
        }
        for i in 4..8 {
            assert_eq!(preds[i], 1, "Expected 1 at index {}, got {}", i, preds[i]);
        }
    }

    #[test]
    fn test_gbc_multiclass() {
        let x = Array2::from_shape_vec((9, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 9);
        // At least training data should mostly be correct.
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/9 correct, got {}/9",
            correct
        );
    }

    #[test]
    fn test_gbc_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_gbc_reproducibility() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);

        let fitted1 = model.fit(&x, &y).unwrap();
        let fitted2 = model.fit(&x, &y).unwrap();

        let preds1 = fitted1.predict(&x).unwrap();
        let preds2 = fitted2.predict(&x).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn test_gbc_feature_importances() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0,
                0.0, 0.0, 7.0, 0.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let importances = fitted.feature_importances();

        assert_eq!(importances.len(), 3);
        assert!(importances[0] > importances[1]);
        assert!(importances[0] > importances[2]);
    }

    #[test]
    fn test_gbc_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1];

        let model = GradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbc_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_random_state(0);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_gbc_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);

        let model = GradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbc_single_class() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = GradientBoostingClassifier::<f64>::new().with_n_estimators(5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbc_zero_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new().with_n_estimators(0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_gbc_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_gbc_f32_support() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = GradientBoostingClassifier::<f32>::new()
            .with_n_estimators(10)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_gbc_subsample() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_subsample(0.5)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_gbc_default_trait() {
        let model = GradientBoostingClassifier::<f64>::default();
        assert_eq!(model.n_estimators, 100);
        assert!((model.learning_rate - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_gbc_non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![10, 10, 10, 20, 20, 20];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(20)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 6);
        for &p in preds.iter() {
            assert!(p == 10 || p == 20);
        }
    }

    // -- Helper tests --

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0f64), 0.5, epsilon = 1e-10);
        assert!(sigmoid(10.0f64) > 0.999);
        assert!(sigmoid(-10.0f64) < 0.001);
    }

    #[test]
    fn test_median_f_odd() {
        let arr = array![3.0, 1.0, 2.0];
        assert_relative_eq!(median_f(&arr), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_f_even() {
        let arr = array![4.0, 1.0, 3.0, 2.0];
        assert_relative_eq!(median_f(&arr), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_median_f_empty() {
        let arr = Array1::<f64>::zeros(0);
        assert_relative_eq!(median_f(&arr), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_f() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q90 = quantile_f(&vals, 0.9);
        assert!(q90 >= 4.0 && q90 <= 5.0);
    }

    #[test]
    fn test_regression_residuals_least_squares() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![0.5, 2.5, 2.0];
        let r = compute_regression_residuals(&y, &f, RegressionLoss::LeastSquares, 0.9);
        assert_relative_eq!(r[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(r[1], -0.5, epsilon = 1e-10);
        assert_relative_eq!(r[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regression_residuals_lad() {
        let y = array![1.0, 2.0, 3.0];
        let f = array![0.5, 2.5, 3.0];
        let r = compute_regression_residuals(&y, &f, RegressionLoss::Lad, 0.9);
        assert_relative_eq!(r[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(r[1], -1.0, epsilon = 1e-10);
        assert_relative_eq!(r[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regression_residuals_huber() {
        let y = array![1.0, 2.0, 10.0, 3.0, 4.0];
        let f = array![1.5, 2.5, 2.0, 3.5, 4.5];
        // abs residuals: [0.5, 0.5, 8.0, 0.5, 0.5]
        // alpha=0.9 quantile index = round(4 * 0.9) = 4 => sorted[4] = 8.0
        // So delta = 8.0, meaning all residuals are within delta and treated as L2.
        let r = compute_regression_residuals(&y, &f, RegressionLoss::Huber, 0.9);
        // All residuals should be y - f.
        assert_relative_eq!(r[0], -0.5, epsilon = 1e-10);
        assert_relative_eq!(r[1], -0.5, epsilon = 1e-10);
        assert_relative_eq!(r[2], 8.0, epsilon = 1e-10);
        assert_relative_eq!(r[3], -0.5, epsilon = 1e-10);
        assert_relative_eq!(r[4], -0.5, epsilon = 1e-10);

        // Test with lower alpha to trigger clipping.
        // alpha=0.1, quantile idx = round(4*0.1) = 0 => sorted[0] = 0.5
        // delta = 0.5, so the 8.0 residual is clipped.
        let r2 = compute_regression_residuals(&y, &f, RegressionLoss::Huber, 0.1);
        assert_relative_eq!(r2[0], -0.5, epsilon = 1e-10);
        // Third residual: diff=8.0 > delta=0.5, so clipped to delta=0.5.
        assert_relative_eq!(r2[2], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_gbc_multiclass_4_classes() {
        let x = Array2::from_shape_vec(
            (12, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(50)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        assert_eq!(preds.len(), 12);
        assert_eq!(fitted.n_classes(), 4);
    }

    #[test]
    fn test_gbc_invalid_learning_rate() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(5)
            .with_learning_rate(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }
}
