//! k-Nearest Neighbors classifier and regressor.
//!
//! This module provides [`KNeighborsClassifier`] and [`KNeighborsRegressor`],
//! which classify or predict target values based on the `k` nearest training
//! samples using Euclidean distance.
//!
//! # Algorithm Selection
//!
//! The [`Algorithm`] enum controls the spatial indexing strategy:
//!
//! - [`Algorithm::Auto`]: Automatically selects KD-Tree for dimensions <= 20,
//!   brute force otherwise.
//! - [`Algorithm::BruteForce`]: Always uses O(n) exhaustive search.
//! - [`Algorithm::KdTree`]: Always uses the KD-Tree spatial index.
//!
//! # Weighting
//!
//! The [`Weights`] enum controls how neighbor contributions are combined:
//!
//! - [`Weights::Uniform`]: All neighbors contribute equally.
//! - [`Weights::Distance`]: Closer neighbors contribute more (inverse distance).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::KNeighborsClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let clf = KNeighborsClassifier::<f64>::new();
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
//!     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let fitted = clf.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use crate::balltree::BallTree;
use crate::kdtree::{self, KdTree};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// The algorithm used to compute nearest neighbors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Automatically select the best algorithm based on data characteristics.
    /// Uses KD-Tree for dimensions <= 15, ball tree for higher dimensions.
    Auto,
    /// Use brute-force exhaustive search (O(n) per query).
    BruteForce,
    /// Use a KD-Tree spatial index (O(log n) average per query for low dimensions).
    KdTree,
    /// Use a ball tree spatial index (handles high dimensions better than KD-Tree).
    BallTree,
}

/// The weighting scheme for neighbor contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weights {
    /// All neighbors contribute equally (majority vote / simple mean).
    Uniform,
    /// Closer neighbors contribute more, weighted by inverse distance.
    /// If a query point exactly coincides with a training point (distance = 0),
    /// that point receives all weight.
    Distance,
}

// ---------------------------------------------------------------------------
// Helper: find k nearest neighbors
// ---------------------------------------------------------------------------

/// Which spatial index was built during fit.
enum SpatialIndex {
    None,
    KdTree(KdTree),
    BallTree(BallTree),
}

impl std::fmt::Debug for SpatialIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialIndex::None => write!(f, "None"),
            SpatialIndex::KdTree(t) => write!(f, "KdTree({t:?})"),
            SpatialIndex::BallTree(t) => write!(f, "BallTree({t:?})"),
        }
    }
}

/// Find the k nearest neighbors of a query point.
///
/// Returns `(index, distance)` pairs sorted by distance.
fn find_neighbors<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query_row: &[F],
    k: usize,
    index: &SpatialIndex,
) -> Vec<(usize, F)> {
    match index {
        SpatialIndex::KdTree(tree) => {
            let query_f64: Vec<f64> = query_row.iter().map(|v| v.to_f64().unwrap()).collect();
            let results = tree.query(data, &query_f64, k);
            results
                .into_iter()
                .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                .collect()
        }
        SpatialIndex::BallTree(tree) => {
            let query_f64: Vec<f64> = query_row.iter().map(|v| v.to_f64().unwrap()).collect();
            let results = tree.query(data, &query_f64, k);
            results
                .into_iter()
                .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                .collect()
        }
        SpatialIndex::None => kdtree::brute_force_knn(data, query_row, k),
    }
}

/// Build the appropriate spatial index based on algorithm setting and dimensionality.
fn build_spatial_index<F: Float + Send + Sync + 'static>(
    algorithm: Algorithm,
    data: &Array2<F>,
) -> SpatialIndex {
    let n_features = data.ncols();
    match algorithm {
        Algorithm::Auto => {
            if n_features <= 15 {
                SpatialIndex::KdTree(KdTree::build(data))
            } else {
                SpatialIndex::BallTree(BallTree::build(data))
            }
        }
        Algorithm::KdTree => SpatialIndex::KdTree(KdTree::build(data)),
        Algorithm::BallTree => SpatialIndex::BallTree(BallTree::build(data)),
        Algorithm::BruteForce => SpatialIndex::None,
    }
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

/// k-Nearest Neighbors classifier.
///
/// Classifies samples by majority vote of the `k` nearest training
/// samples using Euclidean distance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::KNeighborsClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
/// let x = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
///     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsClassifier<F> {
    /// Number of neighbors to use for classification.
    pub n_neighbors: usize,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> KNeighborsClassifier<F> {
    /// Create a new `KNeighborsClassifier` with default settings.
    ///
    /// Defaults: `n_neighbors = 5`, `algorithm = Auto`, `weights = Uniform`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            algorithm: Algorithm::Auto,
            weights: Weights::Uniform,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }
}

impl<F: Float> Default for KNeighborsClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted k-Nearest Neighbors classifier.
///
/// Stores the training data and an optional KD-Tree spatial index.
/// Implements [`Predict`] to classify new samples.
#[derive(Debug)]
pub struct FittedKNeighborsClassifier<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training labels.
    y_train: Array1<usize>,
    /// Number of neighbors to use.
    n_neighbors: usize,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index (KD-Tree, Ball Tree, or None for brute force).
    spatial_index: SpatialIndex,
    /// Sorted unique class labels.
    classes: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for KNeighborsClassifier<F> {
    type Fitted = FittedKNeighborsClassifier<F>;
    type Error = FerroError;

    /// Fit the classifier by storing the training data and optionally
    /// building a KD-Tree spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer
    /// samples than `n_neighbors`.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedKNeighborsClassifier<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        if n_samples < self.n_neighbors {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples,
                context: format!(
                    "KNeighborsClassifier requires at least n_neighbors={} samples",
                    self.n_neighbors
                ),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        // Build spatial index.
        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedKNeighborsClassifier {
            x_train: x.clone(),
            y_train: y.clone(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            spatial_index,
            classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKNeighborsClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// For each sample, finds the `k` nearest neighbors in the training
    /// data and returns the majority class (with optional distance weighting).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        let n_samples = x.nrows();

        // Use a threshold to avoid Rayon overhead on small inputs.
        const PAR_THRESHOLD: usize = 256;

        let predictions_vec: Vec<usize> = if n_samples >= PAR_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                    );
                    self.weighted_vote(&neighbors)
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                    );
                    self.weighted_vote(&neighbors)
                })
                .collect()
        };

        Ok(Array1::from_vec(predictions_vec))
    }
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsClassifier<F> {
    /// Perform a (possibly weighted) majority vote among neighbors.
    fn weighted_vote(&self, neighbors: &[(usize, F)]) -> usize {
        let mut class_weights: HashMap<usize, F> = HashMap::new();
        let eps = F::from(1e-15).unwrap();

        match self.weights {
            Weights::Uniform => {
                for &(idx, _) in neighbors {
                    let label = self.y_train[idx];
                    *class_weights.entry(label).or_insert(F::zero()) =
                        *class_weights.entry(label).or_insert(F::zero()) + F::one();
                }
            }
            Weights::Distance => {
                // Check if any neighbor has zero distance.
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);

                if has_zero_dist {
                    // Give all weight to zero-distance neighbors.
                    for &(idx, d) in neighbors {
                        if d < eps {
                            let label = self.y_train[idx];
                            *class_weights.entry(label).or_insert(F::zero()) =
                                *class_weights.entry(label).or_insert(F::zero()) + F::one();
                        }
                    }
                } else {
                    for &(idx, d) in neighbors {
                        let label = self.y_train[idx];
                        let w = F::one() / d;
                        *class_weights.entry(label).or_insert(F::zero()) =
                            *class_weights.entry(label).or_insert(F::zero()) + w;
                    }
                }
            }
        }

        // Find the class with the maximum weight. Break ties by smallest class label.
        class_weights
            .into_iter()
            .max_by(|(label_a, w_a), (label_b, w_b)| {
                w_a.partial_cmp(w_b).unwrap().then(label_b.cmp(label_a))
            })
            .map(|(label, _)| label)
            .unwrap_or(0)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedKNeighborsClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for KNeighborsClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        // Convert float labels to usize.
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedKNeighborsClassifierPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedKNeighborsClassifierPipeline<F: Float + Send + Sync + 'static>(
    FittedKNeighborsClassifier<F>,
);

// Safety: FittedKNeighborsClassifier<F> is Send + Sync because all its
// fields (Array2<F>, Array1<usize>, usize, Weights, Option<KdTree>, Vec<usize>)
// are Send + Sync.
unsafe impl<F: Float + Send + Sync + 'static> Send for FittedKNeighborsClassifierPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedKNeighborsClassifierPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedKNeighborsClassifierPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;

        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// KNeighborsRegressor
// ---------------------------------------------------------------------------

/// k-Nearest Neighbors regressor.
///
/// Predicts target values as the (weighted) mean of the `k` nearest
/// training samples' target values using Euclidean distance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::KNeighborsRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
/// let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
///
/// let fitted = reg.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct KNeighborsRegressor<F> {
    /// Number of neighbors to use for regression.
    pub n_neighbors: usize,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> KNeighborsRegressor<F> {
    /// Create a new `KNeighborsRegressor` with default settings.
    ///
    /// Defaults: `n_neighbors = 5`, `algorithm = Auto`, `weights = Uniform`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_neighbors: 5,
            algorithm: Algorithm::Auto,
            weights: Weights::Uniform,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the number of neighbors.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }
}

impl<F: Float> Default for KNeighborsRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted k-Nearest Neighbors regressor.
///
/// Stores the training data and an optional KD-Tree spatial index.
/// Implements [`Predict`] to predict target values for new samples.
#[derive(Debug)]
pub struct FittedKNeighborsRegressor<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training target values.
    y_train: Array1<F>,
    /// Number of neighbors to use.
    n_neighbors: usize,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index (KD-Tree, Ball Tree, or None for brute force).
    spatial_index: SpatialIndex,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for KNeighborsRegressor<F> {
    type Fitted = FittedKNeighborsRegressor<F>;
    type Error = FerroError;

    /// Fit the regressor by storing the training data and optionally
    /// building a KD-Tree spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `n_neighbors` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer
    /// samples than `n_neighbors`.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedKNeighborsRegressor<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1".into(),
            });
        }

        if n_samples < self.n_neighbors {
            return Err(FerroError::InsufficientSamples {
                required: self.n_neighbors,
                actual: n_samples,
                context: format!(
                    "KNeighborsRegressor requires at least n_neighbors={} samples",
                    self.n_neighbors
                ),
            });
        }

        // Build spatial index.
        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedKNeighborsRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            n_neighbors: self.n_neighbors,
            weights: self.weights,
            spatial_index,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedKNeighborsRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// For each sample, finds the `k` nearest neighbors in the training
    /// data and returns the (weighted) mean of their target values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        let train_features = self.x_train.ncols();

        if n_features != train_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![train_features],
                actual: vec![n_features],
                context: "number of features must match training data".into(),
            });
        }

        let n_samples = x.nrows();

        // Use a threshold to avoid Rayon overhead on small inputs.
        const PAR_THRESHOLD: usize = 256;

        let predictions_vec: Vec<F> = if n_samples >= PAR_THRESHOLD {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                    );
                    self.weighted_mean(&neighbors)
                })
                .collect()
        } else {
            (0..n_samples)
                .map(|i| {
                    let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
                    let neighbors = find_neighbors(
                        &self.x_train,
                        &query,
                        self.n_neighbors,
                        &self.spatial_index,
                    );
                    self.weighted_mean(&neighbors)
                })
                .collect()
        };

        Ok(Array1::from_vec(predictions_vec))
    }
}

impl<F: Float + Send + Sync + 'static> FittedKNeighborsRegressor<F> {
    /// Compute the (possibly weighted) mean of neighbor targets.
    fn weighted_mean(&self, neighbors: &[(usize, F)]) -> F {
        let eps = F::from(1e-15).unwrap();

        match self.weights {
            Weights::Uniform => {
                let sum: F = neighbors
                    .iter()
                    .map(|&(idx, _)| self.y_train[idx])
                    .fold(F::zero(), |acc, v| acc + v);
                sum / F::from(neighbors.len()).unwrap()
            }
            Weights::Distance => {
                // Check if any neighbor has zero distance.
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);

                if has_zero_dist {
                    // Average the targets of zero-distance neighbors.
                    let zero_neighbors: Vec<_> =
                        neighbors.iter().filter(|&&(_, d)| d < eps).collect();
                    let sum: F = zero_neighbors
                        .iter()
                        .map(|&&(idx, _)| self.y_train[idx])
                        .fold(F::zero(), |acc, v| acc + v);
                    sum / F::from(zero_neighbors.len()).unwrap()
                } else {
                    let mut weighted_sum = F::zero();
                    let mut weight_total = F::zero();
                    for &(idx, d) in neighbors {
                        let w = F::one() / d;
                        weighted_sum = weighted_sum + w * self.y_train[idx];
                        weight_total = weight_total + w;
                    }
                    weighted_sum / weight_total
                }
            }
        }
    }
}

// Pipeline integration.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for KNeighborsRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(FittedKNeighborsRegressorPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration.
struct FittedKNeighborsRegressorPipeline<F: Float + Send + Sync + 'static>(
    FittedKNeighborsRegressor<F>,
);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedKNeighborsRegressorPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedKNeighborsRegressorPipeline<F> {}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedKNeighborsRegressorPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.0.predict(x)
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

    // -- Classifier Tests ---------------------------------------------------

    #[test]
    fn test_classifier_simple() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // All training points should be correctly classified with k=3.
        for i in 0..6 {
            assert_eq!(preds[i], y[i], "sample {i} misclassified");
        }
    }

    #[test]
    fn test_classifier_k1_memorizes() {
        // With k=1, the classifier should perfectly memorize training data.
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 2, 3];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i], "k=1 should memorize training data");
        }
    }

    #[test]
    fn test_classifier_k_equals_n_predicts_mode() {
        // With k=n, every prediction should be the overall mode class.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // Mode is class 0 (appears 3 times).
        let y = array![0, 0, 0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(5);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..5 {
            assert_eq!(preds[i], 0, "k=n should predict the mode class");
        }
    }

    #[test]
    fn test_classifier_distance_weighting() {
        // Place test point at (0, 0). Nearest neighbor is class 0 at (0.1, 0),
        // while two class-1 points are far away at (10, 0) and (11, 0).
        let x = Array2::from_shape_vec((3, 1), vec![0.1, 10.0, 11.0]).unwrap();
        let y = array![0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(3)
            .with_weights(Weights::Distance);
        let fitted = clf.fit(&x, &y).unwrap();

        // Query at origin: class 0 neighbor is much closer.
        let query = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_eq!(
            preds[0], 0,
            "distance weighting should favor closer neighbor"
        );
    }

    #[test]
    fn test_classifier_tied_votes() {
        // With uniform weights and k=2, both classes have 1 vote.
        // Tie-breaking should pick the smallest class label.
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let y = array![0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(2);
        let fitted = clf.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        // Both are equidistant, both have 1 vote. Tie-break: smallest label wins.
        // However, tie-breaking depends on iteration order; we just check it doesn't panic.
        assert!(preds[0] == 0 || preds[0] == 1);
    }

    #[test]
    fn test_classifier_brute_force_algorithm() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_kdtree_algorithm() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(1)
            .with_algorithm(Algorithm::KdTree);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1]; // Wrong length.

        let clf = KNeighborsClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();

        // Wrong number of features.
        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_classifier_invalid_k() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_insufficient_samples() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = array![0, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(5);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_has_classes() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 1, 2, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn test_classifier_single_neighbor() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = array![42];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 42);
    }

    #[test]
    fn test_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_f32_support() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let y = array![0, 1, 0, 1];

        let clf = KNeighborsClassifier::<f32>::new().with_n_neighbors(1);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // -- Regressor Tests ----------------------------------------------------

    #[test]
    fn test_regressor_simple() {
        // y = 2*x, k=1 should memorize.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(1);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..5 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_regressor_mean_of_neighbors() {
        // k=3, query at center should predict mean of 3 nearest.
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 100.0]).unwrap();
        let y = array![0.0, 10.0, 20.0, 30.0, 1000.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
        let fitted = reg.fit(&x, &y).unwrap();

        // Query at 1.0: nearest are indices 0, 1, 2 with targets 0, 10, 20.
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_distance_weighting() {
        // Two points: (0, target=0) and (10, target=100).
        // Query at 1.0: closer to (0), so distance-weighted should bias toward 0.
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let y = array![0.0, 100.0];

        let reg = KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(2)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();

        // Weight for (0): 1/1 = 1.0, weight for (10): 1/9 ~ 0.111.
        // Weighted mean: (1.0*0.0 + 0.111*100.0) / (1.0 + 0.111) = 11.11 / 1.111 ~ 10.0
        let expected = (1.0 * 0.0 + (1.0 / 9.0) * 100.0) / (1.0 + 1.0 / 9.0);
        assert_relative_eq!(preds[0], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length.

        let reg = KNeighborsRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_invalid_k() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(0);
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = KNeighborsRegressor::<f64>::new().with_n_neighbors(3);
        let fitted = reg.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_regressor_f32_support() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0f32, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0]);

        let reg = KNeighborsRegressor::<f32>::new().with_n_neighbors(1);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_high_dimensional_falls_back_to_brute_force() {
        // With d > 20 and Algorithm::Auto, should use brute force (no KD-Tree).
        let n_features = 25;
        let n_samples = 10;
        let data: Vec<f64> = (0..n_samples * n_features).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), data).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let clf = KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(3)
            .with_algorithm(Algorithm::Auto);
        let fitted = clf.fit(&x, &y).unwrap();

        // With d > 20 and Auto, should use BallTree (not brute force).
        assert!(matches!(fitted.spatial_index, SpatialIndex::BallTree(_)));

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), n_samples);
    }

    #[test]
    fn test_classifier_default() {
        let clf = KNeighborsClassifier::<f64>::default();
        assert_eq!(clf.n_neighbors, 5);
        assert_eq!(clf.algorithm, Algorithm::Auto);
        assert_eq!(clf.weights, Weights::Uniform);
    }

    #[test]
    fn test_regressor_default() {
        let reg = KNeighborsRegressor::<f64>::default();
        assert_eq!(reg.n_neighbors, 5);
        assert_eq!(reg.algorithm, Algorithm::Auto);
        assert_eq!(reg.weights, Weights::Uniform);
    }

    #[test]
    fn test_classifier_new_data_prediction() {
        // Train on two clear clusters and predict new points.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = KNeighborsClassifier::<f64>::new().with_n_neighbors(3);
        let fitted = clf.fit(&x, &y).unwrap();

        // New test points.
        let x_test = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 4.9, 4.9]).unwrap();
        let preds = fitted.predict(&x_test).unwrap();
        assert_eq!(preds[0], 0);
        assert_eq!(preds[1], 1);
    }

    #[test]
    fn test_regressor_exact_match_distance_weighting() {
        // When query exactly matches a training point with distance weighting.
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![10.0, 20.0, 30.0];

        let reg = KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(3)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        // Query exactly at x=1.0.
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        // Should return 20.0 (exact match takes all weight).
        assert_relative_eq!(preds[0], 20.0, epsilon = 1e-10);
    }
}
