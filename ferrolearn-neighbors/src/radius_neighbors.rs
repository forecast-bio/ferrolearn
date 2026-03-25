//! Radius-based nearest neighbor classifier and regressor.
//!
//! This module provides [`RadiusNeighborsClassifier`] and
//! [`RadiusNeighborsRegressor`], which use all training samples within a
//! fixed radius of each query point for prediction, rather than a fixed
//! number of neighbors.
//!
//! # When to Use Radius-Based Models
//!
//! - When the number of relevant neighbors varies across the feature space
//! - When you have a natural notion of "closeness" in your domain
//! - When outlier detection is important (points with no neighbors)
//!
//! # Examples
//!
//! ```
//! use ferrolearn_neighbors::{RadiusNeighborsClassifier, Algorithm, Weights};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
//!     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let clf = RadiusNeighborsClassifier::<f64>::new()
//!     .with_radius(1.5)
//!     .with_outlier_label(Some(99));
//! let fitted = clf.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::balltree::BallTree;
use crate::kdtree::{self, KdTree};
use crate::knn::{Algorithm, Weights};

// ---------------------------------------------------------------------------
// Shared spatial index
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
            SpatialIndex::None => write!(f, "BruteForce"),
            SpatialIndex::KdTree(t) => write!(f, "KdTree({t:?})"),
            SpatialIndex::BallTree(t) => write!(f, "BallTree({t:?})"),
        }
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

/// Find all neighbors within `radius` of a query point.
///
/// Returns `(index, distance)` pairs sorted by distance ascending.
fn find_radius_neighbors<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query: &[F],
    radius: F,
    index: &SpatialIndex,
) -> Vec<(usize, F)> {
    let mut results = match index {
        SpatialIndex::BallTree(tree) => {
            let query_f64: Vec<f64> = query.iter().map(|v| v.to_f64().unwrap()).collect();
            let radius_f64 = radius.to_f64().unwrap();
            tree.within_radius(&query_f64, radius_f64)
                .into_iter()
                .map(|(idx, dist)| (idx, F::from(dist).unwrap()))
                .collect()
        }
        _ => {
            // Brute force radius search for KdTree and BruteForce modes.
            brute_force_radius(data, query, radius)
        }
    };

    // Sort by distance ascending.
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Brute-force radius search: find all training points within `radius` of `query`.
fn brute_force_radius<F: Float + Send + Sync + 'static>(
    data: &Array2<F>,
    query: &[F],
    radius: F,
) -> Vec<(usize, F)> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let mut results = Vec::new();

    for i in 0..n_samples {
        let point: Vec<F> = (0..n_features).map(|j| data[[i, j]]).collect();
        let dist = kdtree::euclidean_distance(&point, query);
        if dist <= radius {
            results.push((i, dist));
        }
    }

    results
}

// ===========================================================================
// RadiusNeighborsClassifier
// ===========================================================================

/// Radius-based nearest neighbor classifier.
///
/// Classifies samples by majority vote of all training points within a
/// given radius, using Euclidean distance.
///
/// When a query point has no training neighbors within the radius, the
/// `outlier_label` is used if set. If `outlier_label` is `None` and a
/// point has no neighbors, prediction returns an error.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::RadiusNeighborsClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
///     5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(1.5);
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct RadiusNeighborsClassifier<F> {
    /// Search radius (default 1.0).
    pub radius: F,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    /// Optional label assigned to query points with no neighbors within
    /// the radius. If `None`, an error is returned for such points.
    pub outlier_label: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> RadiusNeighborsClassifier<F> {
    /// Create a new `RadiusNeighborsClassifier` with default settings.
    ///
    /// Defaults: `radius = 1.0`, `weights = Uniform`, `algorithm = Auto`,
    /// `outlier_label = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            radius: F::one(),
            weights: Weights::Uniform,
            algorithm: Algorithm::Auto,
            outlier_label: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the search radius.
    #[must_use]
    pub fn with_radius(mut self, radius: F) -> Self {
        self.radius = radius;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the outlier label for points with no neighbors.
    #[must_use]
    pub fn with_outlier_label(mut self, outlier_label: Option<usize>) -> Self {
        self.outlier_label = outlier_label;
        self
    }
}

impl<F: Float> Default for RadiusNeighborsClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted radius-based nearest neighbor classifier.
///
/// Stores the training data, labels, and spatial index. Implements
/// [`Predict`] to classify new samples by majority vote of neighbors
/// within the configured radius.
#[derive(Debug)]
pub struct FittedRadiusNeighborsClassifier<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training labels.
    y_train: Array1<usize>,
    /// Search radius.
    radius: F,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index.
    spatial_index: SpatialIndex,
    /// Optional outlier label.
    outlier_label: Option<usize>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for RadiusNeighborsClassifier<F>
{
    type Fitted = FittedRadiusNeighborsClassifier<F>;
    type Error = FerroError;

    /// Fit the classifier by storing the training data and building a
    /// spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `radius` is non-positive.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedRadiusNeighborsClassifier<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.radius <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "radius".into(),
                reason: "must be positive".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RadiusNeighborsClassifier requires at least 1 sample".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedRadiusNeighborsClassifier {
            x_train: x.clone(),
            y_train: y.clone(),
            radius: self.radius,
            weights: self.weights,
            spatial_index,
            outlier_label: self.outlier_label,
            classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>>
    for FittedRadiusNeighborsClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// For each sample, finds all training points within `radius` and
    /// returns the majority class (with optional distance weighting).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    /// Returns [`FerroError::InvalidParameter`] if any query point has no
    /// neighbors within the radius and `outlier_label` is `None`.
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
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
            let neighbors = find_radius_neighbors(
                &self.x_train,
                &query,
                self.radius,
                &self.spatial_index,
            );

            if neighbors.is_empty() {
                match self.outlier_label {
                    Some(label) => predictions.push(label),
                    None => {
                        return Err(FerroError::InvalidParameter {
                            name: "radius".into(),
                            reason: format!(
                                "query sample {i} has no neighbors within radius={}; \
                                 set outlier_label to handle this case",
                                self.radius.to_f64().unwrap_or(0.0)
                            ),
                        });
                    }
                }
            } else {
                predictions.push(self.weighted_vote(&neighbors));
            }
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float + Send + Sync + 'static> FittedRadiusNeighborsClassifier<F> {
    /// Perform a (possibly weighted) majority vote among neighbors.
    fn weighted_vote(&self, neighbors: &[(usize, F)]) -> usize {
        let mut class_weights: HashMap<usize, F> = HashMap::new();
        let eps = F::from(1e-15).unwrap();

        match self.weights {
            Weights::Uniform => {
                for &(idx, _) in neighbors {
                    let label = self.y_train[idx];
                    *class_weights.entry(label).or_insert_with(F::zero) =
                        *class_weights.entry(label).or_insert_with(F::zero) + F::one();
                }
            }
            Weights::Distance => {
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);

                if has_zero_dist {
                    for &(idx, d) in neighbors {
                        if d < eps {
                            let label = self.y_train[idx];
                            *class_weights.entry(label).or_insert_with(F::zero) =
                                *class_weights.entry(label).or_insert_with(F::zero) + F::one();
                        }
                    }
                } else {
                    for &(idx, d) in neighbors {
                        let label = self.y_train[idx];
                        let w = F::one() / d;
                        *class_weights.entry(label).or_insert_with(F::zero) =
                            *class_weights.entry(label).or_insert_with(F::zero) + w;
                    }
                }
            }
        }

        // Find the class with the maximum weight. Tie-break: smallest label.
        class_weights
            .into_iter()
            .max_by(|(label_a, w_a), (label_b, w_b)| {
                w_a.partial_cmp(w_b).unwrap().then(label_b.cmp(label_a))
            })
            .map(|(label, _)| label)
            .unwrap_or(0)
    }

    /// Return the unique class labels found during fitting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Return the number of unique classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for classifier.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for RadiusNeighborsClassifier<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedRadiusNeighborsClassifierPipeline(fitted)))
    }
}

/// Pipeline wrapper for the radius-based classifier.
struct FittedRadiusNeighborsClassifierPipeline<F: Float + Send + Sync + 'static>(
    FittedRadiusNeighborsClassifier<F>,
);

unsafe impl<F: Float + Send + Sync + 'static> Send
    for FittedRadiusNeighborsClassifierPipeline<F>
{
}
unsafe impl<F: Float + Send + Sync + 'static> Sync
    for FittedRadiusNeighborsClassifierPipeline<F>
{
}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedRadiusNeighborsClassifierPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ===========================================================================
// RadiusNeighborsRegressor
// ===========================================================================

/// Radius-based nearest neighbor regressor.
///
/// Predicts target values as the (weighted) mean of all training points
/// within a given radius of each query point, using Euclidean distance.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_neighbors::RadiusNeighborsRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![0.0, 2.0, 4.0, 6.0, 8.0];
///
/// let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.5);
/// let fitted = reg.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// assert_eq!(preds.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct RadiusNeighborsRegressor<F> {
    /// Search radius (default 1.0).
    pub radius: F,
    /// The weighting scheme for neighbor contributions.
    pub weights: Weights,
    /// The algorithm to use for neighbor search.
    pub algorithm: Algorithm,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> RadiusNeighborsRegressor<F> {
    /// Create a new `RadiusNeighborsRegressor` with default settings.
    ///
    /// Defaults: `radius = 1.0`, `weights = Uniform`, `algorithm = Auto`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            radius: F::one(),
            weights: Weights::Uniform,
            algorithm: Algorithm::Auto,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the search radius.
    #[must_use]
    pub fn with_radius(mut self, radius: F) -> Self {
        self.radius = radius;
        self
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the algorithm for neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl<F: Float> Default for RadiusNeighborsRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted radius-based nearest neighbor regressor.
///
/// Stores the training data, targets, and spatial index. Implements
/// [`Predict`] to predict target values for new samples.
#[derive(Debug)]
pub struct FittedRadiusNeighborsRegressor<F> {
    /// Training feature data.
    x_train: Array2<F>,
    /// Training target values.
    y_train: Array1<F>,
    /// Search radius.
    radius: F,
    /// Weighting scheme.
    weights: Weights,
    /// Spatial index.
    spatial_index: SpatialIndex,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>>
    for RadiusNeighborsRegressor<F>
{
    type Fitted = FittedRadiusNeighborsRegressor<F>;
    type Error = FerroError;

    /// Fit the regressor by storing the training data and building a
    /// spatial index.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `radius` is non-positive.
    /// Returns [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedRadiusNeighborsRegressor<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.radius <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "radius".into(),
                reason: "must be positive".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RadiusNeighborsRegressor requires at least 1 sample".into(),
            });
        }

        let spatial_index = build_spatial_index(self.algorithm, x);

        Ok(FittedRadiusNeighborsRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            radius: self.radius,
            weights: self.weights,
            spatial_index,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>>
    for FittedRadiusNeighborsRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// For each sample, finds all training points within `radius` and
    /// returns the (weighted) mean of their target values.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    /// Returns [`FerroError::InvalidParameter`] if any query point has no
    /// neighbors within the radius.
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
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let query: Vec<F> = (0..n_features).map(|j| x[[i, j]]).collect();
            let neighbors = find_radius_neighbors(
                &self.x_train,
                &query,
                self.radius,
                &self.spatial_index,
            );

            if neighbors.is_empty() {
                return Err(FerroError::InvalidParameter {
                    name: "radius".into(),
                    reason: format!(
                        "query sample {i} has no neighbors within radius={}",
                        self.radius.to_f64().unwrap_or(0.0)
                    ),
                });
            }

            predictions.push(self.weighted_mean(&neighbors));
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl<F: Float + Send + Sync + 'static> FittedRadiusNeighborsRegressor<F> {
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
                let has_zero_dist = neighbors.iter().any(|&(_, d)| d < eps);

                if has_zero_dist {
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

// Pipeline integration for regressor.
impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for RadiusNeighborsRegressor<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(FittedRadiusNeighborsRegressorPipeline(fitted)))
    }
}

/// Pipeline wrapper for the radius-based regressor.
struct FittedRadiusNeighborsRegressorPipeline<F: Float + Send + Sync + 'static>(
    FittedRadiusNeighborsRegressor<F>,
);

unsafe impl<F: Float + Send + Sync + 'static> Send
    for FittedRadiusNeighborsRegressorPipeline<F>
{
}
unsafe impl<F: Float + Send + Sync + 'static> Sync
    for FittedRadiusNeighborsRegressorPipeline<F>
{
}

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedRadiusNeighborsRegressorPipeline<F>
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

    fn clf_train_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_classifier_basic() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(1.5);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..6 {
            assert_eq!(preds[i], y[i], "sample {i} misclassified");
        }
    }

    #[test]
    fn test_classifier_with_outlier_label() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(0.1)
            .with_outlier_label(Some(99));
        let fitted = clf.fit(&x, &y).unwrap();

        // Query far from all training points.
        let query = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_eq!(preds[0], 99);
    }

    #[test]
    fn test_classifier_no_neighbors_no_outlier_label_errors() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(0.01);
        let fitted = clf.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        assert!(fitted.predict(&query).is_err());
    }

    #[test]
    fn test_classifier_distance_weighting() {
        // Point at (0.1, 0) is near (0,0) [class 0] and far from (10,0) [class 1].
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 10.0, 11.0]).unwrap();
        let y = array![0, 1, 1];

        let clf = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(15.0)
            .with_weights(Weights::Distance);
        let fitted = clf.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        // Class 0 is closer, so distance weighting should favor it.
        assert_eq!(preds[0], 0);
    }

    #[test]
    fn test_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1]; // Wrong length.

        let clf = RadiusNeighborsClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_shape_mismatch_predict() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(1.5);
        let fitted = clf.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_classifier_invalid_radius() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(0.0);
        assert!(clf.fit(&x, &y).is_err());

        let clf_neg = RadiusNeighborsClassifier::<f64>::new().with_radius(-1.0);
        assert!(clf_neg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_empty_training_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = array![];

        let clf = RadiusNeighborsClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_exact_match_vote() {
        // Query exactly on a training point.
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![10, 20, 30];

        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(0.5);
        let fitted = clf.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds[0], 10);
        assert_eq!(preds[1], 20);
        assert_eq!(preds[2], 30);
    }

    #[test]
    fn test_classifier_classes() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(1.5);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_classifier_brute_force() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..6 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_balltree() {
        let (x, y) = clf_train_data();
        let clf = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BallTree);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..6 {
            assert_eq!(preds[i], y[i]);
        }
    }

    #[test]
    fn test_classifier_pipeline() {
        let (x, _) = clf_train_data();
        let y_f64 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let clf = RadiusNeighborsClassifier::<f64>::new().with_radius(1.5);
        let fitted = clf.fit_pipeline(&x, &y_f64).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_f32() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let clf = RadiusNeighborsClassifier::<f32>::new().with_radius(1.5);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_classifier_default() {
        let clf = RadiusNeighborsClassifier::<f64>::default();
        assert_eq!(clf.radius, 1.0);
        assert_eq!(clf.weights, Weights::Uniform);
        assert_eq!(clf.algorithm, Algorithm::Auto);
        assert!(clf.outlier_label.is_none());
    }

    // -- Regressor Tests ----------------------------------------------------

    #[test]
    fn test_regressor_basic() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 2.0, 4.0, 6.0, 8.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.5);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_regressor_mean_of_neighbors() {
        // Points at 0, 1, 2 with targets 0, 10, 20.
        // Query at 1.0 with radius 1.5: neighbors are 0,1,2.
        // Uniform mean = (0 + 10 + 20) / 3 = 10.
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![0.0, 10.0, 20.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.5);
        let fitted = reg.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_exact_match() {
        // Query exactly on a training point with radius that only catches it.
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 5.0, 10.0]).unwrap();
        let y = array![100.0, 200.0, 300.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.0);
        let fitted = reg.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        assert_relative_eq!(preds[0], 100.0, epsilon = 1e-10);
        assert_relative_eq!(preds[1], 200.0, epsilon = 1e-10);
        assert_relative_eq!(preds[2], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_distance_weighting() {
        // Two points: (0, target=0) and (10, target=100).
        // Query at 1.0 with large radius.
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let y = array![0.0, 100.0];

        let reg = RadiusNeighborsRegressor::<f64>::new()
            .with_radius(15.0)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();

        // Weight for (0): 1/1 = 1.0, weight for (10): 1/9.
        let expected = (1.0 * 0.0 + (1.0 / 9.0) * 100.0) / (1.0 + 1.0 / 9.0);
        assert_relative_eq!(preds[0], expected, epsilon = 1e-6);
    }

    #[test]
    fn test_regressor_exact_match_distance_weighting() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![10.0, 20.0, 30.0];

        let reg = RadiusNeighborsRegressor::<f64>::new()
            .with_radius(5.0)
            .with_weights(Weights::Distance);
        let fitted = reg.fit(&x, &y).unwrap();

        // Query exactly at x=1.0: exact match takes all weight.
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let preds = fitted.predict(&query).unwrap();
        assert_relative_eq!(preds[0], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regressor_no_neighbors_errors() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
        let y = array![0.0, 100.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(0.01);
        let fitted = reg.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        assert!(fitted.predict(&query).is_err());
    }

    #[test]
    fn test_regressor_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length.

        let reg = RadiusNeighborsRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_shape_mismatch_predict() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(5.0);
        let fitted = reg.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_regressor_invalid_radius() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(0.0);
        assert!(reg.fit(&x, &y).is_err());

        let reg_neg = RadiusNeighborsRegressor::<f64>::new().with_radius(-1.0);
        assert!(reg_neg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_empty_training_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = array![];

        let reg = RadiusNeighborsRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_brute_force() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![0.0, 10.0, 20.0];

        let reg = RadiusNeighborsRegressor::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BruteForce);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_regressor_balltree() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = array![0.0, 10.0, 20.0];

        let reg = RadiusNeighborsRegressor::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BallTree);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_regressor_pipeline() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 2.0, 4.0, 6.0, 8.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.5);
        let fitted = reg.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_regressor_f32() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0f32, 1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0f32, 10.0, 20.0]);

        let reg = RadiusNeighborsRegressor::<f32>::new().with_radius(1.5);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_regressor_default() {
        let reg = RadiusNeighborsRegressor::<f64>::default();
        assert_eq!(reg.radius, 1.0);
        assert_eq!(reg.weights, Weights::Uniform);
        assert_eq!(reg.algorithm, Algorithm::Auto);
    }

    #[test]
    fn test_regressor_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = array![42.0];

        let reg = RadiusNeighborsRegressor::<f64>::new().with_radius(1.0);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_relative_eq!(preds[0], 42.0, epsilon = 1e-10);
    }

    #[test]
    fn test_classifier_and_regressor_agree_on_neighbors() {
        // Verify that both find the same neighbors for a given radius.
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0],
        )
        .unwrap();
        let y_cls = array![0, 0, 0, 1];
        let y_reg = array![0.0, 1.0, 2.0, 3.0];

        let clf = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BruteForce);
        let reg = RadiusNeighborsRegressor::<f64>::new()
            .with_radius(1.5)
            .with_algorithm(Algorithm::BruteForce);

        let fitted_clf = clf.fit(&x, &y_cls).unwrap();
        let fitted_reg = reg.fit(&x, &y_reg).unwrap();

        // Both should produce predictions of the correct length.
        let preds_clf = fitted_clf.predict(&x).unwrap();
        let preds_reg = fitted_reg.predict(&x).unwrap();
        assert_eq!(preds_clf.len(), 4);
        assert_eq!(preds_reg.len(), 4);
    }
}
