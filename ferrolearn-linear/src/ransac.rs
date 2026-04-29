//! RANSAC (RANdom SAmple Consensus) robust regression.
//!
//! This module provides [`RANSACRegressor`], a meta-estimator that fits a
//! base regressor to inlier data, automatically detecting and excluding
//! outliers.
//!
//! # Algorithm
//!
//! 1. Randomly sample `min_samples` points.
//! 2. Fit the base estimator on the sample.
//! 3. Compute residuals for all points, identify inliers (residual below
//!    `residual_threshold`).
//! 4. If enough inliers, refit on all inliers.
//! 5. Keep the model with the most inliers (ties broken by lowest residual).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ransac::RANSACRegressor;
//! use ferrolearn_linear::LinearRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! // Data with an outlier at index 4.
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 100.0]; // last point is outlier
//!
//! let base = LinearRegression::<f64>::new();
//! let model = RANSACRegressor::new(base);
//! let fitted = model.fit(&x, &y).unwrap();
//!
//! // The outlier should be detected.
//! let mask = fitted.inlier_mask();
//! assert!(!mask[4], "outlier at index 4 should be detected");
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// RANSACRegressor (unfitted)
// ---------------------------------------------------------------------------

/// RANSAC robust regression meta-estimator.
///
/// Wraps a base regressor (e.g., [`LinearRegression`](crate::LinearRegression))
/// and repeatedly fits it on random subsets to find a model robust to
/// outliers.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `E`: The base estimator type.
#[derive(Debug, Clone)]
pub struct RANSACRegressor<F, E> {
    /// The base estimator.
    pub estimator: E,
    /// Minimum number of samples for fitting.
    pub min_samples: Option<usize>,
    /// Residual threshold: points with absolute residual below this are
    /// considered inliers. If `None`, uses the MAD of the target.
    pub residual_threshold: Option<F>,
    /// Maximum number of random trials.
    pub max_trials: usize,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
}

impl<F: Float, E> RANSACRegressor<F, E> {
    /// Create a new `RANSACRegressor` with the given base estimator.
    ///
    /// Defaults: `min_samples = None` (auto: n_features + 1),
    /// `residual_threshold = None` (auto: MAD), `max_trials = 100`,
    /// `random_state = None`.
    #[must_use]
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            random_state: None,
        }
    }

    /// Set the minimum number of samples for fitting.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(min_samples);
        self
    }

    /// Set the residual threshold for inlier detection.
    #[must_use]
    pub fn with_residual_threshold(mut self, threshold: F) -> Self {
        self.residual_threshold = Some(threshold);
        self
    }

    /// Set the maximum number of random trials.
    #[must_use]
    pub fn with_max_trials(mut self, max_trials: usize) -> Self {
        self.max_trials = max_trials;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

// ---------------------------------------------------------------------------
// FittedRANSACRegressor
// ---------------------------------------------------------------------------

/// Fitted RANSAC robust regression model.
///
/// Stores the best estimator fitted on inlier data, and the inlier mask.
#[derive(Debug, Clone)]
pub struct FittedRANSACRegressor<Fitted> {
    /// The fitted base estimator (fitted on inliers).
    fitted_estimator: Fitted,
    /// Boolean mask: true if the sample was classified as an inlier.
    inlier_mask: Vec<bool>,
}

impl<Fitted> FittedRANSACRegressor<Fitted> {
    /// Returns the inlier mask. `true` indicates the sample was an inlier.
    #[must_use]
    pub fn inlier_mask(&self) -> &[bool] {
        &self.inlier_mask
    }
}

// ---------------------------------------------------------------------------
// Helper: Median Absolute Deviation
// ---------------------------------------------------------------------------

/// Compute the median of a slice of floats.
fn median<F: Float>(values: &[F]) -> F {
    let mut sorted: Vec<F> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / (F::one() + F::one())
    } else {
        sorted[n / 2]
    }
}

/// Compute the Median Absolute Deviation (MAD) of a slice.
fn mad<F: Float>(values: &[F]) -> F {
    let med = median(values);
    let abs_devs: Vec<F> = values.iter().map(|&v| (v - med).abs()).collect();
    median(&abs_devs)
}

// ---------------------------------------------------------------------------
// Random subset sampling
// ---------------------------------------------------------------------------

/// Sample `k` distinct indices from `0..n` using Fisher-Yates.
fn sample_indices<R: Rng>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

/// Extract a subset of rows from a 2D array and a 1D array.
fn subset<F: Float>(x: &Array2<F>, y: &Array1<F>, indices: &[usize]) -> (Array2<F>, Array1<F>) {
    let n_features = x.ncols();
    let n = indices.len();
    let mut x_sub = Array2::<F>::zeros((n, n_features));
    let mut y_sub = Array1::<F>::zeros(n);
    for (row, &idx) in indices.iter().enumerate() {
        for col in 0..n_features {
            x_sub[[row, col]] = x[[idx, col]];
        }
        y_sub[row] = y[idx];
    }
    (x_sub, y_sub)
}

// ---------------------------------------------------------------------------
// Fit and Predict
// ---------------------------------------------------------------------------

impl<F, E, Ef> Fit<Array2<F>, Array1<F>> for RANSACRegressor<F, E>
where
    F: Float + Send + Sync + ScalarOperand + num_traits::FromPrimitive + 'static,
    E: Fit<Array2<F>, Array1<F>, Fitted = Ef, Error = FerroError> + Clone,
    Ef: Predict<Array2<F>, Output = Array1<F>, Error = FerroError> + Clone,
{
    type Fitted = FittedRANSACRegressor<Ef>;
    type Error = FerroError;

    /// Fit the RANSAC model by repeatedly sampling and fitting.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::ConvergenceFailure`] if no valid model is found
    /// after `max_trials` iterations.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedRANSACRegressor<E::Fitted>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        let min_samples = self.min_samples.unwrap_or(n_features + 1).max(1);

        if n_samples < min_samples {
            return Err(FerroError::InsufficientSamples {
                required: min_samples,
                actual: n_samples,
                context: "RANSAC requires at least min_samples samples".into(),
            });
        }

        // Compute residual threshold if not provided.
        let threshold = if let Some(t) = self.residual_threshold {
            t
        } else {
            let y_mad = mad(&y.to_vec());
            if y_mad <= F::epsilon() {
                // If MAD is zero (constant target), use a small default.
                F::from(1e-6).unwrap()
            } else {
                y_mad
            }
        };

        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        let mut best_fitted: Option<E::Fitted> = None;
        let mut best_inlier_mask: Option<Vec<bool>> = None;
        let mut best_n_inliers = 0usize;
        let mut best_residual_sum = F::infinity();

        for _ in 0..self.max_trials {
            // Sample random subset.
            let indices = sample_indices(&mut rng, n_samples, min_samples);
            let (x_sub, y_sub) = subset(x, y, &indices);

            // Fit base estimator on the subset.
            let fitted = match self.estimator.fit(&x_sub, &y_sub) {
                Ok(f) => f,
                Err(_) => continue, // Skip failed fits.
            };

            // Compute residuals for all points.
            let preds = match fitted.predict(x) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let mut inlier_mask = vec![false; n_samples];
            let mut n_inliers = 0usize;
            let mut residual_sum = F::zero();

            for i in 0..n_samples {
                let residual = (preds[i] - y[i]).abs();
                if residual <= threshold {
                    inlier_mask[i] = true;
                    n_inliers += 1;
                    residual_sum = residual_sum + residual;
                }
            }

            // Check if this is better than the current best.
            let is_better = n_inliers > best_n_inliers
                || (n_inliers == best_n_inliers && residual_sum < best_residual_sum);

            if is_better && n_inliers >= min_samples {
                // Refit on all inliers.
                let inlier_indices: Vec<usize> = inlier_mask
                    .iter()
                    .enumerate()
                    .filter(|&(_, &is_inlier)| is_inlier)
                    .map(|(i, _)| i)
                    .collect();
                let (x_inlier, y_inlier) = subset(x, y, &inlier_indices);

                if let Ok(refit) = self.estimator.fit(&x_inlier, &y_inlier) {
                    // Recompute inlier mask with the refitted model.
                    if let Ok(new_preds) = refit.predict(x) {
                        let mut new_mask = vec![false; n_samples];
                        let mut new_n_inliers = 0;
                        let mut new_residual_sum = F::zero();
                        for i in 0..n_samples {
                            let r = (new_preds[i] - y[i]).abs();
                            if r <= threshold {
                                new_mask[i] = true;
                                new_n_inliers += 1;
                                new_residual_sum = new_residual_sum + r;
                            }
                        }
                        best_fitted = Some(refit);
                        best_inlier_mask = Some(new_mask);
                        best_n_inliers = new_n_inliers;
                        best_residual_sum = new_residual_sum;
                    }
                } else {
                    // Keep the original fit if refit fails.
                    best_fitted = Some(fitted);
                    best_inlier_mask = Some(inlier_mask);
                    best_n_inliers = n_inliers;
                    best_residual_sum = residual_sum;
                }
            }
        }

        match (best_fitted, best_inlier_mask) {
            (Some(fitted), Some(mask)) => Ok(FittedRANSACRegressor {
                fitted_estimator: fitted,
                inlier_mask: mask,
            }),
            _ => Err(FerroError::ConvergenceFailure {
                iterations: self.max_trials,
                message: "RANSAC could not find a valid model after max_trials iterations".into(),
            }),
        }
    }
}

impl<F, Fitted> Predict<Array2<F>> for FittedRANSACRegressor<Fitted>
where
    F: Float + Send + Sync + 'static,
    Fitted: Predict<Array2<F>, Output = Array1<F>, Error = FerroError>,
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values using the base estimator fitted on inliers.
    ///
    /// # Errors
    ///
    /// Returns any error from the base estimator's predict method.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.fitted_estimator.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearRegression;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_ransac_no_outliers() {
        // Perfect linear data, no outliers.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_residual_threshold(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        // All should be inliers.
        let mask = fitted.inlier_mask();
        assert!(mask.iter().all(|&v| v), "All should be inliers");

        // Predictions should be accurate.
        let preds = fitted.predict(&x).unwrap();
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 0.5);
        }
    }

    #[test]
    fn test_ransac_with_outlier() {
        // y = 2x, but one outlier.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0]; // outlier at idx 5

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_max_trials(200)
            .with_residual_threshold(2.0);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // The outlier at index 5 should be detected.
        assert!(!mask[5], "Outlier at index 5 should not be an inlier");

        // Most other points should be inliers.
        let n_inliers: usize = mask.iter().filter(|&&v| v).count();
        assert!(
            n_inliers >= 4,
            "Expected at least 4 inliers, got {n_inliers}"
        );

        // The prediction at x=3 should be close to 6.
        let x_test = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let pred = fitted.predict(&x_test).unwrap();
        assert!(
            (pred[0] - 6.0).abs() < 3.0,
            "Prediction at x=3 should be near 6.0, got {}",
            pred[0]
        );
    }

    #[test]
    fn test_ransac_multiple_outliers() {
        // y = x + 1, with two outliers.
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![2.0, 3.0, 50.0, 5.0, 6.0, -40.0, 8.0, 9.0]; // outliers at 2 and 5

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(123)
            .with_max_trials(500)
            .with_residual_threshold(2.0);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // Outliers at index 2 and 5 should be detected.
        assert!(!mask[2], "Outlier at index 2 should not be an inlier");
        assert!(!mask[5], "Outlier at index 5 should not be an inlier");
    }

    #[test]
    fn test_ransac_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ransac_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base).with_min_samples(3);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ransac_reproducible_with_seed() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0];

        let base1 = LinearRegression::<f64>::new();
        let model1 = RANSACRegressor::new(base1)
            .with_random_state(42)
            .with_residual_threshold(2.0);
        let fitted1 = model1.fit(&x, &y).unwrap();

        let base2 = LinearRegression::<f64>::new();
        let model2 = RANSACRegressor::new(base2)
            .with_random_state(42)
            .with_residual_threshold(2.0);
        let fitted2 = model2.fit(&x, &y).unwrap();

        // Same seed should produce same inlier mask.
        assert_eq!(fitted1.inlier_mask(), fitted2.inlier_mask());
    }

    #[test]
    fn test_ransac_auto_threshold() {
        // No explicit threshold — should use MAD.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_max_trials(200);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // At least some points should be inliers.
        let n_inliers: usize = mask.iter().filter(|&&v| v).count();
        assert!(
            n_inliers >= 3,
            "Expected at least 3 inliers, got {n_inliers}"
        );
    }
}
