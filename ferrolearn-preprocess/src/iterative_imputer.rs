//! Iterative imputer: fill missing values by modeling each feature as a function
//! of all other features.
//!
//! [`IterativeImputer`] performs round-robin imputation: for each feature with
//! missing values, it fits a simple Ridge regression on the non-missing rows
//! using the other features as predictors, then predicts the missing values.
//! This process is repeated for `max_iter` iterations or until convergence.
//!
//! # Initial Imputation
//!
//! Before the iterative process begins, missing values are filled using a simple
//! strategy (mean by default). This initial imputation provides a starting point
//! for the regression models.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// InitialStrategy
// ---------------------------------------------------------------------------

/// Strategy for the initial imputation before iterative refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialStrategy {
    /// Replace NaN with the column mean.
    Mean,
    /// Replace NaN with the column median.
    Median,
}

// ---------------------------------------------------------------------------
// IterativeImputer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted iterative imputer.
///
/// Calling [`Fit::fit`] learns the imputation model and returns a
/// [`FittedIterativeImputer`] that can impute missing values in new data.
///
/// # Parameters
///
/// - `max_iter` — maximum number of imputation rounds (default 10).
/// - `tol` — convergence tolerance on the total change in imputed values
///   (default 1e-3).
/// - `initial_strategy` — strategy for the initial fill (default `Mean`).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::iterative_imputer::{IterativeImputer, InitialStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
/// let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
/// let fitted = imputer.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert!(!out[[1, 1]].is_nan());
/// assert!(!out[[2, 0]].is_nan());
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct IterativeImputer<F> {
    /// Maximum number of imputation rounds.
    max_iter: usize,
    /// Convergence tolerance.
    tol: F,
    /// Initial imputation strategy.
    initial_strategy: InitialStrategy,
}

impl<F: Float + Send + Sync + 'static> IterativeImputer<F> {
    /// Create a new `IterativeImputer` with the given parameters.
    pub fn new(max_iter: usize, tol: F, initial_strategy: InitialStrategy) -> Self {
        Self {
            max_iter,
            tol,
            initial_strategy,
        }
    }

    /// Return the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> F {
        self.tol
    }

    /// Return the initial imputation strategy.
    #[must_use]
    pub fn initial_strategy(&self) -> InitialStrategy {
        self.initial_strategy
    }
}

impl<F: Float + Send + Sync + 'static> Default for IterativeImputer<F> {
    fn default() -> Self {
        Self::new(
            10,
            F::from(1e-3).unwrap_or_else(F::epsilon),
            InitialStrategy::Mean,
        )
    }
}

// ---------------------------------------------------------------------------
// FittedIterativeImputer
// ---------------------------------------------------------------------------

/// A fitted iterative imputer that stores per-feature Ridge regression
/// coefficients learned during fitting.
///
/// Created by calling [`Fit::fit`] on an [`IterativeImputer`].
#[derive(Debug, Clone)]
pub struct FittedIterativeImputer<F> {
    /// Per-feature initial fill values (used for initial imputation of transform data).
    initial_fill: Array1<F>,
    /// Per-feature Ridge coefficients: `coefs[j]` is the coefficient vector
    /// for predicting feature `j` from the other features.
    /// Only stored for features that had missing values during training.
    feature_models: Vec<Option<FeatureModel<F>>>,
    /// Indices of features that had missing values during training.
    missing_features: Vec<usize>,
    /// Number of iterations that were performed during fitting.
    n_iter: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: F,
    /// Initial strategy.
    initial_strategy: InitialStrategy,
}

/// Ridge regression model for a single feature.
#[derive(Debug, Clone)]
struct FeatureModel<F> {
    /// Coefficients (one per predictor feature).
    coefficients: Array1<F>,
    /// Intercept.
    intercept: F,
}

impl<F: Float + Send + Sync + 'static> FittedIterativeImputer<F> {
    /// Return the number of iterations performed during fitting.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Return the initial fill values per feature.
    #[must_use]
    pub fn initial_fill(&self) -> &Array1<F> {
        &self.initial_fill
    }

    /// Return the initial imputation strategy used during fitting.
    #[must_use]
    pub fn initial_strategy(&self) -> InitialStrategy {
        self.initial_strategy
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute column means, ignoring NaN values.
fn column_means_nan<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n_features = x.ncols();
    let mut means = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = x.column(j);
        let mut sum = F::zero();
        let mut count = 0usize;
        for &v in col {
            if !v.is_nan() {
                sum = sum + v;
                count += 1;
            }
        }
        means[j] = if count > 0 {
            sum / F::from(count).unwrap_or_else(F::one)
        } else {
            F::zero()
        };
    }
    means
}

/// Compute column medians, ignoring NaN values.
fn column_medians_nan<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n_features = x.ncols();
    let mut medians = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = x.column(j);
        let mut vals: Vec<F> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        if vals.is_empty() {
            medians[j] = F::zero();
        } else {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = vals.len();
            medians[j] = if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) / (F::one() + F::one())
            };
        }
    }
    medians
}

/// Fill NaN values in a matrix with the given fill values.
fn initial_fill<F: Float>(x: &Array2<F>, fill: &Array1<F>) -> Array2<F> {
    let mut out = x.to_owned();
    for (mut col, &f) in out.columns_mut().into_iter().zip(fill.iter()) {
        for v in &mut col {
            if v.is_nan() {
                *v = f;
            }
        }
    }
    out
}

/// Fit a simple Ridge regression: y = X * beta + intercept.
/// Uses the closed-form solution: beta = (X^T X + alpha * I)^{-1} X^T y.
///
/// For simplicity we solve this using a small linear system solver.
fn ridge_fit<F: Float>(x: &Array2<F>, y: &Array1<F>, alpha: F) -> Option<(Array1<F>, F)> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 || n_features == 0 {
        return None;
    }

    // Center y
    let y_mean =
        y.iter().copied().fold(F::zero(), |a, v| a + v) / F::from(n_samples).unwrap_or_else(F::one);

    // Center X
    let mut x_means = Array1::zeros(n_features);
    for j in 0..n_features {
        x_means[j] = x.column(j).iter().copied().fold(F::zero(), |a, v| a + v)
            / F::from(n_samples).unwrap_or_else(F::one);
    }

    // Compute X^T X + alpha * I (n_features x n_features)
    let mut xtx = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            let mut s = F::zero();
            for k in 0..n_samples {
                s = s + (x[[k, i]] - x_means[i]) * (x[[k, j]] - x_means[j]);
            }
            xtx[[i, j]] = s;
        }
        xtx[[i, i]] = xtx[[i, i]] + alpha;
    }

    // Compute X^T y (n_features)
    let mut xty = Array1::zeros(n_features);
    for i in 0..n_features {
        let mut s = F::zero();
        for k in 0..n_samples {
            s = s + (x[[k, i]] - x_means[i]) * (y[k] - y_mean);
        }
        xty[i] = s;
    }

    // Solve xtx * beta = xty using Cholesky-like approach (simple Gaussian elimination)
    let beta = solve_linear_system(&xtx, &xty)?;

    // Compute intercept
    let mut intercept = y_mean;
    for j in 0..n_features {
        intercept = intercept - beta[j] * x_means[j];
    }

    Some((beta, intercept))
}

/// Solve A * x = b using Gaussian elimination with partial pivoting.
fn solve_linear_system<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Option<Array1<F>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return None;
    }
    if n == 0 {
        return Some(Array1::zeros(0));
    }

    // Augmented matrix
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from(1e-15).unwrap_or_else(F::min_positive_value) {
            return None; // Singular matrix
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let val = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < F::from(1e-15).unwrap_or_else(F::min_positive_value) {
            return None;
        }
        x[i] = sum / diag;
    }

    Some(x)
}

/// Predict using a Ridge model.
fn ridge_predict<F: Float>(x: &Array2<F>, coefficients: &Array1<F>, intercept: F) -> Array1<F> {
    let n_samples = x.nrows();
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let mut val = intercept;
        for j in 0..x.ncols() {
            val = val + coefficients[j] * x[[i, j]];
        }
        y[i] = val;
    }
    y
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for IterativeImputer<F> {
    type Fitted = FittedIterativeImputer<F>;
    type Error = FerroError;

    /// Fit the iterative imputer by performing round-robin Ridge regression.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::InvalidParameter`] if `max_iter` is zero.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedIterativeImputer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "IterativeImputer::fit".into(),
            });
        }
        if self.max_iter == 0 {
            return Err(FerroError::InvalidParameter {
                name: "max_iter".into(),
                reason: "max_iter must be at least 1".into(),
            });
        }

        let n_features = x.ncols();

        // Compute initial fill values
        let fill_values = match self.initial_strategy {
            InitialStrategy::Mean => column_means_nan(x),
            InitialStrategy::Median => column_medians_nan(x),
        };

        // Create mask of missing values
        let mut missing_mask = Array2::from_elem((n_samples, n_features), false);
        let mut missing_features = Vec::new();
        for j in 0..n_features {
            let mut has_missing = false;
            for i in 0..n_samples {
                if x[[i, j]].is_nan() {
                    missing_mask[[i, j]] = true;
                    has_missing = true;
                }
            }
            if has_missing {
                missing_features.push(j);
            }
        }

        // Initial imputation
        let mut imputed = initial_fill(x, &fill_values);

        // Iterative refinement
        let alpha = F::one(); // Ridge alpha
        let mut n_iter = 0usize;
        let mut feature_models: Vec<Option<FeatureModel<F>>> =
            (0..n_features).map(|_| None).collect();

        for iter_idx in 0..self.max_iter {
            n_iter = iter_idx + 1;
            let prev_imputed = imputed.clone();

            for &j in &missing_features {
                // Build predictor matrix (all features except j) and target (feature j)
                // Only use rows where feature j is NOT missing
                let predictor_cols: Vec<usize> = (0..n_features).filter(|&k| k != j).collect();
                let n_predictors = predictor_cols.len();

                // Collect non-missing rows for feature j
                let non_missing_rows: Vec<usize> =
                    (0..n_samples).filter(|&i| !missing_mask[[i, j]]).collect();

                if non_missing_rows.is_empty() || n_predictors == 0 {
                    continue;
                }

                // Build X_train and y_train
                let n_train = non_missing_rows.len();
                let mut x_train = Array2::zeros((n_train, n_predictors));
                let mut y_train = Array1::zeros(n_train);
                for (row_idx, &i) in non_missing_rows.iter().enumerate() {
                    for (col_idx, &k) in predictor_cols.iter().enumerate() {
                        x_train[[row_idx, col_idx]] = imputed[[i, k]];
                    }
                    y_train[row_idx] = imputed[[i, j]];
                }

                // Fit Ridge regression
                if let Some((coefficients, intercept)) = ridge_fit(&x_train, &y_train, alpha) {
                    // Predict for missing rows
                    let missing_rows: Vec<usize> =
                        (0..n_samples).filter(|&i| missing_mask[[i, j]]).collect();

                    if !missing_rows.is_empty() {
                        let n_missing = missing_rows.len();
                        let mut x_missing = Array2::zeros((n_missing, n_predictors));
                        for (row_idx, &i) in missing_rows.iter().enumerate() {
                            for (col_idx, &k) in predictor_cols.iter().enumerate() {
                                x_missing[[row_idx, col_idx]] = imputed[[i, k]];
                            }
                        }

                        let predictions = ridge_predict(&x_missing, &coefficients, intercept);
                        for (row_idx, &i) in missing_rows.iter().enumerate() {
                            imputed[[i, j]] = predictions[row_idx];
                        }
                    }

                    feature_models[j] = Some(FeatureModel {
                        coefficients,
                        intercept,
                    });
                }
            }

            // Check convergence
            let mut total_change = F::zero();
            let mut total_value = F::zero();
            for &j in &missing_features {
                for i in 0..n_samples {
                    if missing_mask[[i, j]] {
                        let diff = imputed[[i, j]] - prev_imputed[[i, j]];
                        total_change = total_change + diff * diff;
                        total_value = total_value + imputed[[i, j]] * imputed[[i, j]];
                    }
                }
            }

            if total_value > F::zero() {
                let relative_change = (total_change / total_value).sqrt();
                if relative_change < self.tol {
                    break;
                }
            } else if total_change < self.tol * self.tol {
                break;
            }
        }

        Ok(FittedIterativeImputer {
            initial_fill: fill_values,
            feature_models,
            missing_features,
            n_iter,
            max_iter: self.max_iter,
            tol: self.tol,
            initial_strategy: self.initial_strategy,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedIterativeImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Impute missing values in `x` using the learned feature models.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the training data.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.initial_fill.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedIterativeImputer::transform".into(),
            });
        }

        let n_samples = x.nrows();

        // Initial imputation
        let mut imputed = initial_fill(x, &self.initial_fill);

        // Create missing mask
        let mut missing_mask = Array2::from_elem((n_samples, n_features), false);
        for j in 0..n_features {
            for i in 0..n_samples {
                if x[[i, j]].is_nan() {
                    missing_mask[[i, j]] = true;
                }
            }
        }

        // Apply iterative imputation using learned models
        let alpha = F::one();
        for _iter in 0..self.max_iter {
            let prev = imputed.clone();

            for &j in &self.missing_features {
                let predictor_cols: Vec<usize> = (0..n_features).filter(|&k| k != j).collect();
                let n_predictors = predictor_cols.len();

                if n_predictors == 0 {
                    continue;
                }

                // Use the stored model if available, otherwise re-fit on non-missing data
                let model = if let Some(ref m) = self.feature_models[j] {
                    Some((m.coefficients.clone(), m.intercept))
                } else {
                    // Fallback: fit on non-missing rows of transform data
                    let non_missing_rows: Vec<usize> =
                        (0..n_samples).filter(|&i| !missing_mask[[i, j]]).collect();
                    if non_missing_rows.is_empty() {
                        None
                    } else {
                        let n_train = non_missing_rows.len();
                        let mut x_train = Array2::zeros((n_train, n_predictors));
                        let mut y_train = Array1::zeros(n_train);
                        for (row_idx, &i) in non_missing_rows.iter().enumerate() {
                            for (col_idx, &k) in predictor_cols.iter().enumerate() {
                                x_train[[row_idx, col_idx]] = imputed[[i, k]];
                            }
                            y_train[row_idx] = imputed[[i, j]];
                        }
                        ridge_fit(&x_train, &y_train, alpha)
                    }
                };

                if let Some((coefficients, intercept)) = model {
                    let missing_rows: Vec<usize> =
                        (0..n_samples).filter(|&i| missing_mask[[i, j]]).collect();
                    if !missing_rows.is_empty() {
                        let n_missing = missing_rows.len();
                        let mut x_missing = Array2::zeros((n_missing, n_predictors));
                        for (row_idx, &i) in missing_rows.iter().enumerate() {
                            for (col_idx, &k) in predictor_cols.iter().enumerate() {
                                x_missing[[row_idx, col_idx]] = imputed[[i, k]];
                            }
                        }
                        let predictions = ridge_predict(&x_missing, &coefficients, intercept);
                        for (row_idx, &i) in missing_rows.iter().enumerate() {
                            imputed[[i, j]] = predictions[row_idx];
                        }
                    }
                }
            }

            // Check convergence
            let mut total_change = F::zero();
            let mut total_value = F::zero();
            for &j in &self.missing_features {
                for i in 0..n_samples {
                    if missing_mask[[i, j]] {
                        let diff = imputed[[i, j]] - prev[[i, j]];
                        total_change = total_change + diff * diff;
                        total_value = total_value + imputed[[i, j]] * imputed[[i, j]];
                    }
                }
            }
            if total_value > F::zero() {
                let relative_change = (total_change / total_value).sqrt();
                if relative_change < self.tol {
                    break;
                }
            } else if total_change < self.tol * self.tol {
                break;
            }
        }

        Ok(imputed)
    }
}

/// Implement `Transform` on the unfitted imputer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for IterativeImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the imputer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "IterativeImputer".into(),
            reason: "imputer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for IterativeImputer<F> {
    type FitError = FerroError;

    /// Fit the imputer on `x` and return the imputed output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_iterative_imputer_basic() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All values should be non-NaN
        for v in &out {
            assert!(!v.is_nan(), "Output contains NaN");
        }
    }

    #[test]
    fn test_iterative_imputer_no_missing() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_iterative_imputer_convergence() {
        let imputer = IterativeImputer::<f64>::new(100, 1e-6, InitialStrategy::Mean);
        // Correlated features: feature 1 ≈ 2 * feature 0
        let x = array![
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, f64::NAN],
            [f64::NAN, 10.0]
        ];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Check that imputed values are reasonable
        // Feature 1 of row 3 should be close to 8.0 (2 * 4.0)
        assert!(
            (out[[3, 1]] - 8.0).abs() < 2.0,
            "Expected ~8.0, got {}",
            out[[3, 1]]
        );
        // Feature 0 of row 4 should be close to 5.0 (10.0 / 2)
        assert!(
            (out[[4, 0]] - 5.0).abs() < 2.0,
            "Expected ~5.0, got {}",
            out[[4, 0]]
        );
    }

    #[test]
    fn test_iterative_imputer_median_strategy() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Median);
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[2, 1]].is_nan());
    }

    #[test]
    fn test_iterative_imputer_fit_transform() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
        let out = imputer.fit_transform(&x).unwrap();
        for v in &out {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_iterative_imputer_zero_rows_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_iterative_imputer_zero_max_iter_error() {
        let imputer = IterativeImputer::<f64>::new(0, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0]];
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_iterative_imputer_shape_mismatch_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_iterative_imputer_unfitted_transform_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0]];
        assert!(imputer.transform(&x).is_err());
    }

    #[test]
    fn test_iterative_imputer_default() {
        let imputer = IterativeImputer::<f64>::default();
        assert_eq!(imputer.max_iter(), 10);
        assert_eq!(imputer.initial_strategy(), InitialStrategy::Mean);
    }

    #[test]
    fn test_iterative_imputer_n_iter_accessor() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
        assert!(fitted.n_iter() <= 10);
    }

    #[test]
    fn test_iterative_imputer_f32() {
        let imputer = IterativeImputer::<f32>::new(10, 1e-3, InitialStrategy::Mean);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, f32::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[1, 1]].is_nan());
    }
}
