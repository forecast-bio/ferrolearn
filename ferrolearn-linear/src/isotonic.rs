//! Isotonic (monotonic) regression.
//!
//! This module provides [`IsotonicRegression`], a non-parametric regression
//! model that fits a piecewise-constant (step) function subject to a
//! monotonicity constraint. The fitted model uses linear interpolation
//! between breakpoints for prediction.
//!
//! # Algorithm
//!
//! Uses the **Pool Adjacent Violators (PAV)** algorithm, which runs in
//! `O(n)` time.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::isotonic::{IsotonicRegression, OutOfBounds};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = IsotonicRegression::<f64>::new();
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![1.0, 3.0, 2.0, 5.0, 4.0];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! // Predictions are monotonically non-decreasing.
//! for i in 1..preds.len() {
//!     assert!(preds[i] >= preds[i - 1]);
//! }
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Out-of-bounds strategy
// ---------------------------------------------------------------------------

/// Strategy for handling predictions outside the training range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBounds {
    /// Clip predictions to the range of training values.
    Clip,
    /// Return NaN for out-of-range inputs.
    Nan,
    /// Return an error for out-of-range inputs.
    Raise,
}

// ---------------------------------------------------------------------------
// IsotonicRegression (unfitted)
// ---------------------------------------------------------------------------

/// Isotonic regression configuration.
///
/// Fits a piecewise-constant monotonic function using the Pool Adjacent
/// Violators (PAV) algorithm.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct IsotonicRegression<F> {
    /// Whether the fitted function should be increasing (true) or
    /// decreasing (false).
    pub increasing: bool,
    /// Strategy for predictions outside the training range.
    pub out_of_bounds: OutOfBounds,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float> IsotonicRegression<F> {
    /// Create a new `IsotonicRegression` with default settings.
    ///
    /// Defaults: `increasing = true`, `out_of_bounds = Clip`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            increasing: true,
            out_of_bounds: OutOfBounds::Clip,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether the fitted function should be increasing.
    #[must_use]
    pub fn with_increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set the out-of-bounds strategy.
    #[must_use]
    pub fn with_out_of_bounds(mut self, strategy: OutOfBounds) -> Self {
        self.out_of_bounds = strategy;
        self
    }
}

impl<F: Float> Default for IsotonicRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FittedIsotonicRegression
// ---------------------------------------------------------------------------

/// Fitted isotonic regression model.
///
/// Stores the breakpoints of the fitted step function and uses linear
/// interpolation between them for prediction.
#[derive(Debug, Clone)]
pub struct FittedIsotonicRegression<F> {
    /// Sorted x-values of breakpoints.
    x_thresholds: Vec<F>,
    /// Corresponding y-values (monotonic).
    y_thresholds: Vec<F>,
    /// Out-of-bounds strategy.
    out_of_bounds: OutOfBounds,
    /// Whether the function is increasing.
    increasing: bool,
}

impl<F: Float> FittedIsotonicRegression<F> {
    /// Returns whether the fitted function is increasing.
    #[must_use]
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }

    /// Predict a single value using linear interpolation.
    fn predict_single(&self, x: F) -> Result<F, FerroError> {
        if self.x_thresholds.is_empty() {
            return Err(FerroError::NumericalInstability {
                message: "isotonic model has no breakpoints".into(),
            });
        }

        let x_min = self.x_thresholds[0];
        let x_max = *self.x_thresholds.last().unwrap();

        if x < x_min {
            return match self.out_of_bounds {
                OutOfBounds::Clip => Ok(self.y_thresholds[0]),
                OutOfBounds::Nan => Ok(F::nan()),
                OutOfBounds::Raise => Err(FerroError::InvalidParameter {
                    name: "x".into(),
                    reason: "value is below training range".into(),
                }),
            };
        }

        if x > x_max {
            return match self.out_of_bounds {
                OutOfBounds::Clip => Ok(*self.y_thresholds.last().unwrap()),
                OutOfBounds::Nan => Ok(F::nan()),
                OutOfBounds::Raise => Err(FerroError::InvalidParameter {
                    name: "x".into(),
                    reason: "value is above training range".into(),
                }),
            };
        }

        // Binary search for the interval containing x.
        let n = self.x_thresholds.len();

        // Handle exact match at the last point.
        if x == x_max {
            return Ok(*self.y_thresholds.last().unwrap());
        }

        // Find the interval [x_thresholds[i], x_thresholds[i+1]) containing x.
        let mut lo = 0;
        let mut hi = n - 1;
        while lo < hi - 1 {
            let mid = usize::midpoint(lo, hi);
            if self.x_thresholds[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let x0 = self.x_thresholds[lo];
        let x1 = self.x_thresholds[hi];
        let y0 = self.y_thresholds[lo];
        let y1 = self.y_thresholds[hi];

        if (x1 - x0).abs() < F::epsilon() {
            return Ok(y0);
        }

        // Linear interpolation.
        let t = (x - x0) / (x1 - x0);
        Ok(y0 + t * (y1 - y0))
    }
}

// ---------------------------------------------------------------------------
// Pool Adjacent Violators (PAV) algorithm
// ---------------------------------------------------------------------------

/// Run the PAV algorithm to produce a monotonically non-decreasing
/// sequence of (x, y) breakpoints.
fn pav_increasing<F: Float>(xs: &[F], ys: &[F]) -> (Vec<F>, Vec<F>) {
    // Sort by x.
    let n = xs.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());

    let sorted_y: Vec<F> = indices.iter().map(|&i| ys[i]).collect();
    let sorted_x: Vec<F> = indices.iter().map(|&i| xs[i]).collect();

    // PAV: merge adjacent blocks that violate monotonicity.
    // Each block is (sum, count, first_x, last_x).
    struct Block<F> {
        sum: F,
        count: usize,
        first_idx: usize,
        last_idx: usize,
    }

    let mut blocks: Vec<Block<F>> = Vec::with_capacity(n);

    for (i, &y_val) in sorted_y.iter().enumerate().take(n) {
        blocks.push(Block {
            sum: y_val,
            count: 1,
            first_idx: i,
            last_idx: i,
        });

        // Merge with previous blocks as needed.
        while blocks.len() > 1 {
            let len = blocks.len();
            let prev_mean = blocks[len - 2].sum / F::from(blocks[len - 2].count).unwrap();
            let curr_mean = blocks[len - 1].sum / F::from(blocks[len - 1].count).unwrap();

            if prev_mean > curr_mean {
                // Merge.
                let last = blocks.pop().unwrap();
                let prev = blocks.last_mut().unwrap();
                prev.sum = prev.sum + last.sum;
                prev.count += last.count;
                prev.last_idx = last.last_idx;
            } else {
                break;
            }
        }
    }

    // Extract breakpoints: for each block, use the first and last x,
    // and the block mean as y.
    let mut result_x = Vec::new();
    let mut result_y = Vec::new();

    for block in &blocks {
        let mean = block.sum / F::from(block.count).unwrap();
        let bx0 = sorted_x[block.first_idx];
        let bx1 = sorted_x[block.last_idx];

        if result_x.is_empty() || *result_x.last().unwrap() != bx0 {
            result_x.push(bx0);
            result_y.push(mean);
        }
        if bx0 != bx1 {
            result_x.push(bx1);
            result_y.push(mean);
        }
    }

    (result_x, result_y)
}

// ---------------------------------------------------------------------------
// Fit and Predict
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<F>> for IsotonicRegression<F> {
    type Fitted = FittedIsotonicRegression<F>;
    type Error = FerroError;

    /// Fit the isotonic regression model using PAV.
    ///
    /// The input `x` must have exactly one column (univariate regression).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts or if `x` does not have exactly one column.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer than
    /// 2 samples.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedIsotonicRegression<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_features != 1 {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, 1],
                actual: vec![n_samples, n_features],
                context: "IsotonicRegression requires exactly 1 feature".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "IsotonicRegression requires at least 2 samples".into(),
            });
        }

        // Extract the single feature column.
        let xs: Vec<F> = x.column(0).to_vec();
        let ys: Vec<F> = y.to_vec();

        let (mut result_x, mut result_y) = if self.increasing {
            pav_increasing(&xs, &ys)
        } else {
            // For decreasing: negate y, run increasing PAV, negate result.
            let neg_ys: Vec<F> = ys.iter().map(|&v| -v).collect();
            let (rx, ry) = pav_increasing(&xs, &neg_ys);
            let ry_neg: Vec<F> = ry.iter().map(|&v| -v).collect();
            (rx, ry_neg)
        };

        // Ensure at least 2 breakpoints.
        if result_x.len() < 2 {
            // All same x value: duplicate.
            if result_x.len() == 1 {
                result_x.push(result_x[0]);
                result_y.push(result_y[0]);
            } else {
                return Err(FerroError::NumericalInstability {
                    message: "PAV produced no breakpoints".into(),
                });
            }
        }

        Ok(FittedIsotonicRegression {
            x_thresholds: result_x,
            y_thresholds: result_y,
            out_of_bounds: self.out_of_bounds,
            increasing: self.increasing,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedIsotonicRegression<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Uses linear interpolation between breakpoints.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` does not have exactly
    /// one column.
    /// Returns [`FerroError::InvalidParameter`] if `out_of_bounds = Raise`
    /// and a value is outside the training range.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_features != 1 {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples, 1],
                actual: vec![n_samples, n_features],
                context: "IsotonicRegression requires exactly 1 feature".into(),
            });
        }

        let mut result = Array1::<F>::zeros(n_samples);
        for i in 0..n_samples {
            result[i] = self.predict_single(x[[i, 0]])?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_monotonicity_increasing() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 4.0, 2.0, 5.0, 3.0, 7.0];

        let model = IsotonicRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check monotonicity: each prediction should be >= the previous.
        for i in 1..preds.len() {
            assert!(
                preds[i] >= preds[i - 1] - 1e-10,
                "Monotonicity violated at index {i}: {} < {}",
                preds[i],
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_monotonicity_decreasing() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0];

        let model = IsotonicRegression::<f64>::new().with_increasing(false);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check monotonicity: each prediction should be <= the previous.
        for i in 1..preds.len() {
            assert!(
                preds[i] <= preds[i - 1] + 1e-10,
                "Decreasing monotonicity violated at index {i}: {} > {}",
                preds[i],
                preds[i - 1]
            );
        }
    }

    #[test]
    fn test_already_monotonic() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = IsotonicRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_relative_eq!(preds[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_interpolation() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 3.0, 5.0]).unwrap();
        let y = array![1.0, 3.0, 5.0];

        let model = IsotonicRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Predict at intermediate points.
        let x_new = Array2::from_shape_vec((3, 1), vec![2.0, 3.0, 4.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();

        // Linear interpolation: at x=2, y should be 2.0; at x=4, y should be 4.0.
        assert_relative_eq!(preds[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(preds[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(preds[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_bounds_clip() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = IsotonicRegression::<f64>::new().with_out_of_bounds(OutOfBounds::Clip);
        let fitted = model.fit(&x, &y).unwrap();

        let x_oob = Array2::from_shape_vec((2, 1), vec![0.0, 4.0]).unwrap();
        let preds = fitted.predict(&x_oob).unwrap();

        // Should clip to the boundary values.
        assert_relative_eq!(preds[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(preds[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_out_of_bounds_nan() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = IsotonicRegression::<f64>::new().with_out_of_bounds(OutOfBounds::Nan);
        let fitted = model.fit(&x, &y).unwrap();

        let x_oob = Array2::from_shape_vec((2, 1), vec![0.0, 4.0]).unwrap();
        let preds = fitted.predict(&x_oob).unwrap();

        assert!(preds[0].is_nan());
        assert!(preds[1].is_nan());
    }

    #[test]
    fn test_out_of_bounds_raise() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = IsotonicRegression::<f64>::new().with_out_of_bounds(OutOfBounds::Raise);
        let fitted = model.fit(&x, &y).unwrap();

        let x_below = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        assert!(fitted.predict(&x_below).is_err());

        let x_above = Array2::from_shape_vec((1, 1), vec![4.0]).unwrap();
        assert!(fitted.predict(&x_above).is_err());
    }

    #[test]
    fn test_shape_mismatch_features() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let model = IsotonicRegression::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_shape_mismatch_samples() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = IsotonicRegression::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];

        let model = IsotonicRegression::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_pav_all_equal() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![3.0, 3.0, 3.0, 3.0];

        let model = IsotonicRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for i in 0..4 {
            assert_relative_eq!(preds[i], 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_unsorted_x() {
        // PAV should handle unsorted x by sorting internally.
        let x = Array2::from_shape_vec((4, 1), vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        let y = array![3.0, 1.0, 4.0, 2.0];

        let model = IsotonicRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Predict at sorted x values.
        let x_sorted = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let preds = fitted.predict(&x_sorted).unwrap();

        for i in 1..preds.len() {
            assert!(preds[i] >= preds[i - 1] - 1e-10);
        }
    }
}
