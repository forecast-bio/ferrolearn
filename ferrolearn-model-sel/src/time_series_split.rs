//! Time-series cross-validation splitter.
//!
//! [`TimeSeriesSplit`] generates folds where the test set always follows the
//! training set chronologically, making it suitable for evaluating models on
//! time-ordered data without look-ahead bias.
//!
//! # Example
//!
//! ```rust
//! use ferrolearn_model_sel::TimeSeriesSplit;
//!
//! let tss = TimeSeriesSplit::new(5);
//! let folds = tss.split(20).unwrap();
//! assert_eq!(folds.len(), 5);
//! for (train, test) in &folds {
//!     // Test always comes strictly after training data.
//!     assert!(train.iter().all(|&tr| test.iter().all(|&te| tr < te)));
//! }
//! ```

use ferrolearn_core::{FerroError, FerroResult};

use crate::cross_validation::{CrossValidator, FoldSplits};

// ---------------------------------------------------------------------------
// TimeSeriesSplit
// ---------------------------------------------------------------------------

/// Cross-validation splitter for time-series data.
///
/// Generates `n_splits` train/test pairs in which the test window always
/// follows the training window, preserving the temporal ordering of samples.
/// No shuffling is performed.
///
/// For split `i` (0-indexed):
/// - The test window has size `test_size` (defaults to `n_samples / (n_splits + 1)`).
/// - A `gap` of samples is skipped between the end of training and the start of
///   the test window.
/// - The training window ends just before the gap; optionally capped at
///   `max_train_size` most-recent samples.
///
/// # Parameters
///
/// - `n_splits` — number of splits (default 5).
/// - `max_train_size` — if `Some(n)`, limit training set to the `n` most recent
///   samples before the gap.
/// - `test_size` — number of samples in each test window.  If `None`, the
///   default `n_samples / (n_splits + 1)` is used.
/// - `gap` — number of samples to skip between training and test windows
///   (default 0).
///
/// # Example
///
/// ```rust
/// use ferrolearn_model_sel::TimeSeriesSplit;
///
/// let tss = TimeSeriesSplit::new(3)
///     .test_size(Some(4))
///     .gap(1);
/// let folds = tss.split(30).unwrap();
/// assert_eq!(folds.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    /// Number of splits to generate.
    n_splits: usize,
    /// Maximum number of samples in the training set.
    max_train_size: Option<usize>,
    /// Fixed test window size; `None` means auto-compute.
    test_size: Option<usize>,
    /// Gap between end of training and start of test.
    gap: usize,
}

impl TimeSeriesSplit {
    /// Create a new [`TimeSeriesSplit`] with `n_splits` folds.
    ///
    /// All other options default to their standard values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::TimeSeriesSplit;
    ///
    /// let tss = TimeSeriesSplit::new(5);
    /// ```
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            max_train_size: None,
            test_size: None,
            gap: 0,
        }
    }

    /// Set the maximum training size.
    ///
    /// If `Some(n)`, each training fold is limited to the `n` most-recent
    /// samples before the gap.
    #[must_use]
    pub fn max_train_size(mut self, max_train_size: Option<usize>) -> Self {
        self.max_train_size = max_train_size;
        self
    }

    /// Set the test window size.
    ///
    /// If `None`, the default `floor(n_samples / (n_splits + 1))` is used.
    #[must_use]
    pub fn test_size(mut self, test_size: Option<usize>) -> Self {
        self.test_size = test_size;
        self
    }

    /// Set the gap (number of samples skipped) between training and test windows.
    #[must_use]
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate `(train_indices, test_indices)` pairs for each fold.
    ///
    /// The test indices for fold `i` are always strictly greater than the
    /// training indices for that fold (chronological ordering is preserved).
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// - [`FerroError::InsufficientSamples`] if there are not enough samples to
    ///   produce the requested splits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::TimeSeriesSplit;
    ///
    /// let folds = TimeSeriesSplit::new(3).split(12).unwrap();
    /// assert_eq!(folds.len(), 3);
    /// ```
    pub fn split(&self, n_samples: usize) -> FerroResult<FoldSplits> {
        self.split_impl(n_samples)
    }

    // ------------------------------------------------------------------
    // Core splitting logic
    // ------------------------------------------------------------------

    fn split_impl(&self, n_samples: usize) -> FerroResult<FoldSplits> {
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }

        // Determine the test window size.
        let test_sz = match self.test_size {
            Some(ts) => {
                if ts == 0 {
                    return Err(FerroError::InvalidParameter {
                        name: "test_size".into(),
                        reason: "must be at least 1".into(),
                    });
                }
                ts
            }
            None => {
                // sklearn default: floor(n_samples / (n_splits + 1))
                let default_ts = n_samples / (self.n_splits + 1);
                if default_ts == 0 {
                    return Err(FerroError::InsufficientSamples {
                        required: self.n_splits + 1,
                        actual: n_samples,
                        context: format!(
                            "TimeSeriesSplit with n_splits={}: not enough samples to form a \
                             non-empty test window (floor({n_samples} / {}) == 0)",
                            self.n_splits,
                            self.n_splits + 1,
                        ),
                    });
                }
                default_ts
            }
        };

        // The last split's test window ends at index n_samples - 1.
        // Split i (0-indexed, last = n_splits-1):
        //   test_end   = n_samples - (n_splits - 1 - i) * test_sz
        //   test_start = test_end - test_sz
        //   train_end  = test_start - gap
        //   train_start= max(0, train_end - max_train_size) if max_train_size set
        //              = 0 otherwise

        let mut folds = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            // How many test windows come after this one (the offset from the end).
            let windows_after = self.n_splits - 1 - i;
            let test_end = n_samples
                .checked_sub(windows_after * test_sz)
                .ok_or_else(|| FerroError::InsufficientSamples {
                    required: (self.n_splits + 1) * test_sz,
                    actual: n_samples,
                    context: format!(
                        "TimeSeriesSplit with n_splits={}, test_size={test_sz}: \
                         not enough samples",
                        self.n_splits
                    ),
                })?;
            let test_start =
                test_end
                    .checked_sub(test_sz)
                    .ok_or_else(|| FerroError::InsufficientSamples {
                        required: (self.n_splits + 1) * test_sz,
                        actual: n_samples,
                        context: format!(
                            "TimeSeriesSplit with n_splits={}, test_size={test_sz}: \
                         test window underflows",
                            self.n_splits
                        ),
                    })?;

            // Training window ends before the gap.
            let train_end = test_start.checked_sub(self.gap).ok_or_else(|| {
                FerroError::InsufficientSamples {
                    required: self.gap + test_sz,
                    actual: test_start,
                    context: format!(
                        "TimeSeriesSplit with gap={}: gap is too large for split {i}",
                        self.gap
                    ),
                }
            })?;

            if train_end == 0 {
                return Err(FerroError::InsufficientSamples {
                    required: self.gap + test_sz + 1,
                    actual: test_start,
                    context: format!(
                        "TimeSeriesSplit: no training samples available for split {i}"
                    ),
                });
            }

            let train_start = match self.max_train_size {
                Some(mts) => train_end.saturating_sub(mts),
                None => 0,
            };

            let train: Vec<usize> = (train_start..train_end).collect();
            let test: Vec<usize> = (test_start..test_end).collect();
            folds.push((train, test));
        }

        Ok(folds)
    }
}

impl CrossValidator for TimeSeriesSplit {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        self.split_impl(n_samples)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{
        FittedPipelineEstimator as FittedEstTrait, Pipeline, PipelineEstimator,
    };
    use ndarray::{Array1, Array2};

    use crate::cross_val_score;

    // -----------------------------------------------------------------------
    // Helper: a trivial mean-predictor for cross_val_score tests
    // -----------------------------------------------------------------------

    struct MeanEstimator;
    struct FittedMean {
        mean: f64,
    }

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedEstTrait<f64>>, ferrolearn_core::FerroError> {
            Ok(Box::new(FittedMean {
                mean: y.mean().unwrap_or(0.0),
            }))
        }
    }

    impl FittedEstTrait<f64> for FittedMean {
        fn predict_pipeline(
            &self,
            x: &Array2<f64>,
        ) -> Result<Array1<f64>, ferrolearn_core::FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, ferrolearn_core::FerroError> {
        let d = y_true - y_pred;
        Ok(d.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    // -----------------------------------------------------------------------
    // Basic structure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tss_correct_number_of_folds() {
        let folds = TimeSeriesSplit::new(5).split(25).unwrap();
        assert_eq!(folds.len(), 5);
    }

    #[test]
    fn test_tss_default_test_size_fills_last_window() {
        // With n_samples=20, n_splits=4: test_sz = floor(20/5) = 4
        // Last split should have 4 test samples ending at index 20.
        let folds = TimeSeriesSplit::new(4).split(20).unwrap();
        let (_, last_test) = &folds[3];
        assert_eq!(last_test.len(), 4);
        assert_eq!(*last_test.last().unwrap(), 19);
    }

    #[test]
    fn test_tss_test_indices_after_train_indices() {
        let folds = TimeSeriesSplit::new(4).split(20).unwrap();
        for (train, test) in &folds {
            let max_train = train.iter().copied().max().unwrap();
            let min_test = test.iter().copied().min().unwrap();
            assert!(
                max_train < min_test,
                "max train idx {max_train} must be < min test idx {min_test}"
            );
        }
    }

    #[test]
    fn test_tss_training_grows_each_split() {
        // Without max_train_size, each successive training fold should be larger.
        let folds = TimeSeriesSplit::new(5).split(30).unwrap();
        let train_sizes: Vec<usize> = folds.iter().map(|(tr, _)| tr.len()).collect();
        for window in train_sizes.windows(2) {
            assert!(
                window[0] < window[1],
                "training size should grow: {:?}",
                train_sizes
            );
        }
    }

    #[test]
    fn test_tss_fixed_test_size() {
        let folds = TimeSeriesSplit::new(4)
            .test_size(Some(3))
            .split(20)
            .unwrap();
        for (_, test) in &folds {
            assert_eq!(test.len(), 3, "every test fold should have 3 samples");
        }
    }

    #[test]
    fn test_tss_fixed_test_size_non_overlapping_tests() {
        // Test windows for adjacent folds should not overlap.
        let folds = TimeSeriesSplit::new(4)
            .test_size(Some(3))
            .split(24)
            .unwrap();
        for i in 1..folds.len() {
            let prev_test_end = folds[i - 1].1.iter().copied().max().unwrap();
            let curr_test_start = folds[i].1.iter().copied().min().unwrap();
            assert!(
                prev_test_end < curr_test_start,
                "test windows must not overlap between split {} and {}",
                i - 1,
                i
            );
        }
    }

    #[test]
    fn test_tss_gap_separates_train_and_test() {
        let gap = 2_usize;
        let folds = TimeSeriesSplit::new(3)
            .test_size(Some(3))
            .gap(gap)
            .split(30)
            .unwrap();
        for (i, (train, test)) in folds.iter().enumerate() {
            let max_train = train.iter().copied().max().unwrap();
            let min_test = test.iter().copied().min().unwrap();
            // There should be exactly `gap` indices between them.
            assert_eq!(
                min_test - max_train - 1,
                gap,
                "split {i}: gap should be {gap} but got {}",
                min_test - max_train - 1
            );
        }
    }

    #[test]
    fn test_tss_max_train_size_limits_training() {
        let max_train = 5_usize;
        let folds = TimeSeriesSplit::new(3)
            .test_size(Some(3))
            .max_train_size(Some(max_train))
            .split(30)
            .unwrap();
        for (train, _) in &folds {
            assert!(
                train.len() <= max_train,
                "training size {} should be <= {max_train}",
                train.len()
            );
        }
    }

    #[test]
    fn test_tss_max_train_size_uses_most_recent() {
        // The most recent max_train_size indices before the gap should be used.
        let folds = TimeSeriesSplit::new(2)
            .test_size(Some(4))
            .max_train_size(Some(3))
            .gap(0)
            .split(20)
            .unwrap();
        for (train, test) in &folds {
            // The last training index should be immediately before the test start.
            let max_train_idx = train.iter().copied().max().unwrap();
            let min_test_idx = test.iter().copied().min().unwrap();
            assert_eq!(max_train_idx + 1, min_test_idx);
            assert_eq!(train.len(), 3);
        }
    }

    #[test]
    fn test_tss_invalid_n_splits_less_than_2() {
        assert!(TimeSeriesSplit::new(1).split(20).is_err());
    }

    #[test]
    fn test_tss_invalid_test_size_zero() {
        assert!(
            TimeSeriesSplit::new(3)
                .test_size(Some(0))
                .split(20)
                .is_err()
        );
    }

    #[test]
    fn test_tss_insufficient_samples() {
        // Asking for 5 splits with too few samples should error.
        assert!(TimeSeriesSplit::new(10).split(5).is_err());
    }

    #[test]
    fn test_tss_train_indices_are_sorted() {
        let folds = TimeSeriesSplit::new(4).split(20).unwrap();
        for (train, _) in &folds {
            let sorted: Vec<usize> = {
                let mut v = train.clone();
                v.sort_unstable();
                v
            };
            assert_eq!(train, &sorted, "train indices should be in ascending order");
        }
    }

    #[test]
    fn test_tss_test_indices_are_sorted() {
        let folds = TimeSeriesSplit::new(4).split(20).unwrap();
        for (_, test) in &folds {
            let sorted: Vec<usize> = {
                let mut v = test.clone();
                v.sort_unstable();
                v
            };
            assert_eq!(test, &sorted, "test indices should be in ascending order");
        }
    }

    #[test]
    fn test_tss_fold_indices_matches_split() {
        let tss = TimeSeriesSplit::new(4);
        let via_split = tss.split(24).unwrap();
        let via_trait = tss.fold_indices(24).unwrap();
        assert_eq!(via_split, via_trait);
    }

    #[test]
    fn test_tss_integrates_with_cross_val_score() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        let tss = TimeSeriesSplit::new(5);
        let scores = cross_val_score(&pipeline, &x, &y, &tss, mse).unwrap();
        assert_eq!(scores.len(), 5);
        for &s in scores.iter() {
            assert!(s.abs() < 1e-10, "expected 0 MSE, got {s}");
        }
    }

    #[test]
    fn test_tss_large_gap_error() {
        // gap >= n_samples leaves no room for training.
        assert!(
            TimeSeriesSplit::new(2)
                .test_size(Some(2))
                .gap(100)
                .split(10)
                .is_err()
        );
    }

    #[test]
    fn test_tss_n_splits_2_minimal() {
        let folds = TimeSeriesSplit::new(2).test_size(Some(2)).split(6).unwrap();
        assert_eq!(folds.len(), 2);
        // Split 0: test [2,3], train [0,1]
        // Split 1: test [4,5], train [0,1,2,3]
        let (train0, test0) = &folds[0];
        let (train1, test1) = &folds[1];
        assert_eq!(test0, &[2, 3]);
        assert_eq!(test1, &[4, 5]);
        assert_eq!(train0, &[0, 1]);
        assert_eq!(train1, &[0, 1, 2, 3]);
    }
}
