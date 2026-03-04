//! Probability calibration of classifiers using cross-validation.
//!
//! [`CalibratedClassifierCV`] wraps a base classifier (provided as a closure)
//! and calibrates the raw decision scores it produces so they can be
//! interpreted as well-calibrated probabilities.
//!
//! Two calibration methods are available:
//!
//! - **[`CalibrationMethod::Sigmoid`]** — Platt scaling, which fits a sigmoid
//!   `P(y=1|f) = 1 / (1 + exp(A*f + B))` by minimising the negative
//!   log-likelihood via Newton's method.
//! - **[`CalibrationMethod::Isotonic`]** — isotonic regression via the
//!   pool-adjacent-violators (PAV) algorithm, producing a non-decreasing step
//!   function that maps scores to probabilities.
//!
//! During [`CalibratedClassifierCV::fit`] the base model is evaluated in a
//! K-fold cross-validation loop. For each fold the base model is trained on
//! the training partition and raw scores are collected on the held-out
//! partition. A calibration mapping is then fitted on the aggregated
//! out-of-fold scores and true labels.
//!
//! The returned [`FittedCalibratedClassifierCV`] re-fits the base model on
//! the **full** training set and applies the learned calibration mapping at
//! prediction time.

use ferrolearn_core::{FerroError, Predict};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// FitFn type alias
// ---------------------------------------------------------------------------

/// A boxed closure that trains a base classifier and returns a predict
/// function.
///
/// The closure receives `(X, y)` and returns a boxed prediction function
/// that maps a feature matrix to raw decision scores.
pub type FitFn =
    Box<dyn Fn(&Array2<f64>, &Array1<usize>) -> Result<PredictFn, FerroError> + Send + Sync>;

/// A boxed prediction function returned by [`FitFn`].
///
/// Maps a feature matrix to raw decision scores (one per sample).
pub type PredictFn = Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>>;

// ---------------------------------------------------------------------------
// CalibrationMethod
// ---------------------------------------------------------------------------

/// The calibration strategy used by [`CalibratedClassifierCV`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Platt scaling — fits a sigmoid `1 / (1 + exp(A*f + B))` to the
    /// raw scores using Newton's method.
    Sigmoid,
    /// Isotonic regression — fits a non-decreasing step function via the
    /// pool-adjacent-violators algorithm.
    Isotonic,
}

// ---------------------------------------------------------------------------
// CalibratedClassifierCV
// ---------------------------------------------------------------------------

/// Probability calibration of a classifier using cross-validation.
///
/// Wraps a base classifier (expressed as a [`FitFn`] closure) and calibrates
/// the raw decision scores so that they approximate well-calibrated
/// probabilities.
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::calibration::{CalibratedClassifierCV, CalibrationMethod};
/// use ndarray::{Array1, Array2};
/// use ferrolearn_core::Predict;
///
/// // Dummy base classifier that returns the mean of column 0 as a score.
/// let fit_fn = Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
///     Ok(Box::new(|x: &Array2<f64>| {
///         Ok(x.column(0).to_owned())
///     }) as Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, ferrolearn_core::FerroError>>)
/// });
///
/// let cal = CalibratedClassifierCV::new(fit_fn, CalibrationMethod::Sigmoid, 5);
/// ```
pub struct CalibratedClassifierCV {
    /// Closure that trains the base model and returns a predict function.
    fit_fn: FitFn,
    /// Calibration strategy.
    method: CalibrationMethod,
    /// Number of cross-validation folds.
    cv: usize,
}

impl CalibratedClassifierCV {
    /// Create a new [`CalibratedClassifierCV`].
    ///
    /// # Parameters
    ///
    /// - `fit_fn` — a closure that fits the base classifier and returns a
    ///   predict function producing raw scores.
    /// - `method` — the calibration strategy ([`Sigmoid`](CalibrationMethod::Sigmoid)
    ///   or [`Isotonic`](CalibrationMethod::Isotonic)).
    /// - `cv` — number of cross-validation folds (must be >= 2).
    pub fn new(fit_fn: FitFn, method: CalibrationMethod, cv: usize) -> Self {
        Self { fit_fn, method, cv }
    }

    /// Fit the calibrated classifier.
    ///
    /// 1. Performs K-fold cross-validation: for each fold, fits the base model
    ///    on the training partition and collects raw scores on the validation
    ///    partition.
    /// 2. Fits a calibration mapping (sigmoid or isotonic) on the aggregated
    ///    out-of-fold scores and true labels.
    /// 3. Re-fits the base model on the entire training set.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if `cv < 2`, if there are fewer samples than
    /// folds, or if the base model fails to fit/predict.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedCalibratedClassifierCV, FerroError> {
        let n_samples = x.nrows();

        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: format!("must be >= 2, got {}", self.cv),
            });
        }
        if n_samples < self.cv {
            return Err(FerroError::InsufficientSamples {
                required: self.cv,
                actual: n_samples,
                context: "CalibratedClassifierCV: fewer samples than folds".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "CalibratedClassifierCV::fit: y length must equal x rows".into(),
            });
        }

        // Generate K-fold indices.
        let folds = kfold_indices(n_samples, self.cv);

        // Collect out-of-fold scores and labels.
        let mut all_scores: Vec<f64> = Vec::with_capacity(n_samples);
        let mut all_labels: Vec<usize> = Vec::with_capacity(n_samples);
        // We need to keep track of ordering to reconstruct, but simpler to
        // just collect (score, label) pairs in fold order.

        for (train_idx, val_idx) in &folds {
            let x_train = select_rows(x, train_idx);
            let y_train = select_elements(y, train_idx);
            let x_val = select_rows(x, val_idx);

            let predict_fn = (self.fit_fn)(&x_train, &y_train)?;
            let val_scores = predict_fn(&x_val)?;

            for (i, &idx) in val_idx.iter().enumerate() {
                all_scores.push(val_scores[i]);
                all_labels.push(y[idx]);
            }
        }

        let scores_arr = Array1::from_vec(all_scores);
        let labels_arr = Array1::from_vec(all_labels);

        // Fit calibration mapping.
        let calibrator = match self.method {
            CalibrationMethod::Sigmoid => {
                let (a, b) = fit_sigmoid(&scores_arr, &labels_arr)?;
                Calibrator::Sigmoid { a, b }
            }
            CalibrationMethod::Isotonic => {
                let mapping = fit_isotonic(&scores_arr, &labels_arr)?;
                Calibrator::Isotonic { mapping }
            }
        };

        // Re-fit base model on the full dataset.
        let predict_fn = (self.fit_fn)(x, y)?;

        Ok(FittedCalibratedClassifierCV {
            predict_fn,
            calibrator,
        })
    }
}

// ---------------------------------------------------------------------------
// FittedCalibratedClassifierCV
// ---------------------------------------------------------------------------

/// A fitted calibrated classifier that produces calibrated probabilities.
///
/// Obtained by calling [`CalibratedClassifierCV::fit`].
pub struct FittedCalibratedClassifierCV {
    /// The base model predict function (trained on full data).
    predict_fn: PredictFn,
    /// The learned calibration mapping.
    calibrator: Calibrator,
}

impl Predict<Array2<f64>> for FittedCalibratedClassifierCV {
    type Output = Array1<f64>;
    type Error = FerroError;

    /// Predict calibrated probabilities for the given feature matrix.
    ///
    /// First obtains raw scores from the base model, then applies the
    /// calibration mapping to produce probabilities in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if the base model prediction fails.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let raw_scores = (self.predict_fn)(x)?;
        Ok(self.calibrator.transform(&raw_scores))
    }
}

// ---------------------------------------------------------------------------
// Internal calibration types
// ---------------------------------------------------------------------------

/// Internal enum holding the learned calibration mapping.
enum Calibrator {
    /// Platt sigmoid: `P = 1 / (1 + exp(a*f + b))`.
    Sigmoid { a: f64, b: f64 },
    /// Isotonic regression: sorted `(score, probability)` pairs.
    Isotonic { mapping: Vec<(f64, f64)> },
}

impl Calibrator {
    /// Apply the calibration mapping to raw scores.
    fn transform(&self, scores: &Array1<f64>) -> Array1<f64> {
        match self {
            Calibrator::Sigmoid { a, b } => scores.mapv(|f| sigmoid_fn(*a * f + *b)),
            Calibrator::Isotonic { mapping } => scores.mapv(|f| isotonic_lookup(mapping, f)),
        }
    }
}

// ---------------------------------------------------------------------------
// Platt scaling (sigmoid calibration)
// ---------------------------------------------------------------------------

/// Sigmoid function `1 / (1 + exp(-x))`.
fn sigmoid_fn(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

/// Fit Platt scaling parameters A, B by minimising the negative
/// log-likelihood via Newton's method.
///
/// Target probabilities are set following Platt (1999):
///   t_i = (y_i * N_+ + 1) / (N_+ + 2)  if y_i = 1
///   t_i = 1 / (N_- + 2)                 if y_i = 0
fn fit_sigmoid(scores: &Array1<f64>, labels: &Array1<usize>) -> Result<(f64, f64), FerroError> {
    let n = scores.len();
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "Platt scaling: no samples".into(),
        });
    }

    // Compute target probabilities.
    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&l| l != 1).count() as f64;

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let targets: Vec<f64> = labels
        .iter()
        .map(|&l| if l == 1 { t_pos } else { t_neg })
        .collect();

    // Newton's method to minimise NLL.
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let max_iter = 100;
    let tol = 1e-8;

    for _ in 0..max_iter {
        // Compute gradient and Hessian.
        let mut g_a = 0.0_f64;
        let mut g_b = 0.0_f64;
        let mut h_aa = 0.0_f64;
        let mut h_ab = 0.0_f64;
        let mut h_bb = 0.0_f64;

        for i in 0..n {
            let f = scores[i];
            let p = sigmoid_fn(a * f + b);
            let t = targets[i];
            let d = p - t;

            g_a += d * f;
            g_b += d;

            let w = p * (1.0 - p) + 1e-12; // small epsilon for stability
            h_aa += w * f * f;
            h_ab += w * f;
            h_bb += w;
        }

        // Solve 2x2 system: H * delta = -g
        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-15 {
            break;
        }

        let delta_a = -(h_bb * g_a - h_ab * g_b) / det;
        let delta_b = -(h_aa * g_b - h_ab * g_a) / det;

        a += delta_a;
        b += delta_b;

        if delta_a.abs() < tol && delta_b.abs() < tol {
            break;
        }
    }

    Ok((a, b))
}

// ---------------------------------------------------------------------------
// Isotonic regression (PAV algorithm)
// ---------------------------------------------------------------------------

/// Fit isotonic regression using the pool-adjacent-violators algorithm.
///
/// Returns a sorted list of `(score, probability)` breakpoints.
fn fit_isotonic(
    scores: &Array1<f64>,
    labels: &Array1<usize>,
) -> Result<Vec<(f64, f64)>, FerroError> {
    let n = scores.len();
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "Isotonic regression: no samples".into(),
        });
    }

    // Sort by score.
    let mut indexed: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, if l == 1 { 1.0 } else { 0.0 }))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // PAV algorithm — maintain blocks of (sum, count, leftmost_score, rightmost_score).
    let mut blocks: Vec<(f64, usize, f64, f64)> = Vec::with_capacity(n);

    for &(score, label) in &indexed {
        blocks.push((label, 1, score, score));

        // Merge while the last block's mean exceeds the previous block's mean.
        while blocks.len() > 1 {
            let len = blocks.len();
            let (sum1, cnt1, _, _) = blocks[len - 2];
            let (sum2, cnt2, _, _) = blocks[len - 1];

            if sum1 / cnt1 as f64 > sum2 / cnt2 as f64 {
                let (_, _, lo, _) = blocks[len - 2];
                let (_, _, _, hi) = blocks[len - 1];
                blocks[len - 2] = (sum1 + sum2, cnt1 + cnt2, lo, hi);
                blocks.pop();
            } else {
                break;
            }
        }
    }

    // Build the breakpoint mapping.
    let mut mapping: Vec<(f64, f64)> = Vec::with_capacity(blocks.len());
    for &(sum, cnt, lo, hi) in &blocks {
        let prob = sum / cnt as f64;
        // Use the midpoint of the score range for this block.
        let score = (lo + hi) / 2.0;
        mapping.push((score, prob));
    }

    Ok(mapping)
}

/// Look up the calibrated probability for a raw score using an isotonic
/// mapping via piecewise-linear interpolation.
fn isotonic_lookup(mapping: &[(f64, f64)], score: f64) -> f64 {
    if mapping.is_empty() {
        return 0.5;
    }
    if mapping.len() == 1 {
        return mapping[0].1;
    }

    // Clamp to endpoints.
    if score <= mapping[0].0 {
        return mapping[0].1;
    }
    if score >= mapping[mapping.len() - 1].0 {
        return mapping[mapping.len() - 1].1;
    }

    // Binary search for the right interval.
    let pos = mapping.partition_point(|&(s, _)| s < score);
    if pos == 0 {
        return mapping[0].1;
    }
    if pos >= mapping.len() {
        return mapping[mapping.len() - 1].1;
    }

    let (s0, p0) = mapping[pos - 1];
    let (s1, p1) = mapping[pos];
    let denom = s1 - s0;
    if denom.abs() < 1e-15 {
        return (p0 + p1) / 2.0;
    }

    // Linear interpolation.
    p0 + (score - s0) * (p1 - p0) / denom
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate K-fold train/validation index splits.
fn kfold_indices(n_samples: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let base = n_samples / k;
    let remainder = n_samples % k;
    let mut folds = Vec::with_capacity(k);
    let all: Vec<usize> = (0..n_samples).collect();

    let mut start = 0;
    for fold in 0..k {
        let size = base + if fold < remainder { 1 } else { 0 };
        let end = start + size;
        let val_idx: Vec<usize> = (start..end).collect();
        let train_idx: Vec<usize> = all[..start]
            .iter()
            .chain(all[end..].iter())
            .copied()
            .collect();
        folds.push((train_idx, val_idx));
        start = end;
    }

    folds
}

/// Select rows from a 2D array by index.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let n_rows = indices.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &i in indices {
        data.extend(x.row(i).iter().copied());
    }
    Array2::from_shape_vec((n_rows, n_cols), data)
        .expect("select_rows: shape should always be valid")
}

/// Select elements from a 1D array by index.
fn select_elements(y: &Array1<usize>, indices: &[usize]) -> Array1<usize> {
    Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;

    // -- Helper: a simple threshold-based classifier --------------------------

    /// Build a `FitFn` that trains a trivial "mean score" classifier.
    /// For each sample, the score is the mean of its features.
    fn mean_score_fit_fn() -> FitFn {
        Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
            Ok(Box::new(|x: &Array2<f64>| {
                let scores =
                    x.mean_axis(Axis(1))
                        .ok_or_else(|| FerroError::NumericalInstability {
                            message: "empty feature matrix".into(),
                        })?;
                Ok(scores)
            }) as PredictFn)
        })
    }

    /// Build a `FitFn` that always returns a constant score.
    fn constant_score_fit_fn(value: f64) -> FitFn {
        Box::new(move |_x: &Array2<f64>, _y: &Array1<usize>| {
            let v = value;
            Ok(Box::new(move |x: &Array2<f64>| Ok(Array1::from_elem(x.nrows(), v))) as PredictFn)
        })
    }

    // -- CalibrationMethod tests ----------------------------------------------

    #[test]
    fn test_calibration_method_eq() {
        assert_eq!(CalibrationMethod::Sigmoid, CalibrationMethod::Sigmoid);
        assert_eq!(CalibrationMethod::Isotonic, CalibrationMethod::Isotonic);
        assert_ne!(CalibrationMethod::Sigmoid, CalibrationMethod::Isotonic);
    }

    // -- Sigmoid calibration tests --------------------------------------------

    #[test]
    fn test_sigmoid_fn_properties() {
        // sigmoid(0) = 0.5
        assert!((sigmoid_fn(0.0) - 0.5).abs() < 1e-10);
        // sigmoid(large positive) -> 1
        assert!((sigmoid_fn(100.0) - 1.0).abs() < 1e-10);
        // sigmoid(large negative) -> 0
        assert!(sigmoid_fn(-100.0).abs() < 1e-10);
        // sigmoid(-x) = 1 - sigmoid(x)
        let x = 2.5;
        assert!((sigmoid_fn(-x) - (1.0 - sigmoid_fn(x))).abs() < 1e-10);
    }

    #[test]
    fn test_fit_sigmoid_basic() {
        // Positive samples have high scores, negative samples have low scores.
        let scores = Array1::from_vec(vec![-2.0, -1.0, 0.5, 1.0, 2.0, 3.0]);
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let (a, _b) = fit_sigmoid(&scores, &labels).unwrap();

        // Our model is P = sigmoid(a*f + b). For well-separated data where
        // class 1 has higher scores, a should be positive so that higher
        // scores map to higher probabilities.
        assert!(a > 0.0, "Expected a > 0 for well-separated data, got {a}");
    }

    #[test]
    fn test_fit_sigmoid_empty() {
        let scores = Array1::from_vec(vec![]);
        let labels = Array1::from_vec(vec![]);
        assert!(fit_sigmoid(&scores, &labels).is_err());
    }

    // -- Isotonic calibration tests -------------------------------------------

    #[test]
    fn test_fit_isotonic_basic() {
        let scores = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let mapping = fit_isotonic(&scores, &labels).unwrap();

        // Mapping should be non-decreasing.
        for i in 1..mapping.len() {
            assert!(
                mapping[i].1 >= mapping[i - 1].1 - 1e-10,
                "Isotonic mapping should be non-decreasing at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fit_isotonic_empty() {
        let scores = Array1::from_vec(vec![]);
        let labels = Array1::from_vec(vec![]);
        assert!(fit_isotonic(&scores, &labels).is_err());
    }

    #[test]
    fn test_isotonic_lookup_endpoints() {
        let mapping = vec![(0.0, 0.1), (1.0, 0.5), (2.0, 0.9)];
        // Below range -> clamp to first.
        assert!((isotonic_lookup(&mapping, -1.0) - 0.1).abs() < 1e-10);
        // Above range -> clamp to last.
        assert!((isotonic_lookup(&mapping, 3.0) - 0.9).abs() < 1e-10);
        // At breakpoint.
        assert!((isotonic_lookup(&mapping, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_isotonic_lookup_interpolation() {
        let mapping = vec![(0.0, 0.0), (2.0, 1.0)];
        // Midpoint should interpolate to 0.5.
        assert!((isotonic_lookup(&mapping, 1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_isotonic_lookup_empty() {
        assert!((isotonic_lookup(&[], 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_isotonic_lookup_single() {
        let mapping = vec![(1.0, 0.7)];
        assert!((isotonic_lookup(&mapping, 0.0) - 0.7).abs() < 1e-10);
        assert!((isotonic_lookup(&mapping, 2.0) - 0.7).abs() < 1e-10);
    }

    // -- CalibratedClassifierCV fit/predict tests -----------------------------

    #[test]
    fn test_calibrated_classifier_sigmoid_fit_predict() {
        // Construct a dataset where class 1 samples have higher feature values.
        let n = 20;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            if i < n / 2 {
                x_data.push(0.0);
                x_data.push(0.1 * i as f64);
                y_data.push(0);
            } else {
                x_data.push(1.0);
                x_data.push(0.5 + 0.1 * i as f64);
                y_data.push(1);
            }
        }
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let cal = CalibratedClassifierCV::new(mean_score_fit_fn(), CalibrationMethod::Sigmoid, 3);
        let fitted = cal.fit(&x, &y).unwrap();
        let probs = fitted.predict(&x).unwrap();

        // All probabilities should be in [0, 1].
        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {p} out of [0, 1] range"
            );
        }
    }

    #[test]
    fn test_calibrated_classifier_isotonic_fit_predict() {
        let n = 20;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for i in 0..n {
            if i < n / 2 {
                x_data.push(0.0);
                x_data.push(0.1 * i as f64);
                y_data.push(0);
            } else {
                x_data.push(1.0);
                x_data.push(0.5 + 0.1 * i as f64);
                y_data.push(1);
            }
        }
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let cal = CalibratedClassifierCV::new(mean_score_fit_fn(), CalibrationMethod::Isotonic, 3);
        let fitted = cal.fit(&x, &y).unwrap();
        let probs = fitted.predict(&x).unwrap();

        // All probabilities should be in [0, 1].
        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {p} out of [0, 1] range"
            );
        }
    }

    #[test]
    fn test_calibrated_classifier_constant_scores() {
        // When all scores are identical, calibration should still work.
        let n = 10;
        let x = Array2::from_elem((n, 2), 1.0);
        let mut y_data = vec![0; n / 2];
        y_data.extend(vec![1; n - n / 2]);
        let y = Array1::from_vec(y_data);

        let cal =
            CalibratedClassifierCV::new(constant_score_fit_fn(0.5), CalibrationMethod::Sigmoid, 2);
        let fitted = cal.fit(&x, &y).unwrap();
        let probs = fitted.predict(&x).unwrap();

        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {p} out of [0, 1] range"
            );
        }
    }

    #[test]
    fn test_calibrated_classifier_cv_too_small() {
        let x = Array2::from_elem((10, 2), 1.0);
        let y = Array1::from_elem(10, 0);
        let cal = CalibratedClassifierCV::new(mean_score_fit_fn(), CalibrationMethod::Sigmoid, 1);
        assert!(cal.fit(&x, &y).is_err());
    }

    #[test]
    fn test_calibrated_classifier_insufficient_samples() {
        let x = Array2::from_elem((2, 2), 1.0);
        let y = Array1::from_vec(vec![0, 1]);
        let cal = CalibratedClassifierCV::new(mean_score_fit_fn(), CalibrationMethod::Sigmoid, 5);
        assert!(cal.fit(&x, &y).is_err());
    }

    #[test]
    fn test_calibrated_classifier_shape_mismatch() {
        let x = Array2::from_elem((10, 2), 1.0);
        let y = Array1::from_elem(8, 0); // wrong length
        let cal = CalibratedClassifierCV::new(mean_score_fit_fn(), CalibrationMethod::Sigmoid, 3);
        assert!(cal.fit(&x, &y).is_err());
    }

    // -- kfold_indices tests --------------------------------------------------

    #[test]
    fn test_kfold_indices_coverage() {
        let folds = kfold_indices(10, 3);
        assert_eq!(folds.len(), 3);

        let mut all_val: Vec<usize> = folds.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        all_val.sort_unstable();
        assert_eq!(all_val, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_indices_no_overlap() {
        let folds = kfold_indices(12, 4);
        for (train, val) in &folds {
            for v in val {
                assert!(
                    !train.contains(v),
                    "Validation index {v} found in training set"
                );
            }
        }
    }

    // -- select_rows / select_elements ----------------------------------------

    #[test]
    fn test_select_rows() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let selected = select_rows(&x, &[0, 2]);
        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected[[0, 0]], 1.0);
        assert_eq!(selected[[1, 0]], 5.0);
    }

    #[test]
    fn test_select_elements() {
        let y = Array1::from_vec(vec![10, 20, 30, 40]);
        let selected = select_elements(&y, &[1, 3]);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 20);
        assert_eq!(selected[1], 40);
    }
}
