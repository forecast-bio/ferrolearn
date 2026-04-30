//! Decision-threshold classifiers (sklearn's `FixedThresholdClassifier`,
//! `TunedThresholdClassifierCV`).
//!
//! These are meta-classifiers that wrap a binary base estimator producing
//! probability or score outputs and convert them to hard `{0, 1}` predictions
//! using a configurable decision threshold.
//!
//! - [`FixedThresholdClassifier`] applies a user-supplied threshold.
//! - [`TunedThresholdClassifierCV`] picks the threshold by maximising a user-
//!   supplied scoring function over an out-of-fold prediction grid.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};

use crate::cross_validation::{CrossValidator, KFold};

/// Boxed function that, given a feature matrix, returns per-sample scores.
pub type ScoreFn = Box<dyn Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>>;

/// A boxed function that, given training `(X, y)`, returns a [`ScoreFn`].
///
/// This mirrors the calibration / self-training factory pattern used
/// elsewhere in this crate.
pub type FitScoreFn =
    Box<dyn Fn(&Array2<f64>, &Array1<usize>) -> Result<ScoreFn, FerroError> + Send + Sync>;

// ---------------------------------------------------------------------------
// FixedThresholdClassifier
// ---------------------------------------------------------------------------

/// Wrap a base classifier and threshold its score at a fixed value.
pub struct FixedThresholdClassifier {
    fit_fn: FitScoreFn,
    threshold: f64,
}

impl FixedThresholdClassifier {
    /// Construct a new [`FixedThresholdClassifier`] from a `fit_fn` factory
    /// (which trains the base model and returns a score function) and a
    /// fixed decision threshold.
    pub fn new(fit_fn: FitScoreFn, threshold: f64) -> Self {
        Self { fit_fn, threshold }
    }

    /// Train the base classifier and return a fitted predictor.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedFixedThresholdClassifier, FerroError> {
        let scorer = (self.fit_fn)(x, y)?;
        Ok(FittedFixedThresholdClassifier {
            scorer,
            threshold: self.threshold,
        })
    }
}

/// A fitted [`FixedThresholdClassifier`].
pub struct FittedFixedThresholdClassifier {
    scorer: ScoreFn,
    threshold: f64,
}

impl FittedFixedThresholdClassifier {
    /// The fixed decision threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Compute hard `{0, 1}` predictions.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let scores = (self.scorer)(x)?;
        Ok(scores.mapv(|s| if s >= self.threshold { 1usize } else { 0 }))
    }

    /// Return the raw scores from the wrapped base classifier.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        (self.scorer)(x)
    }
}

// ---------------------------------------------------------------------------
// TunedThresholdClassifierCV
// ---------------------------------------------------------------------------

/// Type alias for the score function passed to
/// [`TunedThresholdClassifierCV`].
pub type ThresholdScoring = Box<dyn Fn(&Array1<usize>, &Array1<usize>) -> f64 + Send + Sync>;

/// Pick the decision threshold by k-fold cross-validation, maximising a
/// user-supplied score over a candidate-threshold grid.
pub struct TunedThresholdClassifierCV {
    fit_fn: FitScoreFn,
    cv: usize,
    thresholds: Vec<f64>,
    scoring: ThresholdScoring,
}

impl TunedThresholdClassifierCV {
    /// Construct a new [`TunedThresholdClassifierCV`].
    ///
    /// * `fit_fn` — base estimator factory (same shape as
    ///   [`FixedThresholdClassifier::new`]).
    /// * `cv` — number of CV folds (must be >= 2).
    /// * `thresholds` — candidate thresholds to evaluate. If empty, defaults
    ///   to `[0.0, 0.05, 0.1, ..., 1.0]`.
    /// * `scoring` — closure mapping `(y_true, y_pred) -> f64`. Higher = better.
    pub fn new(
        fit_fn: FitScoreFn,
        cv: usize,
        thresholds: Vec<f64>,
        scoring: ThresholdScoring,
    ) -> Self {
        let grid = if thresholds.is_empty() {
            (0..=20).map(|i| (i as f64) * 0.05).collect()
        } else {
            thresholds
        };
        Self {
            fit_fn,
            cv,
            thresholds: grid,
            scoring,
        }
    }

    /// Run k-fold CV, find the threshold maximising the mean score, then refit
    /// the base estimator on the whole training set with that threshold.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedTunedThresholdClassifierCV, FerroError> {
        if self.cv < 2 {
            return Err(FerroError::InvalidParameter {
                name: "cv".into(),
                reason: format!(
                    "TunedThresholdClassifierCV: cv must be >= 2, got {}",
                    self.cv
                ),
            });
        }
        let n = x.nrows();
        if y.len() != n {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n],
                actual: vec![y.len()],
                context: "TunedThresholdClassifierCV: y length must equal x rows".into(),
            });
        }
        if n < self.cv {
            return Err(FerroError::InsufficientSamples {
                required: self.cv,
                actual: n,
                context: "TunedThresholdClassifierCV".into(),
            });
        }

        // Collect out-of-fold scores via CrossValidator::fold_indices.
        let kf = KFold::new(self.cv);
        let folds = kf.fold_indices(n)?;

        // For each fold, fit base on train rows and score test rows.
        let mut oof_scores = Array1::<f64>::from_elem(n, f64::NAN);
        for (train_idx, test_idx) in &folds {
            let train_x = subset_rows(x, train_idx);
            let train_y = subset_rows_1d(y, train_idx);
            let test_x = subset_rows(x, test_idx);
            let predictor = (self.fit_fn)(&train_x, &train_y)?;
            let test_scores = predictor(&test_x)?;
            for (k, &i) in test_idx.iter().enumerate() {
                oof_scores[i] = test_scores[k];
            }
        }

        // For each candidate threshold, score the OOF predictions.
        let mut best_thr = self.thresholds[0];
        let mut best_score = f64::NEG_INFINITY;
        let mut all_scores = Vec::with_capacity(self.thresholds.len());
        for &thr in &self.thresholds {
            let preds: Array1<usize> = oof_scores.mapv(|s| if s >= thr { 1usize } else { 0 });
            let score = (self.scoring)(y, &preds);
            all_scores.push(score);
            if score > best_score {
                best_score = score;
                best_thr = thr;
            }
        }

        // Refit base on the full training set with the chosen threshold.
        let scorer = (self.fit_fn)(x, y)?;
        Ok(FittedTunedThresholdClassifierCV {
            scorer,
            best_threshold: best_thr,
            best_score,
            cv_scores: all_scores,
            thresholds: self.thresholds.clone(),
        })
    }
}

/// A fitted [`TunedThresholdClassifierCV`].
pub struct FittedTunedThresholdClassifierCV {
    scorer: ScoreFn,
    best_threshold: f64,
    best_score: f64,
    cv_scores: Vec<f64>,
    thresholds: Vec<f64>,
}

impl FittedTunedThresholdClassifierCV {
    /// The threshold that maximised the mean CV score.
    pub fn best_threshold(&self) -> f64 {
        self.best_threshold
    }
    /// The CV score at the best threshold.
    pub fn best_score(&self) -> f64 {
        self.best_score
    }
    /// The CV score at every candidate threshold (same order as
    /// [`thresholds`](Self::thresholds)).
    pub fn cv_scores(&self) -> &[f64] {
        &self.cv_scores
    }
    /// The candidate thresholds the search ran over.
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// Hard `{0, 1}` predictions using the chosen threshold.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let scores = (self.scorer)(x)?;
        Ok(scores.mapv(|s| if s >= self.best_threshold { 1usize } else { 0 }))
    }

    /// Raw scores from the wrapped base classifier.
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        (self.scorer)(x)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn subset_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((indices.len(), x.ncols()));
    for (k, &i) in indices.iter().enumerate() {
        for j in 0..x.ncols() {
            out[[k, j]] = x[[i, j]];
        }
    }
    out
}

fn subset_rows_1d(y: &Array1<usize>, indices: &[usize]) -> Array1<usize> {
    let mut out = Array1::<usize>::zeros(indices.len());
    for (k, &i) in indices.iter().enumerate() {
        out[k] = y[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Trivial scorer: returns column 0 verbatim as the positive-class score.
    fn col0_fit_fn() -> FitScoreFn {
        Box::new(|_x: &Array2<f64>, _y: &Array1<usize>| {
            Ok(Box::new(|x: &Array2<f64>| {
                let n = x.nrows();
                let mut out = Array1::<f64>::zeros(n);
                for i in 0..n {
                    out[i] = x[[i, 0]];
                }
                Ok(out)
            })
                as Box<
                    dyn Fn(&Array2<f64>) -> Result<Array1<f64>, FerroError>,
                >)
        })
    }

    fn accuracy() -> ThresholdScoring {
        Box::new(|y_true: &Array1<usize>, y_pred: &Array1<usize>| {
            let n = y_true.len() as f64;
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(a, b)| a == b)
                .count() as f64;
            correct / n
        })
    }

    #[test]
    fn fixed_threshold_basic() {
        let clf = FixedThresholdClassifier::new(col0_fit_fn(), 0.5);
        let x = array![[0.1, 9.0], [0.6, 9.0], [0.4, 9.0], [0.9, 9.0]];
        let y = array![0usize, 1, 0, 1];
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, y);
        let dec = fitted.decision_function(&x).unwrap();
        assert_eq!(dec.len(), 4);
        assert!((fitted.threshold() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tuned_threshold_picks_best() {
        let x = array![
            [0.1, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
            [0.8, 0.0],
            [0.9, 0.0]
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        let clf = TunedThresholdClassifierCV::new(
            col0_fit_fn(),
            2,
            vec![0.1, 0.3, 0.5, 0.7, 0.9],
            accuracy(),
        );
        let fitted = clf.fit(&x, &y).unwrap();
        // The optimal threshold is 0.5 (separates 0.1/0.2/0.4 from 0.6/0.8/0.9).
        assert!((fitted.best_threshold() - 0.5).abs() < 1e-9);
        assert_eq!(fitted.cv_scores().len(), 5);
        assert_eq!(fitted.thresholds().len(), 5);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    #[test]
    fn tuned_threshold_default_grid() {
        let clf = TunedThresholdClassifierCV::new(col0_fit_fn(), 2, vec![], accuracy());
        let x = array![[0.1, 0.0], [0.9, 0.0]];
        let y = array![0usize, 1];
        // Just confirm we don't blow up with default grid + minimum-viable CV.
        let _ = clf.fit(&x, &y).unwrap();
    }

    #[test]
    fn tuned_threshold_rejects_cv1() {
        let clf = TunedThresholdClassifierCV::new(col0_fit_fn(), 1, vec![0.5], accuracy());
        let x = array![[0.1, 0.0], [0.9, 0.0]];
        let y = array![0usize, 1];
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn tuned_threshold_shape_mismatch() {
        let clf = TunedThresholdClassifierCV::new(col0_fit_fn(), 2, vec![0.5], accuracy());
        let x = array![[0.1, 0.0], [0.9, 0.0]];
        let y = array![0usize];
        assert!(clf.fit(&x, &y).is_err());
    }
}
