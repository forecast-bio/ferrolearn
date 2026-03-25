//! Scorer utility for wrapping scoring functions with metadata.
//!
//! The [`Scorer`] struct associates a scoring function with metadata that tells
//! model-selection utilities (like cross-validation) how to interpret the
//! score. For example, `accuracy_score` has `greater_is_better = true`, while
//! `mean_squared_error` has `greater_is_better = false`.
//!
//! Use [`make_scorer`] to create a scorer from a plain function pointer.

use ferrolearn_core::FerroError;
use ndarray::Array1;
use num_traits::Float;

/// A scoring function wrapped with metadata.
///
/// `Scorer` pairs a function pointer `score_fn` with a flag indicating whether
/// a higher score is better and a human-readable name. Model-selection code
/// (e.g., `GridSearchCV`) can use [`greater_is_better`](Scorer::greater_is_better) to decide
/// whether to maximise or minimise the metric.
///
/// # Type Parameters
///
/// * `F` — the float type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::scorer::{make_scorer, Scorer};
/// use ferrolearn_metrics::regression::mean_absolute_error;
///
/// let scorer: Scorer<f64> = make_scorer(mean_absolute_error, false, "neg_mean_absolute_error");
/// assert!(!scorer.greater_is_better);
/// assert_eq!(scorer.name, "neg_mean_absolute_error");
/// ```
#[derive(Clone)]
pub struct Scorer<F> {
    /// The scoring function.
    pub score_fn: fn(&Array1<F>, &Array1<F>) -> Result<F, FerroError>,
    /// If `true`, a larger score indicates a better model. If `false`, a
    /// smaller score is better (e.g., for loss functions).
    pub greater_is_better: bool,
    /// A human-readable name for this scorer.
    pub name: String,
}

impl<F: Float> std::fmt::Debug for Scorer<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scorer")
            .field("name", &self.name)
            .field("greater_is_better", &self.greater_is_better)
            .finish()
    }
}

impl<F: Float> Scorer<F> {
    /// Evaluate this scorer on a pair of arrays.
    ///
    /// Calls the underlying `score_fn` and returns the result.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying scoring function.
    pub fn score(&self, y_true: &Array1<F>, y_pred: &Array1<F>) -> Result<F, FerroError> {
        (self.score_fn)(y_true, y_pred)
    }

    /// Return the sign multiplier for optimisation.
    ///
    /// Returns `1.0` if `greater_is_better`, otherwise `-1.0`. This is useful
    /// for converting any scorer into a "higher is better" convention by
    /// multiplying the raw score by this factor.
    #[must_use]
    pub fn sign(&self) -> F {
        if self.greater_is_better {
            F::one()
        } else {
            -F::one()
        }
    }
}

/// Create a [`Scorer`] from a scoring function and metadata.
///
/// # Arguments
///
/// * `score_fn`         — a function pointer with signature
///   `fn(&Array1<F>, &Array1<F>) -> Result<F, FerroError>`.
/// * `greater_is_better` — `true` if a higher score is better.
/// * `name`             — a human-readable name for the scorer.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::scorer::{make_scorer, Scorer};
/// use ferrolearn_metrics::regression::r2_score;
///
/// let scorer: Scorer<f64> = make_scorer(r2_score, true, "r2");
/// assert!(scorer.greater_is_better);
/// assert_eq!(scorer.name, "r2");
/// ```
pub fn make_scorer<F: Float>(
    score_fn: fn(&Array1<F>, &Array1<F>) -> Result<F, FerroError>,
    greater_is_better: bool,
    name: &str,
) -> Scorer<F> {
    Scorer {
        score_fn,
        greater_is_better,
        name: name.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // A trivial scoring function for testing.
    fn dummy_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        if y_true.len() != y_pred.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
                context: "dummy_score".into(),
            });
        }
        let n = y_true.len() as f64;
        let sum = y_true
            .iter()
            .zip(y_pred.iter())
            .fold(0.0, |acc, (&t, &p)| acc + (t - p).abs());
        Ok(sum / n)
    }

    #[test]
    fn test_make_scorer_basic() {
        let scorer = make_scorer(dummy_score, false, "dummy_mae");
        assert!(!scorer.greater_is_better);
        assert_eq!(scorer.name, "dummy_mae");
    }

    #[test]
    fn test_scorer_evaluate() {
        let scorer = make_scorer(dummy_score, false, "dummy_mae");
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.5_f64, 2.0, 2.5];
        let result = scorer.score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scorer_sign_greater_is_better() {
        let scorer = make_scorer(dummy_score, true, "test");
        assert_abs_diff_eq!(scorer.sign(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scorer_sign_less_is_better() {
        let scorer = make_scorer(dummy_score, false, "test");
        assert_abs_diff_eq!(scorer.sign(), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scorer_debug() {
        let scorer = make_scorer(dummy_score, true, "my_scorer");
        let debug_str = format!("{:?}", scorer);
        assert!(debug_str.contains("my_scorer"));
        assert!(debug_str.contains("greater_is_better"));
    }

    #[test]
    fn test_scorer_error_propagation() {
        let scorer = make_scorer(dummy_score, false, "dummy_mae");
        let y_true = array![1.0_f64, 2.0];
        let y_pred = array![1.0_f64]; // length mismatch
        assert!(scorer.score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_scorer_with_real_metric() {
        use crate::regression::mean_absolute_error;
        let scorer = make_scorer(mean_absolute_error, false, "neg_mae");
        let y_true = array![1.0_f64, 2.0, 3.0];
        let y_pred = array![1.0_f64, 2.0, 3.0];
        let result = scorer.score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scorer_clone() {
        let scorer = make_scorer(dummy_score, true, "cloneable");
        let cloned = scorer.clone();
        assert_eq!(cloned.name, scorer.name);
        assert_eq!(cloned.greater_is_better, scorer.greater_is_better);
    }
}
