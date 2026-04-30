//! Ridge Classifier.
//!
//! This module provides [`RidgeClassifier`], which applies Ridge regression
//! to classification tasks by converting class labels into a binary indicator
//! matrix and fitting a multivariate Ridge regression.
//!
//! For binary classification, the indicator matrix has a single column
//! (`{-1, +1}`). For multiclass, it has one column per class (one-hot
//! encoding). The predicted class is the one with the highest decision
//! value (`argmax(X @ coef + intercept)`).
//!
//! This approach is significantly faster than logistic regression for
//! large datasets while often achieving competitive accuracy.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ridge_classifier::RidgeClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
//!     5.0, 5.0, 5.0, 6.0, 6.0, 5.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = RidgeClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::linalg;

/// Ridge Classifier.
///
/// Applies Ridge regression (L2-regularized least squares) to classification
/// by converting labels to a binary indicator matrix.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RidgeClassifier<F> {
    /// Regularization strength. Larger values specify stronger regularization.
    pub alpha: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> RidgeClassifier<F> {
    /// Create a new `RidgeClassifier` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            fit_intercept: true,
        }
    }

    /// Set the regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for RidgeClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Ridge Classifier.
///
/// Stores the learned coefficient matrix, intercept vector, and class labels.
#[derive(Debug, Clone)]
pub struct FittedRidgeClassifier<F> {
    /// Coefficient matrix, shape `(n_features, n_targets)`.
    /// For binary, `n_targets = 1`.
    coef_matrix: Array2<F>,
    /// Intercept vector, one per target.
    intercept_vec: Array1<F>,
    /// For HasCoefficients: first column of coef_matrix.
    coefficients: Array1<F>,
    /// For HasCoefficients: first element of intercept_vec.
    intercept: F,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem.
    is_binary: bool,
    /// Number of features.
    n_features: usize,
}

impl<F: Float> FittedRidgeClassifier<F> {
    /// Returns the full coefficient matrix, shape `(n_features, n_targets)`.
    #[must_use]
    pub fn coef_matrix(&self) -> &Array2<F> {
        &self.coef_matrix
    }

    /// Returns the intercept vector.
    #[must_use]
    pub fn intercept_vec(&self) -> &Array1<F> {
        &self.intercept_vec
    }
}

impl<F: Float + ndarray::ScalarOperand + Send + Sync + 'static> FittedRidgeClassifier<F> {
    /// Raw `X @ coef + intercept` per class. Mirrors sklearn
    /// `RidgeClassifier.decision_function`.
    ///
    /// Returns shape `(n_samples, n_classes)`. argmax of each row agrees
    /// with [`Predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }
        Ok(x.dot(&self.coef_matrix) + &self.intercept_vec)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + FromPrimitive + 'static>
    Fit<Array2<F>, Array1<usize>> for RidgeClassifier<F>
{
    type Fitted = FittedRidgeClassifier<F>;
    type Error = FerroError;

    /// Fit the Ridge Classifier by converting labels to a binary indicator
    /// matrix and solving multivariate Ridge regression.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — negative alpha.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 classes.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedRidgeClassifier<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha < F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "RidgeClassifier requires at least 2 distinct classes".into(),
            });
        }

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RidgeClassifier requires at least one sample".into(),
            });
        }

        let is_binary = classes.len() == 2;

        // Build indicator matrix Y.
        let n_targets = if is_binary { 1 } else { classes.len() };
        let mut y_indicator = Array2::<F>::zeros((n_samples, n_targets));

        if is_binary {
            // Binary: encode as {-1, +1}.
            for i in 0..n_samples {
                y_indicator[[i, 0]] = if y[i] == classes[1] {
                    F::one()
                } else {
                    -F::one()
                };
            }
        } else {
            // Multiclass: one-hot.
            for i in 0..n_samples {
                let ci = classes.iter().position(|&c| c == y[i]).unwrap();
                y_indicator[[i, ci]] = F::one();
            }
        }

        // Center data if fit_intercept.
        let (x_work, y_work, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute column means".into(),
                })?;
            let y_mean = y_indicator
                .mean_axis(Axis(0))
                .ok_or_else(|| FerroError::NumericalInstability {
                    message: "failed to compute target means".into(),
                })?;
            let x_c = x - &x_mean;
            let y_c = &y_indicator - &y_mean;
            (x_c, y_c, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y_indicator.clone(), None, None)
        };

        // Solve Ridge for each target column.
        let mut coef_matrix = Array2::<F>::zeros((n_features, n_targets));
        for t in 0..n_targets {
            let y_col = y_work.column(t).to_owned();
            let w = linalg::solve_ridge(&x_work, &y_col, self.alpha)?;
            for j in 0..n_features {
                coef_matrix[[j, t]] = w[j];
            }
        }

        // Compute intercepts.
        let intercept_vec = if let (Some(xm), Some(ym)) = (&x_mean, &y_mean) {
            let xm_dot = xm.dot(&coef_matrix);
            ym - &xm_dot
        } else {
            Array1::<F>::zeros(n_targets)
        };

        let coefficients = coef_matrix.column(0).to_owned();
        let intercept = intercept_vec[0];

        Ok(FittedRidgeClassifier {
            coef_matrix,
            intercept_vec,
            coefficients,
            intercept,
            classes,
            is_binary,
            n_features,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedRidgeClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Computes `X @ coef_matrix + intercept_vec` and takes `argmax` per row.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        // Compute decision values: X @ coef_matrix + intercept_vec.
        let scores = x.dot(&self.coef_matrix) + &self.intercept_vec;

        if self.is_binary {
            for i in 0..n_samples {
                predictions[i] = if scores[[i, 0]] >= F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut best_score = scores[[i, 0]];
                for c in 1..self.classes.len() {
                    if scores[[i, c]] > best_score {
                        best_score = scores[[i, c]];
                        best_class = c;
                    }
                }
                predictions[i] = self.classes[best_class];
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedRidgeClassifier<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedRidgeClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_default_constructor() {
        let m = RidgeClassifier::<f64>::new();
        assert!(m.alpha == 1.0);
        assert!(m.fit_intercept);
    }

    #[test]
    fn test_builder() {
        let m = RidgeClassifier::<f64>::new()
            .with_alpha(0.5)
            .with_fit_intercept(false);
        assert!(m.alpha == 0.5);
        assert!(!m.fit_intercept);
    }

    #[test]
    fn test_binary_classification() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = RidgeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_classification() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
                10.0, 0.0, 10.5, 0.0, 10.0, 0.5,
                0.0, 10.0, 0.5, 10.0, 0.0, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = RidgeClassifier::<f64>::new().with_alpha(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = RidgeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_negative_alpha() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = RidgeClassifier::<f64>::new().with_alpha(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = RidgeClassifier::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_has_classes() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = RidgeClassifier::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_alpha_zero() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = RidgeClassifier::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }
}
