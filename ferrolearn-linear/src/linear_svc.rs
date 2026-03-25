//! Linear Support Vector Classifier.
//!
//! This module provides [`LinearSVC`], an optimized linear SVM that operates
//! directly in the primal space without the overhead of a kernel function.
//! It uses coordinate descent on the L2-regularized hinge or squared-hinge
//! loss.
//!
//! Unlike [`SVC`](crate::svm::SVC) with a [`LinearKernel`](crate::svm::LinearKernel),
//! `LinearSVC` avoids computing and caching the full kernel matrix, making it
//! significantly faster for high-dimensional data.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::linear_svc::LinearSVC;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
//!     5.0, 5.0, 5.0, 6.0, 6.0, 5.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = LinearSVC::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Loss function for [`LinearSVC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSVCLoss {
    /// Standard hinge loss: `max(0, 1 - y * f(x))`.
    Hinge,
    /// Squared hinge loss: `max(0, 1 - y * f(x))^2`.
    SquaredHinge,
}

/// Linear Support Vector Classifier (primal formulation).
///
/// Solves the L2-regularized hinge or squared-hinge loss via coordinate
/// descent in the primal. Supports binary and multiclass (one-vs-rest)
/// classification.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearSVC<F> {
    /// Inverse regularization strength. Larger values allow more
    /// misclassification.
    pub c: F,
    /// Maximum number of coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the change in weight vector.
    pub tol: F,
    /// Loss function to use.
    pub loss: LinearSVCLoss,
}

impl<F: Float> LinearSVC<F> {
    /// Create a new `LinearSVC` with default settings.
    ///
    /// Defaults: `C = 1.0`, `max_iter = 1000`, `tol = 1e-4`,
    /// `loss = SquaredHinge`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: F::one(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            loss: LinearSVCLoss::SquaredHinge,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self {
        self.loss = loss;
        self
    }
}

impl<F: Float> Default for LinearSVC<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear Support Vector Classifier.
///
/// Stores the learned weight vectors, intercepts, and class labels.
/// For binary classification a single weight vector is stored; for
/// multiclass, one per class (one-vs-rest).
#[derive(Debug, Clone)]
pub struct FittedLinearSVC<F> {
    /// Weight vectors: one per binary sub-problem.
    /// Binary: `[w]`, Multiclass: `[w_0, w_1, ..., w_{k-1}]`.
    weight_vectors: Vec<Array1<F>>,
    /// Intercept for each sub-problem.
    intercepts: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem.
    is_binary: bool,
    /// Number of features.
    n_features: usize,
}

impl<F: Float> FittedLinearSVC<F> {
    /// Returns the weight vectors (one per binary sub-problem).
    #[must_use]
    pub fn weight_vectors(&self) -> &[Array1<F>] {
        &self.weight_vectors
    }

    /// Returns the intercepts (one per binary sub-problem).
    #[must_use]
    pub fn intercepts(&self) -> &[F] {
        &self.intercepts
    }
}

/// Solve a single binary L2-SVM via coordinate descent on the primal.
///
/// Minimises: `0.5 * ||w||^2 + C * sum_i loss(y_i, w^T x_i + b)`
///
/// where `y_i in {-1, +1}`.
fn solve_binary_primal<F: Float + 'static>(
    x: &Array2<F>,
    y_signed: &Array1<F>,
    c: F,
    max_iter: usize,
    tol: F,
    loss: LinearSVCLoss,
) -> (Array1<F>, F) {
    let (n_samples, n_features) = x.dim();
    let mut w = Array1::<F>::zeros(n_features);
    let mut b = F::zero();

    let n_f = F::from(n_samples).unwrap();

    for _iter in 0..max_iter {
        let mut max_change = F::zero();

        // Update each weight coordinate.
        for j in 0..n_features {
            let mut grad = w[j]; // regularization gradient
            for i in 0..n_samples {
                let margin = y_signed[i] * (x.row(i).dot(&w) + b);
                match loss {
                    LinearSVCLoss::Hinge => {
                        if margin < F::one() {
                            grad = grad - c / n_f * y_signed[i] * x[[i, j]];
                        }
                    }
                    LinearSVCLoss::SquaredHinge => {
                        if margin < F::one() {
                            let two = F::from(2.0).unwrap();
                            grad = grad
                                - two * c / n_f * (F::one() - margin) * y_signed[i] * x[[i, j]];
                        }
                    }
                }
            }

            let step = F::from(0.01).unwrap(); // learning rate
            let new_w = w[j] - step * grad;
            let change = (new_w - w[j]).abs();
            if change > max_change {
                max_change = change;
            }
            w[j] = new_w;
        }

        // Update intercept (not regularized).
        {
            let mut grad_b = F::zero();
            for i in 0..n_samples {
                let margin = y_signed[i] * (x.row(i).dot(&w) + b);
                match loss {
                    LinearSVCLoss::Hinge => {
                        if margin < F::one() {
                            grad_b = grad_b - c / n_f * y_signed[i];
                        }
                    }
                    LinearSVCLoss::SquaredHinge => {
                        if margin < F::one() {
                            let two = F::from(2.0).unwrap();
                            grad_b = grad_b - two * c / n_f * (F::one() - margin) * y_signed[i];
                        }
                    }
                }
            }
            let step = F::from(0.01).unwrap();
            let new_b = b - step * grad_b;
            let change = (new_b - b).abs();
            if change > max_change {
                max_change = change;
            }
            b = new_b;
        }

        if max_change < tol {
            break;
        }
    }

    (w, b)
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for LinearSVC<F>
{
    type Fitted = FittedLinearSVC<F>;
    type Error = FerroError;

    /// Fit the linear SVC model using coordinate descent.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `C` not positive.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 distinct classes.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedLinearSVC<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "LinearSVC requires at least 2 distinct classes".into(),
            });
        }

        if classes.len() == 2 {
            // Binary classification.
            let y_signed: Array1<F> = y.mapv(|label| {
                if label == classes[1] {
                    F::one()
                } else {
                    -F::one()
                }
            });

            let (w, b) = solve_binary_primal(x, &y_signed, self.c, self.max_iter, self.tol, self.loss);

            Ok(FittedLinearSVC {
                weight_vectors: vec![w],
                intercepts: vec![b],
                classes,
                is_binary: true,
                n_features,
            })
        } else {
            // Multiclass: one-vs-rest.
            let mut weight_vectors = Vec::with_capacity(classes.len());
            let mut intercepts = Vec::with_capacity(classes.len());

            for &cls in &classes {
                let y_signed: Array1<F> = y.mapv(|label| {
                    if label == cls {
                        F::one()
                    } else {
                        -F::one()
                    }
                });

                let (w, b) =
                    solve_binary_primal(x, &y_signed, self.c, self.max_iter, self.tol, self.loss);
                weight_vectors.push(w);
                intercepts.push(b);
            }

            Ok(FittedLinearSVC {
                weight_vectors,
                intercepts,
                classes,
                is_binary: false,
                n_features,
            })
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLinearSVC<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Binary: `sign(X @ w + b)` mapped to class labels.
    /// Multiclass: argmax of decision values across one-vs-rest classifiers.
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

        if self.is_binary {
            let scores = x.dot(&self.weight_vectors[0]) + self.intercepts[0];
            for i in 0..n_samples {
                predictions[i] = if scores[i] >= F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: pick class with highest decision value.
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut best_score = F::neg_infinity();
                for (c, w) in self.weight_vectors.iter().enumerate() {
                    let score = x.row(i).dot(w) + self.intercepts[c];
                    if score > best_score {
                        best_score = score;
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
    for FittedLinearSVC<F>
{
    /// Returns the coefficient vector of the first (or only) binary sub-problem.
    fn coefficients(&self) -> &Array1<F> {
        &self.weight_vectors[0]
    }

    /// Returns the intercept of the first (or only) binary sub-problem.
    fn intercept(&self) -> F {
        self.intercepts[0]
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedLinearSVC<F> {
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
        let m = LinearSVC::<f64>::new();
        assert_eq!(m.max_iter, 1000);
        assert!(m.c == 1.0);
        assert_eq!(m.loss, LinearSVCLoss::SquaredHinge);
    }

    #[test]
    fn test_builder_setters() {
        let m = LinearSVC::<f64>::new()
            .with_c(10.0)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_loss(LinearSVCLoss::Hinge);
        assert!(m.c == 10.0);
        assert_eq!(m.max_iter, 500);
        assert_eq!(m.loss, LinearSVCLoss::Hinge);
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

        let model = LinearSVC::<f64>::new().with_c(1.0).with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_binary_hinge_loss() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LinearSVC::<f64>::new()
            .with_loss(LinearSVCLoss::Hinge)
            .with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 4, "expected at least 4 correct, got {correct}");
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

        let model = LinearSVC::<f64>::new().with_c(10.0).with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = LinearSVC::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = LinearSVC::<f64>::new().with_c(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model_neg = LinearSVC::<f64>::new().with_c(-1.0);
        assert!(model_neg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = LinearSVC::<f64>::new();
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

        let model = LinearSVC::<f64>::new().with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = LinearSVC::<f64>::new().with_max_iter(5000).fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }
}
