//! Logistic regression classifier.
//!
//! This module provides [`LogisticRegression`], a linear classifier that uses
//! the logistic (sigmoid) function for binary classification and softmax for
//! multiclass classification. Parameters are estimated using a custom L-BFGS
//! optimizer with Wolfe line search.
//!
//! The regularization parameter `C` is the inverse of regularization strength
//! (matching scikit-learn's convention): smaller values specify stronger
//! regularization.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::LogisticRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let model = LogisticRegression::<f64>::new();
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
//! ).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::optim::lbfgs::LbfgsOptimizer;

/// Logistic regression classifier.
///
/// Uses L-BFGS optimization to minimize the regularized logistic loss.
/// Supports both binary and multiclass (multinomial) classification.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LogisticRegression<F> {
    /// Inverse regularization strength. Smaller values specify stronger
    /// regularization (matching scikit-learn's convention).
    pub c: F,
    /// Maximum number of L-BFGS iterations.
    pub max_iter: usize,
    /// Convergence tolerance for the optimizer.
    pub tol: F,
    /// Whether to fit an intercept (bias) term.
    pub fit_intercept: bool,
}

impl<F: Float> LogisticRegression<F> {
    /// Create a new `LogisticRegression` with default settings.
    ///
    /// Defaults: `C = 1.0`, `max_iter = 1000`, `tol = 1e-4`,
    /// `fit_intercept = true`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            c: F::one(),
            max_iter: 1000,
            tol: F::from(1e-4).unwrap(),
            fit_intercept: true,
        }
    }

    /// Set the inverse regularization strength.
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

    /// Set whether to fit an intercept term.
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl<F: Float> Default for LogisticRegression<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted logistic regression classifier.
///
/// Stores the learned coefficients, intercept, and class labels.
/// For binary classification, stores a single coefficient vector.
/// For multiclass, stores one coefficient vector per class.
#[derive(Debug, Clone)]
pub struct FittedLogisticRegression<F> {
    /// Learned coefficient vectors.
    /// For binary: shape `(n_features,)` (single vector).
    /// For multiclass: shape `(n_classes, n_features)`.
    coefficients: Array1<F>,
    /// Learned intercept for the primary class (binary).
    intercept: F,
    /// All coefficient vectors for multiclass, shape `(n_classes, n_features)`.
    /// For binary, this has shape `(1, n_features)`.
    weight_matrix: Array2<F>,
    /// Intercept vector, one per class.
    intercept_vec: Array1<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem.
    is_binary: bool,
}

/// Sigmoid function: 1 / (1 + exp(-z)).
fn sigmoid<F: Float>(z: F) -> F {
    if z >= F::zero() {
        F::one() / (F::one() + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (F::one() + ez)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for LogisticRegression<F>
{
    type Fitted = FittedLogisticRegression<F>;
    type Error = FerroError;

    /// Fit the logistic regression model using L-BFGS optimization.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of samples in
    /// `x` and `y` differ.
    /// Returns [`FerroError::InvalidParameter`] if `C` is not positive.
    /// Returns [`FerroError::InsufficientSamples`] if there are fewer
    /// than 2 distinct classes.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedLogisticRegression<F>, FerroError> {
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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "LogisticRegression requires at least one sample".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "LogisticRegression requires at least 2 distinct classes".into(),
            });
        }

        let n_classes = classes.len();

        if n_classes == 2 {
            self.fit_binary(x, y, n_samples, n_features, &classes)
        } else {
            self.fit_multinomial(x, y, n_samples, n_features, &classes)
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> LogisticRegression<F> {
    /// Fit binary logistic regression.
    fn fit_binary(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
        n_samples: usize,
        n_features: usize,
        classes: &[usize],
    ) -> Result<FittedLogisticRegression<F>, FerroError> {
        let n_f = F::from(n_samples).unwrap();
        let reg = F::one() / self.c;

        // Convert labels to 0/1 float.
        let y_binary: Array1<F> = y.mapv(|label| {
            if label == classes[1] {
                F::one()
            } else {
                F::zero()
            }
        });

        // Parameter vector: [w_0, w_1, ..., w_{n_features-1}, (intercept)]
        let n_params = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        let objective = |params: &Array1<F>| -> (F, Array1<F>) {
            let w = params.slice(ndarray::s![..n_features]);
            let b = if self.fit_intercept {
                params[n_features]
            } else {
                F::zero()
            };

            // Compute logits: X @ w + b
            let logits = x.dot(&w.to_owned()) + b;

            // Compute loss and gradient.
            let mut loss = F::zero();
            let mut grad_w = Array1::<F>::zeros(n_features);
            let mut grad_b = F::zero();

            for i in 0..n_samples {
                let p = sigmoid(logits[i]);
                let yi = y_binary[i];

                // Binary cross-entropy loss (negative log-likelihood).
                let eps = F::from(1e-15).unwrap();
                let p_clipped = p.max(eps).min(F::one() - eps);
                loss = loss - (yi * p_clipped.ln() + (F::one() - yi) * (F::one() - p_clipped).ln());

                // Gradient.
                let diff = p - yi;
                let xi = x.row(i);
                for j in 0..n_features {
                    grad_w[j] = grad_w[j] + diff * xi[j];
                }
                if self.fit_intercept {
                    grad_b = grad_b + diff;
                }
            }

            // Average loss and add regularization.
            loss = loss / n_f;
            grad_w.mapv_inplace(|v| v / n_f);
            grad_b = grad_b / n_f;

            // L2 regularization (on weights only, not intercept).
            let reg_loss: F = w.iter().fold(F::zero(), |acc, &wi| acc + wi * wi);
            loss = loss + reg / (F::from(2.0).unwrap()) * reg_loss;

            for j in 0..n_features {
                grad_w[j] = grad_w[j] + reg * w[j];
            }

            let mut grad = Array1::<F>::zeros(n_params);
            for j in 0..n_features {
                grad[j] = grad_w[j];
            }
            if self.fit_intercept {
                grad[n_features] = grad_b;
            }

            (loss, grad)
        };

        let optimizer = LbfgsOptimizer::new(self.max_iter, self.tol);
        let x0 = Array1::<F>::zeros(n_params);
        let params = optimizer.minimize(objective, x0)?;

        let coefficients = params.slice(ndarray::s![..n_features]).to_owned();
        let intercept = if self.fit_intercept {
            params[n_features]
        } else {
            F::zero()
        };

        let weight_matrix = coefficients
            .clone()
            .into_shape_with_order((1, n_features))
            .map_err(|_| FerroError::NumericalInstability {
                message: "failed to reshape coefficients".into(),
            })?;

        Ok(FittedLogisticRegression {
            coefficients,
            intercept,
            weight_matrix,
            intercept_vec: Array1::from_vec(vec![intercept]),
            classes: classes.to_vec(),
            is_binary: true,
        })
    }

    /// Fit multinomial logistic regression.
    fn fit_multinomial(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
        n_samples: usize,
        n_features: usize,
        classes: &[usize],
    ) -> Result<FittedLogisticRegression<F>, FerroError> {
        let n_classes = classes.len();
        let n_f = F::from(n_samples).unwrap();
        let reg = F::one() / self.c;

        // Create class index map.
        let class_indices: Vec<usize> = y
            .iter()
            .map(|&label| classes.iter().position(|&c| c == label).unwrap())
            .collect();

        // One-hot encode targets.
        let mut y_onehot = Array2::<F>::zeros((n_samples, n_classes));
        for (i, &ci) in class_indices.iter().enumerate() {
            y_onehot[[i, ci]] = F::one();
        }

        // Parameter vector: flattened [W (n_classes x n_features), b (n_classes)]
        let n_weight_params = n_classes * n_features;
        let n_params = if self.fit_intercept {
            n_weight_params + n_classes
        } else {
            n_weight_params
        };

        let fit_intercept = self.fit_intercept;

        let objective = move |params: &Array1<F>| -> (F, Array1<F>) {
            // Extract weight matrix W (n_classes x n_features).
            let mut w_mat = Array2::<F>::zeros((n_classes, n_features));
            for c in 0..n_classes {
                for j in 0..n_features {
                    w_mat[[c, j]] = params[c * n_features + j];
                }
            }

            let b_vec: Array1<F> = if fit_intercept {
                Array1::from_shape_fn(n_classes, |c| params[n_weight_params + c])
            } else {
                Array1::zeros(n_classes)
            };

            // Compute logits: X @ W^T + b^T, shape (n_samples, n_classes).
            let logits = x.dot(&w_mat.t()) + &b_vec;

            // Softmax probabilities.
            let probs = softmax_2d(&logits);

            // Multinomial cross-entropy loss.
            let mut loss = F::zero();
            let eps = F::from(1e-15).unwrap();
            for i in 0..n_samples {
                for c in 0..n_classes {
                    let p = probs[[i, c]].max(eps);
                    loss = loss - y_onehot[[i, c]] * p.ln();
                }
            }
            loss = loss / n_f;

            // L2 regularization.
            let reg_loss: F = w_mat.iter().fold(F::zero(), |acc, &wi| acc + wi * wi);
            loss = loss + reg / F::from(2.0).unwrap() * reg_loss;

            // Gradient.
            // diff = probs - y_onehot, shape (n_samples, n_classes)
            let diff = &probs - &y_onehot;

            // grad_W = diff^T @ X / n, shape (n_classes, n_features)
            let grad_w = diff.t().dot(x) / n_f;

            let mut grad = Array1::<F>::zeros(n_params);
            for c in 0..n_classes {
                for j in 0..n_features {
                    grad[c * n_features + j] = grad_w[[c, j]] + reg * w_mat[[c, j]];
                }
            }

            if fit_intercept {
                // grad_b = sum(diff, axis=0) / n
                let grad_b = diff.sum_axis(Axis(0)) / n_f;
                for c in 0..n_classes {
                    grad[n_weight_params + c] = grad_b[c];
                }
            }

            (loss, grad)
        };

        let optimizer = LbfgsOptimizer::new(self.max_iter, self.tol);
        let x0 = Array1::<F>::zeros(n_params);
        let params = optimizer.minimize(objective, x0)?;

        // Extract results.
        let mut weight_matrix = Array2::<F>::zeros((n_classes, n_features));
        for c in 0..n_classes {
            for j in 0..n_features {
                weight_matrix[[c, j]] = params[c * n_features + j];
            }
        }

        let intercept_vec = if self.fit_intercept {
            Array1::from_shape_fn(n_classes, |c| params[n_weight_params + c])
        } else {
            Array1::zeros(n_classes)
        };

        // For HasCoefficients, store the first class coefficients.
        let coefficients = weight_matrix.row(0).to_owned();
        let intercept = intercept_vec[0];

        Ok(FittedLogisticRegression {
            coefficients,
            intercept,
            weight_matrix,
            intercept_vec,
            classes: classes.to_vec(),
            is_binary: false,
        })
    }
}

/// Compute softmax probabilities row-wise for a 2D array.
fn softmax_2d<F: Float>(logits: &Array2<F>) -> Array2<F> {
    let n_rows = logits.nrows();
    let n_cols = logits.ncols();
    let mut probs = Array2::<F>::zeros((n_rows, n_cols));

    for i in 0..n_rows {
        // Numerical stability: subtract max.
        let max_logit = logits
            .row(i)
            .iter()
            .fold(F::neg_infinity(), |a, &b| a.max(b));

        let mut sum = F::zero();
        for j in 0..n_cols {
            let exp_val = (logits[[i, j]] - max_logit).exp();
            probs[[i, j]] = exp_val;
            sum = sum + exp_val;
        }

        if sum > F::zero() {
            for j in 0..n_cols {
                probs[[i, j]] = probs[[i, j]] / sum;
            }
        }
    }

    probs
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> FittedLogisticRegression<F> {
    /// Returns a reference to the full weight matrix.
    ///
    /// For binary classification, shape is `(1, n_features)`.
    /// For multiclass, shape is `(n_classes, n_features)`.
    #[must_use]
    pub fn weight_matrix(&self) -> &Array2<F> {
        &self.weight_matrix
    }

    /// Returns a reference to the intercept vector (one per class).
    #[must_use]
    pub fn intercept_vec(&self) -> &Array1<F> {
        &self.intercept_vec
    }

    /// Returns whether this is a binary classification model.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        self.is_binary
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// For binary classification, returns an array of shape `(n_samples, 2)`.
    /// For multiclass, returns shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        let expected_features = self.weight_matrix.ncols();

        if n_features != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        if self.is_binary {
            let logits = x.dot(&self.coefficients) + self.intercept;
            let n_samples = x.nrows();
            let mut probs = Array2::<F>::zeros((n_samples, 2));
            for i in 0..n_samples {
                let p1 = sigmoid(logits[i]);
                probs[[i, 0]] = F::one() - p1;
                probs[[i, 1]] = p1;
            }
            Ok(probs)
        } else {
            let logits = x.dot(&self.weight_matrix.t()) + &self.intercept_vec;
            Ok(softmax_2d(&logits))
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedLogisticRegression<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Returns the class with the highest predicted probability.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let proba = self.predict_proba(x)?;
        let n_samples = proba.nrows();
        let n_classes = proba.ncols();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_prob = proba[[i, 0]];
            for c in 1..n_classes {
                if proba[[i, c]] > best_prob {
                    best_prob = proba[[i, c]];
                    best_class = c;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedLogisticRegression<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.coefficients
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedLogisticRegression<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for LogisticRegression<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        // Convert f64 labels to usize.
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedLogisticRegressionPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedLogisticRegressionPipeline<F>(FittedLogisticRegression<F>)
where
    F: Float + Send + Sync + 'static;

// Safety: the inner type is Send + Sync.
unsafe impl<F: Float + Send + Sync + 'static> Send for FittedLogisticRegressionPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedLogisticRegressionPipeline<F> {}

impl<F> FittedPipelineEstimator<F> for FittedLogisticRegressionPipeline<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0_f64), 0.5, epsilon = 1e-10);
        assert!(sigmoid(10.0_f64) > 0.99);
        assert!(sigmoid(-10.0_f64) < 0.01);
        // Check symmetry.
        assert_relative_eq!(sigmoid(1.0_f64) + sigmoid(-1.0_f64), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_classification() {
        // Linearly separable binary data.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, // class 0
                5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = LogisticRegression::<f64>::new()
            .with_c(1.0)
            .with_max_iter(1000);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();

        // At minimum, most samples should be correctly classified.
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_binary_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LogisticRegression::<f64>::new().with_c(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();

        // Probabilities should sum to 1.
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }

        // Class 0 should have higher probability for negative x.
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        // Class 1 should have higher probability for positive x.
        assert!(proba[[5, 1]] > proba[[5, 0]]);
    }

    #[test]
    fn test_multiclass_classification() {
        // Three linearly separable clusters.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, // class 0
                5.0, 0.0, 5.5, 0.0, 5.0, 0.5, // class 1
                0.0, 5.0, 0.5, 5.0, 0.0, 5.5, // class 2
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = LogisticRegression::<f64>::new()
            .with_c(10.0)
            .with_max_iter(2000);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_predict_proba() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = LogisticRegression::<f64>::new()
            .with_c(10.0)
            .with_max_iter(2000);
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // Probabilities should sum to 1 for each sample.
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = LogisticRegression::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = LogisticRegression::<f64>::new().with_c(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model_neg = LogisticRegression::<f64>::new().with_c(-1.0);
        assert!(model_neg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0]; // Only one class

        let model = LogisticRegression::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LogisticRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LogisticRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = LogisticRegression::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_no_intercept() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LogisticRegression::<f64>::new().with_fit_intercept(false);
        let fitted = model.fit(&x, &y).unwrap();
        assert_relative_eq!(fitted.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_2d() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let probs = softmax_2d(&logits);

        // Each row should sum to 1.
        assert_relative_eq!(probs.row(0).sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(probs.row(1).sum(), 1.0, epsilon = 1e-10);

        // Uniform logits should give uniform probs.
        assert_relative_eq!(probs[[1, 0]], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(probs[[1, 1]], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(probs[[1, 2]], 1.0 / 3.0, epsilon = 1e-10);
    }
}
