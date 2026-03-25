//! Quadratic Discriminant Analysis (QDA).
//!
//! This module provides [`QDA`], a classifier that models each class with its
//! own covariance matrix, yielding quadratic decision boundaries. Unlike
//! [`LDA`](crate::lda::LDA), which assumes a shared covariance matrix, QDA
//! fits a separate covariance per class.
//!
//! # Algorithm
//!
//! For each class `k`:
//! 1. Compute the class mean `mu_k` and covariance `Sigma_k`.
//! 2. Optionally regularize: `Sigma_k = (1 - reg) * Sigma_k + reg * I`.
//! 3. Compute the log-posterior:
//!    `delta_k(x) = -0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^{-1} (x - mu_k) + log(prior_k)`.
//! 4. Predict the class with the largest `delta_k`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::qda::QDA;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 1.0, 1.5, 1.2, 1.2, 0.8, 5.0, 5.0, 5.5, 4.8, 4.8, 5.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let qda = QDA::<f64>::new();
//! let fitted = qda.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Quadratic Discriminant Analysis configuration.
///
/// Holds hyperparameters. Calling [`Fit::fit`] computes per-class means
/// and covariance matrices and returns a [`FittedQDA`].
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct QDA<F> {
    /// Regularization parameter for covariance matrices.
    ///
    /// Blends each class covariance toward the identity:
    /// `Sigma_k = (1 - reg) * Sigma_k + reg * I`.
    /// Must be in `[0, 1]`. Default: `0.0`.
    pub reg_param: F,
}

impl<F: Float> QDA<F> {
    /// Create a new `QDA` with default settings.
    ///
    /// Default: `reg_param = 0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reg_param: F::zero(),
        }
    }

    /// Set the regularization parameter.
    #[must_use]
    pub fn with_reg_param(mut self, reg_param: F) -> Self {
        self.reg_param = reg_param;
        self
    }
}

impl<F: Float> Default for QDA<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-class model component for QDA.
#[derive(Debug, Clone)]
struct QDAClass<F> {
    /// Class mean, shape `(n_features,)`.
    mean: Array1<F>,
    /// Inverse of the (regularized) covariance matrix, shape `(n_features, n_features)`.
    cov_inv: Array2<F>,
    /// Log-determinant of the covariance matrix.
    log_det: F,
    /// Log-prior probability for this class.
    log_prior: F,
}

/// Fitted Quadratic Discriminant Analysis model.
///
/// Stores per-class means, covariance inverses, and priors. Implements
/// [`Predict`] to produce class labels.
#[derive(Debug, Clone)]
pub struct FittedQDA<F> {
    /// Per-class QDA models.
    class_models: Vec<QDAClass<F>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features seen during fitting.
    n_features: usize,
}

impl<F: Float> FittedQDA<F> {
    /// Returns the class means, one per class.
    #[must_use]
    pub fn means(&self) -> Vec<&Array1<F>> {
        self.class_models.iter().map(|m| &m.mean).collect()
    }
}

/// Compute the inverse and log-determinant of a symmetric positive-definite
/// matrix via Cholesky decomposition.
fn cholesky_inv_and_logdet<F: Float + 'static>(
    a: &Array2<F>,
) -> Result<(Array2<F>, F), FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    // Cholesky decomposition: A = L L^T.
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: "covariance matrix is not positive definite".into(),
                    });
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    // Log-determinant: log|A| = 2 * sum(log(diag(L))).
    let two = F::from(2.0).unwrap();
    let log_det = (0..n)
        .map(|i| l[[i, i]].ln())
        .fold(F::zero(), |a, b| a + b)
        * two;

    // Compute L^{-1} by forward substitution on each column of I.
    let mut l_inv = Array2::<F>::zeros((n, n));
    for col in 0..n {
        l_inv[[col, col]] = F::one() / l[[col, col]];
        for i in (col + 1)..n {
            let mut s = F::zero();
            for k in col..i {
                s = s + l[[i, k]] * l_inv[[k, col]];
            }
            l_inv[[i, col]] = -s / l[[i, i]];
        }
    }

    // A^{-1} = L^{-T} L^{-1}.
    let a_inv = l_inv.t().dot(&l_inv);

    Ok((a_inv, log_det))
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for QDA<F>
{
    type Fitted = FittedQDA<F>;
    type Error = FerroError;

    /// Fit the QDA model by computing per-class means and covariances.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 distinct classes
    ///   or a class has too few samples.
    /// - [`FerroError::InvalidParameter`] — `reg_param` not in `[0, 1]`.
    /// - [`FerroError::NumericalInstability`] — covariance matrix is singular.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedQDA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.reg_param < F::zero() || self.reg_param > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "reg_param".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "QDA requires at least 2 distinct classes".into(),
            });
        }

        let n_f = F::from(n_samples).unwrap();
        let mut class_models = Vec::with_capacity(classes.len());

        for &cls in &classes {
            // Extract samples for this class.
            let indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|&(_, label)| *label == cls)
                .map(|(i, _)| i)
                .collect();

            let n_k = indices.len();
            if n_k < 2 {
                return Err(FerroError::InsufficientSamples {
                    required: 2,
                    actual: n_k,
                    context: format!("class {cls} needs at least 2 samples for QDA"),
                });
            }

            let n_k_f = F::from(n_k).unwrap();

            // Compute class mean.
            let mut mean = Array1::<F>::zeros(n_features);
            for &i in &indices {
                for j in 0..n_features {
                    mean[j] = mean[j] + x[[i, j]];
                }
            }
            mean.mapv_inplace(|v| v / n_k_f);

            // Compute class covariance.
            let mut cov = Array2::<F>::zeros((n_features, n_features));
            for &i in &indices {
                let diff: Array1<F> = x.row(i).to_owned() - &mean;
                for r in 0..n_features {
                    for c in 0..n_features {
                        cov[[r, c]] = cov[[r, c]] + diff[r] * diff[c];
                    }
                }
            }
            // Use unbiased estimator: divide by (n_k - 1).
            let divisor = F::from(n_k - 1).unwrap();
            cov.mapv_inplace(|v| v / divisor);

            // Regularize: Sigma_k = (1 - reg) * Sigma_k + reg * I.
            if self.reg_param > F::zero() {
                let one_minus = F::one() - self.reg_param;
                for r in 0..n_features {
                    for c in 0..n_features {
                        cov[[r, c]] = cov[[r, c]] * one_minus;
                    }
                    cov[[r, r]] = cov[[r, r]] + self.reg_param;
                }
            }

            // Compute inverse and log-determinant.
            let (cov_inv, log_det) = cholesky_inv_and_logdet(&cov)?;

            let log_prior = (n_k_f / n_f).ln();

            class_models.push(QDAClass {
                mean,
                cov_inv,
                log_det,
                log_prior,
            });
        }

        Ok(FittedQDA {
            class_models,
            classes,
            n_features,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedQDA<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Selects the class with the largest log-posterior for each sample.
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
        let half = F::from(0.5).unwrap();

        for i in 0..n_samples {
            let xi = x.row(i);
            let mut best_class = 0;
            let mut best_score = F::neg_infinity();

            for (c, model) in self.class_models.iter().enumerate() {
                let diff: Array1<F> = xi.to_owned() - &model.mean;
                // Mahalanobis: diff^T * cov_inv * diff
                let mahal = diff.dot(&model.cov_inv.dot(&diff));
                let score = -half * model.log_det - half * mahal + model.log_prior;

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedQDA<F> {
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
        let m = QDA::<f64>::new();
        assert!(m.reg_param == 0.0);
    }

    #[test]
    fn test_builder() {
        let m = QDA::<f64>::new().with_reg_param(0.5);
        assert!(m.reg_param == 0.5);
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

        let model = QDA::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_classification() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5,
                10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 10.5, 0.5,
                0.0, 10.0, 0.5, 10.0, 0.0, 10.5, 0.5, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let model = QDA::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 10, "expected at least 10 correct, got {correct}");
    }

    #[test]
    fn test_regularization() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // With regularization should still work.
        let model = QDA::<f64>::new().with_reg_param(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_reg_param() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = QDA::<f64>::new().with_reg_param(-0.1);
        assert!(model.fit(&x, &y).is_err());

        let model2 = QDA::<f64>::new().with_reg_param(1.5);
        assert!(model2.fit(&x, &y).is_err());
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let fitted = QDA::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_classes() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0,
                8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let fitted = QDA::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_means() {
        let x = Array2::from_shape_vec(
            (4, 1),
            vec![1.0, 2.0, 5.0, 6.0],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let fitted = QDA::<f64>::new().with_reg_param(0.1).fit(&x, &y).unwrap();
        let means = fitted.means();
        assert_eq!(means.len(), 2);
    }

    #[test]
    fn test_class_with_too_few_samples() {
        let x = Array2::from_shape_vec(
            (3, 1),
            vec![1.0, 5.0, 6.0],
        )
        .unwrap();
        let y = array![0, 1, 1]; // class 0 has only 1 sample

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }
}
