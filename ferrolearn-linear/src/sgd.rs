//! Stochastic Gradient Descent (SGD) linear models.
//!
//! This module provides [`SGDClassifier`] and [`SGDRegressor`], two linear
//! models trained using stochastic gradient descent. Both support online /
//! streaming learning via the [`PartialFit`] trait and a range of configurable
//! loss functions and learning-rate schedules.
//!
//! # Classifier
//!
//! ```
//! use ferrolearn_linear::sgd::{SGDClassifier, ClassifierLoss};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
//!     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let model = SGDClassifier::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # Regressor
//!
//! ```
//! use ferrolearn_linear::sgd::{SGDRegressor, RegressorLoss};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let model = SGDRegressor::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, PartialFit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

/// A loss function for SGD optimization.
///
/// Provides the loss value and its gradient with respect to the prediction.
pub trait Loss<F: Float>: Clone + Send + Sync {
    /// Compute the loss for a single sample.
    fn loss(&self, y_true: F, y_pred: F) -> F;

    /// Compute the gradient of the loss with respect to `y_pred`.
    fn gradient(&self, y_true: F, y_pred: F) -> F;
}

/// Hinge loss for linear SVM-style classification.
///
/// `L(y, p) = max(0, 1 - y * p)` where `y in {-1, +1}`.
#[derive(Debug, Clone, Copy)]
pub struct Hinge;

impl<F: Float> Loss<F> for Hinge {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let margin = y_true * y_pred;
        if margin < F::one() {
            F::one() - margin
        } else {
            F::zero()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let margin = y_true * y_pred;
        if margin < F::one() {
            -y_true
        } else {
            F::zero()
        }
    }
}

/// Log loss (logistic regression / cross-entropy).
///
/// `L(y, p) = log(1 + exp(-y * p))` where `y in {-1, +1}`.
#[derive(Debug, Clone, Copy)]
pub struct LogLoss;

impl<F: Float> Loss<F> for LogLoss {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z > F::from(18.0).unwrap() {
            (-z).exp()
        } else if z < F::from(-18.0).unwrap() {
            -z
        } else {
            (F::one() + (-z).exp()).ln()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        let exp_nz = if z > F::from(18.0).unwrap() {
            (-z).exp()
        } else if z < F::from(-18.0).unwrap() {
            F::from(1e18).unwrap()
        } else {
            (-z).exp()
        };
        -y_true * exp_nz / (F::one() + exp_nz)
    }
}

/// Squared error loss for regression.
///
/// `L(y, p) = 0.5 * (y - p)^2`.
#[derive(Debug, Clone, Copy)]
pub struct SquaredError;

impl<F: Float> Loss<F> for SquaredError {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = y_true - y_pred;
        F::from(0.5).unwrap() * diff * diff
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        y_pred - y_true
    }
}

/// Modified Huber loss for classification.
///
/// Smooth approximation to hinge with quadratic behaviour near the margin:
///
/// ```text
/// L(y, p) = max(0, 1 - y*p)^2   if y*p >= -1
///         = -4 * y * p            otherwise
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ModifiedHuber;

impl<F: Float> Loss<F> for ModifiedHuber {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z >= -F::one() {
            let margin = F::one() - z;
            if margin > F::zero() {
                margin * margin
            } else {
                F::zero()
            }
        } else {
            -F::from(4.0).unwrap() * z
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let z = y_true * y_pred;
        if z >= -F::one() {
            if z < F::one() {
                F::from(-2.0).unwrap() * y_true * (F::one() - z)
            } else {
                F::zero()
            }
        } else {
            -F::from(4.0).unwrap() * y_true
        }
    }
}

/// Huber loss for robust regression.
///
/// `L(y, p) = 0.5 * (y - p)^2` if `|y - p| <= epsilon`, else
/// `epsilon * (|y - p| - 0.5 * epsilon)`.
#[derive(Debug, Clone, Copy)]
pub struct Huber<F> {
    /// Threshold parameter for switching from quadratic to linear loss.
    pub epsilon: F,
}

impl<F: Float + Send + Sync> Loss<F> for Huber<F> {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = y_true - y_pred;
        let abs_diff = diff.abs();
        if abs_diff <= self.epsilon {
            F::from(0.5).unwrap() * diff * diff
        } else {
            self.epsilon * (abs_diff - F::from(0.5).unwrap() * self.epsilon)
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let diff = y_pred - y_true;
        let abs_diff = diff.abs();
        if abs_diff <= self.epsilon {
            diff
        } else if diff > F::zero() {
            self.epsilon
        } else {
            -self.epsilon
        }
    }
}

/// Epsilon-insensitive loss for support vector regression.
///
/// `L(y, p) = max(0, |y - p| - epsilon)`.
#[derive(Debug, Clone, Copy)]
pub struct EpsilonInsensitive<F> {
    /// Insensitivity margin.
    pub epsilon: F,
}

impl<F: Float + Send + Sync> Loss<F> for EpsilonInsensitive<F> {
    fn loss(&self, y_true: F, y_pred: F) -> F {
        let diff = (y_true - y_pred).abs();
        if diff > self.epsilon {
            diff - self.epsilon
        } else {
            F::zero()
        }
    }

    fn gradient(&self, y_true: F, y_pred: F) -> F {
        let diff = y_pred - y_true;
        if diff > self.epsilon {
            F::one()
        } else if diff < -self.epsilon {
            -F::one()
        } else {
            F::zero()
        }
    }
}

// ---------------------------------------------------------------------------
// Learning rate schedules
// ---------------------------------------------------------------------------

/// Learning rate schedule for SGD.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule<F> {
    /// Fixed learning rate `eta0` throughout training.
    Constant,
    /// Optimal schedule: `eta = 1 / (alpha * t)`.
    Optimal,
    /// Inverse scaling: `eta = eta0 / t^power_t`.
    InvScaling,
    /// Adaptive: starts at `eta0`, halved when loss fails to decrease for
    /// 5 consecutive epochs. Stops when `eta < 1e-6`.
    Adaptive,
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<F>),
}

/// Compute the learning rate for a given step.
fn compute_lr<F: Float>(
    schedule: &LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    power_t: F,
    t: usize,
) -> F {
    let t_f = F::from(t.max(1)).unwrap();
    match schedule {
        LearningRateSchedule::Constant => eta0,
        LearningRateSchedule::Optimal => F::one() / (alpha * t_f),
        LearningRateSchedule::InvScaling => eta0 / t_f.powf(power_t),
        LearningRateSchedule::Adaptive => eta0,
        LearningRateSchedule::_Phantom(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Classifier loss enum
// ---------------------------------------------------------------------------

/// Available loss functions for [`SGDClassifier`].
#[derive(Debug, Clone, Copy)]
pub enum ClassifierLoss {
    /// Hinge loss (linear SVM).
    Hinge,
    /// Log loss (logistic regression).
    Log,
    /// Squared error loss.
    SquaredError,
    /// Modified Huber loss.
    ModifiedHuber,
}

/// Available loss functions for [`SGDRegressor`].
#[derive(Debug, Clone, Copy)]
pub enum RegressorLoss<F> {
    /// Squared error loss (default).
    SquaredError,
    /// Huber loss with the given epsilon.
    Huber(F),
    /// Epsilon-insensitive loss with the given epsilon.
    EpsilonInsensitive(F),
}

// ---------------------------------------------------------------------------
// SGDClassifier
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent classifier.
///
/// Supports binary classification via a decision boundary and multiclass
/// classification via one-vs-all decomposition.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::sgd::SGDClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
///     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let clf = SGDClassifier::<f64>::new();
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SGDClassifier<F> {
    /// The loss function to use.
    pub loss: ClassifierLoss,
    /// The learning rate schedule.
    pub learning_rate: LearningRateSchedule<F>,
    /// Initial learning rate.
    pub eta0: F,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of passes over the training data.
    pub max_iter: usize,
    /// Convergence tolerance. Training stops when the loss improvement
    /// is below this threshold.
    pub tol: F,
    /// Optional random seed for sample shuffling.
    pub random_state: Option<u64>,
    /// Power parameter for inverse scaling schedule.
    pub power_t: F,
}

impl<F: Float> SGDClassifier<F> {
    /// Create a new `SGDClassifier` with default settings.
    ///
    /// Defaults: `loss = Hinge`, `learning_rate = InvScaling`,
    /// `eta0 = 0.01`, `alpha = 0.0001`, `max_iter = 1000`,
    /// `tol = 1e-3`, `power_t = 0.25`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss: ClassifierLoss::Hinge,
            learning_rate: LearningRateSchedule::InvScaling,
            eta0: F::from(0.01).unwrap(),
            alpha: F::from(0.0001).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-3).unwrap(),
            random_state: None,
            power_t: F::from(0.25).unwrap(),
        }
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: ClassifierLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the learning rate schedule.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: LearningRateSchedule<F>) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_eta0(mut self, eta0: F) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of epochs.
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

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the power parameter for inverse scaling.
    #[must_use]
    pub fn with_power_t(mut self, power_t: F) -> Self {
        self.power_t = power_t;
        self
    }
}

impl<F: Float> Default for SGDClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract hyperparameter bundle from an `SGDClassifier`.
fn clf_hyper<F: Float>(clf: &SGDClassifier<F>) -> SGDHyper<F> {
    SGDHyper {
        learning_rate: clf.learning_rate,
        eta0: clf.eta0,
        alpha: clf.alpha,
        max_iter: clf.max_iter,
        tol: clf.tol,
        random_state: clf.random_state,
        power_t: clf.power_t,
    }
}

/// Internal hyperparameter bundle shared between Fit and PartialFit paths.
#[derive(Debug, Clone)]
struct SGDHyper<F> {
    learning_rate: LearningRateSchedule<F>,
    eta0: F,
    alpha: F,
    max_iter: usize,
    tol: F,
    random_state: Option<u64>,
    power_t: F,
}

/// Train a single binary classifier via SGD, updating `weights` and
/// `intercept` in place. `y_binary` must be in `{-1, +1}`.
///
/// Returns the cumulative loss and the step counter after training.
fn train_binary_sgd<F, L>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize)
where
    F: Float + ScalarOperand + Send + Sync + 'static,
    L: Loss<F>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut t = initial_t;
    let mut prev_loss = F::infinity();
    let mut current_eta = hyper.eta0;
    let mut no_improve_count: usize = 0;
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Build the RNG for shuffling.
    let mut rng = match hyper.random_state {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_os_rng(),
    };

    let mut total_loss = F::zero();

    for _epoch in 0..hyper.max_iter {
        indices.shuffle(&mut rng);
        let mut epoch_loss = F::zero();

        for &i in &indices {
            t += 1;

            let eta = match hyper.learning_rate {
                LearningRateSchedule::Adaptive => current_eta,
                _ => compute_lr(
                    &hyper.learning_rate,
                    hyper.eta0,
                    hyper.alpha,
                    hyper.power_t,
                    t,
                ),
            };

            // Compute prediction: w^T x_i + b.
            let mut y_pred = *intercept;
            let xi = x.row(i);
            for j in 0..n_features {
                y_pred = y_pred + weights[j] * xi[j];
            }

            let grad = loss_fn.gradient(y_binary[i], y_pred);
            epoch_loss = epoch_loss + loss_fn.loss(y_binary[i], y_pred);

            // Update weights with gradient + L2 regularization.
            for j in 0..n_features {
                weights[j] = weights[j] - eta * (grad * xi[j] + hyper.alpha * weights[j]);
            }
            *intercept = *intercept - eta * grad;
        }

        epoch_loss = epoch_loss / F::from(n_samples).unwrap();
        total_loss = epoch_loss;

        // Convergence check.
        if (prev_loss - epoch_loss).abs() < hyper.tol {
            break;
        }

        // Adaptive learning rate adjustment.
        if let LearningRateSchedule::Adaptive = hyper.learning_rate {
            if epoch_loss >= prev_loss {
                no_improve_count += 1;
                if no_improve_count >= 5 {
                    current_eta = current_eta / F::from(2.0).unwrap();
                    no_improve_count = 0;
                    if current_eta < F::from(1e-6).unwrap() {
                        break;
                    }
                }
            } else {
                no_improve_count = 0;
            }
        }

        prev_loss = epoch_loss;
    }

    (total_loss, t)
}

/// Fitted SGD classifier.
///
/// Holds the learned weight vectors and intercepts. For binary problems
/// there is a single weight vector; for multiclass problems there is one
/// per class (one-vs-all).
///
/// Implements [`Predict`] and [`PartialFit`] to support both inference and
/// online learning.
#[derive(Debug, Clone)]
pub struct FittedSGDClassifier<F> {
    /// Weight matrix: one row per binary sub-problem.
    /// Binary: shape `(1, n_features)`, multiclass: `(n_classes, n_features)`.
    weight_matrix: Vec<Array1<F>>,
    /// Intercept vector, one per sub-problem.
    intercepts: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features the model was trained on.
    n_features: usize,
    /// The loss function used during training.
    loss: ClassifierLoss,
    /// Hyperparameters for continued training via `partial_fit`.
    hyper: SGDHyper<F>,
    /// Global step counter across all partial_fit calls.
    t: usize,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for SGDClassifier<F>
{
    type Fitted = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Fit the SGD classifier on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sample counts.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes
    /// are present.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// not positive.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedSGDClassifier<F>, FerroError> {
        validate_clf_params(x, y, self.eta0, self.alpha)?;

        let n_features = x.ncols();
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SGDClassifier requires at least 2 distinct classes".into(),
            });
        }

        let hyper = clf_hyper(self);
        let loss_enum = self.loss;

        let (weight_matrix, intercepts, t) =
            fit_ova(x, y, &classes, n_features, &loss_enum, &hyper, 0)?;

        Ok(FittedSGDClassifier {
            weight_matrix,
            intercepts,
            classes,
            n_features,
            loss: loss_enum,
            hyper,
            t,
        })
    }
}

/// Validate classifier input shapes and parameters.
fn validate_clf_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<usize>,
    eta0: F,
    alpha: F,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "SGDClassifier requires at least one sample".into(),
        });
    }
    if eta0 <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "eta0".into(),
            reason: "must be positive".into(),
        });
    }
    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }
    Ok(())
}

/// Result type for one-vs-all training: (weight_matrix, intercepts, step_counter).
type OvaResult<F> = (Vec<Array1<F>>, Vec<F>, usize);

/// Train one-vs-all binary classifiers, returning per-class weights, intercepts,
/// and the cumulative step counter.
fn fit_ova<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    classes: &[usize],
    n_features: usize,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> Result<OvaResult<F>, FerroError> {
    let n_classes = classes.len();
    let mut weight_matrix: Vec<Array1<F>> = Vec::with_capacity(n_classes);
    let mut intercepts: Vec<F> = Vec::with_capacity(n_classes);
    let mut global_t = initial_t;

    if n_classes == 2 {
        // Single binary problem: class[0] -> -1, class[1] -> +1.
        let y_binary: Array1<F> = y.mapv(|label| {
            if label == classes[1] {
                F::one()
            } else {
                -F::one()
            }
        });
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();
        let (_, t) =
            dispatch_train_binary(x, &y_binary, &mut w, &mut b, loss_enum, hyper, global_t);
        global_t = t;
        weight_matrix.push(w);
        intercepts.push(b);
    } else {
        // One-vs-all: one binary problem per class.
        for &cls in classes {
            let y_binary: Array1<F> =
                y.mapv(|label| if label == cls { F::one() } else { -F::one() });
            let mut w = Array1::<F>::zeros(n_features);
            let mut b = F::zero();
            let (_, t) =
                dispatch_train_binary(x, &y_binary, &mut w, &mut b, loss_enum, hyper, global_t);
            global_t = t;
            weight_matrix.push(w);
            intercepts.push(b);
        }
    }

    Ok((weight_matrix, intercepts, global_t))
}

/// Train one-vs-all using existing weight vectors (for partial_fit).
#[allow(clippy::too_many_arguments)]
fn partial_fit_ova<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    classes: &[usize],
    weight_matrix: &mut [Array1<F>],
    intercepts: &mut [F],
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> usize {
    let n_classes = classes.len();
    let mut global_t = initial_t;

    if n_classes == 2 {
        let y_binary: Array1<F> = y.mapv(|label| {
            if label == classes[1] {
                F::one()
            } else {
                -F::one()
            }
        });
        let (_, t) = dispatch_train_binary(
            x,
            &y_binary,
            &mut weight_matrix[0],
            &mut intercepts[0],
            loss_enum,
            hyper,
            global_t,
        );
        global_t = t;
    } else {
        for (idx, &cls) in classes.iter().enumerate() {
            let y_binary: Array1<F> =
                y.mapv(|label| if label == cls { F::one() } else { -F::one() });
            let (_, t) = dispatch_train_binary(
                x,
                &y_binary,
                &mut weight_matrix[idx],
                &mut intercepts[idx],
                loss_enum,
                hyper,
                global_t,
            );
            global_t = t;
        }
    }

    global_t
}

/// Dispatch to the appropriate typed loss training function.
fn dispatch_train_binary<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y_binary: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &ClassifierLoss,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize) {
    match loss_enum {
        ClassifierLoss::Hinge => train_binary_sgd(x, y_binary, w, b, &Hinge, hyper, initial_t),
        ClassifierLoss::Log => train_binary_sgd(x, y_binary, w, b, &LogLoss, hyper, initial_t),
        ClassifierLoss::SquaredError => {
            train_binary_sgd(x, y_binary, w, b, &SquaredError, hyper, initial_t)
        }
        ClassifierLoss::ModifiedHuber => {
            train_binary_sgd(x, y_binary, w, b, &ModifiedHuber, hyper, initial_t)
        }
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedSGDClassifier<F>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// For binary classification, uses `sign(w^T x + b)`.
    /// For multiclass, returns the class whose one-vs-all score is highest.
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

        if self.classes.len() == 2 {
            // Binary: single weight vector.
            let scores = x.dot(&self.weight_matrix[0]) + self.intercepts[0];
            for i in 0..n_samples {
                predictions[i] = if scores[i] >= F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: one-vs-all, pick highest score.
            for i in 0..n_samples {
                let xi = x.row(i);
                let mut best_class = 0;
                let mut best_score = F::neg_infinity();
                for (c, w) in self.weight_matrix.iter().enumerate() {
                    let score = xi.dot(w) + self.intercepts[c];
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<usize>>
    for FittedSGDClassifier<F>
{
    type FitResult = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Incrementally train the classifier on a new batch of data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sizes or `x` has the wrong number of features.
    fn partial_fit(
        mut self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedSGDClassifier<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        // Use a single-epoch hyper for partial_fit.
        let mut hyper = self.hyper.clone();
        hyper.max_iter = 1;

        let t = partial_fit_ova(
            x,
            y,
            &self.classes,
            &mut self.weight_matrix,
            &mut self.intercepts,
            &self.loss,
            &hyper,
            self.t,
        );
        self.t = t;

        Ok(self)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<usize>>
    for SGDClassifier<F>
{
    type FitResult = FittedSGDClassifier<F>;
    type Error = FerroError;

    /// Initial call to `partial_fit` on an unfitted classifier.
    ///
    /// Equivalent to `fit` but with a single epoch, enabling subsequent
    /// incremental calls.
    ///
    /// # Errors
    ///
    /// Same as [`Fit::fit`].
    fn partial_fit(
        self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedSGDClassifier<F>, FerroError> {
        validate_clf_params(x, y, self.eta0, self.alpha)?;

        let n_features = x.ncols();
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SGDClassifier requires at least 2 distinct classes".into(),
            });
        }

        let mut hyper = clf_hyper(&self);
        hyper.max_iter = 1;
        let loss_enum = self.loss;

        let (weight_matrix, intercepts, t) =
            fit_ova(x, y, &classes, n_features, &loss_enum, &hyper, 0)?;

        Ok(FittedSGDClassifier {
            weight_matrix,
            intercepts,
            classes,
            n_features,
            loss: loss_enum,
            hyper: clf_hyper(&self),
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedSGDClassifier<F>
{
    /// Returns the coefficient vector for the first (or only) binary classifier.
    fn coefficients(&self) -> &Array1<F> {
        &self.weight_matrix[0]
    }

    /// Returns the intercept for the first (or only) binary classifier.
    fn intercept(&self) -> F {
        self.intercepts[0]
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for SGDClassifier<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedSGDClassifierPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedSGDClassifierPipeline<F>(FittedSGDClassifier<F>)
where
    F: Float + Send + Sync + 'static;

// Safety: inner type fields are Send + Sync.
unsafe impl<F> Send for FittedSGDClassifierPipeline<F> where F: Float + Send + Sync + 'static {}
unsafe impl<F> Sync for FittedSGDClassifierPipeline<F> where F: Float + Send + Sync + 'static {}

impl<F> FittedPipelineEstimator<F> for FittedSGDClassifierPipeline<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// SGDRegressor
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent regressor.
///
/// Supports several loss functions for regression, trained using stochastic
/// gradient descent with configurable learning rate schedules.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::sgd::SGDRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![2.0, 4.0, 6.0, 8.0];
///
/// let model = SGDRegressor::<f64>::new();
/// let fitted = model.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SGDRegressor<F> {
    /// The loss function to use.
    pub loss: RegressorLoss<F>,
    /// The learning rate schedule.
    pub learning_rate: LearningRateSchedule<F>,
    /// Initial learning rate.
    pub eta0: F,
    /// L2 regularization strength.
    pub alpha: F,
    /// Maximum number of passes over the training data.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Optional random seed for sample shuffling.
    pub random_state: Option<u64>,
    /// Power parameter for inverse scaling schedule.
    pub power_t: F,
}

impl<F: Float> SGDRegressor<F> {
    /// Create a new `SGDRegressor` with default settings.
    ///
    /// Defaults: `loss = SquaredError`, `learning_rate = InvScaling`,
    /// `eta0 = 0.01`, `alpha = 0.0001`, `max_iter = 1000`,
    /// `tol = 1e-3`, `power_t = 0.25`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss: RegressorLoss::SquaredError,
            learning_rate: LearningRateSchedule::InvScaling,
            eta0: F::from(0.01).unwrap(),
            alpha: F::from(0.0001).unwrap(),
            max_iter: 1000,
            tol: F::from(1e-3).unwrap(),
            random_state: None,
            power_t: F::from(0.25).unwrap(),
        }
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: RegressorLoss<F>) -> Self {
        self.loss = loss;
        self
    }

    /// Set the learning rate schedule.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: LearningRateSchedule<F>) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_eta0(mut self, eta0: F) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of epochs.
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

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the power parameter for inverse scaling.
    #[must_use]
    pub fn with_power_t(mut self, power_t: F) -> Self {
        self.power_t = power_t;
        self
    }
}

impl<F: Float> Default for SGDRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract hyperparameter bundle from an `SGDRegressor`.
fn reg_hyper<F: Float>(reg: &SGDRegressor<F>) -> SGDHyper<F> {
    SGDHyper {
        learning_rate: reg.learning_rate,
        eta0: reg.eta0,
        alpha: reg.alpha,
        max_iter: reg.max_iter,
        tol: reg.tol,
        random_state: reg.random_state,
        power_t: reg.power_t,
    }
}

/// Train a single regressor via SGD, updating `weights` and `intercept`
/// in place. Returns the final loss and step counter.
fn train_regressor_sgd<F, L>(
    x: &Array2<F>,
    y: &Array1<F>,
    weights: &mut Array1<F>,
    intercept: &mut F,
    loss_fn: &L,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize)
where
    F: Float + ScalarOperand + Send + Sync + 'static,
    L: Loss<F>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut t = initial_t;
    let mut prev_loss = F::infinity();
    let mut current_eta = hyper.eta0;
    let mut no_improve_count: usize = 0;
    let mut indices: Vec<usize> = (0..n_samples).collect();

    let mut rng = match hyper.random_state {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_os_rng(),
    };

    let mut total_loss = F::zero();

    for _epoch in 0..hyper.max_iter {
        indices.shuffle(&mut rng);
        let mut epoch_loss = F::zero();

        for &i in &indices {
            t += 1;

            let eta = match hyper.learning_rate {
                LearningRateSchedule::Adaptive => current_eta,
                _ => compute_lr(
                    &hyper.learning_rate,
                    hyper.eta0,
                    hyper.alpha,
                    hyper.power_t,
                    t,
                ),
            };

            let xi = x.row(i);
            let mut y_pred = *intercept;
            for j in 0..n_features {
                y_pred = y_pred + weights[j] * xi[j];
            }

            let grad = loss_fn.gradient(y[i], y_pred);
            epoch_loss = epoch_loss + loss_fn.loss(y[i], y_pred);

            for j in 0..n_features {
                weights[j] = weights[j] - eta * (grad * xi[j] + hyper.alpha * weights[j]);
            }
            *intercept = *intercept - eta * grad;
        }

        epoch_loss = epoch_loss / F::from(n_samples).unwrap();
        total_loss = epoch_loss;

        if (prev_loss - epoch_loss).abs() < hyper.tol {
            break;
        }

        if let LearningRateSchedule::Adaptive = hyper.learning_rate {
            if epoch_loss >= prev_loss {
                no_improve_count += 1;
                if no_improve_count >= 5 {
                    current_eta = current_eta / F::from(2.0).unwrap();
                    no_improve_count = 0;
                    if current_eta < F::from(1e-6).unwrap() {
                        break;
                    }
                }
            } else {
                no_improve_count = 0;
            }
        }

        prev_loss = epoch_loss;
    }

    (total_loss, t)
}

/// Dispatch regressor training to the appropriate typed loss function.
fn dispatch_train_regressor<F: Float + Send + Sync + ScalarOperand + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    w: &mut Array1<F>,
    b: &mut F,
    loss_enum: &RegressorLoss<F>,
    hyper: &SGDHyper<F>,
    initial_t: usize,
) -> (F, usize) {
    match loss_enum {
        RegressorLoss::SquaredError => {
            train_regressor_sgd(x, y, w, b, &SquaredError, hyper, initial_t)
        }
        RegressorLoss::Huber(eps) => {
            train_regressor_sgd(x, y, w, b, &Huber { epsilon: *eps }, hyper, initial_t)
        }
        RegressorLoss::EpsilonInsensitive(eps) => train_regressor_sgd(
            x,
            y,
            w,
            b,
            &EpsilonInsensitive { epsilon: *eps },
            hyper,
            initial_t,
        ),
    }
}

/// Fitted SGD regressor.
///
/// Holds the learned weight vector and intercept. Implements [`Predict`]
/// and [`PartialFit`] to support both inference and online learning.
#[derive(Debug, Clone)]
pub struct FittedSGDRegressor<F> {
    /// Learned weight vector (one per feature).
    weights: Array1<F>,
    /// Learned intercept (bias) term.
    intercept: F,
    /// Number of features the model was trained on.
    n_features: usize,
    /// The loss function used during training.
    loss: RegressorLoss<F>,
    /// Hyperparameters for continued training.
    hyper: SGDHyper<F>,
    /// Global step counter.
    t: usize,
}

/// Validate regressor input shapes and parameters.
fn validate_reg_params<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    eta0: F,
    alpha: F,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if n_samples != y.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "y length must match number of samples in X".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "SGDRegressor requires at least one sample".into(),
        });
    }
    if eta0 <= F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "eta0".into(),
            reason: "must be positive".into(),
        });
    }
    if alpha < F::zero() {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: "must be non-negative".into(),
        });
    }
    Ok(())
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<F>>
    for SGDRegressor<F>
{
    type Fitted = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Fit the SGD regressor on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `eta0` or `alpha` are
    /// invalid.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedSGDRegressor<F>, FerroError> {
        validate_reg_params(x, y, self.eta0, self.alpha)?;

        let n_features = x.ncols();
        let hyper = reg_hyper(self);
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();

        let (_, t) = dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0);

        Ok(FittedSGDRegressor {
            weights: w,
            intercept: b,
            n_features,
            loss: self.loss,
            hyper,
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedSGDRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
    ///
    /// Computes `X @ weights + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let preds = x.dot(&self.weights) + self.intercept;
        Ok(preds)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<F>>
    for FittedSGDRegressor<F>
{
    type FitResult = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Incrementally train the regressor on a new batch of data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have mismatched
    /// sizes or `x` has the wrong number of features.
    fn partial_fit(
        mut self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedSGDRegressor<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let mut hyper = self.hyper.clone();
        hyper.max_iter = 1;

        let (_, t) = dispatch_train_regressor(
            x,
            y,
            &mut self.weights,
            &mut self.intercept,
            &self.loss,
            &hyper,
            self.t,
        );
        self.t = t;

        Ok(self)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> PartialFit<Array2<F>, Array1<F>>
    for SGDRegressor<F>
{
    type FitResult = FittedSGDRegressor<F>;
    type Error = FerroError;

    /// Initial call to `partial_fit` on an unfitted regressor.
    ///
    /// Equivalent to `fit` but with a single epoch.
    ///
    /// # Errors
    ///
    /// Same as [`Fit::fit`].
    fn partial_fit(
        self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedSGDRegressor<F>, FerroError> {
        validate_reg_params(x, y, self.eta0, self.alpha)?;

        let n_features = x.ncols();
        let mut hyper = reg_hyper(&self);
        hyper.max_iter = 1;
        let mut w = Array1::<F>::zeros(n_features);
        let mut b = F::zero();

        let (_, t) = dispatch_train_regressor(x, y, &mut w, &mut b, &self.loss, &hyper, 0);

        Ok(FittedSGDRegressor {
            weights: w,
            intercept: b,
            n_features,
            loss: self.loss,
            hyper: reg_hyper(&self),
            t,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F>
    for FittedSGDRegressor<F>
{
    fn coefficients(&self) -> &Array1<F> {
        &self.weights
    }

    fn intercept(&self) -> F {
        self.intercept
    }
}

// Pipeline integration.
impl<F> PipelineEstimator<F> for SGDRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

impl<F> FittedPipelineEstimator<F> for FittedSGDRegressor<F>
where
    F: Float + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Loss function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hinge_loss_correct_side() {
        let h = Hinge;
        // y=1, pred=2 => margin=2 >= 1 => loss=0
        assert!((Loss::<f64>::loss(&h, 1.0, 2.0) - 0.0).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&h, 1.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_wrong_side() {
        let h = Hinge;
        // y=1, pred=-0.5 => margin=-0.5 < 1 => loss=1.5
        assert!((Loss::<f64>::loss(&h, 1.0, -0.5) - 1.5).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&h, 1.0, -0.5) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_zero_pred() {
        let l = LogLoss;
        // y=1, pred=0 => loss=log(1+exp(0))=log(2)
        let loss = Loss::<f64>::loss(&l, 1.0, 0.0);
        assert!((loss - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_large_correct() {
        let l = LogLoss;
        // y=1, pred=20 => very small loss
        let loss = Loss::<f64>::loss(&l, 1.0, 20.0);
        assert!(loss < 1e-5);
    }

    #[test]
    fn test_squared_error_loss() {
        let s = SquaredError;
        assert!((Loss::<f64>::loss(&s, 3.0, 1.0) - 2.0).abs() < 1e-10);
        assert!((Loss::<f64>::gradient(&s, 3.0, 1.0) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_modified_huber_loss() {
        let mh = ModifiedHuber;
        // y=1, pred=2 => z=2 >= 1 => loss=0
        assert!((Loss::<f64>::loss(&mh, 1.0, 2.0)).abs() < 1e-10);
        // y=1, pred=0.5 => z=0.5 => loss=(1-0.5)^2=0.25
        assert!((Loss::<f64>::loss(&mh, 1.0, 0.5) - 0.25).abs() < 1e-10);
        // y=1, pred=-2 => z=-2 < -1 => loss=-4*(-2)=8
        assert!((Loss::<f64>::loss(&mh, 1.0, -2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        let h = Huber { epsilon: 1.0_f64 };
        // |y - p| = 0.5 <= 1.0 => quadratic
        assert!((Loss::<f64>::loss(&h, 1.0, 0.5) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        let h = Huber { epsilon: 1.0_f64 };
        // |y - p| = 3 > 1 => linear: 1*(3 - 0.5) = 2.5
        assert!((Loss::<f64>::loss(&h, 3.0, 0.0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_insensitive_inside() {
        let ei = EpsilonInsensitive { epsilon: 0.1_f64 };
        // |y - p| = 0.05 <= 0.1 => loss=0
        assert!((Loss::<f64>::loss(&ei, 1.0, 0.95)).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_insensitive_outside() {
        let ei = EpsilonInsensitive { epsilon: 0.1_f64 };
        // |y - p| = 0.5 > 0.1 => loss=0.4
        assert!((Loss::<f64>::loss(&ei, 1.0, 0.5) - 0.4).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Learning rate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_lr() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Constant;
        assert!((compute_lr(&lr, 0.1, 0.01, 0.25, 100) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_lr() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Optimal;
        // eta = 1 / (alpha * t) = 1 / (0.01 * 10) = 10.0
        assert!((compute_lr(&lr, 0.1, 0.01, 0.25, 10) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_invscaling_lr() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::InvScaling;
        // eta = 0.1 / 10^0.5 = 0.1 / 3.162... ~= 0.0316...
        let result = compute_lr(&lr, 0.1, 0.01, 0.5, 10);
        let expected = 0.1 / 10.0_f64.sqrt();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_lr_returns_eta0() {
        let lr: LearningRateSchedule<f64> = LearningRateSchedule::Adaptive;
        assert!((compute_lr(&lr, 0.05, 0.01, 0.25, 100) - 0.05).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // SGDClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_classifier_binary() {
        // Well-separated clusters centered near origin for SGD stability.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                -2.0, -2.0, -1.5, -2.0, -2.0, -1.5, -1.5, -1.5, 2.0, 2.0, 1.5, 2.0, 2.0, 1.5, 1.5,
                1.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::Log)
            .with_random_state(42)
            .with_max_iter(1000)
            .with_eta0(0.01);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected >= 6 correct, got {correct}");
    }

    #[test]
    fn test_sgd_classifier_log_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::Log)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 4, "expected >= 4 correct, got {correct}");
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let clf = SGDClassifier::<f64>::new()
            .with_random_state(42)
            .with_max_iter(1000)
            .with_eta0(0.01);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 6,
            "expected >= 6 correct for multiclass, got {correct}"
        );
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_shape_mismatch_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_sgd_classifier_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_invalid_eta0() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_eta0(0.0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_invalid_alpha() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_alpha(-1.0);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_has_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_sgd_classifier_partial_fit() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.partial_fit(&x, &y).unwrap();
        let fitted = fitted.partial_fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_chain() {
        // Test the chaining pattern:
        // model.partial_fit(&b1, &y1)?.partial_fit(&b2, &y2)?.predict(&x)?
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y1 = array![0, 0, 1, 1];
        let x2 =
            Array2::from_shape_vec((4, 2), vec![0.5, 0.5, 1.5, 1.5, 7.5, 7.5, 8.5, 8.5]).unwrap();
        let y2 = array![0, 0, 1, 1];

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let preds = clf
            .partial_fit(&x1, &y1)
            .unwrap()
            .partial_fit(&x2, &y2)
            .unwrap()
            .predict(&x1)
            .unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_classifier_partial_fit_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.partial_fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y_bad = array![0, 1];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_classifier_modified_huber() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::ModifiedHuber)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_squared_error_loss() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_loss(ClassifierLoss::SquaredError)
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_pipeline() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let clf = SGDClassifier::<f64>::new().with_random_state(42);
        let fitted = clf.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_sgd_classifier_constant_lr() {
        let x = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let clf = SGDClassifier::<f64>::new()
            .with_learning_rate(LearningRateSchedule::Constant)
            .with_random_state(42)
            .with_max_iter(200);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_classifier_f32() {
        let x = Array2::from_shape_vec((4, 1), vec![-2.0f32, -1.0, 1.0, 2.0]).unwrap();
        let y = array![0_usize, 0, 1, 1];

        let clf = SGDClassifier::<f32>::new()
            .with_random_state(42)
            .with_max_iter(200);
        let fitted = clf.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    // -----------------------------------------------------------------------
    // SGDRegressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_regressor_basic() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let model = SGDRegressor::<f64>::new()
            .with_random_state(42)
            .with_max_iter(2000)
            .with_eta0(0.01)
            .with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check rough accuracy.
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (*p - actual).abs() < 2.0,
                "prediction {p} too far from {actual}"
            );
        }
    }

    #[test]
    fn test_sgd_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length
        let model = SGDRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_predict_shape_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_invalid_eta0() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let model = SGDRegressor::<f64>::new().with_eta0(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_regressor_has_coefficients() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_sgd_regressor_partial_fit() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.partial_fit(&x, &y).unwrap();
        let fitted = fitted.partial_fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_chain() {
        let x1 = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y1 = array![2.0, 4.0, 6.0];
        let x2 = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y2 = array![8.0, 10.0, 12.0];

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let preds = model
            .partial_fit(&x1, &y1)
            .unwrap()
            .partial_fit(&x2, &y2)
            .unwrap()
            .predict(&x1)
            .unwrap();
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_sgd_regressor_partial_fit_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = array![1.0, 2.0, 3.0];
        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.partial_fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let y_bad = array![1.0, 2.0];
        assert!(fitted.partial_fit(&x_bad, &y_bad).is_err());
    }

    #[test]
    fn test_sgd_regressor_huber_loss() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new()
            .with_loss(RegressorLoss::Huber(1.35))
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_epsilon_insensitive() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::<f64>::new()
            .with_loss(RegressorLoss::EpsilonInsensitive(0.1))
            .with_random_state(42)
            .with_max_iter(500);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_pipeline() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let model = SGDRegressor::<f64>::new().with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_sgd_regressor_f32() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);

        let model = SGDRegressor::<f32>::new().with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.predict(&x).unwrap().len(), 4);
    }

    #[test]
    fn test_sgd_regressor_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let model = SGDRegressor::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let clf = SGDClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgd_classifier_default() {
        let clf = SGDClassifier::<f64>::default();
        assert!(clf.eta0 > 0.0);
        assert!(clf.alpha >= 0.0);
    }

    #[test]
    fn test_sgd_regressor_default() {
        let model = SGDRegressor::<f64>::default();
        assert!(model.eta0 > 0.0);
        assert!(model.alpha >= 0.0);
    }
}
