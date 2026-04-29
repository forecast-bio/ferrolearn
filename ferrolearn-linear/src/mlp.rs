//! Multi-layer Perceptron (MLP) models.
//!
//! This module provides [`MLPClassifier`] and [`MLPRegressor`], feedforward
//! neural networks trained via mini-batch gradient descent with either SGD
//! (with momentum) or Adam optimization.
//!
//! # Classifier
//!
//! ```
//! use ferrolearn_linear::mlp::MLPClassifier;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
//!     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! let clf = MLPClassifier::<f64>::new();
//! let fitted = clf.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! # Regressor
//!
//! ```
//! use ferrolearn_linear::mlp::MLPRegressor;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0];
//!
//! let reg = MLPRegressor::<f64>::new();
//! let fitted = reg.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 4);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::SeedableRng;
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Activation function for hidden (and optionally output) layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit: `max(0, x)`.
    Relu,
    /// Hyperbolic tangent: `tanh(x)`.
    Tanh,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Logistic,
    /// Identity (pass-through): `x`.
    Identity,
}

/// Apply an activation function element-wise in-place.
fn activate_inplace<F: Float>(z: &mut Array2<F>, act: Activation) {
    match act {
        Activation::Relu => {
            z.mapv_inplace(|v| if v > F::zero() { v } else { F::zero() });
        }
        Activation::Tanh => {
            z.mapv_inplace(|v| v.tanh());
        }
        Activation::Logistic => {
            z.mapv_inplace(|v| stable_sigmoid(v));
        }
        Activation::Identity => {} // no-op
    }
}

/// Compute the derivative of the activation with respect to pre-activation `z`,
/// given the already-computed activation `a`. Returns `da/dz` element-wise.
fn activation_derivative<F: Float>(a: &Array2<F>, z: &Array2<F>, act: Activation) -> Array2<F> {
    match act {
        Activation::Relu => z.mapv(|v| if v > F::zero() { F::one() } else { F::zero() }),
        Activation::Tanh => a.mapv(|v| F::one() - v * v),
        Activation::Logistic => a.mapv(|v| v * (F::one() - v)),
        Activation::Identity => Array2::from_elem(a.dim(), F::one()),
    }
}

/// Numerically stable sigmoid function.
fn stable_sigmoid<F: Float>(z: F) -> F {
    if z >= F::zero() {
        F::one() / (F::one() + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (F::one() + ez)
    }
}

/// Row-wise softmax with numerical stability (subtract row max).
fn softmax_rows<F: Float>(logits: &Array2<F>) -> Array2<F> {
    let (n_rows, n_cols) = logits.dim();
    let mut probs = Array2::<F>::zeros((n_rows, n_cols));

    for i in 0..n_rows {
        let max_val = logits
            .row(i)
            .iter()
            .fold(F::neg_infinity(), |a, &b| a.max(b));
        let mut sum = F::zero();
        for j in 0..n_cols {
            let exp_val = (logits[[i, j]] - max_val).exp();
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

// ---------------------------------------------------------------------------
// Solver / optimizer
// ---------------------------------------------------------------------------

/// Optimization solver for MLP training.
#[derive(Debug, Clone, Copy)]
pub enum Solver<F> {
    /// Stochastic gradient descent with optional momentum.
    Sgd {
        /// Learning rate (step size).
        learning_rate: F,
        /// Momentum factor (0 = no momentum).
        momentum: F,
    },
    /// Adam optimizer (adaptive moment estimation).
    Adam {
        /// Learning rate.
        learning_rate: F,
        /// Exponential decay rate for the first moment estimate.
        beta1: F,
        /// Exponential decay rate for the second moment estimate.
        beta2: F,
        /// Small constant for numerical stability.
        epsilon: F,
    },
}

// ---------------------------------------------------------------------------
// Network layer storage
// ---------------------------------------------------------------------------

/// Parameters (weights + biases) for a single dense layer.
#[derive(Debug, Clone)]
struct LayerParams<F> {
    /// Weight matrix of shape `(fan_in, fan_out)`.
    weights: Array2<F>,
    /// Bias vector of shape `(fan_out,)`.
    biases: Array1<F>,
}

/// First and second moment estimates for Adam, per layer.
#[derive(Debug, Clone)]
struct AdamState<F> {
    m_w: Array2<F>,
    v_w: Array2<F>,
    m_b: Array1<F>,
    v_b: Array1<F>,
}

/// Velocity for SGD with momentum, per layer.
#[derive(Debug, Clone)]
struct SgdMomentumState<F> {
    vel_w: Array2<F>,
    vel_b: Array1<F>,
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Xavier (Glorot) uniform initialization.
fn xavier_init<F: Float>(fan_in: usize, fan_out: usize, rng: &mut rand::rngs::StdRng) -> Array2<F> {
    use rand::Rng;
    let limit_f64 = (6.0_f64 / (fan_in as f64 + fan_out as f64)).sqrt();
    let limit = F::from(limit_f64).unwrap();
    Array2::from_shape_fn((fan_in, fan_out), |_| {
        let u: f64 = rng.random_range(-limit_f64..limit_f64);
        F::from(u).unwrap_or(limit)
    })
}

// ---------------------------------------------------------------------------
// Forward / backward pass
// ---------------------------------------------------------------------------

/// Cached activations for backpropagation.
struct ForwardCache<F: Float> {
    /// Pre-activation values `z = x @ W + b` for each layer.
    z_values: Vec<Array2<F>>,
    /// Post-activation values `a = activation(z)` for each layer.
    a_values: Vec<Array2<F>>,
}

/// Run the forward pass through all layers.
///
/// `hidden_activation` is applied to all hidden layers.
/// `output_activation` is applied to the final layer.
fn forward<F: Float + 'static>(
    input: &Array2<F>,
    layers: &[LayerParams<F>],
    hidden_activation: Activation,
    output_activation: Activation,
) -> ForwardCache<F> {
    let n_layers = layers.len();
    let mut z_values = Vec::with_capacity(n_layers);
    let mut a_values = Vec::with_capacity(n_layers);

    let mut current = input.clone();

    for (i, layer) in layers.iter().enumerate() {
        // z = current @ W + b (broadcast b across rows)
        let mut z = current.dot(&layer.weights);
        for row_idx in 0..z.nrows() {
            for col_idx in 0..z.ncols() {
                z[[row_idx, col_idx]] = z[[row_idx, col_idx]] + layer.biases[col_idx];
            }
        }

        let act = if i == n_layers - 1 {
            output_activation
        } else {
            hidden_activation
        };

        let mut a = z.clone();
        // For softmax output, we handle it separately (not in-place activate).
        if i == n_layers - 1 && output_activation == Activation::Identity {
            // Identity for regression — a = z already.
        } else if i == n_layers - 1 && output_activation == Activation::Logistic {
            // For binary classification output, use sigmoid.
            activate_inplace(&mut a, Activation::Logistic);
        } else {
            activate_inplace(&mut a, act);
        }

        z_values.push(z);
        a_values.push(a.clone());
        current = a;
    }

    ForwardCache { z_values, a_values }
}

/// Forward pass returning only the final output (no cache needed).
fn forward_output<F: Float + 'static>(
    input: &Array2<F>,
    layers: &[LayerParams<F>],
    hidden_activation: Activation,
    output_activation: Activation,
    is_multiclass: bool,
) -> Array2<F> {
    let n_layers = layers.len();
    let mut current = input.clone();

    for (i, layer) in layers.iter().enumerate() {
        let mut z = current.dot(&layer.weights);
        for row_idx in 0..z.nrows() {
            for col_idx in 0..z.ncols() {
                z[[row_idx, col_idx]] = z[[row_idx, col_idx]] + layer.biases[col_idx];
            }
        }

        if i == n_layers - 1 {
            if is_multiclass {
                return softmax_rows(&z);
            }
            activate_inplace(&mut z, output_activation);
            return z;
        }
        activate_inplace(&mut z, hidden_activation);
        current = z;
    }

    current
}

/// Compute gradients via backpropagation.
///
/// `delta_output` is the gradient of the loss w.r.t. the output layer's
/// activations, shape `(batch_size, output_size)`.
///
/// Returns weight and bias gradients for each layer.
fn backward<F: Float + 'static>(
    input: &Array2<F>,
    layers: &[LayerParams<F>],
    cache: &ForwardCache<F>,
    delta_output: &Array2<F>,
    hidden_activation: Activation,
    alpha: F,
) -> Vec<(Array2<F>, Array1<F>)> {
    let n_layers = layers.len();
    let batch_f = F::from(input.nrows()).unwrap();
    let mut grads = vec![];

    let mut delta = delta_output.clone();

    for i in (0..n_layers).rev() {
        // The input to this layer.
        let layer_input = if i == 0 {
            input
        } else {
            &cache.a_values[i - 1]
        };

        // grad_w = layer_input^T @ delta / batch_size + alpha * W
        let grad_w = layer_input.t().dot(&delta).mapv(|v| v / batch_f)
            + &layers[i].weights.mapv(|v| v * alpha);
        // grad_b = mean(delta, axis=0)
        let grad_b = delta.sum_axis(Axis(0)).mapv(|v| v / batch_f);

        grads.push((grad_w, grad_b));

        // Propagate delta to previous layer (skip if this is the first layer).
        if i > 0 {
            let d_act = activation_derivative(
                &cache.a_values[i - 1],
                &cache.z_values[i - 1],
                hidden_activation,
            );
            delta = delta.dot(&layers[i].weights.t()) * &d_act;
        }
    }

    grads.reverse();
    grads
}

// ---------------------------------------------------------------------------
// Parameter update helpers
// ---------------------------------------------------------------------------

fn apply_sgd_update<F: Float>(
    layers: &mut [LayerParams<F>],
    grads: &[(Array2<F>, Array1<F>)],
    momentum_states: &mut [SgdMomentumState<F>],
    lr: F,
    momentum: F,
) {
    for (i, (grad_w, grad_b)) in grads.iter().enumerate() {
        // v = momentum * v - lr * grad
        momentum_states[i].vel_w =
            momentum_states[i].vel_w.mapv(|v| v * momentum) - grad_w.mapv(|g| g * lr);
        momentum_states[i].vel_b =
            momentum_states[i].vel_b.mapv(|v| v * momentum) - grad_b.mapv(|g| g * lr);

        layers[i].weights = &layers[i].weights + &momentum_states[i].vel_w;
        layers[i].biases = &layers[i].biases + &momentum_states[i].vel_b;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_adam_update<F: Float>(
    layers: &mut [LayerParams<F>],
    grads: &[(Array2<F>, Array1<F>)],
    adam_states: &mut [AdamState<F>],
    lr: F,
    beta1: F,
    beta2: F,
    epsilon: F,
    t: usize,
) {
    let t_f = F::from(t).unwrap();
    let bc1 = F::one() - beta1.powf(t_f); // bias correction for first moment
    let bc2 = F::one() - beta2.powf(t_f); // bias correction for second moment

    for (i, (grad_w, grad_b)) in grads.iter().enumerate() {
        // Update first moment: m = beta1 * m + (1 - beta1) * grad
        adam_states[i].m_w =
            adam_states[i].m_w.mapv(|v| v * beta1) + grad_w.mapv(|g| g * (F::one() - beta1));
        adam_states[i].m_b =
            adam_states[i].m_b.mapv(|v| v * beta1) + grad_b.mapv(|g| g * (F::one() - beta1));

        // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
        adam_states[i].v_w =
            adam_states[i].v_w.mapv(|v| v * beta2) + grad_w.mapv(|g| g * g * (F::one() - beta2));
        adam_states[i].v_b =
            adam_states[i].v_b.mapv(|v| v * beta2) + grad_b.mapv(|g| g * g * (F::one() - beta2));

        // Bias-corrected estimates.
        let m_hat_w = adam_states[i].m_w.mapv(|v| v / bc1);
        let m_hat_b = adam_states[i].m_b.mapv(|v| v / bc1);
        let v_hat_w = adam_states[i].v_w.mapv(|v| v / bc2);
        let v_hat_b = adam_states[i].v_b.mapv(|v| v / bc2);

        // Update weights: W -= lr * m_hat / (sqrt(v_hat) + eps)
        layers[i].weights =
            &layers[i].weights - &m_hat_w.mapv(|m| m * lr) / &v_hat_w.mapv(|v| v.sqrt() + epsilon);
        layers[i].biases =
            &layers[i].biases - &m_hat_b.mapv(|m| m * lr) / &v_hat_b.mapv(|v| v.sqrt() + epsilon);
    }
}

// ---------------------------------------------------------------------------
// MLPClassifier
// ---------------------------------------------------------------------------

/// Multi-layer Perceptron classifier.
///
/// A feedforward neural network that uses backpropagation for training.
/// Supports binary and multiclass classification with configurable
/// hidden layer sizes, activation functions, and optimizers.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::mlp::MLPClassifier;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 2.0, 3.0, 3.0, 1.0,
///     8.0, 7.0, 9.0, 8.0, 7.0, 9.0,
/// ]).unwrap();
/// let y = array![0, 0, 0, 1, 1, 1];
///
/// let clf = MLPClassifier::<f64>::new()
///     .with_hidden_layer_sizes(vec![10])
///     .with_max_iter(200);
/// let fitted = clf.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MLPClassifier<F> {
    /// Sizes of the hidden layers (e.g., `vec![100]` or `vec![100, 50]`).
    pub hidden_layer_sizes: Vec<usize>,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Optimization solver.
    pub solver: Solver<F>,
    /// Maximum number of training epochs.
    pub max_iter: usize,
    /// Convergence tolerance: training stops when the loss change is below
    /// this value for two consecutive epochs.
    pub tol: F,
    /// Mini-batch size. `None` means `min(200, n_samples)`.
    pub batch_size: Option<usize>,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Whether to use early stopping on a held-out validation set.
    pub early_stopping: bool,
    /// Fraction of training data to use as validation when `early_stopping`
    /// is enabled.
    pub validation_fraction: f64,
    /// L2 regularization strength.
    pub alpha: F,
}

impl<F: Float> MLPClassifier<F> {
    /// Create a new `MLPClassifier` with default settings.
    ///
    /// Defaults: `hidden_layer_sizes = [100]`, `activation = Relu`,
    /// `solver = Adam(lr=0.001)`, `max_iter = 200`, `tol = 1e-4`,
    /// `alpha = 0.0001`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            activation: Activation::Relu,
            solver: Solver::Adam {
                learning_rate: F::from(0.001).unwrap(),
                beta1: F::from(0.9).unwrap(),
                beta2: F::from(0.999).unwrap(),
                epsilon: F::from(1e-8).unwrap(),
            },
            max_iter: 200,
            tol: F::from(1e-4).unwrap(),
            batch_size: None,
            random_state: None,
            early_stopping: false,
            validation_fraction: 0.1,
            alpha: F::from(0.0001).unwrap(),
        }
    }

    /// Set the hidden layer sizes.
    #[must_use]
    pub fn with_hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }

    /// Set the activation function for hidden layers.
    #[must_use]
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set the optimization solver.
    #[must_use]
    pub fn with_solver(mut self, solver: Solver<F>) -> Self {
        self.solver = solver;
        self
    }

    /// Set the maximum number of training epochs.
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

    /// Set the mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable or disable early stopping.
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set the validation fraction for early stopping.
    #[must_use]
    pub fn with_validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> Default for MLPClassifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-layer perceptron classifier.
///
/// Holds the learned network weights, biases, and class labels. Supports
/// both `predict` (class labels) and `predict_proba` (class probabilities).
#[derive(Debug, Clone)]
pub struct FittedMLPClassifier<F> {
    /// Network layer parameters (weights + biases).
    layers: Vec<LayerParams<F>>,
    /// Hidden layer activation function.
    hidden_activation: Activation,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary classification problem.
    is_binary: bool,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for MLPClassifier<F>
{
    type Fitted = FittedMLPClassifier<F>;
    type Error = FerroError;

    /// Fit the MLP classifier on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InvalidParameter`] if hidden layer sizes are
    /// empty or contain zero.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes
    /// are present.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedMLPClassifier<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate inputs.
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
                context: "MLPClassifier requires at least one sample".into(),
            });
        }
        validate_hidden_layers(&self.hidden_layer_sizes)?;

        // Determine classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "MLPClassifier requires at least 2 distinct classes".into(),
            });
        }

        let n_classes = classes.len();
        let is_binary = n_classes == 2;
        let output_size = if is_binary { 1 } else { n_classes };

        // Build one-hot (or binary label) target matrix.
        let y_target = build_classification_target(y, &classes, is_binary);

        // Output activation.
        let output_activation = if is_binary {
            Activation::Logistic
        } else {
            // Softmax is handled specially, we use Identity as a placeholder for
            // the layer itself and apply softmax in the loss computation.
            Activation::Identity
        };

        // Build layer sizes: [n_features, hidden..., output_size].
        let layer_sizes = build_layer_sizes(n_features, &self.hidden_layer_sizes, output_size);

        // Initialize weights.
        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        let mut layers = init_layers::<F>(&layer_sizes, &mut rng);

        // Train.
        let is_multiclass = !is_binary;
        train_network(
            x,
            &y_target,
            &mut layers,
            self.activation,
            output_activation,
            is_multiclass,
            &self.solver,
            self.alpha,
            self.max_iter,
            self.tol,
            self.batch_size,
            self.early_stopping,
            self.validation_fraction,
            &mut rng,
            true, // is_classifier
        )?;

        Ok(FittedMLPClassifier {
            layers,
            hidden_activation: self.activation,
            classes,
            is_binary,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> FittedMLPClassifier<F> {
    /// Predict class probabilities for the given feature matrix.
    ///
    /// For binary classification, returns shape `(n_samples, 2)`.
    /// For multiclass, returns shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let expected_features = self.layers[0].weights.nrows();
        if x.ncols() != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let output_activation = if self.is_binary {
            Activation::Logistic
        } else {
            Activation::Identity
        };

        let raw = forward_output(
            x,
            &self.layers,
            self.hidden_activation,
            output_activation,
            !self.is_binary,
        );

        if self.is_binary {
            // raw has shape (n_samples, 1), expand to (n_samples, 2).
            let n = raw.nrows();
            let mut probs = Array2::<F>::zeros((n, 2));
            for i in 0..n {
                let p1 = raw[[i, 0]];
                probs[[i, 0]] = F::one() - p1;
                probs[[i, 1]] = p1;
            }
            Ok(probs)
        } else {
            Ok(raw)
        }
    }

    /// Returns the number of layers in the network (including output).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMLPClassifier<F>
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

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedMLPClassifier<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration for MLPClassifier.
impl<F> PipelineEstimator<F> for MLPClassifier<F>
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
        Ok(Box::new(FittedMLPClassifierPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration.
struct FittedMLPClassifierPipeline<F>(FittedMLPClassifier<F>)
where
    F: Float + Send + Sync + 'static;

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedMLPClassifierPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedMLPClassifierPipeline<F> {}

impl<F> FittedPipelineEstimator<F> for FittedMLPClassifierPipeline<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// MLPRegressor
// ---------------------------------------------------------------------------

/// Multi-layer Perceptron regressor.
///
/// A feedforward neural network for regression tasks. Uses identity
/// activation on the output layer and minimizes mean squared error.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_linear::mlp::MLPRegressor;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{array, Array2};
///
/// let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = array![2.0, 4.0, 6.0, 8.0];
///
/// let reg = MLPRegressor::<f64>::new()
///     .with_hidden_layer_sizes(vec![10])
///     .with_max_iter(500);
/// let fitted = reg.fit(&x, &y).unwrap();
/// let preds = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MLPRegressor<F> {
    /// Sizes of the hidden layers.
    pub hidden_layer_sizes: Vec<usize>,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Optimization solver.
    pub solver: Solver<F>,
    /// Maximum number of training epochs.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Mini-batch size. `None` means `min(200, n_samples)`.
    pub batch_size: Option<usize>,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Whether to use early stopping on a held-out validation set.
    pub early_stopping: bool,
    /// Fraction of training data to use as validation.
    pub validation_fraction: f64,
    /// L2 regularization strength.
    pub alpha: F,
}

impl<F: Float> MLPRegressor<F> {
    /// Create a new `MLPRegressor` with default settings.
    ///
    /// Defaults: `hidden_layer_sizes = [100]`, `activation = Relu`,
    /// `solver = Adam(lr=0.001)`, `max_iter = 200`, `tol = 1e-4`,
    /// `alpha = 0.0001`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            activation: Activation::Relu,
            solver: Solver::Adam {
                learning_rate: F::from(0.001).unwrap(),
                beta1: F::from(0.9).unwrap(),
                beta2: F::from(0.999).unwrap(),
                epsilon: F::from(1e-8).unwrap(),
            },
            max_iter: 200,
            tol: F::from(1e-4).unwrap(),
            batch_size: None,
            random_state: None,
            early_stopping: false,
            validation_fraction: 0.1,
            alpha: F::from(0.0001).unwrap(),
        }
    }

    /// Set the hidden layer sizes.
    #[must_use]
    pub fn with_hidden_layer_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = sizes;
        self
    }

    /// Set the activation function for hidden layers.
    #[must_use]
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set the optimization solver.
    #[must_use]
    pub fn with_solver(mut self, solver: Solver<F>) -> Self {
        self.solver = solver;
        self
    }

    /// Set the maximum number of training epochs.
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

    /// Set the mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable or disable early stopping.
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set the validation fraction for early stopping.
    #[must_use]
    pub fn with_validation_fraction(mut self, fraction: f64) -> Self {
        self.validation_fraction = fraction;
        self
    }

    /// Set the L2 regularization strength.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> Default for MLPRegressor<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted multi-layer perceptron regressor.
///
/// Holds the learned network weights and biases.
#[derive(Debug, Clone)]
pub struct FittedMLPRegressor<F> {
    /// Network layer parameters.
    layers: Vec<LayerParams<F>>,
    /// Hidden layer activation function.
    hidden_activation: Activation,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<F>>
    for MLPRegressor<F>
{
    type Fitted = FittedMLPRegressor<F>;
    type Error = FerroError;

    /// Fit the MLP regressor on the given data.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// numbers of samples.
    /// Returns [`FerroError::InvalidParameter`] if hidden layer sizes are
    /// empty or contain zero.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedMLPRegressor<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

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
                context: "MLPRegressor requires at least one sample".into(),
            });
        }
        validate_hidden_layers(&self.hidden_layer_sizes)?;

        // Target matrix of shape (n_samples, 1).
        let y_target = y
            .clone()
            .into_shape_with_order((n_samples, 1))
            .map_err(|_| FerroError::NumericalInstability {
                message: "failed to reshape target vector".into(),
            })?;

        let output_size = 1;
        let layer_sizes = build_layer_sizes(n_features, &self.hidden_layer_sizes, output_size);

        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        let mut layers = init_layers::<F>(&layer_sizes, &mut rng);

        train_network(
            x,
            &y_target,
            &mut layers,
            self.activation,
            Activation::Identity,
            false, // is_multiclass
            &self.solver,
            self.alpha,
            self.max_iter,
            self.tol,
            self.batch_size,
            self.early_stopping,
            self.validation_fraction,
            &mut rng,
            false, // is_classifier
        )?;

        Ok(FittedMLPRegressor {
            layers,
            hidden_activation: self.activation,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> FittedMLPRegressor<F> {
    /// Returns the number of layers in the network (including output).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>>
    for FittedMLPRegressor<F>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict continuous values for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let expected_features = self.layers[0].weights.nrows();
        if x.ncols() != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted model".into(),
            });
        }

        let output = forward_output(
            x,
            &self.layers,
            self.hidden_activation,
            Activation::Identity,
            false,
        );
        // Output has shape (n_samples, 1), flatten to (n_samples,).
        Ok(output.column(0).to_owned())
    }
}

// Pipeline integration for MLPRegressor.
impl<F> PipelineEstimator<F> for MLPRegressor<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(FittedMLPRegressorPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration.
struct FittedMLPRegressorPipeline<F>(FittedMLPRegressor<F>)
where
    F: Float + Send + Sync + 'static;

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedMLPRegressorPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedMLPRegressorPipeline<F> {}

impl<F> FittedPipelineEstimator<F> for FittedMLPRegressorPipeline<F>
where
    F: Float + ToPrimitive + FromPrimitive + ScalarOperand + Send + Sync + 'static,
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.0.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Validate that hidden layer sizes are non-empty and all positive.
fn validate_hidden_layers(sizes: &[usize]) -> Result<(), FerroError> {
    if sizes.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "hidden_layer_sizes".into(),
            reason: "must have at least one hidden layer".into(),
        });
    }
    for (i, &s) in sizes.iter().enumerate() {
        if s == 0 {
            return Err(FerroError::InvalidParameter {
                name: format!("hidden_layer_sizes[{i}]"),
                reason: "layer size must be positive".into(),
            });
        }
    }
    Ok(())
}

/// Build the vector of layer sizes: `[input, hidden..., output]`.
fn build_layer_sizes(n_features: usize, hidden: &[usize], output_size: usize) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(hidden.len() + 2);
    sizes.push(n_features);
    sizes.extend_from_slice(hidden);
    sizes.push(output_size);
    sizes
}

/// Initialize network layers with Xavier weights and zero biases.
fn init_layers<F: Float>(
    layer_sizes: &[usize],
    rng: &mut rand::rngs::StdRng,
) -> Vec<LayerParams<F>> {
    let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
    for i in 0..layer_sizes.len() - 1 {
        let fan_in = layer_sizes[i];
        let fan_out = layer_sizes[i + 1];
        layers.push(LayerParams {
            weights: xavier_init(fan_in, fan_out, rng),
            biases: Array1::zeros(fan_out),
        });
    }
    layers
}

/// Build one-hot target matrix for classification.
fn build_classification_target<F: Float>(
    y: &Array1<usize>,
    classes: &[usize],
    is_binary: bool,
) -> Array2<F> {
    let n = y.len();

    if is_binary {
        // Single column: 1 for positive class, 0 for negative.
        let mut target = Array2::<F>::zeros((n, 1));
        for i in 0..n {
            if y[i] == classes[1] {
                target[[i, 0]] = F::one();
            }
        }
        target
    } else {
        // Full one-hot encoding.
        let n_classes = classes.len();
        let mut target = Array2::<F>::zeros((n, n_classes));
        for i in 0..n {
            let ci = classes.iter().position(|&c| c == y[i]).unwrap_or(0);
            target[[i, ci]] = F::one();
        }
        target
    }
}

/// Core training loop shared by classifier and regressor.
#[allow(clippy::too_many_arguments)]
fn train_network<F: Float + ScalarOperand>(
    x: &Array2<F>,
    y_target: &Array2<F>,
    layers: &mut Vec<LayerParams<F>>,
    hidden_activation: Activation,
    output_activation: Activation,
    is_multiclass: bool,
    solver: &Solver<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
    batch_size: Option<usize>,
    early_stopping: bool,
    validation_fraction: f64,
    rng: &mut rand::rngs::StdRng,
    is_classifier: bool,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();

    // Split off validation set if early stopping.
    let (x_train, y_train, x_val, y_val) = if early_stopping {
        let n_val = ((n_samples as f64) * validation_fraction).max(1.0) as usize;
        let n_val = n_val.min(n_samples - 1);
        let n_train = n_samples - n_val;
        let x_t = x.slice(ndarray::s![..n_train, ..]).to_owned();
        let y_t = y_target.slice(ndarray::s![..n_train, ..]).to_owned();
        let x_v = x.slice(ndarray::s![n_train.., ..]).to_owned();
        let y_v = y_target.slice(ndarray::s![n_train.., ..]).to_owned();
        (x_t, y_t, Some(x_v), Some(y_v))
    } else {
        (x.to_owned(), y_target.to_owned(), None, None)
    };

    let n_train = x_train.nrows();
    let effective_batch = batch_size.unwrap_or_else(|| n_train.min(200));
    let effective_batch = effective_batch.max(1).min(n_train);

    // Initialize optimizer state.
    let mut adam_states: Vec<AdamState<F>> = layers
        .iter()
        .map(|l| AdamState {
            m_w: Array2::zeros(l.weights.dim()),
            v_w: Array2::zeros(l.weights.dim()),
            m_b: Array1::zeros(l.biases.len()),
            v_b: Array1::zeros(l.biases.len()),
        })
        .collect();

    let mut sgd_states: Vec<SgdMomentumState<F>> = layers
        .iter()
        .map(|l| SgdMomentumState {
            vel_w: Array2::zeros(l.weights.dim()),
            vel_b: Array1::zeros(l.biases.len()),
        })
        .collect();

    let mut indices: Vec<usize> = (0..n_train).collect();
    let mut prev_loss = F::infinity();
    let mut best_val_loss = F::infinity();
    let mut patience_counter: usize = 0;
    let patience: usize = 10;
    let mut adam_t: usize = 0;

    // Keep a copy of best weights for early stopping.
    let mut best_layers: Vec<LayerParams<F>> = layers.clone();

    for _epoch in 0..max_iter {
        indices.shuffle(rng);

        let mut epoch_loss = F::zero();
        let mut n_batches = 0usize;

        // Mini-batch iteration.
        let mut batch_start = 0;
        while batch_start < n_train {
            let batch_end = (batch_start + effective_batch).min(n_train);
            let batch_idx = &indices[batch_start..batch_end];
            let batch_size_actual = batch_idx.len();

            // Extract batch.
            let x_batch = extract_rows(&x_train, batch_idx);
            let y_batch = extract_rows(&y_train, batch_idx);

            // Forward pass.
            let cache = forward(&x_batch, layers, hidden_activation, output_activation);
            let output = &cache.a_values[cache.a_values.len() - 1];

            // Compute loss and output delta.
            let (batch_loss, delta) = if is_multiclass {
                // Softmax + cross-entropy.
                let probs = softmax_rows(&cache.z_values[cache.z_values.len() - 1]);
                let eps = F::from(1e-15).unwrap();
                let mut loss = F::zero();
                for i in 0..batch_size_actual {
                    for j in 0..y_batch.ncols() {
                        let p = probs[[i, j]].max(eps);
                        loss = loss - y_batch[[i, j]] * p.ln();
                    }
                }
                loss = loss / F::from(batch_size_actual).unwrap();
                // delta = probs - y_true
                let delta = &probs - &y_batch;
                (loss, delta)
            } else if is_classifier {
                // Binary cross-entropy (sigmoid output).
                let eps = F::from(1e-15).unwrap();
                let mut loss = F::zero();
                for i in 0..batch_size_actual {
                    for j in 0..output.ncols() {
                        let p = output[[i, j]].max(eps).min(F::one() - eps);
                        let y_ij = y_batch[[i, j]];
                        loss = loss - (y_ij * p.ln() + (F::one() - y_ij) * (F::one() - p).ln());
                    }
                }
                loss = loss / F::from(batch_size_actual).unwrap();
                // For sigmoid + BCE, delta = output - target.
                let delta = output - &y_batch;
                (loss, delta)
            } else {
                // MSE for regression.
                let diff = output - &y_batch;
                let loss = diff.mapv(|v| v * v).sum() / F::from(batch_size_actual * 2).unwrap();
                // delta = (output - target) for MSE (the 1/n factor is handled in backward).
                let delta = output - &y_batch;
                (loss, delta)
            };

            epoch_loss = epoch_loss + batch_loss;
            n_batches += 1;

            // Backward pass.
            let grads = backward(&x_batch, layers, &cache, &delta, hidden_activation, alpha);

            // Update parameters.
            match solver {
                Solver::Sgd {
                    learning_rate,
                    momentum,
                } => {
                    apply_sgd_update(layers, &grads, &mut sgd_states, *learning_rate, *momentum);
                }
                Solver::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    adam_t += 1;
                    apply_adam_update(
                        layers,
                        &grads,
                        &mut adam_states,
                        *learning_rate,
                        *beta1,
                        *beta2,
                        *epsilon,
                        adam_t,
                    );
                }
            }

            batch_start = batch_end;
        }

        epoch_loss = epoch_loss / F::from(n_batches).unwrap();

        // Early stopping check.
        if early_stopping {
            if let (Some(xv), Some(yv)) = (&x_val, &y_val) {
                let val_loss = compute_loss(
                    xv,
                    yv,
                    layers,
                    hidden_activation,
                    output_activation,
                    is_multiclass,
                    is_classifier,
                );
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_layers = layers.clone();
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        *layers = best_layers;
                        return Ok(());
                    }
                }
            }
        }

        // Convergence check.
        if (prev_loss - epoch_loss).abs() < tol {
            if early_stopping {
                *layers = best_layers;
            }
            return Ok(());
        }
        prev_loss = epoch_loss;
    }

    if early_stopping {
        *layers = best_layers;
    }

    Ok(())
}

/// Compute the loss on a dataset (used for early stopping validation).
fn compute_loss<F: Float + ScalarOperand>(
    x: &Array2<F>,
    y: &Array2<F>,
    layers: &[LayerParams<F>],
    hidden_activation: Activation,
    output_activation: Activation,
    is_multiclass: bool,
    is_classifier: bool,
) -> F {
    let n = x.nrows();
    let output = forward_output(
        x,
        layers,
        hidden_activation,
        output_activation,
        is_multiclass,
    );
    let eps = F::from(1e-15).unwrap();
    let n_f = F::from(n).unwrap();

    if is_multiclass {
        let mut loss = F::zero();
        for i in 0..n {
            for j in 0..y.ncols() {
                let p = output[[i, j]].max(eps);
                loss = loss - y[[i, j]] * p.ln();
            }
        }
        loss / n_f
    } else if is_classifier {
        let mut loss = F::zero();
        for i in 0..n {
            for j in 0..output.ncols() {
                let p = output[[i, j]].max(eps).min(F::one() - eps);
                let y_ij = y[[i, j]];
                loss = loss - (y_ij * p.ln() + (F::one() - y_ij) * (F::one() - p).ln());
            }
        }
        loss / n_f
    } else {
        let diff = &output - y;
        diff.mapv(|v| v * v).sum() / (n_f * F::from(2.0).unwrap())
    }
}

/// Extract selected rows from a 2D array by index.
fn extract_rows<F: Float>(arr: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let n_cols = arr.ncols();
    let n_rows = indices.len();
    let mut result = Array2::<F>::zeros((n_rows, n_cols));
    for (out_row, &idx) in indices.iter().enumerate() {
        for j in 0..n_cols {
            result[[out_row, j]] = arr[[idx, j]];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Activation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stable_sigmoid() {
        assert_relative_eq!(stable_sigmoid(0.0_f64), 0.5, epsilon = 1e-10);
        assert!(stable_sigmoid(10.0_f64) > 0.99);
        assert!(stable_sigmoid(-10.0_f64) < 0.01);
        assert_relative_eq!(
            stable_sigmoid(1.0_f64) + stable_sigmoid(-1.0_f64),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_softmax_rows() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let probs = softmax_rows(&logits);

        assert_relative_eq!(probs.row(0).sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(probs.row(1).sum(), 1.0, epsilon = 1e-10);
        // Uniform logits => uniform probs.
        assert_relative_eq!(probs[[1, 0]], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_activate_relu() {
        let mut z = Array2::from_shape_vec((1, 4), vec![-2.0, -0.5, 0.0, 1.5]).unwrap();
        activate_inplace(&mut z, Activation::Relu);
        assert_relative_eq!(z[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 3]], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_activate_tanh() {
        let mut z = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        activate_inplace(&mut z, Activation::Tanh);
        assert_relative_eq!(z[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(z[[0, 1]], 1.0_f64.tanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_activate_logistic() {
        let mut z = Array2::from_shape_vec((1, 2), vec![0.0, 100.0]).unwrap();
        activate_inplace(&mut z, Activation::Logistic);
        assert_relative_eq!(z[[0, 0]], 0.5, epsilon = 1e-10);
        assert!(z[[0, 1]] > 0.99);
    }

    #[test]
    fn test_activate_identity() {
        let mut z = Array2::from_shape_vec((1, 2), vec![3.14, -2.71]).unwrap();
        let original = z.clone();
        activate_inplace(&mut z, Activation::Identity);
        assert_relative_eq!(z[[0, 0]], original[[0, 0]], epsilon = 1e-15);
        assert_relative_eq!(z[[0, 1]], original[[0, 1]], epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // Xavier init test
    // -----------------------------------------------------------------------

    #[test]
    fn test_xavier_init_shape() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let w: Array2<f64> = xavier_init(10, 5, &mut rng);
        assert_eq!(w.dim(), (10, 5));
        // Values should be in [-sqrt(6/15), sqrt(6/15)].
        let limit = (6.0_f64 / 15.0).sqrt();
        for &v in w.iter() {
            assert!(v >= -limit - 0.01 && v <= limit + 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_hidden_layers_empty() {
        assert!(validate_hidden_layers(&[]).is_err());
    }

    #[test]
    fn test_validate_hidden_layers_zero() {
        assert!(validate_hidden_layers(&[100, 0, 50]).is_err());
    }

    #[test]
    fn test_validate_hidden_layers_ok() {
        assert!(validate_hidden_layers(&[100, 50]).is_ok());
    }

    // -----------------------------------------------------------------------
    // MLPClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length.
        let clf = MLPClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];
        let clf = MLPClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_binary_basic() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, // class 0
                8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected >= 6 correct, got {correct}");
    }

    #[test]
    fn test_classifier_binary_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();

        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);

        // Probabilities should sum to 1.
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_classifier_multiclass() {
        // Three well-separated clusters.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, // class 0
                10.0, 0.0, 10.5, 0.0, 10.0, 0.5, // class 1
                0.0, 10.0, 0.5, 10.0, 0.0, 10.5, // class 2
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![20])
            .with_max_iter(1000)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected >= 7 correct, got {correct}");
    }

    #[test]
    fn test_classifier_multiclass_predict_proba() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 0.0, 10.0, 0.5,
                10.0, 0.0, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![20])
            .with_max_iter(1000)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // Probabilities should sum to 1 for each sample.
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_classifier_has_classes() {
        let x = Array2::from_shape_vec((6, 1), vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_max_iter(100)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_classifier_predict_shape_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y_train = array![0, 0, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_max_iter(50)
            .with_random_state(42);
        let fitted = clf.fit(&x_train, &y_train).unwrap();

        // Wrong number of features.
        let x_test = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_test).is_err());
    }

    #[test]
    fn test_classifier_with_sgd() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_solver(Solver::Sgd {
                learning_rate: 0.01,
                momentum: 0.9,
            })
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "SGD: expected >= 6 correct, got {correct}");
    }

    #[test]
    fn test_classifier_tanh_activation() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 8.0, 8.0, 9.0, 8.0, 8.0, 9.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_activation(Activation::Tanh)
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_two_hidden_layers() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10, 5])
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        // Network should have 3 layers: input->10, 10->5, 5->output.
        assert_eq!(fitted.n_layers(), 3);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_early_stopping() {
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0,
                1.0, 1.0, 0.2, 0.3, 8.0, 8.0, 8.5, 8.0, 8.0, 8.5, 8.5, 8.5, 9.0, 8.0, 8.0, 9.0,
                9.0, 8.5, 8.5, 9.0, 9.0, 9.0, 8.2, 8.3,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_max_iter(1000)
            .with_early_stopping(true)
            .with_validation_fraction(0.2)
            .with_random_state(42);
        let fitted = clf.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_classifier_pipeline_integration() {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let model = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_max_iter(200)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    // -----------------------------------------------------------------------
    // MLPRegressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_regressor_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length.
        let reg = MLPRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_linear_fit() {
        // y = 2x (simple linear relationship).
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_max_iter(2000)
            .with_random_state(42);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check that predictions are reasonably close.
        for (pred, actual) in preds.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 2.0, "pred={pred}, actual={actual}");
        }
    }

    #[test]
    fn test_regressor_predict_shape_mismatch() {
        let x_train =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y_train = array![1.0, 2.0, 3.0, 4.0];

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_max_iter(50)
            .with_random_state(42);
        let fitted = reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features.
        let x_test = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&x_test).is_err());
    }

    #[test]
    fn test_regressor_with_sgd() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_solver(Solver::Sgd {
                learning_rate: 0.01,
                momentum: 0.9,
            })
            .with_max_iter(2000)
            .with_random_state(42);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_regressor_two_hidden_layers() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![10, 5])
            .with_max_iter(500)
            .with_random_state(42);
        let fitted = reg.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_layers(), 3);
    }

    #[test]
    fn test_regressor_multi_feature() {
        // y = x1 + x2
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0],
        )
        .unwrap();
        let y = array![2.0, 3.0, 3.0, 4.0, 5.0, 6.0];

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![20])
            .with_max_iter(2000)
            .with_random_state(42);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        for (pred, actual) in preds.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 3.0, "pred={pred}, actual={actual}");
        }
    }

    #[test]
    fn test_regressor_early_stopping() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|v| v as f64).collect()).unwrap();
        let y = Array1::from_shape_fn(20, |i| (i + 1) as f64 * 2.0);

        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![10])
            .with_max_iter(1000)
            .with_early_stopping(true)
            .with_validation_fraction(0.2)
            .with_random_state(42);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_regressor_pipeline_integration() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_max_iter(200)
            .with_random_state(42);
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Builder pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classifier_builder() {
        let clf = MLPClassifier::<f64>::new()
            .with_hidden_layer_sizes(vec![50, 25])
            .with_activation(Activation::Tanh)
            .with_solver(Solver::Sgd {
                learning_rate: 0.01,
                momentum: 0.9,
            })
            .with_max_iter(100)
            .with_tol(1e-3)
            .with_batch_size(32)
            .with_random_state(123)
            .with_early_stopping(true)
            .with_validation_fraction(0.15)
            .with_alpha(0.001);

        assert_eq!(clf.hidden_layer_sizes, vec![50, 25]);
        assert_eq!(clf.activation, Activation::Tanh);
        assert_eq!(clf.max_iter, 100);
        assert!(clf.early_stopping);
        assert_eq!(clf.batch_size, Some(32));
        assert_eq!(clf.random_state, Some(123));
    }

    #[test]
    fn test_regressor_builder() {
        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![64, 32])
            .with_activation(Activation::Logistic)
            .with_max_iter(300)
            .with_tol(1e-5)
            .with_alpha(0.01);

        assert_eq!(reg.hidden_layer_sizes, vec![64, 32]);
        assert_eq!(reg.activation, Activation::Logistic);
        assert_eq!(reg.max_iter, 300);
    }

    #[test]
    fn test_classifier_default() {
        let clf = MLPClassifier::<f64>::default();
        assert_eq!(clf.hidden_layer_sizes, vec![100]);
        assert_eq!(clf.activation, Activation::Relu);
        assert_eq!(clf.max_iter, 200);
    }

    #[test]
    fn test_regressor_default() {
        let reg = MLPRegressor::<f64>::default();
        assert_eq!(reg.hidden_layer_sizes, vec![100]);
        assert_eq!(reg.activation, Activation::Relu);
        assert_eq!(reg.max_iter, 200);
    }

    #[test]
    fn test_classifier_empty_hidden_layers() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = MLPClassifier::<f64>::new().with_hidden_layer_sizes(vec![]);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_empty_hidden_layers() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];
        let reg = MLPRegressor::<f64>::new().with_hidden_layer_sizes(vec![]);
        assert!(reg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_zero_hidden_layer_size() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];
        let clf = MLPClassifier::<f64>::new().with_hidden_layer_sizes(vec![10, 0]);
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_classifier_no_samples() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<usize>::zeros(0);
        let clf = MLPClassifier::<f64>::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn test_regressor_no_samples() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let reg = MLPRegressor::<f64>::new();
        assert!(reg.fit(&x, &y).is_err());
    }
}
