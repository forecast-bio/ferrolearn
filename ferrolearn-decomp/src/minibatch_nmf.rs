//! Mini-Batch Non-negative Matrix Factorization (MiniBatchNMF).
//!
//! [`MiniBatchNMF`] decomposes a non-negative matrix `X` into two non-negative
//! factors `W` and `H` such that `X ~ W * H`, processing the data in
//! mini-batches for scalability to large datasets.
//!
//! # Algorithm
//!
//! 1. Initialise `W` and `H` (random or NNDSVD).
//! 2. For each mini-batch `X_batch`:
//!    a. Fix `H`, update `W_batch` via coordinate descent on
//!    `||X_batch - W_batch @ H||^2`.
//!    b. Fix `W`, update `H` via multiplicative update:
//!    `H *= (W^T X_batch) / (W^T W H + eps)`.
//!    c. Online averaging of `W` across batches.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::MiniBatchNMF;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let nmf = MiniBatchNMF::<f64>::new(2);
//! let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
//! let fitted = nmf.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 2);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Initialisation strategy for `MiniBatchNMF`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniBatchNMFInit {
    /// Random non-negative initialisation.
    Random,
    /// Non-Negative Double SVD initialisation (simplified).
    Nndsvd,
}

// ---------------------------------------------------------------------------
// MiniBatchNMF (unfitted)
// ---------------------------------------------------------------------------

/// Mini-Batch NMF configuration.
///
/// Holds hyperparameters for the mini-batch NMF decomposition. Calling
/// [`Fit::fit`] performs the iterative procedure and returns a
/// [`FittedMiniBatchNMF`] that can project new data.
#[derive(Debug, Clone)]
pub struct MiniBatchNMF<F> {
    /// Number of components to extract.
    n_components: usize,
    /// Maximum number of iterations over the full dataset.
    max_iter: usize,
    /// Mini-batch size.
    batch_size: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
    /// Initialisation strategy.
    init: MiniBatchNMFInit,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> MiniBatchNMF<F> {
    /// Create a new `MiniBatchNMF` that extracts `n_components` components.
    ///
    /// Defaults: `max_iter = 200`, `batch_size = 1024`, `tol = 1e-4`,
    /// `init = Random`, `random_state = None`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            batch_size: 1024,
            tol: 1e-4,
            random_state: None,
            init: MiniBatchNMFInit::Random,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for reproducible results.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the initialisation strategy.
    #[must_use]
    pub fn with_init(mut self, init: MiniBatchNMFInit) -> Self {
        self.init = init;
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured maximum iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the configured batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Return the configured tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the configured initialisation strategy.
    #[must_use]
    pub fn init(&self) -> MiniBatchNMFInit {
        self.init
    }
}

// ---------------------------------------------------------------------------
// FittedMiniBatchNMF
// ---------------------------------------------------------------------------

/// A fitted Mini-Batch NMF model holding the learned components.
///
/// Created by calling [`Fit::fit`] on a [`MiniBatchNMF`]. Implements
/// [`Transform<Array2<F>>`] to project new data onto the learned components.
#[derive(Debug, Clone)]
pub struct FittedMiniBatchNMF<F> {
    /// Learned component matrix H, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// Frobenius norm of the reconstruction error at convergence.
    reconstruction_err_: F,
    /// Number of iterations performed.
    n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedMiniBatchNMF<F> {
    /// Learned components (H matrix), shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Frobenius norm of the reconstruction error `||X - W*H||_F`.
    #[must_use]
    pub fn reconstruction_err(&self) -> F {
        self.reconstruction_err_
    }

    /// Number of iterations performed during fitting.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Small epsilon to prevent division by zero.
#[inline]
fn eps<F: Float>() -> F {
    F::from(1e-12).unwrap_or_else(F::epsilon)
}

/// Initialise W and H with random non-negative values.
fn init_random<F: Float>(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for elem in w.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    let mut h = Array2::<F>::zeros((n_components, n_features));
    for elem in h.iter_mut() {
        *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::zero) + eps::<F>();
    }

    (w, h)
}

/// Simplified NNDSVD initialisation: compute `X^T X`, use the top eigenvectors.
fn init_nndsvd_simple<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_components: usize,
    seed: u64,
) -> (Array2<F>, Array2<F>) {
    let (n_samples, n_features) = x.dim();

    // Compute column means for scale.
    let mut avg = F::zero();
    for &v in x.iter() {
        avg = avg + v.abs();
    }
    avg = avg / F::from(n_samples * n_features).unwrap();
    if avg < eps::<F>() {
        avg = F::one();
    }
    let scale = avg.sqrt();

    // Compute X^T X.
    let xtx = x.t().dot(x);

    // Simple power iteration to find dominant eigenvectors.
    let mut rng: rand::rngs::StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0f64).unwrap();

    let mut h = Array2::<F>::zeros((n_components, n_features));

    for k in 0..n_components {
        // Random initial vector.
        let mut v = Array2::<F>::zeros((n_features, 1));
        for elem in v.iter_mut() {
            *elem = F::from(uniform.sample(&mut rng)).unwrap_or_else(F::one);
        }

        // Power iteration (20 steps).
        for _ in 0..20 {
            let v_new = xtx.dot(&v);
            let norm: F = v_new.iter().fold(F::zero(), |a, &b| a + b * b).sqrt();
            if norm > eps::<F>() {
                for (dst, &src) in v.iter_mut().zip(v_new.iter()) {
                    *dst = src / norm;
                }
            }
        }

        // Clamp negatives to zero and store as row of H.
        for j in 0..n_features {
            let val = v[[j, 0]];
            h[[k, j]] = if val > F::zero() { val } else { eps::<F>() * scale };
        }
    }

    // W = X * H^T, clamped non-negative.
    let w_raw = x.dot(&h.t());
    let mut w = Array2::<F>::zeros((n_samples, n_components));
    for i in 0..n_samples {
        for k in 0..n_components {
            let val = w_raw[[i, k]];
            w[[i, k]] = if val > F::zero() { val } else { eps::<F>() };
        }
    }

    (w, h)
}

/// Compute the Frobenius norm of `X - W * H`.
fn reconstruction_error<F: Float + 'static>(x: &Array2<F>, w: &Array2<F>, h: &Array2<F>) -> F {
    let wh = w.dot(h);
    let mut err = F::zero();
    for (a, b) in x.iter().zip(wh.iter()) {
        let diff = *a - *b;
        err = err + diff * diff;
    }
    err.sqrt()
}

/// Solve for W_batch via coordinate descent on `||X_batch - W_batch @ H||^2`,
/// keeping H fixed. All values in W_batch are clamped non-negative.
fn update_w_batch<F: Float + 'static>(
    x_batch: &Array2<F>,
    w_batch: &mut Array2<F>,
    h: &Array2<F>,
) {
    let n_batch = x_batch.nrows();
    let n_components = h.nrows();
    let n_features = h.ncols();
    let epsilon = eps::<F>();

    // Pre-compute H * H^T for efficiency.
    let hht = h.dot(&h.t());

    for _cd_iter in 0..5 {
        for i in 0..n_batch {
            for k in 0..n_components {
                // Compute numerator: sum_j x[i,j] * h[k,j] - sum_{l!=k} w[i,l] * hht[l,k]
                let mut num = F::zero();
                for j in 0..n_features {
                    num = num + x_batch[[i, j]] * h[[k, j]];
                }
                for l in 0..n_components {
                    if l != k {
                        num = num - w_batch[[i, l]] * hht[[l, k]];
                    }
                }

                let den = hht[[k, k]] + epsilon;
                let new_val = num / den;
                w_batch[[i, k]] = if new_val > F::zero() {
                    new_val
                } else {
                    F::zero()
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for MiniBatchNMF<F> {
    type Fitted = FittedMiniBatchNMF<F>;
    type Error = FerroError;

    /// Fit Mini-Batch NMF by iterating over mini-batches.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   the number of features, or if the data contains negative values.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedMiniBatchNMF<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components > n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "MiniBatchNMF::fit requires at least 1 sample".into(),
            });
        }

        // Check non-negativity.
        for &v in x.iter() {
            if v < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "X".into(),
                    reason: "MiniBatchNMF requires non-negative input data".into(),
                });
            }
        }

        let n_comp = self.n_components;
        let seed = self.random_state.unwrap_or(42);
        let epsilon = eps::<F>();

        // Initialise W and H.
        let (mut w, mut h) = match self.init {
            MiniBatchNMFInit::Random => init_random(n_samples, n_features, n_comp, seed),
            MiniBatchNMFInit::Nndsvd => init_nndsvd_simple(x, n_comp, seed),
        };

        let batch_size = self.batch_size.min(n_samples);
        let tol_f = F::from(self.tol).unwrap_or_else(F::epsilon);
        let mut prev_err = reconstruction_error(x, &w, &h);
        let mut actual_iter = 0;

        // Use a simple permutation for batching.
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for iteration in 0..self.max_iter {
            actual_iter = iteration + 1;

            // Simple rotation of indices (deterministic).
            indices.rotate_left(batch_size % n_samples.max(1));

            // Process batches.
            let mut batch_start = 0;
            while batch_start < n_samples {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices: Vec<usize> =
                    indices[batch_start..batch_end].to_vec();
                let actual_batch = batch_indices.len();

                // Extract X_batch.
                let mut x_batch = Array2::<F>::zeros((actual_batch, n_features));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for j in 0..n_features {
                        x_batch[[bi, j]] = x[[idx, j]];
                    }
                }

                // Extract W_batch.
                let mut w_batch = Array2::<F>::zeros((actual_batch, n_comp));
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for k in 0..n_comp {
                        w_batch[[bi, k]] = w[[idx, k]];
                    }
                }

                // Update W_batch (fix H, solve for W_batch).
                update_w_batch(&x_batch, &mut w_batch, &h);

                // Write back W_batch.
                for (bi, &idx) in batch_indices.iter().enumerate() {
                    for k in 0..n_comp {
                        w[[idx, k]] = w_batch[[bi, k]];
                    }
                }

                // Update H via multiplicative update: H *= (W^T X_batch) / (W^T W H + eps).
                let wt = w_batch.t();
                let numerator_h = wt.dot(&x_batch);
                let denominator_h = wt.dot(&w_batch).dot(&h);

                for k in 0..n_comp {
                    for j in 0..n_features {
                        let num = numerator_h[[k, j]];
                        let den = denominator_h[[k, j]] + epsilon;
                        h[[k, j]] = h[[k, j]] * (num / den);
                        if h[[k, j]] < F::zero() {
                            h[[k, j]] = epsilon;
                        }
                    }
                }

                batch_start = batch_end;
            }

            // Check convergence.
            let err = reconstruction_error(x, &w, &h);
            if prev_err > epsilon && (prev_err - err).abs() / prev_err < tol_f {
                break;
            }
            prev_err = err;
        }

        let final_err = reconstruction_error(x, &w, &h);

        Ok(FittedMiniBatchNMF {
            components_: h,
            reconstruction_err_: final_err,
            n_iter_: actual_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedMiniBatchNMF<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project new data onto the learned NMF components.
    ///
    /// Solves `min_{W >= 0} ||X - W H||_F^2` for W using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedMiniBatchNMF::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let n_comp = self.components_.nrows();
        let mut w = Array2::<F>::zeros((n_samples, n_comp));
        // Initialise W with uniform values.
        let init_val = F::from(0.1).unwrap_or_else(F::one);
        w.fill(init_val);

        update_w_batch(x, &mut w, &self.components_);
        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minibatch_nmf_basic() {
        let nmf = MiniBatchNMF::<f64>::new(2).with_random_state(42);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (3, 2));
    }

    #[test]
    fn test_minibatch_nmf_components_shape() {
        let nmf = MiniBatchNMF::<f64>::new(3).with_random_state(0);
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (3, 4));
    }

    #[test]
    fn test_minibatch_nmf_nndsvd_init() {
        let nmf = MiniBatchNMF::<f64>::new(2)
            .with_init(MiniBatchNMFInit::Nndsvd)
            .with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 2);
    }

    #[test]
    fn test_minibatch_nmf_components_nonnegative() {
        let nmf = MiniBatchNMF::<f64>::new(2).with_random_state(7);
        let x = array![
            [1.0, 2.0, 0.0],
            [0.0, 5.0, 6.0],
            [7.0, 0.0, 9.0],
            [0.0, 0.0, 1.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        for &v in fitted.components().iter() {
            assert!(v >= 0.0, "negative component value: {v}");
        }
    }

    #[test]
    fn test_minibatch_nmf_negative_input_error() {
        let nmf = MiniBatchNMF::<f64>::new(1);
        let x = array![[1.0, -2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_zero_components_error() {
        let nmf = MiniBatchNMF::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_too_many_components_error() {
        let nmf = MiniBatchNMF::<f64>::new(5);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_empty_data() {
        let nmf = MiniBatchNMF::<f64>::new(1);
        let x = Array2::<f64>::zeros((0, 3));
        assert!(nmf.fit(&x, &()).is_err());
    }

    #[test]
    fn test_minibatch_nmf_transform_shape_mismatch() {
        let nmf = MiniBatchNMF::<f64>::new(1).with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_minibatch_nmf_reconstruction_err_positive() {
        let nmf = MiniBatchNMF::<f64>::new(1).with_random_state(42);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.reconstruction_err() >= 0.0);
        assert!(fitted.reconstruction_err().is_finite());
    }

    #[test]
    fn test_minibatch_nmf_n_iter_positive() {
        let nmf = MiniBatchNMF::<f64>::new(1)
            .with_max_iter(5)
            .with_random_state(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_minibatch_nmf_small_batch() {
        let nmf = MiniBatchNMF::<f64>::new(1)
            .with_batch_size(2)
            .with_random_state(42);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        let fitted = nmf.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
    }

    #[test]
    fn test_minibatch_nmf_f32() {
        let nmf = MiniBatchNMF::<f32>::new(1).with_random_state(0);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = nmf.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_minibatch_nmf_builder_methods() {
        let nmf = MiniBatchNMF::<f64>::new(3)
            .with_max_iter(100)
            .with_batch_size(512)
            .with_tol(1e-5)
            .with_init(MiniBatchNMFInit::Nndsvd)
            .with_random_state(99);
        assert_eq!(nmf.n_components(), 3);
        assert_eq!(nmf.max_iter(), 100);
        assert_eq!(nmf.batch_size(), 512);
        assert!((nmf.tol() - 1e-5).abs() < 1e-15);
        assert_eq!(nmf.init(), MiniBatchNMFInit::Nndsvd);
    }
}
