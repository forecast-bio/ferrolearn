//! Random Fourier features for approximating the RBF kernel.
//!
//! [`RBFSampler`] implements the random Fourier feature mapping described in
//! Rahimi & Recht (2007). It approximates the RBF (Gaussian) kernel by
//! projecting data into a randomized feature space:
//!
//! ```text
//! Z(x) = sqrt(2 / n_components) * cos(x @ W + b)
//! ```
//!
//! where `W ~ N(0, 2*gamma*I)` and `b ~ Uniform(0, 2*pi)`. The inner product
//! in the transformed space approximates the RBF kernel:
//!
//! ```text
//! k(x, y) = exp(-gamma * ||x - y||^2) ~ Z(x)^T Z(y)
//! ```
//!
//! # Examples
//!
//! ```
//! use ferrolearn_kernel::RBFSampler;
//! use ferrolearn_core::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64 * 0.1).collect()).unwrap();
//! let sampler = RBFSampler::<f64>::new();
//! let fitted = sampler.fit(&x, &()).unwrap();
//! let z = fitted.transform(&x).unwrap();
//! assert_eq!(z.ncols(), 100); // default n_components
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Random Fourier feature approximation of the RBF kernel.
///
/// Projects data into a randomized feature space where inner products
/// approximate the RBF kernel. This enables the use of linear methods
/// (e.g., linear regression, SVM) as approximations to their kernelized
/// counterparts, with much lower computational cost for large datasets.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct RBFSampler<F> {
    /// RBF kernel parameter (default 1.0).
    gamma: F,
    /// Number of random Fourier features to generate (default 100).
    n_components: usize,
    /// Optional random seed for reproducibility.
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> RBFSampler<F> {
    /// Create a new `RBFSampler` with default settings.
    ///
    /// Defaults: `gamma = 1.0`, `n_components = 100`, `random_state = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: F::one(),
            n_components: 100,
            random_state: None,
        }
    }

    /// Set the RBF kernel parameter `gamma`.
    ///
    /// Larger `gamma` means a narrower kernel (more localized).
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of random Fourier features.
    #[must_use]
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for RBFSampler<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted RBF sampler holding the random projection weights and offsets.
///
/// Created by calling [`Fit::fit`] on an [`RBFSampler`].
#[derive(Debug, Clone)]
pub struct FittedRBFSampler<F> {
    /// Random weight matrix of shape (n_features, n_components).
    /// Sampled from N(0, 2*gamma).
    random_weights: Array2<F>,
    /// Random offset vector of shape (n_components,).
    /// Sampled from Uniform(0, 2*pi).
    random_offset: Array1<F>,
    /// Scaling factor: sqrt(2 / n_components).
    scale: F,
}

impl<F: Float + Send + Sync + 'static> FittedRBFSampler<F> {
    /// Return the random weight matrix of shape `(n_features, n_components)`.
    #[must_use]
    pub fn random_weights(&self) -> &Array2<F> {
        &self.random_weights
    }

    /// Return the random offset vector of shape `(n_components,)`.
    #[must_use]
    pub fn random_offset(&self) -> &Array1<F> {
        &self.random_offset
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for RBFSampler<F> {
    type Fitted = FittedRBFSampler<F>;
    type Error = FerroError;

    /// Fit the RBF sampler by generating random weights and offsets.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `gamma` is non-positive
    /// or `n_components` is zero.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedRBFSampler<F>, FerroError> {
        if self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive".into(),
            });
        }
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "RBFSampler::fit".into(),
            });
        }

        let n_features = x.ncols();
        let n_components = self.n_components;

        // Standard deviation for the weight distribution: sqrt(2 * gamma)
        let std_dev = (F::from(2.0).unwrap() * self.gamma)
            .sqrt()
            .to_f64()
            .unwrap();

        let mut rng = match self.random_state {
            Some(seed) => rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed),
            None => rand_xoshiro::Xoshiro256PlusPlus::from_os_rng(),
        };

        // Sample W ~ N(0, 2*gamma*I)
        let normal = Normal::new(0.0, std_dev).map_err(|e| FerroError::InvalidParameter {
            name: "gamma".into(),
            reason: format!("failed to create normal distribution: {e}"),
        })?;

        let mut w_data = Vec::with_capacity(n_features * n_components);
        for _ in 0..(n_features * n_components) {
            w_data.push(F::from(normal.sample(&mut rng)).unwrap());
        }
        let random_weights =
            Array2::from_shape_vec((n_features, n_components), w_data).map_err(|e| {
                FerroError::NumericalInstability {
                    message: format!("failed to create weight matrix: {e}"),
                }
            })?;

        // Sample b ~ Uniform(0, 2*pi)
        let two_pi = 2.0 * std::f64::consts::PI;
        let uniform = Uniform::new(0.0, two_pi).map_err(|e| FerroError::InvalidParameter {
            name: "offset_distribution".into(),
            reason: format!("failed to create uniform distribution: {e}"),
        })?;
        let mut b_data = Vec::with_capacity(n_components);
        for _ in 0..n_components {
            b_data.push(F::from(uniform.sample(&mut rng)).unwrap());
        }
        let random_offset = Array1::from_vec(b_data);

        // Scale factor: sqrt(2 / n_components)
        let scale = (F::from(2.0).unwrap() / F::from(n_components).unwrap()).sqrt();

        Ok(FittedRBFSampler {
            random_weights,
            random_offset,
            scale,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedRBFSampler<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data using random Fourier features.
    ///
    /// Computes `Z = sqrt(2/n_components) * cos(X @ W + b)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features in `x`
    /// does not match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features_expected = self.random_weights.nrows();
        if x.ncols() != n_features_expected {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features_expected],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedRBFSampler::transform feature count must match fit data".into(),
            });
        }

        // projection = X @ W + b  (broadcast b across rows)
        let projection = x.dot(&self.random_weights) + &self.random_offset;

        // Z = scale * cos(projection)
        let z = projection.mapv(|v| self.scale * v.cos());

        Ok(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    fn make_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..n * d).map(|_| normal.sample(&mut rng)).collect();
        Array2::from_shape_vec((n, d), data).unwrap()
    }

    #[test]
    fn basic_fit_transform() {
        let x = make_data(50, 5, 42);
        let sampler = RBFSampler::<f64>::new().with_random_state(123);
        let fitted = sampler.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (50, 100));

        // All values should be finite
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn output_shape_matches_n_components() {
        let x = make_data(20, 3, 42);
        let sampler = RBFSampler::<f64>::new()
            .with_n_components(50)
            .with_random_state(0);
        let fitted = sampler.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.ncols(), 50);
        assert_eq!(z.nrows(), 20);
    }

    #[test]
    fn kernel_approximation_quality() {
        // The RBF kernel k(x,y) = exp(-gamma * ||x-y||^2)
        // should be well-approximated by Z(x)^T Z(y) with enough components.
        let gamma = 0.5;
        let x = make_data(20, 3, 42);
        let sampler = RBFSampler::<f64>::new()
            .with_gamma(gamma)
            .with_n_components(5000)
            .with_random_state(1);
        let fitted = sampler.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();

        // Check a few pairs
        for i in 0..5 {
            for j in (i + 1)..5 {
                let xi = x.row(i);
                let xj = x.row(j);
                let diff = &xi - &xj;
                let sq_dist: f64 = diff.iter().map(|d| d * d).sum();
                let exact_k = (-gamma * sq_dist).exp();

                let zi = z.row(i);
                let zj = z.row(j);
                let approx_k: f64 = zi.dot(&zj);

                // With 5000 components, approximation should be within ~0.05
                assert!(
                    (exact_k - approx_k).abs() < 0.1,
                    "Kernel approx failed: exact={exact_k:.4}, approx={approx_k:.4}"
                );
            }
        }
    }

    #[test]
    fn reproducible_with_seed() {
        let x = make_data(10, 3, 42);
        let s1 = RBFSampler::<f64>::new()
            .with_random_state(99)
            .fit(&x, &())
            .unwrap();
        let s2 = RBFSampler::<f64>::new()
            .with_random_state(99)
            .fit(&x, &())
            .unwrap();

        let z1 = s1.transform(&x).unwrap();
        let z2 = s2.transform(&x).unwrap();

        for (a, b) in z1.iter().zip(z2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn different_seeds_different_output() {
        let x = make_data(10, 3, 42);
        let s1 = RBFSampler::<f64>::new()
            .with_random_state(1)
            .fit(&x, &())
            .unwrap();
        let s2 = RBFSampler::<f64>::new()
            .with_random_state(2)
            .fit(&x, &())
            .unwrap();

        let z1 = s1.transform(&x).unwrap();
        let z2 = s2.transform(&x).unwrap();

        // Should differ
        let max_diff = (&z1 - &z2)
            .mapv(f64::abs)
            .into_iter()
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 0.01,
            "Different seeds should produce different output"
        );
    }

    #[test]
    fn rejects_zero_gamma() {
        let x = make_data(10, 3, 42);
        let sampler = RBFSampler::<f64>::new().with_gamma(0.0);
        assert!(sampler.fit(&x, &()).is_err());
    }

    #[test]
    fn rejects_negative_gamma() {
        let x = make_data(10, 3, 42);
        let sampler = RBFSampler::<f64>::new().with_gamma(-1.0);
        assert!(sampler.fit(&x, &()).is_err());
    }

    #[test]
    fn rejects_zero_components() {
        let x = make_data(10, 3, 42);
        let sampler = RBFSampler::<f64>::new().with_n_components(0);
        assert!(sampler.fit(&x, &()).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let x = Array2::<f64>::zeros((0, 3));
        let sampler = RBFSampler::<f64>::new().with_random_state(42);
        assert!(sampler.fit(&x, &()).is_err());
    }

    #[test]
    fn transform_rejects_wrong_features() {
        let x = make_data(10, 3, 42);
        let fitted = RBFSampler::<f64>::new()
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let x_wrong = make_data(5, 4, 42);
        assert!(fitted.transform(&x_wrong).is_err());
    }

    #[test]
    fn transform_single_sample() {
        let x = make_data(10, 3, 42);
        let fitted = RBFSampler::<f64>::new()
            .with_n_components(20)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let x_single = x.row(0).to_owned().insert_axis(ndarray::Axis(0));
        let z = fitted.transform(&x_single).unwrap();
        assert_eq!(z.dim(), (1, 20));
    }

    #[test]
    fn values_bounded() {
        // cos values are in [-1, 1], scaled by sqrt(2/n_components)
        let x = make_data(50, 5, 42);
        let n_components = 100;
        let fitted = RBFSampler::<f64>::new()
            .with_n_components(n_components)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let z = fitted.transform(&x).unwrap();
        let scale = (2.0 / n_components as f64).sqrt();
        for &v in z.iter() {
            assert!(v.abs() <= scale + 1e-10, "Value {v} exceeds bound {scale}");
        }
    }

    #[test]
    fn f32_support() {
        let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.1).collect();
        let x = Array2::from_shape_vec((10, 3), data).unwrap();
        let sampler = RBFSampler::<f32>::new().with_random_state(42);
        let fitted = sampler.fit(&x, &()).unwrap();
        let z = fitted.transform(&x).unwrap();
        assert_eq!(z.dim(), (10, 100));
        for &v in z.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn self_kernel_close_to_one() {
        // k(x, x) = exp(0) = 1.0, so Z(x)^T Z(x) should be close to 1.0
        let x = make_data(10, 3, 42);
        let fitted = RBFSampler::<f64>::new()
            .with_gamma(1.0)
            .with_n_components(5000)
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        let z = fitted.transform(&x).unwrap();

        for i in 0..x.nrows() {
            let zi = z.row(i);
            let self_k: f64 = zi.dot(&zi);
            assert!(
                (self_k - 1.0).abs() < 0.1,
                "Self-kernel for row {i}: {self_k:.4} (expected ~1.0)"
            );
        }
    }

    #[test]
    fn builder_chain() {
        let sampler = RBFSampler::<f64>::new()
            .with_gamma(0.5)
            .with_n_components(200)
            .with_random_state(42);
        assert_eq!(sampler.n_components, 200);
    }
}
