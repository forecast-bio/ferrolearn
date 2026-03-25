//! Random projection transformers for dimensionality reduction.
//!
//! Random projections preserve pairwise distances in expectation (Johnson-Lindenstrauss lemma).
//!
//! - [`GaussianRandomProjection`] — dense Gaussian random matrix
//! - [`SparseRandomProjection`] — sparse random matrix with `{-1, 0, +1}` entries

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// GaussianRandomProjection
// ---------------------------------------------------------------------------

/// Gaussian random projection transformer.
///
/// Projects data into a lower-dimensional space using a random matrix drawn
/// from `N(0, 1/n_components)`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::random_projection::GaussianRandomProjection;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let x = Array2::<f64>::ones((10, 50));
/// let proj = GaussianRandomProjection::<f64>::new(5);
/// let fitted = proj.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[10, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianRandomProjection<F> {
    /// Number of output dimensions.
    n_components: usize,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> GaussianRandomProjection<F> {
    /// Create a new Gaussian random projection with `n_components` output dimensions.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted Gaussian random projection holding the projection matrix.
#[derive(Debug, Clone)]
pub struct FittedGaussianRandomProjection<F> {
    /// Projection matrix of shape `(n_features, n_components)`.
    projection: Array2<F>,
}

impl<F: Float + Send + Sync + 'static> FittedGaussianRandomProjection<F> {
    /// Return a reference to the projection matrix.
    #[must_use]
    pub fn projection(&self) -> &Array2<F> {
        &self.projection
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GaussianRandomProjection<F> {
    type Fitted = FittedGaussianRandomProjection<F>;
    type Error = FerroError;

    /// Fit the projection by generating a random matrix `R ~ N(0, 1/n_components)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_components == 0`.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    fn fit(
        &self,
        x: &Array2<F>,
        _y: &(),
    ) -> Result<FittedGaussianRandomProjection<F>, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be >= 1".into(),
            });
        }
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "GaussianRandomProjection::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let scale = F::one() / F::from(self.n_components).unwrap().sqrt();
        let normal = StandardNormal;
        let mut projection = Array2::zeros((n_features, self.n_components));
        for v in projection.iter_mut() {
            let sample: f64 = normal.sample(&mut rng);
            *v = F::from(sample).unwrap() * scale;
        }

        Ok(FittedGaussianRandomProjection { projection })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>>
    for FittedGaussianRandomProjection<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by computing `X @ R`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols() != n_features`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.projection.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.projection.nrows()],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedGaussianRandomProjection::transform".into(),
            });
        }
        Ok(x.dot(&self.projection))
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for GaussianRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the projection must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "GaussianRandomProjection".into(),
            reason: "projection must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for GaussianRandomProjection<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for GaussianRandomProjection<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedGaussianRandomProjection<F>
{
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// SparseRandomProjection
// ---------------------------------------------------------------------------

/// Sparse random projection transformer.
///
/// Projects data into a lower-dimensional space using a sparse random matrix
/// with entries `{-1, 0, +1}` drawn with probabilities
/// `{d/2, 1 - d, d/2}`, scaled by `sqrt(1 / (d * n_components))`.
///
/// The default density `d = 1 / sqrt(n_features)` is used when not specified.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::random_projection::SparseRandomProjection;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::Array2;
///
/// let x = Array2::<f64>::ones((10, 100));
/// let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
/// let fitted = proj.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.shape(), &[10, 5]);
/// ```
#[derive(Debug, Clone)]
pub struct SparseRandomProjection<F> {
    /// Number of output dimensions.
    n_components: usize,
    /// Density of non-zero entries. `None` means `1/sqrt(n_features)`.
    density: Option<f64>,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SparseRandomProjection<F> {
    /// Create a new sparse random projection with `n_components` output dimensions.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            density: None,
            random_state: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the density of non-zero entries.
    #[must_use]
    pub fn density(mut self, density: f64) -> Self {
        self.density = Some(density);
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Fitted sparse random projection holding the projection matrix.
#[derive(Debug, Clone)]
pub struct FittedSparseRandomProjection<F> {
    /// Projection matrix of shape `(n_features, n_components)`.
    projection: Array2<F>,
}

impl<F: Float + Send + Sync + 'static> FittedSparseRandomProjection<F> {
    /// Return a reference to the projection matrix.
    #[must_use]
    pub fn projection(&self) -> &Array2<F> {
        &self.projection
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SparseRandomProjection<F> {
    type Fitted = FittedSparseRandomProjection<F>;
    type Error = FerroError;

    /// Fit the projection by generating a sparse random matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_components == 0` or
    /// `density` is not in `(0, 1]`.
    /// Returns [`FerroError::InsufficientSamples`] if `x` has zero rows.
    fn fit(
        &self,
        x: &Array2<F>,
        _y: &(),
    ) -> Result<FittedSparseRandomProjection<F>, FerroError> {
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be >= 1".into(),
            });
        }
        if x.nrows() == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SparseRandomProjection::fit".into(),
            });
        }

        let n_features = x.ncols();
        let d = self
            .density
            .unwrap_or_else(|| 1.0 / (n_features as f64).sqrt());

        if d <= 0.0 || d > 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "density".into(),
                reason: format!("must be in (0, 1], got {d}"),
            });
        }

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let scale = F::from(1.0 / (d * self.n_components as f64).sqrt()).unwrap();
        let uniform = rand::distr::Uniform::new(0.0_f64, 1.0).unwrap();

        let mut projection = Array2::zeros((n_features, self.n_components));
        for v in projection.iter_mut() {
            let u: f64 = uniform.sample(&mut rng);
            if u < d / 2.0 {
                *v = scale.neg();
            } else if u < d {
                *v = scale;
            }
            // else: remains 0
        }

        Ok(FittedSparseRandomProjection { projection })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>>
    for FittedSparseRandomProjection<F>
{
    type Output = Array2<F>;
    type Error = FerroError;

    /// Transform data by computing `X @ R`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols() != n_features`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.projection.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.projection.nrows()],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSparseRandomProjection::transform".into(),
            });
        }
        Ok(x.dot(&self.projection))
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SparseRandomProjection<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the projection must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "SparseRandomProjection".into(),
            reason: "projection must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for SparseRandomProjection<F> {
    type FitError = FerroError;

    /// Fit and transform in one step.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for SparseRandomProjection<F> {
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F>
    for FittedSparseRandomProjection<F>
{
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // -- GaussianRandomProjection --

    #[test]
    fn test_gaussian_rp_output_shape() {
        let x = Array2::<f64>::ones((10, 50));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_gaussian_rp_deterministic() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(3).random_state(42);
        let fitted1 = proj.fit(&x, &()).unwrap();
        let out1 = fitted1.transform(&x).unwrap();
        let fitted2 = proj.fit(&x, &()).unwrap();
        let out2 = fitted2.transform(&x).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_rp_zero_components() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = GaussianRandomProjection::<f64>::new(0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_gaussian_rp_empty_input() {
        let x = Array2::<f64>::zeros((0, 10));
        let proj = GaussianRandomProjection::<f64>::new(5);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_gaussian_rp_shape_mismatch() {
        let x_train = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x_train, &()).unwrap();
        let x_bad = Array2::<f64>::ones((5, 15));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_gaussian_rp_fit_transform() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = GaussianRandomProjection::<f64>::new(5).random_state(42);
        let out = proj.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_gaussian_rp_f32() {
        let x = Array2::<f32>::ones((5, 10));
        let proj = GaussianRandomProjection::<f32>::new(3).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 3]);
    }

    // -- SparseRandomProjection --

    #[test]
    fn test_sparse_rp_output_shape() {
        let x = Array2::<f64>::ones((10, 100));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_sparse_rp_deterministic() {
        let x = Array2::<f64>::ones((10, 50));
        let proj = SparseRandomProjection::<f64>::new(3).random_state(42);
        let fitted1 = proj.fit(&x, &()).unwrap();
        let out1 = fitted1.transform(&x).unwrap();
        let fitted2 = proj.fit(&x, &()).unwrap();
        let out2 = fitted2.transform(&x).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_rp_sparsity() {
        let x = Array2::<f64>::ones((5, 100));
        let proj = SparseRandomProjection::<f64>::new(10).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let r = fitted.projection();
        // With density = 1/sqrt(100) = 0.1, about 90% should be zero
        let total = r.len();
        let zeros = r.iter().filter(|&&v| v == 0.0).count();
        let sparsity = zeros as f64 / total as f64;
        assert!(sparsity > 0.5, "expected sparse matrix, got sparsity={sparsity}");
    }

    #[test]
    fn test_sparse_rp_custom_density() {
        let x = Array2::<f64>::ones((5, 20));
        let proj = SparseRandomProjection::<f64>::new(5)
            .density(0.5)
            .random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 5]);
    }

    #[test]
    fn test_sparse_rp_zero_components() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = SparseRandomProjection::<f64>::new(0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_invalid_density() {
        let x = Array2::<f64>::ones((5, 10));
        let proj = SparseRandomProjection::<f64>::new(5).density(0.0);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_empty_input() {
        let x = Array2::<f64>::zeros((0, 10));
        let proj = SparseRandomProjection::<f64>::new(5);
        assert!(proj.fit(&x, &()).is_err());
    }

    #[test]
    fn test_sparse_rp_shape_mismatch() {
        let x_train = Array2::<f64>::ones((10, 20));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let fitted = proj.fit(&x_train, &()).unwrap();
        let x_bad = Array2::<f64>::ones((5, 15));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_sparse_rp_fit_transform() {
        let x = Array2::<f64>::ones((10, 20));
        let proj = SparseRandomProjection::<f64>::new(5).random_state(42);
        let out = proj.fit_transform(&x).unwrap();
        assert_eq!(out.shape(), &[10, 5]);
    }

    #[test]
    fn test_sparse_rp_f32() {
        let x = Array2::<f32>::ones((5, 10));
        let proj = SparseRandomProjection::<f32>::new(3).random_state(42);
        let fitted = proj.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[5, 3]);
    }
}
