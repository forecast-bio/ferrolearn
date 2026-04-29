//! Kernel functions for nonparametric regression.
//!
//! All kernels are univariate: they take scaled distances `u = (x - x_i) / h`
//! and return non-negative weights. Multivariate kernels are constructed as
//! products of univariate kernels (see [`crate::weights`]).

use ndarray::{Array1, ArrayView1};
use num_traits::Float;

/// A univariate kernel function K: R → R+.
///
/// Generic over `F` to enable monomorphization — the compiler inlines
/// kernel evaluation into the weight computation inner loop, enabling
/// auto-vectorization.
pub trait Kernel<F: Float>: Send + Sync {
    /// Evaluate the kernel at scaled distances `u = (x - x_i) / h`.
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F>;

    /// Whether this kernel has compact support (`|u| <= 1`).
    ///
    /// Enables BallTree acceleration for weight computation.
    fn has_compact_support(&self) -> bool;
}

/// Type-erased kernel for runtime selection.
///
/// Wraps `Box<dyn Kernel<F>>` so users can select kernels by name at runtime
/// (e.g., from config files) without losing the `Kernel` trait interface.
pub struct DynKernel<F: Float>(Box<dyn Kernel<F>>);

impl<F: Float + Send + Sync + 'static> DynKernel<F> {
    /// Construct from a kernel name string.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel name is not recognized.
    pub fn from_name(name: &str) -> Result<Self, ferrolearn_core::FerroError> {
        match name {
            "gaussian" => Ok(Self(Box::new(GaussianKernel))),
            "epanechnikov" => Ok(Self(Box::new(EpanechnikovKernel))),
            "tricube" => Ok(Self(Box::new(TricubeKernel))),
            "biweight" => Ok(Self(Box::new(BiweightKernel))),
            "triweight" => Ok(Self(Box::new(TriweightKernel))),
            "uniform" => Ok(Self(Box::new(UniformKernel))),
            "cosine" => Ok(Self(Box::new(CosineKernel))),
            _ => Err(ferrolearn_core::FerroError::InvalidParameter {
                name: "kernel".into(),
                reason: format!("unknown kernel: {name}"),
            }),
        }
    }

    /// Wrap an existing kernel implementation.
    pub fn new(kernel: impl Kernel<F> + 'static) -> Self {
        Self(Box::new(kernel))
    }
}

impl<F: Float + Send + Sync + 'static> Kernel<F> for DynKernel<F> {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        self.0.evaluate(u)
    }
    fn has_compact_support(&self) -> bool {
        self.0.has_compact_support()
    }
}

/// Gaussian kernel: K(u) = (1/√(2π)) exp(-u²/2)
#[derive(Debug, Clone, Copy)]
pub struct GaussianKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for GaussianKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let inv_sqrt_2pi = F::from(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();
        let half = F::from(0.5).unwrap();
        u.mapv(|x| inv_sqrt_2pi * (-half * x * x).exp())
    }

    fn has_compact_support(&self) -> bool {
        false
    }
}

/// Epanechnikov kernel: K(u) = 0.75 (1 - u²) for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct EpanechnikovKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for EpanechnikovKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let coeff = F::from(0.75).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| {
            if x.abs() <= one {
                coeff * (one - x * x)
            } else {
                zero
            }
        })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

/// Tricube kernel: K(u) = (70/81)(1 - |u|³)³ for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct TricubeKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for TricubeKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let coeff = F::from(70.0 / 81.0).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| {
            let abs_x = x.abs();
            if abs_x <= one {
                let t = one - abs_x * abs_x * abs_x;
                coeff * t * t * t
            } else {
                zero
            }
        })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

/// Biweight (quartic) kernel: K(u) = (15/16)(1 - u²)² for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct BiweightKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for BiweightKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let coeff = F::from(15.0 / 16.0).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| {
            if x.abs() <= one {
                let t = one - x * x;
                coeff * t * t
            } else {
                zero
            }
        })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

/// Triweight kernel: K(u) = (35/32)(1 - u²)³ for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct TriweightKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for TriweightKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let coeff = F::from(35.0 / 32.0).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| {
            if x.abs() <= one {
                let t = one - x * x;
                coeff * t * t * t
            } else {
                zero
            }
        })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

/// Uniform (rectangular) kernel: K(u) = 0.5 for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct UniformKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for UniformKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let half = F::from(0.5).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| if x.abs() <= one { half } else { zero })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

/// Cosine kernel: K(u) = (π/4) cos(πu/2) for |u| ≤ 1
#[derive(Debug, Clone, Copy)]
pub struct CosineKernel;

impl<F: Float + Send + Sync + 'static> Kernel<F> for CosineKernel {
    fn evaluate(&self, u: &ArrayView1<F>) -> Array1<F> {
        let pi = F::from(std::f64::consts::PI).unwrap();
        let quarter_pi = pi / F::from(4.0).unwrap();
        let half_pi = pi / F::from(2.0).unwrap();
        let one = F::one();
        let zero = F::zero();
        u.mapv(|x| {
            if x.abs() <= one {
                quarter_pi * (half_pi * x).cos()
            } else {
                zero
            }
        })
    }

    fn has_compact_support(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn gaussian_at_zero() {
        let k = GaussianKernel;
        let u = array![0.0f64];
        let result = k.evaluate(&u.view());
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(result[0], expected, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_symmetry() {
        let k = GaussianKernel;
        let u = array![0.5f64, -0.5, 1.0, -1.0, 2.0, -2.0];
        let result = k.evaluate(&u.view());
        assert_abs_diff_eq!(result[0], result[1], epsilon = 1e-15);
        assert_abs_diff_eq!(result[2], result[3], epsilon = 1e-15);
        assert_abs_diff_eq!(result[4], result[5], epsilon = 1e-15);
    }

    #[test]
    fn gaussian_not_compact() {
        assert!(!Kernel::<f64>::has_compact_support(&GaussianKernel));
    }

    #[test]
    fn epanechnikov_at_zero() {
        let k = EpanechnikovKernel;
        let result = k.evaluate(&array![0.0f64].view());
        assert_abs_diff_eq!(result[0], 0.75, epsilon = 1e-15);
    }

    #[test]
    fn epanechnikov_compact() {
        let k = EpanechnikovKernel;
        let u = array![0.0f64, 0.5, 1.0, 1.01, -1.01, 2.0];
        let result = k.evaluate(&u.view());
        assert!(result[0] > 0.0);
        assert!(result[1] > 0.0);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[4], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[5], 0.0, epsilon = 1e-15);
        assert!(Kernel::<f64>::has_compact_support(&EpanechnikovKernel));
    }

    #[test]
    fn uniform_compact() {
        let k = UniformKernel;
        let u = array![0.0f64, 0.5, 1.0, 1.5];
        let result = k.evaluate(&u.view());
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(result[1], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(result[2], 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn tricube_at_zero() {
        let k = TricubeKernel;
        let result = k.evaluate(&array![0.0f64].view());
        assert_abs_diff_eq!(result[0], 70.0 / 81.0, epsilon = 1e-15);
    }

    #[test]
    fn biweight_at_zero() {
        let k = BiweightKernel;
        let result = k.evaluate(&array![0.0f64].view());
        assert_abs_diff_eq!(result[0], 15.0 / 16.0, epsilon = 1e-15);
    }

    #[test]
    fn triweight_at_zero() {
        let k = TriweightKernel;
        let result = k.evaluate(&array![0.0f64].view());
        assert_abs_diff_eq!(result[0], 35.0 / 32.0, epsilon = 1e-15);
    }

    #[test]
    fn cosine_at_zero() {
        let k = CosineKernel;
        let result = k.evaluate(&array![0.0f64].view());
        assert_abs_diff_eq!(result[0], std::f64::consts::PI / 4.0, epsilon = 1e-15);
    }

    #[test]
    fn all_kernels_nonnegative() {
        let u = array![-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let kernels: Vec<Box<dyn Kernel<f64>>> = vec![
            Box::new(GaussianKernel),
            Box::new(EpanechnikovKernel),
            Box::new(TricubeKernel),
            Box::new(BiweightKernel),
            Box::new(TriweightKernel),
            Box::new(UniformKernel),
            Box::new(CosineKernel),
        ];
        for k in &kernels {
            let result = k.evaluate(&u.view());
            for &v in &result {
                assert!(v >= 0.0, "Kernel produced negative value: {v}");
            }
        }
    }

    #[test]
    fn dyn_kernel_from_name() {
        let k: DynKernel<f64> = DynKernel::from_name("gaussian").unwrap();
        let result = k.evaluate(&array![0.0f64].view());
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(result[0], expected, epsilon = 1e-15);

        assert!(DynKernel::<f64>::from_name("nonexistent").is_err());
    }

    #[test]
    fn dyn_kernel_all_names() {
        let names = [
            "gaussian",
            "epanechnikov",
            "tricube",
            "biweight",
            "triweight",
            "uniform",
            "cosine",
        ];
        for name in &names {
            let k: DynKernel<f64> = DynKernel::from_name(name).unwrap();
            let result = k.evaluate(&array![0.0f64].view());
            assert!(result[0] > 0.0, "{name} at 0 should be positive");
        }
    }

    #[test]
    fn compact_support_kernels_zero_outside() {
        let u = array![1.5f64, 2.0, 3.0];
        let compact: Vec<Box<dyn Kernel<f64>>> = vec![
            Box::new(EpanechnikovKernel),
            Box::new(TricubeKernel),
            Box::new(BiweightKernel),
            Box::new(TriweightKernel),
            Box::new(UniformKernel),
            Box::new(CosineKernel),
        ];
        for k in &compact {
            assert!(k.has_compact_support());
            let result = k.evaluate(&u.view());
            for &v in &result {
                assert_abs_diff_eq!(v, 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn f32_support() {
        let k = GaussianKernel;
        let u = ndarray::array![0.0f32, 1.0f32];
        let result = k.evaluate(&u.view());
        assert!(result[0] > 0.0f32);
        assert!(result[1] > 0.0f32);
        assert!(result[0] > result[1]);
    }
}
