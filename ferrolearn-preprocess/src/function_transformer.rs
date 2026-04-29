//! Function transformer: apply a user-provided function element-wise.
//!
//! Wraps any `Fn(F) -> F` callable and applies it to every element in the
//! input matrix. This is useful for applying non-standard transformations
//! such as `ln`, `sqrt`, or custom domain-specific functions.
//!
//! This transformer is **stateless** — no fitting is required. Call
//! [`Transform::transform`] directly.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ndarray::Array2;
use num_traits::Float;

// ---------------------------------------------------------------------------
// FunctionTransformer
// ---------------------------------------------------------------------------

/// A stateless element-wise function transformer.
///
/// Wraps a boxed `Fn(F) -> F` closure and applies it to every element in
/// the input matrix.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::function_transformer::FunctionTransformer;
/// use ferrolearn_core::traits::Transform;
/// use ndarray::array;
///
/// // Apply natural logarithm element-wise (values must be > 0)
/// let ft = FunctionTransformer::<f64>::new(|v| v.ln());
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let out = ft.transform(&x).unwrap();
/// ```
pub struct FunctionTransformer<F> {
    func: Box<dyn Fn(F) -> F + Send + Sync>,
}

impl<F: Float + Send + Sync + 'static> FunctionTransformer<F> {
    /// Create a new `FunctionTransformer` with the given function.
    ///
    /// The function will be applied element-wise to the input matrix.
    pub fn new<Func>(func: Func) -> Self
    where
        Func: Fn(F) -> F + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
        }
    }
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FunctionTransformer<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTransformer")
            .field("func", &"<fn(F) -> F>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FunctionTransformer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Apply the stored function to every element of `x`.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error for well-formed inputs.
    /// Note: if the user-provided function produces NaN or infinity for
    /// certain inputs, those values will appear in the output without error.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let out = x.mapv(|v| (self.func)(v));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_identity_function() {
        let ft = FunctionTransformer::<f64>::new(|v| v);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let out = ft.transform(&x).unwrap();
        for (a, b) in x.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_sqrt_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.sqrt());
        let x = array![[1.0, 4.0], [9.0, 16.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ln_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.ln());
        let x = array![[1.0, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10); // ln(1) = 0
        assert_abs_diff_eq!(out[[0, 1]], 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_negate_function() {
        let ft = FunctionTransformer::<f64>::new(|v| -v);
        let x = array![[1.0, -2.0, 3.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_function() {
        let ft = FunctionTransformer::<f64>::new(|_| 42.0);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = ft.transform(&x).unwrap();
        for v in &out {
            assert_abs_diff_eq!(*v, 42.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_preserves_shape() {
        let ft = FunctionTransformer::<f64>::new(|v| v * 2.0);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let out = ft.transform(&x).unwrap();
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_clamp_function() {
        let ft = FunctionTransformer::<f64>::new(|v: f64| v.clamp(0.0, 1.0));
        let x = array![[-1.0, 0.5, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 2]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_f32_function() {
        let ft = FunctionTransformer::<f32>::new(|v: f32| v * 2.0);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0]];
        let out = ft.transform(&x).unwrap();
        assert!((out[[0, 0]] - 2.0f32).abs() < 1e-6);
        assert!((out[[1, 1]] - 8.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_closure_captures_environment() {
        let scale = 3.0_f64;
        let ft = FunctionTransformer::<f64>::new(move |v| v * scale);
        let x = array![[1.0, 2.0]];
        let out = ft.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_matrix() {
        let ft = FunctionTransformer::<f64>::new(|v| v);
        let x: Array2<f64> = Array2::zeros((0, 3));
        let out = ft.transform(&x).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }
}
