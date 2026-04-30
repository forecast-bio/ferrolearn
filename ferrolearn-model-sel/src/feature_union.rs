//! Feature union: concatenate outputs of multiple transformers horizontally.
//!
//! [`FeatureUnion`] composes several transformers in parallel, fitting each
//! independently on the same data and horizontally concatenating their outputs.
//! This is the horizontal counterpart to a sequential
//! [`Pipeline`](ferrolearn_core::pipeline::Pipeline).
//!
//! # Example
//!
//! ```
//! use ferrolearn_model_sel::feature_union::FeatureUnion;
//! use ferrolearn_core::pipeline::{PipelineTransformer, FittedPipelineTransformer};
//! use ferrolearn_core::FerroError;
//! use ndarray::{Array1, Array2};
//!
//! // Identity transformer for demonstration.
//! struct Identity;
//! impl PipelineTransformer<f64> for Identity {
//!     fn fit_pipeline(
//!         &self, _x: &Array2<f64>, _y: &Array1<f64>,
//!     ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
//!         Ok(Box::new(FittedIdentity))
//!     }
//! }
//! struct FittedIdentity;
//! impl FittedPipelineTransformer<f64> for FittedIdentity {
//!     fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
//!         Ok(x.clone())
//!     }
//! }
//!
//! let fu = FeatureUnion::<f64>::new()
//!     .add("copy1", Box::new(Identity))
//!     .add("copy2", Box::new(Identity));
//!
//! let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 1.0]);
//! let fitted = fu.fit(&x, &y).unwrap();
//! let out = fitted.transform(&x).unwrap();
//! assert_eq!(out.ncols(), 6); // 3 + 3
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;

// ---------------------------------------------------------------------------
// FeatureUnion (unfitted)
// ---------------------------------------------------------------------------

/// Concatenates the outputs of multiple transformers horizontally.
///
/// Each transformer is fitted independently on the same `(X, y)` data, then
/// at transform time each transformer's output is computed and the results
/// are stacked column-wise.
pub struct FeatureUnion<F: Float + Send + Sync + 'static> {
    /// Named transformer steps.
    transformers: Vec<(String, Box<dyn PipelineTransformer<F>>)>,
}

impl<F: Float + Send + Sync + 'static> FeatureUnion<F> {
    /// Create a new empty `FeatureUnion`.
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    /// Add a named transformer step.
    ///
    /// Transformers are fitted independently and their outputs concatenated
    /// in the order they are added.
    #[must_use]
    pub fn add(mut self, name: &str, transformer: Box<dyn PipelineTransformer<F>>) -> Self {
        self.transformers.push((name.to_owned(), transformer));
        self
    }

    /// Return the number of transformer steps.
    #[must_use]
    pub fn n_transformers(&self) -> usize {
        self.transformers.len()
    }

    /// Return the names of all transformer steps.
    #[must_use]
    pub fn transformer_names(&self) -> Vec<&str> {
        self.transformers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Fit all transformers and return a [`FittedFeatureUnion`].
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if no transformers have been added.
    /// - Propagates errors from individual transformer fitting.
    pub fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedFeatureUnion<F>, FerroError> {
        if self.transformers.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "transformers".into(),
                reason: "FeatureUnion must have at least one transformer".into(),
            });
        }

        let mut fitted_transformers: Vec<(String, Box<dyn FittedPipelineTransformer<F>>)> =
            Vec::with_capacity(self.transformers.len());

        for (name, t) in &self.transformers {
            let fitted = t.fit_pipeline(x, y)?;
            fitted_transformers.push((name.clone(), fitted));
        }

        Ok(FittedFeatureUnion {
            transformers: fitted_transformers,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Default for FeatureUnion<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration for FeatureUnion
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for FeatureUnion<F> {
    /// Fit all sub-transformers and return a boxed [`FittedFeatureUnion`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FeatureUnion::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, y)?;
        Ok(Box::new(fitted))
    }
}

// ---------------------------------------------------------------------------
// FittedFeatureUnion
// ---------------------------------------------------------------------------

/// A fitted feature union that transforms data by applying each fitted
/// transformer and horizontally concatenating the results.
///
/// Created by [`FeatureUnion::fit`].
pub struct FittedFeatureUnion<F: Float + Send + Sync + 'static> {
    /// Fitted transformer steps.
    transformers: Vec<(String, Box<dyn FittedPipelineTransformer<F>>)>,
}

impl<F: Float + Send + Sync + 'static> FittedFeatureUnion<F> {
    /// Return the names of all transformer steps.
    #[must_use]
    pub fn transformer_names(&self) -> Vec<&str> {
        self.transformers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Return the number of transformer steps.
    #[must_use]
    pub fn n_transformers(&self) -> usize {
        self.transformers.len()
    }

    /// Transform data by applying each fitted transformer and concatenating
    /// results column-wise.
    ///
    /// # Errors
    ///
    /// Propagates errors from individual transformer transforms. Also returns
    /// [`FerroError::ShapeMismatch`] if transformer outputs have different
    /// numbers of rows.
    pub fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let mut parts: Vec<Array2<F>> = Vec::with_capacity(self.transformers.len());

        for (name, t) in &self.transformers {
            let part = t.transform_pipeline(x)?;
            if let Some(first) = parts.first() {
                if part.nrows() != first.nrows() {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![first.nrows()],
                        actual: vec![part.nrows()],
                        context: format!(
                            "FittedFeatureUnion::transform — transformer '{}' produced different row count",
                            name
                        ),
                    });
                }
            }
            parts.push(part);
        }

        // Horizontal concatenation
        let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
        ndarray::concatenate(Axis(1), &views).map_err(|e| FerroError::ShapeMismatch {
            expected: vec![],
            actual: vec![],
            context: format!("FittedFeatureUnion::transform — concatenation failed: {e}"),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedFeatureUnion<F> {
    /// Transform using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FittedFeatureUnion::transform`].
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
    use ndarray::array;

    // -- Test fixtures -------------------------------------------------------

    /// Identity transformer: returns the input unchanged.
    struct Identity;

    impl PipelineTransformer<f64> for Identity {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedIdentity))
        }
    }

    struct FittedIdentity;

    impl FittedPipelineTransformer<f64> for FittedIdentity {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.clone())
        }
    }

    /// Doubling transformer: multiplies all values by 2.
    struct Doubler;

    impl PipelineTransformer<f64> for Doubler {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedDoubler))
        }
    }

    struct FittedDoubler;

    impl FittedPipelineTransformer<f64> for FittedDoubler {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.mapv(|v| v * 2.0))
        }
    }

    /// Single-column transformer: returns just the first column.
    struct FirstCol;

    impl PipelineTransformer<f64> for FirstCol {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedFirstCol))
        }
    }

    struct FittedFirstCol;

    impl FittedPipelineTransformer<f64> for FittedFirstCol {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.column(0).insert_axis(Axis(1)).to_owned())
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_identity_union() {
        let fu = FeatureUnion::<f64>::new()
            .add("id1", Box::new(Identity))
            .add("id2", Box::new(Identity));

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let fitted = fu.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 2 + 2 = 4 columns
        assert_eq!(out.ncols(), 4);
        assert_eq!(out.nrows(), 2);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[0, 2]], 1.0);
        assert_eq!(out[[0, 3]], 2.0);
    }

    #[test]
    fn test_identity_and_doubler() {
        let fu = FeatureUnion::<f64>::new()
            .add("id", Box::new(Identity))
            .add("double", Box::new(Doubler));

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let fitted = fu.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 4);
        // First 2 cols: identity
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
        // Last 2 cols: doubled
        assert_eq!(out[[0, 2]], 2.0);
        assert_eq!(out[[0, 3]], 4.0);
    }

    #[test]
    fn test_different_width_outputs() {
        let fu = FeatureUnion::<f64>::new()
            .add("all", Box::new(Identity))
            .add("first", Box::new(FirstCol));

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![0.0, 1.0];
        let fitted = fu.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 3 + 1 = 4 columns
        assert_eq!(out.ncols(), 4);
        assert_eq!(out[[0, 3]], 1.0); // first col of original
    }

    #[test]
    fn test_empty_union_error() {
        let fu = FeatureUnion::<f64>::new();
        let x = array![[1.0, 2.0]];
        let y = array![0.0];
        assert!(fu.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_transformer() {
        let fu = FeatureUnion::<f64>::new().add("double", Box::new(Doubler));

        let x = array![[1.0], [2.0]];
        let y = array![0.0, 1.0];
        let fitted = fu.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_eq!(out[[0, 0]], 2.0);
        assert_eq!(out[[1, 0]], 4.0);
    }

    #[test]
    fn test_transformer_names() {
        let fu = FeatureUnion::<f64>::new()
            .add("alpha", Box::new(Identity))
            .add("beta", Box::new(Doubler));
        assert_eq!(fu.transformer_names(), vec!["alpha", "beta"]);
        assert_eq!(fu.n_transformers(), 2);
    }

    #[test]
    fn test_fitted_transformer_names() {
        let fu = FeatureUnion::<f64>::new()
            .add("alpha", Box::new(Identity))
            .add("beta", Box::new(Doubler));

        let x = array![[1.0]];
        let y = array![0.0];
        let fitted = fu.fit(&x, &y).unwrap();
        assert_eq!(fitted.transformer_names(), vec!["alpha", "beta"]);
        assert_eq!(fitted.n_transformers(), 2);
    }

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;

        let fu = FeatureUnion::<f64>::new()
            .add("id", Box::new(Identity))
            .add("double", Box::new(Doubler));

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let fitted_box = fu.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        assert_eq!(out.ncols(), 4);
    }

    #[test]
    fn test_three_transformers() {
        let fu = FeatureUnion::<f64>::new()
            .add("id", Box::new(Identity))
            .add("double", Box::new(Doubler))
            .add("first", Box::new(FirstCol));

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let fitted = fu.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 2 + 2 + 1 = 5 columns
        assert_eq!(out.ncols(), 5);
    }

    #[test]
    fn test_default() {
        let fu = FeatureUnion::<f64>::default();
        assert_eq!(fu.n_transformers(), 0);
    }
}
