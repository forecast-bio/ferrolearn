//! Column transformer: apply different transformers to different column subsets.
//!
//! [`ColumnTransformer`] applies each registered transformer to its designated
//! column subset, then horizontally concatenates the outputs into a single
//! `Array2<f64>`. Columns not captured by any transformer can be dropped or
//! passed through unchanged via the [`Remainder`] policy.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_preprocess::column_transformer::{
//!     ColumnSelector, ColumnTransformer, Remainder,
//! };
//! use ferrolearn_preprocess::{StandardScaler, MinMaxScaler};
//! use ferrolearn_core::Fit;
//! use ferrolearn_core::Transform;
//! use ndarray::array;
//!
//! let x = array![
//!     [1.0_f64, 2.0, 10.0, 20.0],
//!     [3.0, 4.0, 30.0, 40.0],
//!     [5.0, 6.0, 50.0, 60.0],
//! ];
//!
//! let ct = ColumnTransformer::new(
//!     vec![
//!         ("std".into(),  Box::new(StandardScaler::<f64>::new()), ColumnSelector::Indices(vec![0, 1])),
//!         ("mm".into(),   Box::new(MinMaxScaler::<f64>::new()),   ColumnSelector::Indices(vec![2, 3])),
//!     ],
//!     Remainder::Drop,
//! );
//!
//! let fitted = ct.fit(&x, &()).unwrap();
//! let out    = fitted.transform(&x).unwrap();
//! assert_eq!(out.ncols(), 4);
//! assert_eq!(out.nrows(), 3);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// ColumnSelector
// ---------------------------------------------------------------------------

/// Specifies which columns a transformer should operate on.
///
/// Currently the only supported variant is [`Indices`](ColumnSelector::Indices),
/// which selects columns by their zero-based integer positions.
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    /// Select columns by zero-based index.
    ///
    /// The indices do not need to be sorted, but every index must be strictly
    /// less than the number of columns in the input matrix. Duplicate indices
    /// are allowed; the same column will simply appear twice in the sub-matrix
    /// passed to the transformer.
    Indices(Vec<usize>),
}

impl ColumnSelector {
    /// Resolve the selector to a concrete list of column indices.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if any index is out of range
    /// (i.e., `>= n_features`).
    fn resolve(&self, n_features: usize) -> Result<Vec<usize>, FerroError> {
        match self {
            ColumnSelector::Indices(indices) => {
                for &idx in indices {
                    if idx >= n_features {
                        return Err(FerroError::InvalidParameter {
                            name: "ColumnSelector::Indices".into(),
                            reason: format!(
                                "column index {idx} is out of range for input with {n_features} features"
                            ),
                        });
                    }
                }
                Ok(indices.clone())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Remainder
// ---------------------------------------------------------------------------

/// Policy for columns that are not selected by any transformer.
///
/// When at least one column is not covered by any registered transformer,
/// `Remainder` determines what happens to those columns in the output.
#[derive(Debug, Clone)]
pub enum Remainder {
    /// Discard remainder columns — they do not appear in the output.
    Drop,
    /// Pass remainder columns through unchanged, appended after all
    /// transformer outputs.
    Passthrough,
}

// ---------------------------------------------------------------------------
// Helper: extract a sub-matrix by column indices
// ---------------------------------------------------------------------------

/// Build a new `Array2<f64>` containing only the columns at `indices`.
///
/// Columns are emitted in the order they appear in `indices`.
fn select_columns(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let nrows = x.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        return Array2::zeros((nrows, 0));
    }
    let mut out = Array2::zeros((nrows, ncols));
    for (new_j, &old_j) in indices.iter().enumerate() {
        out.column_mut(new_j).assign(&x.column(old_j));
    }
    out
}

/// Horizontally concatenate a slice of `Array2<f64>` matrices.
///
/// All matrices must have the same number of rows.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if row counts differ.
fn hstack(matrices: &[Array2<f64>]) -> Result<Array2<f64>, FerroError> {
    if matrices.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }
    let nrows = matrices[0].nrows();
    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();

    // Handle the case where the first matrix establishes nrows = 0 separately.
    if total_cols == 0 {
        return Ok(Array2::zeros((nrows, 0)));
    }

    let mut out = Array2::zeros((nrows, total_cols));
    let mut col_offset = 0;
    for m in matrices {
        if m.nrows() != nrows {
            return Err(FerroError::ShapeMismatch {
                expected: vec![nrows, m.ncols()],
                actual: vec![m.nrows(), m.ncols()],
                context: "ColumnTransformer hstack: row count mismatch".into(),
            });
        }
        let end = col_offset + m.ncols();
        if m.ncols() > 0 {
            out.slice_mut(ndarray::s![.., col_offset..end]).assign(m);
        }
        col_offset = end;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ColumnTransformer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted column transformer.
///
/// Applies each registered transformer to its designated column subset, then
/// horizontally concatenates all outputs. The [`Remainder`] policy controls
/// what happens to columns not covered by any transformer.
///
/// # Transformer order
///
/// Transformers are applied and their outputs concatenated in the order they
/// were registered. Remainder columns (when
/// `remainder = `[`Remainder::Passthrough`]) are appended last.
///
/// # Overlapping selections
///
/// Each transformer receives its own copy of the selected columns, so
/// overlapping `ColumnSelector`s are fully supported.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::column_transformer::{
///     ColumnSelector, ColumnTransformer, Remainder,
/// };
/// use ferrolearn_preprocess::StandardScaler;
/// use ferrolearn_core::Fit;
/// use ferrolearn_core::Transform;
/// use ndarray::array;
///
/// let x = array![[1.0_f64, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]];
/// let ct = ColumnTransformer::new(
///     vec![("scaler".into(), Box::new(StandardScaler::<f64>::new()), ColumnSelector::Indices(vec![0, 1]))],
///     Remainder::Passthrough,
/// );
/// let fitted = ct.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // 2 scaled columns + 1 passthrough column
/// assert_eq!(out.ncols(), 3);
/// ```
pub struct ColumnTransformer {
    /// Named transformer steps with their column selectors.
    transformers: Vec<(String, Box<dyn PipelineTransformer<f64>>, ColumnSelector)>,
    /// Policy for columns not covered by any transformer.
    remainder: Remainder,
}

impl ColumnTransformer {
    /// Create a new `ColumnTransformer`.
    ///
    /// # Parameters
    ///
    /// - `transformers`: A list of `(name, transformer, selector)` triples.
    /// - `remainder`: Policy for uncovered columns (`Drop` or `Passthrough`).
    #[must_use]
    pub fn new(
        transformers: Vec<(String, Box<dyn PipelineTransformer<f64>>, ColumnSelector)>,
        remainder: Remainder,
    ) -> Self {
        Self {
            transformers,
            remainder,
        }
    }
}

// ---------------------------------------------------------------------------
// Fit implementation
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for ColumnTransformer {
    type Fitted = FittedColumnTransformer;
    type Error = FerroError;

    /// Fit each transformer on its selected column subset.
    ///
    /// Validates that all selected column indices are within bounds before
    /// fitting any transformer.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if any column index is out of range.
    /// - Propagates any error returned by an individual transformer's
    ///   `fit_pipeline` call.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedColumnTransformer, FerroError> {
        let n_features = x.ncols();
        let n_rows = x.nrows();

        // A dummy y vector required by PipelineTransformer::fit_pipeline.
        let dummy_y = Array1::<f64>::zeros(n_rows);

        // Resolve all selectors up front to validate indices eagerly.
        let mut resolved_selectors: Vec<Vec<usize>> = Vec::with_capacity(self.transformers.len());
        for (name, _, selector) in &self.transformers {
            let indices = selector.resolve(n_features).map_err(|e| {
                // Enrich the error with the transformer name.
                FerroError::InvalidParameter {
                    name: format!("ColumnTransformer step '{name}'"),
                    reason: e.to_string(),
                }
            })?;
            resolved_selectors.push(indices);
        }

        // Build the set of covered column indices (for remainder computation).
        let covered: std::collections::HashSet<usize> = resolved_selectors
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        let remainder_indices: Vec<usize> =
            (0..n_features).filter(|c| !covered.contains(c)).collect();

        // Fit each transformer on its sub-matrix.
        let mut fitted_transformers: Vec<FittedSubTransformer> =
            Vec::with_capacity(self.transformers.len());

        for ((name, transformer, _), indices) in
            self.transformers.iter().zip(resolved_selectors.into_iter())
        {
            let sub_x = select_columns(x, &indices);
            let fitted = transformer.fit_pipeline(&sub_x, &dummy_y)?;
            fitted_transformers.push((name.clone(), fitted, indices));
        }

        Ok(FittedColumnTransformer {
            fitted_transformers,
            remainder: self.remainder.clone(),
            remainder_indices,
            n_features_in: n_features,
        })
    }
}

// ---------------------------------------------------------------------------
// PipelineTransformer implementation
// ---------------------------------------------------------------------------

impl PipelineTransformer<f64> for ColumnTransformer {
    /// Fit the column transformer using the pipeline interface.
    ///
    /// The `y` argument is ignored; it exists only for API compatibility.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

// ---------------------------------------------------------------------------
// FittedColumnTransformer
// ---------------------------------------------------------------------------

/// A named, fitted sub-transformer with its column indices.
type FittedSubTransformer = (String, Box<dyn FittedPipelineTransformer<f64>>, Vec<usize>);

/// A fitted column transformer holding fitted sub-transformers and metadata.
///
/// Created by calling [`Fit::fit`] on a [`ColumnTransformer`].
/// Implements [`Transform<Array2<f64>>`] to apply the fitted transformers and
/// concatenate their outputs, as well as [`FittedPipelineTransformer`] for use
/// inside a [`ferrolearn_core::pipeline::Pipeline`].
pub struct FittedColumnTransformer {
    /// Fitted transformers with their associated column indices.
    fitted_transformers: Vec<FittedSubTransformer>,
    /// Remainder policy from the original [`ColumnTransformer`].
    remainder: Remainder,
    /// Column indices not covered by any transformer.
    remainder_indices: Vec<usize>,
    /// Number of input features seen during fitting.
    n_features_in: usize,
}

impl FittedColumnTransformer {
    /// Return the number of input features seen during fitting.
    #[must_use]
    pub fn n_features_in(&self) -> usize {
        self.n_features_in
    }

    /// Return the names of all registered transformer steps.
    #[must_use]
    pub fn transformer_names(&self) -> Vec<&str> {
        self.fitted_transformers
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect()
    }

    /// Return the remainder column indices (columns not selected by any transformer).
    #[must_use]
    pub fn remainder_indices(&self) -> &[usize] {
        &self.remainder_indices
    }
}

// ---------------------------------------------------------------------------
// Transform implementation
// ---------------------------------------------------------------------------

impl Transform<Array2<f64>> for FittedColumnTransformer {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Transform data by applying each fitted transformer to its column subset,
    /// then horizontally concatenating all outputs.
    ///
    /// When `remainder = Passthrough`, the unselected columns are appended
    /// after all transformer outputs. When `remainder = Drop`, they are
    /// discarded.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if the input does not have
    ///   `n_features_in` columns.
    /// - Propagates any error from individual transformer `transform_pipeline`
    ///   calls.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedColumnTransformer::transform".into(),
            });
        }

        let mut parts: Vec<Array2<f64>> = Vec::with_capacity(self.fitted_transformers.len() + 1);

        for (_, fitted, indices) in &self.fitted_transformers {
            let sub_x = select_columns(x, indices);
            let transformed = fitted.transform_pipeline(&sub_x)?;
            parts.push(transformed);
        }

        // Append remainder columns if requested.
        if matches!(self.remainder, Remainder::Passthrough) && !self.remainder_indices.is_empty() {
            let remainder_sub = select_columns(x, &self.remainder_indices);
            parts.push(remainder_sub);
        }

        hstack(&parts)
    }
}

// ---------------------------------------------------------------------------
// FittedPipelineTransformer implementation
// ---------------------------------------------------------------------------

impl FittedPipelineTransformer<f64> for FittedColumnTransformer {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// make_column_transformer convenience function
// ---------------------------------------------------------------------------

/// Convenience function to build a [`ColumnTransformer`] with auto-generated
/// step names.
///
/// Steps are named `"transformer-0"`, `"transformer-1"`, etc.
///
/// # Parameters
///
/// - `transformers`: A list of `(transformer, selector)` pairs.
/// - `remainder`: Policy for uncovered columns (`Drop` or `Passthrough`).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::column_transformer::{
///     make_column_transformer, ColumnSelector, Remainder,
/// };
/// use ferrolearn_preprocess::StandardScaler;
/// use ferrolearn_core::Fit;
/// use ferrolearn_core::Transform;
/// use ndarray::array;
///
/// let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0]];
/// let ct = make_column_transformer(
///     vec![(Box::new(StandardScaler::<f64>::new()), ColumnSelector::Indices(vec![0, 1]))],
///     Remainder::Drop,
/// );
/// let fitted = ct.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 2);
/// ```
#[must_use]
pub fn make_column_transformer(
    transformers: Vec<(Box<dyn PipelineTransformer<f64>>, ColumnSelector)>,
    remainder: Remainder,
) -> ColumnTransformer {
    let named: Vec<(String, Box<dyn PipelineTransformer<f64>>, ColumnSelector)> = transformers
        .into_iter()
        .enumerate()
        .map(|(i, (t, s))| (format!("transformer-{i}"), t, s))
        .collect();
    ColumnTransformer::new(named, remainder)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ferrolearn_core::pipeline::{Pipeline, PipelineEstimator};
    use ndarray::{Array2, array};

    use crate::{MinMaxScaler, StandardScaler};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a simple 4-column test matrix (rows = 4, cols = 4).
    fn make_x() -> Array2<f64> {
        array![
            [1.0, 2.0, 10.0, 20.0],
            [2.0, 4.0, 20.0, 40.0],
            [3.0, 6.0, 30.0, 60.0],
            [4.0, 8.0, 40.0, 80.0],
        ]
    }

    // -----------------------------------------------------------------------
    // 1. Basic 2-transformer usage
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_two_transformers_drop_remainder() {
        let x = make_x(); // 4×4
        let ct = ColumnTransformer::new(
            vec![
                (
                    "std".into(),
                    Box::new(StandardScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0, 1]),
                ),
                (
                    "mm".into(),
                    Box::new(MinMaxScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![2, 3]),
                ),
            ],
            Remainder::Drop,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        // All 4 columns covered → no remainder; output is 4 cols
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4);
    }

    // -----------------------------------------------------------------------
    // 2. Remainder::Drop drops uncovered columns
    // -----------------------------------------------------------------------

    #[test]
    fn test_remainder_drop() {
        let x = make_x(); // 4×4
        // Only cover cols 0 and 1 — cols 2 and 3 should be dropped.
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Drop,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 2, "uncovered cols should be dropped");
    }

    // -----------------------------------------------------------------------
    // 3. Remainder::Passthrough passes uncovered columns through unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn test_remainder_passthrough() {
        let x = make_x(); // 4×4
        // Only cover cols 0 and 1 — cols 2 and 3 should pass through.
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Passthrough,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4, "passthrough: 2 transformed + 2 remainder");

        // The last 2 columns should be the original cols 2 and 3.
        for i in 0..4 {
            assert_abs_diff_eq!(out[[i, 2]], x[[i, 2]], epsilon = 1e-12);
            assert_abs_diff_eq!(out[[i, 3]], x[[i, 3]], epsilon = 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Invalid column index (out of range)
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_column_index_out_of_range() {
        let x = make_x(); // 4×4 — valid indices are 0..3
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 99]), // 99 is out of range
            )],
            Remainder::Drop,
        );
        let result = ct.fit(&x, &());
        assert!(result.is_err(), "expected error for out-of-range index");
    }

    // -----------------------------------------------------------------------
    // 5. Empty transformer list with Remainder::Drop
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_transformer_list_drop() {
        let x = make_x();
        let ct = ColumnTransformer::new(vec![], Remainder::Drop);
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // No transformers, remainder dropped → empty output
        assert_eq!(out.nrows(), 0, "hstack of nothing with no passthrough");
    }

    // -----------------------------------------------------------------------
    // 6. Empty transformer list with Remainder::Passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_transformer_list_passthrough() {
        let x = make_x(); // 4×4
        let ct = ColumnTransformer::new(vec![], Remainder::Passthrough);
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // No transformers, all columns pass through unchanged.
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(out[[i, j]], x[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Overlapping column selections
    // -----------------------------------------------------------------------

    #[test]
    fn test_overlapping_column_selections() {
        let x = make_x(); // 4×4
        // Both transformers select col 0 (overlapping is allowed).
        let ct = ColumnTransformer::new(
            vec![
                (
                    "std1".into(),
                    Box::new(StandardScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0, 1]),
                ),
                (
                    "mm1".into(),
                    Box::new(MinMaxScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0, 2]), // col 0 also used here
                ),
            ],
            Remainder::Drop,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        // Output: 2 cols from std1 + 2 cols from mm1 = 4 cols
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4);
    }

    // -----------------------------------------------------------------------
    // 8. Single transformer
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_transformer() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let ct = ColumnTransformer::new(
            vec![(
                "mm".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Drop,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        assert_eq!(out.nrows(), 3);
        assert_eq!(out.ncols(), 2);

        // MinMax on cols 0 and 1: first row → 0.0, last row → 1.0
        assert_abs_diff_eq!(out[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[2, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[2, 1]], 1.0, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------------
    // 9. make_column_transformer convenience function
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_column_transformer_auto_names() {
        let x = make_x();
        let ct = make_column_transformer(
            vec![
                (
                    Box::new(StandardScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0, 1]),
                ),
                (
                    Box::new(MinMaxScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![2, 3]),
                ),
            ],
            Remainder::Drop,
        );

        let fitted = ct.fit(&x, &()).unwrap();
        assert_eq!(
            fitted.transformer_names(),
            vec!["transformer-0", "transformer-1"]
        );

        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4);
    }

    // -----------------------------------------------------------------------
    // 10. Pipeline integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_integration() {
        // Wrap a ColumnTransformer as a pipeline step.
        let x = make_x();
        let y = Array1::<f64>::zeros(4);

        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1, 2, 3]),
            )],
            Remainder::Drop,
        );

        // Use a trivial estimator that sums rows.
        struct SumEstimator;
        impl PipelineEstimator<f64> for SumEstimator {
            fn fit_pipeline(
                &self,
                _x: &Array2<f64>,
                _y: &Array1<f64>,
            ) -> Result<Box<dyn ferrolearn_core::pipeline::FittedPipelineEstimator<f64>>, FerroError>
            {
                Ok(Box::new(FittedSum))
            }
        }
        struct FittedSum;
        impl ferrolearn_core::pipeline::FittedPipelineEstimator<f64> for FittedSum {
            fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
                let sums: Vec<f64> = x.rows().into_iter().map(|r| r.sum()).collect();
                Ok(Array1::from_vec(sums))
            }
        }

        let pipeline = Pipeline::new()
            .transform_step("ct", Box::new(ct))
            .estimator_step("sum", Box::new(SumEstimator));

        use ferrolearn_core::Fit as _;
        let fitted_pipeline = pipeline.fit(&x, &y).unwrap();

        use ferrolearn_core::Predict as _;
        let preds = fitted_pipeline.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);
    }

    // -----------------------------------------------------------------------
    // 11. Transform shape correctness — number of output columns
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_shape_all_selected_drop() {
        let x = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let ct = ColumnTransformer::new(
            vec![
                (
                    "s".into(),
                    Box::new(StandardScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0]),
                ),
                (
                    "m".into(),
                    Box::new(MinMaxScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![1, 2]),
                ),
            ],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    // -----------------------------------------------------------------------
    // 12. Transform shape — partial selection + passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_shape_partial_passthrough() {
        // 5-column input, transform 2 cols, passthrough 3
        let x =
            Array2::<f64>::from_shape_vec((3, 5), (1..=15).map(|v| v as f64).collect()).unwrap();
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Passthrough,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[3, 5]);
    }

    // -----------------------------------------------------------------------
    // 13. n_features_in accessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_n_features_in() {
        let x = make_x(); // 4×4
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0]),
            )],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_features_in(), 4);
    }

    // -----------------------------------------------------------------------
    // 14. Shape mismatch on transform (wrong number of columns)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shape_mismatch_on_transform() {
        let x = make_x(); // 4×4
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();

        // Now pass a matrix with only 2 columns — should fail.
        let x_bad = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let result = fitted.transform(&x_bad);
        assert!(result.is_err(), "expected shape mismatch error");
    }

    // -----------------------------------------------------------------------
    // 15. remainder_indices accessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_remainder_indices_accessor() {
        let x = make_x(); // 4×4
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 2]),
            )],
            Remainder::Passthrough,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        // Remainder should be cols 1 and 3.
        assert_eq!(fitted.remainder_indices(), &[1, 3]);
    }

    // -----------------------------------------------------------------------
    // 16. StandardScaler output values are correct (zero-mean)
    // -----------------------------------------------------------------------

    #[test]
    fn test_standard_scaler_zero_mean_in_output() {
        let x = array![[1.0_f64, 100.0, 0.5], [2.0, 200.0, 1.5], [3.0, 300.0, 2.5],];
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        // Cols 0 and 1 of output should have mean ≈ 0.
        for j in 0..2 {
            let mean: f64 = out.column(j).iter().sum::<f64>() / 3.0;
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // 17. MinMaxScaler output values are in [0, 1]
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_max_values_in_range() {
        let x = make_x();
        let ct = ColumnTransformer::new(
            vec![(
                "mm".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1, 2, 3]),
            )],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();

        for j in 0..4 {
            let col_min = out.column(j).iter().copied().fold(f64::INFINITY, f64::min);
            let col_max = out
                .column(j)
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            assert_abs_diff_eq!(col_min, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(col_max, 1.0, epsilon = 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // 18. Pipeline transformer interface (fit_pipeline / transform_pipeline)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_transformer_interface() {
        let x = make_x();
        let y = Array1::<f64>::zeros(4);
        let ct = ColumnTransformer::new(
            vec![(
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Passthrough,
        );
        let fitted_box = ct.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 4);
    }

    // -----------------------------------------------------------------------
    // 19. Remainder passthrough values are identical to input values
    // -----------------------------------------------------------------------

    #[test]
    fn test_passthrough_values_are_exact() {
        let x = array![[10.0_f64, 20.0, 30.0], [40.0, 50.0, 60.0],];
        // Only transform col 0; cols 1 and 2 pass through.
        let ct = ColumnTransformer::new(
            vec![(
                "mm".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0]),
            )],
            Remainder::Passthrough,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // out[:, 1] == x[:, 1] and out[:, 2] == x[:, 2]
        assert_abs_diff_eq!(out[[0, 1]], 20.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[1, 1]], 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 2]], 30.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[1, 2]], 60.0, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // 20. Transformer names from explicit ColumnTransformer::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_transformer_names_explicit() {
        let x = make_x();
        let ct = ColumnTransformer::new(
            vec![
                (
                    "alpha".into(),
                    Box::new(StandardScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![0]),
                ),
                (
                    "beta".into(),
                    Box::new(MinMaxScaler::<f64>::new()),
                    ColumnSelector::Indices(vec![1]),
                ),
            ],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        assert_eq!(fitted.transformer_names(), vec!["alpha", "beta"]);
    }

    // -----------------------------------------------------------------------
    // 21. make_column_transformer with single step
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_column_transformer_single() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let ct = make_column_transformer(
            vec![(
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            )],
            Remainder::Drop,
        );
        let fitted = ct.fit(&x, &()).unwrap();
        assert_eq!(fitted.transformer_names(), vec!["transformer-0"]);
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
    }

    // -----------------------------------------------------------------------
    // 22. Edge case: all columns as remainder with Passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_remainder_passthrough_unchanged() {
        let x = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let ct = ColumnTransformer::new(vec![], Remainder::Passthrough);
        let fitted = ct.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(out[[i, j]], x[[i, j]], epsilon = 1e-12);
            }
        }
    }
}
