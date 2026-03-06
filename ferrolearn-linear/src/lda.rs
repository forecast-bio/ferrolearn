//! Linear Discriminant Analysis (LDA).
//!
//! LDA is both a supervised dimensionality reduction technique and a
//! linear classifier. It finds the directions that maximise the separation
//! between classes while minimising within-class scatter.
//!
//! # Algorithm
//!
//! 1. Compute class means `μ_c` and the overall mean `μ`.
//! 2. Compute the within-class scatter matrix
//!    `Sw = Σ_c Σ_{x ∈ c} (x - μ_c)(x - μ_c)^T`.
//! 3. Compute the between-class scatter matrix
//!    `Sb = Σ_c n_c (μ_c - μ)(μ_c - μ)^T`.
//! 4. Solve the generalised eigenvalue problem `Sw⁻¹ Sb v = λ v`.
//! 5. Project data onto the top-`k` eigenvectors.
//!
//! The number of discriminant directions is at most `min(n_classes - 1, n_features)`.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::lda::LDA;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let lda = LDA::new(Some(1));
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 1.0, 1.5, 1.2, 1.2, 0.8, 5.0, 5.0, 5.5, 4.8, 4.8, 5.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//! let fitted = lda.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::{Array1, Array2};
use num_traits::{Float, NumCast};

// ---------------------------------------------------------------------------
// LDA (unfitted)
// ---------------------------------------------------------------------------

/// Linear Discriminant Analysis configuration.
///
/// Holds hyperparameters. Calling [`Fit::fit`] computes the discriminant
/// directions and returns a [`FittedLDA`].
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LDA<F> {
    /// Number of discriminant components to retain.
    ///
    /// If `None`, defaults to `min(n_classes - 1, n_features)` at fit time.
    n_components: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> LDA<F> {
    /// Create a new `LDA`.
    ///
    /// - `n_components`: number of discriminant directions to retain.
    ///   Pass `None` to use `min(n_classes - 1, n_features)`.
    #[must_use]
    pub fn new(n_components: Option<usize>) -> Self {
        Self {
            n_components,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the configured number of components (may be `None`).
    #[must_use]
    pub fn n_components(&self) -> Option<usize> {
        self.n_components
    }
}

impl<F: Float + Send + Sync + 'static> Default for LDA<F> {
    fn default() -> Self {
        Self::new(None)
    }
}

// ---------------------------------------------------------------------------
// FittedLDA
// ---------------------------------------------------------------------------

/// A fitted LDA model.
///
/// Created by calling [`Fit::fit`] on an [`LDA`]. Implements:
/// - [`Transform<Array2<F>>`] — project data onto discriminant axes.
/// - [`Predict<Array2<F>>`] — classify by nearest centroid in projected space.
#[derive(Debug, Clone)]
pub struct FittedLDA<F> {
    /// Projection matrix, shape `(n_features, n_components)`.
    ///
    /// New data is projected via `X @ scalings`.
    scalings: Array2<F>,

    /// Class means in the projected space, shape `(n_classes, n_components)`.
    means: Array2<F>,

    /// Ratio of explained variance per discriminant direction.
    explained_variance_ratio: Array1<F>,

    /// Class labels corresponding to rows of `means`.
    classes: Vec<usize>,

    /// Number of features seen during fitting.
    n_features: usize,
}

impl<F: Float + Send + Sync + 'static> FittedLDA<F> {
    /// Projection (scalings) matrix, shape `(n_features, n_components)`.
    #[must_use]
    pub fn scalings(&self) -> &Array2<F> {
        &self.scalings
    }

    /// Class centroids in the projected space, shape `(n_classes, n_components)`.
    #[must_use]
    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    /// Explained-variance ratio per discriminant direction.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio
    }

    /// Sorted class labels as seen during fitting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers (generic over F)
// ---------------------------------------------------------------------------

/// Jacobi symmetric eigendecomposition.
///
/// Returns `(eigenvalues, eigenvectors_columns)` — column `i` is the
/// eigenvector for `eigenvalues[i]`.  Eigenvalues are **not** sorted.
fn jacobi_eigen_f<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }
    let tol = F::from(1e-12).unwrap_or(F::epsilon());

    for _ in 0..max_iter {
        // Find the largest off-diagonal entry.
        let mut max_off = F::zero();
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = mat[[i, j]].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < tol {
            let eigenvalues = Array1::from_shape_fn(n, |i| mat[[i, i]]);
            return Ok((eigenvalues, v));
        }
        let app = mat[[p, p]];
        let aqq = mat[[q, q]];
        let apq = mat[[p, q]];
        let two = F::from(2.0).unwrap();
        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or(F::one())
        } else {
            let tau = (aqq - app) / (two * apq);
            let t = if tau >= F::zero() {
                F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            } else {
                -F::one() / (tau.abs() + (F::one() + tau * tau).sqrt())
            };
            t.atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        let mut new_mat = mat.clone();
        for i in 0..n {
            if i != p && i != q {
                let mip = mat[[i, p]];
                let miq = mat[[i, q]];
                new_mat[[i, p]] = c * mip - s * miq;
                new_mat[[p, i]] = new_mat[[i, p]];
                new_mat[[i, q]] = s * mip + c * miq;
                new_mat[[q, i]] = new_mat[[i, q]];
            }
        }
        new_mat[[p, p]] = c * c * app - two * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + two * s * c * apq + c * c * aqq;
        new_mat[[p, q]] = F::zero();
        new_mat[[q, p]] = F::zero();
        mat = new_mat;
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }
    Err(FerroError::ConvergenceFailure {
        iterations: max_iter,
        message: "Jacobi eigendecomposition did not converge (LDA)".into(),
    })
}

/// Gaussian elimination with partial pivoting to solve `A x = b`.
fn gaussian_solve_f<F: Float>(
    n: usize,
    a: &Array2<F>,
    b: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }
    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < F::from(1e-12).unwrap_or(F::epsilon()) {
            return Err(FerroError::NumericalInstability {
                message: "singular matrix during LDA inversion".into(),
            });
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < F::from(1e-12).unwrap_or(F::epsilon()) {
            return Err(FerroError::NumericalInstability {
                message: "near-zero pivot during LDA back substitution".into(),
            });
        }
        x[i] = sum / aug[[i, i]];
    }
    Ok(x)
}

/// Compute `Sw⁻¹ @ Sb` column by column.
///
/// Returns the matrix `M = Sw⁻¹ Sb` of shape `(n, n)`.
fn sw_inv_sb<F: Float + Send + Sync + 'static>(
    sw: &Array2<F>,
    sb: &Array2<F>,
) -> Result<Array2<F>, FerroError> {
    let n = sw.nrows();
    let mut result = Array2::<F>::zeros((n, n));
    for j in 0..n {
        let col_sb = Array1::from_shape_fn(n, |i| sb[[i, j]]);
        let col = gaussian_solve_f(n, sw, &col_sb)?;
        for i in 0..n {
            result[[i, j]] = col[i];
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for LDA<F> {
    type Fitted = FittedLDA<F>;
    type Error = FerroError;

    /// Fit the LDA model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples are provided.
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds the
    ///   maximum allowed (`min(n_classes - 1, n_features)`).
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::NumericalInstability`] if `Sw` is singular.
    /// - [`FerroError::ConvergenceFailure`] if eigendecomposition does not converge.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedLDA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "LDA: y length must match number of rows in X".into(),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LDA requires at least 2 samples".into(),
            });
        }

        // Gather sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_classes,
                context: "LDA requires at least 2 distinct classes".into(),
            });
        }

        // Determine effective n_components.
        let max_components = (n_classes - 1).min(n_features);
        let n_comp = match self.n_components {
            None => max_components,
            Some(0) => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: "must be at least 1".into(),
                });
            }
            Some(k) if k > max_components => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: format!(
                        "n_components ({k}) exceeds max allowed ({max_components} = min(n_classes-1, n_features))"
                    ),
                });
            }
            Some(k) => k,
        };

        // --- Step 1: compute overall mean and per-class means ----------------
        let n_f = F::from(n_samples).unwrap();
        let mut overall_mean = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let s = col.iter().copied().fold(F::zero(), |a, b| a + b);
            overall_mean[j] = s / n_f;
        }

        // class_means[c] = mean of samples in class c
        let mut class_means: Vec<Array1<F>> = Vec::with_capacity(n_classes);
        let mut class_counts: Vec<usize> = Vec::with_capacity(n_classes);
        for &cls in &classes {
            let mut mean = Array1::<F>::zeros(n_features);
            let mut cnt = 0usize;
            for (i, &label) in y.iter().enumerate() {
                if label == cls {
                    for j in 0..n_features {
                        mean[j] = mean[j] + x[[i, j]];
                    }
                    cnt += 1;
                }
            }
            if cnt == 0 {
                return Err(FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: format!("LDA: class {cls} has no samples"),
                });
            }
            let cnt_f = F::from(cnt).unwrap();
            mean.mapv_inplace(|v| v / cnt_f);
            class_means.push(mean);
            class_counts.push(cnt);
        }

        // --- Step 2: within-class scatter Sw ----------------------------------
        let mut sw = Array2::<F>::zeros((n_features, n_features));
        for (ci, &cls) in classes.iter().enumerate() {
            let mu_c = &class_means[ci];
            for (i, &label) in y.iter().enumerate() {
                if label == cls {
                    // diff = x[i] - mu_c
                    let diff: Vec<F> = (0..n_features).map(|j| x[[i, j]] - mu_c[j]).collect();
                    for r in 0..n_features {
                        for c in 0..n_features {
                            sw[[r, c]] = sw[[r, c]] + diff[r] * diff[c];
                        }
                    }
                }
            }
        }

        // Add a small regularisation to Sw to avoid singularity.
        let reg = F::from(1e-6).unwrap();
        for i in 0..n_features {
            sw[[i, i]] = sw[[i, i]] + reg;
        }

        // --- Step 3: between-class scatter Sb ---------------------------------
        let mut sb = Array2::<F>::zeros((n_features, n_features));
        for (ci, &nc) in class_counts.iter().enumerate() {
            let nc_f = F::from(nc).unwrap();
            let diff: Vec<F> = (0..n_features)
                .map(|j| class_means[ci][j] - overall_mean[j])
                .collect();
            for r in 0..n_features {
                for c in 0..n_features {
                    sb[[r, c]] = sb[[r, c]] + nc_f * diff[r] * diff[c];
                }
            }
        }

        // --- Step 4: solve generalised eigenvalue problem Sw⁻¹ Sb v = λ v ----
        let m = sw_inv_sb(&sw, &sb)?;
        let max_jacobi = n_features * n_features * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_f(&m, max_jacobi)?;

        // Sort eigenvalues descending.
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Clamp negative eigenvalues.
        let total_ev: F = eigenvalues
            .iter()
            .copied()
            .map(|v| if v > F::zero() { v } else { F::zero() })
            .fold(F::zero(), |a, b| a + b);

        // --- Step 5: build scalings matrix (n_features × n_comp) -------------
        let mut scalings = Array2::<F>::zeros((n_features, n_comp));
        let mut explained_variance_ratio = Array1::<F>::zeros(n_comp);
        for (k, &idx) in indices.iter().take(n_comp).enumerate() {
            let ev = eigenvalues[idx];
            let ev_clamped = if ev > F::zero() { ev } else { F::zero() };
            explained_variance_ratio[k] = if total_ev > F::zero() {
                ev_clamped / total_ev
            } else {
                F::zero()
            };
            for j in 0..n_features {
                scalings[[j, k]] = eigenvectors[[j, idx]];
            }
        }

        // --- Project class means into the discriminant space -----------------
        // means[c, k] = class_means[c] · scalings[:, k]
        let mut means = Array2::<F>::zeros((n_classes, n_comp));
        for ci in 0..n_classes {
            let mu_row = class_means[ci].view();
            for k in 0..n_comp {
                let mut dot = F::zero();
                for j in 0..n_features {
                    dot = dot + mu_row[j] * scalings[[j, k]];
                }
                means[[ci, k]] = dot;
            }
        }

        Ok(FittedLDA {
            scalings,
            means,
            explained_variance_ratio,
            classes,
            n_features,
        })
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedLDA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project `x` onto the discriminant axes: `X @ scalings`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols()` does not match the
    /// number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedLDA::transform".into(),
            });
        }
        Ok(x.dot(&self.scalings))
    }
}

// ---------------------------------------------------------------------------
// Predict (nearest centroid in projected space)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLDA<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Classify samples by nearest centroid in the projected space.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let projected = self.transform(x)?;
        let n_samples = projected.nrows();
        let n_comp = projected.ncols();
        let n_classes = self.classes.len();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class = 0usize;
            let mut best_dist = F::infinity();
            for ci in 0..n_classes {
                let mut dist = F::zero();
                for k in 0..n_comp {
                    let d = projected[[i, k]] - self.means[[ci, k]];
                    dist = dist + d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_class = ci;
                }
            }
            predictions[i] = self.classes[best_class];
        }
        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineEstimator<F> for LDA<F> {
    /// Fit LDA using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedLDAPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedLDAPipeline<F>(FittedLDA<F>);

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedLDAPipeline<F> {
    /// Predict via the pipeline interface, returning float class labels.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| NumCast::from(v).unwrap_or(F::nan())))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn linearly_separable_2d() -> (Array2<f64>, Array1<usize>) {
        // Two well-separated Gaussian clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.2, 0.8, 0.9, 1.1, 1.3, // class 0
                6.0, 6.0, 6.2, 5.8, 5.9, 6.1, 6.3, 5.7, // class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn three_class_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.1, 0.1, 0.5, // class 0
                5.0, 0.0, 5.2, 0.3, 4.8, 0.1, // class 1
                0.0, 5.0, 0.1, 5.2, 0.3, 4.8, // class 2
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, y)
    }

    // ------------------------------------------------------------------

    #[test]
    fn test_lda_fit_returns_fitted() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.scalings().ncols(), 1);
        assert_eq!(fitted.scalings().nrows(), 2);
    }

    #[test]
    fn test_lda_default_n_components() {
        // With 2 classes the default n_components = min(1, n_features) = 1.
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::default();
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.scalings().ncols(), 1);
    }

    #[test]
    fn test_lda_transform_shape() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let proj = fitted.transform(&x).unwrap();
        assert_eq!(proj.dim(), (8, 1));
    }

    #[test]
    fn test_lda_predict_accuracy_binary() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| *p == *a).count();
        assert_eq!(correct, 8, "All 8 samples should be classified correctly");
    }

    #[test]
    fn test_lda_predict_three_classes() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| *p == *a).count();
        assert!(correct >= 7, "Expected at least 7/9 correct, got {correct}");
    }

    #[test]
    fn test_lda_explained_variance_ratio_positive() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        for &v in fitted.explained_variance_ratio().iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_lda_explained_variance_ratio_le_1() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        let total: f64 = fitted.explained_variance_ratio().iter().sum();
        assert!(total <= 1.0 + 1e-9, "total={total}");
    }

    #[test]
    fn test_lda_classes_accessor() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0usize, 1]);
    }

    #[test]
    fn test_lda_means_shape() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.means().dim(), (3, 2));
    }

    #[test]
    fn test_lda_transform_shape_mismatch() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let x_bad = Array2::<f64>::zeros((3, 5));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_lda_predict_shape_mismatch() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let x_bad = Array2::<f64>::zeros((3, 5));
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_lda_error_zero_n_components() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(0));
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_n_components_too_large() {
        let (x, y) = linearly_separable_2d(); // 2 classes → max 1 component
        let lda = LDA::<f64>::new(Some(5));
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_single_class() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0usize, 0, 0, 0];
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_shape_mismatch_fit() {
        let x = Array2::<f64>::zeros((4, 2));
        let y = array![0usize, 1]; // wrong length
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_insufficient_samples() {
        let x = Array2::<f64>::zeros((1, 2));
        let y = array![0usize];
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_scalings_accessor() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.scalings().dim(), (2, 1));
    }

    #[test]
    fn test_lda_pipeline_estimator() {
        use ferrolearn_core::pipeline::PipelineEstimator;

        let (x, y_usize) = linearly_separable_2d();
        let y_f64 = y_usize.mapv(|v| v as f64);
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit_pipeline(&x, &y_f64).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_lda_n_components_getter() {
        let lda = LDA::<f64>::new(Some(2));
        assert_eq!(lda.n_components(), Some(2));
        let lda_none = LDA::<f64>::new(None);
        assert_eq!(lda_none.n_components(), None);
    }

    #[test]
    fn test_lda_transform_then_predict_consistent() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        // Manually compute nearest-centroid prediction from transform output.
        let projected = fitted.transform(&x).unwrap();
        let preds_predict = fitted.predict(&x).unwrap();
        let n_samples = projected.nrows();
        let n_comp = projected.ncols();
        let n_classes = fitted.classes().len();
        for i in 0..n_samples {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for ci in 0..n_classes {
                let mut d = 0.0;
                for k in 0..n_comp {
                    let diff = projected[[i, k]] - fitted.means()[[ci, k]];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            assert_eq!(preds_predict[i], fitted.classes()[best]);
        }
    }

    #[test]
    fn test_lda_projected_class_separation() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let projected = fitted.transform(&x).unwrap();

        // Means of class 0 and class 1 in projected space should be far apart.
        let mean0: f64 = projected
            .rows()
            .into_iter()
            .zip(y.iter())
            .filter(|&(_, label)| *label == 0)
            .map(|(row, _)| row[0])
            .sum::<f64>()
            / 4.0;
        let mean1: f64 = projected
            .rows()
            .into_iter()
            .zip(y.iter())
            .filter(|&(_, label)| *label == 1)
            .map(|(row, _)| row[0])
            .sum::<f64>()
            / 4.0;

        assert!(
            (mean0 - mean1).abs() > 0.5,
            "Projected means should differ, got {mean0} vs {mean1}"
        );
    }

    #[test]
    fn test_lda_transform_known_data() {
        // With perfectly separated data the transform should yield two clearly
        // distinct groups.
        let x = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let proj = fitted.transform(&x).unwrap();
        // The first two samples should project to one side, the other two to the other side.
        let sign0 = proj[[0, 0]].signum();
        let sign1 = proj[[2, 0]].signum();
        // They should be on opposite sides of the origin (or at least the split is correct).
        assert_ne!(
            sign0 as i32, sign1 as i32,
            "Classes should be on opposite sides"
        );
    }

    #[test]
    fn test_lda_abs_diff_eq_means_dimensions() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        // Each class mean in projected space should be a 1-component vector.
        assert_eq!(fitted.means().ncols(), 1);
        let m0 = fitted.means()[[0, 0]];
        let m1 = fitted.means()[[1, 0]];
        // For well-separated data the projected means should differ by > 1.0.
        assert!((m0 - m1).abs() > 0.5, "m0={m0}, m1={m1}");
        let _ = assert_abs_diff_eq!(0.0_f64, 0.0_f64); // use the import
    }
}
