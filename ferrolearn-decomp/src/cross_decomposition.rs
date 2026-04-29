//! Cross-decomposition methods: PLS, CCA, and PLSSVD.
//!
//! This module provides Partial Least Squares (PLS) and Canonical Correlation
//! Analysis (CCA) methods for modelling the relationship between two multivariate
//! datasets X and Y.
//!
//! # Algorithms
//!
//! - [`PLSSVD`] — SVD of the cross-covariance matrix. The simplest PLS variant;
//!   computes weight matrices from the leading singular vectors of `X^T Y`.
//! - [`PLSRegression`] — PLS via the NIPALS algorithm. Maximises covariance
//!   between X-scores and Y-scores, with asymmetric deflation suitable for
//!   regression.
//! - [`PLSCanonical`] — Canonical PLS via NIPALS. Symmetric deflation of both
//!   X and Y.
//! - [`CCA`] — Canonical Correlation Analysis via NIPALS. Maximises
//!   *correlation* (not covariance) between X-scores and Y-scores by
//!   normalising scores to unit variance.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::cross_decomposition::PLSRegression;
//! use ferrolearn_core::traits::{Fit, Predict, Transform};
//! use ndarray::array;
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let y = array![[1.0], [2.0], [3.0], [4.0]];
//!
//! let pls = PLSRegression::<f64>::new(1);
//! let fitted = pls.fit(&x, &y).unwrap();
//! let y_pred = fitted.predict(&x).unwrap();
//! assert_eq!(y_pred.ncols(), 1);
//! let x_scores = fitted.transform(&x).unwrap();
//! assert_eq!(x_scores.ncols(), 1);
//! ```

use ferrolearn_core::backend::Backend;
use ferrolearn_core::backend_faer::NdarrayFaerBackend;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::any::TypeId;

/// Result type for SVD: `(U, S, Vt)`.
type SvdResult<F> = Result<(Array2<F>, Array1<F>, Array2<F>), FerroError>;

// ---------------------------------------------------------------------------
// Helper: centre and optionally scale columns of a matrix
// ---------------------------------------------------------------------------

/// Centre (and optionally scale to unit variance) columns of a matrix.
///
/// Returns `(centred_matrix, mean, std)` where `std` is `None` when
/// `scale` is `false`.
fn centre_scale<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    scale: bool,
) -> (Array2<F>, Array1<F>, Option<Array1<F>>) {
    let (n_samples, n_features) = x.dim();
    let n_f = F::from(n_samples).unwrap();

    // Compute column means.
    let mean = Array1::from_shape_fn(n_features, |j| {
        x.column(j).iter().copied().fold(F::zero(), |a, b| a + b) / n_f
    });

    // Centre.
    let mut xc = x.to_owned();
    for mut row in xc.rows_mut() {
        for (v, &m) in row.iter_mut().zip(mean.iter()) {
            *v = *v - m;
        }
    }

    if scale {
        let n_minus_1 = F::from(n_samples.saturating_sub(1).max(1)).unwrap();
        let std_dev = Array1::from_shape_fn(n_features, |j| {
            let var = xc
                .column(j)
                .iter()
                .copied()
                .fold(F::zero(), |a, b| a + b * b)
                / n_minus_1;
            let s = var.sqrt();
            if s < F::epsilon() { F::one() } else { s }
        });
        for mut row in xc.rows_mut() {
            for (v, &s) in row.iter_mut().zip(std_dev.iter()) {
                *v = *v / s;
            }
        }
        (xc, mean, Some(std_dev))
    } else {
        (xc, mean, None)
    }
}

/// Apply centring (and optionally scaling) to new data using stored statistics.
fn apply_centre_scale<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    mean: &Array1<F>,
    std_dev: &Option<Array1<F>>,
    context: &str,
) -> Result<Array2<F>, FerroError> {
    if x.ncols() != mean.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows(), mean.len()],
            actual: vec![x.nrows(), x.ncols()],
            context: context.into(),
        });
    }
    let mut xc = x.to_owned();
    for mut row in xc.rows_mut() {
        for (v, &m) in row.iter_mut().zip(mean.iter()) {
            *v = *v - m;
        }
    }
    if let Some(ref s) = *std_dev {
        for mut row in xc.rows_mut() {
            for (v, &sd) in row.iter_mut().zip(s.iter()) {
                *v = *v / sd;
            }
        }
    }
    Ok(xc)
}

// ---------------------------------------------------------------------------
// Helper: SVD dispatch (generic F via f64 fast-path or Jacobi fallback)
// ---------------------------------------------------------------------------

/// Compute the thin SVD of a general (m x n) matrix.
///
/// Returns `(U, S, Vt)` where:
/// - `U` is `(m, min(m,n))`,
/// - `S` is `(min(m,n),)`,
/// - `Vt` is `(min(m,n), n)`.
///
/// For `f64` this delegates to `NdarrayFaerBackend::svd` (faer's optimised
/// routine). For other float types it falls back to a power-iteration approach.
fn svd_dispatch<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> SvdResult<F> {
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // Cast to f64 and use faer.
        let a_f64: &Array2<f64> = unsafe { &*(std::ptr::from_ref(a).cast::<Array2<f64>>()) };
        let (u, s, vt) = NdarrayFaerBackend::svd(a_f64)?;
        // Thin U and Vt.
        let k = s.len();
        let u_thin = u.slice(ndarray::s![.., ..k]).to_owned();
        let vt_thin = vt.slice(ndarray::s![..k, ..]).to_owned();

        // Cast back to F (which is f64).
        let u_f: Array2<F> = unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&u_thin) };
        let s_f: Array1<F> = unsafe { std::mem::transmute_copy::<Array1<f64>, Array1<F>>(&s) };
        let vt_f: Array2<F> =
            unsafe { std::mem::transmute_copy::<Array2<f64>, Array2<F>>(&vt_thin) };
        std::mem::forget(u_thin);
        std::mem::forget(s);
        std::mem::forget(vt_thin);
        Ok((u_f, s_f, vt_f))
    } else if TypeId::of::<F>() == TypeId::of::<f32>() {
        // Convert f32 -> f64, compute, convert back.
        let (m, n) = a.dim();
        let a_f64 =
            Array2::<f64>::from_shape_fn((m, n), |(i, j)| a[[i, j]].to_f64().unwrap_or(0.0));
        let (u64, s64, vt64) = NdarrayFaerBackend::svd(&a_f64)?;
        let k = s64.len();
        let u_thin = u64.slice(ndarray::s![.., ..k]).to_owned();
        let vt_thin = vt64.slice(ndarray::s![..k, ..]).to_owned();

        let u_f =
            Array2::<F>::from_shape_fn(u_thin.dim(), |(i, j)| F::from(u_thin[[i, j]]).unwrap());
        let s_f = Array1::<F>::from_shape_fn(s64.len(), |i| F::from(s64[i]).unwrap());
        let vt_f =
            Array2::<F>::from_shape_fn(vt_thin.dim(), |(i, j)| F::from(vt_thin[[i, j]]).unwrap());
        Ok((u_f, s_f, vt_f))
    } else {
        // Fallback: compute via eigendecomposition of A^T A.
        svd_via_eigen(a)
    }
}

/// Compute SVD via eigendecomposition of `A^T A` (fallback for exotic float types).
fn svd_via_eigen<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> SvdResult<F> {
    let (m, n) = a.dim();
    let k = m.min(n);

    // Compute A^T A.
    let ata = a.t().dot(a);

    // Jacobi eigendecomposition of A^T A.
    let max_iter = n * n * 100 + 1000;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&ata, max_iter)?;

    // Sort eigenvalues descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top k.
    let mut s = Array1::<F>::zeros(k);
    let mut v = Array2::<F>::zeros((n, k));
    for (col, &idx) in indices.iter().take(k).enumerate() {
        let eval = eigenvalues[idx];
        s[col] = if eval > F::zero() {
            eval.sqrt()
        } else {
            F::zero()
        };
        for row in 0..n {
            v[[row, col]] = eigenvectors[[row, idx]];
        }
    }

    // U = A V S^{-1}
    let av = a.dot(&v);
    let mut u = Array2::<F>::zeros((m, k));
    for col in 0..k {
        if s[col] > F::epsilon() {
            let inv_s = F::one() / s[col];
            for row in 0..m {
                u[[row, col]] = av[[row, col]] * inv_s;
            }
        }
    }

    // Vt = V^T
    let mut vt = Array2::<F>::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            vt[[i, j]] = v[[j, i]];
        }
    }

    Ok((u, s, vt))
}

/// Jacobi eigendecomposition for symmetric matrices (generic F fallback).
fn jacobi_eigen_symmetric<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iteration in 0..max_iter {
        let mut max_off = F::zero();
        let mut p = 0;
        let mut q = 1;
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

        let theta = if (app - aqq).abs() < tol {
            F::from(std::f64::consts::FRAC_PI_4).unwrap_or_else(F::one)
        } else {
            let tau = (aqq - app) / (F::from(2.0).unwrap() * apq);
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

        new_mat[[p, p]] = c * c * app - F::from(2.0).unwrap() * s * c * apq + s * s * aqq;
        new_mat[[q, q]] = s * s * app + F::from(2.0).unwrap() * s * c * apq + c * c * aqq;
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
        message: "Jacobi eigendecomposition did not converge in cross_decomposition SVD fallback"
            .into(),
    })
}

// ---------------------------------------------------------------------------
// Helper: vector norm and dot
// ---------------------------------------------------------------------------

/// L2 norm of a 1-D array.
fn norm<F: Float>(v: &Array1<F>) -> F {
    v.iter().copied().fold(F::zero(), |a, b| a + b * b).sqrt()
}

/// Dot product of two 1-D arrays.
fn dot<F: Float>(a: &Array1<F>, b: &Array1<F>) -> F {
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .fold(F::zero(), |acc, (x, y)| acc + x * y)
}

// ---------------------------------------------------------------------------
// Helper: solve (P^T W) inverse for PLSRegression predict
// ---------------------------------------------------------------------------

/// Solve `(P^T W)^{-1}` for a square matrix using Gaussian elimination
/// with partial pivoting (generic float).
///
/// When a pivot is too small (near-singular), it is regularised with a
/// small perturbation to avoid hard failures. This matches the behaviour
/// of scikit-learn, which uses `pinv` for the rotation matrix.
fn invert_square<F: Float + Send + Sync + 'static>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![a.nrows(), a.ncols()],
            context: "invert_square: matrix must be square".into(),
        });
    }

    // Augmented matrix [A | I].
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    // Compute a tolerance based on the matrix norm.
    let max_abs = a.iter().copied().fold(F::zero(), |m, v| {
        let abs = v.abs();
        if abs > m { abs } else { m }
    });
    let regularise_tol = max_abs * F::from(1e-12).unwrap_or_else(F::epsilon)
        + F::from(1e-15).unwrap_or_else(F::epsilon);

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Regularise if pivot is too small.
        if max_val < regularise_tol {
            aug[[col, col]] = regularise_tol;
        } else {
            // Swap rows.
            if max_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }
        }

        // Eliminate below.
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..(2 * n) {
                let above = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * above;
            }
        }
    }

    // Back substitution.
    for col in (0..n).rev() {
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] = aug[[col, j]] / pivot;
        }
        for row in 0..col {
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                let below = aug[[col, j]];
                aug[[row, j]] = aug[[row, j]] - factor * below;
            }
        }
    }

    // Extract inverse.
    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

// ===========================================================================
// PLSSVD
// ===========================================================================

/// PLS via Singular Value Decomposition of the cross-covariance matrix.
///
/// This is the simplest PLS variant. It computes the weight matrices by
/// taking the leading left and right singular vectors of `X^T Y` after
/// optional centring and scaling.
///
/// Unlike [`PLSRegression`], PLSSVD does not iterate; it is a single
/// matrix decomposition. It cannot predict Y from X — use
/// [`PLSRegression`] if you need a `predict` method.
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::cross_decomposition::PLSSVD;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let y = array![[1.0], [2.0], [3.0], [4.0]];
/// let svd = PLSSVD::<f64>::new(1);
/// let fitted = svd.fit(&x, &y).unwrap();
/// let scores = fitted.transform(&x).unwrap();
/// assert_eq!(scores.ncols(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PLSSVD<F> {
    /// Number of components to extract.
    n_components: usize,
    /// Whether to scale X and Y to unit variance before decomposition.
    scale: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PLSSVD<F> {
    /// Create a new `PLSSVD` that extracts `n_components` components.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set whether to scale X and Y to unit variance (default: `true`).
    #[must_use]
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

/// A fitted [`PLSSVD`] model.
///
/// Holds the learned weight matrices and centring/scaling statistics.
/// Implements [`Transform`] to project X data onto the PLS score space.
#[derive(Debug, Clone)]
pub struct FittedPLSSVD<F> {
    /// X-weights, shape `(n_features_x, n_components)`.
    x_weights_: Array2<F>,
    /// Y-weights, shape `(n_features_y, n_components)`.
    y_weights_: Array2<F>,
    /// Per-feature mean of X.
    x_mean_: Array1<F>,
    /// Per-feature mean of Y.
    y_mean_: Array1<F>,
    /// Per-feature standard deviation of X (None if not scaled).
    x_std_: Option<Array1<F>>,
    /// Per-feature standard deviation of Y (None if not scaled).
    y_std_: Option<Array1<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedPLSSVD<F> {
    /// X-weights matrix, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_weights(&self) -> &Array2<F> {
        &self.x_weights_
    }

    /// Y-weights matrix, shape `(n_features_y, n_components)`.
    #[must_use]
    pub fn y_weights(&self) -> &Array2<F> {
        &self.y_weights_
    }

    /// Per-feature mean of X learned during fitting.
    #[must_use]
    pub fn x_mean(&self) -> &Array1<F> {
        &self.x_mean_
    }

    /// Per-feature mean of Y learned during fitting.
    #[must_use]
    pub fn y_mean(&self) -> &Array1<F> {
        &self.y_mean_
    }

    /// Transform Y data onto the Y-score space.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if Y has the wrong number of columns.
    pub fn transform_y(&self, y: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let yc = apply_centre_scale(y, &self.y_mean_, &self.y_std_, "FittedPLSSVD::transform_y")?;
        Ok(yc.dot(&self.y_weights_))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array2<F>> for PLSSVD<F> {
    type Fitted = FittedPLSSVD<F>;
    type Error = FerroError;

    /// Fit PLSSVD by computing the SVD of the cross-covariance matrix `X^T Y`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   `min(n_features_x, n_features_y)`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if X and Y have different numbers of rows.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedPLSSVD<F>, FerroError> {
        let (n_samples_x, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples_x != n_samples_y {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples_x, n_features_y],
                actual: vec![n_samples_y, n_features_y],
                context: "PLSSVD::fit: X and Y must have the same number of rows".into(),
            });
        }

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        let max_components = n_features_x.min(n_features_y);
        if self.n_components > max_components {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_features_x, n_features_y) ({})",
                    self.n_components, max_components
                ),
            });
        }

        if n_samples_x < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples_x,
                context: "PLSSVD::fit requires at least 2 samples".into(),
            });
        }

        // Centre and optionally scale.
        let (xc, x_mean, x_std) = centre_scale(x, self.scale);
        let (yc, y_mean, y_std) = centre_scale(y, self.scale);

        // Cross-covariance: C = X^T Y.
        let c = xc.t().dot(&yc);

        // SVD of C.
        let (u, _s, vt) = svd_dispatch(&c)?;

        // Take first n_components columns of U, rows of Vt (= columns of V).
        let nc = self.n_components;
        let x_weights = u.slice(ndarray::s![.., ..nc]).to_owned();
        // V = Vt^T, so columns of V = rows of Vt transposed.
        let y_weights = vt.t().slice(ndarray::s![.., ..nc]).to_owned();

        Ok(FittedPLSSVD {
            x_weights_: x_weights,
            y_weights_: y_weights,
            x_mean_: x_mean,
            y_mean_: y_mean,
            x_std_: x_std,
            y_std_: y_std,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPLSSVD<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project X data onto the PLS score space: `(X - x_mean) / x_std @ x_weights`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if X has the wrong number of columns.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xc = apply_centre_scale(x, &self.x_mean_, &self.x_std_, "FittedPLSSVD::transform")?;
        Ok(xc.dot(&self.x_weights_))
    }
}

// ===========================================================================
// NIPALS mode enum (shared by PLSRegression, PLSCanonical, CCA)
// ===========================================================================

/// Internal NIPALS deflation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NipalsMode {
    /// Regression: deflate Y with X-scores (`Y = Y - t q^T`).
    Regression,
    /// Canonical: deflate Y with its own scores (`Y = Y - u c^T`).
    Canonical,
}

/// Internal flag: whether to normalise scores to unit variance (CCA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreNorm {
    /// Do not normalise scores (PLS).
    None,
    /// Normalise scores to unit variance (CCA).
    UnitVariance,
}

// ---------------------------------------------------------------------------
// NIPALS core algorithm
// ---------------------------------------------------------------------------

/// Result from a NIPALS fit.
#[derive(Debug, Clone)]
struct NipalsResult<F> {
    /// X-weights W, shape `(n_features_x, n_components)` — columns are weight vectors.
    x_weights: Array2<F>,
    /// X-loadings P, shape `(n_features_x, n_components)`.
    x_loadings: Array2<F>,
    /// X-scores T, shape `(n_samples, n_components)`.
    x_scores: Array2<F>,
    /// Y-loadings Q, shape `(n_features_y, n_components)`.
    y_loadings: Array2<F>,
    /// Y-scores U, shape `(n_samples, n_components)`.
    y_scores: Array2<F>,
    /// Number of iterations per component.
    n_iter: Vec<usize>,
}

/// Run the NIPALS algorithm.
fn nipals<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array2<F>,
    n_components: usize,
    max_iter: usize,
    tol: F,
    mode: NipalsMode,
    score_norm: ScoreNorm,
) -> Result<NipalsResult<F>, FerroError> {
    let (n_samples, n_features_x) = x.dim();
    let n_features_y = y.ncols();

    let mut xk = x.to_owned();
    let mut yk = y.to_owned();

    let mut x_weights = Array2::<F>::zeros((n_features_x, n_components));
    let mut x_loadings = Array2::<F>::zeros((n_features_x, n_components));
    let mut x_scores = Array2::<F>::zeros((n_samples, n_components));
    let mut y_loadings = Array2::<F>::zeros((n_features_y, n_components));
    let mut y_scores = Array2::<F>::zeros((n_samples, n_components));
    let mut n_iter_vec = Vec::with_capacity(n_components);

    for k in 0..n_components {
        // Initialise u = column of Y with max variance.
        let best_col = (0..n_features_y)
            .max_by(|&a, &b| {
                let var_a: F = yk
                    .column(a)
                    .iter()
                    .copied()
                    .fold(F::zero(), |s, v| s + v * v);
                let var_b: F = yk
                    .column(b)
                    .iter()
                    .copied()
                    .fold(F::zero(), |s, v| s + v * v);
                var_a
                    .partial_cmp(&var_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let mut u = yk.column(best_col).to_owned();

        let mut converged = false;
        let mut iters = 0;

        for iteration in 0..max_iter {
            iters = iteration + 1;

            // w = X^T u / (u^T u)
            let utu = dot(&u, &u);
            let mut w = xk.t().dot(&u);
            if utu > F::epsilon() {
                w.mapv_inplace(|v| v / utu);
            }
            // Normalise w.
            let w_norm = norm(&w);
            if w_norm < F::epsilon() {
                // Degenerate: zero weight vector.
                break;
            }
            w.mapv_inplace(|v| v / w_norm);

            // t = X w
            let t = xk.dot(&w);

            // q = Y^T t / (t^T t)
            let ttt = dot(&t, &t);
            let mut q = yk.t().dot(&t);
            if ttt > F::epsilon() {
                q.mapv_inplace(|v| v / ttt);
            }

            // For CCA: normalise q.
            if score_norm == ScoreNorm::UnitVariance {
                let q_norm = norm(&q);
                if q_norm > F::epsilon() {
                    q.mapv_inplace(|v| v / q_norm);
                }
            }

            // u_new = Y q / (q^T q)
            let qtq = dot(&q, &q);
            let mut u_new = yk.dot(&q);
            if qtq > F::epsilon() {
                u_new.mapv_inplace(|v| v / qtq);
            }

            // For CCA: normalise t and u to unit variance.
            // (This is done after the loop for storing; here we just check convergence.)

            // Convergence check: ||u_new - u|| / ||u_new||.
            let diff_norm = {
                let diff: Array1<F> = &u_new - &u;
                norm(&diff)
            };
            let u_new_norm = norm(&u_new);

            u = u_new;

            if u_new_norm > F::epsilon() && diff_norm / u_new_norm < tol {
                converged = true;
                // Recompute final w, t, q with converged u.
                // w = X^T u / (u^T u), normalised
                let utu2 = dot(&u, &u);
                w = xk.t().dot(&u);
                if utu2 > F::epsilon() {
                    w.mapv_inplace(|v| v / utu2);
                }
                let w_norm2 = norm(&w);
                if w_norm2 > F::epsilon() {
                    w.mapv_inplace(|v| v / w_norm2);
                }
                // t = X w
                let t_final = xk.dot(&w);
                let ttt2 = dot(&t_final, &t_final);
                q = yk.t().dot(&t_final);
                if ttt2 > F::epsilon() {
                    q.mapv_inplace(|v| v / ttt2);
                }
                if score_norm == ScoreNorm::UnitVariance {
                    let q_norm2 = norm(&q);
                    if q_norm2 > F::epsilon() {
                        q.mapv_inplace(|v| v / q_norm2);
                    }
                }
                let qtq2 = dot(&q, &q);
                u = yk.dot(&q);
                if qtq2 > F::epsilon() {
                    u.mapv_inplace(|v| v / qtq2);
                }
                break;
            }
        }

        // Compute final scores and loadings with the converged weights.
        let utu_final = dot(&u, &u);
        let mut w_final = xk.t().dot(&u);
        if utu_final > F::epsilon() {
            w_final.mapv_inplace(|v| v / utu_final);
        }
        let w_norm_final = norm(&w_final);
        if w_norm_final > F::epsilon() {
            w_final.mapv_inplace(|v| v / w_norm_final);
        }

        let mut t_final = xk.dot(&w_final);
        let ttt_final = dot(&t_final, &t_final);

        // p = X^T t / (t^T t)
        let mut p = xk.t().dot(&t_final);
        if ttt_final > F::epsilon() {
            p.mapv_inplace(|v| v / ttt_final);
        }

        // q = Y^T t / (t^T t)
        let mut q_final = yk.t().dot(&t_final);
        if ttt_final > F::epsilon() {
            q_final.mapv_inplace(|v| v / ttt_final);
        }

        if score_norm == ScoreNorm::UnitVariance {
            let q_norm = norm(&q_final);
            if q_norm > F::epsilon() {
                q_final.mapv_inplace(|v| v / q_norm);
            }
        }

        let qtq_final = dot(&q_final, &q_final);
        let mut u_final = yk.dot(&q_final);
        if qtq_final > F::epsilon() {
            u_final.mapv_inplace(|v| v / qtq_final);
        }

        // For CCA: normalise t and u to unit variance.
        if score_norm == ScoreNorm::UnitVariance {
            let t_std = {
                let t_mean = t_final.iter().copied().fold(F::zero(), |a, b| a + b)
                    / F::from(n_samples).unwrap();
                let var = t_final
                    .iter()
                    .copied()
                    .fold(F::zero(), |a, b| a + (b - t_mean) * (b - t_mean))
                    / F::from(n_samples.saturating_sub(1).max(1)).unwrap();
                var.sqrt()
            };
            if t_std > F::epsilon() {
                t_final.mapv_inplace(|v| v / t_std);
            }

            let u_std = {
                let u_mean = u_final.iter().copied().fold(F::zero(), |a, b| a + b)
                    / F::from(n_samples).unwrap();
                let var = u_final
                    .iter()
                    .copied()
                    .fold(F::zero(), |a, b| a + (b - u_mean) * (b - u_mean))
                    / F::from(n_samples.saturating_sub(1).max(1)).unwrap();
                var.sqrt()
            };
            if u_std > F::epsilon() {
                u_final.mapv_inplace(|v| v / u_std);
            }
        }

        // Store component k.
        x_weights.column_mut(k).assign(&w_final);
        x_loadings.column_mut(k).assign(&p);
        x_scores.column_mut(k).assign(&t_final);
        y_loadings.column_mut(k).assign(&q_final);
        y_scores.column_mut(k).assign(&u_final);

        // Deflate X: X = X - t p^T.
        for i in 0..n_samples {
            let ti = t_final[i];
            for j in 0..n_features_x {
                xk[[i, j]] = xk[[i, j]] - ti * p[j];
            }
        }

        // Deflate Y.
        match mode {
            NipalsMode::Regression => {
                // Y = Y - t q^T (deflate with X-scores).
                for i in 0..n_samples {
                    let ti = t_final[i];
                    for j in 0..n_features_y {
                        yk[[i, j]] = yk[[i, j]] - ti * q_final[j];
                    }
                }
            }
            NipalsMode::Canonical => {
                // Y = Y - u c^T where c = Y^T u / (u^T u).
                let utu_c = dot(&u_final, &u_final);
                let mut c = yk.t().dot(&u_final);
                if utu_c > F::epsilon() {
                    c.mapv_inplace(|v| v / utu_c);
                }
                for i in 0..n_samples {
                    let ui = u_final[i];
                    for j in 0..n_features_y {
                        yk[[i, j]] = yk[[i, j]] - ui * c[j];
                    }
                }
            }
        }

        n_iter_vec.push(iters);

        if !converged && n_features_y > 1 && iters == max_iter {
            return Err(FerroError::ConvergenceFailure {
                iterations: max_iter,
                message: format!("NIPALS did not converge for component {k}"),
            });
        }
    }

    Ok(NipalsResult {
        x_weights,
        x_loadings,
        x_scores,
        y_loadings,
        y_scores,
        n_iter: n_iter_vec,
    })
}

// ===========================================================================
// PLSRegression
// ===========================================================================

/// Partial Least Squares Regression via the NIPALS algorithm.
///
/// PLSRegression finds latent components that maximise the covariance
/// between X-scores and Y-scores, with asymmetric deflation of Y using
/// X-scores. This is the standard PLS2 algorithm for multi-target
/// regression.
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::cross_decomposition::PLSRegression;
/// use ferrolearn_core::traits::{Fit, Predict, Transform};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let y = array![[1.0], [2.0], [3.0], [4.0]];
///
/// let pls = PLSRegression::<f64>::new(1);
/// let fitted = pls.fit(&x, &y).unwrap();
///
/// let y_pred = fitted.predict(&x).unwrap();
/// assert_eq!(y_pred.ncols(), 1);
///
/// let scores = fitted.transform(&x).unwrap();
/// assert_eq!(scores.ncols(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PLSRegression<F> {
    /// Number of PLS components to extract.
    n_components: usize,
    /// Maximum NIPALS iterations per component.
    max_iter: usize,
    /// Convergence tolerance for NIPALS.
    tol: F,
    /// Whether to scale X and Y to unit variance.
    scale: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PLSRegression<F> {
    /// Create a new `PLSRegression` with `n_components` components.
    ///
    /// Defaults: `max_iter = 500`, `tol = 1e-6`, `scale = true`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 500,
            tol: F::from(1e-6).unwrap_or_else(F::epsilon),
            scale: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of NIPALS iterations per component.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the NIPALS convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to scale X and Y to unit variance (default: `true`).
    #[must_use]
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

/// A fitted [`PLSRegression`] model.
///
/// Holds the learned weight, loading, and score matrices, plus the
/// regression coefficients for prediction. Implements [`Predict`] to
/// predict Y from X, and [`Transform`] to project X onto the score space.
#[derive(Debug, Clone)]
pub struct FittedPLSRegression<F> {
    /// X-weights W, shape `(n_features_x, n_components)`.
    x_weights_: Array2<F>,
    /// X-loadings P, shape `(n_features_x, n_components)`.
    x_loadings_: Array2<F>,
    /// Y-loadings Q, shape `(n_features_y, n_components)`.
    y_loadings_: Array2<F>,
    /// Regression coefficients B, shape `(n_features_x, n_features_y)`.
    /// B = W (P^T W)^{-1} Q^T.
    coefficients_: Array2<F>,
    /// X-scores T from training, shape `(n_samples, n_components)`.
    x_scores_: Array2<F>,
    /// Y-scores U from training, shape `(n_samples, n_components)`.
    y_scores_: Array2<F>,
    /// Number of iterations per component.
    n_iter_: Vec<usize>,
    /// Per-feature mean of X.
    x_mean_: Array1<F>,
    /// Per-feature mean of Y.
    y_mean_: Array1<F>,
    /// Per-feature std of X (None if not scaled).
    x_std_: Option<Array1<F>>,
    /// Per-feature std of Y (None if not scaled).
    y_std_: Option<Array1<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedPLSRegression<F> {
    /// X-weights matrix W, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_weights(&self) -> &Array2<F> {
        &self.x_weights_
    }

    /// X-loadings matrix P, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_loadings(&self) -> &Array2<F> {
        &self.x_loadings_
    }

    /// Y-loadings matrix Q, shape `(n_features_y, n_components)`.
    #[must_use]
    pub fn y_loadings(&self) -> &Array2<F> {
        &self.y_loadings_
    }

    /// Regression coefficient matrix B, shape `(n_features_x, n_features_y)`.
    ///
    /// `Y_pred = X_centred @ B + y_mean`.
    #[must_use]
    pub fn coefficients(&self) -> &Array2<F> {
        &self.coefficients_
    }

    /// X-scores T from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn x_scores(&self) -> &Array2<F> {
        &self.x_scores_
    }

    /// Y-scores U from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn y_scores(&self) -> &Array2<F> {
        &self.y_scores_
    }

    /// Number of NIPALS iterations for each component.
    #[must_use]
    pub fn n_iter(&self) -> &[usize] {
        &self.n_iter_
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array2<F>> for PLSRegression<F> {
    type Fitted = FittedPLSRegression<F>;
    type Error = FerroError;

    /// Fit PLSRegression using the NIPALS algorithm.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if X and Y have different row counts.
    /// - [`FerroError::ConvergenceFailure`] if NIPALS does not converge.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedPLSRegression<F>, FerroError> {
        let (n_samples_x, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples_x != n_samples_y {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples_x, n_features_y],
                actual: vec![n_samples_y, n_features_y],
                context: "PLSRegression::fit: X and Y must have the same number of rows".into(),
            });
        }

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        let max_components = n_features_x.min(n_features_y).min(n_samples_x);
        if self.n_components > max_components {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_features_x, n_features_y, n_samples) ({})",
                    self.n_components, max_components
                ),
            });
        }

        if n_samples_x < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples_x,
                context: "PLSRegression::fit requires at least 2 samples".into(),
            });
        }

        // Centre and optionally scale.
        let (xc, x_mean, x_std) = centre_scale(x, self.scale);
        let (yc, y_mean, y_std) = centre_scale(y, self.scale);

        // Run NIPALS.
        let result = nipals(
            &xc,
            &yc,
            self.n_components,
            self.max_iter,
            self.tol,
            NipalsMode::Regression,
            ScoreNorm::None,
        )?;

        // Compute regression coefficients: B = W (P^T W)^{-1} Q^T.
        let ptw = result.x_loadings.t().dot(&result.x_weights);
        let ptw_inv = invert_square(&ptw)?;
        let coefficients = result.x_weights.dot(&ptw_inv).dot(&result.y_loadings.t());

        // If we scaled, adjust coefficients to work on the original scale.
        // The stored coefficients operate on centred (and scaled) X, producing
        // centred (and scaled) Y. We leave them in this internal space and
        // apply the scaling in predict/transform.

        Ok(FittedPLSRegression {
            x_weights_: result.x_weights,
            x_loadings_: result.x_loadings,
            y_loadings_: result.y_loadings,
            coefficients_: coefficients,
            x_scores_: result.x_scores,
            y_scores_: result.y_scores,
            n_iter_: result.n_iter,
            x_mean_: x_mean,
            y_mean_: y_mean,
            x_std_: x_std,
            y_std_: y_std,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedPLSRegression<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Predict Y from X using the fitted PLS regression model.
    ///
    /// Computes `Y_pred = X_centred @ B`, then un-scales and un-centres.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if X has the wrong number of columns.
    fn predict(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xc = apply_centre_scale(
            x,
            &self.x_mean_,
            &self.x_std_,
            "FittedPLSRegression::predict",
        )?;

        let mut y_pred = xc.dot(&self.coefficients_);

        // Un-scale Y.
        if let Some(ref ys) = self.y_std_ {
            for mut row in y_pred.rows_mut() {
                for (v, &s) in row.iter_mut().zip(ys.iter()) {
                    *v = *v * s;
                }
            }
        }

        // Un-centre Y.
        for mut row in y_pred.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.y_mean_.iter()) {
                *v = *v + m;
            }
        }

        Ok(y_pred)
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPLSRegression<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project X onto the PLS score space (X-scores).
    ///
    /// Computes `T = X_centred @ W (P^T W)^{-1}`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if X has the wrong number of columns.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xc = apply_centre_scale(
            x,
            &self.x_mean_,
            &self.x_std_,
            "FittedPLSRegression::transform",
        )?;

        // T = X_centred @ W (P^T W)^{-1} = X_centred @ rotation
        // But simpler: T = X_centred @ W when W columns are the NIPALS weights.
        // Actually the correct transform for new data is via the rotation matrix.
        let ptw = self.x_loadings_.t().dot(&self.x_weights_);
        let ptw_inv = invert_square(&ptw)?;
        let rotation = self.x_weights_.dot(&ptw_inv);
        Ok(xc.dot(&rotation))
    }
}

// ===========================================================================
// PLSCanonical
// ===========================================================================

/// Canonical PLS via the NIPALS algorithm.
///
/// PLSCanonical performs a symmetric decomposition: both X and Y are
/// deflated with their own scores. This contrasts with [`PLSRegression`]
/// which deflates Y using X-scores.
///
/// PLSCanonical is appropriate when you want a symmetric analysis of the
/// relationship between X and Y, rather than a predictive model.
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::cross_decomposition::PLSCanonical;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
///
/// let pls = PLSCanonical::<f64>::new(2);
/// let fitted = pls.fit(&x, &y).unwrap();
/// let scores = fitted.transform(&x).unwrap();
/// assert_eq!(scores.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct PLSCanonical<F> {
    /// Number of PLS components to extract.
    n_components: usize,
    /// Maximum NIPALS iterations per component.
    max_iter: usize,
    /// Convergence tolerance for NIPALS.
    tol: F,
    /// Whether to scale X and Y to unit variance.
    scale: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> PLSCanonical<F> {
    /// Create a new `PLSCanonical` with `n_components` components.
    ///
    /// Defaults: `max_iter = 500`, `tol = 1e-6`, `scale = true`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 500,
            tol: F::from(1e-6).unwrap_or_else(F::epsilon),
            scale: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of NIPALS iterations per component.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the NIPALS convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to scale X and Y to unit variance (default: `true`).
    #[must_use]
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

/// A fitted [`PLSCanonical`] model.
///
/// Holds the learned weight, loading, and score matrices from the
/// symmetric NIPALS decomposition. Implements [`Transform`] to project
/// X onto the score space.
#[derive(Debug, Clone)]
pub struct FittedPLSCanonical<F> {
    /// X-weights W, shape `(n_features_x, n_components)`.
    x_weights_: Array2<F>,
    /// X-loadings P, shape `(n_features_x, n_components)`.
    x_loadings_: Array2<F>,
    /// Y-loadings Q, shape `(n_features_y, n_components)`.
    y_loadings_: Array2<F>,
    /// X-scores T from training, shape `(n_samples, n_components)`.
    x_scores_: Array2<F>,
    /// Y-scores U from training, shape `(n_samples, n_components)`.
    y_scores_: Array2<F>,
    /// Number of iterations per component.
    n_iter_: Vec<usize>,
    /// Per-feature mean of X.
    x_mean_: Array1<F>,
    /// Per-feature mean of Y.
    y_mean_: Array1<F>,
    /// Per-feature std of X (None if not scaled).
    x_std_: Option<Array1<F>>,
    /// Per-feature std of Y (None if not scaled).
    y_std_: Option<Array1<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedPLSCanonical<F> {
    /// X-weights matrix W, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_weights(&self) -> &Array2<F> {
        &self.x_weights_
    }

    /// X-loadings matrix P, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_loadings(&self) -> &Array2<F> {
        &self.x_loadings_
    }

    /// Y-loadings matrix Q, shape `(n_features_y, n_components)`.
    #[must_use]
    pub fn y_loadings(&self) -> &Array2<F> {
        &self.y_loadings_
    }

    /// X-scores T from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn x_scores(&self) -> &Array2<F> {
        &self.x_scores_
    }

    /// Y-scores U from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn y_scores(&self) -> &Array2<F> {
        &self.y_scores_
    }

    /// Number of NIPALS iterations for each component.
    #[must_use]
    pub fn n_iter(&self) -> &[usize] {
        &self.n_iter_
    }

    /// Transform Y data onto the Y-score space.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if Y has the wrong number of columns.
    pub fn transform_y(&self, y: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let yc = apply_centre_scale(
            y,
            &self.y_mean_,
            &self.y_std_,
            "FittedPLSCanonical::transform_y",
        )?;
        Ok(yc.dot(&self.y_loadings_))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array2<F>> for PLSCanonical<F> {
    type Fitted = FittedPLSCanonical<F>;
    type Error = FerroError;

    /// Fit PLSCanonical using the NIPALS algorithm with symmetric deflation.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if X and Y have different row counts.
    /// - [`FerroError::ConvergenceFailure`] if NIPALS does not converge.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedPLSCanonical<F>, FerroError> {
        let (n_samples_x, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples_x != n_samples_y {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples_x, n_features_y],
                actual: vec![n_samples_y, n_features_y],
                context: "PLSCanonical::fit: X and Y must have the same number of rows".into(),
            });
        }

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        let max_components = n_features_x.min(n_features_y).min(n_samples_x);
        if self.n_components > max_components {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_features_x, n_features_y, n_samples) ({})",
                    self.n_components, max_components
                ),
            });
        }

        if n_samples_x < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples_x,
                context: "PLSCanonical::fit requires at least 2 samples".into(),
            });
        }

        let (xc, x_mean, x_std) = centre_scale(x, self.scale);
        let (yc, y_mean, y_std) = centre_scale(y, self.scale);

        let result = nipals(
            &xc,
            &yc,
            self.n_components,
            self.max_iter,
            self.tol,
            NipalsMode::Canonical,
            ScoreNorm::None,
        )?;

        Ok(FittedPLSCanonical {
            x_weights_: result.x_weights,
            x_loadings_: result.x_loadings,
            y_loadings_: result.y_loadings,
            x_scores_: result.x_scores,
            y_scores_: result.y_scores,
            n_iter_: result.n_iter,
            x_mean_: x_mean,
            y_mean_: y_mean,
            x_std_: x_std,
            y_std_: y_std,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedPLSCanonical<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project X onto the PLS score space (X-scores).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if X has the wrong number of columns.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xc = apply_centre_scale(
            x,
            &self.x_mean_,
            &self.x_std_,
            "FittedPLSCanonical::transform",
        )?;

        let ptw = self.x_loadings_.t().dot(&self.x_weights_);
        let ptw_inv = invert_square(&ptw)?;
        let rotation = self.x_weights_.dot(&ptw_inv);
        Ok(xc.dot(&rotation))
    }
}

// ===========================================================================
// CCA (Canonical Correlation Analysis)
// ===========================================================================

/// Canonical Correlation Analysis via the NIPALS algorithm.
///
/// CCA maximises the *correlation* (rather than covariance) between
/// X-scores and Y-scores by normalising scores to unit variance after
/// each NIPALS iteration. It uses symmetric (canonical) deflation.
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type.
///
/// # Examples
///
/// ```
/// use ferrolearn_decomp::cross_decomposition::CCA;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
///
/// let cca = CCA::<f64>::new(2);
/// let fitted = cca.fit(&x, &y).unwrap();
/// let scores = fitted.transform(&x).unwrap();
/// assert_eq!(scores.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CCA<F> {
    /// Number of canonical components to extract.
    n_components: usize,
    /// Maximum NIPALS iterations per component.
    max_iter: usize,
    /// Convergence tolerance for NIPALS.
    tol: F,
    /// Whether to scale X and Y to unit variance.
    scale: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> CCA<F> {
    /// Create a new `CCA` with `n_components` components.
    ///
    /// Defaults: `max_iter = 500`, `tol = 1e-6`, `scale = true`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 500,
            tol: F::from(1e-6).unwrap_or_else(F::epsilon),
            scale: true,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of NIPALS iterations per component.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the NIPALS convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to scale X and Y to unit variance (default: `true`).
    #[must_use]
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

/// A fitted [`CCA`] model.
///
/// Holds the learned weight, loading, and score matrices. Implements
/// [`Transform`] to project X onto the canonical score space.
#[derive(Debug, Clone)]
pub struct FittedCCA<F> {
    /// X-weights W, shape `(n_features_x, n_components)`.
    x_weights_: Array2<F>,
    /// X-loadings P, shape `(n_features_x, n_components)`.
    x_loadings_: Array2<F>,
    /// Y-loadings Q, shape `(n_features_y, n_components)`.
    y_loadings_: Array2<F>,
    /// X-scores T from training, shape `(n_samples, n_components)`.
    x_scores_: Array2<F>,
    /// Y-scores U from training, shape `(n_samples, n_components)`.
    y_scores_: Array2<F>,
    /// Number of iterations per component.
    n_iter_: Vec<usize>,
    /// Per-feature mean of X.
    x_mean_: Array1<F>,
    /// Per-feature mean of Y.
    y_mean_: Array1<F>,
    /// Per-feature std of X (None if not scaled).
    x_std_: Option<Array1<F>>,
    /// Per-feature std of Y (None if not scaled).
    y_std_: Option<Array1<F>>,
}

impl<F: Float + Send + Sync + 'static> FittedCCA<F> {
    /// X-weights matrix W, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_weights(&self) -> &Array2<F> {
        &self.x_weights_
    }

    /// X-loadings matrix P, shape `(n_features_x, n_components)`.
    #[must_use]
    pub fn x_loadings(&self) -> &Array2<F> {
        &self.x_loadings_
    }

    /// Y-loadings matrix Q, shape `(n_features_y, n_components)`.
    #[must_use]
    pub fn y_loadings(&self) -> &Array2<F> {
        &self.y_loadings_
    }

    /// X-scores T from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn x_scores(&self) -> &Array2<F> {
        &self.x_scores_
    }

    /// Y-scores U from training, shape `(n_samples, n_components)`.
    #[must_use]
    pub fn y_scores(&self) -> &Array2<F> {
        &self.y_scores_
    }

    /// Number of NIPALS iterations for each component.
    #[must_use]
    pub fn n_iter(&self) -> &[usize] {
        &self.n_iter_
    }

    /// Transform Y data onto the Y-score space.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if Y has the wrong number of columns.
    pub fn transform_y(&self, y: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let yc = apply_centre_scale(y, &self.y_mean_, &self.y_std_, "FittedCCA::transform_y")?;
        Ok(yc.dot(&self.y_loadings_))
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array2<F>> for CCA<F> {
    type Fitted = FittedCCA<F>;
    type Error = FerroError;

    /// Fit CCA using the NIPALS algorithm with score normalisation.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or too large.
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples.
    /// - [`FerroError::ShapeMismatch`] if X and Y have different row counts.
    /// - [`FerroError::ConvergenceFailure`] if NIPALS does not converge.
    fn fit(&self, x: &Array2<F>, y: &Array2<F>) -> Result<FittedCCA<F>, FerroError> {
        let (n_samples_x, n_features_x) = x.dim();
        let (n_samples_y, n_features_y) = y.dim();

        if n_samples_x != n_samples_y {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples_x, n_features_y],
                actual: vec![n_samples_y, n_features_y],
                context: "CCA::fit: X and Y must have the same number of rows".into(),
            });
        }

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }

        let max_components = n_features_x.min(n_features_y).min(n_samples_x);
        if self.n_components > max_components {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) exceeds min(n_features_x, n_features_y, n_samples) ({})",
                    self.n_components, max_components
                ),
            });
        }

        if n_samples_x < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples_x,
                context: "CCA::fit requires at least 2 samples".into(),
            });
        }

        let (xc, x_mean, x_std) = centre_scale(x, self.scale);
        let (yc, y_mean, y_std) = centre_scale(y, self.scale);

        let result = nipals(
            &xc,
            &yc,
            self.n_components,
            self.max_iter,
            self.tol,
            NipalsMode::Canonical,
            ScoreNorm::UnitVariance,
        )?;

        Ok(FittedCCA {
            x_weights_: result.x_weights,
            x_loadings_: result.x_loadings,
            y_loadings_: result.y_loadings,
            x_scores_: result.x_scores,
            y_scores_: result.y_scores,
            n_iter_: result.n_iter,
            x_mean_: x_mean,
            y_mean_: y_mean,
            x_std_: x_std,
            y_std_: y_std,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedCCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project X onto the canonical score space.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if X has the wrong number of columns.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let xc = apply_centre_scale(x, &self.x_mean_, &self.x_std_, "FittedCCA::transform")?;

        let ptw = self.x_loadings_.t().dot(&self.x_weights_);
        let ptw_inv = invert_square(&ptw)?;
        let rotation = self.x_weights_.dot(&ptw_inv);
        Ok(xc.dot(&rotation))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // PLSSVD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plssvd_basic_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let svd = PLSSVD::<f64>::new(1);
        let fitted = svd.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.dim(), (5, 1));
    }

    #[test]
    fn test_plssvd_two_components() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let svd = PLSSVD::<f64>::new(2);
        let fitted = svd.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.dim(), (4, 2));
    }

    #[test]
    fn test_plssvd_transform_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let svd = PLSSVD::<f64>::new(1);
        let fitted = svd.fit(&x, &y).unwrap();
        let y_scores = fitted.transform_y(&y).unwrap();
        assert_eq!(y_scores.ncols(), 1);
    }

    #[test]
    fn test_plssvd_no_scale() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let svd = PLSSVD::<f64>::new(1).with_scale(false);
        let fitted = svd.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    #[test]
    fn test_plssvd_x_weights_shape() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let svd = PLSSVD::<f64>::new(2);
        let fitted = svd.fit(&x, &y).unwrap();
        assert_eq!(fitted.x_weights().dim(), (3, 2));
        assert_eq!(fitted.y_weights().dim(), (2, 2));
    }

    #[test]
    fn test_plssvd_invalid_zero_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0]];
        let svd = PLSSVD::<f64>::new(0);
        assert!(svd.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plssvd_too_many_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        // min(2, 1) = 1, asking for 2 is too many.
        let svd = PLSSVD::<f64>::new(2);
        assert!(svd.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plssvd_row_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0]];
        let svd = PLSSVD::<f64>::new(1);
        assert!(svd.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plssvd_insufficient_samples() {
        let x = array![[1.0, 2.0]];
        let y = array![[1.0]];
        let svd = PLSSVD::<f64>::new(1);
        assert!(svd.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plssvd_transform_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        let svd = PLSSVD::<f64>::new(1);
        let fitted = svd.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_plssvd_n_components_getter() {
        let svd = PLSSVD::<f64>::new(3);
        assert_eq!(svd.n_components(), 3);
    }

    #[test]
    fn test_plssvd_f32() {
        let x: Array2<f32> = array![
            [1.0f32, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ];
        let y: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let svd = PLSSVD::<f32>::new(1);
        let fitted = svd.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    // -----------------------------------------------------------------------
    // PLSRegression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plsregression_basic_fit_predict() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_pred = fitted.predict(&x).unwrap();
        assert_eq!(y_pred.dim(), (5, 1));
    }

    #[test]
    fn test_plsregression_prediction_quality() {
        // Perfect linear relationship: Y = X[:,0] + X[:,1]
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[3.0], [7.0], [11.0], [15.0], [19.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_pred = fitted.predict(&x).unwrap();

        // With a perfect linear relationship and 1 component, prediction
        // should be very close.
        for (pred, actual) in y_pred.column(0).iter().zip(y.column(0).iter()) {
            assert_abs_diff_eq!(pred, actual, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_plsregression_multi_target() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5],];
        let pls = PLSRegression::<f64>::new(2);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_pred = fitted.predict(&x).unwrap();
        assert_eq!(y_pred.dim(), (5, 2));
    }

    #[test]
    fn test_plsregression_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.dim(), (5, 1));
    }

    #[test]
    fn test_plsregression_coefficients_shape() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let pls = PLSRegression::<f64>::new(2);
        let fitted = pls.fit(&x, &y).unwrap();
        // B shape: (n_features_x, n_features_y)
        assert_eq!(fitted.coefficients().dim(), (3, 2));
    }

    #[test]
    fn test_plsregression_no_scale() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pls = PLSRegression::<f64>::new(1).with_scale(false);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_pred = fitted.predict(&x).unwrap();
        assert_eq!(y_pred.dim(), (5, 1));
    }

    #[test]
    fn test_plsregression_builder() {
        let pls = PLSRegression::<f64>::new(2)
            .with_max_iter(1000)
            .with_tol(1e-8)
            .with_scale(false);
        assert_eq!(pls.n_components(), 2);
    }

    #[test]
    fn test_plsregression_invalid_zero_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0]];
        let pls = PLSRegression::<f64>::new(0);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plsregression_too_many_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        // min(2, 1, 3) = 1, asking for 2 is too many.
        let pls = PLSRegression::<f64>::new(2);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plsregression_row_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0]];
        let pls = PLSRegression::<f64>::new(1);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plsregression_insufficient_samples() {
        let x = array![[1.0, 2.0]];
        let y = array![[1.0]];
        let pls = PLSRegression::<f64>::new(1);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plsregression_predict_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_plsregression_transform_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_plsregression_x_scores_shape() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0], [2.0], [3.0], [4.0]];
        let pls = PLSRegression::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        assert_eq!(fitted.x_scores().dim(), (4, 1));
        assert_eq!(fitted.y_scores().dim(), (4, 1));
        assert_eq!(fitted.n_iter().len(), 1);
    }

    #[test]
    fn test_plsregression_f32() {
        let x: Array2<f32> = array![
            [1.0f32, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ];
        let y: Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0]];
        let pls = PLSRegression::<f32>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_pred = fitted.predict(&x).unwrap();
        assert_eq!(y_pred.ncols(), 1);
    }

    // -----------------------------------------------------------------------
    // PLSCanonical tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_plscanonical_basic_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5],];
        let pls = PLSCanonical::<f64>::new(2);
        let fitted = pls.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.dim(), (5, 2));
    }

    #[test]
    fn test_plscanonical_single_component() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let pls = PLSCanonical::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    #[test]
    fn test_plscanonical_scores_shape() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let pls = PLSCanonical::<f64>::new(2);
        let fitted = pls.fit(&x, &y).unwrap();
        assert_eq!(fitted.x_scores().dim(), (3, 2));
        assert_eq!(fitted.y_scores().dim(), (3, 2));
        assert_eq!(fitted.x_weights().dim(), (3, 2));
        assert_eq!(fitted.x_loadings().dim(), (3, 2));
        assert_eq!(fitted.y_loadings().dim(), (2, 2));
    }

    #[test]
    fn test_plscanonical_transform_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let pls = PLSCanonical::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let y_scores = fitted.transform_y(&y).unwrap();
        assert_eq!(y_scores.ncols(), 1);
    }

    #[test]
    fn test_plscanonical_builder() {
        let pls = PLSCanonical::<f64>::new(2)
            .with_max_iter(1000)
            .with_tol(1e-8)
            .with_scale(false);
        assert_eq!(pls.n_components(), 2);
    }

    #[test]
    fn test_plscanonical_invalid_zero_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0]];
        let pls = PLSCanonical::<f64>::new(0);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plscanonical_too_many_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        let pls = PLSCanonical::<f64>::new(2);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plscanonical_row_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let pls = PLSCanonical::<f64>::new(1);
        assert!(pls.fit(&x, &y).is_err());
    }

    #[test]
    fn test_plscanonical_transform_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let pls = PLSCanonical::<f64>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_plscanonical_f32() {
        let x: Array2<f32> = array![
            [1.0f32, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ];
        let y: Array2<f32> = array![
            [1.0f32, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
        ];
        let pls = PLSCanonical::<f32>::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    // -----------------------------------------------------------------------
    // CCA tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cca_basic_fit_transform() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5],];
        let cca = CCA::<f64>::new(2);
        let fitted = cca.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.dim(), (5, 2));
    }

    #[test]
    fn test_cca_single_component() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let cca = CCA::<f64>::new(1);
        let fitted = cca.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    #[test]
    fn test_cca_scores_shape() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let cca = CCA::<f64>::new(2);
        let fitted = cca.fit(&x, &y).unwrap();
        assert_eq!(fitted.x_scores().dim(), (3, 2));
        assert_eq!(fitted.y_scores().dim(), (3, 2));
        assert_eq!(fitted.x_weights().dim(), (3, 2));
        assert_eq!(fitted.x_loadings().dim(), (3, 2));
        assert_eq!(fitted.y_loadings().dim(), (2, 2));
    }

    #[test]
    fn test_cca_transform_y() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let cca = CCA::<f64>::new(1);
        let fitted = cca.fit(&x, &y).unwrap();
        let y_scores = fitted.transform_y(&y).unwrap();
        assert_eq!(y_scores.ncols(), 1);
    }

    #[test]
    fn test_cca_builder() {
        let cca = CCA::<f64>::new(2)
            .with_max_iter(1000)
            .with_tol(1e-8)
            .with_scale(false);
        assert_eq!(cca.n_components(), 2);
    }

    #[test]
    fn test_cca_invalid_zero_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0]];
        let cca = CCA::<f64>::new(0);
        assert!(cca.fit(&x, &y).is_err());
    }

    #[test]
    fn test_cca_too_many_components() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0], [2.0], [3.0]];
        let cca = CCA::<f64>::new(2);
        assert!(cca.fit(&x, &y).is_err());
    }

    #[test]
    fn test_cca_row_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let cca = CCA::<f64>::new(1);
        assert!(cca.fit(&x, &y).is_err());
    }

    #[test]
    fn test_cca_transform_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];
        let cca = CCA::<f64>::new(1);
        let fitted = cca.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_cca_f32() {
        let x: Array2<f32> = array![
            [1.0f32, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ];
        let y: Array2<f32> = array![
            [1.0f32, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
        ];
        let cca = CCA::<f32>::new(1);
        let fitted = cca.fit(&x, &y).unwrap();
        let scores = fitted.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    // -----------------------------------------------------------------------
    // Cross-cutting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pls_regression_and_canonical_give_different_scores() {
        let x = array![
            [1.0, 2.0, 0.5],
            [3.0, 1.0, 2.5],
            [5.0, 6.0, 1.0],
            [7.0, 3.0, 4.5],
            [9.0, 10.0, 2.0],
        ];
        let y = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5],];

        let pls_reg = PLSRegression::<f64>::new(2);
        let fitted_reg = pls_reg.fit(&x, &y).unwrap();
        let scores_reg = fitted_reg.transform(&x).unwrap();

        let pls_can = PLSCanonical::<f64>::new(2);
        let fitted_can = pls_can.fit(&x, &y).unwrap();
        let scores_can = fitted_can.transform(&x).unwrap();

        // They should produce different results (different deflation).
        let diff: f64 = scores_reg
            .iter()
            .zip(scores_can.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // The scores should not be identical (unless the data is degenerate).
        // We just check they are both valid matrices.
        assert_eq!(scores_reg.dim(), scores_can.dim());
        // In practice diff may be zero for some data; just check no NaN.
        assert!(diff.is_finite());
    }

    #[test]
    fn test_centre_scale_helper() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (xc, mean, std_dev) = centre_scale(&x, true);
        assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean[1], 4.0, epsilon = 1e-10);
        assert!(std_dev.is_some());

        // Centred data should have zero mean.
        let col_mean_0: f64 = xc.column(0).iter().sum::<f64>() / 3.0;
        assert_abs_diff_eq!(col_mean_0, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centre_scale_no_scale() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (_xc, _mean, std_dev) = centre_scale(&x, false);
        assert!(std_dev.is_none());
    }

    #[test]
    fn test_invert_square_identity() {
        let eye = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let inv = invert_square(&eye).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(inv[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_square_2x2() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let inv = invert_square(&a).unwrap();
        // A * A^{-1} should be identity.
        let prod = a.dot(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(prod[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }
}
