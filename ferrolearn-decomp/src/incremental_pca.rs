//! Incremental Principal Component Analysis (IncrementalPCA).
//!
//! [`IncrementalPCA`] performs PCA incrementally by processing data in batches.
//! This is useful for datasets that are too large to fit in memory, or when
//! data arrives as a stream.
//!
//! # Algorithm
//!
//! Maintains a running mean and uses an incremental SVD update:
//!
//! For each batch `X_batch`:
//! 1. Centre the batch using the running (cumulative) mean.
//! 2. Stack the centred batch with the existing components scaled by their
//!    singular values: `M = vstack([components * sqrt(n_samples_seen - 1), X_centred])`.
//! 3. Compute a thin SVD of `M`.
//! 4. Extract updated `components` (rows of V^T), `singular_values`, and
//!    `explained_variance` from the SVD.
//! 5. Update the running mean.
//!
//! The [`Fit::fit`] method processes the dataset in `batch_size` chunks
//! internally. Use [`IncrementalPCA::partial_fit`] to update the model with
//! one batch at a time (for streaming use cases).
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::IncrementalPCA;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::array;
//!
//! let ipca = IncrementalPCA::<f64>::new(1);
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let fitted = ipca.fit(&x, &()).unwrap();
//! let projected = fitted.transform(&x).unwrap();
//! assert_eq!(projected.ncols(), 1);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// IncrementalPCA (unfitted)
// ---------------------------------------------------------------------------

/// Incremental PCA configuration.
///
/// Holds `n_components` and an optional `batch_size`. Calling [`Fit::fit`]
/// processes the data in batches and returns a [`FittedIncrementalPCA`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct IncrementalPCA<F> {
    /// Number of principal components to retain.
    n_components: usize,
    /// Number of samples per batch. If `None`, the whole dataset is processed
    /// in a single batch (equivalent to standard PCA on the full data).
    batch_size: Option<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> IncrementalPCA<F> {
    /// Create a new `IncrementalPCA` that retains `n_components` components.
    ///
    /// If `batch_size` is not set, the whole dataset is processed at once.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            batch_size: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the batch size.
    ///
    /// Each call to the internal loop will process this many samples.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Return the number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured batch size, if any.
    #[must_use]
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }
}

// ---------------------------------------------------------------------------
// FittedIncrementalPCA
// ---------------------------------------------------------------------------

/// A fitted Incremental PCA model.
///
/// Created either by calling [`Fit::fit`] on an [`IncrementalPCA`] or by
/// calling [`IncrementalPCA::partial_fit`] one batch at a time.
///
/// Implements [`Transform<Array2<F>>`] to project new data onto the
/// learned principal components.
#[derive(Debug, Clone)]
pub struct FittedIncrementalPCA<F> {
    /// Principal component directions, shape `(n_components, n_features)`.
    components_: Array2<F>,
    /// Variance explained by each component (singular_value^2 / n_samples_seen).
    explained_variance_: Array1<F>,
    /// Ratio of variance explained by each component to total variance.
    explained_variance_ratio_: Array1<F>,
    /// Per-feature running mean.
    mean_: Array1<F>,
    /// Number of samples seen so far.
    n_samples_seen_: usize,
    /// Singular values of the current incremental SVD.
    singular_values_: Array1<F>,
}

impl<F: Float + Send + Sync + 'static> FittedIncrementalPCA<F> {
    /// Principal components, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<F> {
        &self.components_
    }

    /// Explained variance per component.
    #[must_use]
    pub fn explained_variance(&self) -> &Array1<F> {
        &self.explained_variance_
    }

    /// Explained variance ratio per component.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio_
    }

    /// Running per-feature mean.
    #[must_use]
    pub fn mean(&self) -> &Array1<F> {
        &self.mean_
    }

    /// Number of samples seen during fitting.
    #[must_use]
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen_
    }

    /// Singular values of the incremental SVD.
    #[must_use]
    pub fn singular_values(&self) -> &Array1<F> {
        &self.singular_values_
    }

    /// Map reduced data back to the original feature space. Mirrors
    /// sklearn `IncrementalPCA.inverse_transform`. Returns shape
    /// `(n_samples, n_features)` via `X_reduced @ components + mean`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x_reduced.ncols()` does
    /// not equal the number of components.
    pub fn inverse_transform(&self, x_reduced: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_components = self.components_.nrows();
        if x_reduced.ncols() != n_components {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x_reduced.nrows(), n_components],
                actual: vec![x_reduced.nrows(), x_reduced.ncols()],
                context: "FittedIncrementalPCA::inverse_transform".into(),
            });
        }
        let mut result = x_reduced.dot(&self.components_);
        for mut row in result.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v + m;
            }
        }
        Ok(result)
    }

    /// Process one additional batch, updating the model in-place.
    ///
    /// This is the core of the incremental algorithm. See the module-level
    /// documentation for the algorithm details.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the batch is empty.
    /// - [`FerroError::ShapeMismatch`] if the batch has the wrong number of
    ///   features (after the first batch has been seen).
    /// - [`FerroError::NumericalInstability`] if a numerical failure occurs.
    pub fn partial_fit_batch(&mut self, x_batch: &Array2<F>) -> Result<(), FerroError> {
        let (batch_n, n_features) = x_batch.dim();

        if batch_n == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "IncrementalPCA::partial_fit_batch requires non-empty batch".into(),
            });
        }

        // Check feature consistency if we have already seen samples.
        if self.n_samples_seen_ > 0 && n_features != self.mean_.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.mean_.len()],
                actual: vec![n_features],
                context: "IncrementalPCA::partial_fit_batch: feature dimension mismatch".into(),
            });
        }

        let n_components = self.components_.nrows();

        // ----------------------------------------------------------------
        // Step 1 & 5: Update running mean and centre the batch.
        // ----------------------------------------------------------------
        let batch_mean = column_mean(x_batch);

        let new_n = self.n_samples_seen_ + batch_n;
        let new_n_f = F::from(new_n).unwrap_or_else(F::one);

        // Incremental mean update: new_mean = (old_n * old_mean + batch_n * batch_mean) / new_n
        let updated_mean = if self.n_samples_seen_ == 0 {
            batch_mean
        } else {
            let old_n_f = F::from(self.n_samples_seen_).unwrap_or_else(F::zero);
            let batch_n_f = F::from(batch_n).unwrap_or_else(F::one);
            let mut m = Array1::zeros(n_features);
            for j in 0..n_features {
                m[j] = (old_n_f * self.mean_[j] + batch_n_f * batch_mean[j]) / new_n_f;
            }
            m
        };

        // Centre the batch using the *updated* mean.
        let mut x_centred = x_batch.to_owned();
        for mut row in x_centred.rows_mut() {
            for (v, &m) in row.iter_mut().zip(updated_mean.iter()) {
                *v = *v - m;
            }
        }

        // ----------------------------------------------------------------
        // Step 2: Stack old components (weighted by singular values) with
        //         the centred batch.
        // ----------------------------------------------------------------
        // The stacked matrix M has shape (n_components + batch_n, n_features).
        // If n_samples_seen == 0, M is just x_centred.
        let m_mat: Array2<F> = if self.n_samples_seen_ == 0 || n_components == 0 {
            x_centred.clone()
        } else {
            // Scale components by singular values:
            // weighted_components[k, :] = singular_values[k] * components[k, :]
            let mut weighted = Array2::zeros((n_components, n_features));
            for k in 0..n_components {
                let sv = self.singular_values_[k];
                for j in 0..n_features {
                    weighted[[k, j]] = sv * self.components_[[k, j]];
                }
            }
            // Stack: [weighted; x_centred]
            stack_vertical(&weighted, &x_centred)
        };

        // ----------------------------------------------------------------
        // Step 3: Thin SVD of M.
        // ----------------------------------------------------------------
        let max_rank = m_mat.nrows().min(m_mat.ncols()).min(n_components);
        if max_rank == 0 {
            self.mean_ = updated_mean;
            self.n_samples_seen_ = new_n;
            return Ok(());
        }

        let (_, sigma, vt) = thin_svd(&m_mat, max_rank)?;

        // ----------------------------------------------------------------
        // Step 4: Update components, singular values, and explained variance.
        // ----------------------------------------------------------------
        // vt has shape (max_rank, n_features); each row is a component.
        for k in 0..n_components.min(max_rank) {
            for j in 0..n_features {
                self.components_[[k, j]] = vt[[k, j]];
            }
            self.singular_values_[k] = if k < sigma.len() { sigma[k] } else { F::zero() };
        }
        // Zero out any components beyond max_rank if n_components > max_rank.
        for k in max_rank..n_components {
            for j in 0..n_features {
                self.components_[[k, j]] = F::zero();
            }
            self.singular_values_[k] = F::zero();
        }

        // Recompute explained variance.
        // explained_variance[k] = sigma[k]^2 / (n_samples_seen - 1)
        let denom = F::from(new_n.saturating_sub(1).max(1)).unwrap_or_else(F::one);
        let mut total_var = F::zero();
        for k in 0..n_components {
            let sv = self.singular_values_[k];
            self.explained_variance_[k] = sv * sv / denom;
            total_var = total_var + self.explained_variance_[k];
        }

        if total_var > F::zero() {
            for k in 0..n_components {
                self.explained_variance_ratio_[k] = self.explained_variance_[k] / total_var;
            }
        } else {
            for k in 0..n_components {
                self.explained_variance_ratio_[k] = F::zero();
            }
        }

        // Update state.
        self.mean_ = updated_mean;
        self.n_samples_seen_ = new_n;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IncrementalPCA: partial_fit (public streaming API)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> IncrementalPCA<F> {
    /// Process a single batch and return the updated fitted model.
    ///
    /// Calling `partial_fit` repeatedly on successive batches gives the same
    /// result as calling `fit` on the concatenation of all batches (up to
    /// floating-point rounding).
    ///
    /// The first call initialises the model; subsequent calls update it.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components == 0` or
    ///   `n_components >= n_features`.
    /// - [`FerroError::InsufficientSamples`] if the batch is empty.
    /// - [`FerroError::ShapeMismatch`] if the batch has the wrong number of
    ///   features after the first batch.
    pub fn partial_fit(
        &self,
        x_batch: &Array2<F>,
        state: Option<FittedIncrementalPCA<F>>,
    ) -> Result<FittedIncrementalPCA<F>, FerroError> {
        let n_features = x_batch.ncols();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_features".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components >= n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) must be < n_features ({})",
                    self.n_components, n_features
                ),
            });
        }

        let mut fitted = state.unwrap_or_else(|| FittedIncrementalPCA {
            components_: Array2::zeros((self.n_components, n_features)),
            explained_variance_: Array1::zeros(self.n_components),
            explained_variance_ratio_: Array1::zeros(self.n_components),
            mean_: Array1::zeros(n_features),
            n_samples_seen_: 0,
            singular_values_: Array1::zeros(self.n_components),
        });

        fitted.partial_fit_batch(x_batch)?;
        Ok(fitted)
    }
}

// ---------------------------------------------------------------------------
// Fit trait
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for IncrementalPCA<F> {
    type Fitted = FittedIncrementalPCA<F>;
    type Error = FerroError;

    /// Fit the model by processing the data in mini-batches.
    ///
    /// If `batch_size` is `None`, the entire dataset is processed in one batch.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components == 0`,
    ///   `n_components >= n_features`, or `batch_size == 0`.
    /// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedIncrementalPCA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_features".into(),
                reason: "must be at least 1".into(),
            });
        }
        if self.n_components >= n_features {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: format!(
                    "n_components ({}) must be < n_features ({})",
                    self.n_components, n_features
                ),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "IncrementalPCA::fit requires at least 2 samples".into(),
            });
        }

        let batch_size = match self.batch_size {
            Some(bs) => {
                if bs == 0 {
                    return Err(FerroError::InvalidParameter {
                        name: "batch_size".into(),
                        reason: "must be at least 1 when specified".into(),
                    });
                }
                bs
            }
            None => n_samples,
        };

        let mut state: Option<FittedIncrementalPCA<F>> = None;
        let mut start = 0;
        while start < n_samples {
            let end = (start + batch_size).min(n_samples);
            let batch = x.slice(ndarray::s![start..end, ..]).to_owned();
            if batch.nrows() > 0 {
                state = Some(self.partial_fit(&batch, state)?);
            }
            start = end;
        }

        state.ok_or_else(|| FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "IncrementalPCA::fit: no batches processed".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedIncrementalPCA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project data onto the principal components: `(X - mean) @ components^T`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.mean_.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedIncrementalPCA::transform".into(),
            });
        }

        let mut x_centred = x.to_owned();
        for mut row in x_centred.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.mean_.iter()) {
                *v = *v - m;
            }
        }

        // Project: X_centred @ components^T
        Ok(x_centred.dot(&self.components_.t()))
    }
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers (no external SVD library needed)
// ---------------------------------------------------------------------------

/// Compute the per-column mean of a matrix.
fn column_mean<F: Float>(x: &Array2<F>) -> Array1<F> {
    let (n, p) = x.dim();
    let n_f = F::from(n).unwrap_or_else(F::one);
    let mut mean = Array1::zeros(p);
    for j in 0..p {
        let s = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
        mean[j] = s / n_f;
    }
    mean
}

/// Stack two matrices vertically: `[a; b]`.
fn stack_vertical<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let na = a.nrows();
    let nb = b.nrows();
    let p = a.ncols();
    let mut out = Array2::zeros((na + nb, p));
    for i in 0..na {
        for j in 0..p {
            out[[i, j]] = a[[i, j]];
        }
    }
    for i in 0..nb {
        for j in 0..p {
            out[[na + i, j]] = b[[i, j]];
        }
    }
    out
}

/// `(U, sigma, Vt)` triple returned by [`thin_svd`].
type SvdTriple<F> = (Array2<F>, Array1<F>, Array2<F>);

/// Thin SVD via Jacobi eigendecomposition of the smaller of M^T M or M M^T.
///
/// Returns `(U, sigma, Vt)` where:
/// - `sigma` has length `max_rank`, sorted descending.
/// - `Vt` has shape `(max_rank, n_features)`.
/// - `U` has shape `(n_rows, max_rank)`.
fn thin_svd<F: Float + Send + Sync + 'static>(
    m: &Array2<F>,
    max_rank: usize,
) -> Result<SvdTriple<F>, FerroError> {
    let (nr, nc) = m.dim();
    if nr == 0 || nc == 0 || max_rank == 0 {
        return Ok((
            Array2::zeros((nr, 0)),
            Array1::zeros(0),
            Array2::zeros((0, nc)),
        ));
    }

    let rank = max_rank.min(nr).min(nc);

    // Decide whether to work with M M^T (small rows) or M^T M (small cols).
    if nr <= nc {
        // M M^T is (nr x nr) — cheaper when nr <= nc.
        let mmt = m.dot(&m.t());
        let max_iter = nr * nr * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&mmt, max_iter)?;

        let mut indices: Vec<usize> = (0..nr).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = Array1::zeros(rank);
        let mut u = Array2::zeros((nr, rank));
        let mut vt = Array2::zeros((rank, nc));

        for (out_k, &eigen_idx) in indices.iter().take(rank).enumerate() {
            let ev = eigenvalues[eigen_idx];
            let sv = if ev > F::zero() { ev.sqrt() } else { F::zero() };
            sigma[out_k] = sv;

            for i in 0..nr {
                u[[i, out_k]] = eigenvectors[[i, eigen_idx]];
            }

            // V = M^T U / sigma
            if sv > F::from(1e-30).unwrap_or_else(F::epsilon) {
                for j in 0..nc {
                    let mut val = F::zero();
                    for i in 0..nr {
                        val = val + m[[i, j]] * u[[i, out_k]];
                    }
                    vt[[out_k, j]] = val / sv;
                }
            }
        }

        Ok((u, sigma, vt))
    } else {
        // M^T M is (nc x nc) — cheaper when nc < nr.
        let mtm = m.t().dot(m);
        let max_iter = nc * nc * 100 + 1000;
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&mtm, max_iter)?;

        let mut indices: Vec<usize> = (0..nc).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = Array1::zeros(rank);
        let mut u = Array2::zeros((nr, rank));
        let mut vt = Array2::zeros((rank, nc));

        for (out_k, &eigen_idx) in indices.iter().take(rank).enumerate() {
            let ev = eigenvalues[eigen_idx];
            let sv = if ev > F::zero() { ev.sqrt() } else { F::zero() };
            sigma[out_k] = sv;

            for j in 0..nc {
                vt[[out_k, j]] = eigenvectors[[j, eigen_idx]];
            }

            // U = M V / sigma
            if sv > F::from(1e-30).unwrap_or_else(F::epsilon) {
                for i in 0..nr {
                    let mut val = F::zero();
                    for j in 0..nc {
                        val = val + m[[i, j]] * vt[[out_k, j]];
                    }
                    u[[i, out_k]] = val / sv;
                }
            }
        }

        Ok((u, sigma, vt))
    }
}

/// Jacobi eigendecomposition for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` where column `i` of `eigenvectors`
/// corresponds to `eigenvalues[i]`. The output is **not** sorted.
fn jacobi_eigen_symmetric<F: Float + Send + Sync + 'static>(
    a: &Array2<F>,
    max_iter: usize,
) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let n = a.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((
            Array1::from_vec(vec![a[[0, 0]]]),
            Array2::from_shape_vec((1, 1), vec![F::one()]).unwrap(),
        ));
    }

    let mut mat = a.to_owned();
    let mut v = Array2::<F>::zeros((n, n));
    for i in 0..n {
        v[[i, i]] = F::one();
    }

    let tol = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iteration in 0..max_iter {
        // Find the off-diagonal element with the largest absolute value.
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
        message: "Jacobi eigendecomposition did not converge in IncrementalPCA".into(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // Basic shape and structure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_output_shape() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (1, 2));
        assert_eq!(fitted.explained_variance().len(), 1);
        assert_eq!(fitted.explained_variance_ratio().len(), 1);
        assert_eq!(fitted.mean().len(), 2);
        assert_eq!(fitted.n_samples_seen(), 4);
    }

    #[test]
    fn test_transform_output_shape() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 1));
    }

    #[test]
    fn test_fit_two_components() {
        let ipca = IncrementalPCA::<f64>::new(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let fitted = ipca.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (2, 3));

        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.dim(), (4, 2));
    }

    #[test]
    fn test_mean_is_correct() {
        // Column means for [[0,0],[2,4]] should be [1, 2].
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[0.0, 0.0], [2.0, 4.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.mean()[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.mean()[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_explained_variance_positive() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        for &v in fitted.explained_variance() {
            assert!(v >= 0.0, "explained variance should be non-negative: {v}");
        }
    }

    #[test]
    fn test_explained_variance_ratio_in_unit_interval() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();
        assert!(
            (0.0..=1.0 + 1e-10).contains(&ratio_sum),
            "ratio sum {ratio_sum} not in [0,1]"
        );
    }

    #[test]
    fn test_batch_size_single_batch() {
        // batch_size == n_samples should give the same result as no batch_size.
        let ipca_no_bs = IncrementalPCA::<f64>::new(1);
        let ipca_bs = IncrementalPCA::<f64>::new(1).with_batch_size(4);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let fitted_no_bs = ipca_no_bs.fit(&x, &()).unwrap();
        let fitted_bs = ipca_bs.fit(&x, &()).unwrap();

        // Means should be identical.
        for (a, b) in fitted_no_bs.mean().iter().zip(fitted_bs.mean().iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
        assert_eq!(fitted_no_bs.n_samples_seen(), fitted_bs.n_samples_seen());
    }

    #[test]
    fn test_batch_size_two_batches() {
        let ipca = IncrementalPCA::<f64>::new(1).with_batch_size(2);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_samples_seen(), 4);
        assert_eq!(fitted.components().dim(), (1, 2));
    }

    #[test]
    fn test_partial_fit_chaining() {
        // Fit in two batches using partial_fit.
        let ipca = IncrementalPCA::<f64>::new(1);
        let b1 = array![[1.0, 2.0], [3.0, 4.0]];
        let b2 = array![[5.0, 6.0], [7.0, 8.0]];

        let state1 = ipca.partial_fit(&b1, None).unwrap();
        assert_eq!(state1.n_samples_seen(), 2);

        let state2 = ipca.partial_fit(&b2, Some(state1)).unwrap();
        assert_eq!(state2.n_samples_seen(), 4);
    }

    #[test]
    fn test_transform_shape_mismatch_error() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_invalid_n_components_zero_error() {
        let ipca = IncrementalPCA::<f64>::new(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(ipca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_invalid_n_components_ge_n_features_error() {
        let ipca = IncrementalPCA::<f64>::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(ipca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_insufficient_samples_error() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[1.0, 2.0]]; // only 1 sample
        assert!(ipca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_zero_batch_size_error() {
        let ipca = IncrementalPCA::<f64>::new(1).with_batch_size(0);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(ipca.fit(&x, &()).is_err());
    }

    #[test]
    fn test_f32_support() {
        let ipca = IncrementalPCA::<f32>::new(1);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        let projected = fitted.transform(&x).unwrap();
        assert_eq!(projected.ncols(), 1);
    }

    #[test]
    fn test_components_approx_unit_length() {
        let ipca = IncrementalPCA::<f64>::new(1);
        let x = array![[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],];
        let fitted = ipca.fit(&x, &()).unwrap();
        let c = fitted.components();
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_n_samples_seen() {
        let ipca = IncrementalPCA::<f64>::new(1).with_batch_size(3);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let fitted = ipca.fit(&x, &()).unwrap();
        assert_eq!(fitted.n_samples_seen(), 5);
    }

    #[test]
    fn test_getters() {
        let ipca = IncrementalPCA::<f64>::new(2).with_batch_size(50);
        assert_eq!(ipca.n_components(), 2);
        assert_eq!(ipca.batch_size(), Some(50));

        let ipca2 = IncrementalPCA::<f64>::new(1);
        assert!(ipca2.batch_size().is_none());
    }
}
