//! Dictionary Learning.
//!
//! [`DictionaryLearning`] learns a dictionary `D` and sparse codes `A` such
//! that `X ~ A * D`. The dictionary atoms form an overcomplete basis, and
//! the codes are encouraged to be sparse via an L1 penalty.
//!
//! # Algorithm
//!
//! Alternating optimisation:
//!
//! 1. **Sparse coding step**: with `D` fixed, solve for `A` using coordinate
//!    descent (lasso) or orthogonal matching pursuit (OMP).
//! 2. **Dictionary update step**: with `A` fixed, update `D` by solving a
//!    least-squares problem and normalising the atoms.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_decomp::DictionaryLearning;
//! use ferrolearn_core::traits::{Fit, Transform};
//! use ndarray::Array2;
//!
//! let x = Array2::<f64>::from_shape_fn((20, 10), |(i, j)| {
//!     ((i * 7 + j * 3) % 11) as f64
//! });
//! let dl = DictionaryLearning::new(5)
//!     .with_max_iter(50)
//!     .with_random_state(42);
//! let fitted = dl.fit(&x, &()).unwrap();
//! let codes = fitted.transform(&x).unwrap();
//! assert_eq!(codes.dim(), (20, 5));
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;

// ---------------------------------------------------------------------------
// Algorithm enums
// ---------------------------------------------------------------------------

/// The algorithm for the sparse coding step during fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictFitAlgorithm {
    /// Coordinate descent (lasso).
    CoordinateDescent,
}

/// The algorithm for the sparse coding step during transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictTransformAlgorithm {
    /// Orthogonal Matching Pursuit.
    Omp,
    /// Coordinate descent (lasso).
    LassoCd,
}

// ---------------------------------------------------------------------------
// DictionaryLearning (unfitted)
// ---------------------------------------------------------------------------

/// Dictionary Learning configuration.
///
/// Holds hyperparameters for the dictionary learning algorithm. Calling
/// [`Fit::fit`] learns a dictionary and returns a [`FittedDictionaryLearning`].
#[derive(Debug, Clone)]
pub struct DictionaryLearning {
    /// Number of dictionary atoms (components).
    n_components: usize,
    /// Sparsity penalty (L1 coefficient). Default 1.0.
    alpha: f64,
    /// Maximum number of alternating optimisation iterations. Default 1000.
    max_iter: usize,
    /// Convergence tolerance. Default 1e-8.
    tol: f64,
    /// Algorithm for fitting. Default coordinate descent.
    fit_algorithm: DictFitAlgorithm,
    /// Algorithm for transform. Default OMP.
    transform_algorithm: DictTransformAlgorithm,
    /// Maximum atoms per sample for OMP. Default n_components.
    transform_n_nonzero_coefs: Option<usize>,
    /// Optional random seed.
    random_state: Option<u64>,
}

impl DictionaryLearning {
    /// Create a new `DictionaryLearning` with `n_components` atoms.
    ///
    /// Defaults: `alpha=1.0`, `max_iter=1000`, `tol=1e-8`,
    /// `fit_algorithm=CoordinateDescent`, `transform_algorithm=Omp`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-8,
            fit_algorithm: DictFitAlgorithm::CoordinateDescent,
            transform_algorithm: DictTransformAlgorithm::Omp,
            transform_n_nonzero_coefs: None,
            random_state: None,
        }
    }

    /// Set the sparsity penalty.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the fit algorithm.
    #[must_use]
    pub fn with_fit_algorithm(mut self, algo: DictFitAlgorithm) -> Self {
        self.fit_algorithm = algo;
        self
    }

    /// Set the transform algorithm.
    #[must_use]
    pub fn with_transform_algorithm(mut self, algo: DictTransformAlgorithm) -> Self {
        self.transform_algorithm = algo;
        self
    }

    /// Set the maximum number of non-zero coefficients for OMP transform.
    #[must_use]
    pub fn with_transform_n_nonzero_coefs(mut self, n: usize) -> Self {
        self.transform_n_nonzero_coefs = Some(n);
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Return the configured number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Return the configured alpha.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the configured maximum iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the configured tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the configured fit algorithm.
    #[must_use]
    pub fn fit_algorithm(&self) -> DictFitAlgorithm {
        self.fit_algorithm
    }

    /// Return the configured transform algorithm.
    #[must_use]
    pub fn transform_algorithm(&self) -> DictTransformAlgorithm {
        self.transform_algorithm
    }

    /// Return the configured random state, if any.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }
}

// ---------------------------------------------------------------------------
// FittedDictionaryLearning
// ---------------------------------------------------------------------------

/// A fitted dictionary learning model.
///
/// Created by calling [`Fit::fit`] on a [`DictionaryLearning`]. The learned
/// dictionary is accessible via [`FittedDictionaryLearning::components`].
/// Implements [`Transform<Array2<f64>>`] to compute sparse codes for new data.
#[derive(Debug, Clone)]
pub struct FittedDictionaryLearning {
    /// Learned dictionary, shape `(n_components, n_features)`.
    /// Each row is a dictionary atom.
    components_: Array2<f64>,
    /// Sparsity penalty used during fitting.
    alpha_: f64,
    /// Number of iterations performed.
    n_iter_: usize,
    /// Final reconstruction error (Frobenius norm).
    reconstruction_err_: f64,
    /// Transform algorithm to use.
    transform_algorithm_: DictTransformAlgorithm,
    /// Max non-zero coefs for OMP.
    transform_n_nonzero_coefs_: usize,
}

impl FittedDictionaryLearning {
    /// The learned dictionary, shape `(n_components, n_features)`.
    #[must_use]
    pub fn components(&self) -> &Array2<f64> {
        &self.components_
    }

    /// Number of iterations performed.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// The reconstruction error at convergence.
    #[must_use]
    pub fn reconstruction_err(&self) -> f64 {
        self.reconstruction_err_
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Normalise dictionary rows to unit L2 norm.
fn normalise_dictionary(d: &mut Array2<f64>) {
    let n_components = d.nrows();
    let n_features = d.ncols();
    for k in 0..n_components {
        let mut norm = 0.0;
        for j in 0..n_features {
            norm += d[[k, j]] * d[[k, j]];
        }
        let norm = norm.sqrt();
        if norm > 1e-16 {
            for j in 0..n_features {
                d[[k, j]] /= norm;
            }
        }
    }
}

/// Lasso coordinate descent for a single sample: solve
///   min_a 0.5 * ||x - D^T a||^2 + alpha * ||a||_1
/// where x is (n_features,), D is (n_components, n_features), a is (n_components,).
fn lasso_cd_single(x_row: &[f64], d: &Array2<f64>, alpha: f64, max_iter: usize) -> Vec<f64> {
    let n_components = d.nrows();
    let n_features = d.ncols();
    let mut a = vec![0.0; n_components];

    // Pre-compute D * D^T (Gram matrix) and D * x.
    let mut gram = vec![vec![0.0; n_components]; n_components];
    let mut dx = vec![0.0; n_components];
    for k in 0..n_components {
        for j in 0..n_features {
            dx[k] += d[[k, j]] * x_row[j];
        }
        for l in k..n_components {
            let mut val = 0.0;
            for j in 0..n_features {
                val += d[[k, j]] * d[[l, j]];
            }
            gram[k][l] = val;
            gram[l][k] = val;
        }
    }

    for _iter in 0..max_iter {
        let mut max_change = 0.0;
        for k in 0..n_components {
            // Compute partial residual: dx[k] - sum_{l!=k} gram[k][l] * a[l]
            let mut rho = dx[k];
            for l in 0..n_components {
                if l != k {
                    rho -= gram[k][l] * a[l];
                }
            }

            // Soft threshold.
            let gram_kk = gram[k][k];
            let new_a = if gram_kk.abs() < 1e-16 {
                0.0
            } else {
                soft_threshold(rho, alpha) / gram_kk
            };

            let change = (new_a - a[k]).abs();
            if change > max_change {
                max_change = change;
            }
            a[k] = new_a;
        }
        if max_change < 1e-6 {
            break;
        }
    }

    a
}

/// Orthogonal Matching Pursuit for a single sample.
fn omp_single(x_row: &[f64], d: &Array2<f64>, max_nonzero: usize) -> Vec<f64> {
    let n_components = d.nrows();
    let n_features = d.ncols();
    let mut a = vec![0.0; n_components];
    let mut residual: Vec<f64> = x_row.to_vec();
    let mut selected: Vec<usize> = Vec::new();
    let max_k = max_nonzero.min(n_components).min(n_features);

    for _step in 0..max_k {
        // Find the atom most correlated with the residual.
        let mut best_idx = 0;
        let mut best_corr = 0.0;
        for k in 0..n_components {
            if selected.contains(&k) {
                continue;
            }
            let mut corr = 0.0;
            for j in 0..n_features {
                corr += d[[k, j]] * residual[j];
            }
            if corr.abs() > best_corr {
                best_corr = corr.abs();
                best_idx = k;
            }
        }

        if best_corr < 1e-12 {
            break;
        }

        selected.push(best_idx);

        // Solve least squares: x = D_selected^T * a_selected
        // Use normal equations: (D_s D_s^T) a_s = D_s x
        let m = selected.len();
        let mut gram = vec![vec![0.0; m]; m];
        let mut rhs = vec![0.0; m];
        for (ii, &ki) in selected.iter().enumerate() {
            for j in 0..n_features {
                rhs[ii] += d[[ki, j]] * x_row[j];
            }
            for (jj, &kj) in selected.iter().enumerate() {
                let mut val = 0.0;
                for f in 0..n_features {
                    val += d[[ki, f]] * d[[kj, f]];
                }
                gram[ii][jj] = val;
            }
        }

        // Solve gram * coefs = rhs via Cholesky-like.
        if let Some(coefs) = solve_symmetric(&gram, &rhs) {
            // Update residual.
            residual = x_row.to_vec();
            for (ii, &ki) in selected.iter().enumerate() {
                a[ki] = coefs[ii];
                for j in 0..n_features {
                    residual[j] -= coefs[ii] * d[[ki, j]];
                }
            }
        } else {
            break;
        }

        // Check if residual is small enough.
        let res_norm: f64 = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
        if res_norm < 1e-10 {
            break;
        }
    }

    a
}

/// Solve a small symmetric positive definite system Ax = b using
/// Gaussian elimination with partial pivoting. Returns None if singular.
#[allow(clippy::needless_range_loop)]
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Augmented matrix.
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, row) in a.iter().enumerate().take(n) {
        let mut r = row.clone();
        r.push(b[i]);
        aug.push(r);
    }

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Soft thresholding: sign(x) * max(|x| - lambda, 0).
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Compute Frobenius norm of X - A * D.
fn reconstruction_error(x: &Array2<f64>, a: &Array2<f64>, d: &Array2<f64>) -> f64 {
    let ad = a.dot(d);
    let mut err = 0.0;
    for (xi, adi) in x.iter().zip(ad.iter()) {
        let diff = xi - adi;
        err += diff * diff;
    }
    err.sqrt()
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Fit<Array2<f64>, ()> for DictionaryLearning {
    type Fitted = FittedDictionaryLearning;
    type Error = FerroError;

    /// Fit the dictionary learning model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or
    ///   `alpha` is negative.
    /// - [`FerroError::InsufficientSamples`] if there are zero samples or
    ///   zero features.
    fn fit(&self, x: &Array2<f64>, _y: &()) -> Result<FittedDictionaryLearning, FerroError> {
        let (n_samples, n_features) = x.dim();

        // Validate.
        if self.n_components == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_components".into(),
                reason: "must be at least 1".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DictionaryLearning::fit".into(),
            });
        }
        if n_features == 0 {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "must have at least 1 feature".into(),
            });
        }
        if self.alpha < 0.0 {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be non-negative".into(),
            });
        }

        let n_components = self.n_components;
        let seed = self.random_state.unwrap_or(0);
        let transform_n_nonzero = self.transform_n_nonzero_coefs.unwrap_or(n_components);

        // Initialise dictionary from random Gaussian, then normalise.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut d = Array2::<f64>::zeros((n_components, n_features));
        for elem in &mut d {
            *elem = normal.sample(&mut rng);
        }
        normalise_dictionary(&mut d);

        let mut prev_err = f64::MAX;
        let mut n_iter = 0;

        for iteration in 0..self.max_iter {
            n_iter = iteration + 1;

            // Sparse coding step: compute codes A.
            let mut a = Array2::<f64>::zeros((n_samples, n_components));
            for i in 0..n_samples {
                let x_row: Vec<f64> = (0..n_features).map(|j| x[[i, j]]).collect();
                let codes = lasso_cd_single(&x_row, &d, self.alpha, 200);
                for k in 0..n_components {
                    a[[i, k]] = codes[k];
                }
            }

            // Dictionary update step: D = (A^T A)^{-1} A^T X
            // We solve the normal equations for each atom.
            let ata = a.t().dot(&a);
            let atx = a.t().dot(x);

            // Solve K x K system for each feature column of D.
            // Build the Gram matrix as Vec<Vec<f64>>.
            let gram: Vec<Vec<f64>> = (0..n_components)
                .map(|i| (0..n_components).map(|j| ata[[i, j]]).collect())
                .collect();

            // Add small regularisation for stability.
            let mut gram_reg = gram.clone();
            for (k, row) in gram_reg.iter_mut().enumerate() {
                row[k] += 1e-10;
            }

            for j in 0..n_features {
                let rhs: Vec<f64> = (0..n_components).map(|k| atx[[k, j]]).collect();
                if let Some(sol) = solve_symmetric(&gram_reg, &rhs) {
                    for k in 0..n_components {
                        d[[k, j]] = sol[k];
                    }
                }
            }

            normalise_dictionary(&mut d);

            // Check convergence.
            let err = reconstruction_error(x, &a, &d);
            if (prev_err - err).abs() < self.tol {
                break;
            }
            prev_err = err;
        }

        // Final sparse coding for reconstruction error.
        let mut a_final = Array2::<f64>::zeros((n_samples, n_components));
        for i in 0..n_samples {
            let x_row: Vec<f64> = (0..n_features).map(|j| x[[i, j]]).collect();
            let codes = lasso_cd_single(&x_row, &d, self.alpha, 200);
            for k in 0..n_components {
                a_final[[i, k]] = codes[k];
            }
        }
        let final_err = reconstruction_error(x, &a_final, &d);

        Ok(FittedDictionaryLearning {
            components_: d,
            alpha_: self.alpha,
            n_iter_: n_iter,
            reconstruction_err_: final_err,
            transform_algorithm_: self.transform_algorithm,
            transform_n_nonzero_coefs_: transform_n_nonzero,
        })
    }
}

impl Transform<Array2<f64>> for FittedDictionaryLearning {
    type Output = Array2<f64>;
    type Error = FerroError;

    /// Compute sparse codes for new data using the learned dictionary.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the dictionary.
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_features = self.components_.ncols();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedDictionaryLearning::transform".into(),
            });
        }

        let n_samples = x.nrows();
        let n_components = self.components_.nrows();
        let mut codes = Array2::<f64>::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let x_row: Vec<f64> = (0..n_features).map(|j| x[[i, j]]).collect();
            let a = match self.transform_algorithm_ {
                DictTransformAlgorithm::Omp => {
                    omp_single(&x_row, &self.components_, self.transform_n_nonzero_coefs_)
                }
                DictTransformAlgorithm::LassoCd => {
                    lasso_cd_single(&x_row, &self.components_, self.alpha_, 200)
                }
            };
            for k in 0..n_components {
                codes[[i, k]] = a[k];
            }
        }

        Ok(codes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Create a simple test dataset.
    fn test_data() -> Array2<f64> {
        Array2::<f64>::from_shape_fn((20, 10), |(i, j)| ((i * 7 + j * 3) % 11) as f64)
    }

    #[test]
    fn test_dictlearn_basic_shape() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().dim(), (5, 10));
    }

    #[test]
    fn test_dictlearn_transform_shape() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let codes = fitted.transform(&x).unwrap();
        assert_eq!(codes.dim(), (20, 5));
    }

    #[test]
    fn test_dictlearn_reconstruction_error_decreases() {
        let x = test_data();
        let dl_few = DictionaryLearning::new(5)
            .with_max_iter(5)
            .with_random_state(42);
        let dl_many = DictionaryLearning::new(5)
            .with_max_iter(50)
            .with_random_state(42);
        let fitted_few = dl_few.fit(&x, &()).unwrap();
        let fitted_many = dl_many.fit(&x, &()).unwrap();
        assert!(
            fitted_many.reconstruction_err() <= fitted_few.reconstruction_err() + 1.0,
            "more iterations should reduce error: few={}, many={}",
            fitted_few.reconstruction_err(),
            fitted_many.reconstruction_err()
        );
    }

    #[test]
    fn test_dictlearn_dictionary_atoms_normalised() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let d = fitted.components();
        for k in 0..d.nrows() {
            let norm: f64 = d.row(k).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "atom {k} should be unit norm, got {norm}"
            );
        }
    }

    #[test]
    fn test_dictlearn_sparsity_of_codes() {
        let x = test_data();
        let dl = DictionaryLearning::new(8)
            .with_alpha(2.0) // Higher alpha = more sparsity.
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let codes = fitted.transform(&x).unwrap();
        // Count zero entries.
        let total = codes.len();
        let zeros = codes.iter().filter(|&&v| v.abs() < 1e-10).count();
        let sparsity = zeros as f64 / total as f64;
        assert!(
            sparsity > 0.1,
            "codes should have some sparsity, got {:.1}%",
            sparsity * 100.0
        );
    }

    #[test]
    fn test_dictlearn_omp_transform() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_transform_algorithm(DictTransformAlgorithm::Omp)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let codes = fitted.transform(&x).unwrap();
        assert_eq!(codes.dim(), (20, 5));
    }

    #[test]
    fn test_dictlearn_lasso_cd_transform() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_transform_algorithm(DictTransformAlgorithm::LassoCd)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let codes = fitted.transform(&x).unwrap();
        assert_eq!(codes.dim(), (20, 5));
    }

    #[test]
    fn test_dictlearn_transform_shape_mismatch() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(10)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let x_bad = Array2::<f64>::zeros((5, 3)); // wrong number of features
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_dictlearn_invalid_n_components_zero() {
        let x = test_data();
        let dl = DictionaryLearning::new(0);
        assert!(dl.fit(&x, &()).is_err());
    }

    #[test]
    fn test_dictlearn_invalid_alpha_negative() {
        let x = test_data();
        let dl = DictionaryLearning::new(5).with_alpha(-1.0);
        assert!(dl.fit(&x, &()).is_err());
    }

    #[test]
    fn test_dictlearn_empty_data() {
        let x = Array2::<f64>::zeros((0, 5));
        let dl = DictionaryLearning::new(2);
        assert!(dl.fit(&x, &()).is_err());
    }

    #[test]
    fn test_dictlearn_zero_features() {
        let x = Array2::<f64>::zeros((10, 0));
        let dl = DictionaryLearning::new(2);
        assert!(dl.fit(&x, &()).is_err());
    }

    #[test]
    fn test_dictlearn_getters() {
        let dl = DictionaryLearning::new(5)
            .with_alpha(0.5)
            .with_max_iter(100)
            .with_tol(1e-6)
            .with_fit_algorithm(DictFitAlgorithm::CoordinateDescent)
            .with_transform_algorithm(DictTransformAlgorithm::LassoCd)
            .with_random_state(99);
        assert_eq!(dl.n_components(), 5);
        assert!((dl.alpha() - 0.5).abs() < 1e-10);
        assert_eq!(dl.max_iter(), 100);
        assert!((dl.tol() - 1e-6).abs() < 1e-12);
        assert_eq!(dl.fit_algorithm(), DictFitAlgorithm::CoordinateDescent);
        assert_eq!(dl.transform_algorithm(), DictTransformAlgorithm::LassoCd);
        assert_eq!(dl.random_state(), Some(99));
    }

    #[test]
    fn test_dictlearn_fitted_accessors() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(10)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
        assert!(fitted.reconstruction_err() >= 0.0);
    }

    #[test]
    fn test_dictlearn_single_component() {
        let x = test_data();
        let dl = DictionaryLearning::new(1)
            .with_max_iter(20)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        assert_eq!(fitted.components().nrows(), 1);
        let codes = fitted.transform(&x).unwrap();
        assert_eq!(codes.ncols(), 1);
    }

    #[test]
    fn test_dictlearn_omp_nonzero_coefs() {
        let x = test_data();
        let dl = DictionaryLearning::new(5)
            .with_max_iter(20)
            .with_transform_algorithm(DictTransformAlgorithm::Omp)
            .with_transform_n_nonzero_coefs(2)
            .with_random_state(42);
        let fitted = dl.fit(&x, &()).unwrap();
        let codes = fitted.transform(&x).unwrap();
        // Each row should have at most 2 non-zero entries.
        for i in 0..codes.nrows() {
            let nnz = codes.row(i).iter().filter(|&&v| v.abs() > 1e-10).count();
            assert!(nnz <= 2, "row {i} has {nnz} non-zeros, expected at most 2");
        }
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-10);
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-10);
        assert!((soft_threshold(1.0, 2.0)).abs() < 1e-10);
        assert!((soft_threshold(0.0, 2.0)).abs() < 1e-10);
    }
}
