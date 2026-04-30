//! Label Spreading for semi-supervised classification.
//!
//! This module provides [`LabelSpreading`], a graph-based semi-supervised
//! learning algorithm similar to [`LabelPropagation`](super::label_propagation::LabelPropagation)
//! but using the **normalized graph Laplacian** for smoother label propagation.
//!
//! # Algorithm
//!
//! 1. Build an affinity matrix `W` using either an RBF or KNN kernel.
//! 2. Construct the normalized Laplacian propagation matrix
//!    `S = D^{-1/2} W D^{-1/2}` where `D` is the diagonal degree matrix.
//! 3. Initialize label distributions `Y` from the known labels.
//! 4. Iterate: `F(t+1) = alpha * S * F(t) + (1 - alpha) * Y`.
//! 5. Convergence is reached when `||F(t+1) - F(t)|| < tol` or `max_iter`
//!    is exceeded.
//!
//! The `alpha` parameter controls the trade-off between the initial label
//! information and the graph structure. With `alpha = 0`, labels are fixed
//! at the initial values; with `alpha = 1`, labels are fully determined by
//! graph propagation.
//!
//! Labels of `-1` in the target vector indicate unlabeled points.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::LabelSpreading;
//! use ferrolearn_core::Fit;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);
//!
//! let model = LabelSpreading::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// The kernel used to build the affinity matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSpreadingKernel {
    /// RBF (Gaussian) kernel: `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
    Rbf,
    /// KNN kernel: `W[i,j] = 1` if j is among the k nearest neighbors of i
    /// (or vice versa), `0` otherwise.
    Knn,
}

/// Label Spreading semi-supervised classifier (unfitted).
///
/// Holds hyperparameters. Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedLabelSpreading`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LabelSpreading<F> {
    /// The kernel to use for building the affinity matrix.
    pub kernel: LabelSpreadingKernel,
    /// Gamma parameter for the RBF kernel.
    pub gamma: F,
    /// Number of neighbors for the KNN kernel.
    pub n_neighbors: usize,
    /// Maximum number of spreading iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Clamping factor: controls the balance between initial labels and
    /// graph structure. Must be in `[0, 1)`.
    pub alpha: F,
}

impl<F: Float> LabelSpreading<F> {
    /// Create a new `LabelSpreading` with default parameters.
    ///
    /// Defaults: `kernel = Rbf`, `gamma = 20.0`, `n_neighbors = 7`,
    /// `max_iter = 30`, `tol = 1e-4`, `alpha = 0.2`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernel: LabelSpreadingKernel::Rbf,
            gamma: F::from(20.0).unwrap_or_else(F::one),
            n_neighbors: 7,
            max_iter: 30,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            alpha: F::from(0.2).unwrap_or_else(F::zero),
        }
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: LabelSpreadingKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the gamma parameter for the RBF kernel.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of neighbors for the KNN kernel.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the clamping factor `alpha`.
    ///
    /// Must be in `[0, 1)`. A value of `0` means labels are fully determined
    /// by the initial labels (no spreading). Higher values give more weight
    /// to the graph structure.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> Default for LabelSpreading<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Label Spreading model.
///
/// Stores the final labels and the label distribution matrix.
/// Implements [`Predict`] on new data by finding the nearest labeled point.
#[derive(Debug, Clone)]
pub struct FittedLabelSpreading<F> {
    /// Final labels for each training sample.
    labels_: Array1<isize>,
    /// Label distribution matrix, shape `(n_samples, n_classes)`.
    label_distributions_: Array2<F>,
    /// Training data, stored for predict.
    x_train_: Array2<F>,
    /// Number of classes.
    n_classes_: usize,
}

impl<F: Float> FittedLabelSpreading<F> {
    /// Return the final labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the label distribution matrix.
    ///
    /// Shape: `(n_samples, n_classes)`. Each row sums to approximately 1.
    #[must_use]
    pub fn label_distributions(&self) -> &Array2<F> {
        &self.label_distributions_
    }

    /// Return the number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes_
    }
}

impl<F: Float + Send + Sync + 'static> FittedLabelSpreading<F> {
    /// Per-class probability for each query sample. Mirrors sklearn
    /// `LabelSpreading.predict_proba`. For each query the result is the
    /// nearest training point's label distribution row.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count does not
    /// match the training data.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        let expected = self.x_train_.ncols();
        if n_features != expected {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![n_features],
                context: "number of features must match the training data".into(),
            });
        }
        let n_new = x.nrows();
        let n_train = self.x_train_.nrows();
        let mut out = Array2::<F>::zeros((n_new, self.n_classes_));
        for i in 0..n_new {
            let ri = x.row(i);
            let si = ri.as_slice().unwrap_or(&[]);
            let mut best_j = 0;
            let mut best_dist = F::max_value();
            for j in 0..n_train {
                let rj = self.x_train_.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                let d = sq_euclidean(si, sj);
                if d < best_dist {
                    best_dist = d;
                    best_j = j;
                }
            }
            for c in 0..self.n_classes_ {
                out[[i, c]] = self.label_distributions_[[best_j, c]];
            }
        }
        Ok(out)
    }

    /// Mean accuracy on the given test data and labels. Mirrors sklearn
    /// `ClassifierMixin.score`. Test samples with `y == -1` are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<isize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let mut total = 0usize;
        let mut correct = 0usize;
        for (p, t) in preds.iter().zip(y.iter()) {
            if *t == -1 {
                continue;
            }
            total += 1;
            if p == t {
                correct += 1;
            }
        }
        if total == 0 {
            return Ok(F::zero());
        }
        Ok(F::from(correct).unwrap() / F::from(total).unwrap())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two slices.
#[inline]
fn sq_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Build the RBF affinity matrix.
fn build_rbf_affinity<F: Float>(x: &Array2<F>, gamma: F) -> Vec<F> {
    let n = x.nrows();
    let mut w = vec![F::zero(); n * n];

    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        for j in (i + 1)..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            let d = sq_euclidean(si, sj);
            let v = (-gamma * d).exp();
            w[i * n + j] = v;
            w[j * n + i] = v;
        }
    }
    w
}

/// Build the KNN affinity matrix.
fn build_knn_affinity<F: Float>(x: &Array2<F>, k: usize) -> Vec<F> {
    let n = x.nrows();
    let k = k.min(n - 1);
    let mut w = vec![F::zero(); n * n];

    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);

        let mut dists: Vec<(usize, F)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let rj = x.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                (j, sq_euclidean(si, sj))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, _) in dists.iter().take(k) {
            w[i * n + j] = F::one();
            w[j * n + i] = F::one();
        }
    }

    w
}

/// Compute the normalized Laplacian propagation matrix S = D^{-1/2} W D^{-1/2}.
fn normalized_laplacian<F: Float>(w: &[F], n: usize) -> Vec<F> {
    // Compute D^{-1/2}.
    let mut d_inv_sqrt = vec![F::zero(); n];
    for i in 0..n {
        let row_sum: F = (0..n).fold(F::zero(), |acc, j| acc + w[i * n + j]);
        if row_sum > F::zero() {
            d_inv_sqrt[i] = F::one() / row_sum.sqrt();
        }
    }

    // S[i,j] = D_inv_sqrt[i] * W[i,j] * D_inv_sqrt[j].
    let mut s = vec![F::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = d_inv_sqrt[i] * w[i * n + j] * d_inv_sqrt[j];
        }
    }

    s
}

/// Run the label spreading iterations.
/// Update rule: F(t+1) = alpha * S * F(t) + (1 - alpha) * Y
fn spread<F: Float>(
    s_matrix: &[F],
    initial_y: &Array2<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
) -> Array2<F> {
    let n = initial_y.nrows();
    let n_classes = initial_y.ncols();
    let one_minus_alpha = F::one() - alpha;

    let mut f_current = initial_y.clone();
    let mut f_next = Array2::zeros((n, n_classes));

    for _ in 0..max_iter {
        // f_next = alpha * S * f_current + (1 - alpha) * Y
        for i in 0..n {
            for c in 0..n_classes {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + s_matrix[i * n + j] * f_current[[j, c]];
                }
                f_next[[i, c]] = alpha * sum + one_minus_alpha * initial_y[[i, c]];
            }
        }

        // Normalize rows.
        for i in 0..n {
            let row_sum: F = (0..n_classes).fold(F::zero(), |acc, c| acc + f_next[[i, c]]);
            if row_sum > F::zero() {
                for c in 0..n_classes {
                    f_next[[i, c]] = f_next[[i, c]] / row_sum;
                }
            }
        }

        // Check convergence.
        let mut diff = F::zero();
        for i in 0..n {
            for c in 0..n_classes {
                let d = f_next[[i, c]] - f_current[[i, c]];
                diff = diff + d * d;
            }
        }

        std::mem::swap(&mut f_current, &mut f_next);

        if diff.sqrt() < tol {
            break;
        }
    }

    f_current
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<isize>> for LabelSpreading<F> {
    type Fitted = FittedLabelSpreading<F>;
    type Error = FerroError;

    /// Fit the Label Spreading model.
    ///
    /// Labels of `-1` indicate unlabeled points.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is not in `[0, 1)`,
    /// `gamma` is not positive (for RBF), or there are no labeled points.
    fn fit(&self, x: &Array2<F>, y: &Array1<isize>) -> Result<FittedLabelSpreading<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedLabelSpreading {
                labels_: Array1::zeros(0),
                label_distributions_: Array2::zeros((0, 0)),
                x_train_: x.clone(),
                n_classes_: 0,
            });
        }

        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y must have the same length as x rows".into(),
            });
        }

        if self.alpha < F::zero() || self.alpha >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in [0, 1)".into(),
            });
        }

        if self.kernel == LabelSpreadingKernel::Rbf && self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive for RBF kernel".into(),
            });
        }

        if self.kernel == LabelSpreadingKernel::Knn && self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1 for KNN kernel".into(),
            });
        }

        // Identify labeled points.
        let n_labeled = y.iter().filter(|&&l| l >= 0).count();
        if n_labeled == 0 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "must have at least one labeled sample (label >= 0)".into(),
            });
        }

        // Find number of classes.
        let n_classes = y
            .iter()
            .filter(|&&l| l >= 0)
            .map(|&l| l as usize)
            .max()
            .unwrap_or(0)
            + 1;

        // Build the affinity matrix.
        let w = match self.kernel {
            LabelSpreadingKernel::Rbf => build_rbf_affinity(x, self.gamma),
            LabelSpreadingKernel::Knn => build_knn_affinity(x, self.n_neighbors),
        };

        // Build the normalized Laplacian propagation matrix S.
        let s = normalized_laplacian(&w, n_samples);

        // Build initial label distribution Y.
        let mut initial_y = Array2::from_elem((n_samples, n_classes), F::zero());
        for (i, &label) in y.iter().enumerate() {
            if label >= 0 {
                let c = label as usize;
                if c < n_classes {
                    initial_y[[i, c]] = F::one();
                }
            } else {
                // Unlabeled: uniform distribution.
                let uniform = F::one() / F::from(n_classes).unwrap_or_else(F::one);
                for c in 0..n_classes {
                    initial_y[[i, c]] = uniform;
                }
            }
        }

        // Run spreading.
        let label_distributions = spread(&s, &initial_y, self.alpha, self.max_iter, self.tol);

        // Extract final labels (argmax of each row).
        let labels: Array1<isize> = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let mut best_c = 0;
                    let mut best_v = label_distributions[[i, 0]];
                    for c in 1..n_classes {
                        if label_distributions[[i, c]] > best_v {
                            best_v = label_distributions[[i, c]];
                            best_c = c;
                        }
                    }
                    best_c as isize
                })
                .collect(),
        );

        Ok(FittedLabelSpreading {
            labels_: labels,
            label_distributions_: label_distributions,
            x_train_: x.clone(),
            n_classes_: n_classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLabelSpreading<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict labels for new data by finding the nearest training point
    /// and returning its label.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count does not match.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let n_features = x.ncols();
        let expected_features = self.x_train_.ncols();

        if n_features != expected_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected_features],
                actual: vec![n_features],
                context: "number of features must match the training data".into(),
            });
        }

        let n_new = x.nrows();
        let n_train = self.x_train_.nrows();
        let mut labels = Array1::zeros(n_new);

        for i in 0..n_new {
            let ri = x.row(i);
            let si = ri.as_slice().unwrap_or(&[]);
            let mut best_j = 0;
            let mut best_dist = F::max_value();

            for j in 0..n_train {
                let rj = self.x_train_.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                let d = sq_euclidean(si, sj);
                if d < best_dist {
                    best_dist = d;
                    best_j = j;
                }
            }

            labels[i] = self.labels_[best_j];
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two blobs with partially labeled data.
    fn make_semi_supervised() -> (Array2<f64>, Array1<isize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1,
                10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);
        (x, y)
    }

    #[test]
    fn test_label_spreading_basic() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // Points near (0,0) should get label 0.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[3], 0);

        // Points near (10,10) should get label 1.
        assert_eq!(labels[4], 1);
        assert_eq!(labels[5], 1);
        assert_eq!(labels[6], 1);
        assert_eq!(labels[7], 1);
    }

    #[test]
    fn test_alpha_zero_recovers_initial() {
        // With alpha = 0: F = (1-0)*Y = Y, so labels are the initial ones.
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0])
            .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let model = LabelSpreading::<f64>::new().with_alpha(0.0);
        let fitted = model.fit(&x, &y).unwrap();

        // All labeled, alpha=0 means Y is unchanged.
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[1], 0);
        assert_eq!(fitted.labels()[2], 1);
        assert_eq!(fitted.labels()[3], 1);
    }

    #[test]
    fn test_convergence() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new().with_max_iter(1000);
        let fitted = model.fit(&x, &y).unwrap();

        // Should have converged with reasonable results.
        assert_eq!(fitted.labels().len(), 8);

        // Label distributions should sum to ~1 for each sample.
        let dists = fitted.label_distributions();
        for i in 0..8 {
            let row_sum: f64 = (0..dists.ncols()).map(|c| dists[[i, c]]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_knn_kernel() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new()
            .with_kernel(LabelSpreadingKernel::Knn)
            .with_n_neighbors(3);
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);
    }

    #[test]
    fn test_predict_on_new_data() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        assert_eq!(new_labels[0], 0);
        assert_eq!(new_labels[1], 1);
    }

    #[test]
    fn test_invalid_alpha() {
        let (x, y) = make_semi_supervised();

        // alpha >= 1.0 is invalid.
        let model = LabelSpreading::<f64>::new().with_alpha(1.0);
        assert!(model.fit(&x, &y).is_err());

        // alpha < 0 is invalid.
        let model = LabelSpreading::<f64>::new().with_alpha(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_no_labeled_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![-1, -1, -1, -1]);

        let model = LabelSpreading::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_label_distributions_shape() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let dist = fitted.label_distributions();
        assert_eq!(dist.nrows(), 8);
        assert_eq!(dist.ncols(), 2);
    }

    #[test]
    fn test_n_classes() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let bad_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&bad_x).is_err());
    }

    #[test]
    fn test_y_length_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelSpreading::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<isize>::zeros(0);

        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.labels().len(), 0);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);

        let model = LabelSpreading::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.labels().len(), 6);
    }

    #[test]
    fn test_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 0.0, 10.0, 0.1,
                10.0, 0.0, 10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1, 2, -1, -1]);

        let model = LabelSpreading::<f64>::new().with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[3], 1);
        assert_eq!(fitted.labels()[6], 2);
    }

    #[test]
    fn test_default_constructor() {
        let model = LabelSpreading::<f64>::default();
        assert_eq!(model.kernel, LabelSpreadingKernel::Rbf);
        assert!(model.gamma > 0.0);
        assert_eq!(model.n_neighbors, 7);
        assert_relative_eq!(model.alpha, 0.2);
    }

    #[test]
    fn test_invalid_gamma() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelSpreading::<f64>::new().with_gamma(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_alpha_affects_results() {
        let (x, y) = make_semi_supervised();

        // With alpha close to 0, results should be close to initial labels.
        let model_low = LabelSpreading::<f64>::new().with_alpha(0.01);
        let fitted_low = model_low.fit(&x, &y).unwrap();

        // With alpha close to 1, results should be more influenced by graph.
        let model_high = LabelSpreading::<f64>::new().with_alpha(0.99);
        let fitted_high = model_high.fit(&x, &y).unwrap();

        // Both should produce the same labels for well-separated clusters.
        assert_eq!(fitted_low.labels()[0], 0);
        assert_eq!(fitted_high.labels()[0], 0);
        assert_eq!(fitted_low.labels()[4], 1);
        assert_eq!(fitted_high.labels()[4], 1);
    }
}
