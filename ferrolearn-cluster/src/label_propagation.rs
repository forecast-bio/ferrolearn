//! Label Propagation for semi-supervised classification.
//!
//! This module provides [`LabelPropagation`], a graph-based semi-supervised
//! learning algorithm that propagates known labels through a similarity graph
//! to classify unlabeled data points.
//!
//! # Algorithm
//!
//! 1. Build an affinity matrix `W` using either an RBF kernel or a KNN kernel.
//! 2. Construct the propagation matrix `T = D^{-1} W` where `D` is the
//!    diagonal degree matrix.
//! 3. Initialize label distributions `Y` from the known labels.
//! 4. Iterate: `F(t+1) = T * F(t)`, then **clamp** labeled points to their
//!    original labels.
//! 5. Convergence is reached when `||F(t+1) - F(t)|| < tol` or `max_iter`
//!    is exceeded.
//!
//! Labels of `-1` in the target vector indicate unlabeled points.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::LabelPropagation;
//! use ferrolearn_core::Fit;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//! // First and fourth points are labeled; rest are unlabeled (-1).
//! let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);
//!
//! let model = LabelPropagation::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// The kernel used to build the affinity matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPropagationKernel {
    /// RBF (Gaussian) kernel: `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
    Rbf,
    /// KNN kernel: `W[i,j] = 1` if j is among the k nearest neighbors of i
    /// (or vice versa), `0` otherwise.
    Knn,
}

/// Label Propagation semi-supervised classifier (unfitted).
///
/// Holds hyperparameters. Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedLabelPropagation`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LabelPropagation<F> {
    /// The kernel to use for building the affinity matrix.
    pub kernel: LabelPropagationKernel,
    /// Gamma parameter for the RBF kernel.
    pub gamma: F,
    /// Number of neighbors for the KNN kernel.
    pub n_neighbors: usize,
    /// Maximum number of propagation iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
}

impl<F: Float> LabelPropagation<F> {
    /// Create a new `LabelPropagation` with default parameters.
    ///
    /// Defaults: `kernel = Rbf`, `gamma = 20.0`, `n_neighbors = 7`,
    /// `max_iter = 1000`, `tol = 1e-4`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernel: LabelPropagationKernel::Rbf,
            gamma: F::from(20.0).unwrap_or_else(F::one),
            n_neighbors: 7,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
        }
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: LabelPropagationKernel) -> Self {
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
}

impl<F: Float> Default for LabelPropagation<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Label Propagation model.
///
/// Stores the final labels and the label distribution matrix.
/// Implements [`Predict`] on new data by finding the nearest labeled point.
#[derive(Debug, Clone)]
pub struct FittedLabelPropagation<F> {
    /// Final labels for each training sample.
    labels_: Array1<isize>,
    /// Label distribution matrix, shape `(n_samples, n_classes)`.
    label_distributions_: Array2<F>,
    /// Training data, stored for predict.
    x_train_: Array2<F>,
    /// Number of classes.
    n_classes_: usize,
}

impl<F: Float> FittedLabelPropagation<F> {
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
    // Diagonal is zero (no self-loops in standard label propagation).
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

        // Compute distances from i to all other points.
        let mut dists: Vec<(usize, F)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let rj = x.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                (j, sq_euclidean(si, sj))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Set the k nearest neighbors to 1.
        for &(j, _) in dists.iter().take(k) {
            w[i * n + j] = F::one();
            w[j * n + i] = F::one(); // Symmetrize.
        }
    }

    w
}

/// Row-normalize affinity matrix: T = D^{-1} * W.
fn row_normalize<F: Float>(w: &mut [F], n: usize) {
    for i in 0..n {
        let row_sum: F = (0..n).fold(F::zero(), |acc, j| acc + w[i * n + j]);
        if row_sum > F::zero() {
            for j in 0..n {
                w[i * n + j] = w[i * n + j] / row_sum;
            }
        }
    }
}

/// Run the label propagation iterations.
/// Returns the final label distributions, shape (n_samples, n_classes).
fn propagate<F: Float>(
    t_matrix: &[F],
    initial_y: &Array2<F>,
    labeled_mask: &[bool],
    max_iter: usize,
    tol: F,
) -> Array2<F> {
    let n = initial_y.nrows();
    let n_classes = initial_y.ncols();

    let mut f_current = initial_y.clone();
    let mut f_next = Array2::zeros((n, n_classes));

    for _ in 0..max_iter {
        // f_next = T * f_current
        for i in 0..n {
            for c in 0..n_classes {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + t_matrix[i * n + j] * f_current[[j, c]];
                }
                f_next[[i, c]] = sum;
            }
        }

        // Clamp labeled points to their original labels.
        for i in 0..n {
            if labeled_mask[i] {
                for c in 0..n_classes {
                    f_next[[i, c]] = initial_y[[i, c]];
                }
            }
        }

        // Normalize rows so they sum to 1.
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

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<isize>> for LabelPropagation<F> {
    type Fitted = FittedLabelPropagation<F>;
    type Error = FerroError;

    /// Fit the Label Propagation model.
    ///
    /// Labels of `-1` indicate unlabeled points.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `gamma` is not positive
    /// (for RBF kernel) or if there are no labeled points.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<isize>,
    ) -> Result<FittedLabelPropagation<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedLabelPropagation {
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

        if self.kernel == LabelPropagationKernel::Rbf && self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive for RBF kernel".into(),
            });
        }

        if self.kernel == LabelPropagationKernel::Knn && self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1 for KNN kernel".into(),
            });
        }

        // Identify labeled and unlabeled points.
        let labeled_mask: Vec<bool> = y.iter().map(|&l| l >= 0).collect();
        let n_labeled = labeled_mask.iter().filter(|&&m| m).count();

        if n_labeled == 0 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "must have at least one labeled sample (label >= 0)".into(),
            });
        }

        // Find the number of classes.
        let n_classes = y
            .iter()
            .filter(|&&l| l >= 0)
            .map(|&l| l as usize)
            .max()
            .unwrap_or(0)
            + 1;

        // Build the affinity matrix.
        let mut w = match self.kernel {
            LabelPropagationKernel::Rbf => build_rbf_affinity(x, self.gamma),
            LabelPropagationKernel::Knn => build_knn_affinity(x, self.n_neighbors),
        };

        // Row-normalize: T = D^{-1} * W.
        row_normalize(&mut w, n_samples);

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

        // Run propagation.
        let label_distributions = propagate(&w, &initial_y, &labeled_mask, self.max_iter, self.tol);

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

        Ok(FittedLabelPropagation {
            labels_: labels,
            label_distributions_: label_distributions,
            x_train_: x.clone(),
            n_classes_: n_classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLabelPropagation<F> {
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
        // Label first and fifth points; rest unlabeled.
        let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);
        (x, y)
    }

    #[test]
    fn test_label_propagation_basic() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // Labeled points should keep their labels.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);

        // Points near (0,0) should get label 0.
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[3], 0);

        // Points near (10,10) should get label 1.
        assert_eq!(labels[5], 1);
        assert_eq!(labels[6], 1);
        assert_eq!(labels[7], 1);
    }

    #[test]
    fn test_knn_kernel() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new()
            .with_kernel(LabelPropagationKernel::Knn)
            .with_n_neighbors(3);
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        // Same expected behavior.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);
    }

    #[test]
    fn test_predict_on_new_data() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        assert_eq!(new_labels[0], 0);
        assert_eq!(new_labels[1], 1);
    }

    #[test]
    fn test_all_labeled() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0])
            .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // All labels should be preserved since all are labeled.
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[1], 0);
        assert_eq!(fitted.labels()[2], 1);
        assert_eq!(fitted.labels()[3], 1);
    }

    #[test]
    fn test_no_labeled_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![-1, -1, -1, -1]);

        let model = LabelPropagation::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_distributions_shape() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let dist = fitted.label_distributions();
        assert_eq!(dist.nrows(), 8);
        assert_eq!(dist.ncols(), 2); // 2 classes.
    }

    #[test]
    fn test_n_classes() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let bad_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&bad_x);
        assert!(result.is_err());
    }

    #[test]
    fn test_y_length_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]); // Wrong length.

        let model = LabelPropagation::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<isize>::zeros(0);

        let model = LabelPropagation::<f64>::new();
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

        let model = LabelPropagation::<f32>::new();
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

        let model = LabelPropagation::<f64>::new().with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[3], 1);
        assert_eq!(fitted.labels()[6], 2);
    }

    #[test]
    fn test_default_constructor() {
        let model = LabelPropagation::<f64>::default();
        assert_eq!(model.kernel, LabelPropagationKernel::Rbf);
        assert!(model.gamma > 0.0);
        assert_eq!(model.n_neighbors, 7);
    }

    #[test]
    fn test_invalid_gamma() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelPropagation::<f64>::new().with_gamma(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }
}
