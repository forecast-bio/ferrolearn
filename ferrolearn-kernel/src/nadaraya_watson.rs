//! Nadaraya-Watson kernel regression (local constant).
//!
//! `ŷ(x) = Σ K((x - x_i)/h) y_i / Σ K((x - x_i)/h)`

use ferrolearn_neighbors::balltree::BallTree;
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::bandwidth::{self, BandwidthStrategy, CvStrategy};
use crate::kernels::{DynKernel, GaussianKernel, Kernel};
use crate::weights;

/// Boundary correction method.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundaryCorrection {
    /// Reflect data near boundaries.
    Reflection,
    /// Use local linear regression near boundaries.
    LocalLinear,
}

/// Nadaraya-Watson kernel regression (local constant).
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
/// - `K`: Kernel type. Use a concrete kernel (e.g., `GaussianKernel`) for
///   maximum performance, or `DynKernel<F>` for runtime selection.
#[derive(Debug)]
pub struct NadarayaWatson<F: Float + Send + Sync + 'static, K: Kernel<F> = DynKernel<F>> {
    /// Kernel function.
    pub kernel: K,
    /// Bandwidth strategy.
    pub bandwidth: BandwidthStrategy<F>,
    /// Optional boundary correction.
    pub boundary_correction: Option<BoundaryCorrection>,
}

impl<F: Float + Send + Sync + 'static> NadarayaWatson<F, GaussianKernel> {
    /// Create with Gaussian kernel and cross-validated bandwidth.
    pub fn new() -> Self {
        Self {
            kernel: GaussianKernel,
            bandwidth: BandwidthStrategy::CrossValidated {
                cv: CvStrategy::Loo,
                per_dimension: false,
            },
            boundary_correction: None,
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for NadarayaWatson<F, GaussianKernel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static, K: Kernel<F>> NadarayaWatson<F, K> {
    /// Create with a specific kernel and bandwidth strategy.
    pub fn with_kernel(kernel: K, bandwidth: BandwidthStrategy<F>) -> Self {
        Self {
            kernel,
            bandwidth,
            boundary_correction: None,
        }
    }

    /// Set boundary correction method.
    #[must_use]
    pub fn boundary(mut self, correction: BoundaryCorrection) -> Self {
        self.boundary_correction = Some(correction);
        self
    }
}

/// Minimum samples to build a BallTree for compact-support kernels.
const BALLTREE_THRESHOLD: usize = 500;

/// Fitted Nadaraya-Watson model.
pub struct FittedNadarayaWatson<F: Float + Send + Sync + 'static, K: Kernel<F> = DynKernel<F>> {
    /// Training features.
    pub x_train: Array2<F>,
    /// Training targets.
    pub y_train: Array1<F>,
    /// Fitted per-dimension bandwidth.
    pub bandwidth: Array1<F>,
    /// Kernel function.
    pub kernel: K,
    /// Boundary correction.
    pub boundary_correction: Option<BoundaryCorrection>,
    /// Data bounds for boundary detection.
    pub x_min: Array1<F>,
    /// Data upper bounds.
    pub x_max: Array1<F>,
    /// Optional BallTree for compact-support kernel acceleration.
    ball_tree: Option<BallTree>,
}

impl<F, K> Fit<Array2<F>, Array1<F>> for NadarayaWatson<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F> + Clone,
{
    type Fitted = FittedNadarayaWatson<F, K>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedNadarayaWatson<F, K>, FerroError> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "NadarayaWatson::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        let bw = bandwidth::resolve_bandwidth(
            &self.bandwidth,
            x,
            y,
            &self.kernel,
            0, // polynomial order 0 for NW
        )?;

        // Data bounds for boundary detection
        let x_min = x
            .columns()
            .into_iter()
            .map(|col| col.iter().copied().fold(F::infinity(), F::min))
            .collect();
        let x_max = x
            .columns()
            .into_iter()
            .map(|col| col.iter().copied().fold(F::neg_infinity(), F::max))
            .collect();

        // Build BallTree for compact-support kernels on large datasets
        let ball_tree = if self.kernel.has_compact_support() && n_samples >= BALLTREE_THRESHOLD {
            Some(BallTree::build(x))
        } else {
            None
        };

        Ok(FittedNadarayaWatson {
            x_train: x.clone(),
            y_train: y.clone(),
            bandwidth: bw,
            kernel: self.kernel.clone(),
            boundary_correction: self.boundary_correction,
            x_min,
            x_max,
            ball_tree,
        })
    }
}

impl<F, K> Predict<Array2<F>> for FittedNadarayaWatson<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    type Output = Array1<F>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        if x.ncols() != self.x_train.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.x_train.ncols()],
                actual: vec![x.nrows(), x.ncols()],
                context: "predict feature count must match training data".into(),
            });
        }

        match self.boundary_correction {
            Some(BoundaryCorrection::Reflection) => self.predict_reflection(x),
            Some(BoundaryCorrection::LocalLinear) => self.predict_local_linear(x),
            None => self.predict_standard(x),
        }
    }
}

impl<F, K> FittedNadarayaWatson<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    /// Standard NW prediction (no boundary correction).
    fn predict_standard(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let w = if let Some(tree) = &self.ball_tree {
            weights::compute_kernel_weights_balltree(
                x,
                &self.x_train,
                &self.bandwidth,
                &self.kernel,
                tree,
            )
        } else {
            weights::compute_kernel_weights(x, &self.x_train, &self.bandwidth, &self.kernel)
        };
        Ok(weights::nw_predict_from_weights(&w, &self.y_train))
    }

    /// Prediction with reflection boundary correction.
    fn predict_reflection(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let (x_aug, y_aug) = self.reflect_data();
        let w = weights::compute_kernel_weights(x, &x_aug, &self.bandwidth, &self.kernel);
        Ok(weights::nw_predict_from_weights(&w, &y_aug))
    }

    /// Prediction with local linear boundary correction.
    fn predict_local_linear(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_pred = x.nrows();
        let mut predictions = Array1::zeros(n_pred);

        for i in 0..n_pred {
            let xi = x.row(i);
            let is_boundary = self.is_boundary_point(&xi);

            if is_boundary {
                predictions[i] = self.local_linear_at_point(&xi)?;
            } else {
                let x_single = xi.to_owned().insert_axis(ndarray::Axis(0));
                let w = weights::compute_kernel_weights(
                    &x_single,
                    &self.x_train,
                    &self.bandwidth,
                    &self.kernel,
                );
                predictions[i] = weights::nw_predict_from_weights(&w, &self.y_train)[0];
            }
        }

        Ok(predictions)
    }

    /// Check if a point is near the data boundary.
    fn is_boundary_point(&self, x: &ndarray::ArrayView1<F>) -> bool {
        for j in 0..x.len() {
            if x[j] < self.x_min[j] + self.bandwidth[j] || x[j] > self.x_max[j] - self.bandwidth[j]
            {
                return true;
            }
        }
        false
    }

    /// Local linear prediction at a single boundary point.
    fn local_linear_at_point(&self, x: &ndarray::ArrayView1<F>) -> Result<F, FerroError> {
        let n_train = self.x_train.nrows();
        let n_features = self.x_train.ncols();
        let x_single = x.to_owned().insert_axis(ndarray::Axis(0));

        let w_row = weights::compute_kernel_weights(
            &x_single,
            &self.x_train,
            &self.bandwidth,
            &self.kernel,
        );
        let w = w_row.row(0);
        let w_sum: F = w.iter().copied().fold(F::zero(), |a, b| a + b);

        if w_sum < F::from(1e-10).unwrap() {
            return Ok(self.y_train.sum() / F::from(self.y_train.len()).unwrap());
        }

        // Weighted least squares: y = a + b*(X - x)
        // Design matrix: [1, diff_1, diff_2, ...]
        let n_cols = 1 + n_features;
        let mut xtw_x = Array2::<F>::zeros((n_cols, n_cols));
        let mut xtw_y = Array1::<F>::zeros(n_cols);
        let reg = F::from(1e-10).unwrap();

        for k in 0..n_train {
            let wi = w[k];
            if wi <= F::zero() {
                continue;
            }
            let mut design = vec![F::one()];
            for j in 0..n_features {
                design.push(self.x_train[[k, j]] - x[j]);
            }
            for a in 0..n_cols {
                xtw_y[a] = xtw_y[a] + wi * design[a] * self.y_train[k];
                for b in 0..n_cols {
                    xtw_x[[a, b]] = xtw_x[[a, b]] + wi * design[a] * design[b];
                }
            }
        }

        // Add regularization
        for a in 0..n_cols {
            xtw_x[[a, a]] = xtw_x[[a, a]] + reg;
        }

        // Solve via Cholesky or fall back to weighted average
        match solve_symmetric(&xtw_x, &xtw_y) {
            Some(beta) if beta[0].is_finite() => Ok(beta[0]),
            _ => Ok(w.dot(&self.y_train) / w_sum),
        }
    }

    /// Reflect data near boundaries.
    fn reflect_data(&self) -> (Array2<F>, Array1<F>) {
        let n_features = self.x_train.ncols();
        let mut x_list = vec![self.x_train.clone()];
        let mut y_list = vec![self.y_train.clone()];

        for dim in 0..n_features {
            let h = self.bandwidth[dim];

            // Near lower boundary
            for i in 0..self.x_train.nrows() {
                if self.x_train[[i, dim]] < self.x_min[dim] + h {
                    let mut reflected = self.x_train.row(i).to_owned();
                    reflected[dim] = F::from(2.0).unwrap() * self.x_min[dim] - reflected[dim];
                    x_list.push(reflected.insert_axis(ndarray::Axis(0)));
                    y_list.push(Array1::from_vec(vec![self.y_train[i]]));
                }
            }

            // Near upper boundary
            for i in 0..self.x_train.nrows() {
                if self.x_train[[i, dim]] > self.x_max[dim] - h {
                    let mut reflected = self.x_train.row(i).to_owned();
                    reflected[dim] = F::from(2.0).unwrap() * self.x_max[dim] - reflected[dim];
                    x_list.push(reflected.insert_axis(ndarray::Axis(0)));
                    y_list.push(Array1::from_vec(vec![self.y_train[i]]));
                }
            }
        }

        let x_aug = ndarray::concatenate(
            ndarray::Axis(0),
            &x_list.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let y_aug = ndarray::concatenate(
            ndarray::Axis(0),
            &y_list.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        (x_aug, y_aug)
    }

    /// Get the normalized weight matrix (hat matrix rows) for query points.
    pub fn get_weights(&self, x: &Array2<F>) -> Array2<F> {
        weights::compute_normalized_weights(x, &self.x_train, &self.bandwidth, &self.kernel)
    }
}

/// Simple symmetric positive definite solver (Cholesky-like).
fn solve_symmetric<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Option<Array1<F>> {
    let n = a.nrows();
    // Gaussian elimination with partial pivoting
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < F::from(1e-15).unwrap() {
            return None;
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                aug[[row, j]] = aug[[row, j]] - factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut result = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * result[j];
        }
        if aug[[i, i]].abs() < F::from(1e-15).unwrap() {
            return None;
        }
        result[i] = sum / aug[[i, i]];
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::GaussianKernel;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    fn make_sin_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x_data: Vec<f64> = (0..n)
            .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
            .collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y = x.column(0).mapv(f64::sin);
        (x, y)
    }

    #[test]
    fn fit_predict_basic() {
        let (x, y) = make_sin_data(50);
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.5));
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 50);
        // Predictions should be finite
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn predict_constant() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(f64::from).collect()).unwrap();
        let y = Array1::from_elem(10, 5.0);
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert_abs_diff_eq!(p, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn predict_in_range() {
        // NW predictions should be in [min(y), max(y)]
        let (x, y) = make_sin_data(50);
        let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.5));
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        for &p in &pred {
            assert!(
                p >= y_min - 1e-10 && p <= y_max + 1e-10,
                "Prediction {p} outside [{y_min}, {y_max}]"
            );
        }
    }

    #[test]
    fn small_bandwidth_interpolates() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.01));
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for i in 0..5 {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn large_bandwidth_averages() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];
        let y_mean = y.sum() / 5.0;
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(100.0));
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert_abs_diff_eq!(p, y_mean, epsilon = 0.1);
        }
    }

    #[test]
    fn fit_rejects_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = array![1.0f64];
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn fit_rejects_mismatched_y() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0f64]; // Wrong length
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0f64];
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
        let fitted = model.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    #[test]
    fn silverman_bandwidth() {
        let (x, y) = make_sin_data(50);
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Silverman);
        let fitted = model.fit(&x, &y).unwrap();
        assert!(fitted.bandwidth[0] > 0.0);
        let pred = fitted.predict(&x).unwrap();
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn boundary_reflection() {
        // y = x on [0, 1], predict at boundary x=1.0
        let x_data: Vec<f64> = (0..50).map(|i| f64::from(i) / 49.0).collect();
        let x = Array2::from_shape_vec((50, 1), x_data).unwrap();
        let y = x.column(0).to_owned();
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.2))
            .boundary(BoundaryCorrection::Reflection);
        let fitted = model.fit(&x, &y).unwrap();
        let x_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let pred = fitted.predict(&x_test).unwrap();
        // With reflection, boundary bias should be reduced
        assert!(
            (pred[0] - 1.0).abs() < 0.15,
            "Boundary prediction {:.4} should be close to 1.0",
            pred[0]
        );
    }

    #[test]
    fn multivariate_2d() {
        let n = 100;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| vec![i as f64 / n as f64, (i as f64 / n as f64).powi(2)])
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let model = NadarayaWatson::with_kernel(
            GaussianKernel,
            BandwidthStrategy::PerDimension(array![0.2, 0.2]),
        );
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), n);
        for &p in &pred {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn row_permutation_invariant() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0f64];

        // Original order
        let model = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(1.0));
        let pred1 = model.fit(&x, &y).unwrap().predict(&x).unwrap();

        // Permuted order
        let perm = [3, 0, 4, 1, 2];
        let x_perm =
            Array2::from_shape_vec((5, 1), perm.iter().map(|&i| x[[i, 0]]).collect()).unwrap();
        let y_perm = Array1::from_vec(perm.iter().map(|&i| y[i]).collect());
        let pred2 = model.fit(&x_perm, &y_perm).unwrap().predict(&x).unwrap();

        for i in 0..5 {
            assert_abs_diff_eq!(pred1[i], pred2[i], epsilon = 1e-12);
        }
    }
}
