//! Local polynomial kernel regression.
//!
//! Fits a weighted polynomial locally at each prediction point.
//! Order 0 is equivalent to Nadaraya-Watson; order 1 is local linear.

use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::bandwidth::{self, BandwidthStrategy, CvStrategy};
use crate::kernels::{DynKernel, GaussianKernel, Kernel};
use crate::weights;

/// Polynomial order strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStrategy {
    /// Fixed polynomial order.
    Fixed(usize),
    /// Cross-validated order selection.
    CrossValidated {
        /// Maximum order to consider.
        max_order: usize,
    },
}

/// Local polynomial kernel regression.
#[derive(Debug)]
pub struct LocalPolynomialRegression<F: Float + Send + Sync + 'static, K: Kernel<F> = DynKernel<F>>
{
    /// Kernel function.
    pub kernel: K,
    /// Bandwidth strategy.
    pub bandwidth: BandwidthStrategy<F>,
    /// Polynomial order strategy.
    pub order: OrderStrategy,
    /// Tikhonov regularization parameter.
    pub regularization: F,
}

impl<F: Float + Send + Sync + 'static> LocalPolynomialRegression<F, GaussianKernel> {
    /// Create with Gaussian kernel, order 1, cross-validated bandwidth.
    pub fn new() -> Self {
        Self {
            kernel: GaussianKernel,
            bandwidth: BandwidthStrategy::CrossValidated {
                cv: CvStrategy::Loo,
                per_dimension: false,
            },
            order: OrderStrategy::Fixed(1),
            regularization: F::from(1e-10).unwrap(),
        }
    }
}

impl<F: Float + Send + Sync + 'static> Default for LocalPolynomialRegression<F, GaussianKernel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync + 'static, K: Kernel<F>> LocalPolynomialRegression<F, K> {
    /// Create with a specific kernel and settings.
    pub fn with_kernel(kernel: K, bandwidth: BandwidthStrategy<F>, order: usize) -> Self {
        Self {
            kernel,
            bandwidth,
            order: OrderStrategy::Fixed(order),
            regularization: F::from(1e-10).unwrap(),
        }
    }

    /// Set regularization parameter.
    #[must_use]
    pub fn regularization(mut self, reg: F) -> Self {
        self.regularization = reg;
        self
    }
}

/// Fitted local polynomial regression model.
#[derive(Debug)]
pub struct FittedLocalPolynomialRegression<
    F: Float + Send + Sync + 'static,
    K: Kernel<F> = DynKernel<F>,
> {
    /// Training features.
    pub x_train: Array2<F>,
    /// Training targets.
    pub y_train: Array1<F>,
    /// Fitted per-dimension bandwidth.
    pub bandwidth: Array1<F>,
    /// Kernel function.
    pub kernel: K,
    /// Resolved polynomial order.
    pub order: usize,
    /// Regularization parameter.
    pub regularization: F,
}

impl<F, K> Fit<Array2<F>, Array1<F>> for LocalPolynomialRegression<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F> + Clone,
{
    type Fitted = FittedLocalPolynomialRegression<F, K>;
    type Error = FerroError;

    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedLocalPolynomialRegression<F, K>, FerroError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LocalPolynomialRegression::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        // Resolve order
        let order = match &self.order {
            OrderStrategy::Fixed(o) => *o,
            OrderStrategy::CrossValidated { max_order } => {
                select_order_cv(x, y, &self.kernel, &self.bandwidth, *max_order)?
            }
        };

        // Resolve bandwidth
        let bw = bandwidth::resolve_bandwidth(&self.bandwidth, x, y, &self.kernel, order)?;

        Ok(FittedLocalPolynomialRegression {
            x_train: x.clone(),
            y_train: y.clone(),
            bandwidth: bw,
            kernel: self.kernel.clone(),
            order,
            regularization: self.regularization,
        })
    }
}

impl<F, K> Predict<Array2<F>> for FittedLocalPolynomialRegression<F, K>
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

        let n_pred = x.nrows();
        let mut predictions = Array1::zeros(n_pred);

        for i in 0..n_pred {
            predictions[i] = self.predict_point(&x.row(i));
        }

        Ok(predictions)
    }
}

impl<F, K> FittedLocalPolynomialRegression<F, K>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    /// Predict at a single point using local polynomial regression.
    fn predict_point(&self, x: &ndarray::ArrayView1<F>) -> F {
        let n_train = self.x_train.nrows();
        let x_single = x.to_owned().insert_axis(ndarray::Axis(0));

        let w_matrix = weights::compute_kernel_weights(
            &x_single,
            &self.x_train,
            &self.bandwidth,
            &self.kernel,
        );
        let w = w_matrix.row(0);

        // Build polynomial design matrix
        let design = build_design_matrix(&self.x_train, x, self.order);
        let n_cols = design.ncols();

        // Weighted least squares with Tikhonov regularization:
        // Augment [√W * X; √λ * I] and [√W * y; 0]
        let sqrt_reg = self.regularization.sqrt();

        // Build augmented system
        let n_aug = n_train + n_cols;
        let mut a = Array2::<F>::zeros((n_aug, n_cols));
        let mut b = Array1::<F>::zeros(n_aug);

        for k in 0..n_train {
            let sqrt_w = w[k].max(F::zero()).sqrt();
            for j in 0..n_cols {
                a[[k, j]] = sqrt_w * design[[k, j]];
            }
            b[k] = sqrt_w * self.y_train[k];
        }
        for j in 0..n_cols {
            a[[n_train + j, j]] = sqrt_reg;
        }

        // Solve via normal equations: A^T A beta = A^T b
        let ata = a.t().dot(&a);
        let atb = a.t().dot(&b);

        match solve_spd(&ata, &atb) {
            Some(beta) if beta[0].is_finite() => beta[0],
            _ => {
                // Fallback: Nadaraya-Watson
                let w_sum: F = w.iter().copied().fold(F::zero(), |a, b| a + b);
                if w_sum > F::zero() {
                    w.dot(&self.y_train) / w_sum
                } else {
                    self.y_train.sum() / F::from(self.y_train.len()).unwrap()
                }
            }
        }
    }
}

/// Build polynomial design matrix centered at query point.
///
/// Columns are: `[1, (X_1 - x_1), ..., (X_d - x_d), (X_1-x_1)^2, (X_1-x_1)(X_2-x_2), ...]`
fn build_design_matrix<F: Float>(
    x_train: &Array2<F>,
    x_query: &ndarray::ArrayView1<F>,
    order: usize,
) -> Array2<F> {
    let n = x_train.nrows();
    let d = x_train.ncols();

    // Compute number of columns
    let n_cols = count_polynomial_terms(d, order);
    let mut design = Array2::zeros((n, n_cols));

    // Compute differences
    let mut diff = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            diff[[i, j]] = x_train[[i, j]] - x_query[j];
        }
    }

    // Column 0: intercept
    for i in 0..n {
        design[[i, 0]] = F::one();
    }

    let mut col = 1;

    // Order 1 through `order`
    for deg in 1..=order {
        // Generate all combinations_with_replacement(0..d, deg)
        let combos = combinations_with_replacement(d, deg);
        for combo in &combos {
            for i in 0..n {
                let mut val = F::one();
                for &idx in combo {
                    val = val * diff[[i, idx]];
                }
                design[[i, col]] = val;
            }
            col += 1;
        }
    }

    design
}

/// Count polynomial terms: sum of C(d+k-1, k) for k=0..=order.
fn count_polynomial_terms(d: usize, order: usize) -> usize {
    let mut total = 0;
    for k in 0..=order {
        total += n_choose_k(d + k - 1, k);
    }
    total
}

/// Binomial coefficient.
fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Generate all combinations with replacement of `d` items taken `k` at a time.
fn combinations_with_replacement(d: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    fn helper(
        d: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..d {
            current.push(i);
            helper(d, k, i, current, result);
            current.pop();
        }
    }
    let mut current = Vec::new();
    helper(d, k, 0, &mut current, &mut result);
    result
}

/// Solve symmetric positive definite system via Cholesky-like approach.
fn solve_spd<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Option<Array1<F>> {
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
                aug[[row, j]] = aug[[row, j]] - factor * aug[[col, j]];
            }
        }
    }

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

/// Select polynomial order via cross-validation.
fn select_order_cv<F, K>(
    x: &Array2<F>,
    y: &Array1<F>,
    kernel: &K,
    bw_strategy: &BandwidthStrategy<F>,
    max_order: usize,
) -> Result<usize, FerroError>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F> + Clone,
{
    let mut best_order = 0;
    let mut best_error = F::infinity();

    for order in 0..=max_order {
        let bw = bandwidth::resolve_bandwidth(bw_strategy, x, y, kernel, order)?;

        // Simple LOOCV
        let n = x.nrows();
        let mut sse = F::zero();
        for i in 0..n {
            let mut x_train_rows = Vec::with_capacity(n - 1);
            let mut y_train_vec = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j != i {
                    x_train_rows.push(x.row(j).to_owned());
                    y_train_vec.push(y[j]);
                }
            }
            let x_train = ndarray::stack(
                ndarray::Axis(0),
                &x_train_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
            )
            .unwrap();
            let y_train = Array1::from_vec(y_train_vec);

            // Build model manually for this fold
            let model = LocalPolynomialRegression {
                kernel: kernel.clone(),
                bandwidth: BandwidthStrategy::PerDimension(bw.clone()),
                order: OrderStrategy::Fixed(order),
                regularization: F::from(1e-10).unwrap(),
            };
            let fitted = model.fit(&x_train, &y_train)?;
            let x_test = x.row(i).to_owned().insert_axis(ndarray::Axis(0));
            let pred = fitted.predict(&x_test)?;
            let err = y[i] - pred[0];
            sse = sse + err * err;
        }

        let mse = sse / F::from(n).unwrap();
        if mse < best_error {
            best_error = mse;
            best_order = order;
        }
    }

    Ok(best_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::GaussianKernel;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn design_matrix_order0() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let q = array![2.0f64];
        let d = build_design_matrix(&x, &q.view(), 0);
        assert_eq!(d.ncols(), 1);
        assert_eq!(d.nrows(), 3);
        for i in 0..3 {
            assert_abs_diff_eq!(d[[i, 0]], 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn design_matrix_order1_1d() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let q = array![2.0f64];
        let d = build_design_matrix(&x, &q.view(), 1);
        // Columns: [1, (x-2)]
        assert_eq!(d.ncols(), 2);
        assert_abs_diff_eq!(d[[0, 1]], -1.0, epsilon = 1e-15); // 1-2
        assert_abs_diff_eq!(d[[1, 1]], 0.0, epsilon = 1e-15); // 2-2
        assert_abs_diff_eq!(d[[2, 1]], 1.0, epsilon = 1e-15); // 3-2
    }

    #[test]
    fn design_matrix_order2_2d() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = array![0.0f64, 0.0];
        let d = build_design_matrix(&x, &q.view(), 2);
        // Order 0: [1]
        // Order 1: [x0, x1]
        // Order 2: [x0^2, x0*x1, x1^2]
        assert_eq!(d.ncols(), 6);
    }

    #[test]
    fn polynomial_terms_count() {
        assert_eq!(count_polynomial_terms(1, 0), 1);
        assert_eq!(count_polynomial_terms(1, 1), 2);
        assert_eq!(count_polynomial_terms(1, 2), 3);
        assert_eq!(count_polynomial_terms(2, 1), 3);
        assert_eq!(count_polynomial_terms(2, 2), 6);
    }

    #[test]
    fn fit_predict_linear_data() {
        // y = 2x + 1, local linear should fit perfectly
        let n = 50;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi + 1.0);

        let model = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(1.0),
            1,
        );
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        for i in 0..n {
            assert_abs_diff_eq!(pred[i], y[i], epsilon = 0.1);
        }
    }

    #[test]
    fn boundary_bias_reduction() {
        // y = x on [0, 1]: NW has boundary bias, LPR order 1 should not
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).to_owned();

        let model = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(0.15),
            1,
        );
        let fitted = model.fit(&x, &y).unwrap();

        // Predict at boundary x=1.0
        let x_test = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let pred = fitted.predict(&x_test).unwrap();

        // Local linear should have small bias at boundary
        assert!(
            (pred[0] - 1.0).abs() < 0.05,
            "LPR boundary prediction {:.4} should be close to 1.0",
            pred[0]
        );
    }

    #[test]
    fn order0_matches_nw() {
        // LPR order 0 should match Nadaraya-Watson
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);

        let nw = crate::NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Fixed(0.5));
        let nw_fitted = nw.fit(&x, &y).unwrap();
        let nw_pred = nw_fitted.predict(&x).unwrap();

        let lpr = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(0.5),
            0,
        )
        .regularization(0.0);
        let lpr_fitted = lpr.fit(&x, &y).unwrap();
        let lpr_pred = lpr_fitted.predict(&x).unwrap();

        for i in 0..n {
            assert_abs_diff_eq!(nw_pred[i], lpr_pred[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn quadratic_data_order2() {
        // y = x^2, order 2 should fit well in interior
        let n = 50;
        let x_data: Vec<f64> = (0..n)
            .map(|i| -2.0 + 4.0 * i as f64 / (n - 1) as f64)
            .collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| xi * xi);

        let model = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(0.5),
            2,
        );
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Interior points should be well-fitted
        let interior_start = 10;
        let interior_end = 40;
        let mut max_err = 0.0f64;
        for i in interior_start..interior_end {
            let err = (pred[i] - y[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 0.5, "Max interior error: {max_err}");
    }

    #[test]
    fn collinear_no_panic() {
        // Perfectly collinear features: X2 = X1
        let n = 30;
        let x_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let v = i as f64 * 0.1;
                vec![v, v]
            })
            .collect();
        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi + 1.0);

        let model = LocalPolynomialRegression::with_kernel(
            GaussianKernel,
            BandwidthStrategy::Fixed(0.5),
            1,
        );
        let fitted = model.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Should produce finite predictions
        for &p in &pred {
            assert!(p.is_finite(), "Collinear prediction should be finite");
        }
    }
}
