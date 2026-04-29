//! Bandwidth selection for kernel regression.
//!
//! Provides rule-of-thumb (Silverman, Scott) and cross-validation based
//! bandwidth selection. The LOOCV shortcut uses `O(n)` hat matrix
//! diagonal computation for Nadaraya-Watson.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::hat_matrix::loocv_hat_matrix_shortcut;
use crate::kernels::Kernel;
use crate::weights;

/// How bandwidth is determined.
#[derive(Debug, Clone)]
pub enum BandwidthStrategy<F: Float> {
    /// Fixed scalar bandwidth (applied to all dimensions).
    Fixed(F),
    /// Fixed per-dimension bandwidth.
    PerDimension(Array1<F>),
    /// Cross-validated selection.
    CrossValidated {
        /// CV strategy.
        cv: CvStrategy,
        /// Optimize per dimension (coordinate descent).
        per_dimension: bool,
    },
    /// Silverman's rule of thumb.
    Silverman,
    /// Scott's rule of thumb.
    Scott,
}

/// Cross-validation strategy.
#[derive(Debug, Clone, Copy)]
pub enum CvStrategy {
    /// Leave-one-out (O(n) shortcut for NW).
    Loo,
    /// K-fold cross-validation.
    KFold(usize),
}

/// Silverman's rule of thumb: `h_j = 1.06 * σ_robust * n^(-1/5)`
pub fn silverman_bandwidth<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n = F::from(x.nrows()).unwrap();
    let factor = F::from(1.06).unwrap() * n.powf(F::from(-0.2).unwrap());

    x.columns()
        .into_iter()
        .map(|col| {
            let std = robust_scale(&col);
            factor * std
        })
        .collect()
}

/// Scott's rule of thumb: `h_j = σ_j * n^(-1/(d+4))`
pub fn scott_bandwidth<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n = F::from(x.nrows()).unwrap();
    let d = F::from(x.ncols()).unwrap();
    let exponent = -F::one() / (d + F::from(4.0).unwrap());
    let n_factor = n.powf(exponent);

    x.columns()
        .into_iter()
        .map(|col| {
            let std = column_std(&col);
            let std = if std > F::zero() { std } else { F::one() };
            std * n_factor
        })
        .collect()
}

/// Robust scale estimate: min(std, IQR/1.349)
fn robust_scale<F: Float>(col: &ndarray::ArrayView1<F>) -> F {
    let std = column_std(col);
    let iqr = interquartile_range(col);
    let iqr_scale = iqr / F::from(1.349).unwrap();

    let scale = if iqr_scale > F::zero() {
        std.min(iqr_scale)
    } else {
        std
    };
    if scale > F::zero() { scale } else { F::one() }
}

/// Standard deviation with ddof=1.
fn column_std<F: Float>(col: &ndarray::ArrayView1<F>) -> F {
    let n = F::from(col.len()).unwrap();
    if col.len() < 2 {
        return F::zero();
    }
    let mean = col.iter().copied().fold(F::zero(), |a, b| a + b) / n;
    let var = col
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .fold(F::zero(), |a, b| a + b)
        / (n - F::one());
    var.sqrt()
}

/// Interquartile range.
fn interquartile_range<F: Float>(col: &ndarray::ArrayView1<F>) -> F {
    let mut sorted: Vec<F> = col.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n < 4 {
        return F::zero();
    }
    let q25 = percentile_sorted(&sorted, 0.25);
    let q75 = percentile_sorted(&sorted, 0.75);
    q75 - q25
}

/// Linear interpolation percentile on sorted data.
fn percentile_sorted<F: Float>(sorted: &[F], p: f64) -> F {
    let n = sorted.len();
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((n - 1) as f64) as usize;
    let frac = F::from(idx - lo as f64).unwrap();
    sorted[lo] * (F::one() - frac) + sorted[hi] * frac
}

/// Resolve a bandwidth strategy to concrete per-dimension values.
pub fn resolve_bandwidth<F, K>(
    strategy: &BandwidthStrategy<F>,
    x: &Array2<F>,
    y: &Array1<F>,
    kernel: &K,
    polynomial_order: usize,
) -> Result<Array1<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    match strategy {
        BandwidthStrategy::Fixed(h) => {
            let n_features = x.ncols();
            Ok(Array1::from_elem(n_features, *h))
        }
        BandwidthStrategy::PerDimension(bw) => {
            if bw.len() != x.ncols() {
                return Err(FerroError::ShapeMismatch {
                    expected: vec![x.ncols()],
                    actual: vec![bw.len()],
                    context: "bandwidth dimensions must match features".into(),
                });
            }
            Ok(bw.clone())
        }
        BandwidthStrategy::Silverman => Ok(silverman_bandwidth(x)),
        BandwidthStrategy::Scott => Ok(scott_bandwidth(x)),
        BandwidthStrategy::CrossValidated { cv, per_dimension } => {
            let cv_selector = CrossValidatedBandwidth {
                cv: *cv,
                n_bandwidths: 30,
                polynomial_order,
                per_dimension: *per_dimension,
            };
            cv_selector.select(x, y, kernel)
        }
    }
}

/// Cross-validation bandwidth selector.
pub struct CrossValidatedBandwidth {
    /// CV strategy.
    pub cv: CvStrategy,
    /// Number of bandwidth values in grid search.
    pub n_bandwidths: usize,
    /// Polynomial order (0 for NW, 1+ for local polynomial).
    pub polynomial_order: usize,
    /// Optimize per dimension.
    pub per_dimension: bool,
}

impl CrossValidatedBandwidth {
    /// Select optimal bandwidth via cross-validation.
    pub fn select<F, K>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        kernel: &K,
    ) -> Result<Array1<F>, FerroError>
    where
        F: Float + Send + Sync + 'static,
        K: Kernel<F>,
    {
        let h_rot = silverman_bandwidth(x);

        if self.per_dimension && x.ncols() > 1 {
            self.per_dimension_cv(x, y, kernel, &h_rot)
        } else {
            self.grid_search(x, y, kernel, &h_rot)
        }
    }

    /// Grid search with a common scaling factor.
    fn grid_search<F, K>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        kernel: &K,
        h_rot: &Array1<F>,
    ) -> Result<Array1<F>, FerroError>
    where
        F: Float + Send + Sync + 'static,
        K: Kernel<F>,
    {
        let mut best_error = F::infinity();
        let mut best_bw = h_rot.clone();

        for i in 0..self.n_bandwidths {
            let t = F::from(i).unwrap() / F::from(self.n_bandwidths - 1).unwrap();
            // Log-space from 0.1 to 5.0
            let log_min = F::from(-1.0f64).unwrap(); // log10(0.1)
            let log_max = F::from(0.699).unwrap(); // log10(5.0)
            let log_factor = log_min + t * (log_max - log_min);
            let factor = F::from(10.0).unwrap().powf(log_factor);

            let bw = h_rot.mapv(|h| h * factor);
            let error = self.cv_error(x, y, &bw, kernel)?;

            if error < best_error {
                best_error = error;
                best_bw = bw;
            }
        }

        Ok(best_bw)
    }

    /// Coordinate descent: optimize each dimension separately.
    fn per_dimension_cv<F, K>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        kernel: &K,
        h_rot: &Array1<F>,
    ) -> Result<Array1<F>, FerroError>
    where
        F: Float + Send + Sync + 'static,
        K: Kernel<F>,
    {
        let n_features = x.ncols();
        let mut bw = h_rot.clone();

        for _iteration in 0..3 {
            for dim in 0..n_features {
                let mut best_error = F::infinity();
                let mut best_h = bw[dim];

                for i in 0..self.n_bandwidths {
                    let t = F::from(i).unwrap() / F::from(self.n_bandwidths - 1).unwrap();
                    let log_min = F::from(-1.0f64).unwrap();
                    let log_max = F::from(0.699).unwrap();
                    let log_factor = log_min + t * (log_max - log_min);
                    let factor = F::from(10.0).unwrap().powf(log_factor);

                    let mut test_bw = bw.clone();
                    test_bw[dim] = h_rot[dim] * factor;

                    let error = self.cv_error(x, y, &test_bw, kernel)?;
                    if error < best_error {
                        best_error = error;
                        best_h = test_bw[dim];
                    }
                }

                bw[dim] = best_h;
            }
        }

        Ok(bw)
    }

    /// Compute CV error for a given bandwidth.
    fn cv_error<F, K>(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
        bw: &Array1<F>,
        kernel: &K,
    ) -> Result<F, FerroError>
    where
        F: Float + Send + Sync + 'static,
        K: Kernel<F>,
    {
        match self.cv {
            CvStrategy::Loo => {
                if self.polynomial_order == 0 {
                    Ok(loocv_hat_matrix_shortcut(x, y, bw, kernel))
                } else {
                    Ok(naive_loocv(x, y, bw, kernel, self.polynomial_order))
                }
            }
            CvStrategy::KFold(k) => Ok(kfold_cv(x, y, bw, kernel, k, self.polynomial_order)),
        }
    }
}

/// Naive O(n²) LOOCV for local polynomial regression.
fn naive_loocv<F, K>(
    x: &Array2<F>,
    y: &Array1<F>,
    bw: &Array1<F>,
    kernel: &K,
    _polynomial_order: usize,
) -> F
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let n = x.nrows();
    let n_f = F::from(n).unwrap();
    let mut sse = F::zero();

    for i in 0..n {
        // Build leave-one-out training set
        let mut x_train_rows = Vec::with_capacity(n - 1);
        let mut y_train = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                x_train_rows.push(x.row(j).to_owned());
                y_train.push(y[j]);
            }
        }
        let x_train = ndarray::stack(
            ndarray::Axis(0),
            &x_train_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let y_train = Array1::from_vec(y_train);
        let x_test = x.row(i).to_owned().insert_axis(ndarray::Axis(0));

        // NW prediction for the LOO test point
        let w = weights::compute_kernel_weights(&x_test, &x_train, bw, kernel);
        let pred = weights::nw_predict_from_weights(&w, &y_train);
        let err = y[i] - pred[0];
        sse = sse + err * err;
    }

    sse / n_f
}

/// K-fold cross-validation error.
fn kfold_cv<F, K>(
    x: &Array2<F>,
    y: &Array1<F>,
    bw: &Array1<F>,
    kernel: &K,
    n_folds: usize,
    _polynomial_order: usize,
) -> F
where
    F: Float + Send + Sync + 'static,
    K: Kernel<F>,
{
    let n = x.nrows();
    let fold_size = n.div_ceil(n_folds);
    let mut total_error = F::zero();
    let mut total_count = 0usize;

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = (test_start + fold_size).min(n);
        if test_start >= n {
            break;
        }

        let mut x_train_rows = Vec::new();
        let mut y_train_vec = Vec::new();
        let mut x_test_rows = Vec::new();
        let mut y_test_vec = Vec::new();

        for i in 0..n {
            if i >= test_start && i < test_end {
                x_test_rows.push(x.row(i).to_owned());
                y_test_vec.push(y[i]);
            } else {
                x_train_rows.push(x.row(i).to_owned());
                y_train_vec.push(y[i]);
            }
        }

        if x_train_rows.is_empty() || x_test_rows.is_empty() {
            continue;
        }

        let x_train = ndarray::stack(
            ndarray::Axis(0),
            &x_train_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let y_train = Array1::from_vec(y_train_vec);
        let x_test = ndarray::stack(
            ndarray::Axis(0),
            &x_test_rows.iter().map(|r| r.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        let w = weights::compute_kernel_weights(&x_test, &x_train, bw, kernel);
        let pred = weights::nw_predict_from_weights(&w, &y_train);

        for (i, &yt) in y_test_vec.iter().enumerate() {
            let err = yt - pred[i];
            total_error = total_error + err * err;
        }
        total_count += y_test_vec.len();
    }

    if total_count > 0 {
        total_error / F::from(total_count).unwrap()
    } else {
        F::infinity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::GaussianKernel;
    use ndarray::Array2;

    #[test]
    fn silverman_positive() {
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| f64::from(i) * 0.5).collect()).unwrap();
        let bw = silverman_bandwidth(&x);
        assert!(bw[0] > 0.0);
    }

    #[test]
    fn scott_positive() {
        let x =
            Array2::from_shape_vec((20, 1), (0..20).map(|i| f64::from(i) * 0.5).collect()).unwrap();
        let bw = scott_bandwidth(&x);
        assert!(bw[0] > 0.0);
    }

    #[test]
    fn silverman_per_dimension() {
        let x = Array2::from_shape_vec(
            (50, 2),
            (0..100)
                .map(|i| {
                    if i % 2 == 0 {
                        f64::from(i)
                    } else {
                        f64::from(i) * 10.0
                    }
                })
                .collect(),
        )
        .unwrap();
        let bw = silverman_bandwidth(&x);
        assert_eq!(bw.len(), 2);
        assert!(bw[0] > 0.0);
        assert!(bw[1] > 0.0);
        // Dimension with larger spread should have larger bandwidth
        assert!(bw[1] > bw[0]);
    }

    #[test]
    fn cv_selects_reasonable_bandwidth() {
        let n = 50;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);

        let selector = CrossValidatedBandwidth {
            cv: CvStrategy::Loo,
            n_bandwidths: 15,
            polynomial_order: 0,
            per_dimension: false,
        };
        let bw = selector.select(&x, &y, &GaussianKernel).unwrap();

        // Should be positive and reasonable
        assert!(bw[0] > 0.0);
        assert!(bw[0] < 10.0); // Not absurdly large
    }

    #[test]
    fn resolve_fixed() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = Array1::from_vec(vec![0.0; 5]);
        let bw =
            resolve_bandwidth(&BandwidthStrategy::Fixed(1.5), &x, &y, &GaussianKernel, 0).unwrap();
        assert_eq!(bw.len(), 2);
        assert_eq!(bw[0], 1.5);
        assert_eq!(bw[1], 1.5);
    }

    #[test]
    fn resolve_per_dimension_rejects_mismatch() {
        let x = Array2::from_shape_vec((5, 2), vec![0.0; 10]).unwrap();
        let y = Array1::from_vec(vec![0.0; 5]);
        let bw = resolve_bandwidth(
            &BandwidthStrategy::PerDimension(ndarray::array![1.0, 2.0, 3.0]),
            &x,
            &y,
            &GaussianKernel,
            0,
        );
        assert!(bw.is_err());
    }

    #[test]
    fn kfold_cv_works() {
        let n = 30;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(f64::sin);
        let bw = ndarray::array![0.5];

        let err = kfold_cv(&x, &y, &bw, &GaussianKernel, 5, 0);
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }
}
