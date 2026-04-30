//! Sparse inverse-covariance estimation via the graphical lasso.
//!
//! [`GraphicalLasso`] estimates a sparse precision matrix by maximising the
//! L1-penalised Gaussian log-likelihood
//!
//! ```text
//! max_{P ≻ 0}   log|P| - tr(S P) - alpha * ||P||_1
//! ```
//!
//! where `S` is the empirical covariance and `alpha` is the L1 penalty.
//! [`GraphicalLassoCV`] picks `alpha` by k-fold cross-validation over a grid.
//!
//! The implementation follows the coordinate-descent (Friedman, Hastie &
//! Tibshirani, 2008) outer iteration: at each pass over the columns, solve a
//! lasso regression of column `i` against the other columns, then update the
//! covariance and precision blocks accordingly.

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::helpers::empirical_covariance;

/// Sparse inverse-covariance estimator (graphical lasso).
#[derive(Debug, Clone)]
pub struct GraphicalLasso<F> {
    /// L1 regularisation strength.
    alpha: F,
    /// Outer-loop iteration cap.
    max_iter: usize,
    /// Coordinate-descent inner iteration cap.
    max_inner_iter: usize,
    /// Convergence tolerance on the change in covariance between outer
    /// iterations (Frobenius norm).
    tol: F,
    /// If `true`, skip mean centering during the empirical covariance step.
    assume_centered: bool,
}

impl<F: Float + Send + Sync + 'static> GraphicalLasso<F> {
    /// Construct a new [`GraphicalLasso`] with the given L1 penalty.
    #[must_use]
    pub fn new(alpha: F) -> Self {
        Self {
            alpha,
            max_iter: 100,
            max_inner_iter: 100,
            tol: F::from(1e-4).unwrap_or(F::epsilon()),
            assume_centered: false,
        }
    }

    /// Set the maximum number of outer iterations (default `100`).
    #[must_use]
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the maximum number of inner coordinate-descent iterations
    /// per column (default `100`).
    #[must_use]
    pub fn max_inner_iter(mut self, n: usize) -> Self {
        self.max_inner_iter = n;
        self
    }

    /// Set the outer convergence tolerance (default `1e-4`).
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// If `true`, skip mean centering during the empirical covariance step.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

/// A fitted [`GraphicalLasso`] model.
#[derive(Debug, Clone)]
pub struct FittedGraphicalLasso<F> {
    covariance: Array2<F>,
    precision: Array2<F>,
    location: Array1<F>,
    n_iter: usize,
}

impl<F: Float + Send + Sync + 'static> FittedGraphicalLasso<F> {
    /// The estimated covariance matrix.
    pub fn covariance(&self) -> &Array2<F> {
        &self.covariance
    }
    /// The estimated precision matrix (inverse covariance).
    pub fn precision(&self) -> &Array2<F> {
        &self.precision
    }
    /// The per-feature mean used during centering.
    pub fn location(&self) -> &Array1<F> {
        &self.location
    }
    /// Number of outer iterations run.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GraphicalLasso<F> {
    type Fitted = FittedGraphicalLasso<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGraphicalLasso<F>, FerroError> {
        let n = x.nrows();
        let p = x.ncols();
        if n < 2 || p < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n.min(p),
                context: "GraphicalLasso requires n >= 2 and p >= 2".into(),
            });
        }

        // Per-column means (used both for location and centering when needed).
        let n_f = F::from(n).ok_or_else(|| FerroError::InvalidParameter {
            name: "n".into(),
            reason: "could not convert".into(),
        })?;
        let mut mean = Array1::<F>::zeros(p);
        if !self.assume_centered {
            for j in 0..p {
                let s = x.column(j).iter().copied().fold(F::zero(), |a, b| a + b);
                mean[j] = s / n_f;
            }
        }

        let emp_cov = empirical_covariance(x, self.assume_centered)?;
        let (cov, prec, n_iter) = solve_glasso(
            &emp_cov,
            self.alpha,
            self.max_iter,
            self.max_inner_iter,
            self.tol,
        );

        Ok(FittedGraphicalLasso {
            covariance: cov,
            precision: prec,
            location: mean,
            n_iter,
        })
    }
}

/// Cross-validated [`GraphicalLasso`].
#[derive(Debug, Clone)]
pub struct GraphicalLassoCV<F> {
    alphas: Vec<F>,
    n_folds: usize,
    max_iter: usize,
    tol: F,
    assume_centered: bool,
}

impl<F: Float + Send + Sync + 'static> GraphicalLassoCV<F> {
    /// Construct a new cross-validated graphical lasso over the given alpha
    /// grid.
    #[must_use]
    pub fn new(alphas: Vec<F>) -> Self {
        Self {
            alphas,
            n_folds: 3,
            max_iter: 100,
            tol: F::from(1e-4).unwrap_or(F::epsilon()),
            assume_centered: false,
        }
    }

    /// Set the number of CV folds (default `3`).
    #[must_use]
    pub fn n_folds(mut self, n: usize) -> Self {
        self.n_folds = n;
        self
    }

    /// Set the per-fit max iterations (default `100`).
    #[must_use]
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the per-fit tolerance (default `1e-4`).
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// If `true`, skip mean centering during the empirical covariance step.
    #[must_use]
    pub fn assume_centered(mut self, value: bool) -> Self {
        self.assume_centered = value;
        self
    }
}

/// A fitted [`GraphicalLassoCV`] model.
#[derive(Debug, Clone)]
pub struct FittedGraphicalLassoCV<F> {
    inner: FittedGraphicalLasso<F>,
    best_alpha: F,
    cv_scores: Vec<F>,
}

impl<F: Float + Send + Sync + 'static> FittedGraphicalLassoCV<F> {
    /// The chosen alpha that maximised the CV log-likelihood.
    pub fn best_alpha(&self) -> F {
        self.best_alpha
    }
    /// The per-fold per-alpha mean log-likelihoods (one entry per `alpha`).
    pub fn cv_scores(&self) -> &[F] {
        &self.cv_scores
    }
    /// The estimated covariance matrix at the chosen alpha.
    pub fn covariance(&self) -> &Array2<F> {
        self.inner.covariance()
    }
    /// The estimated precision matrix at the chosen alpha.
    pub fn precision(&self) -> &Array2<F> {
        self.inner.precision()
    }
    /// The per-feature mean used during centering.
    pub fn location(&self) -> &Array1<F> {
        self.inner.location()
    }
    /// Number of outer iterations run on the chosen alpha refit.
    pub fn n_iter(&self) -> usize {
        self.inner.n_iter()
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for GraphicalLassoCV<F> {
    type Fitted = FittedGraphicalLassoCV<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedGraphicalLassoCV<F>, FerroError> {
        if self.alphas.is_empty() {
            return Err(FerroError::InvalidParameter {
                name: "alphas".into(),
                reason: "GraphicalLassoCV: alpha grid must be non-empty".into(),
            });
        }
        let n = x.nrows();
        if self.n_folds < 2 || n < self.n_folds {
            return Err(FerroError::InvalidParameter {
                name: "n_folds".into(),
                reason: format!(
                    "GraphicalLassoCV: need n_folds in [2, n_samples]; got {} folds and {n} samples",
                    self.n_folds
                ),
            });
        }
        let p = x.ncols();
        let fold_size = n / self.n_folds;

        let mut scores: Vec<F> = Vec::with_capacity(self.alphas.len());
        for &alpha in &self.alphas {
            let mut acc = F::zero();
            let mut count = 0usize;
            for fold in 0..self.n_folds {
                let lo = fold * fold_size;
                let hi = if fold + 1 == self.n_folds {
                    n
                } else {
                    lo + fold_size
                };

                let mut train = Vec::with_capacity((n - (hi - lo)) * p);
                let mut test = Vec::with_capacity((hi - lo) * p);
                for i in 0..n {
                    let row = x.row(i);
                    if (lo..hi).contains(&i) {
                        for v in row.iter() {
                            test.push(*v);
                        }
                    } else {
                        for v in row.iter() {
                            train.push(*v);
                        }
                    }
                }
                let train_arr = Array2::from_shape_vec((n - (hi - lo), p), train).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "fold".into(),
                        reason: format!("could not reshape train fold: {e}"),
                    }
                })?;
                let test_arr = Array2::from_shape_vec((hi - lo, p), test).map_err(|e| {
                    FerroError::InvalidParameter {
                        name: "fold".into(),
                        reason: format!("could not reshape test fold: {e}"),
                    }
                })?;
                let model = GraphicalLasso::<F>::new(alpha)
                    .max_iter(self.max_iter)
                    .tol(self.tol)
                    .assume_centered(self.assume_centered);
                let fitted = model.fit(&train_arr, &())?;
                let test_emp = empirical_covariance(&test_arr, self.assume_centered)?;
                let ll = crate::helpers::log_likelihood(&test_emp, fitted.precision())?;
                acc = acc + ll;
                count += 1;
            }
            let mean = acc
                / F::from(count).ok_or_else(|| FerroError::InvalidParameter {
                    name: "n_folds".into(),
                    reason: "could not convert".into(),
                })?;
            scores.push(mean);
        }

        // Pick the alpha that maximises CV log-likelihood.
        let mut best_idx = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
        let best_alpha = self.alphas[best_idx];

        let model = GraphicalLasso::<F>::new(best_alpha)
            .max_iter(self.max_iter)
            .tol(self.tol)
            .assume_centered(self.assume_centered);
        let inner = model.fit(x, &())?;

        Ok(FittedGraphicalLassoCV {
            inner,
            best_alpha,
            cv_scores: scores,
        })
    }
}

/// Function-style equivalent of [`GraphicalLasso::fit`] returning
/// `(covariance, precision)`.
pub fn graphical_lasso<F>(
    emp_cov: &Array2<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
) -> Result<(Array2<F>, Array2<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = emp_cov.nrows();
    if n != emp_cov.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![emp_cov.nrows(), emp_cov.ncols()],
            context: "graphical_lasso: emp_cov must be square".into(),
        });
    }
    let (cov, prec, _n_iter) = solve_glasso(emp_cov, alpha, max_iter, 100, tol);
    Ok((cov, prec))
}

// ---------------------------------------------------------------------------
// Coordinate-descent solver (Friedman et al., 2008)
// ---------------------------------------------------------------------------

fn solve_glasso<F: Float>(
    emp_cov: &Array2<F>,
    alpha: F,
    max_iter: usize,
    max_inner_iter: usize,
    tol: F,
) -> (Array2<F>, Array2<F>, usize) {
    let p = emp_cov.nrows();
    // Initialise W = S + alpha I (sklearn convention)
    let mut w = emp_cov.clone();
    for i in 0..p {
        w[[i, i]] = w[[i, i]] + alpha;
    }
    // Inverse-style precision matrix bookkeeping; we recompute at the end
    // via a Cholesky-style block decomposition.
    let mut iter = 0usize;
    for it in 0..max_iter {
        iter = it + 1;
        let w_old = w.clone();
        for j in 0..p {
            // Build W_11 = W with row j and column j removed (size (p-1)x(p-1))
            // and s_12 = emp_cov column j (excluding diagonal).
            let mut w11 = Array2::<F>::zeros((p - 1, p - 1));
            let mut s12 = Array1::<F>::zeros(p - 1);
            let mut idx_a = 0usize;
            for a in 0..p {
                if a == j {
                    continue;
                }
                let mut idx_b = 0usize;
                for b in 0..p {
                    if b == j {
                        continue;
                    }
                    w11[[idx_a, idx_b]] = w[[a, b]];
                    idx_b += 1;
                }
                s12[idx_a] = emp_cov[[a, j]];
                idx_a += 1;
            }

            // Solve lasso: minimise 0.5 * beta^T W11 beta - s12^T beta + alpha ||beta||_1
            let beta = coord_descent_lasso(&w11, &s12, alpha, max_inner_iter, tol);
            // Update W column/row j: w_12 = W_11 * beta
            let mut w12 = Array1::<F>::zeros(p - 1);
            for a in 0..(p - 1) {
                let mut acc = F::zero();
                for b in 0..(p - 1) {
                    acc = acc + w11[[a, b]] * beta[b];
                }
                w12[a] = acc;
            }
            let mut idx = 0usize;
            for a in 0..p {
                if a == j {
                    continue;
                }
                w[[a, j]] = w12[idx];
                w[[j, a]] = w12[idx];
                idx += 1;
            }
        }
        // Convergence: change in W
        let mut diff = F::zero();
        for i in 0..p {
            for j in 0..p {
                let d = w[[i, j]] - w_old[[i, j]];
                diff = diff + d * d;
            }
        }
        if diff.sqrt() < tol {
            break;
        }
    }

    // Build precision matrix from final W via the standard glasso identity.
    // For each column j: theta_jj = 1 / (W_jj - w_12^T beta_j); theta_12 = -beta_j * theta_jj
    let mut prec = Array2::<F>::zeros((p, p));
    for j in 0..p {
        let mut w11 = Array2::<F>::zeros((p - 1, p - 1));
        let mut w_12 = Array1::<F>::zeros(p - 1);
        let mut idx_a = 0usize;
        for a in 0..p {
            if a == j {
                continue;
            }
            let mut idx_b = 0usize;
            for b in 0..p {
                if b == j {
                    continue;
                }
                w11[[idx_a, idx_b]] = w[[a, b]];
                idx_b += 1;
            }
            w_12[idx_a] = w[[a, j]];
            idx_a += 1;
        }
        let mut s12 = Array1::<F>::zeros(p - 1);
        let mut k = 0usize;
        for a in 0..p {
            if a == j {
                continue;
            }
            s12[k] = emp_cov[[a, j]];
            k += 1;
        }
        let beta = coord_descent_lasso(&w11, &s12, alpha, max_inner_iter, tol);
        let mut dot = F::zero();
        for a in 0..(p - 1) {
            dot = dot + w_12[a] * beta[a];
        }
        let denom = w[[j, j]] - dot;
        let theta_jj = if denom.abs() > F::epsilon() {
            F::one() / denom
        } else {
            F::one() / F::epsilon()
        };
        prec[[j, j]] = theta_jj;
        let mut idx = 0usize;
        for a in 0..p {
            if a == j {
                continue;
            }
            let v = -beta[idx] * theta_jj;
            prec[[a, j]] = v;
            prec[[j, a]] = v;
            idx += 1;
        }
    }
    (w, prec, iter)
}

/// Coordinate-descent lasso solver for the inner sub-problem
/// `min 0.5 beta^T W beta - s^T beta + alpha ||beta||_1`.
fn coord_descent_lasso<F: Float>(
    w: &Array2<F>,
    s: &Array1<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
) -> Array1<F> {
    let n = s.len();
    let mut beta = Array1::<F>::zeros(n);
    for _ in 0..max_iter {
        let mut max_change = F::zero();
        for j in 0..n {
            let mut residual = s[j];
            for k in 0..n {
                if k != j {
                    residual = residual - w[[j, k]] * beta[k];
                }
            }
            let denom = w[[j, j]];
            let new_val = if denom > F::epsilon() {
                soft_threshold(residual, alpha) / denom
            } else {
                F::zero()
            };
            let change = (new_val - beta[j]).abs();
            if change > max_change {
                max_change = change;
            }
            beta[j] = new_val;
        }
        if max_change < tol {
            break;
        }
    }
    beta
}

#[inline]
fn soft_threshold<F: Float>(x: F, gamma: F) -> F {
    if x > gamma {
        x - gamma
    } else if x < -gamma {
        x + gamma
    } else {
        F::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Array2<f64> {
        // 10 samples × 3 features with mild correlation.
        array![
            [1.0, 2.0, 1.5],
            [3.0, 4.0, 3.5],
            [5.0, 6.0, 5.5],
            [7.0, 8.0, 7.5],
            [2.0, 3.0, 2.5],
            [4.0, 5.0, 4.5],
            [6.0, 7.0, 6.5],
            [8.0, 9.0, 8.5],
            [1.5, 2.5, 2.0],
            [9.0, 10.0, 9.5],
        ]
    }

    #[test]
    fn test_graphical_lasso_basic() {
        let est = GraphicalLasso::<f64>::new(0.1).max_iter(50).tol(1e-3);
        let fitted = est.fit(&data(), &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (3, 3));
        assert_eq!(fitted.precision().dim(), (3, 3));
        assert!(fitted.n_iter() > 0);
    }

    #[test]
    fn test_graphical_lasso_function() {
        let emp = empirical_covariance(&data(), false).unwrap();
        let (cov, prec) = graphical_lasso(&emp, 0.1, 50, 1e-3).unwrap();
        assert_eq!(cov.dim(), (3, 3));
        assert_eq!(prec.dim(), (3, 3));
    }

    #[test]
    fn test_graphical_lasso_cv() {
        let est = GraphicalLassoCV::<f64>::new(vec![0.05, 0.1, 0.2])
            .n_folds(2)
            .max_iter(20)
            .tol(1e-2);
        let fitted = est.fit(&data(), &()).unwrap();
        assert_eq!(fitted.covariance().dim(), (3, 3));
        assert!(fitted.cv_scores().len() == 3);
        let alpha = fitted.best_alpha();
        assert!([0.05, 0.1, 0.2].contains(&alpha));
    }

    #[test]
    fn test_graphical_lasso_too_small() {
        let x: Array2<f64> = array![[1.0]];
        assert!(GraphicalLasso::<f64>::new(0.1).fit(&x, &()).is_err());
    }
}
