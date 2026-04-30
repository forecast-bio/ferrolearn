//! Chained multi-output estimators and the output-code multi-class
//! classifier (sklearn's `ClassifierChain`, `RegressorChain`,
//! `OutputCodeClassifier`).
//!
//! - [`ClassifierChain`] — like [`MultiOutputClassifier`](crate::MultiOutputClassifier)
//!   but each per-target classifier sees the previous targets' predictions
//!   as additional features.
//! - [`RegressorChain`] — same idea for regression.
//! - [`OutputCodeClassifier`] — reduces a `K`-class problem to a sequence of
//!   binary problems via a random ±1 code matrix.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipeline, Pipeline};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// helper: append columns to a feature matrix
// ---------------------------------------------------------------------------
fn hcat(x: &Array2<f64>, extras: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
    let n = x.nrows();
    if extras.nrows() != n {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, extras.ncols()],
            actual: vec![extras.nrows(), extras.ncols()],
            context: "chain hcat: row counts differ".into(),
        });
    }
    let total_cols = x.ncols() + extras.ncols();
    let mut out = Array2::<f64>::zeros((n, total_cols));
    for i in 0..n {
        for j in 0..x.ncols() {
            out[[i, j]] = x[[i, j]];
        }
        for j in 0..extras.ncols() {
            out[[i, x.ncols() + j]] = extras[[i, j]];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ClassifierChain
// ---------------------------------------------------------------------------

/// Multi-output classifier that fits one binary classifier per target,
/// with each classifier seeing the previous targets' predictions as
/// additional features.
pub struct ClassifierChain {
    make_pipeline: PipelineFactory,
    order: Option<Vec<usize>>,
}

impl ClassifierChain {
    /// Construct a new [`ClassifierChain`].
    #[must_use]
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self {
            make_pipeline,
            order: None,
        }
    }

    /// Set an explicit ordering for the chain. By default, targets are
    /// processed in column order `0, 1, ..., n_targets - 1`.
    #[must_use]
    pub fn order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    /// Fit the chain.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<FittedClassifierChain, FerroError> {
        fit_chain(&self.make_pipeline, &self.order, x, y, "ClassifierChain")
            .map(|(estimators, order)| FittedClassifierChain { estimators, order })
    }
}

/// A fitted [`ClassifierChain`].
pub struct FittedClassifierChain {
    estimators: Vec<FittedPipeline<f64>>,
    order: Vec<usize>,
}

impl FittedClassifierChain {
    /// Number of targets in the chain.
    pub fn n_targets(&self) -> usize {
        self.order.len()
    }
    /// Order in which targets were fitted.
    pub fn order(&self) -> &[usize] {
        &self.order
    }
}

impl Predict<Array2<f64>> for FittedClassifierChain {
    type Output = Array2<f64>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_samples = x.nrows();
        let n_targets = self.estimators.len();
        let mut chained = Array2::<f64>::zeros((n_samples, 0));
        let mut out = Array2::<f64>::zeros((n_samples, n_targets));
        for (step, est) in self.estimators.iter().enumerate() {
            let inputs = if chained.ncols() == 0 {
                x.clone()
            } else {
                hcat(x, &chained)?
            };
            let preds = est.predict(&inputs)?;
            // append column to chained
            let mut new_chained = Array2::<f64>::zeros((n_samples, chained.ncols() + 1));
            for i in 0..n_samples {
                for j in 0..chained.ncols() {
                    new_chained[[i, j]] = chained[[i, j]];
                }
                new_chained[[i, chained.ncols()]] = preds[i];
            }
            chained = new_chained;
            let target_col = self.order[step];
            for i in 0..n_samples {
                out[[i, target_col]] = preds[i];
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// RegressorChain
// ---------------------------------------------------------------------------

/// Multi-output regressor variant of [`ClassifierChain`].
pub struct RegressorChain {
    make_pipeline: PipelineFactory,
    order: Option<Vec<usize>>,
}

impl RegressorChain {
    /// Construct a new [`RegressorChain`].
    #[must_use]
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self {
            make_pipeline,
            order: None,
        }
    }

    /// Set an explicit ordering for the chain.
    #[must_use]
    pub fn order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    /// Fit the regressor chain.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<FittedRegressorChain, FerroError> {
        fit_chain(&self.make_pipeline, &self.order, x, y, "RegressorChain")
            .map(|(estimators, order)| FittedRegressorChain { estimators, order })
    }
}

/// A fitted [`RegressorChain`].
pub struct FittedRegressorChain {
    estimators: Vec<FittedPipeline<f64>>,
    order: Vec<usize>,
}

impl FittedRegressorChain {
    /// Number of targets in the chain.
    pub fn n_targets(&self) -> usize {
        self.order.len()
    }
    /// Order in which targets were fitted.
    pub fn order(&self) -> &[usize] {
        &self.order
    }
}

impl Predict<Array2<f64>> for FittedRegressorChain {
    type Output = Array2<f64>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let n_samples = x.nrows();
        let n_targets = self.estimators.len();
        let mut chained = Array2::<f64>::zeros((n_samples, 0));
        let mut out = Array2::<f64>::zeros((n_samples, n_targets));
        for (step, est) in self.estimators.iter().enumerate() {
            let inputs = if chained.ncols() == 0 {
                x.clone()
            } else {
                hcat(x, &chained)?
            };
            let preds = est.predict(&inputs)?;
            let mut new_chained = Array2::<f64>::zeros((n_samples, chained.ncols() + 1));
            for i in 0..n_samples {
                for j in 0..chained.ncols() {
                    new_chained[[i, j]] = chained[[i, j]];
                }
                new_chained[[i, chained.ncols()]] = preds[i];
            }
            chained = new_chained;
            let target_col = self.order[step];
            for i in 0..n_samples {
                out[[i, target_col]] = preds[i];
            }
        }
        Ok(out)
    }
}

// Shared chain-fitting helper.
fn fit_chain(
    factory: &PipelineFactory,
    order: &Option<Vec<usize>>,
    x: &Array2<f64>,
    y: &Array2<f64>,
    context: &str,
) -> Result<(Vec<FittedPipeline<f64>>, Vec<usize>), FerroError> {
    let n_samples = x.nrows();
    let n_targets = y.ncols();
    if y.nrows() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples, n_targets],
            actual: vec![y.nrows(), n_targets],
            context: format!("{context}: y rows must equal x rows"),
        });
    }
    if n_targets == 0 {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: format!("{context}: target matrix must have at least one column"),
        });
    }
    let chain_order: Vec<usize> = match order {
        Some(o) => {
            if o.len() != n_targets {
                return Err(FerroError::InvalidParameter {
                    name: "order".into(),
                    reason: format!(
                        "{context}: order length ({}) must equal n_targets ({n_targets})",
                        o.len()
                    ),
                });
            }
            o.clone()
        }
        None => (0..n_targets).collect(),
    };

    let mut estimators = Vec::with_capacity(n_targets);
    let mut chained = Array2::<f64>::zeros((n_samples, 0));
    for &t in &chain_order {
        let inputs = if chained.ncols() == 0 {
            x.clone()
        } else {
            hcat(x, &chained)?
        };
        let y_col = y.column(t).to_owned();
        let pipeline = factory();
        let fitted = pipeline.fit(&inputs, &y_col)?;
        let preds = fitted.predict(&inputs)?;
        // Append true (training) targets to the chain — sklearn's default is
        // to feed the *true* labels during fit (not predictions) because that
        // matches the canonical "perfect previous step" assumption.
        let mut new_chained = Array2::<f64>::zeros((n_samples, chained.ncols() + 1));
        for i in 0..n_samples {
            for j in 0..chained.ncols() {
                new_chained[[i, j]] = chained[[i, j]];
            }
            new_chained[[i, chained.ncols()]] = y[[i, t]];
        }
        chained = new_chained;
        let _ = preds; // silence unused — predict result not needed during fit
        estimators.push(fitted);
    }
    Ok((estimators, chain_order))
}

// ---------------------------------------------------------------------------
// OutputCodeClassifier
// ---------------------------------------------------------------------------

/// Reduce a `K`-class classification problem to a sequence of binary problems
/// via an error-correcting output code.
///
/// `code_size` controls how many binary classifiers are trained:
/// `n_codes = max(2, ceil(code_size * K))`. Each row of the code matrix is
/// the binary signature of one class; predictions are made by computing the
/// row in the code matrix closest (in Hamming distance) to the per-classifier
/// vote.
pub struct OutputCodeClassifier {
    make_pipeline: PipelineFactory,
    code_size: f64,
    random_state: Option<u64>,
}

impl OutputCodeClassifier {
    /// Construct a new [`OutputCodeClassifier`].
    #[must_use]
    pub fn new(make_pipeline: PipelineFactory) -> Self {
        Self {
            make_pipeline,
            code_size: 1.5,
            random_state: None,
        }
    }

    /// Set the code-matrix size relative to the number of classes
    /// (default `1.5`).
    #[must_use]
    pub fn code_size(mut self, size: f64) -> Self {
        self.code_size = size;
        self
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the output-code classifier.
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<usize>,
    ) -> Result<FittedOutputCodeClassifier, FerroError> {
        let n_samples = x.nrows();
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "OutputCodeClassifier: y length must equal x rows".into(),
            });
        }
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OutputCodeClassifier::fit".into(),
            });
        }
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        if k < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_classes".into(),
                reason: "OutputCodeClassifier needs at least 2 distinct classes".into(),
            });
        }

        // Generate code matrix: shape (k, n_codes), entries in {-1, +1}.
        let n_codes = ((self.code_size * k as f64).ceil() as usize).max(2);
        let mut rng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };
        let mut code = Array2::<f64>::zeros((k, n_codes));
        use rand::Rng;
        for i in 0..k {
            for j in 0..n_codes {
                code[[i, j]] = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
            }
        }

        // Map class -> code row.
        let class_to_idx: std::collections::HashMap<usize, usize> =
            classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Fit one binary classifier per code column. Encode +1 as label 1,
        // -1 as label 0 (sklearn convention, our base estimator outputs in
        // [0, 1]).
        let mut estimators = Vec::with_capacity(n_codes);
        for j in 0..n_codes {
            let y_col: Array1<f64> = y
                .iter()
                .map(|c| {
                    if code[[class_to_idx[c], j]] > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let pipeline = (self.make_pipeline)();
            let fitted = pipeline.fit(x, &y_col)?;
            estimators.push(fitted);
        }

        Ok(FittedOutputCodeClassifier {
            estimators,
            code,
            classes,
        })
    }
}

/// A fitted [`OutputCodeClassifier`].
pub struct FittedOutputCodeClassifier {
    estimators: Vec<FittedPipeline<f64>>,
    code: Array2<f64>,
    classes: Vec<usize>,
}

impl FittedOutputCodeClassifier {
    /// Sorted class labels seen during training.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }
    /// Number of binary base classifiers fitted.
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }
}

impl Predict<Array2<f64>> for FittedOutputCodeClassifier {
    type Output = Array1<usize>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_codes = self.estimators.len();
        let mut votes = Array2::<f64>::zeros((n_samples, n_codes));
        for (j, est) in self.estimators.iter().enumerate() {
            let preds = est.predict(x)?;
            for i in 0..n_samples {
                votes[[i, j]] = if preds[i] > 0.5 { 1.0 } else { -1.0 };
            }
        }
        // For each sample, pick the class whose code row minimises Euclidean
        // (= Hamming-on-bits) distance to the vote vector.
        let mut out = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..self.classes.len() {
                let mut d = 0.0_f64;
                for j in 0..n_codes {
                    let diff = votes[[i, j]] - self.code[[c, j]];
                    d += diff * diff;
                }
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            out[i] = self.classes[best];
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};

    /// Trivial estimator that just returns the mean of the training y column
    /// for every prediction. Plenty for shape-only tests.
    struct MeanEstimator;
    struct FittedMeanEstimator(f64);
    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            let mean = y.iter().copied().sum::<f64>() / y.len() as f64;
            Ok(Box::new(FittedMeanEstimator(mean)))
        }
    }
    impl FittedPipelineEstimator<f64> for FittedMeanEstimator {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.0))
        }
    }

    fn factory() -> PipelineFactory {
        Box::new(|| Pipeline::<f64>::new().estimator_step("mean", Box::new(MeanEstimator)))
    }

    #[test]
    fn classifier_chain_fits_and_predicts_shape() {
        let x = Array2::<f64>::zeros((6, 2));
        let y = ndarray::array![
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ];
        let cc = ClassifierChain::new(factory());
        let fitted = cc.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_targets(), 2);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.dim(), (6, 2));
    }

    #[test]
    fn regressor_chain_fits_and_predicts_shape() {
        let x = Array2::<f64>::zeros((5, 3));
        let y = ndarray::array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5]];
        let rc = RegressorChain::new(factory());
        let fitted = rc.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.dim(), (5, 2));
    }

    #[test]
    fn classifier_chain_explicit_order() {
        let x = Array2::<f64>::zeros((4, 2));
        let y = ndarray::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let cc = ClassifierChain::new(factory()).order(vec![1, 0]);
        let fitted = cc.fit(&x, &y).unwrap();
        assert_eq!(fitted.order(), &[1, 0]);
    }

    #[test]
    fn output_code_basic_shapes() {
        let x = Array2::<f64>::zeros((9, 2));
        let y = Array1::from(vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2]);
        let occ = OutputCodeClassifier::new(factory())
            .code_size(2.0)
            .random_state(7);
        let fitted = occ.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1, 2]);
        let n = fitted.n_estimators();
        assert!(n >= 2);
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 9);
        for &p in preds.iter() {
            assert!([0, 1, 2].contains(&p));
        }
    }
}
