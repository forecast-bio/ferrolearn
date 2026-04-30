//! Dummy (baseline) estimators that ignore the input features.
//!
//! These are used as sanity-check baselines: a serious model must beat the
//! dummy, otherwise its predictive power is no better than the chosen
//! constant or marginal-distribution rule.
//!
//! - [`DummyClassifier`] supports the strategies most_frequent, prior,
//!   stratified, uniform, and constant.
//! - [`DummyRegressor`] supports mean, median, quantile, and constant.

use std::collections::HashMap;

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// DummyClassifier
// ---------------------------------------------------------------------------

/// Strategy used by [`DummyClassifier`] to choose predictions.
#[derive(Debug, Clone, PartialEq)]
pub enum DummyClassifierStrategy {
    /// Always predict the most frequent training label.
    MostFrequent,
    /// Same as [`Self::MostFrequent`]; included to mirror sklearn naming.
    Prior,
    /// Sample predictions from the empirical class prior.
    Stratified,
    /// Sample predictions uniformly at random from the observed classes.
    Uniform,
    /// Always predict a fixed user-supplied constant.
    Constant(usize),
}

/// Baseline classifier that ignores its input features.
#[derive(Debug, Clone)]
pub struct DummyClassifier {
    strategy: DummyClassifierStrategy,
    random_state: Option<u64>,
}

impl DummyClassifier {
    /// Construct a new [`DummyClassifier`] with the given strategy.
    #[must_use]
    pub fn new(strategy: DummyClassifierStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set the RNG seed used by stochastic strategies.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Default for DummyClassifier {
    fn default() -> Self {
        Self::new(DummyClassifierStrategy::Prior)
    }
}

/// Fitted [`DummyClassifier`].
#[derive(Debug, Clone)]
pub struct FittedDummyClassifier {
    strategy: DummyClassifierStrategy,
    classes: Vec<usize>,
    class_priors: Vec<f64>,
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for DummyClassifier {
    type Fitted = FittedDummyClassifier;
    type Error = FerroError;

    fn fit(&self, _x: &Array2<F>, y: &Array1<usize>) -> Result<Self::Fitted, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DummyClassifier::fit".into(),
            });
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &c in y.iter() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut classes: Vec<usize> = counts.keys().copied().collect();
        classes.sort_unstable();
        let n = y.len() as f64;
        let class_priors: Vec<f64> = classes.iter().map(|c| counts[c] as f64 / n).collect();

        if let DummyClassifierStrategy::Constant(c) = self.strategy
            && !classes.contains(&c)
        {
            return Err(FerroError::InvalidParameter {
                name: "constant".into(),
                reason: format!("DummyClassifier: constant {c} not present in training labels"),
            });
        }

        Ok(FittedDummyClassifier {
            strategy: self.strategy.clone(),
            classes,
            class_priors,
            random_state: self.random_state,
        })
    }
}

impl FittedDummyClassifier {
    /// Most-frequent class as observed during training.
    #[must_use]
    pub fn most_frequent(&self) -> usize {
        let mut best = self.classes[0];
        let mut best_p = self.class_priors[0];
        for (i, &p) in self.class_priors.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = self.classes[i];
            }
        }
        best
    }

    fn make_rng(&self, salt: u64) -> SmallRng {
        match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(salt)),
            None => SmallRng::from_os_rng(),
        }
    }

    fn weighted_choice(&self, rng: &mut SmallRng) -> usize {
        let r: f64 = {
            use rand::Rng;
            rng.random::<f64>()
        };
        let mut acc = 0.0_f64;
        for (i, &p) in self.class_priors.iter().enumerate() {
            acc += p;
            if r <= acc {
                return self.classes[i];
            }
        }
        *self.classes.last().expect("at least one class")
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDummyClassifier {
    type Output = Array1<usize>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n = x.nrows();
        let out = match &self.strategy {
            DummyClassifierStrategy::MostFrequent | DummyClassifierStrategy::Prior => {
                Array1::from_elem(n, self.most_frequent())
            }
            DummyClassifierStrategy::Constant(c) => Array1::from_elem(n, *c),
            DummyClassifierStrategy::Stratified => {
                let mut rng = self.make_rng(0);
                let mut buf = Vec::with_capacity(n);
                for _ in 0..n {
                    buf.push(self.weighted_choice(&mut rng));
                }
                Array1::from(buf)
            }
            DummyClassifierStrategy::Uniform => {
                let mut rng = self.make_rng(0);
                let mut buf = Vec::with_capacity(n);
                for _ in 0..n {
                    buf.push(*self.classes.choose(&mut rng).expect("at least one class"));
                }
                Array1::from(buf)
            }
        };
        Ok(out)
    }
}

impl HasClasses for FittedDummyClassifier {
    fn classes(&self) -> &[usize] {
        &self.classes
    }
    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// ---------------------------------------------------------------------------
// DummyRegressor
// ---------------------------------------------------------------------------

/// Strategy used by [`DummyRegressor`] to choose predictions.
#[derive(Debug, Clone, PartialEq)]
pub enum DummyRegressorStrategy<F> {
    /// Predict the mean of the training targets.
    Mean,
    /// Predict the median of the training targets.
    Median,
    /// Predict the empirical `quantile` (in `[0, 1]`) of the training targets.
    Quantile(f64),
    /// Always predict a fixed user-supplied constant.
    Constant(F),
}

/// Baseline regressor that ignores its input features.
#[derive(Debug, Clone)]
pub struct DummyRegressor<F> {
    strategy: DummyRegressorStrategy<F>,
}

impl<F> DummyRegressor<F> {
    /// Construct a new [`DummyRegressor`] with the given strategy.
    #[must_use]
    pub fn new(strategy: DummyRegressorStrategy<F>) -> Self {
        Self { strategy }
    }
}

impl<F: Float> Default for DummyRegressor<F> {
    fn default() -> Self {
        Self::new(DummyRegressorStrategy::Mean)
    }
}

/// Fitted [`DummyRegressor`].
#[derive(Debug, Clone)]
pub struct FittedDummyRegressor<F> {
    constant: F,
}

impl<F> FittedDummyRegressor<F> {
    /// The constant returned for every input row.
    pub fn constant(&self) -> &F {
        &self.constant
    }
}

impl<F: Float + Send + Sync + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for DummyRegressor<F>
{
    type Fitted = FittedDummyRegressor<F>;
    type Error = FerroError;

    fn fit(&self, _x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DummyRegressor::fit".into(),
            });
        }
        let constant = match &self.strategy {
            DummyRegressorStrategy::Mean => {
                let n = F::from(y.len()).ok_or_else(|| FerroError::InvalidParameter {
                    name: "n_samples".into(),
                    reason: "could not convert to F".into(),
                })?;
                y.iter().fold(F::zero(), |acc, &v| acc + v) / n
            }
            DummyRegressorStrategy::Median => quantile_value(y, 0.5)?,
            DummyRegressorStrategy::Quantile(q) => quantile_value(y, *q)?,
            DummyRegressorStrategy::Constant(c) => *c,
        };
        Ok(FittedDummyRegressor { constant })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDummyRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.constant))
    }
}

fn quantile_value<F: Float + FromPrimitive>(y: &Array1<F>, q: f64) -> Result<F, FerroError> {
    if !(0.0..=1.0).contains(&q) {
        return Err(FerroError::InvalidParameter {
            name: "quantile".into(),
            reason: format!("must be in [0, 1], got {q}"),
        });
    }
    let mut sorted: Vec<F> = y.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Ok(sorted[lo])
    } else {
        let frac = F::from(pos - lo as f64).ok_or_else(|| FerroError::InvalidParameter {
            name: "fraction".into(),
            reason: "could not convert to F".into(),
        })?;
        Ok(sorted[lo] + (sorted[hi] - sorted[lo]) * frac)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn x() -> Array2<f64> {
        Array2::<f64>::zeros((5, 2))
    }

    #[test]
    fn dummy_classifier_most_frequent() {
        let y = array![0usize, 0, 1, 2, 0];
        let clf = DummyClassifier::new(DummyClassifierStrategy::MostFrequent);
        let fitted: FittedDummyClassifier = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        assert!(preds.iter().all(|&v| v == 0));
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn dummy_classifier_constant() {
        let y = array![0usize, 1, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Constant(2));
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        assert!(preds.iter().all(|&v| v == 2));
    }

    #[test]
    fn dummy_classifier_constant_invalid() {
        let y = array![0usize, 1];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Constant(99));
        assert!(clf.fit(&x(), &y).is_err());
    }

    #[test]
    fn dummy_classifier_stratified_in_range() {
        let y = array![0usize, 0, 0, 1, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Stratified).random_state(7);
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        for p in preds.iter() {
            assert!(matches!(*p, 0 | 1 | 2));
        }
    }

    #[test]
    fn dummy_classifier_uniform_in_range() {
        let y = array![0usize, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Uniform).random_state(5);
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        for p in preds.iter() {
            assert!(matches!(*p, 0 | 1 | 2));
        }
    }

    #[test]
    fn dummy_regressor_mean() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Mean);
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_median() {
        let y: Array1<f64> = array![1.0, 5.0, 2.0, 4.0, 3.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Median);
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_quantile_25() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Quantile(0.25));
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_constant() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Constant(42.0));
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 42.0).abs() < 1e-12);
        let preds = fitted.predict(&Array2::<f64>::zeros((4, 2))).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
