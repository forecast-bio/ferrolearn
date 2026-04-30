//! Bernoulli Restricted Boltzmann Machine.
//!
//! [`BernoulliRBM`] models the joint distribution of a binary visible layer
//! and a binary hidden layer with energy
//!
//! ```text
//! E(v, h) = -v^T W h - b_v^T v - b_h^T h
//! ```
//!
//! It is trained via one-step Contrastive Divergence (CD-1). Inputs are
//! interpreted as Bernoulli probabilities; the [`transform`] method returns
//! the probability that each hidden unit is active given the visible vector,
//! which is the standard scikit-learn convention.
//!
//! [`transform`]: FittedBernoulliRBM::transform

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Bernoulli Restricted Boltzmann Machine.
#[derive(Debug, Clone)]
pub struct BernoulliRBM<F> {
    n_components: usize,
    learning_rate: F,
    n_iter: usize,
    batch_size: usize,
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> BernoulliRBM<F> {
    /// Construct a new [`BernoulliRBM`] with the given hidden-layer size.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            learning_rate: F::from(0.1).unwrap(),
            n_iter: 10,
            batch_size: 10,
            random_state: None,
        }
    }

    /// Set the SGD learning rate (default `0.1`).
    #[must_use]
    pub fn learning_rate(mut self, lr: F) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of full passes over the data (default `10`).
    #[must_use]
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the mini-batch size (default `10`).
    #[must_use]
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// A fitted [`BernoulliRBM`].
#[derive(Debug, Clone)]
pub struct FittedBernoulliRBM<F> {
    /// Weight matrix of shape `(n_components, n_features)`.
    pub components_: Array2<F>,
    /// Hidden-layer bias of length `n_components`.
    pub intercept_hidden_: Array1<F>,
    /// Visible-layer bias of length `n_features`.
    pub intercept_visible_: Array1<F>,
    /// Number of training iterations actually run.
    pub n_iter_: usize,
}

impl<F: Float + Send + Sync + 'static> FittedBernoulliRBM<F> {
    /// Compute the probability of each hidden unit being active given `v`.
    ///
    /// `v` should have shape `(n_samples, n_features)` and contain values
    /// in `[0, 1]` (Bernoulli probabilities).
    pub fn transform(&self, v: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if v.ncols() != self.intercept_visible_.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.intercept_visible_.len()],
                actual: vec![v.ncols()],
                context: "BernoulliRBM::transform: feature count mismatch".into(),
            });
        }
        let n = v.nrows();
        let h = self.intercept_hidden_.len();
        let mut out = Array2::<F>::zeros((n, h));
        for i in 0..n {
            for j in 0..h {
                let mut acc = self.intercept_hidden_[j];
                for k in 0..self.intercept_visible_.len() {
                    acc = acc + v[[i, k]] * self.components_[[j, k]];
                }
                out[[i, j]] = sigmoid(acc);
            }
        }
        Ok(out)
    }

    /// Run one Gibbs sampling step from `v` to a new visible vector via the
    /// hidden layer, returning the reconstructed visible probabilities.
    pub fn gibbs(&self, v: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let h_prob = self.transform(v)?;
        // Reconstruct visible probabilities P(v=1 | h)
        let n = h_prob.nrows();
        let n_v = self.intercept_visible_.len();
        let mut out = Array2::<F>::zeros((n, n_v));
        for i in 0..n {
            for k in 0..n_v {
                let mut acc = self.intercept_visible_[k];
                for j in 0..self.intercept_hidden_.len() {
                    acc = acc + h_prob[[i, j]] * self.components_[[j, k]];
                }
                out[[i, k]] = sigmoid(acc);
            }
        }
        Ok(out)
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for BernoulliRBM<F> {
    type Fitted = FittedBernoulliRBM<F>;
    type Error = FerroError;

    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedBernoulliRBM<F>, FerroError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples == 0 || n_features == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: n_samples.min(n_features),
                context: "BernoulliRBM::fit".into(),
            });
        }

        // Sanity: inputs should be in [0, 1]; we don't enforce here but the
        // CD update relies on it. Casting to f64 for RNG samples then back to F.
        let mut rng = match self.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE),
        };
        let init_normal =
            Normal::new(0.0_f64, 0.01_f64).map_err(|e| FerroError::InvalidParameter {
                name: "init normal".into(),
                reason: e.to_string(),
            })?;
        let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
            name: "uniform".into(),
            reason: e.to_string(),
        })?;

        let mut components = Array2::<F>::zeros((self.n_components, n_features));
        for j in 0..self.n_components {
            for k in 0..n_features {
                let v: f64 = init_normal.sample(&mut rng);
                components[[j, k]] = F::from(v).ok_or_else(|| FerroError::InvalidParameter {
                    name: "init weight".into(),
                    reason: "could not convert".into(),
                })?;
            }
        }
        let mut intercept_hidden = Array1::<F>::zeros(self.n_components);
        let mut intercept_visible = Array1::<F>::zeros(n_features);

        let lr = self.learning_rate;
        let n_iter = self.n_iter;
        let bs = self.batch_size.min(n_samples);
        let n_batches = n_samples.div_ceil(bs);

        for _epoch in 0..n_iter {
            // Shuffle indices for this epoch.
            let mut indices: Vec<usize> = (0..n_samples).collect();
            // Fisher–Yates with our RNG
            for i in (1..n_samples).rev() {
                let r = unif.sample(&mut rng);
                let j = ((r * (i as f64 + 1.0)).floor() as usize).min(i);
                indices.swap(i, j);
            }

            for batch_idx in 0..n_batches {
                let start = batch_idx * bs;
                let end = (start + bs).min(n_samples);
                let batch: Vec<usize> = indices[start..end].to_vec();
                let m = batch.len();
                let m_f = F::from(m).ok_or_else(|| FerroError::InvalidParameter {
                    name: "batch size".into(),
                    reason: "could not convert".into(),
                })?;

                // Positive phase: P(h=1 | v) for each sample
                let mut h_pos = Array2::<F>::zeros((m, self.n_components));
                for (bi, &si) in batch.iter().enumerate() {
                    for j in 0..self.n_components {
                        let mut acc = intercept_hidden[j];
                        for k in 0..n_features {
                            acc = acc + x[[si, k]] * components[[j, k]];
                        }
                        h_pos[[bi, j]] = sigmoid(acc);
                    }
                }

                // Sample h ~ Bernoulli(h_pos)
                let mut h_sample = Array2::<F>::zeros((m, self.n_components));
                for bi in 0..m {
                    for j in 0..self.n_components {
                        let r: f64 = unif.sample(&mut rng);
                        let p = h_pos[[bi, j]].to_f64().unwrap_or(0.0);
                        h_sample[[bi, j]] = if r < p { F::one() } else { F::zero() };
                    }
                }

                // Negative phase: reconstruct v then re-sample h
                let mut v_neg = Array2::<F>::zeros((m, n_features));
                for bi in 0..m {
                    for k in 0..n_features {
                        let mut acc = intercept_visible[k];
                        for j in 0..self.n_components {
                            acc = acc + h_sample[[bi, j]] * components[[j, k]];
                        }
                        v_neg[[bi, k]] = sigmoid(acc);
                    }
                }
                let mut h_neg = Array2::<F>::zeros((m, self.n_components));
                for bi in 0..m {
                    for j in 0..self.n_components {
                        let mut acc = intercept_hidden[j];
                        for k in 0..n_features {
                            acc = acc + v_neg[[bi, k]] * components[[j, k]];
                        }
                        h_neg[[bi, j]] = sigmoid(acc);
                    }
                }

                // Update components: W += lr / m * (h_pos^T v - h_neg^T v_neg)
                for j in 0..self.n_components {
                    for k in 0..n_features {
                        let mut delta = F::zero();
                        for (bi, &si) in batch.iter().enumerate() {
                            delta = delta + h_pos[[bi, j]] * x[[si, k]]
                                - h_neg[[bi, j]] * v_neg[[bi, k]];
                        }
                        components[[j, k]] = components[[j, k]] + lr * delta / m_f;
                    }
                }
                // Hidden bias update
                for j in 0..self.n_components {
                    let mut delta = F::zero();
                    for bi in 0..m {
                        delta = delta + h_pos[[bi, j]] - h_neg[[bi, j]];
                    }
                    intercept_hidden[j] = intercept_hidden[j] + lr * delta / m_f;
                }
                // Visible bias update
                for k in 0..n_features {
                    let mut delta = F::zero();
                    for (bi, &si) in batch.iter().enumerate() {
                        delta = delta + x[[si, k]] - v_neg[[bi, k]];
                    }
                    intercept_visible[k] = intercept_visible[k] + lr * delta / m_f;
                }
            }
        }

        Ok(FittedBernoulliRBM {
            components_: components,
            intercept_hidden_: intercept_hidden,
            intercept_visible_: intercept_visible,
            n_iter_: n_iter,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedBernoulliRBM<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        FittedBernoulliRBM::transform(self, x)
    }
}

#[inline]
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Array2<f64> {
        // Two pairs of correlated features in [0, 1]
        array![
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn rbm_fit_smoke() {
        let rbm = BernoulliRBM::<f64>::new(2)
            .learning_rate(0.1)
            .n_iter(5)
            .batch_size(3)
            .random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        assert_eq!(fitted.components_.dim(), (2, 4));
        assert_eq!(fitted.intercept_hidden_.len(), 2);
        assert_eq!(fitted.intercept_visible_.len(), 4);
        assert_eq!(fitted.n_iter_, 5);
    }

    #[test]
    fn rbm_transform_shape_and_range() {
        let rbm = BernoulliRBM::<f64>::new(3)
            .learning_rate(0.1)
            .n_iter(2)
            .random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let h = fitted.transform(&data()).unwrap();
        assert_eq!(h.dim(), (6, 3));
        for v in h.iter() {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn rbm_gibbs_round_trip_shape() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let v_recon = fitted.gibbs(&data()).unwrap();
        assert_eq!(v_recon.dim(), (6, 4));
    }

    #[test]
    fn rbm_feature_dim_mismatch() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let fitted = rbm.fit(&data(), &()).unwrap();
        let bad: Array2<f64> = Array2::zeros((2, 9));
        assert!(fitted.transform(&bad).is_err());
    }

    #[test]
    fn rbm_empty_input_rejected() {
        let rbm = BernoulliRBM::<f64>::new(2).n_iter(2).random_state(7);
        let bad: Array2<f64> = Array2::zeros((0, 4));
        assert!(rbm.fit(&bad, &()).is_err());
    }
}
