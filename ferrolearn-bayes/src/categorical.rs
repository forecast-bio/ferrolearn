//! Categorical Naive Bayes classifier.
//!
//! This module provides [`CategoricalNB`], suitable for features that take on
//! one of K discrete categorical values (encoded as non-negative integers cast
//! to floats). Each feature column may have a different number of categories.
//!
//! The log-likelihood for feature `j` in class `c` taking value `k` is:
//!
//! ```text
//! log P(x_j = k | c) = log( (N_cjk + alpha) / (N_c + alpha * K_j) )
//! ```
//!
//! where `N_cjk` is the count of feature `j` equal to `k` in class `c`,
//! `N_c` is the total number of samples in class `c`, `K_j` is the number
//! of distinct categories for feature `j`, and `alpha` is the Laplace
//! smoothing parameter.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_bayes::CategoricalNB;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 3),
//!     vec![
//!         0.0, 1.0, 2.0,
//!         1.0, 0.0, 2.0,
//!         0.0, 1.0, 1.0,
//!         2.0, 0.0, 0.0,
//!         2.0, 1.0, 0.0,
//!         1.0, 0.0, 1.0,
//!     ],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = CategoricalNB::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::marker::PhantomData;

/// Specification of the minimum number of categories per feature.
///
/// Used by [`CategoricalNB::with_min_categories`] /
/// [`CategoricalNB::with_min_categories_per_feature`] to ensure the count
/// tables have enough slots for categories that may not appear in the
/// training data but could appear at predict / partial_fit time.
#[derive(Debug, Clone)]
pub enum MinCategories {
    /// Same minimum count broadcast across every feature.
    Scalar(usize),
    /// Explicit minimum count per feature; length must equal `n_features`
    /// at fit time.
    PerFeature(Vec<usize>),
}

/// Categorical Naive Bayes classifier.
///
/// Suitable for features where each column takes on one of several discrete
/// categorical values (encoded as non-negative integers cast to floats).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct CategoricalNB<F: Float + Send + Sync + 'static> {
    /// Additive (Laplace) smoothing parameter. Default: `1.0`.
    alpha: F,
    /// Optional user-supplied class priors. If set, these are used
    /// instead of computing priors from the data.
    class_prior: Option<Vec<F>>,
    /// Whether to learn class priors from the data. When `false` and
    /// `class_prior` is `None`, uniform priors `1 / n_classes` are used.
    /// Default: `true`.
    fit_prior: bool,
    /// When `false`, `alpha` values below `1e-10` are silently raised to
    /// `1e-10` (legacy behavior). Default: `true`.
    force_alpha: bool,
    /// Optional minimum number of categories per feature. When set, the
    /// fitted count tables are sized to `max(observed_max + 1, min_cats[j])`,
    /// pre-populating slots for category values that might appear at
    /// predict / `partial_fit` time but were not in the training data.
    min_categories: Option<MinCategories>,
    _marker: PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> CategoricalNB<F> {
    /// Create a new `CategoricalNB` with default settings.
    ///
    /// Defaults: `alpha = 1.0`, `class_prior = None`, `fit_prior = true`,
    /// `force_alpha = true`, `min_categories = None`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: F::one(),
            class_prior: None,
            fit_prior: true,
            force_alpha: true,
            min_categories: None,
            _marker: PhantomData,
        }
    }

    /// Set the Laplace smoothing parameter.
    ///
    /// Invalid alpha values are caught at `fit()` time.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set user-supplied class priors.
    ///
    /// The priors must sum to 1.0 and have length equal to the number
    /// of classes discovered during fitting.
    #[must_use]
    pub fn with_class_prior(mut self, priors: Vec<F>) -> Self {
        self.class_prior = Some(priors);
        self
    }

    /// Toggle whether to learn class priors from data. Mirrors sklearn's
    /// `fit_prior`. When `false` and no `class_prior` is set, uniform priors
    /// are used.
    #[must_use]
    pub fn with_fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }

    /// Toggle the `force_alpha` policy. See struct field doc.
    #[must_use]
    pub fn with_force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }

    /// Set the same minimum category count for every feature.
    #[must_use]
    pub fn with_min_categories(mut self, min: usize) -> Self {
        self.min_categories = Some(MinCategories::Scalar(min));
        self
    }

    /// Set per-feature minimum category counts. Length must match
    /// `n_features` at fit time.
    #[must_use]
    pub fn with_min_categories_per_feature(mut self, mins: Vec<usize>) -> Self {
        self.min_categories = Some(MinCategories::PerFeature(mins));
        self
    }
}

impl<F: Float + Send + Sync + 'static> Default for CategoricalNB<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Categorical Naive Bayes classifier.
///
/// Stores the per-class log prior, the raw per-(feature, class, category)
/// integer counts, and the cached log probabilities derived from them.
/// `partial_fit` increments the counts and recomputes log probabilities.
#[derive(Debug, Clone)]
pub struct FittedCategoricalNB<F: Float + Send + Sync + 'static> {
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Log prior probability for each class, length `n_classes`.
    class_log_prior: Vec<F>,
    /// For each feature i and class c, store log P(x_i = k | c) for each
    /// category k. Indexed as `feature_log_prob[feature_idx][class_idx][category_idx]`.
    /// Categories are mapped from their integer value to a contiguous index via
    /// `categories[feature_idx]`. Cached; recomputed by `recompute_log_prob`.
    feature_log_prob: Vec<Vec<Vec<F>>>,
    /// Raw per-(feature, class, category) integer counts. Same shape as
    /// `feature_log_prob`. Used by `partial_fit` to incrementally update.
    category_counts: Vec<Vec<Vec<usize>>>,
    /// For each feature, the sorted list of known category values.
    /// `categories[feature_idx]` is a `Vec<usize>` of category integer values.
    /// May contain min_categories padding values that were not in the
    /// training data.
    categories: Vec<Vec<usize>>,
    /// Per-class sample counts. Index aligns with `classes`.
    class_counts: Vec<usize>,
    /// Number of features the model was fitted on.
    n_features: usize,
    /// Smoothing parameter (post-clamp), carried for `partial_fit`.
    alpha: F,
    /// Optional explicit class priors carried for prior recomputation.
    class_prior: Option<Vec<F>>,
    /// Whether to refit empirical priors during `partial_fit`.
    fit_prior: bool,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for CategoricalNB<F> {
    type Fitted = FittedCategoricalNB<F>;
    type Error = FerroError;

    /// Fit the Categorical NB model.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of rows.
    /// - [`FerroError::InsufficientSamples`] if there are no samples.
    /// - [`FerroError::InvalidParameter`] if `alpha <= 0`.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedCategoricalNB<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "CategoricalNB requires at least one sample".into(),
            });
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.alpha <= F::zero() && self.force_alpha {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "alpha must be positive for CategoricalNB \
                    (or set force_alpha=false to clamp to 1e-10)"
                    .into(),
            });
        }
        let alpha = crate::clamp_alpha(self.alpha, self.force_alpha);

        // Validate min_categories shape against n_features (only relevant
        // for PerFeature; Scalar broadcasts to all).
        if let Some(MinCategories::PerFeature(ref mins)) = self.min_categories
            && mins.len() != n_features
        {
            return Err(FerroError::InvalidParameter {
                name: "min_categories".into(),
                reason: format!(
                    "PerFeature length {} does not match n_features {}",
                    mins.len(),
                    n_features
                ),
            });
        }

        // Collect sorted unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Build per-class sample counts and indices.
        let mut class_counts = vec![0usize; n_classes];
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (sample_idx, &label) in y.iter().enumerate() {
            let ci = classes.iter().position(|&c| c == label).unwrap();
            class_counts[ci] += 1;
            class_indices[ci].push(sample_idx);
        }

        // For each feature, discover categories and build raw count tables.
        let mut category_counts: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_features);
        let mut categories_per_feature: Vec<Vec<usize>> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Discover observed unique category values for this feature.
            let mut cats: Vec<usize> = Vec::new();
            for i in 0..n_samples {
                let val = x[[i, j]].to_usize().unwrap_or(0);
                cats.push(val);
            }
            cats.sort_unstable();
            cats.dedup();

            // Apply min_categories: ensure cats covers 0..min_cats[j] so the
            // count tables have slots even for unobserved category indices.
            let min_cats_j = match self.min_categories {
                Some(MinCategories::Scalar(m)) => m,
                Some(MinCategories::PerFeature(ref v)) => v[j],
                None => 0,
            };
            if min_cats_j > 0 {
                for cv in 0..min_cats_j {
                    if cats.binary_search(&cv).is_err() {
                        let pos = cats.partition_point(|&c| c < cv);
                        cats.insert(pos, cv);
                    }
                }
            }

            // Per-class, per-category count table.
            let mut counts_for_feature: Vec<Vec<usize>> =
                vec![vec![0usize; cats.len()]; n_classes];
            for (ci, indices) in class_indices.iter().enumerate() {
                for &sample_idx in indices {
                    let val = x[[sample_idx, j]].to_usize().unwrap_or(0);
                    if let Ok(cat_idx) = cats.binary_search(&val) {
                        counts_for_feature[ci][cat_idx] += 1;
                    }
                }
            }

            category_counts.push(counts_for_feature);
            categories_per_feature.push(cats);
        }

        // Cached log probabilities derived from counts.
        let feature_log_prob =
            recompute_feature_log_prob(&category_counts, &class_counts, alpha);

        // Resolve priors.
        let class_log_prior =
            resolve_class_log_prior(&class_counts, n_classes, &self.class_prior, self.fit_prior)?;

        Ok(FittedCategoricalNB {
            classes,
            class_log_prior,
            feature_log_prob,
            category_counts,
            categories: categories_per_feature,
            class_counts,
            n_features,
            alpha,
            class_prior: self.class_prior.clone(),
            fit_prior: self.fit_prior,
        })
    }
}

/// Recompute `feature_log_prob[j][c][k]` from raw `category_counts` and
/// `class_counts`, using Laplace smoothing.
fn recompute_feature_log_prob<F: Float>(
    category_counts: &[Vec<Vec<usize>>],
    class_counts: &[usize],
    alpha: F,
) -> Vec<Vec<Vec<F>>> {
    let n_features = category_counts.len();
    let n_classes = class_counts.len();
    let mut out: Vec<Vec<Vec<F>>> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let n_cats = category_counts[j].first().map_or(0, Vec::len);
        let n_cats_f = F::from(n_cats).unwrap();
        let mut per_class: Vec<Vec<F>> = Vec::with_capacity(n_classes);
        for ci in 0..n_classes {
            let n_c_f = F::from(class_counts[ci]).unwrap();
            let denom = n_c_f + alpha * n_cats_f;
            let mut row: Vec<F> = Vec::with_capacity(n_cats);
            for k in 0..n_cats {
                let count_f = F::from(category_counts[j][ci][k]).unwrap();
                row.push(((count_f + alpha) / denom).ln());
            }
            per_class.push(row);
        }
        out.push(per_class);
    }
    out
}

/// Resolve `class_log_prior` from `class_counts`, honoring an optional
/// explicit `class_prior` and the `fit_prior` flag.
fn resolve_class_log_prior<F: Float>(
    class_counts: &[usize],
    n_classes: usize,
    class_prior: &Option<Vec<F>>,
    fit_prior: bool,
) -> Result<Vec<F>, FerroError> {
    let mut out = vec![F::zero(); n_classes];
    if let Some(priors) = class_prior {
        if priors.len() != n_classes {
            return Err(FerroError::InvalidParameter {
                name: "class_prior".into(),
                reason: format!(
                    "length {} does not match number of classes {}",
                    priors.len(),
                    n_classes
                ),
            });
        }
        for (ci, &p) in priors.iter().enumerate() {
            out[ci] = p.ln();
        }
    } else if fit_prior {
        let total: usize = class_counts.iter().sum();
        let total_f = F::from(total).unwrap();
        for (ci, &c) in class_counts.iter().enumerate() {
            out[ci] = (F::from(c).unwrap() / total_f).ln();
        }
    } else {
        let uniform = (F::one() / F::from(n_classes).unwrap()).ln();
        for ci in 0..n_classes {
            out[ci] = uniform;
        }
    }
    Ok(out)
}

impl<F: Float + Send + Sync + 'static> FittedCategoricalNB<F> {
    /// Incrementally update the model with new data.
    ///
    /// Increments the per-(feature, class, category) counts and recomputes
    /// the cached log probabilities. Unlike sklearn's strict integer-index
    /// model, ferrolearn's CategoricalNB allows new category values to
    /// appear at `partial_fit` time — they are appended to `categories[j]`
    /// and a new count slot is allocated for every class.
    ///
    /// New class labels not seen at `fit` time are likewise added.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row
    ///   counts or the feature count does not match the fitted model.
    pub fn partial_fit(&mut self, x: &Array2<F>, y: &Array1<usize>) -> Result<(), FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Ok(());
        }

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }

        // Discover and integrate any new class labels (preserving sorted
        // order so `classes` stays a sorted unique list).
        for &label in y {
            if self.classes.binary_search(&label).is_err() {
                let pos = self.classes.partition_point(|&c| c < label);
                self.classes.insert(pos, label);
                self.class_counts.insert(pos, 0);
                self.class_log_prior.insert(pos, F::neg_infinity());
                for j in 0..self.n_features {
                    let n_cats = self.categories[j].len();
                    self.category_counts[j].insert(pos, vec![0usize; n_cats]);
                    self.feature_log_prob[j].insert(pos, vec![F::zero(); n_cats]);
                }
            }
        }

        // Walk every sample, extending `categories[j]` for unseen values.
        for sample_idx in 0..n_samples {
            let label = y[sample_idx];
            let ci = self.classes.binary_search(&label).unwrap();
            self.class_counts[ci] += 1;
            for j in 0..self.n_features {
                let val = x[[sample_idx, j]].to_usize().unwrap_or(0);
                let cat_idx = match self.categories[j].binary_search(&val) {
                    Ok(idx) => idx,
                    Err(insert_pos) => {
                        // New category: insert into categories[j] and add a
                        // count slot for every class.
                        self.categories[j].insert(insert_pos, val);
                        for c in 0..self.classes.len() {
                            self.category_counts[j][c].insert(insert_pos, 0);
                        }
                        insert_pos
                    }
                };
                self.category_counts[j][ci][cat_idx] += 1;
            }
        }

        // Recompute cached log probabilities and class priors.
        self.feature_log_prob =
            recompute_feature_log_prob(&self.category_counts, &self.class_counts, self.alpha);
        self.class_log_prior = resolve_class_log_prior(
            &self.class_counts,
            self.classes.len(),
            &self.class_prior,
            self.fit_prior,
        )?;

        Ok(())
    }

    /// Look up the log probability for a given feature, class, and category value.
    ///
    /// If the category value was not seen during training, returns a uniform
    /// log probability based on the number of known categories for that feature.
    fn log_prob_for(&self, feature_idx: usize, class_idx: usize, cat_value: usize) -> F {
        let cats = &self.categories[feature_idx];
        if let Ok(cat_idx) = cats.binary_search(&cat_value) {
            self.feature_log_prob[feature_idx][class_idx][cat_idx]
        } else {
            // Unseen category: use uniform probability 1 / (n_known_cats + 1).
            // This gracefully degrades for unseen categories.
            let n_cats_plus_one = F::from(cats.len() + 1).unwrap();
            (F::one() / n_cats_plus_one).ln()
        }
    }

    /// Compute joint log-likelihood for each class.
    ///
    /// Returns shape `(n_samples, n_classes)`.
    fn joint_log_likelihood(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut scores = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for ci in 0..n_classes {
                let mut score = self.class_log_prior[ci];
                for j in 0..self.n_features {
                    let cat_value = x[[i, j]].to_usize().unwrap_or(0);
                    score = score + self.log_prob_for(j, ci, cat_value);
                }
                scores[[i, ci]] = score;
            }
        }

        scores
    }

    /// Predict class probabilities for the given feature matrix.
    ///
    /// Returns shape `(n_samples, n_classes)` where each row sums to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }

        let log_scores = self.joint_log_likelihood(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            // Numerically stable softmax: subtract row max before exp.
            let max_score = log_scores
                .row(i)
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b));

            let mut row_sum = F::zero();
            for ci in 0..n_classes {
                let p = (log_scores[[i, ci]] - max_score).exp();
                proba[[i, ci]] = p;
                row_sum = row_sum + p;
            }
            for ci in 0..n_classes {
                proba[[i, ci]] = proba[[i, ci]] / row_sum;
            }
        }

        Ok(proba)
    }

    /// Compute the unnormalized joint log-likelihood `log P(c) + log P(x|c)`.
    ///
    /// Returns shape `(n_samples, n_classes)`. Matches sklearn
    /// `CategoricalNB._joint_log_likelihood`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_joint_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }
        Ok(self.joint_log_likelihood(x))
    }

    /// Compute log of class probabilities (numerically stable).
    ///
    /// Returns shape `(n_samples, n_classes)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let jll = self.predict_joint_log_proba(x)?;
        Ok(crate::log_softmax_rows(&jll))
    }

    /// Mean accuracy on the given test data and labels.
    ///
    /// Equivalent to sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the fitted model.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let n = y.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(F::from(correct).unwrap() / F::from(n).unwrap())
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedCategoricalNB<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does
    /// not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![x.ncols()],
                context: "number of features must match fitted CategoricalNB".into(),
            });
        }

        let scores = self.joint_log_likelihood(x);
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_class = 0;
            let mut best_score = scores[[i, 0]];
            for ci in 1..n_classes {
                if scores[[i, ci]] > best_score {
                    best_score = scores[[i, ci]];
                    best_class = ci;
                }
            }
            predictions[i] = self.classes[best_class];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + 'static> HasClasses for FittedCategoricalNB<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// Pipeline integration.
impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> PipelineEstimator<F>
    for CategoricalNB<F>
{
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedCategoricalNBPipeline(fitted)))
    }
}

struct FittedCategoricalNBPipeline<F: Float + Send + Sync + 'static>(FittedCategoricalNB<F>);

unsafe impl<F: Float + Send + Sync + 'static> Send for FittedCategoricalNBPipeline<F> {}
unsafe impl<F: Float + Send + Sync + 'static> Sync for FittedCategoricalNBPipeline<F> {}

impl<F: Float + ToPrimitive + FromPrimitive + Send + Sync + 'static> FittedPipelineEstimator<F>
    for FittedCategoricalNBPipeline<F>
{
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| F::from_usize(v).unwrap_or_else(F::nan)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_categorical_data() -> (Array2<f64>, Array1<usize>) {
        // Categorical features: 3 features, each taking values in {0, 1, 2}.
        // Class 0 tends to have low values, class 1 tends to have high values.
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 1.0, // class 0
                1.0, 0.0, 0.0, // class 0
                0.0, 1.0, 0.0, // class 0
                0.0, 0.0, 0.0, // class 0
                2.0, 2.0, 1.0, // class 1
                2.0, 1.0, 2.0, // class 1
                1.0, 2.0, 2.0, // class 1
                2.0, 2.0, 2.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_categorical_nb_fit_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        // Should get most or all correct on training data.
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_categorical_nb_predict_proba_sums_to_one() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 8);
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert_relative_eq!(proba.row(i).sum(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_categorical_nb_has_classes() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_categorical_nb_alpha_smoothing_effect() {
        let (x, y) = make_categorical_data();

        // With small alpha, sharper probabilities.
        let model_sharp = CategoricalNB::<f64>::new().with_alpha(0.01);
        let fitted_sharp = model_sharp.fit(&x, &y).unwrap();
        let proba_sharp = fitted_sharp.predict_proba(&x).unwrap();

        // With large alpha, smoother probabilities (closer to uniform).
        let model_smooth = CategoricalNB::<f64>::new().with_alpha(100.0);
        let fitted_smooth = model_smooth.fit(&x, &y).unwrap();
        let proba_smooth = fitted_smooth.predict_proba(&x).unwrap();

        // Smoothed probabilities for the dominant class on a class-0 sample
        // should be less extreme.
        let sharp_max = proba_sharp[[0, 0]].max(proba_sharp[[0, 1]]);
        let smooth_max = proba_smooth[[0, 0]].max(proba_smooth[[0, 1]]);
        assert!(smooth_max < sharp_max);
    }

    #[test]
    fn test_categorical_nb_invalid_alpha_zero() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(0.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_categorical_nb_invalid_alpha_negative() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new().with_alpha(-1.0);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
        match result.unwrap_err() {
            FerroError::InvalidParameter { name, .. } => assert_eq!(name, "alpha"),
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((4, 3), vec![0.0; 12]).unwrap();
        let y = array![0usize, 1]; // Wrong length
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_shape_mismatch_predict() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let x_bad = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
        assert!(fitted.predict_proba(&x_bad).is_err());
    }

    #[test]
    fn test_categorical_nb_empty_data() {
        let x = Array2::<f64>::zeros((0, 3));
        let y = Array1::<usize>::zeros(0);
        let model = CategoricalNB::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_categorical_nb_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = array![2usize, 2, 2];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[2]);
        let preds = fitted.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 2));
    }

    #[test]
    fn test_categorical_nb_default() {
        let model = CategoricalNB::<f64>::default();
        assert_relative_eq!(model.alpha, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_categorical_nb_unseen_category() {
        // Fit on categories {0, 1}, then predict with a sample containing
        // category 5 (unseen). Should not panic, should return valid probabilities.
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // Predict with unseen category 5 in feature 0.
        let x_new = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
        let preds = fitted.predict(&x_new).unwrap();
        assert_eq!(preds.len(), 1);

        let proba = fitted.predict_proba(&x_new).unwrap();
        assert_relative_eq!(proba.row(0).sum(), 1.0, epsilon = 1e-10);
        // Both probabilities should be between 0 and 1.
        assert!(proba[[0, 0]] > 0.0 && proba[[0, 0]] < 1.0);
        assert!(proba[[0, 1]] > 0.0 && proba[[0, 1]] < 1.0);
    }

    #[test]
    fn test_categorical_nb_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                1.0, 1.0, // class 1
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
                2.0, 2.0, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert_eq!(correct, 9);
    }

    #[test]
    fn test_categorical_nb_pipeline() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit_pipeline(&x, &y).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_categorical_nb_predict_proba_ordering() {
        let (x, y) = make_categorical_data();
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // First 4 samples are class 0 — class 0 probability should be higher.
        for i in 0..4 {
            assert!(
                proba[[i, 0]] > proba[[i, 1]],
                "sample {i}: P(c=0)={} should be > P(c=1)={}",
                proba[[i, 0]],
                proba[[i, 1]]
            );
        }
        // Last 4 samples are class 1 — class 1 probability should be higher.
        for i in 4..8 {
            assert!(
                proba[[i, 1]] > proba[[i, 0]],
                "sample {i}: P(c=1)={} should be > P(c=0)={}",
                proba[[i, 1]],
                proba[[i, 0]]
            );
        }
    }

    #[test]
    fn test_categorical_nb_f32() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = CategoricalNB::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 4);

        let proba = fitted.predict_proba(&x).unwrap();
        for i in 0..proba.nrows() {
            let sum: f32 = proba.row(i).sum();
            assert!((sum - 1.0f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_categorical_nb_unordered_classes() {
        // Classes are not 0..n, and not contiguous.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        )
        .unwrap();
        let y = array![5usize, 5, 5, 10, 10, 10];
        let model = CategoricalNB::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[5, 10]);

        let preds = fitted.predict(&x).unwrap();
        // First 3 should predict class 5.
        for i in 0..3 {
            assert_eq!(preds[i], 5);
        }
        // Last 3 should predict class 10.
        for i in 3..6 {
            assert_eq!(preds[i], 10);
        }
    }
}
