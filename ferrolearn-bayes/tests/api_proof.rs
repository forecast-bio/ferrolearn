//! Proof-of-API integration test for ferrolearn-bayes.
//!
//! Audit deliverable for crosslink #260 (under #241). Exercises every
//! public API surface of the crate end-to-end so that future PRs that
//! change the public API have a green-or-red signal here. Adding a method
//! here is the canonical way to confirm a sklearn-parity feature actually
//! exists in the public API.
//!
//! Coverage matrix (one block per estimator):
//!
//! | Estimator      | new | with_alpha | with_class_prior | with_fit_prior | with_force_alpha | with_binarize | with_norm | with_min_categories | fit | predict | predict_proba | predict_log_proba | predict_joint_log_proba | score | partial_fit | classes/n_classes |
//! |----------------|:---:|:----------:|:----------------:|:--------------:|:----------------:|:-------------:|:---------:|:-------------------:|:---:|:-------:|:-------------:|:-----------------:|:-----------------------:|:-----:|:-----------:|:-----------------:|
//! | GaussianNB     |  ✓  |     n/a    |        ✓ (var)   |       n/a      |        n/a       |      n/a      |    n/a    |         n/a         |  ✓  |    ✓    |       ✓       |         ✓         |            ✓            |   ✓   |      ✓      |         ✓         |
//! | MultinomialNB  |  ✓  |      ✓     |         ✓        |        ✓       |         ✓        |      n/a      |    n/a    |         n/a         |  ✓  |    ✓    |       ✓       |         ✓         |            ✓            |   ✓   |      ✓      |         ✓         |
//! | BernoulliNB    |  ✓  |      ✓     |         ✓        |        ✓       |         ✓        |       ✓       |    n/a    |         n/a         |  ✓  |    ✓    |       ✓       |         ✓         |            ✓            |   ✓   |      ✓      |         ✓         |
//! | ComplementNB   |  ✓  |      ✓     |         ✓        |        ✓       |         ✓        |      n/a      |     ✓     |         n/a         |  ✓  |    ✓    |       ✓       |         ✓         |            ✓            |   ✓   |      ✓      |         ✓         |
//! | CategoricalNB  |  ✓  |      ✓     |         ✓        |        ✓       |         ✓        |      n/a      |    n/a    |          ✓          |  ✓  |    ✓    |       ✓       |         ✓         |            ✓            |   ✓   |      ✓      |         ✓         |
//!
//! Plus a block for the `conjugate` module's free functions.

use approx::assert_relative_eq;
use ferrolearn_bayes::categorical::MinCategories;
use ferrolearn_bayes::conjugate::{NormalNormalPosterior, posterior_normal_normal};
use ferrolearn_bayes::{BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB};
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array2, array};

/// Verify a probability matrix has the right shape, values in [0, 1], and
/// rows that sum to 1 within tolerance.
fn assert_proba_well_formed(proba: &Array2<f64>, n_samples: usize, n_classes: usize) {
    assert_eq!(proba.nrows(), n_samples, "predict_proba row count");
    assert_eq!(proba.ncols(), n_classes, "predict_proba col count");
    for i in 0..n_samples {
        let row_sum: f64 = proba.row(i).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        for ci in 0..n_classes {
            assert!(
                (0.0..=1.0).contains(&proba[[i, ci]]),
                "row {i} col {ci} = {} not in [0, 1]",
                proba[[i, ci]]
            );
        }
    }
}

/// Verify log_proba matches log(predict_proba) within tolerance.
fn assert_log_proba_consistent(log_proba: &Array2<f64>, proba: &Array2<f64>) {
    assert_eq!(log_proba.dim(), proba.dim());
    for i in 0..log_proba.nrows() {
        for ci in 0..log_proba.ncols() {
            let p = proba[[i, ci]];
            // Avoid log(0) at boundaries.
            if p > 1e-100 {
                assert_relative_eq!(log_proba[[i, ci]], p.ln(), epsilon = 1e-9, max_relative = 1e-9);
            }
        }
    }
}

// =============================================================================
// GaussianNB — continuous features, ctor params: priors, var_smoothing
// =============================================================================
#[test]
fn api_proof_gaussian_nb() {
    // Two well-separated Gaussian clusters.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 2.5, 1.2, 1.8, 6.0, 7.0, 5.8, 6.5, 6.2, 7.2],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // ── Default ctor ──
    let m = GaussianNB::<f64>::new();
    let fitted = m.fit(&x, &y).unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    assert_eq!(fitted.n_classes(), 2);
    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let log_proba = fitted.predict_log_proba(&x).unwrap();
    let jll = fitted.predict_joint_log_proba(&x).unwrap();
    let acc = fitted.score(&x, &y).unwrap();

    assert_eq!(preds.len(), 6);
    assert_proba_well_formed(&proba, 6, 2);
    assert_log_proba_consistent(&log_proba, &proba);
    assert_eq!(jll.dim(), (6, 2));
    assert_relative_eq!(acc, 1.0, epsilon = 1e-10);

    // ── Builders: with_var_smoothing + with_class_prior ──
    let m2 = GaussianNB::<f64>::new()
        .with_var_smoothing(1e-5)
        .with_class_prior(vec![0.3, 0.7]);
    let fitted2 = m2.fit(&x, &y).unwrap();
    let proba2 = fitted2.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba2, 6, 2);

    // ── partial_fit ──
    let mut fitted3 = GaussianNB::<f64>::new().fit(&x, &y).unwrap();
    let x2 = Array2::from_shape_vec((2, 2), vec![1.3, 2.2, 6.1, 6.9]).unwrap();
    let y2 = array![0usize, 1];
    fitted3.partial_fit(&x2, &y2).unwrap();
    let _ = fitted3.predict(&x2).unwrap();
}

// =============================================================================
// MultinomialNB — count features, ctor params: alpha, fit_prior,
// class_prior, force_alpha
// =============================================================================
#[test]
fn api_proof_multinomial_nb() {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            3.0, 1.0, 0.0, 2.0, 0.0, 1.0, 4.0, 2.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 3.0, 0.0, 2.0,
            5.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // ── Default ctor + every public method ──
    let m = MultinomialNB::<f64>::new();
    let fitted = m.fit(&x, &y).unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let log_proba = fitted.predict_log_proba(&x).unwrap();
    let jll = fitted.predict_joint_log_proba(&x).unwrap();
    let acc = fitted.score(&x, &y).unwrap();
    assert_eq!(preds.len(), 6);
    assert_proba_well_formed(&proba, 6, 2);
    assert_log_proba_consistent(&log_proba, &proba);
    assert_eq!(jll.dim(), (6, 2));
    assert!(acc > 0.0);

    // ── Every builder ──
    let m2 = MultinomialNB::<f64>::new()
        .with_alpha(0.5)
        .with_class_prior(vec![0.4, 0.6])
        .with_fit_prior(false)
        .with_force_alpha(false);
    let fitted2 = m2.fit(&x, &y).unwrap();
    let _ = fitted2.predict(&x).unwrap();

    // ── fit_prior=false without class_prior → uniform priors ──
    let m3 = MultinomialNB::<f64>::new().with_fit_prior(false);
    let fitted3 = m3.fit(&x, &y).unwrap();
    let _ = fitted3.predict_proba(&x).unwrap();

    // ── partial_fit ──
    let mut fitted4 = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    let x2 = Array2::from_shape_vec((2, 3), vec![5.0, 1.0, 0.0, 0.0, 1.0, 6.0]).unwrap();
    let y2 = array![0usize, 1];
    fitted4.partial_fit(&x2, &y2).unwrap();
    let _ = fitted4.predict(&x2).unwrap();
}

// =============================================================================
// BernoulliNB — binary features, ctor params: alpha, binarize, fit_prior,
// class_prior, force_alpha
// =============================================================================
#[test]
fn api_proof_bernoulli_nb() {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            1.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // ── Default ctor ──
    let m = BernoulliNB::<f64>::new();
    let fitted = m.fit(&x, &y).unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let log_proba = fitted.predict_log_proba(&x).unwrap();
    let jll = fitted.predict_joint_log_proba(&x).unwrap();
    let acc = fitted.score(&x, &y).unwrap();
    assert_eq!(preds.len(), 6);
    assert_proba_well_formed(&proba, 6, 2);
    assert_log_proba_consistent(&log_proba, &proba);
    assert_eq!(jll.dim(), (6, 2));
    assert!(acc > 0.0);

    // ── Every builder ──
    let m2 = BernoulliNB::<f64>::new()
        .with_alpha(0.5)
        .with_binarize(0.5)
        .with_class_prior(vec![0.5, 0.5])
        .with_fit_prior(false)
        .with_force_alpha(false);
    let fitted2 = m2.fit(&x, &y).unwrap();
    // Binarized prediction should still produce valid probabilities even
    // when given continuous-valued features.
    let x_cont = Array2::from_shape_vec((2, 3), vec![0.7, 0.3, 0.9, 0.1, 0.8, 0.2]).unwrap();
    let proba2 = fitted2.predict_proba(&x_cont).unwrap();
    assert_proba_well_formed(&proba2, 2, 2);

    // ── partial_fit ──
    let mut fitted3 = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
    let x2 = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let y2 = array![0usize, 1];
    fitted3.partial_fit(&x2, &y2).unwrap();
    let _ = fitted3.predict(&x2).unwrap();
}

// =============================================================================
// ComplementNB — count features, ctor params: alpha, class_prior,
// fit_prior, force_alpha, norm
// =============================================================================
#[test]
fn api_proof_complement_nb() {
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            5.0, 1.0, 0.0, 4.0, 0.0, 1.0, 6.0, 2.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 0.0, 2.0,
            6.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];

    // ── Default ctor ──
    let m = ComplementNB::<f64>::new();
    let fitted = m.fit(&x, &y).unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let log_proba = fitted.predict_log_proba(&x).unwrap();
    let jll = fitted.predict_joint_log_proba(&x).unwrap();
    let acc = fitted.score(&x, &y).unwrap();
    assert_eq!(preds.len(), 6);
    assert_proba_well_formed(&proba, 6, 2);
    assert_log_proba_consistent(&log_proba, &proba);
    assert_eq!(jll.dim(), (6, 2));
    assert!(acc > 0.0);

    // ── Every builder, including with_norm ──
    let m2 = ComplementNB::<f64>::new()
        .with_alpha(0.5)
        .with_class_prior(vec![0.5, 0.5])
        .with_fit_prior(false)
        .with_force_alpha(false)
        .with_norm(true);
    let fitted2 = m2.fit(&x, &y).unwrap();
    let proba2 = fitted2.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba2, 6, 2);

    // ── partial_fit ──
    let mut fitted3 = ComplementNB::<f64>::new().fit(&x, &y).unwrap();
    let x2 = Array2::from_shape_vec((2, 3), vec![5.0, 1.0, 0.0, 0.0, 1.0, 6.0]).unwrap();
    let y2 = array![0usize, 1];
    fitted3.partial_fit(&x2, &y2).unwrap();
    let _ = fitted3.predict(&x2).unwrap();
}

// =============================================================================
// CategoricalNB — discrete categorical features, ctor params: alpha,
// class_prior, fit_prior, force_alpha, min_categories
// =============================================================================
#[test]
fn api_proof_categorical_nb() {
    let x = Array2::from_shape_vec(
        (8, 3),
        vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

    // ── Default ctor ──
    let m = CategoricalNB::<f64>::new();
    let fitted = m.fit(&x, &y).unwrap();
    assert_eq!(fitted.classes(), &[0, 1]);
    let preds = fitted.predict(&x).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();
    let log_proba = fitted.predict_log_proba(&x).unwrap();
    let jll = fitted.predict_joint_log_proba(&x).unwrap();
    let acc = fitted.score(&x, &y).unwrap();
    assert_eq!(preds.len(), 8);
    assert_proba_well_formed(&proba, 8, 2);
    assert_log_proba_consistent(&log_proba, &proba);
    assert_eq!(jll.dim(), (8, 2));
    assert!(acc > 0.0);

    // ── Every builder, including min_categories scalar form ──
    let m2 = CategoricalNB::<f64>::new()
        .with_alpha(0.5)
        .with_class_prior(vec![0.5, 0.5])
        .with_fit_prior(false)
        .with_force_alpha(false)
        .with_min_categories(5);
    let fitted2 = m2.fit(&x, &y).unwrap();
    let proba2 = fitted2.predict_proba(&x).unwrap();
    assert_proba_well_formed(&proba2, 8, 2);

    // ── min_categories per-feature form ──
    let m3 = CategoricalNB::<f64>::new().with_min_categories_per_feature(vec![5, 4, 3]);
    let fitted3 = m3.fit(&x, &y).unwrap();
    let _ = fitted3.predict(&x).unwrap();

    // ── partial_fit including a category that wasn't in the training set
    // (ferrolearn extends categories[j] to accommodate it) ──
    let mut fitted4 = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();
    let x2 = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 5.0, 5.0, 5.0]).unwrap();
    let y2 = array![0usize, 1];
    fitted4.partial_fit(&x2, &y2).unwrap();
    let _ = fitted4.predict(&x2).unwrap();

    // ── Confirm MinCategories enum is publicly constructible (smoke) ──
    let _scalar = MinCategories::Scalar(3);
    let _per = MinCategories::PerFeature(vec![3, 4, 2]);
}

// =============================================================================
// conjugate module — Normal-Normal posterior
// =============================================================================
#[test]
fn api_proof_conjugate_normal_normal() {
    // Empty observations → posterior == prior.
    let post = posterior_normal_normal(0.5, 2.0, &[]);
    assert_relative_eq!(post.mean, 0.5, epsilon = 1e-12);
    assert_relative_eq!(post.var, 2.0, epsilon = 1e-12);

    // Worked example from the doc comment.
    let post: NormalNormalPosterior =
        posterior_normal_normal(0.0, 1.0, &[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
    assert_relative_eq!(post.mean, 1.5, epsilon = 1e-12);
    assert_relative_eq!(post.var, 0.25, epsilon = 1e-12);

    // Degenerate prior (≤0) is clipped, not NaN.
    let post = posterior_normal_normal(0.0, -1.0, &[(1.0, 1.0)]);
    assert!(post.mean.is_finite());
    assert!(post.var.is_finite() && post.var > 0.0);
}

// =============================================================================
// Cross-cutting: f32 numeric type compiles + works for every NB
// =============================================================================
#[test]
fn api_proof_f32_compiles() {
    let x32 = Array2::from_shape_vec((4, 2), vec![1.0f32, 2.0, 1.0, 2.5, 5.0, 6.0, 5.5, 6.0]).unwrap();
    let y = array![0usize, 0, 1, 1];

    let _ = GaussianNB::<f32>::new().fit(&x32, &y).unwrap().predict(&x32).unwrap();
    let _ = MultinomialNB::<f32>::new().fit(&x32, &y).unwrap().predict(&x32).unwrap();
    let _ = BernoulliNB::<f32>::new().fit(&x32, &y).unwrap().predict(&x32).unwrap();
    let _ = ComplementNB::<f32>::new().fit(&x32, &y).unwrap().predict(&x32).unwrap();

    let x_cat = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let _ = CategoricalNB::<f32>::new().fit(&x_cat, &y).unwrap().predict(&x_cat).unwrap();
}

// =============================================================================
// Cross-cutting: Default impls on every NB
// =============================================================================
#[test]
fn api_proof_default_impls() {
    let _: GaussianNB<f64> = Default::default();
    let _: MultinomialNB<f64> = Default::default();
    let _: BernoulliNB<f64> = Default::default();
    let _: ComplementNB<f64> = Default::default();
    let _: CategoricalNB<f64> = Default::default();
}

// =============================================================================
// Cross-cutting: Shape-mismatch errors are surfaced for every NB
// =============================================================================
#[test]
fn api_proof_shape_mismatch_errors() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 5.0, 6.0, 5.8, 6.2]).unwrap();
    let y = array![0usize, 0, 1, 1];
    let bad_x = Array2::from_shape_vec((2, 5), vec![0.0; 10]).unwrap();
    let bad_y = array![0usize, 0];

    let g = GaussianNB::<f64>::new().fit(&x, &y).unwrap();
    assert!(g.predict(&bad_x).is_err());
    assert!(g.predict_proba(&bad_x).is_err());
    assert!(g.predict_log_proba(&bad_x).is_err());
    assert!(g.predict_joint_log_proba(&bad_x).is_err());
    assert!(g.score(&bad_x, &bad_y).is_err());

    let m = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    assert!(m.predict(&bad_x).is_err());
    assert!(m.predict_proba(&bad_x).is_err());
    assert!(m.predict_log_proba(&bad_x).is_err());

    let b = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
    assert!(b.predict(&bad_x).is_err());
    assert!(b.predict_proba(&bad_x).is_err());

    let c = ComplementNB::<f64>::new().fit(&x, &y).unwrap();
    assert!(c.predict(&bad_x).is_err());
    assert!(c.predict_proba(&bad_x).is_err());

    let cat = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();
    assert!(cat.predict(&bad_x).is_err());
    assert!(cat.predict_proba(&bad_x).is_err());
}
