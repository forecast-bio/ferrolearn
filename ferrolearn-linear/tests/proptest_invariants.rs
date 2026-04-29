//! Property-based tests for linear model mathematical invariants.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::{ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for generating regression data with n_samples > n_features,
/// non-constant y, and values in a reasonable range.
fn regression_data_strategy() -> impl Strategy<Value = (Array2<f64>, Array1<f64>, usize, usize)> {
    // n_features in 2..=4, n_samples in n_features+2..=n_features+10
    (2usize..=4usize)
        .prop_flat_map(|n_feat| {
            let min_samples = n_feat + 2;
            let max_samples = n_feat + 10;
            (min_samples..=max_samples, Just(n_feat))
        })
        .prop_flat_map(|(n_samples, n_feat)| {
            let x_strat = proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat);
            let y_strat = proptest::collection::vec(-10.0..10.0f64, n_samples);
            (x_strat, y_strat, Just(n_samples), Just(n_feat))
        })
        .prop_map(|(x_data, y_data, n_samples, n_feat)| {
            let x = Array2::from_shape_vec((n_samples, n_feat), x_data).unwrap();
            let y = Array1::from_vec(y_data);
            (x, y, n_samples, n_feat)
        })
}

/// Strategy for generating binary classification data (2 classes).
fn binary_classification_data_strategy() -> impl Strategy<Value = (Array2<f64>, Array1<usize>)> {
    (6usize..=12usize, 2usize..=4usize)
        .prop_flat_map(|(n_samples, n_feat)| {
            let half = n_samples / 2;
            let x_strat = proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat);
            // Ensure at least 1 sample per class by constructing labels manually
            (x_strat, Just(n_samples), Just(n_feat), Just(half))
        })
        .prop_map(|(x_data, n_samples, n_feat, half)| {
            let x = Array2::from_shape_vec((n_samples, n_feat), x_data).unwrap();
            let mut labels = vec![0usize; half];
            labels.extend(vec![1usize; n_samples - half]);
            let y = Array1::from_vec(labels);
            (x, y)
        })
}

/// Strategy for multiclass classification data (3 classes).
fn multiclass_data_strategy() -> impl Strategy<Value = (Array2<f64>, Array1<usize>)> {
    (9usize..=15usize, 2usize..=3usize)
        .prop_flat_map(|(n_samples, n_feat)| {
            let third = n_samples / 3;
            let x_strat = proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat);
            (x_strat, Just(n_samples), Just(n_feat), Just(third))
        })
        .prop_map(|(x_data, n_samples, n_feat, third)| {
            let x = Array2::from_shape_vec((n_samples, n_feat), x_data).unwrap();
            let mut labels = vec![0usize; third];
            labels.extend(vec![1usize; third]);
            labels.extend(vec![2usize; n_samples - 2 * third]);
            let y = Array1::from_vec(labels);
            (x, y)
        })
}

// ---------------------------------------------------------------------------
// LinearRegression invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn linear_regression_coef_len_equals_n_features(
        (x, y, _n_samples, n_feat) in regression_data_strategy()
    ) {
        let model = LinearRegression::<f64>::new();
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        prop_assert_eq!(fitted.coefficients().len(), n_feat);
    }

    #[test]
    fn ridge_coef_len_equals_n_features(
        (x, y, _n_samples, n_feat) in regression_data_strategy()
    ) {
        let model = Ridge::<f64>::new().with_alpha(1.0);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        prop_assert_eq!(fitted.coefficients().len(), n_feat);
    }

    #[test]
    fn lasso_coef_len_equals_n_features(
        (x, y, _n_samples, n_feat) in regression_data_strategy()
    ) {
        let model = Lasso::<f64>::new().with_alpha(0.1);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        prop_assert_eq!(fitted.coefficients().len(), n_feat);
    }

    #[test]
    fn elastic_net_coef_len_equals_n_features(
        (x, y, _n_samples, n_feat) in regression_data_strategy()
    ) {
        let model = ElasticNet::<f64>::new().with_alpha(0.1).with_l1_ratio(0.5);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        prop_assert_eq!(fitted.coefficients().len(), n_feat);
    }

    #[test]
    fn ridge_alpha_zero_matches_ols(
        (x, y, _n_samples, _n_feat) in regression_data_strategy()
    ) {
        let ols = LinearRegression::<f64>::new();
        let ridge = Ridge::<f64>::new().with_alpha(0.0);

        let ols_result = ols.fit(&x, &y);
        let ridge_result = ridge.fit(&x, &y);

        prop_assume!(ols_result.is_ok() && ridge_result.is_ok());

        let ols_fitted = ols_result.unwrap();
        let ridge_fitted = ridge_result.unwrap();

        let ols_preds = ols_fitted.predict(&x).unwrap();
        let ridge_preds = ridge_fitted.predict(&x).unwrap();

        for (o, r) in ols_preds.iter().zip(ridge_preds.iter()) {
            prop_assert!((o - r).abs() < 0.1,
                "OLS pred {} vs Ridge(alpha=0) pred {} differ by more than 0.1", o, r);
        }
    }

    #[test]
    fn lasso_high_alpha_sparser_than_ols(
        (x, y, _n_samples, _n_feat) in regression_data_strategy()
    ) {
        let ols = LinearRegression::<f64>::new();
        let lasso = Lasso::<f64>::new().with_alpha(5.0);

        let ols_result = ols.fit(&x, &y);
        let lasso_result = lasso.fit(&x, &y);

        prop_assume!(ols_result.is_ok() && lasso_result.is_ok());

        let ols_fitted = ols_result.unwrap();
        let lasso_fitted = lasso_result.unwrap();

        // Count number of zero (or near-zero) coefficients
        let ols_zeros = ols_fitted.coefficients().iter().filter(|c| c.abs() < 1e-8).count();
        let lasso_zeros = lasso_fitted.coefficients().iter().filter(|c| c.abs() < 1e-8).count();

        prop_assert!(lasso_zeros >= ols_zeros,
            "Lasso zeros ({}) should be >= OLS zeros ({})", lasso_zeros, ols_zeros);
    }
}

// ---------------------------------------------------------------------------
// LogisticRegression invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn logistic_predict_labels_subset_of_training_labels_binary(
        (x, y) in binary_classification_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(200);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        let mut train_labels: Vec<usize> = y.to_vec();
        train_labels.sort_unstable();
        train_labels.dedup();

        for &p in &preds {
            prop_assert!(train_labels.contains(&p),
                "Predicted label {} not in training labels {:?}", p, train_labels);
        }
    }

    #[test]
    fn logistic_predict_proba_rows_sum_to_one_binary(
        (x, y) in binary_classification_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(200);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            prop_assert!((row_sum - 1.0).abs() < 1e-10,
                "Row {} sum {} != 1.0", i, row_sum);
        }
    }

    #[test]
    fn logistic_predict_proba_values_in_01_binary(
        (x, y) in binary_classification_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(200);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for &val in &proba {
            prop_assert!((0.0..=1.0).contains(&val),
                "Probability {} outside [0, 1]", val);
        }
    }

    #[test]
    fn logistic_predict_matches_argmax_proba_binary(
        (x, y) in binary_classification_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(200);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        let classes = fitted.classes();

        for i in 0..proba.nrows() {
            let mut best_idx = 0;
            let mut best_val = proba[[i, 0]];
            for c in 1..proba.ncols() {
                if proba[[i, c]] > best_val {
                    best_val = proba[[i, c]];
                    best_idx = c;
                }
            }
            prop_assert_eq!(preds[i], classes[best_idx],
                "predict[{}]={} != argmax(predict_proba)={}", i, preds[i], classes[best_idx]);
        }
    }

    #[test]
    fn logistic_predict_proba_rows_sum_to_one_multiclass(
        (x, y) in multiclass_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(500);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            prop_assert!((row_sum - 1.0).abs() < 1e-10,
                "Row {} sum {} != 1.0", i, row_sum);
        }
    }

    #[test]
    fn logistic_predict_proba_values_in_01_multiclass(
        (x, y) in multiclass_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(500);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        for &val in &proba {
            prop_assert!((0.0..=1.0).contains(&val),
                "Probability {} outside [0, 1]", val);
        }
    }
}

use ferrolearn_core::introspection::HasClasses;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn logistic_predict_labels_subset_of_training_labels_multiclass(
        (x, y) in multiclass_data_strategy()
    ) {
        let model = LogisticRegression::<f64>::new().with_c(1.0).with_max_iter(500);
        let result = model.fit(&x, &y);
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let preds = fitted.predict(&x).unwrap();

        let classes = fitted.classes();

        for &p in &preds {
            prop_assert!(classes.contains(&p),
                "Predicted label {} not in training classes {:?}", p, classes);
        }
    }
}
