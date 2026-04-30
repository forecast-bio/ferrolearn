//! Proof-of-API integration test for ferrolearn-neural.
//!
//! Audit deliverable for crosslink #328 (under #252). Exercises every
//! public estimator after the move of MLP from ferrolearn-linear (#326)
//! and the addition of BernoulliRBM (#327).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neural::{Activation, BernoulliRBM, MLPClassifier, MLPRegressor, Solver};
use ndarray::{Array1, Array2, array};

fn sgd_solver() -> Solver<f64> {
    Solver::Sgd {
        learning_rate: 0.01,
        momentum: 0.9,
    }
}

fn adam_solver() -> Solver<f64> {
    Solver::Adam {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    }
}

#[test]
fn api_proof_mlp_classifier() {
    let x: Array2<f64> = array![
        [0.0, 0.0],
        [0.1, 0.2],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 9.9],
        [9.8, 10.2],
    ];
    let y = array![0usize, 0, 0, 1, 1, 1];

    for solver in [sgd_solver(), adam_solver()] {
        for activation in [
            Activation::Relu,
            Activation::Tanh,
            Activation::Logistic,
            Activation::Identity,
        ] {
            let clf = MLPClassifier::<f64>::new()
                .with_hidden_layer_sizes(vec![4])
                .with_activation(activation)
                .with_solver(solver)
                .with_max_iter(50)
                .with_tol(1e-4)
                .with_batch_size(2)
                .with_alpha(1e-4)
                .with_random_state(7);
            let fitted = clf.fit(&x, &y).unwrap();
            let preds = fitted.predict(&x).unwrap();
            assert_eq!(preds.len(), 6);
            let _proba = fitted.predict_proba(&x).unwrap();
            let _log_proba = fitted.predict_log_proba(&x).unwrap();
        }
    }

    // Default + early stopping
    let clf = MLPClassifier::<f64>::default()
        .with_hidden_layer_sizes(vec![4])
        .with_max_iter(20)
        .with_early_stopping(true)
        .with_validation_fraction(0.2)
        .with_random_state(7);
    let _ = clf.fit(&x, &y).unwrap();
}

#[test]
fn api_proof_mlp_regressor() {
    let x: Array2<f64> = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];

    for solver in [sgd_solver(), adam_solver()] {
        let reg = MLPRegressor::<f64>::new()
            .with_hidden_layer_sizes(vec![5])
            .with_activation(Activation::Relu)
            .with_solver(solver)
            .with_max_iter(100)
            .with_random_state(7);
        let fitted = reg.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 5);
    }
}

#[test]
fn api_proof_bernoulli_rbm() {
    let x: Array2<f64> = array![
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let rbm = BernoulliRBM::<f64>::new(2)
        .learning_rate(0.1)
        .n_iter(5)
        .batch_size(3)
        .random_state(7);
    let fitted = rbm.fit(&x, &()).unwrap();
    assert_eq!(fitted.components_.dim(), (2, 4));
    assert_eq!(fitted.intercept_hidden_.len(), 2);
    assert_eq!(fitted.intercept_visible_.len(), 4);
    assert_eq!(fitted.n_iter_, 5);

    let h = fitted.transform(&x).unwrap();
    assert_eq!(h.dim(), (6, 2));
    for v in h.iter() {
        assert!((0.0..=1.0).contains(v));
    }

    let v_recon = fitted.gibbs(&x).unwrap();
    assert_eq!(v_recon.dim(), (6, 4));

    use ferrolearn_core::traits::Transform;
    let h2 = <_ as Transform<Array2<f64>>>::transform(&fitted, &x).unwrap();
    assert_eq!(h2.dim(), (6, 2));
}
