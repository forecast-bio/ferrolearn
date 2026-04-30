//! Proof-of-API integration test for ferrolearn-covariance.
//!
//! Audit deliverable for crosslink #328 (under #252). Exercises every
//! public estimator and helper after the move from ferrolearn-decomp
//! (#324) and the GraphicalLasso + function-style additions (#325).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_covariance::{
    EllipticEnvelope, EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, LedoitWolf, MinCovDet,
    OAS, ShrunkCovariance, empirical_covariance, fast_mcd, graphical_lasso, ledoit_wolf,
    ledoit_wolf_shrinkage, log_likelihood, oas, shrunk_covariance,
};
use ndarray::{Array2, array};

fn data() -> Array2<f64> {
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
fn api_proof_empirical_and_shrunk() {
    let est = EmpiricalCovariance::<f64>::new();
    let fitted = est.fit(&data(), &()).unwrap();
    assert_eq!(fitted.covariance().dim(), (3, 3));
    assert_eq!(fitted.location().len(), 3);

    let est2 = EmpiricalCovariance::<f64>::new().assume_centered(true);
    let _ = est2.fit(&data(), &()).unwrap();

    let s = ShrunkCovariance::<f64>::new(0.1);
    let _ = s.fit(&data(), &()).unwrap();
}

#[test]
fn api_proof_ledoit_wolf_oas() {
    let lw = LedoitWolf::<f64>::new();
    let fitted = lw.fit(&data(), &()).unwrap();
    assert!(fitted.shrinkage() >= 0.0 && fitted.shrinkage() <= 1.0);
    assert_eq!(fitted.covariance().dim(), (3, 3));

    let oas_est = OAS::<f64>::new();
    let f2 = oas_est.fit(&data(), &()).unwrap();
    assert!(f2.shrinkage() >= 0.0 && f2.shrinkage() <= 1.0);
}

#[test]
fn api_proof_mcd_and_envelope() {
    let mcd = MinCovDet::<f64>::new()
        .support_fraction(0.75)
        .random_state(7);
    let f = mcd.fit(&data(), &()).unwrap();
    assert_eq!(f.covariance().dim(), (3, 3));
    assert_eq!(f.location().len(), 3);
    assert_eq!(f.support().len(), 10);

    let env = EllipticEnvelope::<f64>::new()
        .support_fraction(0.75)
        .random_state(7);
    let fenv = env.fit(&data(), &()).unwrap();
    let preds = fenv.predict(&data()).unwrap();
    assert_eq!(preds.len(), 10);
    assert!(preds.iter().all(|&v| v == 1 || v == -1));
}

#[test]
fn api_proof_helpers() {
    let cov = empirical_covariance(&data(), false).unwrap();
    assert_eq!(cov.dim(), (3, 3));

    let s = shrunk_covariance(&cov, 0.4).unwrap();
    assert_eq!(s.dim(), (3, 3));

    let (lw_cov, lw_s) = ledoit_wolf(&data(), false).unwrap();
    assert_eq!(lw_cov.dim(), (3, 3));
    let lw_s2 = ledoit_wolf_shrinkage(&data(), false).unwrap();
    assert!((lw_s - lw_s2).abs() < 1e-12);

    let (oas_cov, _oas_s) = oas(&data(), false).unwrap();
    assert_eq!(oas_cov.dim(), (3, 3));

    // log_likelihood: identity emp_cov + identity precision
    let mut id = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        id[[i, i]] = 1.0;
    }
    let _ = log_likelihood(&id, &id).unwrap();

    let (loc, mc, supp) = fast_mcd(&data(), 0.75, Some(7)).unwrap();
    assert_eq!(loc.len(), 3);
    assert_eq!(mc.dim(), (3, 3));
    assert_eq!(supp.len(), 10);
}

#[test]
fn api_proof_graphical_lasso() {
    let gl = GraphicalLasso::<f64>::new(0.1)
        .max_iter(50)
        .max_inner_iter(50)
        .tol(1e-3)
        .assume_centered(false);
    let f = gl.fit(&data(), &()).unwrap();
    assert_eq!(f.covariance().dim(), (3, 3));
    assert_eq!(f.precision().dim(), (3, 3));
    assert!(f.n_iter() > 0);

    let emp = empirical_covariance(&data(), false).unwrap();
    let (cov, prec) = graphical_lasso(&emp, 0.1, 50, 1e-3).unwrap();
    assert_eq!(cov.dim(), (3, 3));
    assert_eq!(prec.dim(), (3, 3));

    let glcv = GraphicalLassoCV::<f64>::new(vec![0.05, 0.1, 0.2])
        .n_folds(2)
        .max_iter(20)
        .tol(1e-2);
    let fcv = glcv.fit(&data(), &()).unwrap();
    assert_eq!(fcv.covariance().dim(), (3, 3));
    assert_eq!(fcv.cv_scores().len(), 3);
    assert!([0.05, 0.1, 0.2].contains(&fcv.best_alpha()));
}
