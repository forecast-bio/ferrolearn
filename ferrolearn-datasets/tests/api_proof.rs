//! Proof-of-API integration test for ferrolearn-datasets.
//!
//! Audit deliverable for crosslink #323 (under #251). Exercises every
//! toy loader and every synthetic generator, including the 7 generators
//! added in #318 (make_friedman1/2/3, make_low_rank_matrix, make_spd_matrix,
//! make_sparse_spd_matrix, make_gaussian_quantiles, make_hastie_10_2,
//! make_multilabel_classification).

use ferrolearn_datasets::{
    load_breast_cancer, load_diabetes, load_digits, load_iris, load_linnerud, load_olivetti_faces,
    load_wine, make_blobs, make_circles, make_classification, make_friedman1, make_friedman2,
    make_friedman3, make_gaussian_quantiles, make_hastie_10_2, make_low_rank_matrix, make_moons,
    make_multilabel_classification, make_regression, make_s_curve, make_sparse_spd_matrix,
    make_sparse_uncorrelated, make_spd_matrix, make_swiss_roll,
};

#[test]
fn api_proof_toy_loaders() {
    let iris = load_iris::<f64>().unwrap();
    assert_eq!(iris.0.nrows(), 150);
    let wine = load_wine::<f64>().unwrap();
    assert_eq!(wine.0.nrows(), 178);
    let bc = load_breast_cancer::<f64>().unwrap();
    assert_eq!(bc.0.nrows(), 569);
    let dia = load_diabetes::<f64>().unwrap();
    assert_eq!(dia.0.nrows(), 442);
    let dig = load_digits::<f64>().unwrap();
    assert_eq!(dig.0.ncols(), 64);
    let lin = load_linnerud::<f64>().unwrap();
    assert_eq!(lin.0.nrows(), 20);
    let oli = load_olivetti_faces::<f64>().unwrap();
    assert_eq!(oli.0.nrows(), 400);
}

#[test]
fn api_proof_classic_generators() {
    let (x, y) = make_classification::<f64>(50, 5, 3, Some(42)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (50, 5));
    assert_eq!(y.len(), 50);

    let (x, y) = make_regression::<f64>(40, 4, 2, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 4));
    assert_eq!(y.len(), 40);

    let (x, y) = make_blobs::<f64>(30, 2, 3, 1.0, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (30, 2));
    assert_eq!(y.len(), 30);

    let (x, y) = make_moons::<f64>(40, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 2));
    assert_eq!(y.len(), 40);

    let (x, y) = make_circles::<f64>(40, 0.5, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 2));
    assert_eq!(y.len(), 40);

    let (x, t) = make_swiss_roll::<f64>(40, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 3));
    assert_eq!(t.len(), 40);

    let (x, t) = make_s_curve::<f64>(40, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 3));
    assert_eq!(t.len(), 40);

    let (x, y) = make_sparse_uncorrelated::<f64>(20, 5, Some(3)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (20, 5));
    assert_eq!(y.len(), 20);
}

#[test]
fn api_proof_friedman_family() {
    let (x, y) = make_friedman1::<f64>(50, 6, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (50, 6));
    assert_eq!(y.len(), 50);

    let (x, y) = make_friedman2::<f64>(50, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (50, 4));
    assert_eq!(y.len(), 50);

    let (x, y) = make_friedman3::<f64>(50, 0.1, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (50, 4));
    assert_eq!(y.len(), 50);

    // friedman1 requires n_features >= 5
    assert!(make_friedman1::<f64>(10, 4, 0.0, Some(0)).is_err());
}

#[test]
fn api_proof_matrix_generators() {
    let m = make_low_rank_matrix::<f64>(20, 10, 3, 0.5, Some(7)).unwrap();
    assert_eq!((m.nrows(), m.ncols()), (20, 10));

    let s = make_spd_matrix::<f64>(8, Some(7)).unwrap();
    assert_eq!((s.nrows(), s.ncols()), (8, 8));
    // symmetric check
    for i in 0..8 {
        for j in 0..8 {
            assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-9);
        }
    }

    let sp = make_sparse_spd_matrix::<f64>(8, 0.3, Some(7)).unwrap();
    assert_eq!((sp.nrows(), sp.ncols()), (8, 8));
}

#[test]
fn api_proof_classification_extras() {
    let (x, y) = make_gaussian_quantiles::<f64>(50, 4, 3, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (50, 4));
    assert_eq!(y.len(), 50);
    assert!(y.iter().all(|&v| v < 3));

    let (x, y) = make_hastie_10_2::<f64>(40, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (40, 10));
    assert_eq!(y.len(), 40);
    assert!(y.iter().all(|&v| v == 0 || v == 1));

    let (x, y) = make_multilabel_classification::<f64>(30, 5, 4, 2, Some(7)).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (30, 5));
    assert_eq!((y.nrows(), y.ncols()), (30, 4));
}
