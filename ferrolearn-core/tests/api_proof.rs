//! Proof-of-API integration test for ferrolearn-core.
//!
//! Audit deliverable for crosslink #323 (under #251). Exercises every
//! public trait, type, and re-export at the crate root: error types,
//! Dataset, Backend (default faer), introspection traits, Pipeline,
//! StreamingFitter (via a minimal in-test PartialFit implementor), and
//! the typed-pipeline builder entry point.

use ferrolearn_core::backend::Backend;
use ferrolearn_core::dataset::Dataset;
use ferrolearn_core::error::{FerroError, FerroResult};
use ferrolearn_core::introspection::{HasClasses, HasCoefficients, HasFeatureImportances};
use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::streaming::StreamingFitter;
use ferrolearn_core::traits::PartialFit;
use ferrolearn_core::typed_pipeline::TypedPipeline;
use ferrolearn_core::{DefaultBackend, FitTransform, Predict, Transform};
use ndarray::{Array1, Array2, array};

#[test]
fn api_proof_error_types() {
    let err = FerroError::ShapeMismatch {
        expected: vec![10],
        actual: vec![20],
        context: "api_proof".into(),
    };
    let _: FerroResult<()> = Err(err);

    let _ = FerroError::InsufficientSamples {
        required: 5,
        actual: 1,
        context: "api_proof".into(),
    };
    let _ = FerroError::ConvergenceFailure {
        iterations: 100,
        message: "did not converge".into(),
    };
    let _ = FerroError::InvalidParameter {
        name: "x".into(),
        reason: "too small".into(),
    };
    let _ = FerroError::NumericalInstability {
        message: "div by 0".into(),
    };
    let _ = FerroError::SerdeError {
        message: "bad json".into(),
    };
}

#[test]
fn api_proof_dataset_for_array2() {
    let x: Array2<f64> = Array2::zeros((10, 4));
    assert_eq!(x.n_samples(), 10);
    assert_eq!(x.n_features(), 4);
    assert!(!x.is_sparse());
    let x32: Array2<f32> = Array2::zeros((3, 5));
    assert_eq!(x32.n_samples(), 3);
}

#[test]
fn api_proof_default_backend_basic_ops() {
    let a: Array2<f64> = array![[2.0, 1.0], [1.0, 3.0]];
    let b: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
    let c = <DefaultBackend as Backend>::gemm(&a, &b).unwrap();
    assert_eq!(c, a);

    let (u, s, vt) = <DefaultBackend as Backend>::svd(&a).unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(vt.shape(), &[2, 2]);

    let (q, r) = <DefaultBackend as Backend>::qr(&a).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let chol = <DefaultBackend as Backend>::cholesky(&a).unwrap();
    assert_eq!(chol.shape(), &[2, 2]);

    let rhs: Array1<f64> = array![1.0, 1.0];
    let sol = <DefaultBackend as Backend>::solve(&a, &rhs).unwrap();
    assert_eq!(sol.len(), 2);

    let (vals, vecs) = <DefaultBackend as Backend>::eigh(&a).unwrap();
    assert_eq!(vals.len(), 2);
    assert_eq!(vecs.shape(), &[2, 2]);

    let _ = <DefaultBackend as Backend>::det(&a).unwrap();
    let inv = <DefaultBackend as Backend>::inv(&a).unwrap();
    assert_eq!(inv.shape(), &[2, 2]);
}

// Custom in-test classifier proving the introspection traits compile and
// resolve to expected behaviour.
struct DummyClf;
impl HasClasses for DummyClf {
    fn classes(&self) -> &[usize] {
        &[0, 1, 2]
    }
    fn n_classes(&self) -> usize {
        3
    }
}
struct DummyReg {
    coef: Array1<f64>,
    intercept: f64,
    importances: Array1<f64>,
}
impl HasCoefficients<f64> for DummyReg {
    fn coefficients(&self) -> &Array1<f64> {
        &self.coef
    }
    fn intercept(&self) -> f64 {
        self.intercept
    }
}
impl HasFeatureImportances<f64> for DummyReg {
    fn feature_importances(&self) -> &Array1<f64> {
        &self.importances
    }
}

#[test]
fn api_proof_introspection_traits() {
    let clf = DummyClf;
    assert_eq!(clf.classes(), &[0, 1, 2]);
    assert_eq!(clf.n_classes(), 3);

    let reg = DummyReg {
        coef: array![1.0, 2.0, 3.0],
        intercept: 0.5,
        importances: array![0.5, 0.3, 0.2],
    };
    assert_eq!(reg.coefficients().len(), 3);
    assert!((reg.intercept() - 0.5).abs() < 1e-12);
    assert_eq!(reg.feature_importances().len(), 3);
}

#[test]
fn api_proof_pipeline_construction() {
    // Empty pipeline is valid to construct; fit will fail without an estimator.
    let pipe = Pipeline::<f64>::new();
    let x: Array2<f64> = Array2::zeros((4, 2));
    let y: Array1<f64> = Array1::zeros(4);
    use ferrolearn_core::Fit;
    let _ = pipe.fit(&x, &y).is_err();

    // TypedPipeline builder root.
    let _ = TypedPipeline::new();
}

// Minimal in-test PartialFit model used to prove StreamingFitter compiles.
// PartialFit requires FitResult: Predict<X> + PartialFit<X, Y>, so the
// model must also implement Predict<Vec<f64>>.
#[derive(Default)]
struct SumModel {
    total: f64,
}
impl PartialFit<Vec<f64>, Vec<f64>> for SumModel {
    type FitResult = SumModel;
    type Error = FerroError;

    fn partial_fit(mut self, x: &Vec<f64>, _y: &Vec<f64>) -> Result<Self::FitResult, Self::Error> {
        self.total += x.iter().sum::<f64>();
        Ok(self)
    }
}
impl Predict<Vec<f64>> for SumModel {
    type Output = f64;
    type Error = FerroError;
    fn predict(&self, _x: &Vec<f64>) -> Result<f64, FerroError> {
        Ok(self.total)
    }
}

#[test]
fn api_proof_streaming_fitter() {
    let model = SumModel::default();
    let batches: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![1.0, 2.0], vec![]),
        (vec![3.0], vec![]),
        (vec![4.0], vec![]),
    ];
    let fitted = StreamingFitter::new(model)
        .n_epochs(2)
        .fit_batches(batches)
        .unwrap();
    // 2 epochs of (1+2)+3+4 = 10 each → 20
    assert!((fitted.total - 20.0).abs() < 1e-12);

    let model2 = SumModel::default();
    let batches2: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![1.0, 2.0, 3.0], vec![])];
    let fitted2 = StreamingFitter::new(model2)
        .fit_batches_single_epoch(batches2)
        .unwrap();
    assert!((fitted2.total - 6.0).abs() < 1e-12);
}

// Touch the FitTransform / Transform / Predict re-exports.
#[test]
fn api_proof_trait_reexports_in_scope() {
    // Just need them in scope to validate they are pub-re-exported at crate
    // root. The trait bounds are real (not stubs) and parametrise over
    // the input type; we instantiate them with `Vec<f64>` to prove the
    // re-export resolves.
    fn _take_predict<P: Predict<Vec<f64>, Output = f64, Error = FerroError>>(p: &P) {
        let _ = p.predict(&vec![1.0]);
    }
    fn _take_transform<T: Transform<Vec<f64>, Output = Vec<f64>, Error = FerroError>>(t: &T) {
        let _ = t.transform(&vec![1.0]);
    }
    fn _take_fit_transform<T: FitTransform<Vec<f64>, Output = Vec<f64>, Error = FerroError>>(
        t: &T,
    ) {
        let _ = t.transform(&vec![1.0]);
    }
}
