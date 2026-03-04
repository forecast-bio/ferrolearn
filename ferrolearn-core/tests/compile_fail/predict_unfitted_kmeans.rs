// Type-system proof: calling predict() on an unfitted KMeans must not compile.
// KMeans<F> implements Fit but not Predict. Only FittedKMeans<F> (returned by
// fit()) implements Predict.

use ferrolearn_cluster::KMeans;
use ferrolearn_core::Predict;
use ndarray::Array2;

fn main() {
    let model = KMeans::<f64>::new(3);
    let x = Array2::<f64>::zeros((6, 2));
    let _ = model.predict(&x);
}
