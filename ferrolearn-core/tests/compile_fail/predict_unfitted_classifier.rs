// Type-system proof: calling predict() on an unfitted LogisticRegression must
// not compile. LogisticRegression<F> implements Fit but not Predict. Only
// FittedLogisticRegression<F> (returned by fit()) implements Predict.

use ferrolearn_core::Predict;
use ferrolearn_linear::LogisticRegression;
use ndarray::Array2;

fn main() {
    let model = LogisticRegression::<f64>::new();
    let x = Array2::<f64>::zeros((4, 2));
    let _ = model.predict(&x);
}
