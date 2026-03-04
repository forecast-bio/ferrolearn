// Type-system proof: calling predict() on an unfitted LinearRegression must
// not compile. LinearRegression<F> implements Fit but not Predict. Only
// FittedLinearRegression<F> (returned by fit()) implements Predict.

use ferrolearn_core::Predict;
use ferrolearn_linear::LinearRegression;
use ndarray::Array2;

fn main() {
    let model = LinearRegression::<f64>::new();
    let x = Array2::<f64>::zeros((3, 2));
    let _ = model.predict(&x);
}
