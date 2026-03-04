// Type-system proof: calling transform() on an unfitted PCA must not compile.
// PCA<F> implements Fit but not Transform. Only FittedPCA<F> (returned by
// fit()) implements Transform.

use ferrolearn_core::Transform;
use ferrolearn_decomp::PCA;
use ndarray::Array2;

fn main() {
    let pca = PCA::<f64>::new(2);
    let x = Array2::<f64>::zeros((5, 3));
    let _ = pca.transform(&x);
}
