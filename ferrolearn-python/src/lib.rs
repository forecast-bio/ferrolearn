mod classifiers;
mod clusterers;
mod conversions;
mod regressors;
mod transformers;

use pyo3::prelude::*;

#[pymodule]
fn _ferrolearn_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Regressors
    m.add_class::<regressors::RsLinearRegression>()?;
    m.add_class::<regressors::RsRidge>()?;
    m.add_class::<regressors::RsLasso>()?;
    m.add_class::<regressors::RsElasticNet>()?;

    // Classifiers
    m.add_class::<classifiers::RsLogisticRegression>()?;
    m.add_class::<classifiers::RsDecisionTreeClassifier>()?;
    m.add_class::<classifiers::RsRandomForestClassifier>()?;
    m.add_class::<classifiers::RsKNeighborsClassifier>()?;
    m.add_class::<classifiers::RsGaussianNB>()?;

    // Transformers
    m.add_class::<transformers::RsStandardScaler>()?;
    m.add_class::<transformers::RsPCA>()?;

    // Clusterers
    m.add_class::<clusterers::RsKMeans>()?;

    Ok(())
}
