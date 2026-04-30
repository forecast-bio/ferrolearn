mod classifiers;
mod clusterers;
mod conversions;
mod extras;
mod regressors;
mod transformers;

use pyo3::prelude::*;

#[pymodule]
fn _ferrolearn_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Original 12
    m.add_class::<regressors::RsLinearRegression>()?;
    m.add_class::<regressors::RsRidge>()?;
    m.add_class::<regressors::RsLasso>()?;
    m.add_class::<regressors::RsElasticNet>()?;
    m.add_class::<classifiers::RsLogisticRegression>()?;
    m.add_class::<classifiers::RsDecisionTreeClassifier>()?;
    m.add_class::<classifiers::RsRandomForestClassifier>()?;
    m.add_class::<classifiers::RsKNeighborsClassifier>()?;
    m.add_class::<classifiers::RsGaussianNB>()?;
    m.add_class::<transformers::RsStandardScaler>()?;
    m.add_class::<transformers::RsPCA>()?;
    m.add_class::<clusterers::RsKMeans>()?;

    // Extras — linear regressors
    m.add_class::<extras::RsBayesianRidge>()?;
    m.add_class::<extras::RsARDRegression>()?;
    m.add_class::<extras::RsHuberRegressor>()?;
    m.add_class::<extras::RsQuantileRegressor>()?;

    // Extras — tree regressors
    m.add_class::<extras::RsDecisionTreeRegressor>()?;
    m.add_class::<extras::RsRandomForestRegressor>()?;
    m.add_class::<extras::RsExtraTreesRegressor>()?;
    m.add_class::<extras::RsGradientBoostingRegressor>()?;
    m.add_class::<extras::RsHistGradientBoostingRegressor>()?;

    // Extras — neighbors regressors / kernel regressors
    m.add_class::<extras::RsKNeighborsRegressor>()?;
    m.add_class::<extras::RsKernelRidge>()?;

    // Extras — linear classifiers
    m.add_class::<extras::RsRidgeClassifier>()?;
    m.add_class::<extras::RsLinearSVC>()?;
    m.add_class::<extras::RsQDA>()?;

    // Extras — bayes classifiers
    m.add_class::<extras::RsMultinomialNB>()?;
    m.add_class::<extras::RsBernoulliNB>()?;
    m.add_class::<extras::RsComplementNB>()?;

    // Extras — tree classifiers
    m.add_class::<extras::RsExtraTreeClassifier>()?;
    m.add_class::<extras::RsExtraTreesClassifier>()?;
    m.add_class::<extras::RsAdaBoostClassifier>()?;
    m.add_class::<extras::RsGradientBoostingClassifier>()?;
    m.add_class::<extras::RsHistGradientBoostingClassifier>()?;
    m.add_class::<extras::RsBaggingClassifier>()?;

    // Extras — neighbors
    m.add_class::<extras::RsNearestCentroid>()?;

    // Extras — clusterers
    m.add_class::<extras::RsMiniBatchKMeans>()?;
    m.add_class::<extras::RsDBSCAN>()?;
    m.add_class::<extras::RsAgglomerativeClustering>()?;
    m.add_class::<extras::RsBirch>()?;
    m.add_class::<extras::RsGaussianMixture>()?;

    // Extras — decomp
    m.add_class::<extras::RsIncrementalPCA>()?;
    m.add_class::<extras::RsTruncatedSVD>()?;
    m.add_class::<extras::RsFastICA>()?;
    m.add_class::<extras::RsNMF>()?;
    m.add_class::<extras::RsKernelPCA>()?;
    m.add_class::<extras::RsSparsePCA>()?;
    m.add_class::<extras::RsFactorAnalysis>()?;

    // Extras — preprocess
    m.add_class::<extras::RsMinMaxScaler>()?;
    m.add_class::<extras::RsMaxAbsScaler>()?;
    m.add_class::<extras::RsRobustScaler>()?;
    m.add_class::<extras::RsPowerTransformer>()?;

    // Extras — kernel approx
    m.add_class::<extras::RsNystroem>()?;
    m.add_class::<extras::RsRBFSampler>()?;

    Ok(())
}
