//! Additional PyO3 bindings (Phase 2 binding expansion) — covers everything
//! beyond the 12 originally-bound estimators so the head-to-head bench can
//! exercise the full ferrolearn surface against scikit-learn.

#![allow(non_snake_case)]

use crate::conversions::*;
use ferrolearn_core::{Fit, Predict, Transform};
use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ===========================================================================
// Linear regressors (extras)
// ===========================================================================

macro_rules! py_regressor {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let y_nd = numpy1_to_ndarray(y);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &y_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray1<f64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let preds = fitted.predict(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray1_to_numpy(py, &preds))
            }
        }
    };
}

macro_rules! py_classifier {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let y_nd = numpy1_to_ndarray_usize(y);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &y_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray1<i64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let preds = fitted.predict(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray1_usize_to_numpy(py, &preds))
            }
        }
    };
}

macro_rules! py_transformer {
    (
        $cls_name:ident, $py_name:literal, $fitted_path:path,
        ($($field:ident : $ty:ty = $default:expr),* $(,)?),
        $build_block:block
    ) => {
        #[pyclass(name = $py_name)]
        pub struct $cls_name {
            $($field: $ty,)*
            fitted: Option<$fitted_path>,
        }

        #[pymethods]
        impl $cls_name {
            #[new]
            #[pyo3(signature = ($($field = $default),*))]
            fn new($($field: $ty),*) -> Self {
                Self { $($field,)* fitted: None }
            }

            fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
                $(let $field = self.$field.clone();)*
                let x_nd = numpy2_to_ndarray(x);
                let model = $build_block;
                let fitted = model.fit(&x_nd, &())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                self.fitted = Some(fitted);
                Ok(())
            }

            fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>)
                -> PyResult<Bound<'py, PyArray2<f64>>>
            {
                let fitted = self.fitted.as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
                let x_nd = numpy2_to_ndarray(x);
                let xt = fitted.transform(&x_nd)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(ndarray2_to_numpy(py, &xt))
            }
        }
    };
}

// ===========================================================================
// Linear regressors
// ===========================================================================

py_regressor!(
    RsBayesianRidge, "_RsBayesianRidge",
    ferrolearn_linear::FittedBayesianRidge<f64>,
    (max_iter: usize = 300, tol: f64 = 1e-3, fit_intercept: bool = true),
    {
        ferrolearn_linear::BayesianRidge::<f64>::new()
            .with_max_iter(max_iter).with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

py_regressor!(
    RsARDRegression, "_RsARDRegression",
    ferrolearn_linear::FittedARDRegression<f64>,
    (max_iter: usize = 300, tol: f64 = 1e-3, fit_intercept: bool = true),
    {
        ferrolearn_linear::ARDRegression::<f64>::new()
            .with_max_iter(max_iter).with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

py_regressor!(
    RsHuberRegressor, "_RsHuberRegressor",
    ferrolearn_linear::FittedHuberRegressor<f64>,
    (epsilon: f64 = 1.35, alpha: f64 = 0.0001, max_iter: usize = 100,
     tol: f64 = 1e-5, fit_intercept: bool = true),
    {
        ferrolearn_linear::HuberRegressor::<f64>::new()
            .with_epsilon(epsilon).with_alpha(alpha).with_max_iter(max_iter)
            .with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

py_regressor!(
    RsQuantileRegressor, "_RsQuantileRegressor",
    ferrolearn_linear::FittedQuantileRegressor<f64>,
    (quantile: f64 = 0.5, alpha: f64 = 1.0, max_iter: usize = 10000,
     tol: f64 = 1e-6, fit_intercept: bool = true),
    {
        ferrolearn_linear::QuantileRegressor::<f64>::new()
            .with_quantile(quantile).with_alpha(alpha).with_max_iter(max_iter)
            .with_tol(tol).with_fit_intercept(fit_intercept)
    }
);

// ===========================================================================
// Tree regressors
// ===========================================================================

py_regressor!(
    RsDecisionTreeRegressor, "_RsDecisionTreeRegressor",
    ferrolearn_tree::FittedDecisionTreeRegressor<f64>,
    (max_depth: Option<usize> = None, min_samples_split: usize = 2,
     min_samples_leaf: usize = 1),
    {
        ferrolearn_tree::DecisionTreeRegressor::<f64>::new()
            .with_max_depth(max_depth).with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
    }
);

#[pyclass(name = "_RsRandomForestRegressor")]
pub struct RsRandomForestRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedRandomForestRegressor<f64>>,
}

#[pymethods]
impl RsRandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, random_state=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>, min_samples_split: usize,
           min_samples_leaf: usize, random_state: Option<u64>) -> Self {
        Self { n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::RandomForestRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsExtraTreesRegressor")]
pub struct RsExtraTreesRegressor {
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedExtraTreesRegressor<f64>>,
}

#[pymethods]
impl RsExtraTreesRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::ExtraTreesRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsGradientBoostingRegressor")]
pub struct RsGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedGradientBoostingRegressor<f64>>,
}

#[pymethods]
impl RsGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3), random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, learning_rate, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsHistGradientBoostingRegressor")]
pub struct RsHistGradientBoostingRegressor {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedHistGradientBoostingRegressor<f64>>,
}

#[pymethods]
impl RsHistGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, learning_rate, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let mut m = ferrolearn_tree::HistGradientBoostingRegressor::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
    }
}

py_regressor!(
    RsKNeighborsRegressor, "_RsKNeighborsRegressor",
    ferrolearn_neighbors::FittedKNeighborsRegressor<f64>,
    (n_neighbors: usize = 5),
    {
        ferrolearn_neighbors::KNeighborsRegressor::<f64>::new()
            .with_n_neighbors(n_neighbors)
    }
);

py_regressor!(
    RsKernelRidge, "_RsKernelRidge",
    ferrolearn_kernel::FittedKernelRidge<f64>,
    (alpha: f64 = 1.0),
    {
        ferrolearn_kernel::KernelRidge::<f64>::new().with_alpha(alpha)
    }
);

// ===========================================================================
// Linear classifiers
// ===========================================================================

py_classifier!(
    RsRidgeClassifier, "_RsRidgeClassifier",
    ferrolearn_linear::FittedRidgeClassifier<f64>,
    (alpha: f64 = 1.0, fit_intercept: bool = true),
    {
        ferrolearn_linear::RidgeClassifier::<f64>::new()
            .with_alpha(alpha).with_fit_intercept(fit_intercept)
    }
);

py_classifier!(
    RsLinearSVC, "_RsLinearSVC",
    ferrolearn_linear::FittedLinearSVC<f64>,
    (c: f64 = 1.0, max_iter: usize = 1000, tol: f64 = 1e-4),
    {
        ferrolearn_linear::LinearSVC::<f64>::new()
            .with_c(c).with_max_iter(max_iter).with_tol(tol)
    }
);

py_classifier!(
    RsQDA, "_RsQDA",
    ferrolearn_linear::FittedQDA<f64>,
    (reg_param: f64 = 0.0),
    {
        ferrolearn_linear::QDA::<f64>::new().with_reg_param(reg_param)
    }
);

// ===========================================================================
// Bayes (extras)
// ===========================================================================

py_classifier!(
    RsMultinomialNB, "_RsMultinomialNB",
    ferrolearn_bayes::FittedMultinomialNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true),
    {
        ferrolearn_bayes::MultinomialNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior)
    }
);

py_classifier!(
    RsBernoulliNB, "_RsBernoulliNB",
    ferrolearn_bayes::FittedBernoulliNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, binarize: f64 = 0.0),
    {
        ferrolearn_bayes::BernoulliNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_binarize(binarize)
    }
);

py_classifier!(
    RsComplementNB, "_RsComplementNB",
    ferrolearn_bayes::FittedComplementNB<f64>,
    (alpha: f64 = 1.0, fit_prior: bool = true, norm: bool = false),
    {
        ferrolearn_bayes::ComplementNB::<f64>::new()
            .with_alpha(alpha).with_fit_prior(fit_prior).with_norm(norm)
    }
);

// ===========================================================================
// Tree classifiers (extras)
// ===========================================================================

py_classifier!(
    RsExtraTreeClassifier, "_RsExtraTreeClassifier",
    ferrolearn_tree::FittedExtraTreeClassifier<f64>,
    (max_depth: Option<usize> = None, min_samples_split: usize = 2,
     min_samples_leaf: usize = 1),
    {
        ferrolearn_tree::ExtraTreeClassifier::<f64>::new()
            .with_max_depth(max_depth).with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
    }
);

#[pyclass(name = "_RsExtraTreesClassifier")]
pub struct RsExtraTreesClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedExtraTreesClassifier<f64>>,
}

#[pymethods]
impl RsExtraTreesClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::ExtraTreesClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsAdaBoostClassifier")]
pub struct RsAdaBoostClassifier {
    n_estimators: usize,
    learning_rate: f64,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedAdaBoostClassifier<f64>>,
}

#[pymethods]
impl RsAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, random_state: Option<u64>) -> Self {
        Self { n_estimators, learning_rate, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::AdaBoostClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsGradientBoostingClassifier")]
pub struct RsGradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedGradientBoostingClassifier<f64>>,
}

#[pymethods]
impl RsGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=Some(3), random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, learning_rate, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsHistGradientBoostingClassifier")]
pub struct RsHistGradientBoostingClassifier {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedHistGradientBoostingClassifier<f64>>,
}

#[pymethods]
impl RsHistGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, max_depth: Option<usize>, random_state: Option<u64>) -> Self {
        Self { n_estimators, learning_rate, max_depth, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::HistGradientBoostingClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

#[pyclass(name = "_RsBaggingClassifier")]
pub struct RsBaggingClassifier {
    n_estimators: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedBaggingClassifier<f64>>,
}

#[pymethods]
impl RsBaggingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=10, random_state=None))]
    fn new(n_estimators: usize, random_state: Option<u64>) -> Self {
        Self { n_estimators, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut m = ferrolearn_tree::BaggingClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

py_classifier!(
    RsNearestCentroid, "_RsNearestCentroid",
    ferrolearn_neighbors::FittedNearestCentroid<f64>,
    (),
    {
        ferrolearn_neighbors::NearestCentroid::<f64>::new()
    }
);

// ===========================================================================
// Cluster (extras) — these don't fit the supervised pattern; predict is fitted.labels_
// ===========================================================================

#[pyclass(name = "_RsMiniBatchKMeans")]
pub struct RsMiniBatchKMeans {
    n_clusters: usize,
    max_iter: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedMiniBatchKMeans<f64>>,
}

#[pymethods]
impl RsMiniBatchKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=100, random_state=None))]
    fn new(n_clusters: usize, max_iter: usize, random_state: Option<u64>) -> Self {
        Self { n_clusters, max_iter, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::MiniBatchKMeans::<f64>::new(self.n_clusters)
            .with_max_iter(self.max_iter);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

#[pyclass(name = "_RsDBSCAN")]
pub struct RsDBSCAN {
    eps: f64,
    min_samples: usize,
    fitted: Option<ferrolearn_cluster::FittedDBSCAN<f64>>,
}

#[pymethods]
impl RsDBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let m = ferrolearn_cluster::DBSCAN::<f64>::new(self.eps).with_min_samples(self.min_samples);
        let fitted = m.fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let lbls = f.labels();
        let arr: Array1<i64> = lbls.mapv(|v| v as i64);
        Ok(PyArray1::from_array(py, &arr))
    }
}

#[pyclass(name = "_RsAgglomerativeClustering")]
pub struct RsAgglomerativeClustering {
    n_clusters: usize,
    fitted: Option<ferrolearn_cluster::FittedAgglomerativeClustering<f64>>,
}

#[pymethods]
impl RsAgglomerativeClustering {
    #[new]
    #[pyo3(signature = (n_clusters=2))]
    fn new(n_clusters: usize) -> Self {
        Self { n_clusters, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let m = ferrolearn_cluster::AgglomerativeClustering::<f64>::new(self.n_clusters);
        let fitted = m.fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

#[pyclass(name = "_RsBirch")]
pub struct RsBirch {
    n_clusters: Option<usize>,
    threshold: f64,
    fitted: Option<ferrolearn_cluster::FittedBirch<f64>>,
}

#[pymethods]
impl RsBirch {
    #[new]
    #[pyo3(signature = (n_clusters=None, threshold=0.5))]
    fn new(n_clusters: Option<usize>, threshold: f64) -> Self {
        Self { n_clusters, threshold, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::Birch::<f64>::new().with_threshold(self.threshold);
        if let Some(n) = self.n_clusters { m = m.with_n_clusters(n); }
        let fitted = m.fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, f.labels()))
    }
}

#[pyclass(name = "_RsGaussianMixture")]
pub struct RsGaussianMixture {
    n_components: usize,
    max_iter: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedGaussianMixture<f64>>,
}

#[pymethods]
impl RsGaussianMixture {
    #[new]
    #[pyo3(signature = (n_components=1, max_iter=100, random_state=None))]
    fn new(n_components: usize, max_iter: usize, random_state: Option<u64>) -> Self {
        Self { n_components, max_iter, random_state, fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut m = ferrolearn_cluster::GaussianMixture::<f64>::new(self.n_components)
            .with_max_iter(self.max_iter);
        if let Some(s) = self.random_state { m = m.with_random_state(s); }
        let fitted = m.fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let f = self.fitted.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = f.predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }
}

// ===========================================================================
// Decomp (extras)
// ===========================================================================

py_transformer!(
    RsIncrementalPCA, "_RsIncrementalPCA",
    ferrolearn_decomp::FittedIncrementalPCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::IncrementalPCA::<f64>::new(n_components) }
);

py_transformer!(
    RsTruncatedSVD, "_RsTruncatedSVD",
    ferrolearn_decomp::FittedTruncatedSVD<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::TruncatedSVD::<f64>::new(n_components) }
);

py_transformer!(
    RsFastICA, "_RsFastICA",
    ferrolearn_decomp::FittedFastICA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::FastICA::<f64>::new(n_components) }
);

py_transformer!(
    RsNMF, "_RsNMF",
    ferrolearn_decomp::FittedNMF<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::NMF::<f64>::new(n_components) }
);

py_transformer!(
    RsKernelPCA, "_RsKernelPCA",
    ferrolearn_decomp::FittedKernelPCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::KernelPCA::<f64>::new(n_components) }
);

py_transformer!(
    RsSparsePCA, "_RsSparsePCA",
    ferrolearn_decomp::FittedSparsePCA<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::SparsePCA::<f64>::new(n_components) }
);

py_transformer!(
    RsFactorAnalysis, "_RsFactorAnalysis",
    ferrolearn_decomp::FittedFactorAnalysis<f64>,
    (n_components: usize = 2),
    { ferrolearn_decomp::FactorAnalysis::<f64>::new(n_components) }
);

// ===========================================================================
// Preprocess (extras)
// ===========================================================================

py_transformer!(
    RsMinMaxScaler, "_RsMinMaxScaler",
    ferrolearn_preprocess::FittedMinMaxScaler<f64>,
    (),
    { ferrolearn_preprocess::MinMaxScaler::<f64>::new() }
);

py_transformer!(
    RsMaxAbsScaler, "_RsMaxAbsScaler",
    ferrolearn_preprocess::FittedMaxAbsScaler<f64>,
    (),
    { ferrolearn_preprocess::MaxAbsScaler::<f64>::new() }
);

py_transformer!(
    RsRobustScaler, "_RsRobustScaler",
    ferrolearn_preprocess::FittedRobustScaler<f64>,
    (),
    { ferrolearn_preprocess::RobustScaler::<f64>::new() }
);

py_transformer!(
    RsPowerTransformer, "_RsPowerTransformer",
    ferrolearn_preprocess::FittedPowerTransformer<f64>,
    (),
    { ferrolearn_preprocess::PowerTransformer::<f64>::new() }
);

py_transformer!(
    RsNystroem, "_RsNystroem",
    ferrolearn_kernel::FittedNystroem<f64>,
    (),
    { ferrolearn_kernel::Nystroem::<f64>::new() }
);

py_transformer!(
    RsRBFSampler, "_RsRBFSampler",
    ferrolearn_kernel::FittedRBFSampler<f64>,
    (),
    { ferrolearn_kernel::RBFSampler::<f64>::new() }
);
