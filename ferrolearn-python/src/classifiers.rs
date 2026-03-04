//! PyO3 bindings for classification models.

use crate::conversions::*;
use ferrolearn_core::{Fit, HasClasses, HasCoefficients, HasFeatureImportances, Predict};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// LogisticRegression
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLogisticRegression")]
pub struct RsLogisticRegression {
    c: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLogisticRegression<f64>>,
}

#[pymethods]
impl RsLogisticRegression {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(c: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            c,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_linear::LogisticRegression::<f64>::new()
            .with_c(self.c)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_fit_intercept(self.fit_intercept);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.coefficients()))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.intercept())
    }
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsDecisionTreeClassifier")]
pub struct RsDecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    fitted: Option<ferrolearn_tree::FittedDecisionTreeClassifier<f64>>,
}

#[pymethods]
impl RsDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_tree::DecisionTreeClassifier::<f64>::new()
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn feature_importances_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// RandomForestClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsRandomForestClassifier")]
pub struct RsRandomForestClassifier {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_tree::FittedRandomForestClassifier<f64>>,
}

#[pymethods]
impl RsRandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let mut model = ferrolearn_tree::RandomForestClassifier::<f64>::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_min_samples_split(self.min_samples_split)
            .with_min_samples_leaf(self.min_samples_leaf);
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }

    #[getter]
    fn feature_importances_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.feature_importances()))
    }
}

// ---------------------------------------------------------------------------
// KNeighborsClassifier
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsKNeighborsClassifier")]
pub struct RsKNeighborsClassifier {
    n_neighbors: usize,
    fitted: Option<ferrolearn_neighbors::FittedKNeighborsClassifier<f64>>,
}

#[pymethods]
impl RsKNeighborsClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5))]
    fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model = ferrolearn_neighbors::KNeighborsClassifier::<f64>::new()
            .with_n_neighbors(self.n_neighbors);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }
}

// ---------------------------------------------------------------------------
// GaussianNB
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsGaussianNB")]
pub struct RsGaussianNB {
    var_smoothing: f64,
    fitted: Option<ferrolearn_bayes::FittedGaussianNB<f64>>,
}

#[pymethods]
impl RsGaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        Self {
            var_smoothing,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, i64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray_usize(y);
        let model =
            ferrolearn_bayes::GaussianNB::<f64>::new().with_var_smoothing(self.var_smoothing);
        let fitted = model
            .fit(&x_nd, &y_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_usize_to_numpy(py, &preds))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let proba = fitted
            .predict_proba(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &proba))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let classes = fitted.classes();
        let arr = ndarray::Array1::from_vec(classes.iter().map(|&c| c as i64).collect());
        Ok(PyArray1::from_array(py, &arr))
    }
}
