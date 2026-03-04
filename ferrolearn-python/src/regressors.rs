//! PyO3 bindings for regression models.

use crate::conversions::*;
use ferrolearn_core::{Fit, HasCoefficients, Predict};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLinearRegression")]
pub struct RsLinearRegression {
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLinearRegression<f64>>,
}

#[pymethods]
impl RsLinearRegression {
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model =
            ferrolearn_linear::LinearRegression::<f64>::new().with_fit_intercept(self.fit_intercept);
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
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
// Ridge
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsRidge")]
pub struct RsRidge {
    alpha: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedRidge<f64>>,
}

#[pymethods]
impl RsRidge {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::Ridge::<f64>::new()
            .with_alpha(self.alpha)
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
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
// Lasso
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsLasso")]
pub struct RsLasso {
    alpha: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedLasso<f64>>,
}

#[pymethods]
impl RsLasso {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(alpha: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::Lasso::<f64>::new()
            .with_alpha(self.alpha)
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
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
// ElasticNet
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsElasticNet")]
pub struct RsElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    fitted: Option<ferrolearn_linear::FittedElasticNet<f64>>,
}

#[pymethods]
impl RsElasticNet {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=true))]
    fn new(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64, fit_intercept: bool) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            fit_intercept,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let y_nd = numpy1_to_ndarray(y);
        let model = ferrolearn_linear::ElasticNet::<f64>::new()
            .with_alpha(self.alpha)
            .with_l1_ratio(self.l1_ratio)
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let preds = fitted
            .predict(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray1_to_numpy(py, &preds))
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
