//! PyO3 bindings for transformer models.

use crate::conversions::*;
use ferrolearn_core::{Fit, Transform};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsStandardScaler")]
pub struct RsStandardScaler {
    fitted: Option<ferrolearn_preprocess::FittedStandardScaler<f64>>,
}

#[pymethods]
impl RsStandardScaler {
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_preprocess::StandardScaler::<f64>::new();
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .inverse_transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.mean()))
    }

    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.std()))
    }
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsPCA")]
pub struct RsPCA {
    n_components: usize,
    fitted: Option<ferrolearn_decomp::FittedPCA<f64>>,
}

#[pymethods]
impl RsPCA {
    #[new]
    #[pyo3(signature = (n_components=2))]
    fn new(n_components: usize) -> Self {
        Self {
            n_components,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let model = ferrolearn_decomp::PCA::<f64>::new(self.n_components);
        let fitted = model
            .fit(&x_nd, &())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        let x_nd = numpy2_to_ndarray(x);
        let result = fitted
            .inverse_transform(&x_nd)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(ndarray2_to_numpy(py, &result))
    }

    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.components()))
    }

    #[getter]
    fn explained_variance_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.explained_variance()))
    }

    #[getter]
    fn explained_variance_ratio_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.explained_variance_ratio()))
    }

    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.mean()))
    }

    #[getter]
    fn singular_values_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_to_numpy(py, fitted.singular_values()))
    }
}
