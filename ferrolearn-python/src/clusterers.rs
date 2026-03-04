//! PyO3 bindings for clustering models.

use crate::conversions::*;
use ferrolearn_core::{Fit, Predict, Transform};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

#[pyclass(name = "_RsKMeans")]
pub struct RsKMeans {
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    random_state: Option<u64>,
    fitted: Option<ferrolearn_cluster::FittedKMeans<f64>>,
}

#[pymethods]
impl RsKMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            n_init,
            random_state,
            fitted: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let x_nd = numpy2_to_ndarray(x);
        let mut model = ferrolearn_cluster::KMeans::<f64>::new(self.n_clusters)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol)
            .with_n_init(self.n_init);
        if let Some(seed) = self.random_state {
            model = model.with_random_state(seed);
        }
        let fitted = model
            .fit(&x_nd, &())
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

    #[getter]
    fn cluster_centers_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray2_to_numpy(py, fitted.cluster_centers()))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(ndarray1_usize_to_numpy(py, fitted.labels()))
    }

    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.inertia())
    }

    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("not fitted"))?;
        Ok(fitted.n_iter())
    }
}
