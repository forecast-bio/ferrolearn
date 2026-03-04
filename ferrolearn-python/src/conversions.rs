//! Conversion utilities between ndarray and numpy arrays.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Convert a numpy 2D array (read-only) to an ndarray Array2<f64>.
pub fn numpy2_to_ndarray(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

/// Convert a numpy 1D array (read-only) to an ndarray Array1<f64>.
pub fn numpy1_to_ndarray(x: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    x.as_array().to_owned()
}

/// Convert a numpy 1D array of i64 to an ndarray Array1<usize> (for class labels).
pub fn numpy1_to_ndarray_usize(y: PyReadonlyArray1<'_, i64>) -> Array1<usize> {
    y.as_array().mapv(|v| v as usize)
}

/// Convert an ndarray Array1<f64> to a numpy 1D array.
pub fn ndarray1_to_numpy<'py>(py: Python<'py>, a: &Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_array(py, a)
}

/// Convert an ndarray Array2<f64> to a numpy 2D array.
pub fn ndarray2_to_numpy<'py>(py: Python<'py>, a: &Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_array(py, a)
}

/// Convert an ndarray Array1<usize> to a numpy 1D array of i64.
pub fn ndarray1_usize_to_numpy<'py>(
    py: Python<'py>,
    a: &Array1<usize>,
) -> Bound<'py, PyArray1<i64>> {
    let converted = a.mapv(|v| v as i64);
    PyArray1::from_array(py, &converted)
}
