//! Pipelines: [`RiPipeline`].

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use refimage::pipeline::Pipeline;

use crate::compat::refimage_module;

/// A processing recipe at the boundary with Python: `refimage.Pipeline` there, a
/// [`Pipeline`] here. It crosses as its JSON form (the serde representation both sides
/// share), so every op, including warps and coordinate maps, carries exactly.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RiPipeline(pub Pipeline);

impl<'py> IntoPyObject<'py> for RiPipeline {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        (&self).into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for &RiPipeline {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json =
            serde_json::to_string(&self.0).map_err(|e| PyValueError::new_err(e.to_string()))?;
        refimage_module(py)?
            .getattr("Pipeline")?
            .call_method1("from_json", (json,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for RiPipeline {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        if !obj.is_instance(&refimage_module(obj.py())?.getattr("Pipeline")?)? {
            return Err(PyTypeError::new_err("expected a refimage.Pipeline"));
        }
        let json: String = obj.call_method0("to_json")?.extract()?;
        serde_json::from_str(&json)
            .map(Self)
            .map_err(|e| PyValueError::new_err(format!("refimage.Pipeline JSON: {e}")))
    }
}

impl From<Pipeline> for RiPipeline {
    fn from(p: Pipeline) -> Self {
        Self(p)
    }
}
