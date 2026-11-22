//! Finding the `refimage` Python package, once, and checking that it can exchange values
//! with this crate.

use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;

/// The version of the exchange format ([`parts`](crate::parts)). The `refimage` package
/// and this crate exchange values only when both name the same one.
pub const PROTOCOL: u32 = 1;

/// This crate's version, which the `refimage` package's must be compatible with.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `refimage._native`, once checked.
static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// The leading `X.Y.Z` of a Cargo (`1.0.0-pre6`) or Python (`1.0.0rc6`, `1.0.1.dev4`)
/// version.
fn release(version: &str) -> Option<[u64; 3]> {
    let mut parts = [0u64; 3];
    let mut rest = version;
    for slot in &mut parts {
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        let (n, tail) = rest.split_at(digits);
        *slot = n.parse().ok()?;
        rest = tail.strip_prefix('.').unwrap_or("");
    }
    Some(parts)
}

/// Whether the package at version `python` can serve this crate at `rust`: Cargo's caret
/// rule (the first non-zero part of `X.Y.Z` agrees, and all parts before it), on the
/// release numbers.
#[must_use]
pub fn compatible(rust: &str, python: &str) -> bool {
    match (release(rust), release(python)) {
        (Some([0, 0, z]), Some([0, 0, w])) => z == w,
        (Some([0, y, _]), Some([0, v, _])) => y == v,
        (Some([x, _, _]), Some([u, _, _])) => x == u,
        _ => false,
    }
}

fn check(py: Python<'_>) -> PyResult<Py<PyModule>> {
    let package = py.import("refimage").map_err(|e| {
        PyImportError::new_err(format!(
            "refimage-pyo3 {VERSION} needs the refimage Python package ({e})"
        ))
    })?;
    let version: String = package.getattr("__version__")?.extract()?;
    if !compatible(VERSION, &version) {
        return Err(PyImportError::new_err(format!(
            "refimage-pyo3 {VERSION} cannot exchange values with refimage {version}: install a \
             compatible refimage, or build against the refimage-pyo3 that matches it"
        )));
    }
    let module = py.import("refimage._native")?;
    let protocol: u32 = match module.getattr("_interop_protocol") {
        Ok(f) => f.call0()?.extract()?,
        Err(_) => 0,
    };
    if protocol != PROTOCOL {
        return Err(PyImportError::new_err(format!(
            "refimage {version} exchanges values with protocol {protocol}, refimage-pyo3 \
             {VERSION} with protocol {PROTOCOL}: install matching versions"
        )));
    }
    Ok(module.unbind())
}

/// `refimage._native`, after checking (once per process) that the installed `refimage` is
/// compatible with this crate. Every conversion calls it; calling it directly makes an
/// extension fail early, at import, rather than at its first conversion.
///
/// # Errors
/// `ImportError` if `refimage` is not installed, or its version or exchange protocol does
/// not match this crate's. The check runs again on the next call after a failure.
pub fn refimage_module(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    Ok(MODULE.get_or_try_init(py, || check(py))?.bind(py))
}

/// A `refimage.RefImageError` (the package's own exception class) with `msg`, for an
/// extension that wants to raise what `refimage` raises.
pub fn refimage_error(py: Python<'_>, msg: impl std::fmt::Display) -> PyErr {
    let msg = msg.to_string();
    match refimage_module(py).and_then(|m| m.getattr("RefImageError")) {
        Ok(cls) => match cls.call1((msg.clone(),)) {
            Ok(exc) => PyErr::from_value(exc),
            Err(e) => e,
        },
        Err(e) => e,
    }
}
