// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(
text_signature = "(special_unitary_matrix)"
)]
pub fn decompose_two_qubit_product_gate(
    py: Python<'_>,
    special_unitary_matrix: &PyAny,
) -> PyObject {
    "This is a string.".into_py(py)
}
