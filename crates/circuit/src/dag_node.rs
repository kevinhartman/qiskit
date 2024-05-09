// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit_instruction::CircuitInstruction;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList, PySequence, PyString, PyTuple};
use pyo3::{intern, PyObject, PyResult};

static SEMANTIC_EQ_PYTHON: GILOnceCell<PyResult<PyObject>> = GILOnceCell::new();

/// Parent class for DAGOpNode, DAGInNode, and DAGOutNode.
#[pyclass(module = "qiskit._accelerate.circuit", subclass)]
#[derive(Clone, Debug)]
pub struct DAGNode {
    #[pyo3(get, set)]
    pub _node_id: isize,
}

#[pymethods]
impl DAGNode {
    #[new]
    #[pyo3(signature=(nid=-1))]
    fn new(nid: isize) -> Self {
        DAGNode { _node_id: nid }
    }

    fn __getstate__(&self) -> isize {
        self._node_id
    }

    fn __setstate__(&mut self, nid: isize) {
        self._node_id = nid;
    }

    fn __lt__(&self, other: &DAGNode) -> bool {
        self._node_id < other._node_id
    }

    fn __gt__(&self, other: &DAGNode) -> bool {
        self._node_id > other._node_id
    }

    fn __str__(_self: &Bound<DAGNode>) -> String {
        format!("{}", _self.as_ptr() as usize)
    }

    fn __hash__(&self, py: Python) -> PyResult<isize> {
        self._node_id.into_py(py).bind(py).hash()
    }

    /// Check if DAG nodes are considered equivalent, e.g., as a node_match for
    /// :func:`rustworkx.is_isomorphic_node_match`.
    ///
    /// Args:
    ///     node1 (DAGOpNode, DAGInNode, DAGOutNode): A node to compare.
    ///     node2 (DAGOpNode, DAGInNode, DAGOutNode): The other node to compare.
    ///     bit_indices1 (dict): Dictionary mapping Bit instances to their index
    ///         within the circuit containing node1
    ///     bit_indices2 (dict): Dictionary mapping Bit instances to their index
    ///         within the circuit containing node2
    ///
    /// Return:
    ///     Bool: If node1 == node2
    #[staticmethod]
    #[pyo3(signature = (*args), text_signature = "(node1, node2, bit_indices1, bit_indices2)")]
    fn semantic_eq<'py>(py: Python<'py>, args: &Bound<PyTuple>) -> PyResult<Bound<'py, PyAny>> {
        let semantic_eq_fn = SEMANTIC_EQ_PYTHON
            .get_or_init(py, || {
                let module = PyModule::import_bound(py, "qiskit.dagcircuit.dagnode")?;
                Ok(module.getattr("_semantic_eq")?.unbind())
            })
            .as_ref()
            .map_err(|e| e.clone_ref(py))?;
        semantic_eq_fn.bind(py).call1(args)
    }
}

/// Object to represent an Instruction at a node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOpNode {
    pub instruction: CircuitInstruction,
    #[pyo3(get)]
    pub sort_key: PyObject,
}

type DAGOpNodeState = (((PyObject, Py<PyTuple>, Py<PyTuple>), PyObject), isize);

#[pymethods]
impl DAGOpNode {
    #[new]
    fn new(
        py: Python,
        op: PyObject,
        qargs: Option<&Bound<PySequence>>,
        cargs: Option<&Bound<PySequence>>,
        dag: Option<&Bound<PyAny>>,
    ) -> PyResult<(Self, DAGNode)> {
        let qargs =
            qargs.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;
        let cargs =
            cargs.map_or_else(|| Ok(PyTuple::empty_bound(py)), PySequenceMethods::to_tuple)?;

        let sort_key = match dag {
            Some(dag) => {
                let cache = dag
                    .getattr(intern!(py, "_key_cache"))?
                    .downcast_into_exact::<PyDict>()?;
                let cache_key = PyTuple::new_bound(py, [&qargs, &cargs]);
                match cache.get_item(&cache_key)? {
                    Some(key) => key,
                    None => {
                        let indices: PyResult<Vec<_>> = qargs
                            .iter()
                            .chain(cargs.iter())
                            .map(|bit| {
                                dag.call_method1(intern!(py, "find_bit"), (bit,))?
                                    .getattr(intern!(py, "index"))
                            })
                            .collect();
                        let index_strs: Vec<_> =
                            indices?.into_iter().map(|i| format!("{:04}", i)).collect();
                        let key = PyString::new_bound(py, index_strs.join(",").as_str());
                        cache.set_item(&cache_key, &key)?;
                        key.into_any()
                    }
                }
            }
            None => qargs.str()?.into_any(),
        };

        Ok((
            DAGOpNode {
                instruction: CircuitInstruction {
                    operation: op,
                    qubits: qargs.unbind(),
                    clbits: cargs.unbind(),
                },
                sort_key: sort_key.unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __getstate__(slf: PyRef<Self>, py: Python) -> PyObject {
        (
            (slf.instruction.__getstate__(py), &slf.sort_key),
            (slf.as_ref()._node_id),
        )
            .into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (((operation, qubits, clbits), sort_key), nid): DAGOpNodeState = state.extract()?;
        slf.instruction = CircuitInstruction {
            operation,
            qubits,
            clbits,
        };
        slf.sort_key = sort_key;
        slf.as_mut()._node_id = nid;
        Ok(())
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyObject {
        (
            &self.instruction.operation,
            &self.instruction.qubits,
            &self.instruction.clbits,
            None::<PyObject>,
        )
            .into_py(py)
    }

    #[getter]
    fn get_op(&self, py: Python) -> PyObject {
        self.instruction.operation.clone_ref(py)
    }

    #[setter]
    fn set_op(&mut self, op: PyObject) {
        self.instruction.operation = op;
    }

    #[getter]
    fn get_qargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.qubits.clone_ref(py)
    }

    #[setter]
    fn set_qargs(&mut self, qargs: Py<PyTuple>) {
        self.instruction.qubits = qargs;
    }

    #[getter]
    fn get_cargs(&self, py: Python) -> Py<PyTuple> {
        self.instruction.clbits.clone_ref(py)
    }

    #[setter]
    fn set_cargs(&mut self, cargs: Py<PyTuple>) {
        self.instruction.clbits = cargs;
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self, py: Python) -> PyResult<PyObject> {
        Ok(self
            .instruction
            .operation
            .bind(py)
            .getattr(intern!(py, "name"))?
            .unbind())
    }

    /// Sets the Instruction name corresponding to the op for this node
    #[setter]
    fn set_name(&self, py: Python, new_name: PyObject) -> PyResult<()> {
        self.instruction
            .operation
            .bind(py)
            .setattr(intern!(py, "name"), new_name)
    }

    /// Returns a representation of the DAGOpNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "DAGOpNode(op={}, qargs={}, cargs={})",
            self.instruction.operation.bind(py).str()?,
            self.instruction.qubits.bind(py).str()?,
            self.instruction.clbits.bind(py).str()?
        ))
    }
}

/// Object to represent an incoming wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGInNode {
    #[pyo3(get)]
    wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

#[pymethods]
impl DAGInNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGInNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __getstate__(slf: PyRef<Self>, py: Python) -> PyObject {
        (&slf.wire, slf.as_ref()._node_id).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (wire, nid): (PyObject, isize) = state.extract()?;
        slf.wire = wire;
        slf.as_mut()._node_id = nid;
        Ok(())
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyObject {
        (&self.wire,).into_py(py)
    }

    /// Returns a representation of the DAGInNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGInNode(wire={})", self.wire.bind(py).str()?))
    }
}

/// Object to represent an outgoing wire node in the DAGCircuit.
#[pyclass(module = "qiskit._accelerate.circuit", extends=DAGNode)]
pub struct DAGOutNode {
    #[pyo3(get)]
    wire: PyObject,
    #[pyo3(get)]
    sort_key: PyObject,
}

#[pymethods]
impl DAGOutNode {
    #[new]
    fn new(py: Python, wire: PyObject) -> PyResult<(Self, DAGNode)> {
        Ok((
            DAGOutNode {
                wire,
                sort_key: PyList::empty_bound(py).str()?.into_any().unbind(),
            },
            DAGNode { _node_id: -1 },
        ))
    }

    fn __getstate__(slf: PyRef<Self>, py: Python) -> PyObject {
        (&slf.wire, slf.as_ref()._node_id).into_py(py)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyAny>) -> PyResult<()> {
        let (wire, nid): (PyObject, isize) = state.extract()?;
        slf.wire = wire;
        slf.as_mut()._node_id = nid;
        Ok(())
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyObject {
        (&self.wire,).into_py(py)
    }

    /// Returns a representation of the DAGOutNode
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("DAGOutNode(wire={})", self.wire.bind(py).str()?))
    }
}
