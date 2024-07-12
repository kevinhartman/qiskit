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

// This module contains objects imported from Python that are reused. These are
// typically data model classes that are used to identify an object, or for
// python side casting

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

use crate::operations::{StandardGate, STANDARD_GATE_SIZE};

/// Helper wrapper around `GILOnceCell` instances that are just intended to store a Python object
/// that is lazily imported.
pub struct ImportOnceCell {
    module: &'static str,
    object: &'static str,
    cell: GILOnceCell<Py<PyAny>>,
}

impl ImportOnceCell {
    const fn new(module: &'static str, object: &'static str) -> Self {
        Self {
            module,
            object,
            cell: GILOnceCell::new(),
        }
    }

    /// Get the underlying GIL-independent reference to the contained object, importing if
    /// required.
    #[inline]
    pub fn get(&self, py: Python) -> &Py<PyAny> {
        self.cell.get_or_init(py, || {
            py.import_bound(self.module)
                .unwrap()
                .getattr(self.object)
                .unwrap()
                .unbind()
        })
    }

    /// Get a GIL-bound reference to the contained object, importing if required.
    #[inline]
    pub fn get_bound<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.get(py).bind(py)
    }
}

pub static BUILTIN_LIST: ImportOnceCell = ImportOnceCell::new("builtins", "list");
pub static OPERATION: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.operation", "Operation");
pub static INSTRUCTION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.instruction", "Instruction");
pub static GATE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.gate", "Gate");
pub static QUBIT: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.quantumregister", "Qubit");
pub static CLBIT: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.classicalregister", "Clbit");
pub static QUANTUM_REGISTER: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.quantumregister", "QuantumRegister");
pub static CLASSICAL_REGISTER: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.classicalregister", "ClassicalRegister");
pub static PARAMETER_EXPRESSION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parameterexpression", "ParameterExpression");
pub static QUANTUM_CIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.quantumcircuit", "QuantumCircuit");
pub static SINGLETON_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.singleton", "SingletonGate");
pub static SINGLETON_CONTROLLED_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.singleton", "SingletonControlledGate");
pub static VARIABLE_MAPPER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit._classical_resource_map", "VariableMapper");
pub static CONTROL_FLOW_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "ControlFlowOp");
pub static SWITCH_CASE_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "SwitchCaseOp");
pub static STORE_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "Store");
pub static EXPR: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.classical.expr", "Expr");
pub static ITER_VARS: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.classical.expr", "iter_vars");

pub static DAG_NODE: ImportOnceCell = ImportOnceCell::new("qiskit.dagcircuit", "DAGNode");

/// A mapping from the enum variant in crate::operations::StandardGate to the python
/// module path and class name to import it. This is used to populate the conversion table
/// when a gate is added directly via the StandardGate path and there isn't a Python object
/// to poll the _standard_gate attribute for.
///
/// NOTE: the order here is significant it must match the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
static STDGATE_IMPORT_PATHS: [[&str; 2]; STANDARD_GATE_SIZE] = [
    // ZGate = 0
    ["qiskit.circuit.library.standard_gates.z", "ZGate"],
    // YGate = 1
    ["qiskit.circuit.library.standard_gates.y", "YGate"],
    // XGate = 2
    ["qiskit.circuit.library.standard_gates.x", "XGate"],
    // CZGate = 3
    ["qiskit.circuit.library.standard_gates.z", "CZGate"],
    // CYGate = 4
    ["qiskit.circuit.library.standard_gates.y", "CYGate"],
    // CXGate = 5
    ["qiskit.circuit.library.standard_gates.x", "CXGate"],
    // CCXGate = 6
    ["qiskit.circuit.library.standard_gates.x", "CCXGate"],
    // RXGate = 7
    ["qiskit.circuit.library.standard_gates.rx", "RXGate"],
    // RYGate = 8
    ["qiskit.circuit.library.standard_gates.ry", "RYGate"],
    // RZGate = 9
    ["qiskit.circuit.library.standard_gates.rz", "RZGate"],
    // ECRGate = 10
    ["qiskit.circuit.library.standard_gates.ecr", "ECRGate"],
    // SwapGate = 11
    ["qiskit.circuit.library.standard_gates.swap", "SwapGate"],
    // SXGate = 12
    ["qiskit.circuit.library.standard_gates.sx", "SXGate"],
    // GlobalPhaseGate = 13
    [
        "qiskit.circuit.library.standard_gates.global_phase",
        "GlobalPhaseGate",
    ],
    // IGate = 14
    ["qiskit.circuit.library.standard_gates.i", "IGate"],
    // HGate = 15
    ["qiskit.circuit.library.standard_gates.h", "HGate"],
    // PhaseGate = 16
    ["qiskit.circuit.library.standard_gates.p", "PhaseGate"],
    // UGate = 17
    ["qiskit.circuit.library.standard_gates.u", "UGate"],
];

/// A mapping from the enum variant in crate::operations::StandardGate to the python object for the
/// class that matches it. This is typically used when we need to convert from the internal rust
/// representation to a Python object for a python user to interact with.
///
/// NOTE: the order here is significant it must match the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
static mut STDGATE_PYTHON_GATES: GILOnceCell<[Option<PyObject>; STANDARD_GATE_SIZE]> =
    GILOnceCell::new();

#[inline]
pub fn populate_std_gate_map(py: Python, rs_gate: StandardGate, py_gate: PyObject) {
    let gate_map = unsafe {
        match STDGATE_PYTHON_GATES.get_mut() {
            Some(gate_map) => gate_map,
            None => {
                let array: [Option<PyObject>; STANDARD_GATE_SIZE] = std::array::from_fn(|_| None);
                STDGATE_PYTHON_GATES.set(py, array).unwrap();
                STDGATE_PYTHON_GATES.get_mut().unwrap()
            }
        }
    };
    let gate_cls = &gate_map[rs_gate as usize];
    if gate_cls.is_none() {
        gate_map[rs_gate as usize] = Some(py_gate.clone_ref(py));
    }
}

#[inline]
pub fn get_std_gate_class(py: Python, rs_gate: StandardGate) -> PyResult<PyObject> {
    let gate_map =
        unsafe { STDGATE_PYTHON_GATES.get_or_init(py, || std::array::from_fn(|_| None)) };
    let gate = &gate_map[rs_gate as usize];
    let populate = gate.is_none();
    let out_gate = match gate {
        Some(gate) => gate.clone_ref(py),
        None => {
            let [py_mod, py_class] = STDGATE_IMPORT_PATHS[rs_gate as usize];
            py.import_bound(py_mod)?.getattr(py_class)?.unbind()
        }
    };
    if populate {
        populate_std_gate_map(py, rs_gate, out_gate.clone_ref(py));
    }
    Ok(out_gate)
}
