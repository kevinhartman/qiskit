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

use crate::bit_data::BitData;
use crate::circuit_instruction::{
    convert_py_to_operation_type, CircuitInstruction, ExtraInstructionAttributes,
    OperationTypeConstruct,
};
use crate::circuit_instruction::{operation_type_and_data_to_py, PackedInstruction};
use crate::dag_node::{DAGInNode, DAGNode, DAGOpNode, DAGOutNode};
use crate::dot_utils::build_dot;
use crate::error::DAGCircuitError;
use crate::imports::{
    CLASSICAL_REGISTER, CLBIT, CONTROL_FLOW_OP, DAG_NODE, EXPR, ITER_VARS, QUBIT, STORE_OP,
    SWITCH_CASE_OP, VARIABLE_MAPPER,
};
use crate::interner::{Index, IndexedInterner, Interner};
use crate::operations::{Operation, OperationType, Param};
use crate::rustworkx_core_vnext::isomorphism;
use crate::{interner, BitType, Clbit, Qubit, TupleLikeArg};
use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::{hash_map, HashMap, HashSet};
use indexmap::set::Slice;
use indexmap::{IndexMap, IndexSet};
use petgraph::prelude::*;
use pyo3::callback::IntoPyCallbackOutput;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi::PyCFunction;
use pyo3::prelude::*;
use pyo3::types::iter::BoundTupleIterator;
use pyo3::types::{
    IntoPyDict, PyDict, PyFloat, PyFrozenSet, PyInt, PyIterator, PyList, PySequence, PySet,
    PySlice, PyString, PyTuple, PyType,
};
use pyo3::{intern, PyObject, PyResult, PyVisit};
use rustworkx_core::err::ContractError;
use rustworkx_core::graph_ext::ContractNodesDirected;
use rustworkx_core::petgraph;
use rustworkx_core::petgraph::prelude::StableDiGraph;
use rustworkx_core::petgraph::stable_graph::{DefaultIx, IndexType, Neighbors, NodeIndex};
use rustworkx_core::petgraph::visit::{IntoNodeReferences, NodeCount, NodeRef};
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::traversal::{
    ancestors as core_ancestors, bfs_successors as core_bfs_successors,
    descendants as core_descendants,
};
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::f64::consts::PI;
use std::ffi::c_double;
use std::hash::{Hash, Hasher};

trait IntoUnique {
    type Output;
    fn unique(self) -> Self::Output;
}

struct UniqueIterator<I, N: Hash + Eq> {
    neighbors: I,
    seen: HashSet<N>,
}

impl<I> IntoUnique for I
where
    I: Iterator,
    I::Item: Hash + Eq + Clone,
{
    type Output = UniqueIterator<I, I::Item>;

    fn unique(self) -> Self::Output {
        UniqueIterator {
            neighbors: self,
            seen: HashSet::new(),
        }
    }
}

impl<I> Iterator for UniqueIterator<I, I::Item>
where
    I: Iterator,
    I::Item: Hash + Eq + Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // First any outgoing edges
        while let Some(node) = self.neighbors.next() {
            if !self.seen.contains(&node) {
                self.seen.insert(node.clone());
                return Some(node);
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) enum NodeType {
    QubitIn(Qubit),
    QubitOut(Qubit),
    ClbitIn(Clbit),
    ClbitOut(Clbit),
    Operation(PackedInstruction),
}

impl NodeType {
    #[inline]
    pub fn key(&self) -> (Option<Index>, Option<Index>) {
        match self {
            NodeType::Operation(packed) => (Some(packed.qubits_id), Some(packed.clbits_id)),
            _ => (None, None),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Wire {
    Qubit(Qubit),
    Clbit(Clbit),
    Var(PyObject),
}

impl Hash for Wire {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Wire::Qubit(q) => {
                state.write_u8(1);
                q.hash(state);
            }
            Wire::Clbit(c) => {
                state.write_u8(2);
                c.hash(state);
            }
            Wire::Var(v) => Python::with_gil(|py| {
                state.write_u8(3);
                state.write_isize(v.bind(py).hash().unwrap_or(0));
            }),
        }
    }
}

impl PartialEq for Wire {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Wire::Qubit(q1), Wire::Qubit(q2)) => q1 == q2,
            (Wire::Clbit(c1), Wire::Clbit(c2)) => c1 == c2,
            (Wire::Var(v1), Wire::Var(v2)) => {
                v1.is(v2) || Python::with_gil(|py| v1.bind(py).eq(v2).unwrap())
            }
            _ => false,
        }
    }
}

impl Eq for Wire {}

// TODO: Remove me.
// This is a temporary map type used to store a mapping of
// Var to NodeIndex to hold us over until Var is ported to
// Rust. Currently, we need this because PyObject cannot be
// used as the key to an IndexMap.
//
// Once we've got Var ported, Wire should also become Hash + Eq
// and we can consider combining input/output nodes maps.
#[derive(Clone, Debug)]
struct _VarIndexMap {
    dict: Py<PyDict>,
}

impl _VarIndexMap {
    pub fn new(py: Python) -> Self {
        Self {
            dict: PyDict::new_bound(py).unbind(),
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = PyObject> {
        Python::with_gil(|py| {
            self.dict
                .bind(py)
                .keys()
                .into_iter()
                .map(|k| k.unbind())
                .collect::<Vec<_>>()
                .into_iter()
        })
    }

    pub fn contains_key(&self, key: &PyObject) -> bool {
        Python::with_gil(|py| self.dict.bind(py).contains(key).unwrap())
    }

    pub fn get(&self, key: &PyObject) -> Option<NodeIndex> {
        Python::with_gil(|py| {
            self.dict
                .bind(py)
                .get_item(key)
                .unwrap()
                .map(|v| NodeIndex::new(v.extract().unwrap()))
        })
    }

    pub fn insert(&mut self, key: PyObject, value: NodeIndex) {
        Python::with_gil(|py| {
            self.dict
                .bind(py)
                .set_item(key, value.index().into_py(py))
                .unwrap()
        })
    }
}

/// Quantum circuit as a directed acyclic graph.
///
/// There are 3 types of nodes in the graph: inputs, outputs, and operations.
/// The nodes are connected by directed edges that correspond to qubits and
/// bits.
#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct DAGCircuit {
    #[pyo3(get, set)]
    name: Option<Py<PyString>>,
    #[pyo3(get, set)]
    metadata: Option<Py<PyDict>>,
    calibrations: HashMap<String, Py<PyDict>>,

    pub(crate) dag: StableDiGraph<NodeType, Wire>,

    #[pyo3(get)]
    qregs: Py<PyDict>,
    #[pyo3(get)]
    cregs: Py<PyDict>,

    /// The cache used to intern instruction qargs.
    qargs_cache: IndexedInterner<Vec<Qubit>>,
    /// The cache used to intern instruction cargs.
    cargs_cache: IndexedInterner<Vec<Clbit>>,
    /// Qubits registered in the circuit.
    pub(crate) qubits: BitData<Qubit>,
    /// Clbits registered in the circuit.
    pub(crate) clbits: BitData<Clbit>,
    /// Global phase.
    global_phase: PyObject,
    /// Duration.
    #[pyo3(get, set)]
    duration: Option<PyObject>,
    /// Unit of duration.
    #[pyo3(get, set)]
    unit: String,

    // Note: these are tracked separately from `qubits` and `clbits`
    // because it's not yet clear if the Rust concept of a native Qubit
    // and Clbit should correspond directly to the numerical Python
    // index that users see in the Python API.
    /// The index locations of bits, and their positions within
    /// registers.
    qubit_locations: Py<PyDict>,
    clbit_locations: Py<PyDict>,

    /// Map from qubit to input nodes of the graph.
    qubit_input_map: IndexMap<Qubit, NodeIndex>,
    /// Map from qubit to output nodes of the graph.
    qubit_output_map: IndexMap<Qubit, NodeIndex>,

    /// Map from clbit to input nodes of the graph.
    clbit_input_map: IndexMap<Clbit, NodeIndex>,
    /// Map from clbit to output nodes of the graph.
    clbit_output_map: IndexMap<Clbit, NodeIndex>,

    // TODO: use IndexMap<Wire, NodeIndex> once Var is ported to Rust
    /// Map from var to input nodes of the graph.
    var_input_map: _VarIndexMap,
    /// Map from var to output nodes of the graph.
    var_output_map: _VarIndexMap,

    /// Operation kind to count
    op_names: HashMap<String, usize>,

    // Python modules we need to frequently access (for now).
    control_flow_module: PyControlFlowModule,
    circuit_module: PyCircuitModule,
}

#[derive(Clone, Debug)]
struct PyControlFlowModule {
    condition_resources: Py<PyAny>,
    node_resources: Py<PyAny>,
    control_flow_op_names: Py<PyFrozenSet>,
}

#[derive(Clone, Debug)]
struct PyLegacyResources {
    clbits: Py<PyTuple>,
    cregs: Py<PyTuple>,
}

impl PyControlFlowModule {
    fn new(py: Python) -> PyResult<Self> {
        let module = PyModule::import_bound(py, "qiskit.circuit.controlflow")?;
        Ok(PyControlFlowModule {
            condition_resources: module.getattr("condition_resources")?.unbind(),
            node_resources: module.getattr("node_resources")?.unbind(),
            control_flow_op_names: module
                .getattr("CONTROL_FLOW_OP_NAMES")?
                .downcast_into_exact()?
                .unbind(),
        })
    }

    fn condition_resources(&self, condition: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self
            .condition_resources
            .bind(condition.py())
            .call1((condition,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }

    fn node_resources(&self, node: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self.node_resources.bind(node.py()).call1((node,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }
}

#[derive(Clone, Debug)]
struct PyCircuitModule {
    clbit: Py<PyAny>,
    qubit: Py<PyAny>,
    classical_register: Py<PyAny>,
    quantum_register: Py<PyAny>,
    control_flow_op: Py<PyAny>,
    for_loop_op: Py<PyAny>,
    if_else_op: Py<PyAny>,
    while_loop_op: Py<PyAny>,
    switch_case_op: Py<PyAny>,
    operation: Py<PyAny>,
    store: Py<PyAny>,
    gate: Py<PyAny>,
    parameter_expression: Py<PyAny>,
    variable_mapper: Py<PyAny>,
}

impl PyCircuitModule {
    fn new(py: Python) -> PyResult<Self> {
        let module = PyModule::import_bound(py, "qiskit.circuit")?;
        Ok(PyCircuitModule {
            clbit: module.getattr("Clbit")?.unbind(),
            qubit: module.getattr("Qubit")?.unbind(),
            classical_register: module.getattr("ClassicalRegister")?.unbind(),
            quantum_register: module.getattr("QuantumRegister")?.unbind(),
            control_flow_op: module.getattr("ControlFlowOp")?.unbind(),
            for_loop_op: module.getattr("ForLoopOp")?.unbind(),
            if_else_op: module.getattr("IfElseOp")?.unbind(),
            while_loop_op: module.getattr("WhileLoopOp")?.unbind(),
            switch_case_op: module.getattr("SwitchCaseOp")?.unbind(),
            operation: module.getattr("Operation")?.unbind(),
            store: module.getattr("Store")?.unbind(),
            gate: module.getattr("Gate")?.unbind(),
            parameter_expression: module.getattr("ParameterExpression")?.unbind(),
            variable_mapper: module
                .getattr("_classical_resource_map")?
                .getattr("VariableMapper")?
                .unbind(),
        })
    }
}

struct PyVariableMapper {
    mapper: Py<PyAny>,
}

impl PyVariableMapper {
    fn new(
        py: Python,
        target_cregs: Bound<PyAny>,
        bit_map: Option<Bound<PyDict>>,
        var_map: Option<Bound<PyDict>>,
        add_register: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let kwargs: HashMap<&str, Option<Py<PyAny>>> =
            HashMap::from_iter([("add_register", add_register)]);
        Ok(PyVariableMapper {
            mapper: VARIABLE_MAPPER
                .get_bound(py)
                .call(
                    (target_cregs, bit_map, var_map),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .unbind(),
        })
    }

    fn map_condition<'py>(
        &self,
        condition: &Bound<'py, PyAny>,
        allow_reorder: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = condition.py();
        let kwargs: HashMap<&str, Py<PyAny>> =
            HashMap::from_iter([("allow_reorder", allow_reorder.into_py(py))]);
        self.mapper.bind(py).call_method(
            intern!(py, "map_condition"),
            (condition,),
            Some(&kwargs.into_py_dict_bound(py)),
        )
    }

    fn map_target<'py>(&self, target: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = target.py();
        self.mapper
            .bind(py)
            .call_method1(intern!(py, "map_target"), (target,))
    }

    fn map_expr<'py>(&self, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = node.py();
        self.mapper
            .bind(py)
            .call_method1(intern!(py, "map_expr"), (node,))
    }
}

#[pyfunction]
fn reject_new_register(reg: &Bound<PyAny>) -> PyResult<()> {
    Err(DAGCircuitError::new_err(format!(
        "No register with '{:?}' to map this expression onto.",
        reg.getattr("bits")?
    )))
}

impl IntoPy<Py<PyAny>> for PyVariableMapper {
    fn into_py(self, _py: Python<'_>) -> Py<PyAny> {
        self.mapper
    }
}

#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
struct BitLocations {
    #[pyo3(get)]
    index: Py<PyAny>,
    #[pyo3(get)]
    registers: Py<PyList>,
}

#[pymethods]
impl DAGCircuit {
    #[new]
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(DAGCircuit {
            name: None,
            metadata: None,
            calibrations: HashMap::new(),
            dag: StableDiGraph::default(),
            qregs: PyDict::new_bound(py).unbind(),
            cregs: PyDict::new_bound(py).unbind(),
            qargs_cache: IndexedInterner::new(),
            cargs_cache: IndexedInterner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            global_phase: PyFloat::new_bound(py, 0 as c_double).into_any().unbind(),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: PyDict::new_bound(py).unbind(),
            clbit_locations: PyDict::new_bound(py).unbind(),
            qubit_input_map: IndexMap::new(),
            qubit_output_map: IndexMap::new(),
            clbit_input_map: IndexMap::new(),
            clbit_output_map: IndexMap::new(),
            var_input_map: _VarIndexMap::new(py),
            var_output_map: _VarIndexMap::new(py),
            op_names: HashMap::new(),

            // Python module wrappers
            control_flow_module: PyControlFlowModule::new(py)?,
            circuit_module: PyCircuitModule::new(py)?,
        })
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter]
    pub fn qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached().clone_ref(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter]
    pub fn clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached().clone_ref(py)
    }

    /// Return a list of the wires in order.
    #[getter]
    fn get_wires(&self, py: Python<'_>) -> Py<PyList> {
        let wires: Vec<&PyObject> = self
            .qubits
            .bits()
            .iter()
            .chain(self.clbits.bits().iter())
            .collect();
        PyList::new_bound(py, wires).unbind()
    }

    /// Returns the number of nodes in the dag.
    #[getter]
    fn get_node_counter(&self) -> usize {
        self.dag.node_count()
    }

    /// Return the global phase of the circuit.
    #[getter]
    fn get_global_phase(&self) -> &PyObject {
        &self.global_phase
    }

    /// Set the global phase of the circuit.
    ///
    /// Args:
    ///     angle (float, :class:`.ParameterExpression`): The phase angle.
    #[setter]
    fn set_global_phase(&mut self, py: Python<'_>, angle: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(angle) = angle.downcast::<PyFloat>() {
            self.global_phase = PyFloat::new_bound(
                py,
                if !angle.is_truthy()? {
                    0 as c_double
                } else {
                    angle.value() % (2f64 * PI)
                },
            )
            .into_any()
            .unbind();
        } else {
            self.global_phase = angle.clone().unbind()
        }
        Ok(())
    }

    /// Return calibration dictionary.
    ///
    /// The custom pulse definition of a given gate is of the form
    ///    {'gate_name': {(qubits, params): schedule}}
    #[getter]
    fn get_calibrations(&self, py: Python) -> HashMap<String, Py<PyDict>> {
        self.calibrations.clone()
    }

    /// Set the circuit calibration data from a dictionary of calibration definition.
    ///
    ///  Args:
    ///      calibrations (dict): A dictionary of input in the format
    ///          {'gate_name': {(qubits, gate_params): schedule}}
    #[setter]
    fn set_calibrations(&mut self, calibrations: HashMap<String, Py<PyDict>>) {
        self.calibrations = calibrations;
    }

    /// Register a low-level, custom pulse definition for the given gate.
    ///
    /// Args:
    ///     gate (Union[Gate, str]): Gate information.
    ///     qubits (Union[int, Tuple[int]]): List of qubits to be measured.
    ///     schedule (Schedule): Schedule information.
    ///     params (Optional[List[Union[float, Parameter]]]): A list of parameters.
    ///
    /// Raises:
    ///     Exception: if the gate is of type string and params is None.
    fn add_calibration<'py>(
        &mut self,
        py: Python<'py>,
        mut gate: Bound<'py, PyAny>,
        mut qubits: Bound<'py, PyAny>,
        schedule: Py<PyAny>,
        mut params: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        if gate.is_instance(self.circuit_module.gate.bind(py))? {
            params = Some(gate.getattr(intern!(py, "params"))?);
            gate = gate.getattr(intern!(py, "name"))?;
        }

        if let Some(operands) = params {
            let add_calibration = PyModule::from_code_bound(
                py,
                r#"
def _format(operand):
    try:
        # Using float/complex value as a dict key is not good idea.
        # This makes the mapping quite sensitive to the rounding error.
        # However, the mechanism is already tied to the execution model (i.e. pulse gate)
        # and we cannot easily update this rule.
        # The same logic exists in QuantumCircuit.add_calibration.
        evaluated = complex(operand)
        if np.isreal(evaluated):
            evaluated = float(evaluated.real)
            if evaluated.is_integer():
                evaluated = int(evaluated)
        return evaluated
    except TypeError:
        # Unassigned parameter
        return operand
    "#,
                "add_calibration.py",
                "add_calibration",
            )?;

            let format = add_calibration.getattr("_format")?;
            let mapped: PyResult<Vec<_>> = operands.iter()?.map(|p| format.call1((p?,))).collect();
            params = Some(PyTuple::new_bound(py, mapped).into_any());
        } else {
            params = Some(PyTuple::empty_bound(py).into_any());
        }

        let calibrations = self
            .calibrations
            .entry(gate.extract()?)
            .or_insert_with(|| PyDict::new_bound(py).unbind())
            .bind(py);

        if !qubits.is_instance_of::<PyTuple>() {
            qubits = PyTuple::new_bound(py, [qubits]).into_any();
        }

        calibrations.set_item((qubits, params.unwrap()).to_object(py), schedule)?;
        Ok(())
    }

    /// Return True if the dag has a calibration defined for the node operation. In this
    /// case, the operation does not need to be translated to the device basis.
    fn has_calibration_for(&self, py: Python, node: PyRef<DAGOpNode>) -> PyResult<bool> {
        let node = node.as_ref().node.unwrap();
        if let Some(NodeType::Operation(packed)) = self.dag.node_weight(node) {
            let op_name = packed.op.name().to_string();
            if !self.calibrations.contains_key(&op_name) {
                return Ok(false);
            }
            let mut params = Vec::new();
            for p in &packed.params {
                if let Param::ParameterExpression(exp) = p {
                    let exp = exp.bind(py);
                    if !exp.getattr(intern!(py, "parameters"))?.is_truthy()? {
                        let as_py_float = exp.call_method0(intern!(py, "__float__"))?;
                        params.push(as_py_float.unbind());
                        continue;
                    }
                }
                params.push(p.to_object(py));
            }
            let qubits: Vec<BitType> = self
                .qargs_cache
                .intern(packed.qubits_id)
                .iter()
                .cloned()
                .map(|b| b.into())
                .collect();
            let params = PyTuple::new_bound(py, params);
            self.calibrations[&op_name]
                .bind(py)
                .contains((qubits, params).to_object(py))
        } else {
            Ok(false)
        }
    }

    /// Remove all operation nodes with the given name.
    fn remove_all_ops_named(&mut self, opname: &str) {
        let mut to_remove = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if opname == packed.op.name() {
                    to_remove.push(id);
                }
            }
        }
        for node in to_remove {
            self.remove_op_node(node);
        }
    }

    /// Add individual qubit wires.
    fn add_qubits(&mut self, py: Python, qubits: &Bound<PySequence>) -> PyResult<()> {
        let bits: Vec<Bound<PyAny>> = qubits.extract()?;
        for bit in bits.iter() {
            if !bit.is_instance(self.circuit_module.qubit.bind(py))? {
                return Err(DAGCircuitError::new_err("not a Qubit instance."));
            }

            if self.qubits.find(bit).is_some() {
                return Err(DAGCircuitError::new_err(format!("duplicate qubit {}", bit)));
            }
        }

        for bit in bits.iter() {
            self.add_qubit_unchecked(py, bit)?;
        }
        Ok(())
    }

    /// Add individual qubit wires.
    fn add_clbits(&mut self, py: Python, clbits: &Bound<PySequence>) -> PyResult<()> {
        let bits: Vec<Bound<PyAny>> = clbits.extract()?;
        for bit in bits.iter() {
            if !bit.is_instance(self.circuit_module.clbit.bind(py))? {
                return Err(DAGCircuitError::new_err("not a Clbit instance."));
            }

            if self.clbits.find(bit).is_some() {
                return Err(DAGCircuitError::new_err(format!("duplicate clbit {}", bit)));
            }
        }

        for bit in bits.iter() {
            self.add_clbit_unchecked(py, bit)?;
        }
        Ok(())
    }

    /// Add all wires in a quantum register.
    fn add_qreg(&mut self, py: Python, qreg: &Bound<PyAny>) -> PyResult<()> {
        if !qreg.is_instance(self.circuit_module.quantum_register.bind(py))? {
            return Err(DAGCircuitError::new_err("not a QuantumRegister instance."));
        }

        let register_name = qreg.getattr(intern!(py, "name"))?;
        if self.qregs.bind(py).contains(&register_name)? {
            return Err(DAGCircuitError::new_err(format!(
                "duplicate register {}",
                register_name
            )));
        }
        self.qregs.bind(py).set_item(register_name, qreg)?;

        for (index, bit) in qreg.iter()?.enumerate() {
            let bit = bit?;
            if self.qubits.find(&bit).is_none() {
                self.add_qubit_unchecked(py, &bit)?;
            }
            let locations: PyRef<BitLocations> = self
                .qubit_locations
                .bind(py)
                .get_item(&bit)?
                .unwrap()
                .extract()?;
            locations.registers.bind(py).append((qreg, index))?;
        }
        Ok(())
    }

    /// Add all wires in a classical register.
    fn add_creg(&mut self, py: Python, creg: &Bound<PyAny>) -> PyResult<()> {
        if !creg.is_instance(self.circuit_module.classical_register.bind(py))? {
            return Err(DAGCircuitError::new_err(
                "not a ClassicalRegister instance.",
            ));
        }

        let register_name = creg.getattr(intern!(py, "name"))?;
        if self.cregs.bind(py).contains(&register_name)? {
            return Err(DAGCircuitError::new_err(format!(
                "duplicate register {}",
                register_name
            )));
        }
        self.cregs.bind(py).set_item(register_name, creg)?;

        for (index, bit) in creg.iter()?.enumerate() {
            let bit = bit?;
            if self.clbits.find(&bit).is_none() {
                self.add_clbit_unchecked(py, &bit)?;
            }
            let locations: PyRef<BitLocations> = self
                .clbit_locations
                .bind(py)
                .get_item(&bit)?
                .unwrap()
                .extract()?;
            locations.registers.bind(py).append((creg, index))?;
        }
        Ok(())
    }

    /// Finds locations in the circuit, by mapping the Qubit and Clbit to positional index
    /// BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))
    ///
    /// Args:
    ///     bit (Bit): The bit to locate.
    ///
    /// Returns:
    ///     namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
    ///         contains the index at which the ``Bit`` can be found (in either
    ///         :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its
    ///         type). The second element (``registers``) is a list of ``(register, index)``
    ///         pairs with an entry for each :obj:`~Register` in the circuit which contains the
    ///         :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).
    ///
    ///   Raises:
    ///     DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.
    ///     DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
    fn find_bit<'py>(&self, py: Python<'py>, bit: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if bit.is_instance(self.circuit_module.qubit.bind(py))? {
            return self.qubit_locations.bind(py).get_item(bit)?.ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                    bit
                ))
            });
        }

        if bit.is_instance(self.circuit_module.clbit.bind(py))? {
            return self.clbit_locations.bind(py).get_item(bit)?.ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                    bit
                ))
            });
        }

        Err(DAGCircuitError::new_err(format!(
            "Could not locate bit of unknown type: {}",
            bit.get_type()
        )))
    }

    /// Remove classical bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// Args:
    ///     clbits (List[Clbit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*clbits))]
    fn remove_clbits(&mut self, py: Python, clbits: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_bits = Vec::new();
        for bit in clbits.iter() {
            if !bit.is_instance(self.circuit_module.clbit.bind(py))? {
                non_bits.push(bit);
            }
        }
        if !non_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "clbits not of type Clbit: {:?}",
                non_bits
            )));
        }

        let clbits: IndexSet<Clbit> = self.clbits.map_bits(clbits)?.collect();
        let mut busy_bits = Vec::new();
        for bit in clbits.iter() {
            if !self.is_wire_idle(&Wire::Clbit(*bit))? {
                busy_bits.push(self.clbits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "clbits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut cregs_to_remove = Vec::new();
        for creg in self.cregs.bind(py).values() {
            for bit in creg.iter()? {
                let bit = bit?;
                if clbits.contains(&self.clbits.find(&bit).unwrap()) {
                    cregs_to_remove.push(creg);
                    break;
                }
            }
        }
        self.remove_cregs(py, &PyTuple::new_bound(py, cregs_to_remove))?;

        // Remove DAG in/out nodes etc.
        for bit in clbits.iter() {
            self.remove_idle_wire(Wire::Clbit(*bit))?;
        }

        // Update bit data.
        self.clbits.remove_indices(py, clbits)?;

        // Update bit locations.
        let bit_locations = self.clbit_locations.bind(py);
        for (i, bit) in self.clbits.bits().iter().enumerate() {
            bit_locations.set_item(
                bit,
                bit_locations
                    .get_item(bit)?
                    .unwrap()
                    .call_method1(intern!(py, "_replace"), (i,))?,
            )?;
        }
        Ok(())
    }

    /// Remove classical registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a creg is not a ClassicalRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*cregs))]
    fn remove_cregs(&mut self, py: Python, cregs: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_regs = Vec::new();
        let mut unknown_regs = Vec::new();
        for reg in cregs.iter() {
            if !reg.is_instance(self.circuit_module.classical_register.bind(py))? {
                non_regs.push(reg);
            } else if self.cregs.bind(py).values().contains(&reg)? {
                // TODO: make check not quadratic
                unknown_regs.push(reg);
            }
        }
        if !non_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "cregs not of type ClassicalRegister: {:?}",
                non_regs
            )));
        }
        if !unknown_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "cregs not in circuit: {:?}",
                unknown_regs
            )));
        }

        for creg in cregs {
            self.cregs
                .bind(py)
                .del_item(creg.getattr(intern!(py, "name"))?)?;
            for (i, bit) in creg.iter()?.enumerate() {
                let bit = bit?;
                let bit_position = self
                    .clbit_locations
                    .bind(py)
                    .get_item(bit)?
                    .unwrap()
                    .downcast_into_exact::<BitLocations>()?;
                bit_position
                    .borrow()
                    .registers
                    .bind(py)
                    .as_any()
                    .call_method1(intern!(py, "remove"), ((&creg, i),))?;
            }
        }
        Ok(())
    }

    /// Remove quantum bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// Args:
    ///     qubits (List[~qiskit.circuit.Qubit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a qubit is not a :obj:`~.circuit.Qubit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*qubits))]
    fn remove_qubits(&mut self, py: Python, qubits: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_bits = Vec::new();
        for bit in qubits.iter() {
            if !bit.is_instance(self.circuit_module.clbit.bind(py))? {
                non_bits.push(bit);
            }
        }
        if !non_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qubits not of type Qubit: {:?}",
                non_bits
            )));
        }

        let qubits: IndexSet<Qubit> = self.qubits.map_bits(qubits)?.collect();
        let mut busy_bits = Vec::new();
        for bit in qubits.iter() {
            if !self.is_wire_idle(&Wire::Qubit(*bit))? {
                busy_bits.push(self.qubits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qubits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut qregs_to_remove = Vec::new();
        for qreg in self.qregs.bind(py).values() {
            for bit in qreg.iter()? {
                let bit = bit?;
                if qubits.contains(&self.qubits.find(&bit).unwrap()) {
                    qregs_to_remove.push(qreg);
                    break;
                }
            }
        }
        self.remove_qregs(py, &PyTuple::new_bound(py, qregs_to_remove))?;

        // Remove DAG in/out nodes etc.
        for bit in qubits.iter() {
            self.remove_idle_wire(Wire::Qubit(*bit))?;
        }

        // Update bit data.
        self.qubits.remove_indices(py, qubits)?;

        // Update bit locations.
        let bit_locations = self.qubit_locations.bind(py);
        for (i, bit) in self.qubits.bits().iter().enumerate() {
            bit_locations.set_item(
                bit,
                bit_locations
                    .get_item(bit)?
                    .unwrap()
                    .call_method1(intern!(py, "_replace"), (i,))?,
            )?;
        }
        Ok(())
    }

    /// Remove quantum registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a qreg is not a QuantumRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*qregs))]
    fn remove_qregs(&mut self, py: Python, qregs: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_regs = Vec::new();
        let mut unknown_regs = Vec::new();
        for reg in qregs.iter() {
            if !reg.is_instance(self.circuit_module.quantum_register.bind(py))? {
                non_regs.push(reg);
            } else if self.qregs.bind(py).values().contains(&reg)? {
                // TODO: make check not quadratic
                unknown_regs.push(reg);
            }
        }
        if !non_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qregs not of type QuantumRegister: {:?}",
                non_regs
            )));
        }
        if !unknown_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qregs not in circuit: {:?}",
                unknown_regs
            )));
        }

        for qreg in qregs {
            self.qregs
                .bind(py)
                .del_item(qreg.getattr(intern!(py, "name"))?)?;
            for (i, bit) in qreg.iter()?.enumerate() {
                let bit = bit?;
                let bit_position = self
                    .qubit_locations
                    .bind(py)
                    .get_item(bit)?
                    .unwrap()
                    .downcast_into_exact::<BitLocations>()?;
                bit_position
                    .borrow()
                    .registers
                    .bind(py)
                    .as_any()
                    .call_method1(intern!(py, "remove"), ((&qreg, i),))?;
            }
        }
        Ok(())
    }

    /// Verify that the condition is valid.
    ///
    /// Args:
    ///     name (string): used for error reporting
    ///     condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)
    ///
    /// Raises:
    ///     DAGCircuitError: if conditioning on an invalid register
    fn _check_condition(&self, py: Python, name: &str, condition: &Bound<PyAny>) -> PyResult<()> {
        if condition.is_none() {
            return Ok(());
        }

        let resources = self.control_flow_module.condition_resources(condition)?;
        for reg in resources.cregs.bind(py) {
            if !self
                .cregs
                .bind(py)
                .contains(reg.getattr(intern!(py, "name"))?)?
            {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid creg in condition for {}",
                    name
                )));
            }
        }

        for bit in resources.clbits.bind(py) {
            if self.clbits.find(&bit).is_none() {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid clbits in condition for {}",
                    name
                )));
            }
        }

        Ok(())
    }

    /// Return a copy of self with the same structure but empty.
    ///
    /// That structure includes:
    ///     * name and other metadata
    ///     * global phase
    ///     * duration
    ///     * all the qubits and clbits, including the registers.
    ///
    /// Returns:
    ///     DAGCircuit: An empty copy of self.
    fn copy_empty_like(&self, py: Python) -> PyResult<Self> {
        let mut target_dag = DAGCircuit::new(py)?;
        target_dag.name = self.name.as_ref().map(|n| n.clone_ref(py));
        target_dag.global_phase = self.global_phase.clone_ref(py);
        target_dag.duration = self.duration.as_ref().map(|d| d.clone_ref(py));
        target_dag.unit = self.unit.clone();
        target_dag.metadata = self.metadata.as_ref().map(|m| m.clone_ref(py));
        target_dag.qargs_cache = self.qargs_cache.clone();
        target_dag.cargs_cache = self.cargs_cache.clone();

        for bit in self.qubits.bits() {
            target_dag.add_qubit_unchecked(py, bit.bind(py))?;
        }
        for bit in self.clbits.bits() {
            target_dag.add_clbit_unchecked(py, bit.bind(py))?;
        }
        for reg in self.qregs.bind(py).values() {
            target_dag.add_qreg(py, &reg)?;
        }
        for reg in self.cregs.bind(py).values() {
            target_dag.add_creg(py, &reg)?;
        }
        Ok(target_dag)
    }

    /// Apply an operation to the output of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if a leaf node is connected to multiple outputs
    #[pyo3(name = "apply_operation_back", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_back(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let old_op = op.clone().unbind();
        let op = convert_py_to_operation_type(py, op.unbind())?;
        let qargs = qargs.map(|q| q.value);
        let cargs = cargs.map(|c| c.value);
        let node = {
            let qubits_id = Interner::intern(
                &mut self.qargs_cache,
                self.qubits.map_bits(qargs.iter().flatten())?.collect(),
            )?;
            let clbits_id = Interner::intern(
                &mut self.cargs_cache,
                self.clbits.map_bits(cargs.iter().flatten())?.collect(),
            )?;
            let instr = PackedInstruction::new(
                op.operation,
                qubits_id,
                clbits_id,
                op.params,
                op.label,
                op.duration,
                op.unit,
                op.condition,
                #[cfg(feature = "cache_pygates")]
                Some(old_op.clone_ref(py)),
            );

            if check {
                if let Some(condition) = instr.condition() {
                    self._check_condition(py, instr.op.name(), condition.bind(py))?;
                }

                for b in self.qargs_cache.intern(instr.qubits_id) {
                    if !self.qubit_output_map.contains_key(b) {
                        return Err(DAGCircuitError::new_err(format!(
                            "qubit {} not found in output map",
                            self.qubits.get(*b).unwrap()
                        )));
                    }
                }

                for b in self.cargs_cache.intern(instr.clbits_id) {
                    if !self.clbit_output_map.contains_key(b) {
                        return Err(DAGCircuitError::new_err(format!(
                            "clbit {} not found in output map",
                            self.clbits.get(*b).unwrap()
                        )));
                    }
                }

                if self.may_have_additional_wires(py, &instr) {
                    let (clbits, vars) = self.additional_wires(py, &instr)?;
                    for b in clbits {
                        if !self.clbit_output_map.contains_key(&b) {
                            return Err(DAGCircuitError::new_err(format!(
                                "clbit {} not found in output map",
                                self.clbits.get(b).unwrap()
                            )));
                        }
                    }
                    for v in vars {
                        if !self.var_output_map.contains_key(&v) {
                            return Err(DAGCircuitError::new_err(format!(
                                "var {} not found in output map",
                                v
                            )));
                        }
                    }
                }
            }
            self.push_back(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Apply an operation to the input of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if initial nodes connected to multiple out edges
    #[pyo3(name = "apply_operation_front", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_front(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let old_op = op.clone().unbind();
        let op = convert_py_to_operation_type(py, op.unbind())?;
        let qargs = qargs.map(|q| q.value);
        let cargs = cargs.map(|c| c.value);
        let node = {
            let qubits_id = Interner::intern(
                &mut self.qargs_cache,
                self.qubits.map_bits(qargs.iter().flatten())?.collect(),
            )?;
            let clbits_id = Interner::intern(
                &mut self.cargs_cache,
                self.clbits.map_bits(cargs.iter().flatten())?.collect(),
            )?;
            let instr = PackedInstruction::new(
                op.operation,
                qubits_id,
                clbits_id,
                op.params,
                op.label,
                op.duration,
                op.unit,
                op.condition,
                #[cfg(feature = "cache_pygates")]
                Some(old_op.clone_ref(py)),
            );

            if check {
                if let Some(condition) = instr.condition() {
                    self._check_condition(py, instr.op.name(), condition.bind(py))?;
                }

                for b in self.qargs_cache.intern(instr.qubits_id) {
                    if !self.qubit_output_map.contains_key(b) {
                        return Err(DAGCircuitError::new_err(format!(
                            "qubit {} not found in output map",
                            self.qubits.get(*b).unwrap()
                        )));
                    }
                }

                for b in self.cargs_cache.intern(instr.clbits_id) {
                    if !self.clbit_output_map.contains_key(b) {
                        return Err(DAGCircuitError::new_err(format!(
                            "clbit {} not found in output map",
                            self.clbits.get(*b).unwrap()
                        )));
                    }
                }

                if self.may_have_additional_wires(py, &instr) {
                    let (clbits, vars) = self.additional_wires(py, &instr)?;
                    for b in clbits {
                        if !self.clbit_output_map.contains_key(&b) {
                            return Err(DAGCircuitError::new_err(format!(
                                "clbit {} not found in output map",
                                self.clbits.get(b).unwrap()
                            )));
                        }
                    }
                    for v in vars {
                        if !self.var_output_map.contains_key(&v) {
                            return Err(DAGCircuitError::new_err(format!(
                                "var {} not found in output map",
                                v
                            )));
                        }
                    }
                }
            }
            self.push_front(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Compose the ``other`` circuit onto the output of this circuit.
    ///
    /// A subset of input wires of ``other`` are mapped
    /// to a subset of output wires of this circuit.
    ///
    /// ``other`` can be narrower or of equal width to ``self``.
    ///
    /// Args:
    ///     other (DAGCircuit): circuit to compose with self
    ///     qubits (list[~qiskit.circuit.Qubit|int]): qubits of self to compose onto.
    ///     clbits (list[Clbit|int]): clbits of self to compose onto.
    ///     front (bool): If True, front composition will be performed (not implemented yet)
    ///     inplace (bool): If True, modify the object. Otherwise return composed circuit.
    ///
    /// Returns:
    ///     DAGCircuit: the composed dag (returns None if inplace==True).
    ///
    /// Raises:
    ///     DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
    #[pyo3(signature = (other, qubits=None, clbits=None, front=false, inplace=true))]
    fn compose(
        slf: PyRefMut<Self>,
        py: Python,
        other: &DAGCircuit,
        qubits: Option<Bound<PyList>>,
        clbits: Option<Bound<PyList>>,
        front: bool,
        inplace: bool,
    ) -> PyResult<Option<PyObject>> {
        if front {
            return Err(DAGCircuitError::new_err(
                "Front composition not supported yet.",
            ));
        }

        if other.qubits.len() > slf.qubits.len() || other.clbits.len() > slf.clbits.len() {
            return Err(DAGCircuitError::new_err(
                "Trying to compose with another DAGCircuit which has more 'in' edges.",
            ));
        }

        // Number of qubits and clbits must match number in circuit or None
        let identity_qubit_map = other
            .qubits
            .bits()
            .iter()
            .zip(slf.qubits.bits())
            .into_py_dict_bound(py);
        let identity_clbit_map = other
            .clbits
            .bits()
            .iter()
            .zip(slf.clbits.bits())
            .into_py_dict_bound(py);

        let qubit_map: Bound<PyDict> = match qubits {
            None => identity_qubit_map.clone(),
            Some(qubits) => {
                if qubits.len() != other.qubits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in qubits parameter does not",
                        " match number of qubits in the circuit."
                    )));
                }

                let self_qubits = slf.qubits.cached().bind(py);
                let other_qubits = other.qubits.cached().bind(py);
                let dict = PyDict::new_bound(py);
                for (i, q) in qubits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_qubits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_qubits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let clbit_map: Bound<PyDict> = match clbits {
            None => identity_clbit_map.clone(),
            Some(clbits) => {
                if clbits.len() != other.clbits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in clbits parameter does not",
                        " match number of clbits in the circuit."
                    )));
                }

                let self_clbits = slf.clbits.cached().bind(py);
                let other_clbits = other.clbits.cached().bind(py);
                let dict = PyDict::new_bound(py);
                for (i, q) in clbits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_clbits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_clbits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let edge_map = if qubit_map.is_empty() && clbit_map.is_empty() {
            // try to do a 1-1 mapping in order
            identity_qubit_map
                .iter()
                .chain(identity_clbit_map.iter())
                .into_py_dict_bound(py)
        } else {
            qubit_map
                .iter()
                .chain(clbit_map.iter())
                .into_py_dict_bound(py)
        };

        // Chck duplicates in wire map.
        {
            let edge_map_values: Vec<_> = edge_map.values().iter().collect();
            if PySet::new_bound(py, edge_map_values.as_slice())?.len() != edge_map.len() {
                return Err(DAGCircuitError::new_err("duplicates in wire_map"));
            }
        }

        // Compose
        let mut dag: PyRefMut<DAGCircuit> = if inplace {
            slf
        } else {
            Py::new(py, slf.clone())?.into_bound(py).borrow_mut()
        };

        dag.global_phase = dag.global_phase.bind(py).add(&other.global_phase)?.unbind();

        for (gate, cals) in other.calibrations.iter() {
            dag.calibrations[gate]
                .bind(py)
                .update(&cals.bind(py).as_mapping())?;
        }

        let variable_mapper = PyVariableMapper::new(
            py,
            dag.cregs.bind(py).values().into_any(),
            Some(edge_map.clone()),
            None,
            Some(wrap_pyfunction_bound!(reject_new_register, py)?.to_object(py)),
        )?;

        for node in other.topological_nodes()? {
            match &other.dag[node] {
                NodeType::QubitIn(q) => {
                    let bit = other.qubits.get(*q).unwrap().bind(py);
                    let m_wire = edge_map.get_item(bit)?.unwrap_or_else(|| bit.clone());
                    let bit_in_dag = dag.qubits.find(bit);
                    if bit_in_dag.is_none()
                        || !dag.qubit_output_map.contains_key(&bit_in_dag.unwrap())
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {}[{}] not in self",
                            m_wire.getattr("name")?,
                            m_wire.getattr("index")?
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::ClbitIn(c) => {
                    let bit = other.clbits.get(*c).unwrap().bind(py);
                    let m_wire = edge_map.get_item(bit)?.unwrap_or_else(|| bit.clone());
                    let bit_in_dag = dag.clbits.find(bit);
                    if bit_in_dag.is_none()
                        || !dag.clbit_output_map.contains_key(&bit_in_dag.unwrap())
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {}[{}] not in self",
                            m_wire.getattr("name")?,
                            m_wire.getattr("index")?
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::Operation(op) => {
                    let m_qargs = {
                        let qubits = other
                            .qubits
                            .map_indices(other.qargs_cache.intern(op.qubits_id).as_slice());
                        let mut mapped = Vec::with_capacity(qubits.len());
                        for bit in qubits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .unwrap_or_else(|| bit.bind(py).clone()),
                            );
                        }
                        PyTuple::new_bound(py, mapped)
                    };
                    let m_cargs = {
                        let clbits = other
                            .clbits
                            .map_indices(other.cargs_cache.intern(op.clbits_id).as_slice());
                        let mut mapped = Vec::with_capacity(clbits.len());
                        for bit in clbits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .unwrap_or_else(|| bit.bind(py).clone()),
                            );
                        }
                        PyTuple::new_bound(py, mapped)
                    };

                    let mut py_op = op.unpack_py_op(py)?.into_bound(py);
                    if let Some(condition) = op.condition() {
                        // TODO: do we need to check for condition.is_none()?
                        let condition = variable_mapper.map_condition(condition.bind(py), true)?;
                        if !op.op.control_flow() {
                            py_op = py_op.call_method1(
                                intern!(py, "c_if"),
                                condition.downcast::<PyTuple>()?,
                            )?;
                        } else {
                            py_op.setattr(intern!(py, "condition"), condition)?;
                        }
                    } else if py_op.is_instance(SWITCH_CASE_OP.get_bound(py))? {
                        py_op.setattr(
                            intern!(py, "target"),
                            variable_mapper.map_target(&py_op.getattr(intern!(py, "target"))?)?,
                        )?;
                    };

                    dag.py_apply_operation_back(
                        py,
                        py_op,
                        Some(TupleLikeArg { value: m_qargs }),
                        Some(TupleLikeArg { value: m_cargs }),
                        false,
                    )?;
                }
                NodeType::QubitOut(_) | NodeType::ClbitOut(_) => (),
            }
        }
        // if qubits is None:
        //     qubit_map = identity_qubit_map
        // elif len(qubits) != len(other.qubits):
        //     raise DAGCircuitError
        // else:
        //     qubit_map = {
        //         other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
        //         for i, q in enumerate(qubits)
        //     }
        // if clbits is None:
        //     clbit_map = identity_clbit_map
        // elif len(clbits) != len(other.clbits):
        //     raise DAGCircuitError(
        //         "Number of items in clbits parameter does not"
        //         " match number of clbits in the circuit."
        //     )
        // else:
        //     clbit_map = {
        //         other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
        //         for i, c in enumerate(clbits)
        //     }
        // edge_map = {**qubit_map, **clbit_map} or None
        //
        // # if no edge_map, try to do a 1-1 mapping in order
        // if edge_map is None:
        //     edge_map = {**identity_qubit_map, **identity_clbit_map}
        //
        // # Check the edge_map for duplicate values
        // if len(set(edge_map.values())) != len(edge_map):
        //     raise DAGCircuitError("duplicates in wire_map")
        //
        // # Compose
        // if inplace:
        //     dag = self
        // else:
        //     dag = copy.deepcopy(self)
        // dag.global_phase += other.global_phase
        //
        // for gate, cals in other.calibrations.items():
        //     dag._calibrations[gate].update(cals)
        //
        // # Ensure that the error raised here is a `DAGCircuitError` for backwards compatibility.
        // def _reject_new_register(reg):
        //     raise DAGCircuitError(f"No register with '{reg.bits}' to map this expression onto.")
        //
        // variable_mapper = _classical_resource_map.VariableMapper(
        //     dag.cregs.values(), edge_map, _reject_new_register
        // )
        // for nd in other.topological_nodes():
        //     if isinstance(nd, DAGInNode):
        //         # if in edge_map, get new name, else use existing name
        //         m_wire = edge_map.get(nd.wire, nd.wire)
        //         # the mapped wire should already exist
        //         if m_wire not in dag.output_map:
        //             raise DAGCircuitError(
        //                 "wire %s[%d] not in self" % (m_wire.register.name, m_wire.index)
        //             )
        //         if nd.wire not in other._wires:
        //             raise DAGCircuitError(
        //                 "inconsistent wire type for %s[%d] in other"
        //                 % (nd.register.name, nd.wire.index)
        //             )
        //     elif isinstance(nd, DAGOutNode):
        //         # ignore output nodes
        //         pass
        //     elif isinstance(nd, DAGOpNode):
        //         m_qargs = [edge_map.get(x, x) for x in nd.qargs]
        //         m_cargs = [edge_map.get(x, x) for x in nd.cargs]
        //         op = nd.op.copy()
        //         if (condition := getattr(op, "condition", None)) is not None:
        //             if not isinstance(op, ControlFlowOp):
        //                 op = op.c_if(*variable_mapper.map_condition(condition, allow_reorder=True))
        //             else:
        //                 op.condition = variable_mapper.map_condition(condition, allow_reorder=True)
        //         elif isinstance(op, SwitchCaseOp):
        //             op.target = variable_mapper.map_target(op.target)
        //         dag.apply_operation_back(op, m_qargs, m_cargs, check=False)
        //     else:
        //         raise DAGCircuitError("bad node type %s" % type(nd))
        //
        // if not inplace:
        //     return dag
        // else:
        //     return None

        if !inplace {
            Ok(Some(dag.into_py(py)))
        } else {
            Ok(None)
        }
    }

    /// Reverse the operations in the ``self`` circuit.
    ///
    /// Returns:
    ///     DAGCircuit: the reversed dag.
    fn reverse_ops(&mut self) -> PyResult<()> {
        // # TODO: speed up
        // # pylint: disable=cyclic-import
        // from qiskit.converters import dag_to_circuit, circuit_to_dag
        //
        // qc = dag_to_circuit(self)
        // reversed_qc = qc.reverse_ops()
        // reversed_dag = circuit_to_dag(reversed_qc)
        // return reversed_dag
        todo!()
    }

    /// Return idle wires.
    ///
    /// Args:
    ///     ignore (list(str)): List of node names to ignore. Default: []
    ///
    /// Yields:
    ///     Bit: Bit in idle wire.
    ///
    /// Raises:
    ///     DAGCircuitError: If the DAG is invalid
    fn idle_wires(&self, py: Python, ignore: Option<&Bound<PyList>>) -> PyResult<Py<PyIterator>> {
        let mut result: Vec<PyObject> = Vec::new();
        let wires = self
            .qubit_input_map
            .keys()
            .cloned()
            .map(Wire::Qubit)
            .chain(self.clbit_input_map.keys().cloned().map(Wire::Clbit))
            .chain(self.var_input_map.keys().map(Wire::Var));
        match ignore {
            Some(ignore) => {
                // Convert the list to a Rust set.
                let ignore_set = ignore
                    .into_iter()
                    .map(|s| s.extract())
                    .collect::<PyResult<HashSet<String>>>()?;
                for wire in wires {
                    let nodes_found = self.nodes_on_wire(&wire, true).into_iter().any(|node| {
                        let weight = self.dag.node_weight(node).unwrap();
                        if let NodeType::Operation(packed) = weight {
                            !ignore_set.contains(&packed.op.name().to_string())
                        } else {
                            false
                        }
                    });

                    if !nodes_found {
                        result.push(match wire {
                            Wire::Qubit(qubit) => self.qubits.get(qubit).unwrap().clone_ref(py),
                            Wire::Clbit(clbit) => self.clbits.get(clbit).unwrap().clone_ref(py),
                            Wire::Var(var) => var,
                        });
                    }
                }
            }
            None => {
                for wire in wires {
                    if self.is_wire_idle(&wire)? {
                        result.push(match wire {
                            Wire::Qubit(qubit) => self.qubits.get(qubit).unwrap().clone_ref(py),
                            Wire::Clbit(clbit) => self.clbits.get(clbit).unwrap().clone_ref(py),
                            Wire::Var(var) => var,
                        });
                    }
                }
            }
        }
        Ok(PyTuple::new_bound(py, result).into_any().iter()?.unbind())
    }

    /// Return the number of operations.  If there is control flow present, this count may only
    /// be an estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops with
    ///         known-length iterators are counted unrolled.  If-else blocks sum both of the two
    ///         branches.  While loops are counted as if the loop body runs once only.  Defaults to
    ///         ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to
    ///         avoid silently returning a mostly meaningless number.
    ///
    /// Returns:
    ///     int: the circuit size
    ///
    /// Raises:
    ///     DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with
    ///         ``recurse=True``, or any control flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn size(&self, py: Python, recurse: bool) -> PyResult<usize> {
        let length = self.dag.node_count() - self.width() * 2;
        if !recurse {
            // TODO: do this once in module struct `new`.
            let control_flow_op_names: PyResult<Vec<String>> = self
                .control_flow_module
                .control_flow_op_names
                .bind(py)
                .iter()
                .map(|s| s.extract())
                .collect();
            if control_flow_op_names?
                .into_iter()
                .any(|n| self.op_names.contains_key(&n))
            {
                return Err(DAGCircuitError::new_err(concat!(
                    "Size with control flow is ambiguous.",
                    " You may use `recurse=True` to get a result",
                    " but see this method's documentation for the meaning of this."
                )));
            }
            return Ok(length);
        }

        // length = len(self._multi_graph) - 2 * len(self._wires)
        // if not recurse:
        //     if any(x in self._op_names for x in CONTROL_FLOW_OP_NAMES):
        //         raise DAGCircuitError(
        //             "Size with control flow is ambiguous."
        //             " You may use `recurse=True` to get a result,"
        //             " but see this method's documentation for the meaning of this."
        //         )
        //     return length
        // # pylint: disable=cyclic-import
        // from qiskit.converters import circuit_to_dag
        //
        // for node in self.op_nodes(ControlFlowOp):
        //     if isinstance(node.op, ForLoopOp):
        //         indexset = node.op.params[0]
        //         inner = len(indexset) * circuit_to_dag(node.op.blocks[0]).size(recurse=True)
        //     elif isinstance(node.op, WhileLoopOp):
        //         inner = circuit_to_dag(node.op.blocks[0]).size(recurse=True)
        //     elif isinstance(node.op, (IfElseOp, SwitchCaseOp)):
        //         inner = sum(circuit_to_dag(block).size(recurse=True) for block in node.op.blocks)
        //     else:
        //         raise DAGCircuitError(f"unknown control-flow type: '{node.op.name}'")
        //     # Replace the "1" for the node itself with the actual count.
        //     length += inner - 1
        // return length
        todo!()
    }

    /// Return the circuit depth.  If there is control flow present, this count may only be an
    /// estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops
    ///         with known-length iterators are counted as if the loop had been manually unrolled
    ///         (*i.e.* with each iteration of the loop body written out explicitly).
    ///         If-else blocks take the longer case of the two branches.  While loops are counted as
    ///         if the loop body runs once only.  Defaults to ``False`` and raises
    ///         :class:`.DAGCircuitError` if any control flow is present, to avoid silently
    ///         returning a nonsensical number.
    ///
    /// Returns:
    ///     int: the circuit depth
    ///
    /// Raises:
    ///     DAGCircuitError: if not a directed acyclic graph
    ///     DAGCircuitError: if unknown control flow is present in a recursive call, or any control
    ///         flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn depth(&self, recurse: bool) -> PyResult<usize> {
        // if recurse:
        //     from qiskit.converters import circuit_to_dag  # pylint: disable=cyclic-import
        //
        //     node_lookup = {}
        //     for node in self.op_nodes(ControlFlowOp):
        //         weight = len(node.op.params[0]) if isinstance(node.op, ForLoopOp) else 1
        //         if weight == 0:
        //             node_lookup[node._node_id] = 0
        //         else:
        //             node_lookup[node._node_id] = weight * max(
        //                 circuit_to_dag(block).depth(recurse=True) for block in node.op.blocks
        //             )
        //
        //     def weight_fn(_source, target, _edge):
        //         return node_lookup.get(target, 1)
        //
        // else:
        //     if any(x in self._op_names for x in CONTROL_FLOW_OP_NAMES):
        //         raise DAGCircuitError(
        //             "Depth with control flow is ambiguous."
        //             " You may use `recurse=True` to get a result,"
        //             " but see this method's documentation for the meaning of this."
        //         )
        //     weight_fn = None
        //
        // try:
        //     depth = rx.dag_longest_path_length(self._multi_graph, weight_fn) - 1
        // except rx.DAGHasCycle as ex:
        //     raise DAGCircuitError("not a DAG") from ex
        // return depth if depth >= 0 else 0
        todo!()
    }

    /// Return the total number of qubits + clbits used by the circuit.
    /// This function formerly returned the number of qubits by the calculation
    /// return len(self._wires) - self.num_clbits()
    /// but was changed by issue #2564 to return number of qubits + clbits
    /// with the new function DAGCircuit.num_qubits replacing the former
    /// semantic of DAGCircuit.width().
    fn width(&self) -> usize {
        self.qubits.len() + self.clbits.len()
    }

    /// Return the total number of qubits used by the circuit.
    /// num_qubits() replaces former use of width().
    /// DAGCircuit.width() now returns qubits + clbits for
    /// consistency with Circuit.width() [qiskit-terra #2564].
    fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Return the total number of classical bits used by the circuit.
    fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Compute how many components the circuit can decompose into.
    fn num_tensor_factors(&self) -> usize {
        // return rx.number_weakly_connected_components(self._multi_graph)
        todo!()
    }

    fn __eq__(&self, py: Python, other: &DAGCircuit) -> PyResult<bool> {
        // Try to convert to float, but in case of unbound ParameterExpressions
        // a TypeError will be raise, fallback to normal equality in those
        // cases.
        let self_phase = match self
            .global_phase
            .bind(py)
            .call_method0(intern!(py, "__float__"))
        {
            Err(e) if !e.is_instance_of::<PyTypeError>(py) => {
                return Err(e);
            }
            res => res.ok(),
        };
        let other_phase = match other
            .global_phase
            .bind(py)
            .call_method0(intern!(py, "__float__"))
        {
            Err(e) if !e.is_instance_of::<PyTypeError>(py) => {
                return Err(e);
            }
            res => res.ok(),
        };
        match (self_phase, other_phase) {
            (Some(self_phase), Some(other_phase)) => {
                let self_phase: f64 = self_phase.extract()?;
                let other_phase: f64 = other_phase.extract()?;
                if (((self_phase - other_phase + PI) % (2.0 * PI)) - PI).abs() > 1.0e-10 {
                    return Ok(false);
                }
            }
            _ => {
                if !self.global_phase.bind(py).eq(other.global_phase.bind(py))? {
                    return Ok(false);
                }
            }
        }

        if self.calibrations.len() != other.calibrations.len() {
            return Ok(false);
        }

        for (k, v1) in &self.calibrations {
            match other.calibrations.get(k) {
                Some(v2) => {
                    if !v1.bind(py).eq(v2.bind(py))? {
                        return Ok(false);
                    }
                }
                None => {
                    return Ok(false);
                }
            }
        }

        let self_bit_indices = {
            let indices = self
                .qubits
                .bits()
                .iter()
                .chain(self.clbits.bits())
                .enumerate()
                .map(|(idx, bit)| (bit, idx));
            indices.into_py_dict_bound(py)
        };

        let other_bit_indices = {
            let indices = other
                .qubits
                .bits()
                .iter()
                .chain(other.clbits.bits())
                .enumerate()
                .map(|(idx, bit)| (bit, idx));
            indices.into_py_dict_bound(py)
        };

        // Check if qregs are the same.
        let self_qregs = self.qregs.bind(py);
        let other_qregs = other.qregs.bind(py);
        if self_qregs.len() != other_qregs.len() {
            return Ok(false);
        }
        for (regname, self_bits) in self_qregs {
            let self_bits = self_bits.downcast_into_exact::<PyList>()?;
            let other_bits = match other_qregs.get_item(regname)? {
                Some(bits) => bits.downcast_into_exact::<PyList>()?,
                None => return Ok(false),
            };
            if !self
                .qubits
                .map_bits(self_bits)?
                .eq(other.qubits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check if cregs are the same.
        let self_cregs = self.cregs.bind(py);
        let other_cregs = other.cregs.bind(py);
        if self_cregs.len() != other_cregs.len() {
            return Ok(false);
        }

        for (regname, self_bits) in self_cregs {
            let self_bits = self_bits.downcast_into_exact::<PyList>()?;
            let other_bits = match other_cregs.get_item(regname)? {
                Some(bits) => bits.downcast_into_exact::<PyList>()?,
                None => return Ok(false),
            };
            if !self
                .clbits
                .map_bits(self_bits)?
                .eq(other.clbits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check for VF2 isomorphic match.
        let semantic_eq = DAG_NODE.get_bound(py).getattr(intern!(py, "semantic_eq"))?;
        let node_match = |n1: &NodeType, n2: &NodeType| -> PyResult<bool> {
            // Note: we pretend that the node IDs are 0, since we know that semantic_eq
            // doesn't use node IDs in its comparison. We should eventually port
            // semantic_eq to Rust to entirely skip conversion to Python DAGNodes.
            let n1 = self.unpack_into(py, NodeIndex::new(0), n1)?;
            let n2 = self.unpack_into(py, NodeIndex::new(0), n2)?;
            Ok(semantic_eq
                .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                .extract()?)
        };

        isomorphism::vf2::is_isomorphic(
            &self.dag,
            &other.dag,
            node_match,
            isomorphism::vf2::NoSemanticMatch,
            true,
            Ordering::Equal,
            true,
            None,
        )
        .map_err(|e| match e {
            isomorphism::vf2::IsIsomorphicError::NodeMatcherErr(e) => e,
            _ => {
                unreachable!()
            }
        })
    }

    /// Yield nodes in topological order.
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the
    ///         :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
    ///         used as the sort key for each node.
    ///
    /// Returns:
    ///     generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
    #[pyo3(name = "topological_nodes")]
    fn py_topological_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            // This path (user provided key func) is not ideal, since we no longer
            // use a string key after moving to Rust, in favor of using a tuple
            // of the qargs and cargs interner IDs of the node.
            let key = |node: NodeIndex| -> PyResult<String> {
                let node = self.get_node(py, node)?;
                Ok(key.call1((node,))?.extract()?)
            };
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(ref e) => {
                        e.clone_ref(py)
                    }
                })?
                .into_iter()
                .map(|n| self.get_node(py, n))
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_nodes()?
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new_bound(py, nodes?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Yield op nodes in topological order.
    ///
    /// Allowed to pass in specific key to break ties in top order
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the
    ///         :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
    ///         used as the sort key for each node.
    ///
    /// Returns:
    ///     generator(DAGOpNode): op node in topological order
    fn topological_op_nodes(&self, key: Option<Bound<PyAny>>) -> PyResult<Py<PyIterator>> {
        // return (nd for nd in self.topological_nodes(key) if isinstance(nd, DAGOpNode))
        todo!()
    }

    /// Replace a block of nodes with a single node.
    ///
    /// This is used to consolidate a block of DAGOpNodes into a single
    /// operation. A typical example is a block of gates being consolidated
    /// into a single ``UnitaryGate`` representing the unitary matrix of the
    /// block.
    ///
    /// Args:
    ///     node_block (List[DAGNode]): A list of dag nodes that represents the
    ///         node block to be replaced
    ///     op (qiskit.circuit.Operation): The operation to replace the
    ///         block with
    ///     wire_pos_map (Dict[Bit, int]): The dictionary mapping the bits to their positions in the
    ///         output ``qargs`` or ``cargs``. This is necessary to reconstruct the arg order over
    ///         multiple gates in the combined single op node.  If a :class:`.Bit` is not in the
    ///         dictionary, it will not be added to the args; this can be useful when dealing with
    ///         control-flow operations that have inherent bits in their ``condition`` or ``target``
    ///         fields.
    ///     cycle_check (bool): When set to True this method will check that
    ///         replacing the provided ``node_block`` with a single node
    ///         would introduce a cycle (which would invalidate the
    ///         ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle
    ///         would be introduced. This checking comes with a run time
    ///         penalty. If you can guarantee that your input ``node_block`` is
    ///         a contiguous block and won't introduce a cycle when it's
    ///         contracted to a single node, this can be set to ``False`` to
    ///         improve the runtime performance of this method.
    ///
    /// Raises:
    ///     DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing
    ///         the specified block introduces a cycle or if ``node_block`` is
    ///         empty.
    ///
    /// Returns:
    ///     DAGOpNode: The op node that replaces the block.
    #[pyo3(signature = (node_block, op, wire_pos_map, cycle_check=true))]
    fn replace_block_with_op(
        &mut self,
        py: Python,
        node_block: Vec<PyRef<DAGNode>>,
        op: Bound<PyAny>,
        wire_pos_map: &Bound<PyDict>,
        cycle_check: bool,
    ) -> PyResult<Py<PyAny>> {
        // If node block is empty return early
        if node_block.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Can't replace an empty 'node_block'",
            ));
        }

        let mut qubit_pos_map: HashMap<Qubit, usize> = HashMap::new();
        let mut clbit_pos_map: HashMap<Clbit, usize> = HashMap::new();
        for (bit, index) in wire_pos_map.iter() {
            if bit.is_instance(self.circuit_module.qubit.bind(py))? {
                qubit_pos_map.insert(self.qubits.find(&bit).unwrap(), index.extract()?);
            } else if bit.is_instance(self.circuit_module.clbit.bind(py))? {
                clbit_pos_map.insert(self.clbits.find(&bit).unwrap(), index.extract()?);
            } else {
                return Err(DAGCircuitError::new_err(
                    "Wire map keys must be Qubit or Clbit instances.",
                ));
            }
        }

        let block_ids: Vec<_> = node_block.iter().map(|n| n.node.unwrap()).collect();

        let mut block_op_names = Vec::new();
        let mut block_qargs: IndexSet<Qubit> = IndexSet::new();
        let mut block_cargs: IndexSet<Clbit> = IndexSet::new();
        for nd in &block_ids {
            let weight = self.dag.node_weight(*nd);
            match weight {
                Some(NodeType::Operation(packed)) => {
                    block_op_names.push(packed.op.name().to_string());
                    block_qargs.extend(self.qargs_cache.intern(packed.qubits_id));
                    block_cargs.extend(self.cargs_cache.intern(packed.clbits_id));

                    let condition = packed
                        .extra_attrs
                        .iter()
                        .flat_map(|e| e.condition.as_ref().map(|c| c.bind(py)))
                        .next();
                    if let Some(condition) = condition {
                        block_cargs.extend(
                            self.clbits.map_bits(
                                self.control_flow_module
                                    .condition_resources(condition)?
                                    .clbits
                                    .bind(py),
                            )?,
                        );
                        continue;
                    }

                    // Add classical bits from SwitchCaseOp, if applicable.
                    if let OperationType::Instruction(ref op) = packed.op {
                        let op = op.instruction.bind(py);
                        if op.is_instance(self.circuit_module.switch_case_op.bind(py))? {
                            let target = op.getattr(intern!(py, "target"))?;
                            if target.is_instance(self.circuit_module.clbit.bind(py))? {
                                block_cargs.insert(self.clbits.find(&target).unwrap());
                            } else if target
                                .is_instance(self.circuit_module.classical_register.bind(py))?
                            {
                                block_cargs.extend(
                                    self.clbits
                                        .map_bits(target.extract::<Vec<Bound<PyAny>>>()?)?,
                                );
                            } else {
                                block_cargs.extend(
                                    self.clbits.map_bits(
                                        self.control_flow_module
                                            .node_resources(&target)?
                                            .clbits
                                            .bind(py),
                                    )?,
                                );
                            }
                        }
                    }
                }
                Some(_) => {
                    return Err(DAGCircuitError::new_err(
                        "Nodes in 'node_block' must be of type 'DAGOpNode'.",
                    ))
                }
                None => {
                    return Err(DAGCircuitError::new_err(
                        "Node in 'node_block' not found in DAG.",
                    ))
                }
            }
        }

        let mut block_qargs: Vec<Qubit> = block_qargs
            .into_iter()
            .filter(|q| qubit_pos_map.contains_key(q))
            .collect();
        block_qargs.sort_by_key(|q| qubit_pos_map[q]);

        let mut block_cargs: Vec<Clbit> = block_cargs
            .into_iter()
            .filter(|c| clbit_pos_map.contains_key(c))
            .collect();
        block_cargs.sort_by_key(|c| clbit_pos_map[c]);

        let old_op = op.unbind();
        let op = convert_py_to_operation_type(py, old_op.clone_ref(py))?;
        let op_name = op.operation.name().to_string();
        let qubits_id = Interner::intern(&mut self.qargs_cache, block_qargs)?;
        let clbits_id = Interner::intern(&mut self.cargs_cache, block_cargs)?;
        let weight = NodeType::Operation(PackedInstruction::new(
            op.operation,
            qubits_id,
            clbits_id,
            op.params,
            op.label,
            op.duration,
            op.unit,
            op.condition,
            #[cfg(feature = "cache_pygates")]
            Some(old_op),
        ));

        let new_node = self
            .dag
            .contract_nodes(block_ids, weight, cycle_check)
            .map_err(|e| match e {
                ContractError::DAGWouldCycle => DAGCircuitError::new_err(
                    "Replacing the specified node block would introduce a cycle",
                ),
            })?;

        self.increment_op(op_name);
        for name in block_op_names {
            self.decrement_op(name);
        }

        self.get_node(py, new_node)
    }

    fn iter_vars(&self, py: Python) -> PyResult<Vec<PyObject>> {
        todo!()
    }

    /// Replace one node with dag.
    ///
    /// Args:
    ///     node (DAGOpNode): node to substitute
    ///     input_dag (DAGCircuit): circuit that will substitute the node.
    ///     wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
    ///         in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
    ///         and the order gets matched to the node wires by qargs first, then cargs, then
    ///         conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
    ///         that the ``node`` acts on.
    ///
    ///         Standalone :class:`~.expr.Var` nodes cannot currently be remapped as part of the
    ///         substitution; the ``input_dag`` should be defined over the correct set of variables
    ///         already.
    ///
    ///         ..
    ///             The rule about not remapping `Var`s is to avoid performance pitfalls and reduce
    ///             complexity; the creator of the input DAG should easily be able to arrange for
    ///             the correct `Var`s to be used, and doing so avoids us needing to recurse through
    ///             control-flow operations to do deep remappings.
    ///     propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on
    ///         the operation within ``node`` is propagated to each node in the ``input_dag``.  If
    ///         ``False``, then the ``input_dag`` is assumed to faithfully implement suitable
    ///         conditional logic already.  This is ignored for :class:`.ControlFlowOp`\\ s (i.e.
    ///         treated as if it is ``False``); replacements of those must already fulfill the same
    ///         conditional logic or this function would be close to useless for them.
    ///
    /// Returns:
    ///     dict: maps node IDs from `input_dag` to their new node incarnations in `self`.
    ///
    /// Raises:
    ///     DAGCircuitError: if met with unexpected predecessor/successors
    #[pyo3(signature = (node, input_dag, wires=None, propagate_condition=true))]
    fn substitute_node_with_dag(
        &mut self,
        py: Python,
        node: &DAGNode,
        input_dag: &DAGCircuit,
        wires: Option<Bound<PyAny>>,
        propagate_condition: bool,
    ) -> PyResult<Py<PyDict>> {
        let instr = match &self.dag[node.node.unwrap()] {
            NodeType::Operation(p) => p,
            _ => {
                return Err(DAGCircuitError::new_err("expected DAGOpNode"));
            }
        };
        let wire_map = if wires.is_some() && wires.unwrap().is_instance_of::<PyDict>() {
            todo!()
        } else {
            let wires: Vec<Wire> = if let Some(wires) = wires {
                let wires: Vec<Bound<PyAny>> = wires.extract()?;
                wires
                    .into_iter()
                    .map(|w| {
                        Ok(if w.is_instance(QUBIT.get_bound(py))? {
                            Wire::Qubit(input_dag.qubits.find(&w).unwrap())
                        } else if w.is_instance(CLBIT.get_bound(py))? {
                            Wire::Clbit(input_dag.clbits.find(&w).unwrap())
                        } else {
                            Wire::Var(w.unbind())
                        })
                    })
                    .collect::<PyResult<_>>()?
            } else {
                input_dag
                    .qubits
                    .into_iter()
                    .map(|q| Wire::Qubit(*q))
                    .chain(input_dag.clbits.into_iter().map(|c| Wire::Clbit(*c)))
                    .chain(input_dag.iter_vars(py)?.into_iter().map(|v| Wire::Var(v)))
                    .collect()
            };

            let node_cargs: IndexSet<&Clbit, ahash::RandomState> =
                IndexSet::from_iter(self.cargs_cache.intern(instr.clbits_id));
            let mut node_wire_order: Vec<Wire> = self
                .qargs_cache
                .intern(instr.qubits_id)
                .into_iter()
                .map(Wire::Qubit)
                .chain(
                    self.qargs_cache
                        .intern(instr.clbits_id)
                        .into_iter()
                        .map(Wire::Clbit),
                )
                .collect();

            if !propagate_condition && self.may_have_additional_wires(py, &instr) {
                let (clbits, vars) = self.additional_wires(py, &instr)?;
                for clbit in clbits {

                }
            }
        };
        // if isinstance(wires, dict):
        //     wire_map = wires
        // else:
        //     wires = input_dag.wires if wires is None else wires
        //     node_cargs = set(node.cargs)
        //     node_wire_order = list(node.qargs) + list(node.cargs)
        //     # If we're not propagating it, the number of wires in the input DAG should include the
        //     # condition as well.
        //     if not propagate_condition and _may_have_additional_wires(node.op):
        //         node_wire_order += [
        //             wire for wire in _additional_wires(node.op) if wire not in node_cargs
        //         ]
        //     if len(wires) != len(node_wire_order):
        //         raise DAGCircuitError(
        //             f"bit mapping invalid: expected {len(node_wire_order)}, got {len(wires)}"
        //         )
        //     wire_map = dict(zip(wires, node_wire_order))
        //     if len(wire_map) != len(node_wire_order):
        //         raise DAGCircuitError("bit mapping invalid: some bits have duplicate entries")
        // for input_dag_wire, our_wire in wire_map.items():
        //     if our_wire not in self.input_map:
        //         raise DAGCircuitError(f"bit mapping invalid: {our_wire} is not in this DAG")
        //     if isinstance(our_wire, expr.Var) or isinstance(input_dag_wire, expr.Var):
        //         raise DAGCircuitError("`Var` nodes cannot be remapped during substitution")
        //     # Support mapping indiscriminately between Qubit and AncillaQubit, etc.
        //     check_type = Qubit if isinstance(our_wire, Qubit) else Clbit
        //     if not isinstance(input_dag_wire, check_type):
        //         raise DAGCircuitError(
        //             f"bit mapping invalid: {input_dag_wire} and {our_wire} are different bit types"
        //         )
        // if _may_have_additional_wires(node.op):
        //     node_vars = {var for var in _additional_wires(node.op) if isinstance(var, expr.Var)}
        // else:
        //     node_vars = set()
        // dag_vars = set(input_dag.iter_vars())
        // if dag_vars - node_vars:
        //     raise DAGCircuitError(
        //         "Cannot replace a node with a DAG with more variables."
        //         f" Variables in node: {node_vars}."
        //         f" Variables in DAG: {dag_vars}."
        //     )
        // for var in dag_vars:
        //     wire_map[var] = var
        //
        // reverse_wire_map = {b: a for a, b in wire_map.items()}
        // # It doesn't make sense to try and propagate a condition from a control-flow op; a
        // # replacement for the control-flow op should implement the operation completely.
        // if (
        //     propagate_condition
        //     and not isinstance(node.op, ControlFlowOp)
        //     and (op_condition := getattr(node.op, "condition", None)) is not None
        // ):
        //     in_dag = input_dag.copy_empty_like()
        //     # The remapping of `condition` below is still using the old code that assumes a 2-tuple.
        //     # This is because this remapping code only makes sense in the case of non-control-flow
        //     # operations being replaced.  These can only have the 2-tuple conditions, and the
        //     # ability to set a condition at an individual node level will be deprecated and removed
        //     # in favour of the new-style conditional blocks.  The extra logic in here to add
        //     # additional wires into the map as necessary would hugely complicate matters if we tried
        //     # to abstract it out into the `VariableMapper` used elsewhere.
        //     target, value = op_condition
        //     if isinstance(target, Clbit):
        //         new_target = reverse_wire_map.get(target, Clbit())
        //         if new_target not in wire_map:
        //             in_dag.add_clbits([new_target])
        //             wire_map[new_target], reverse_wire_map[target] = target, new_target
        //         target_cargs = {new_target}
        //     else:  # ClassicalRegister
        //         mapped_bits = [reverse_wire_map.get(bit, Clbit()) for bit in target]
        //         for ours, theirs in zip(target, mapped_bits):
        //             # Update to any new dummy bits we just created to the wire maps.
        //             wire_map[theirs], reverse_wire_map[ours] = ours, theirs
        //         new_target = ClassicalRegister(bits=mapped_bits)
        //         in_dag.add_creg(new_target)
        //         target_cargs = set(new_target)
        //     new_condition = (new_target, value)
        //     for in_node in input_dag.topological_op_nodes():
        //         if getattr(in_node.op, "condition", None) is not None:
        //             raise DAGCircuitError(
        //                 "cannot propagate a condition to an element that already has one"
        //             )
        //         if target_cargs.intersection(in_node.cargs):
        //             # This is for backwards compatibility with early versions of the method, as it is
        //             # a tested part of the API.  In the newer model of a condition being an integral
        //             # part of the operation (not a separate property to be copied over), this error
        //             # is overzealous, because it forbids a custom instruction from implementing the
        //             # condition within its definition rather than at the top level.
        //             raise DAGCircuitError(
        //                 "cannot propagate a condition to an element that acts on those bits"
        //             )
        //         new_op = copy.copy(in_node.op)
        //         if new_condition:
        //             if not isinstance(new_op, ControlFlowOp):
        //                 new_op = new_op.c_if(*new_condition)
        //             else:
        //                 new_op.condition = new_condition
        //         in_dag.apply_operation_back(new_op, in_node.qargs, in_node.cargs, check=False)
        // else:
        //     in_dag = input_dag
        //
        // if in_dag.global_phase:
        //     self.global_phase += in_dag.global_phase
        //
        // # Add wire from pred to succ if no ops on mapped wire on ``in_dag``
        // # rustworkx's substitute_node_with_subgraph lacks the DAGCircuit
        // # context to know what to do in this case (the method won't even see
        // # these nodes because they're filtered) so we manually retain the
        // # edges prior to calling substitute_node_with_subgraph and set the
        // # edge_map_fn callback kwarg to skip these edges when they're
        // # encountered.
        // for in_dag_wire, self_wire in wire_map.items():
        //     input_node = in_dag.input_map[in_dag_wire]
        //     output_node = in_dag.output_map[in_dag_wire]
        //     if in_dag._multi_graph.has_edge(input_node._node_id, output_node._node_id):
        //         pred = self._multi_graph.find_predecessors_by_edge(
        //             node._node_id, lambda edge, wire=self_wire: edge == wire
        //         )[0]
        //         succ = self._multi_graph.find_successors_by_edge(
        //             node._node_id, lambda edge, wire=self_wire: edge == wire
        //         )[0]
        //         self._multi_graph.add_edge(pred._node_id, succ._node_id, self_wire)
        // for contracted_var in node_vars - dag_vars:
        //     pred = self._multi_graph.find_predecessors_by_edge(
        //         node._node_id, lambda edge, wire=contracted_var: edge == wire
        //     )[0]
        //     succ = self._multi_graph.find_successors_by_edge(
        //         node._node_id, lambda edge, wire=contracted_var: edge == wire
        //     )[0]
        //     self._multi_graph.add_edge(pred._node_id, succ._node_id, contracted_var)
        //
        // # Exclude any nodes from in_dag that are not a DAGOpNode or are on
        // # wires outside the set specified by the wires kwarg
        // def filter_fn(node):
        //     if not isinstance(node, DAGOpNode):
        //         return False
        //     for _, _, wire in in_dag.edges(node):
        //         if wire not in wire_map:
        //             return False
        //     return True
        //
        // # Map edges into and out of node to the appropriate node from in_dag
        // def edge_map_fn(source, _target, self_wire):
        //     wire = reverse_wire_map[self_wire]
        //     # successor edge
        //     if source == node._node_id:
        //         wire_output_id = in_dag.output_map[wire]._node_id
        //         out_index = in_dag._multi_graph.predecessor_indices(wire_output_id)[0]
        //         # Edge directly from from input nodes to output nodes in in_dag are
        //         # already handled prior to calling rustworkx. Don't map these edges
        //         # in rustworkx.
        //         if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
        //             return None
        //     # predecessor edge
        //     else:
        //         wire_input_id = in_dag.input_map[wire]._node_id
        //         out_index = in_dag._multi_graph.successor_indices(wire_input_id)[0]
        //         # Edge directly from from input nodes to output nodes in in_dag are
        //         # already handled prior to calling rustworkx. Don't map these edges
        //         # in rustworkx.
        //         if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
        //             return None
        //     return out_index
        //
        // # Adjust edge weights from in_dag
        // def edge_weight_map(wire):
        //     return wire_map[wire]
        //
        // node_map = self._multi_graph.substitute_node_with_subgraph(
        //     node._node_id, in_dag._multi_graph, edge_map_fn, filter_fn, edge_weight_map
        // )
        // self._decrement_op(node.name)
        //
        // variable_mapper = _classical_resource_map.VariableMapper(
        //     self.cregs.values(), wire_map, add_register=self.add_creg
        // )
        // # Iterate over nodes of input_circuit and update wires in node objects migrated
        // # from in_dag
        // for old_node_index, new_node_index in node_map.items():
        //     # update node attributes
        //     old_node = in_dag._multi_graph[old_node_index]
        //     if isinstance(old_node.op, SwitchCaseOp):
        //         m_op = SwitchCaseOp(
        //             variable_mapper.map_target(old_node.op.target),
        //             old_node.op.cases_specifier(),
        //             label=old_node.op.label,
        //         )
        //     elif getattr(old_node.op, "condition", None) is not None:
        //         m_op = old_node.op
        //         if not isinstance(old_node.op, ControlFlowOp):
        //             new_condition = variable_mapper.map_condition(m_op.condition)
        //             if new_condition is not None:
        //                 m_op = m_op.c_if(*new_condition)
        //         else:
        //             m_op.condition = variable_mapper.map_condition(m_op.condition)
        //     else:
        //         m_op = old_node.op
        //     m_qargs = [wire_map[x] for x in old_node.qargs]
        //     m_cargs = [wire_map[x] for x in old_node.cargs]
        //     new_node = DAGOpNode(m_op, qargs=m_qargs, cargs=m_cargs, dag=self)
        //     new_node._node_id = new_node_index
        //     self._multi_graph[new_node_index] = new_node
        //     self._increment_op(new_node.name)
        //
        // return {k: self._multi_graph[v] for k, v in node_map.items()}
        todo!()
    }

    /// Replace an DAGOpNode with a single operation. qargs, cargs and
    /// conditions for the new operation will be inferred from the node to be
    /// replaced. The new operation will be checked to match the shape of the
    /// replaced operation.
    ///
    /// Args:
    ///     node (DAGOpNode): Node to be replaced
    ///     op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
    ///         instance to be added to the DAG
    ///     inplace (bool): Optional, default False. If True, existing DAG node
    ///         will be modified to include op. Otherwise, a new DAG node will
    ///         be used.
    ///     propagate_condition (bool): Optional, default True.  If True, a condition on the
    ///         ``node`` to be replaced will be applied to the new ``op``.  This is the legacy
    ///         behaviour.  If either node is a control-flow operation, this will be ignored.  If
    ///         the ``op`` already has a condition, :exc:`.DAGCircuitError` is raised.
    ///
    /// Returns:
    ///     DAGOpNode: the new node containing the added operation.
    ///
    /// Raises:
    ///     DAGCircuitError: If replacement operation was incompatible with
    ///     location of target node.
    #[pyo3(signature = (node, op, inplace=false, propagate_condition=true))]
    fn substitute_node(
        &mut self,
        node: PyRefMut<DAGOpNode>,
        op: &Bound<PyAny>,
        inplace: bool,
        propagate_condition: bool,
    ) -> Py<PyAny> {
        // if not isinstance(node, DAGOpNode):
        //     raise DAGCircuitError("Only DAGOpNodes can be replaced.")
        //
        // if node.op.num_qubits != op.num_qubits or node.op.num_clbits != op.num_clbits:
        //     raise DAGCircuitError(
        //         "Cannot replace node of width ({} qubits, {} clbits) with "
        //         "operation of mismatched width ({} qubits, {} clbits).".format(
        //             node.op.num_qubits, node.op.num_clbits, op.num_qubits, op.num_clbits
        //         )
        //     )
        //
        // # This might include wires that are inherent to the node, like in its `condition` or
        // # `target` fields, so might be wider than `node.op.num_{qu,cl}bits`.
        // current_wires = {wire for _, _, wire in self.edges(node)}
        // new_wires = set(node.qargs) | set(node.cargs)
        // if (new_condition := getattr(op, "condition", None)) is not None:
        //     new_wires.update(condition_resources(new_condition).clbits)
        // elif isinstance(op, SwitchCaseOp):
        //     if isinstance(op.target, Clbit):
        //         new_wires.add(op.target)
        //     elif isinstance(op.target, ClassicalRegister):
        //         new_wires.update(op.target)
        //     else:
        //         new_wires.update(node_resources(op.target).clbits)
        //
        // if propagate_condition and not (
        //     isinstance(node.op, ControlFlowOp) or isinstance(op, ControlFlowOp)
        // ):
        //     if new_condition is not None:
        //         raise DAGCircuitError(
        //             "Cannot propagate a condition to an operation that already has one."
        //         )
        //     if (old_condition := getattr(node.op, "condition", None)) is not None:
        //         if not isinstance(op, Instruction):
        //             raise DAGCircuitError("Cannot add a condition on a generic Operation.")
        //         if not isinstance(node.op, ControlFlowOp):
        //             op = op.c_if(*old_condition)
        //         else:
        //             op.condition = old_condition
        //         new_wires.update(condition_resources(old_condition).clbits)
        //
        // if new_wires != current_wires:
        //     # The new wires must be a non-strict subset of the current wires; if they add new wires,
        //     # we'd not know where to cut the existing wire to insert the new dependency.
        //     raise DAGCircuitError(
        //         f"New operation '{op}' does not span the same wires as the old node '{node}'."
        //         f" New wires: {new_wires}, old wires: {current_wires}."
        //     )
        //
        // if inplace:
        //     if op.name != node.op.name:
        //         self._increment_op(op)
        //         self._decrement_op(node.op)
        //     node.op = op
        //     return node
        //
        // new_node = copy.copy(node)
        // new_node.op = op
        // self._multi_graph[node._node_id] = new_node
        // if op.name != node.op.name:
        //     self._increment_op(op)
        //     self._decrement_op(node.op)
        // return new_node
        todo!()
    }

    /// Decompose the circuit into sets of qubits with no gates connecting them.
    ///
    /// Args:
    ///     remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from
    ///         the separated circuits. If ``False``, each output circuit will contain the
    ///         same number of qubits as ``self``.
    ///
    /// Returns:
    ///     List[DAGCircuit]: The circuits resulting from separating ``self`` into sets
    ///         of disconnected qubits
    ///
    /// Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of
    /// clbits as ``self``. The global phase information in ``self`` will not be maintained
    /// in the subcircuits returned by this method.
    #[pyo3(signature = (remove_idle_qubits=false))]
    fn separable_circuits(&self, remove_idle_qubits: bool) -> Py<PyList> {
        // connected_components = rx.weakly_connected_components(self._multi_graph)
        //
        // # Collect each disconnected subgraph
        // disconnected_subgraphs = []
        // for components in connected_components:
        //     disconnected_subgraphs.append(self._multi_graph.subgraph(list(components)))
        //
        // # Helper function for ensuring rustworkx nodes are returned in lexicographical,
        // # topological order
        // def _key(x):
        //     return x.sort_key
        //
        // # Create new DAGCircuit objects from each of the rustworkx subgraph objects
        // decomposed_dags = []
        // for subgraph in disconnected_subgraphs:
        //     new_dag = self.copy_empty_like()
        //     new_dag.global_phase = 0
        //     subgraph_is_classical = True
        //     for node in rx.lexicographical_topological_sort(subgraph, key=_key):
        //         if isinstance(node, DAGInNode):
        //             if isinstance(node.wire, Qubit):
        //                 subgraph_is_classical = False
        //         if not isinstance(node, DAGOpNode):
        //             continue
        //         new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        //
        //     # Ignore DAGs created for empty clbits
        //     if not subgraph_is_classical:
        //         decomposed_dags.append(new_dag)
        //
        // if remove_idle_qubits:
        //     for dag in decomposed_dags:
        //         dag.remove_qubits(*(bit for bit in dag.idle_wires() if isinstance(bit, Qubit)))
        //
        // return decomposed_dags
        todo!()
    }

    /// Swap connected nodes e.g. due to commutation.
    ///
    /// Args:
    ///     node1 (OpNode): predecessor node
    ///     node2 (OpNode): successor node
    ///
    /// Raises:
    ///     DAGCircuitError: if either node is not an OpNode or nodes are not connected
    fn swap_nodes(&mut self, node1: &DAGNode, node2: &DAGNode) -> PyResult<()> {
        // if not (isinstance(node1, DAGOpNode) and isinstance(node2, DAGOpNode)):
        //     raise DAGCircuitError("nodes to swap are not both DAGOpNodes")
        // try:
        //     connected_edges = self._multi_graph.get_all_edge_data(node1._node_id, node2._node_id)
        // except rx.NoEdgeBetweenNodes as no_common_edge:
        //     raise DAGCircuitError("attempt to swap unconnected nodes") from no_common_edge
        // node1_id = node1._node_id
        // node2_id = node2._node_id
        // for edge in connected_edges[::-1]:
        //     edge_find = lambda x, y=edge: x == y
        //     edge_parent = self._multi_graph.find_predecessors_by_edge(node1_id, edge_find)[0]
        //     self._multi_graph.remove_edge(edge_parent._node_id, node1_id)
        //     self._multi_graph.add_edge(edge_parent._node_id, node2_id, edge)
        //     edge_child = self._multi_graph.find_successors_by_edge(node2_id, edge_find)[0]
        //     self._multi_graph.remove_edge(node1_id, node2_id)
        //     self._multi_graph.add_edge(node2_id, node1_id, edge)
        //     self._multi_graph.remove_edge(node2_id, edge_child._node_id)
        //     self._multi_graph.add_edge(node1_id, edge_child._node_id, edge)
        todo!()
    }

    /// Get the node in the dag.
    ///
    /// Args:
    ///     node_id(int): Node identifier.
    ///
    /// Returns:
    ///     node: the node.
    fn node(&self, py: Python, node_id: isize) -> PyResult<Py<PyAny>> {
        self.get_node(py, NodeIndex::new(node_id as usize))
    }

    /// Iterator for node values.
    ///
    /// Yield:
    ///     node: the node.
    fn nodes(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result: PyResult<Vec<_>> = self
            .dag
            .node_references()
            .map(|(node, weight)| self.unpack_into(py, node, weight))
            .collect();
        let tup = PyTuple::new_bound(py, result?);
        Ok(tup.into_any().iter().unwrap().unbind())
    }

    /// Iterator for edge values with source and destination node.
    ///
    /// This works by returning the outgoing edges from the specified nodes. If
    /// no nodes are specified all edges from the graph are returned.
    ///
    /// Args:
    ///     nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):
    ///         Either a list of nodes or a single input node. If none is specified,
    ///         all edges are returned from the graph.
    ///
    /// Yield:
    ///     edge: the edge as a tuple with the format
    ///         (source node, destination node, edge wire)
    fn edges(&self, nodes: Option<Bound<PyAny>>, py: Python) -> PyResult<Py<PyIterator>> {
        let get_node_index = |obj: &Bound<PyAny>| -> PyResult<NodeIndex> {
            Ok(obj.downcast::<DAGNode>()?.borrow().node.unwrap())
        };

        let actual_nodes: Vec<_> = match nodes {
            None => self.dag.node_indices().collect(),
            Some(nodes) => {
                let mut out = Vec::new();
                if let Ok(node) = get_node_index(&nodes) {
                    out.push(node);
                } else {
                    for node in nodes.iter()? {
                        out.push(get_node_index(&node?)?);
                    }
                }
                out
            }
        };

        let mut edges = Vec::new();
        for node in actual_nodes {
            for edge in self.dag.edges_directed(node, Outgoing) {
                edges.push((
                    self.get_node(py, edge.source())?,
                    self.get_node(py, edge.target())?,
                    match edge.weight() {
                        Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                        Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                        Wire::Var(var) => var,
                    },
                ))
            }
        }

        Ok(PyTuple::new_bound(py, edges)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Get the list of "op" nodes in the dag.
    ///
    /// Args:
    ///     op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
    ///         return. If None, return all op nodes.
    ///     include_directives (bool): include `barrier`, `snapshot` etc.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of node ids containing the given op.
    #[pyo3(signature=(op=None, include_directives=true))]
    fn op_nodes(
        &self,
        py: Python,
        op: Option<&Bound<PyType>>,
        include_directives: bool,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if !include_directives && packed.op.directive() {
                    continue;
                }
                if let Some(op_type) = op {
                    if !packed.op.is_instance(op_type)? {
                        continue;
                    }
                }
                nodes.push(self.unpack_into(py, node, weight)?);
            }
        }
        Ok(nodes)
    }

    /// Get the list of gate nodes in the dag.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of DAGOpNodes that represent gates.
    fn gate_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if let OperationType::Gate(_) = packed.op {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Get the set of "op" nodes with the given name.
    fn named_nodes(&self, py: Python<'_>, names: Vec<String>) -> PyResult<Vec<Py<PyAny>>> {
        let mut result: Vec<Py<PyAny>> = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                let name = packed.op.name();
                if names.contains(&name.to_string()) {
                    result.push(self.unpack_into(py, id, weight)?);
                }
            }
        }
        Ok(result)
    }

    /// Get list of 2 qubit operations. Ignore directives like snapshot and barrier.
    fn two_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if packed.op.directive() {
                    continue;
                }

                let qargs = self.qargs_cache.intern(packed.qubits_id);
                if qargs.len() == 2 {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Get list of 3+ qubit operations. Ignore directives like snapshot and barrier.
    fn multi_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if packed.op.directive() {
                    continue;
                }

                let qargs = self.qargs_cache.intern(packed.qubits_id);
                if qargs.len() >= 3 {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes.
    fn longest_path(&self, py: Python) {
        // return [self._multi_graph[x] for x in rx.dag_longest_path(self._multi_graph)]
        todo!()
    }

    /// Returns iterator of the successors of a node as DAGOpNodes and DAGOutNodes."""
    fn successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, successors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node as DAGOpNodes and DAGInNodes.
    fn predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Checks if a second node is in the successors of node.
    fn is_successor(&self, node: &DAGNode, node_succ: &DAGNode) -> bool {
        self.dag
            .find_edge(node.node.unwrap(), node_succ.node.unwrap())
            .is_some()
    }

    /// Checks if a second node is in the predecessors of node.
    fn is_predecessor(&self, node: &DAGNode, node_pred: &DAGNode) -> bool {
        self.dag
            .find_edge(node_pred.node.unwrap(), node.node.unwrap())
            .is_some()
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "quantum_predecessors")]
    fn py_quantum_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .quantum_predecessors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "quantum_successors")]
    fn py_quantum_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .quantum_successors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, successors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGInNodes.
    fn classical_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Incoming);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Clbit(_) => Some(e.source()),
            _ => None,
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns set of the ancestors of a node as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "ancestors")]
    fn py_ancestors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let ancestors: PyResult<Vec<PyObject>> = self
            .ancestors(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new_bound(py, &ancestors?)?.unbind())
    }

    /// Returns set of the descendants of a node as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "descendants")]
    fn py_descendants(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let descendants: PyResult<Vec<PyObject>> = self
            .descendants(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new_bound(py, &descendants?)?.unbind())
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    #[pyo3(name = "bfs_successors")]
    fn py_bfs_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successor_index: PyResult<Vec<(PyObject, Vec<PyObject>)>> = self
            .bfs_successors(node.node.unwrap())
            .map(|(node, nodes)| -> PyResult<(PyObject, Vec<PyObject>)> {
                Ok((
                    self.get_node(py, node)?,
                    nodes
                        .iter()
                        .map(|sub_node| self.get_node(py, *sub_node))
                        .collect::<PyResult<Vec<_>>>()?,
                ))
            })
            .collect();
        Ok(PyList::new_bound(py, successor_index?)
            .into_any()
            .iter()?
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGOutNodes.
    fn classical_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Incoming);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Clbit(_) => Some(e.target()),
            _ => None,
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    #[pyo3(name = "remove_op_node")]
    fn py_remove_op_node(&mut self, node: PyRef<DAGOpNode>) -> PyResult<()> {
        self.remove_op_node(node.as_ref().node.unwrap());
        Ok(())
    }

    /// Remove all of the ancestor operation nodes of node.
    fn remove_ancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: Vec<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && match self.dag.node_weight(*next) {
                        Some(NodeType::Operation(_)) => true,
                        _ => false,
                    }
            })
            .collect();
        for a in ancestors {
            self.dag.remove_node(a);
        }
        Ok(())
    }

    /// Remove all of the descendant operation nodes of node.
    fn remove_descendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: Vec<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && match self.dag.node_weight(*next) {
                        Some(NodeType::Operation(_)) => true,
                        _ => false,
                    }
            })
            .collect();
        for d in descendants {
            self.dag.remove_node(d);
        }
        Ok(())
    }

    /// Remove all of the non-ancestors operation nodes of node.
    fn remove_nonancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: HashSet<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && match self.dag.node_weight(*next) {
                        Some(NodeType::Operation(_)) => true,
                        _ => false,
                    }
            })
            .collect();
        let non_ancestors: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !ancestors.contains(node_id))
            .collect();
        for na in non_ancestors {
            self.dag.remove_node(na);
        }
        Ok(())
    }

    /// Remove all of the non-descendants operation nodes of node.
    fn remove_nondescendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: HashSet<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && match self.dag.node_weight(*next) {
                        Some(NodeType::Operation(_)) => true,
                        _ => false,
                    }
            })
            .collect();
        let non_descendants: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !descendants.contains(node_id))
            .collect();
        for nd in non_descendants {
            self.dag.remove_node(nd);
        }
        Ok(())
    }

    /// Return a list of op nodes in the first layer of this dag.
    fn front_layers(&self) -> Py<PyList> {
        // graph_layers = self.multigraph_layers()
        // try:
        //     next(graph_layers)  # Remove input nodes
        // except StopIteration:
        //     return []
        //
        // op_nodes = [node for node in next(graph_layers) if isinstance(node, DAGOpNode)]
        //
        // return op_nodes
        todo!()
    }

    /// Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.
    ///
    /// A layer is a circuit whose gates act on disjoint qubits, i.e.,
    /// a layer has depth 1. The total number of layers equals the
    /// circuit depth d. The layers are indexed from 0 to d-1 with the
    /// earliest layer at index 0. The layers are constructed using a
    /// greedy algorithm. Each returned layer is a dict containing
    /// {"graph": circuit graph, "partition": list of qubit lists}.
    ///
    /// The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,
    /// and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent
    /// via DAGNode.semantic_eq(node1, node2).
    ///
    /// TODO: Gates that use the same cbits will end up in different
    /// layers as this is currently implemented. This may not be
    /// the desired behavior.
    fn layers(&self) -> Py<PyIterator> {
        // graph_layers = self.multigraph_layers()
        // try:
        //     next(graph_layers)  # Remove input nodes
        // except StopIteration:
        //     return
        //
        // for graph_layer in graph_layers:
        //
        //     # Get the op nodes from the layer, removing any input and output nodes.
        //     op_nodes = [node for node in graph_layer if isinstance(node, DAGOpNode)]
        //
        //     # Sort to make sure they are in the order they were added to the original DAG
        //     # It has to be done by node_id as graph_layer is just a list of nodes
        //     # with no implied topology
        //     # Drawing tools rely on _node_id to infer order of node creation
        //     # so we need this to be preserved by layers()
        //     op_nodes.sort(key=lambda nd: nd._node_id)
        //
        //     # Stop yielding once there are no more op_nodes in a layer.
        //     if not op_nodes:
        //         return
        //
        //     # Construct a shallow copy of self
        //     new_layer = self.copy_empty_like()
        //
        //     for node in op_nodes:
        //         # this creates new DAGOpNodes in the new_layer
        //         new_layer.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        //
        //     # The quantum registers that have an operation in this layer.
        //     support_list = [
        //         op_node.qargs
        //         for op_node in new_layer.op_nodes()
        //         if not getattr(op_node.op, "_directive", False)
        //     ]
        //
        //     yield {"graph": new_layer, "partition": support_list}
        todo!()
    }

    /// Yield a layer for all gates of this circuit.
    ///
    /// A serial layer is a circuit with one gate. The layers have the
    /// same structure as in layers().
    fn serial_layers(&self) -> PyResult<Py<PyIterator>> {
        // for next_node in self.topological_op_nodes():
        //     new_layer = self.copy_empty_like()
        //
        //     # Save the support of the operation we add to the layer
        //     support_list = []
        //     # Operation data
        //     op = copy.copy(next_node.op)
        //     qargs = copy.copy(next_node.qargs)
        //     cargs = copy.copy(next_node.cargs)
        //
        //     # Add node to new_layer
        //     new_layer.apply_operation_back(op, qargs, cargs, check=False)
        //     # Add operation to partition
        //     if not getattr(next_node.op, "_directive", False):
        //         support_list.append(list(qargs))
        //     l_dict = {"graph": new_layer, "partition": support_list}
        //     yield l_dict
        todo!()
    }

    /// Yield layers of the multigraph.
    fn multigraph_layers(&self) -> PyResult<Py<PyIterator>> {
        /// first_layer = [x._node_id for x in self.input_map.values()]
        /// return iter(rx.layers(self._multi_graph, first_layer))
        todo!()
    }

    /// Return a set of non-conditional runs of "op" nodes with the given names.
    ///
    /// For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
    /// would produce the tuple of cx nodes as an element of the set returned
    /// from a call to collect_runs(["cx"]). If instead the cx nodes were
    /// "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
    /// pair in a tuple. The namelist can contain names that are not
    /// in the circuit's basis.
    ///
    /// Nodes must have only one successor to continue the run.
    #[pyo3(name = "collect_runs")]
    fn py_collect_runs(&self, py: Python, namelist: &Bound<PyList>) -> PyResult<Py<PySet>> {
        let mut name_list_set = HashSet::with_capacity(namelist.len());
        for name in namelist.iter() {
            name_list_set.insert(name.extract::<String>()?);
        }
        match self.collect_runs(name_list_set) {
            Some(runs) => {
                let run_iter = runs.map(|node_indices| {
                    PyTuple::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_set = PySet::empty_bound(py)?;
                for run_tuple in run_iter {
                    out_set.add(run_tuple)?;
                }
                Ok(out_set.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    #[pyo3(name = "collect_1q_runs")]
    fn py_collect_1q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_1q_runs() {
            Some(runs) => {
                let runs_iter = runs.map(|node_indices| {
                    PyList::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_list = PyList::empty_bound(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    #[pyo3(name = "collect_2q_runs")]
    fn py_collect_2q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_2q_runs() {
            Some(runs) => {
                let runs_iter = runs.into_iter().map(|node_indices| {
                    PyList::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_list = PyList::empty_bound(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Iterator for nodes that affect a given wire.
    ///
    /// Args:
    ///     wire (Bit): the wire to be looked at.
    ///     only_ops (bool): True if only the ops nodes are wanted;
    ///                 otherwise, all nodes are returned.
    /// Yield:
    ///      Iterator: the successive nodes on the given wire
    ///
    /// Raises:
    ///     DAGCircuitError: if the given wire doesn't exist in the DAG
    #[pyo3(name = "nodes_on_wire", signature = (wire, only_ops=false))]
    fn py_nodes_on_wire(
        &self,
        py: Python,
        wire: &Bound<PyAny>,
        only_ops: bool,
    ) -> PyResult<Py<PyIterator>> {
        let wire = if wire.is_instance(self.circuit_module.qubit.bind(py))? {
            self.qubits.find(wire).map(Wire::Qubit)
        } else if wire.is_instance(self.circuit_module.clbit.bind(py))? {
            self.clbits.find(wire).map(Wire::Clbit)
        } else {
            None
        }
        .ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given wire {:?} is not present in the circuit",
                wire
            ))
        })?;

        let nodes = self
            .nodes_on_wire(&wire, only_ops)
            .into_iter()
            .map(|n| self.get_node(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new_bound(py, nodes).into_any().iter()?.unbind())
    }

    /// Count the occurrences of operation names.
    ///
    /// Args:
    ///     recurse: if ``True`` (default), then recurse into control-flow operations.  In all
    ///         cases, this counts only the number of times the operation appears in any possible
    ///         block; both branches of if-elses are counted, and for- and while-loop blocks are
    ///         only counted once.
    ///
    /// Returns:
    ///     Mapping[str, int]: a mapping of operation names to the number of times it appears.
    #[pyo3(signature = (*, recurse=true))]
    fn count_ops(&self, recurse: bool) -> PyResult<usize> {
        // if not recurse or not CONTROL_FLOW_OP_NAMES.intersection(self._op_names):
        //     return self._op_names.copy()
        //
        // # pylint: disable=cyclic-import
        // from qiskit.converters import circuit_to_dag
        //
        // def inner(dag, counts):
        //     for name, count in dag._op_names.items():
        //         counts[name] += count
        //     for node in dag.op_nodes(ControlFlowOp):
        //         for block in node.op.blocks:
        //             counts = inner(circuit_to_dag(block), counts)
        //     return counts
        //
        // return dict(inner(self, defaultdict(int)))
        todo!()
    }

    /// Count the occurrences of operation names on the longest path.
    ///
    /// Returns a dictionary of counts keyed on the operation name.
    fn count_ops_longest_path(&self) -> PyResult<usize> {
        // op_dict = {}
        // path = self.longest_path()
        // path = path[1:-1]  # remove qubits at beginning and end of path
        // for node in path:
        //     name = node.op.name
        //     if name not in op_dict:
        //         op_dict[name] = 1
        //     else:
        //         op_dict[name] += 1
        // return op_dict
        todo!()
    }

    /// Returns causal cone of a qubit.
    ///
    /// A qubit's causal cone is the set of qubits that can influence the output of that
    /// qubit through interactions, whether through multi-qubit gates or operations. Knowing
    /// the causal cone of a qubit can be useful when debugging faulty circuits, as it can
    /// help identify which wire(s) may be causing the problem.
    ///
    /// This method does not consider any classical data dependency in the ``DAGCircuit``,
    /// classical bit wires are ignored for the purposes of building the causal cone.
    ///
    /// Args:
    ///     qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.
    ///
    /// Returns:
    ///     Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.
    fn quantum_causal_cone(&self, py: Python, qubit: &Bound<PyAny>) -> PyResult<Py<PySet>> {
        // Retrieve the output node from the qubit
        let output_qubit = self.qubits.find(qubit).ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given qubit {:?} is not present in the circuit",
                qubit
            ))
        })?;
        let output_node_index = self.qubit_output_map.get(&output_qubit).ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given qubit {:?} is not present in qubit_output_map",
                qubit
            ))
        })?;

        let mut qubits_in_cone: HashSet<&Qubit> = HashSet::from([&output_qubit]);
        let mut queue: VecDeque<NodeIndex> =
            self.quantum_predecessors(*output_node_index).collect();

        // The processed_non_directive_nodes stores the set of processed non-directive nodes.
        // This is an optimization to avoid considering the same non-directive node multiple
        // times when reached from different paths.
        // The directive nodes (such as barriers or measures) are trickier since when processing
        // them we only add their predecessors that intersect qubits_in_cone. Hence, directive
        // nodes have to be considered multiple times.
        let mut processed_non_directive_nodes: HashSet<NodeIndex> = HashSet::new();

        while !queue.is_empty() {
            let cur_index = queue.pop_front().unwrap();

            if let NodeType::Operation(packed) = self.dag.node_weight(cur_index).unwrap() {
                if !packed.op.directive() {
                    // If the operation is not a directive (in particular not a barrier nor a measure),
                    // we do not do anything if it was already processed. Otherwise, we add its qubits
                    // to qubits_in_cone, and append its predecessors to queue.
                    if processed_non_directive_nodes.contains(&cur_index) {
                        continue;
                    }
                    qubits_in_cone.extend(self.qargs_cache.intern(packed.qubits_id).iter());
                    processed_non_directive_nodes.insert(cur_index);

                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            queue.push_back(pred_index);
                        }
                    }
                } else {
                    // Directives (such as barriers and measures) may be defined over all the qubits,
                    // yet not all of these qubits should be considered in the causal cone. So we
                    // only add those predecessors that have qubits in common with qubits_in_cone.
                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            if self
                                .qargs_cache
                                .intern(pred_packed.qubits_id)
                                .iter()
                                .any(|x| qubits_in_cone.contains(x))
                            {
                                queue.push_back(pred_index);
                            }
                        }
                    }
                }
            }
        }

        let qubits_in_cone_vec: Vec<_> = qubits_in_cone.iter().map(|&&qubit| qubit).collect();
        let elements = self.qubits.map_indices(&qubits_in_cone_vec[..]);
        Ok(PySet::new_bound(py, elements)?.unbind())
    }

    /// Return a dictionary of circuit properties.
    fn properties(&self, py: Python) -> PyResult<HashMap<&str, usize>> {
        let mut summary = HashMap::from_iter([
            ("size", self.size(py, false)?),
            ("depth", self.depth(false)?),
            ("width", self.width()),
            ("qubits", self.num_qubits()),
            ("bits", self.num_clbits()),
            ("factors", self.num_tensor_factors()),
            ("operations", self.count_ops(true)?),
        ]);
        Ok(summary)
    }

    /// Draws the dag circuit.
    ///
    /// This function needs `Graphviz <https://www.graphviz.org/>`_ to be
    /// installed. Graphviz is not a python package and can't be pip installed
    /// (the ``graphviz`` package on PyPI is a Python interface library for
    /// Graphviz and does not actually install Graphviz). You can refer to
    /// `the Graphviz documentation <https://www.graphviz.org/download/>`__ on
    /// how to install it.
    ///
    /// Args:
    ///     scale (float): scaling factor
    ///     filename (str): file path to save image to (format inferred from name)
    ///     style (str):
    ///         'plain': B&W graph;
    ///         'color' (default): color input/output/op nodes
    ///
    /// Returns:
    ///     Ipython.display.Image: if in Jupyter notebook and not saving to file,
    ///     otherwise None.
    #[pyo3(signature=(scale=0.7, filename=None, style="color"))]
    fn draw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        scale: f64,
        filename: Option<&str>,
        style: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = PyModule::import_bound(py, "qiskit.visualization.dag_visualization")?;
        module.call_method1("dag_drawer", (slf, scale, filename, style))
    }

    fn _to_dot<'py>(
        &self,
        py: Python<'py>,
        graph_attrs: Option<BTreeMap<String, String>>,
        node_attrs: Option<PyObject>,
        edge_attrs: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyString>> {
        let mut buffer = Vec::<u8>::new();
        build_dot(py, self, &mut buffer, graph_attrs, node_attrs, edge_attrs)?;
        Ok(PyString::new_bound(py, std::str::from_utf8(&buffer)?))
    }
}

impl DAGCircuit {
    /// Return an iterator of gate runs with non-conditional op nodes of given names
    pub fn collect_runs(
        &self,
        namelist: HashSet<String>,
    ) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(namelist.contains(inst.op.name())
                    && match &inst.extra_attrs {
                        None => true,
                        Some(attrs) => attrs.condition.is_none(),
                    }),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    pub fn collect_1q_runs(&self) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(inst.op.num_qubits() == 1
                    && inst.op.num_clbits() == 0
                    && inst.op.matrix(&inst.params).is_some()
                    && match &inst.extra_attrs {
                        None => true,
                        Some(attrs) => attrs.condition.is_none(),
                    }),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    pub fn collect_2q_runs(&self) -> Option<Vec<Vec<NodeIndex>>> {
        let filter_fn = move |node_index: NodeIndex| -> Result<Option<bool>, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => match &inst.op {
                    OperationType::Standard(gate) => Ok(Some(
                        gate.num_qubits() <= 2
                            && match &inst.extra_attrs {
                                None => true,
                                Some(attrs) => attrs.condition.is_none(),
                            }
                            && !inst.is_parameterized(),
                    )),
                    OperationType::Gate(gate) => Ok(Some(
                        gate.num_qubits() <= 2
                            && match &inst.extra_attrs {
                                None => true,
                                Some(attrs) => attrs.condition.is_none(),
                            }
                            && !inst.is_parameterized(),
                    )),
                    _ => Ok(Some(false)),
                },
                _ => Ok(None),
            }
        };

        let color_fn = move |edge_index: EdgeIndex| -> Result<Option<usize>, Infallible> {
            let wire = self.dag.edge_weight(edge_index).unwrap();
            match wire {
                Wire::Qubit(index) => Ok(Some(index.0 as usize)),
                _ => Ok(None),
            }
        };
        rustworkx_core::dag_algo::collect_bicolor_runs(&self.dag, filter_fn, color_fn).unwrap()
    }

    fn increment_op(&mut self, op: String) {
        match self.op_names.entry(op) {
            hash_map::Entry::Occupied(mut o) => {
                *o.get_mut() += 1;
            }
            hash_map::Entry::Vacant(v) => {
                v.insert(1);
            }
        }
    }

    fn decrement_op(&mut self, op: String) {
        match self.op_names.entry(op) {
            hash_map::Entry::Occupied(mut o) => {
                if *o.get() > 0usize {
                    *o.get_mut() -= 1;
                } else {
                    o.remove();
                }
            }
            _ => panic!("Cannot decrement something not added!"),
        }
    }

    fn quantum_predecessors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Incoming)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.source()),
                _ => None,
            })
            .unique()
    }

    fn quantum_successors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Outgoing)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.target()),
                _ => None,
            })
            .unique()
    }

    /// Apply a [PackedInstruction] to the back of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    fn push_back(&mut self, py: Python, instr: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = instr.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &instr) {
                let mut clbits: IndexSet<Clbit> =
                    IndexSet::from_iter(self.cargs_cache.intern(instr.clbits_id).iter().cloned());
                let (additional_clbits, additional_vars) = self.additional_wires(py, &instr)?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (
                    self.cargs_cache
                        .intern(instr.clbits_id)
                        .iter()
                        .copied()
                        .collect(),
                    None,
                )
            }
        };

        self.increment_op(op_name.to_string());

        let qubits_id = instr.qubits_id;
        let new_node = self.dag.add_node(NodeType::Operation(instr));

        // Put the new node in-between the previously "last" nodes on each wire
        // and the output map.
        let output_nodes: Vec<NodeIndex> = self
            .qargs_cache
            .intern(qubits_id)
            .iter()
            .map(|q| self.qubit_output_map.get(q).copied().unwrap())
            .chain(
                all_cbits
                    .iter()
                    .map(|c| self.clbit_output_map.get(c).copied().unwrap()),
            )
            .chain(
                vars.iter()
                    .flatten()
                    .map(|v| self.var_output_map.get(v).unwrap()),
            )
            .collect();

        for output_node in output_nodes {
            let last_edges: Vec<_> = self
                .dag
                .edges_directed(output_node, Incoming)
                .map(|e| (e.source(), e.id(), e.weight().clone()))
                .collect();
            for (source, old_edge, weight) in last_edges.into_iter() {
                self.dag.add_edge(source, new_node, weight.clone());
                self.dag.add_edge(new_node, output_node, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    /// Apply a [PackedInstruction] to the front of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    fn push_front(&mut self, py: Python, inst: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = inst.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &inst) {
                let mut clbits: IndexSet<Clbit> =
                    IndexSet::from_iter(self.cargs_cache.intern(inst.clbits_id).iter().cloned());
                let (additional_clbits, additional_vars) = self.additional_wires(py, &inst)?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (
                    self.cargs_cache
                        .intern(inst.clbits_id)
                        .iter()
                        .copied()
                        .collect(),
                    None,
                )
            }
        };

        self.increment_op(op_name.to_string());

        let qubits_id = inst.qubits_id;
        let new_node = self.dag.add_node(NodeType::Operation(inst));

        // Put the new node in-between the input map and the previously
        // "first" nodes on each wire.
        let input_nodes: Vec<NodeIndex> = self
            .qargs_cache
            .intern(qubits_id)
            .iter()
            .map(|q| self.qubit_input_map.get(q).copied().unwrap())
            .chain(
                all_cbits
                    .iter()
                    .map(|c| self.clbit_input_map.get(c).copied().unwrap()),
            )
            .collect();

        for input_node in input_nodes {
            let first_edges: Vec<_> = self
                .dag
                .edges_directed(input_node, Outgoing)
                .map(|e| (e.target(), e.id(), e.weight().clone()))
                .collect();
            for (target, old_edge, weight) in first_edges.into_iter() {
                self.dag.add_edge(input_node, new_node, weight.clone());
                self.dag.add_edge(new_node, target, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    fn topological_nodes(&self) -> PyResult<impl Iterator<Item = NodeIndex>> {
        let key = |node: NodeIndex| -> Result<(Option<Index>, Option<Index>), Infallible> {
            Ok(self.dag.node_weight(node).unwrap().key())
        };
        let nodes =
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(_) => {
                        unreachable!()
                    }
                })?;
        Ok(nodes.into_iter())
    }

    fn is_wire_idle(&self, wire: &Wire) -> PyResult<bool> {
        let (input_node, output_node) = match wire {
            Wire::Qubit(qubit) => (self.qubit_input_map[qubit], self.qubit_output_map[qubit]),
            Wire::Clbit(clbit) => (self.clbit_input_map[clbit], self.clbit_output_map[clbit]),
            Wire::Var(var) => (
                self.var_input_map.get(var).unwrap(),
                self.var_output_map.get(var).unwrap(),
            ),
        };

        let child = self
            .dag
            .neighbors_directed(input_node, Outgoing)
            .next()
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Invalid dagcircuit input node {:?} has no output",
                    input_node
                ))
            })?;

        Ok(child == output_node)
    }

    fn may_have_additional_wires(&self, py: Python, instr: &PackedInstruction) -> bool {
        let has_condition = match instr.condition() {
            None => false,
            Some(condition) => !condition.bind(py).is_none(),
        };

        if has_condition {
            return true;
        }

        if let OperationType::Instruction(ref inst) = instr.op {
            inst.instruction
                .bind(py)
                .is_instance(CONTROL_FLOW_OP.get_bound(py))
                .unwrap()
                || inst
                    .instruction
                    .bind(py)
                    .is_instance(STORE_OP.get_bound(py))
                    .unwrap()
        } else {
            false
        }
    }

    fn additional_wires(
        &self,
        py: Python,
        instr: &PackedInstruction,
    ) -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
        let wires_from_expr = |node: &Bound<PyAny>| -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
            let mut clbits = Vec::new();
            let mut vars = Vec::new();
            for var in ITER_VARS.get_bound(py).call1((node,))?.iter()? {
                let var = var?;
                let var_var = var.getattr("var")?;
                if var_var.is_instance(CLBIT.get_bound(py))? {
                    clbits.push(self.clbits.find(&var_var).unwrap());
                } else if var_var.is_instance(CLASSICAL_REGISTER.get_bound(py))? {
                    for bit in var_var.iter().unwrap() {
                        clbits.push(self.clbits.find(&bit?).unwrap());
                    }
                } else {
                    vars.push(var.unbind());
                }
            }
            Ok((clbits, vars))
        };

        let condition = instr
            .extra_attrs
            .iter()
            .flat_map(|e| e.condition.as_ref().map(|c| c.bind(py)))
            .next();
        // let mut bits = Vec::new();
        let mut clbits = Vec::new();
        let mut vars = Vec::new();
        if let Some(condition) = condition {
            if !condition.is_none() {
                if condition.is_instance(EXPR.get_bound(py)).unwrap() {
                    let (expr_clbits, expr_vars) = wires_from_expr(condition)?;
                    for bit in expr_clbits {
                        clbits.push(bit);
                    }
                    for var in expr_vars {
                        vars.push(var);
                    }
                } else {
                    for bit in self
                        .control_flow_module
                        .condition_resources(&condition)?
                        .clbits
                        .bind(py)
                    {
                        clbits.push(self.clbits.find(&bit).unwrap());
                    }
                }
            }
        }

        if let OperationType::Instruction(ref inst) = instr.op {
            let op = inst.instruction.bind(py);
            if op.is_instance(CONTROL_FLOW_OP.get_bound(py))? {
                for var in op.call_method0("iter_captured_vars")?.iter()? {
                    vars.push(var?.unbind())
                }
                if op.is_instance(SWITCH_CASE_OP.get_bound(py))? {
                    let target = op.getattr(intern!(py, "target"))?;
                    if target.is_instance(CLBIT.get_bound(py))? {
                        clbits.push(self.clbits.find(&target).unwrap());
                    } else if target.is_instance(CLASSICAL_REGISTER.get_bound(py))? {
                        for bit in target.iter()? {
                            clbits.push(self.clbits.find(&bit?).unwrap());
                        }
                    } else {
                        let (expr_clbits, expr_vars) = wires_from_expr(&target)?;
                        for bit in expr_clbits {
                            clbits.push(bit);
                        }
                        for var in expr_vars {
                            vars.push(var);
                        }
                    }
                }
            } else if op.is_instance(STORE_OP.get_bound(py))? {
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("lvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("rvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
            }
        }
        Ok((clbits, vars))
    }

    /// Add a qubit or bit to the circuit.
    ///
    /// Args:
    ///     wire: the wire to be added
    ///
    ///     This adds a pair of in and out nodes connected by an edge.
    ///
    /// Raises:
    ///     DAGCircuitError: if trying to add duplicate wire
    fn add_wire(&mut self, wire: Wire) -> PyResult<()> {
        let (in_node, out_node) = match wire {
            Wire::Qubit(qubit) => {
                match (
                    self.qubit_input_map.entry(qubit),
                    self.qubit_output_map.entry(qubit),
                ) {
                    (indexmap::map::Entry::Vacant(input), indexmap::map::Entry::Vacant(output)) => {
                        Ok((
                            *input.insert(self.dag.add_node(NodeType::QubitIn(qubit))),
                            *output.insert(self.dag.add_node(NodeType::QubitOut(qubit))),
                        ))
                    }
                    (_, _) => Err(DAGCircuitError::new_err("wire already exists!")),
                }
            }
            Wire::Clbit(clbit) => {
                match (
                    self.clbit_input_map.entry(clbit),
                    self.clbit_output_map.entry(clbit),
                ) {
                    (indexmap::map::Entry::Vacant(input), indexmap::map::Entry::Vacant(output)) => {
                        Ok((
                            *input.insert(self.dag.add_node(NodeType::ClbitIn(clbit))),
                            *output.insert(self.dag.add_node(NodeType::ClbitOut(clbit))),
                        ))
                    }
                    (_, _) => Err(DAGCircuitError::new_err("wire already exists!")),
                }
            }
            Wire::Var(_) => todo!(),
        }?;

        self.dag.add_edge(in_node, out_node, wire);
        Ok(())
    }

    /// Get the nodes on the given wire.
    ///
    /// Note: result is empty if the wire is not in the DAG.
    fn nodes_on_wire(&self, wire: &Wire, only_ops: bool) -> Vec<NodeIndex> {
        let mut nodes = Vec::new();
        let mut current_node = match wire {
            Wire::Qubit(qubit) => self.qubit_input_map.get(qubit),
            Wire::Clbit(clbit) => self.clbit_input_map.get(clbit),
            Wire::Var(_) => todo!(),
        }
        .cloned();

        while let Some(node) = current_node {
            if only_ops {
                let node_weight = self.dag.node_weight(node).unwrap();
                if let NodeType::Operation(_) = node_weight {
                    nodes.push(node);
                }
            } else {
                nodes.push(node);
            }

            let edges = self.dag.edges_directed(node, Outgoing);
            current_node = edges.into_iter().find_map(|edge| {
                if edge.weight() == wire {
                    Some(edge.target())
                } else {
                    None
                }
            });
        }
        nodes
    }

    fn remove_idle_wire(&mut self, wire: Wire) -> PyResult<()> {
        let (in_node, out_node) = match wire {
            Wire::Qubit(qubit) => (
                self.qubit_input_map.shift_remove(&qubit),
                self.qubit_output_map.shift_remove(&qubit),
            ),
            Wire::Clbit(clbit) => (
                self.clbit_input_map.shift_remove(&clbit),
                self.clbit_output_map.shift_remove(&clbit),
            ),
            Wire::Var(_) => todo!(),
        };

        self.dag.remove_node(in_node.unwrap());
        self.dag.remove_node(out_node.unwrap());
        Ok(())
    }

    fn add_qubit_unchecked(&mut self, py: Python, bit: &Bound<PyAny>) -> PyResult<Qubit> {
        let qubit = self.qubits.add(py, bit, false)?;
        self.qubit_locations.bind(py).set_item(
            bit,
            Py::new(
                py,
                BitLocations {
                    index: (self.qubits.len() - 1).into_py(py),
                    registers: PyList::empty_bound(py).unbind(),
                },
            )?,
        )?;
        self.add_wire(Wire::Qubit(qubit))?;
        Ok(qubit)
    }

    fn add_clbit_unchecked(&mut self, py: Python, bit: &Bound<PyAny>) -> PyResult<Clbit> {
        let clbit = self.clbits.add(py, bit, false)?;
        self.clbit_locations.bind(py).set_item(
            bit,
            Py::new(
                py,
                BitLocations {
                    index: (self.clbits.len() - 1).into_py(py),
                    registers: PyList::empty_bound(py).unbind(),
                },
            )?,
        )?;
        self.add_wire(Wire::Clbit(clbit))?;
        Ok(clbit)
    }

    pub(crate) fn get_node(&self, py: Python, node: NodeIndex) -> PyResult<Py<PyAny>> {
        self.unpack_into(py, node, self.dag.node_weight(node).unwrap())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    fn remove_op_node(&mut self, index: NodeIndex) {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::new();
        for (source, in_weight) in self
            .dag
            .edges_directed(index, Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(index, Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight.clone()));
                }
            }
        }
        for (source, target, weight) in edge_list {
            self.dag.add_edge(source, target, weight);
        }

        match self.dag.remove_node(index) {
            Some(NodeType::Operation(packed)) => Python::with_gil(|py| {
                let op_name = packed.op.name().to_string();
                self.decrement_op(op_name);
            }),
            _ => panic!("Must be called with valid operation node!"),
        }
    }

    /// Returns an iterator of the ancestors indices of a node.
    pub fn ancestors<'a>(&'a self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + 'a {
        core_ancestors(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of the descendants of a node as DAGOpNodes and DAGOutNodes.
    pub fn descendants<'a>(&'a self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + 'a {
        core_descendants(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    pub fn bfs_successors<'a>(
        &'a self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + 'a {
        core_bfs_successors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    fn unpack_into(&self, py: Python, id: NodeIndex, weight: &NodeType) -> PyResult<Py<PyAny>> {
        let dag_node = match weight {
            NodeType::QubitIn(qubit) => Py::new(
                py,
                DAGInNode::new(py, id, self.qubits.get(*qubit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::QubitOut(qubit) => Py::new(
                py,
                DAGOutNode::new(py, id, self.qubits.get(*qubit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::ClbitIn(clbit) => Py::new(
                py,
                DAGInNode::new(py, id, self.clbits.get(*clbit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::ClbitOut(clbit) => Py::new(
                py,
                DAGOutNode::new(py, id, self.clbits.get(*clbit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::Operation(packed) => {
                let qargs = self.qargs_cache.intern(packed.qubits_id);
                let cargs = self.cargs_cache.intern(packed.clbits_id);
                Py::new(
                    py,
                    DAGOpNode::new(
                        py,
                        id,
                        packed.op.clone(),
                        self.qubits.map_indices(qargs.as_slice()),
                        self.clbits.map_indices(cargs.as_slice()),
                        packed.params.clone(),
                        packed.extra_attrs.clone(),
                        (packed.qubits_id, packed.clbits_id).into_py(py),
                    ),
                )?
                .into_any()
            }
        };
        Ok(dag_node)
    }
}
