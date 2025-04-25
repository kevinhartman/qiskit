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

use crate::circuit_data::CircuitData;
use crate::imports::{get_std_gate_class, BARRIER, DELAY, MEASURE, RESET};
use crate::imports::{PARAMETER_EXPRESSION, QUANTUM_CIRCUIT, UNITARY_GATE};
use crate::{gate_matrix, impl_intopyobject_for_copy_pyclass, Qubit};
use approx::relative_eq;
use std::f64::consts::PI;
use std::ops::{Deref, Index};
use std::{fmt, vec};

use nalgebra::{Matrix2, Matrix4};
use ndarray::{array, aview2, Array2};
use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

use numpy::IntoPyArray;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyFloat, PyIterator, PyList, PyTuple};
use pyo3::{intern, IntoPyObjectExt, Python};

#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef)]
pub enum Param {
    Numeric(NumericParam),
    Obj(PyObject),
}

#[derive(Clone, Debug, IntoPyObject, IntoPyObjectRef)]
pub enum NumericParam {
    Float(f64),
    ParameterExpression(PyObject),
}

impl From<NumericParam> for Param {
    fn from(value: NumericParam) -> Self {
        Self::Numeric(value)
    }
}

impl NumericParam {
    pub fn eq(&self, py: Python, other: &NumericParam) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(a == b),
            [Self::Float(a), Self::ParameterExpression(b)] => b.bind(py).eq(a),
            [Self::ParameterExpression(a), Self::Float(b)] => a.bind(py).eq(b),
            [Self::ParameterExpression(a), Self::ParameterExpression(b)] => a.bind(py).eq(b),
        }
    }

    pub fn is_close(&self, py: Python, other: &NumericParam, max_relative: f64) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(relative_eq!(a, b, max_relative = max_relative)),
            _ => self.eq(py, other),
        }
    }

    /// Extract from a Python object without numeric coercion to float.  The default conversion will
    /// coerce integers into floats, but in things like `assign_parameters`, this is not always
    /// desirable.
    pub fn extract_no_coerce(ob: &Bound<PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyFloat>() {
            Ok(NumericParam::Float(ob.extract()?))
        } else if ob.is_instance(PARAMETER_EXPRESSION.get_bound(ob.py()))? {
            Ok(NumericParam::ParameterExpression(ob.clone().unbind()))
        } else {
            Err(PyValueError::new_err("not a numeric parameter"))
        }
    }

    /// Clones the [NumericParam] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            NumericParam::ParameterExpression(exp) => {
                NumericParam::ParameterExpression(exp.clone_ref(py))
            }
            NumericParam::Float(float) => NumericParam::Float(*float),
        }
    }
}

impl Param {
    pub fn as_numeric(&self) -> Option<&NumericParam> {
        match self {
            Param::Numeric(param) => Some(param),
            _ => None,
        }
    }

    pub fn eq(&self, py: Python, other: &Param) -> PyResult<bool> {
        match [self, other] {
            [Self::Numeric(a), Self::Numeric(b)] => a.eq(py, b),
            [Self::Numeric(NumericParam::ParameterExpression(a)), Self::Obj(b)] => a.bind(py).eq(b),
            [Self::Obj(a), Self::Numeric(NumericParam::ParameterExpression(b))] => a.bind(py).eq(b),
            _ => Ok(false),
        }
    }
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        if let Ok(numeric) = b.extract::<NumericParam>() {
            Ok(Param::Numeric(numeric))
        } else {
            Ok(Param::Obj(b.clone().unbind()))
        }
    }
}

impl<'py> FromPyObject<'py> for NumericParam {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))? {
            NumericParam::ParameterExpression(b.clone().unbind())
        } else {
            let Ok(val) = b.extract::<f64>()?;
            NumericParam::Float(val)
        })
    }
}

impl NumericParam {
    /// Get an iterator over any Python-space `Parameter` instances tracked within this `Param`.
    pub fn iter_parameters<'py>(&self, py: Python<'py>) -> PyResult<ParamParameterIter<'py>> {
        let parameters_attr = intern!(py, "parameters");
        match self {
            NumericParam::Float(_) => Ok(ParamParameterIter(None)),
            NumericParam::ParameterExpression(expr) => Ok(ParamParameterIter(Some(
                expr.bind(py).getattr(parameters_attr)?.try_iter()?,
            ))),
        }
    }
}

impl Param {
    /// Get an iterator over any Python-space `Parameter` instances tracked within this `Param`.
    pub fn iter_parameters<'py>(&self, py: Python<'py>) -> PyResult<ParamParameterIter<'py>> {
        let parameters_attr = intern!(py, "parameters");
        match self {
            Param::Numeric(numeric) => numeric.iter_parameters(py),
            Param::Obj(obj) => {
                let obj = obj.bind(py);
                if obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                    Ok(ParamParameterIter(Some(
                        obj.getattr(parameters_attr)?.try_iter()?,
                    )))
                } else {
                    Ok(ParamParameterIter(None))
                }
            }
        }
    }

    /// Clones the [Param] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            Param::Numeric(numeric) => Param::Numeric(numeric.clone_ref(py)),
            Param::Obj(obj) => Param::Obj(obj.clone_ref(py)),
        }
    }
}

// This impl allows for shared usage between [Param] and &[Param].
// Such blanked impl doesn't exist inherently due to Rust's type system limitations.
// See https://doc.rust-lang.org/std/convert/trait.AsRef.html#reflexivity for more information.
impl AsRef<Param> for Param {
    fn as_ref(&self) -> &Param {
        self
    }
}

impl AsRef<NumericParam> for NumericParam {
    fn as_ref(&self) -> &NumericParam {
        self
    }
}

// Conveniently converts an f64 into a `NumericParam`.
impl From<f64> for NumericParam {
    fn from(value: f64) -> Self {
        NumericParam::Float(value)
    }
}

/// Struct to provide iteration over Python-space `Parameter` instances within a `Param`.
pub struct ParamParameterIter<'py>(Option<Bound<'py, PyIterator>>);
impl<'py> Iterator for ParamParameterIter<'py> {
    type Item = PyResult<Bound<'py, PyAny>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(|iter| iter.next())
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn control_flow(&self) -> bool;
    // fn blocks(&self) -> Vec<CircuitData>;
    // fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
    // fn definition(&self, params: &[Param]) -> Option<CircuitData>;
    fn standard_gate(&self) -> Option<StandardGate>;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
#[derive(Debug)]
pub enum OperationRef<'a> {
    StandardGate(StandardGate),
    StandardInstruction(StandardInstruction),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(&'a UnitaryGate),
}

impl Operation for OperationRef<'_> {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Self::StandardGate(standard) => standard.name(),
            Self::StandardInstruction(instruction) => instruction.name(),
            Self::Gate(gate) => gate.name(),
            Self::Instruction(instruction) => instruction.name(),
            Self::Operation(operation) => operation.name(),
            Self::Unitary(unitary) => unitary.name(),
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        match self {
            Self::StandardGate(standard) => standard.num_qubits(),
            Self::StandardInstruction(instruction) => instruction.num_qubits(),
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(instruction) => instruction.num_qubits(),
            Self::Operation(operation) => operation.num_qubits(),
            Self::Unitary(unitary) => unitary.num_qubits(),
        }
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        match self {
            Self::StandardGate(standard) => standard.num_clbits(),
            Self::StandardInstruction(instruction) => instruction.num_clbits(),
            Self::Gate(gate) => gate.num_clbits(),
            Self::Instruction(instruction) => instruction.num_clbits(),
            Self::Operation(operation) => operation.num_clbits(),
            Self::Unitary(unitary) => unitary.num_clbits(),
        }
    }
    #[inline]
    fn num_params(&self) -> u32 {
        match self {
            Self::StandardGate(standard) => standard.num_params(),
            Self::StandardInstruction(instruction) => instruction.num_params(),
            Self::Gate(gate) => gate.num_params(),
            Self::Instruction(instruction) => instruction.num_params(),
            Self::Operation(operation) => operation.num_params(),
            Self::Unitary(unitary) => unitary.num_params(),
        }
    }
    #[inline]
    fn control_flow(&self) -> bool {
        match self {
            Self::StandardGate(standard) => standard.control_flow(),
            Self::StandardInstruction(instruction) => instruction.control_flow(),
            Self::Gate(gate) => gate.control_flow(),
            Self::Instruction(instruction) => instruction.control_flow(),
            Self::Operation(operation) => operation.control_flow(),
            Self::Unitary(unitary) => unitary.control_flow(),
        }
    }
    #[inline]
    fn standard_gate(&self) -> Option<StandardGate> {
        match self {
            Self::StandardGate(standard) => standard.standard_gate(),
            Self::StandardInstruction(instruction) => instruction.standard_gate(),
            Self::Gate(gate) => gate.standard_gate(),
            Self::Instruction(instruction) => instruction.standard_gate(),
            Self::Operation(operation) => operation.standard_gate(),
            Self::Unitary(unitary) => unitary.standard_gate(),
        }
    }
    #[inline]
    fn directive(&self) -> bool {
        match self {
            Self::StandardGate(standard) => standard.directive(),
            Self::StandardInstruction(instruction) => instruction.directive(),
            Self::Gate(gate) => gate.directive(),
            Self::Instruction(instruction) => instruction.directive(),
            Self::Operation(operation) => operation.directive(),
            Self::Unitary(unitary) => unitary.directive(),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum DelayUnit {
    NS,
    PS,
    US,
    MS,
    S,
    DT,
    EXPR,
}

unsafe impl ::bytemuck::CheckedBitPattern for DelayUnit {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 7
    }
}
unsafe impl ::bytemuck::NoUninit for DelayUnit {}

impl fmt::Display for DelayUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DelayUnit::NS => "ns",
                DelayUnit::PS => "ps",
                DelayUnit::US => "us",
                DelayUnit::MS => "ms",
                DelayUnit::S => "s",
                DelayUnit::DT => "dt",
                DelayUnit::EXPR => "expr",
            }
        )
    }
}

impl<'py> FromPyObject<'py> for DelayUnit {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        let str: String = b.extract()?;
        Ok(match str.as_str() {
            "ns" => DelayUnit::NS,
            "ps" => DelayUnit::PS,
            "us" => DelayUnit::US,
            "ms" => DelayUnit::MS,
            "s" => DelayUnit::S,
            "dt" => DelayUnit::DT,
            "expr" => DelayUnit::EXPR,
            unknown_unit => {
                return Err(PyValueError::new_err(format!(
                    "Unit '{}' is invalid.",
                    unknown_unit
                )));
            }
        })
    }
}

/// An internal type used to further discriminate the payload of a `PackedOperation` when its
/// discriminant is `PackedOperationType::StandardInstruction`.
///
/// This is also used to tag standard instructions via the `_standard_instruction_type` class
/// attribute in the corresponding Python class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int)]
#[repr(u8)]
pub(crate) enum StandardInstructionType {
    Barrier = 0,
    Delay = 1,
    Measure = 2,
    Reset = 3,
}

unsafe impl ::bytemuck::CheckedBitPattern for StandardInstructionType {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 4
    }
}
unsafe impl ::bytemuck::NoUninit for StandardInstructionType {}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
pub enum StandardInstruction {
    Barrier(u32),
    Delay(DelayUnit),
    Measure,
    Reset,
}

// This must be kept up-to-date with `StandardInstruction` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_INSTRUCTION_SIZE: usize = 4;

impl Operation for StandardInstruction {
    fn name(&self) -> &str {
        match self {
            StandardInstruction::Barrier(_) => "barrier",
            StandardInstruction::Delay(_) => "delay",
            StandardInstruction::Measure => "measure",
            StandardInstruction::Reset => "reset",
        }
    }

    fn num_qubits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(num_qubits) => *num_qubits,
            StandardInstruction::Delay(_) => 1,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 1,
        }
    }

    fn num_clbits(&self) -> u32 {
        match self {
            StandardInstruction::Barrier(_) => 0,
            StandardInstruction::Delay(_) => 0,
            StandardInstruction::Measure => 1,
            StandardInstruction::Reset => 0,
        }
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        match self {
            StandardInstruction::Barrier(_) => true,
            StandardInstruction::Delay(_) => false,
            StandardInstruction::Measure => false,
            StandardInstruction::Reset => false,
        }
    }
}

impl StandardInstruction {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<&[Param]>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = label
            .map(|label| [("label", label.into_py_any(py)?)].into_py_dict(py))
            .transpose()?;
        let out = match self {
            StandardInstruction::Barrier(num_qubits) => {
                BARRIER.get_bound(py).call((num_qubits,), kwargs.as_ref())?
            }
            StandardInstruction::Delay(unit) => {
                let duration = &params.unwrap()[0];
                DELAY
                    .get_bound(py)
                    .call1((duration.into_py_any(py)?, unit.to_string()))?
            }
            StandardInstruction::Measure => MEASURE.get_bound(py).call((), kwargs.as_ref())?,
            StandardInstruction::Reset => RESET.get_bound(py).call((), kwargs.as_ref())?,
        };

        Ok(out.unbind())
    }
}

pub trait AsMatrix {
    type Matrix;

    fn matrix(&self) -> Self::Matrix;
}

pub trait IntoBlockReferences {
    type BlockRef;
    type BlockReferences: Iterator<Item = Self::BlockRef>;

    fn blocks(&self) -> Self::BlockReferences;
}

pub trait AsCircuit {
    fn definition(&self) -> Option<CircuitData>;
}

pub trait ParameterizedOperation {
    type ParamType;
    type Parameters: IntoIterator<Item = Self::ParamType>;

    fn params(&self) -> Self::Parameters;
}

/// Represents an operation with bound parameters, what is known as an "instruction" in classical
/// computing.
///
/// Notably, the `StandardGateRef` and `StandardInstructionRef` store a reference to the
/// instruction's parameters while the Python object variants own their parameters and
/// can get them from the PyObject.
///
/// This intentionally does NOT implement [ParameterizedOperation] itself since not all variants implement it
/// using the same associated types.
#[derive(Debug)]
pub enum ParameterizedOperationRef<'a> {
    StandardGate(StandardGateRef<'a>),
    StandardInstruction(StandardInstructionRef<'a>),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
    Unitary(UnitaryGateRef<'a>),
}

impl<'a> ParameterizedOperationRef<'a> {
    fn blocks(&self) -> impl Iterator<Item = CircuitData> {
        match self {
            ParameterizedOperationRef::Instruction(i) => i.blocks(),
            _ => None,
        }
    }

    pub fn matrix(&self) -> Option<Array2<Complex64>> {
        match self {
            Self::StandardGate(s) => s.matrix(),
            Self::Gate(g) => g.matrix(),
            Self::Unitary(u) => Some(u.matrix()),
            _ => None,
        }
    }

    fn definition(&self) -> Option<CircuitData> {
        match self {
            Self::StandardGate(s) => s.definition(),
            Self::Gate(g) => g.definition(),
            Self::Instruction(i) => i.definition(),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct StandardGateRef<'a> {
    gate: StandardGate,
    params: &'a [NumericParam],
}

impl<'a> StandardGateRef {
    pub fn inverse(&self) -> Option<(StandardGate, SmallVec<[NumericParam; 3]>)> {
        let params = self.params;
        match self.gate {
            StandardGate::GlobalPhase => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::GlobalPhase,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::H => Some((StandardGate::H, smallvec![])),
            StandardGate::I => Some((StandardGate::I, smallvec![])),
            StandardGate::X => Some((StandardGate::X, smallvec![])),
            StandardGate::Y => Some((StandardGate::Y, smallvec![])),
            StandardGate::Z => Some((StandardGate::Z, smallvec![])),
            StandardGate::Phase => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::Phase,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::R => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::R,
                        smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                    )
                },
            )),
            StandardGate::RX => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RX,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::RY => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RY,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::RZ => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RZ,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::S => Some((StandardGate::Sdg, smallvec![])),
            StandardGate::Sdg => Some((StandardGate::S, smallvec![])),
            StandardGate::SX => Some((StandardGate::SXdg, smallvec![])),
            StandardGate::SXdg => Some((StandardGate::SX, smallvec![])),
            StandardGate::T => Some((StandardGate::Tdg, smallvec![])),
            StandardGate::Tdg => Some((StandardGate::T, smallvec![])),
            StandardGate::U => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::U,
                        smallvec![
                            multiply_param(&params[0], -1.0, py),
                            multiply_param(&params[2], -1.0, py),
                            multiply_param(&params[1], -1.0, py),
                        ],
                    )
                },
            )),
            StandardGate::U1 => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::U1,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::U2 => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::U2,
                        smallvec![
                            add_param(&multiply_param(&params[1], -1.0, py), -PI, py),
                            add_param(&multiply_param(&params[0], -1.0, py), PI, py),
                        ],
                    )
                },
            )),
            StandardGate::U3 => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::U3,
                        smallvec![
                            multiply_param(&params[0], -1.0, py),
                            multiply_param(&params[2], -1.0, py),
                            multiply_param(&params[1], -1.0, py),
                        ],
                    )
                },
            )),
            StandardGate::CH => Some((StandardGate::CH, smallvec![])),
            StandardGate::CX => Some((StandardGate::CX, smallvec![])),
            StandardGate::CY => Some((StandardGate::CY, smallvec![])),
            StandardGate::CZ => Some((StandardGate::CZ, smallvec![])),
            StandardGate::DCX => None, // the inverse in not a StandardGate
            StandardGate::ECR => Some((StandardGate::ECR, smallvec![])),
            StandardGate::Swap => Some((StandardGate::Swap, smallvec![])),
            StandardGate::ISwap => None, // the inverse in not a StandardGate
            StandardGate::CPhase => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CPhase,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::CRX => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CRX,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::CRY => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CRY,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::CRZ => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CRZ,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::CS => Some((StandardGate::CSdg, smallvec![])),
            StandardGate::CSdg => Some((StandardGate::CS, smallvec![])),
            StandardGate::CSX => None, // the inverse in not a StandardGate
            StandardGate::CU => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CU,
                        smallvec![
                            multiply_param(&params[0], -1.0, py),
                            multiply_param(&params[2], -1.0, py),
                            multiply_param(&params[1], -1.0, py),
                            multiply_param(&params[3], -1.0, py),
                        ],
                    )
                },
            )),
            StandardGate::CU1 => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CU1,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::CU3 => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::CU3,
                        smallvec![
                            multiply_param(&params[0], -1.0, py),
                            multiply_param(&params[2], -1.0, py),
                            multiply_param(&params[1], -1.0, py),
                        ],
                    )
                },
            )),
            StandardGate::RXX => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RXX,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::RYY => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RYY,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::RZZ => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RZZ,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::RZX => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::RZX,
                        smallvec![multiply_param(&params[0], -1.0, py)],
                    )
                },
            )),
            StandardGate::XXMinusYY => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::XXMinusYY,
                        smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                    )
                },
            )),
            StandardGate::XXPlusYY => Some(Python::with_gil(
                |py| -> (StandardGate, SmallVec<[NumericParam; 3]>) {
                    (
                        StandardGate::XXPlusYY,
                        smallvec![multiply_param(&params[0], -1.0, py), params[1].clone()],
                    )
                },
            )),
            StandardGate::CCX => Some((StandardGate::CCX, smallvec![])),
            StandardGate::CCZ => Some((StandardGate::CCZ, smallvec![])),
            StandardGate::CSwap => Some((StandardGate::CSwap, smallvec![])),
            StandardGate::RCCX => None, // the inverse in not a StandardGate
            StandardGate::C3X => Some((StandardGate::C3X, smallvec![])),
            StandardGate::C3SX => None, // the inverse in not a StandardGate
            StandardGate::RC3X => None, // the inverse in not a StandardGate
        }
    }
}

impl<'a> AsMatrix for StandardGateRef<'a> {
    type Matrix = Option<Array2<Complex64>>;

    fn matrix(&self) -> Self::Matrix {
        let params = self.params;
        match self.gate {
            StandardGate::GlobalPhase => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::H => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            StandardGate::I => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            StandardGate::X => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Y => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Z => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Phase => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::R => match params {
                [NumericParam::Float(theta), NumericParam::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            StandardGate::RX => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::RY => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::RZ => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::S => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Sdg => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
            StandardGate::SX => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::SXdg => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
                _ => None,
            },
            StandardGate::T => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Tdg => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            StandardGate::U => match params {
                [NumericParam::Float(theta), NumericParam::Float(phi), NumericParam::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            StandardGate::U1 => match params[0] {
                NumericParam::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            StandardGate::U2 => match params {
                [NumericParam::Float(phi), NumericParam::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            StandardGate::U3 => match params {
                [NumericParam::Float(theta), NumericParam::Float(phi), NumericParam::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            StandardGate::CH => match params {
                [] => Some(aview2(&gate_matrix::CH_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CX => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CY => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CZ => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            StandardGate::DCX => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::ECR => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            StandardGate::Swap => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            StandardGate::ISwap => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CPhase => match params {
                [NumericParam::Float(lam)] => Some(aview2(&gate_matrix::cp_gate(*lam)).to_owned()),
                _ => None,
            },
            StandardGate::CRX => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::CRY => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::CRZ => match params {
                [NumericParam::Float(theta)] => {
                    Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::CS => match params {
                [] => Some(aview2(&gate_matrix::CS_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CSdg => match params {
                [] => Some(aview2(&gate_matrix::CSDG_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CSX => match params {
                [] => Some(aview2(&gate_matrix::CSX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CU => match params {
                [NumericParam::Float(theta), NumericParam::Float(phi), NumericParam::Float(lam), NumericParam::Float(gamma)] => {
                    Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned())
                }
                _ => None,
            },
            StandardGate::CU1 => match params[0] {
                NumericParam::Float(lam) => Some(aview2(&gate_matrix::cu1_gate(lam)).to_owned()),
                _ => None,
            },
            StandardGate::CU3 => match params {
                [NumericParam::Float(theta), NumericParam::Float(phi), NumericParam::Float(lam)] => {
                    Some(aview2(&gate_matrix::cu3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            StandardGate::RXX => match params[0] {
                NumericParam::Float(theta) => {
                    Some(aview2(&gate_matrix::rxx_gate(theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::RYY => match params[0] {
                NumericParam::Float(theta) => {
                    Some(aview2(&gate_matrix::ryy_gate(theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::RZZ => match params[0] {
                NumericParam::Float(theta) => {
                    Some(aview2(&gate_matrix::rzz_gate(theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::RZX => match params[0] {
                NumericParam::Float(theta) => {
                    Some(aview2(&gate_matrix::rzx_gate(theta)).to_owned())
                }
                _ => None,
            },
            StandardGate::XXMinusYY => match params {
                [NumericParam::Float(theta), NumericParam::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            StandardGate::XXPlusYY => match params {
                [NumericParam::Float(theta), NumericParam::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            StandardGate::CCX => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CCZ => match params {
                [] => Some(aview2(&gate_matrix::CCZ_GATE).to_owned()),
                _ => None,
            },
            StandardGate::CSwap => match params {
                [] => Some(aview2(&gate_matrix::CSWAP_GATE).to_owned()),
                _ => None,
            },
            StandardGate::RCCX => match params {
                [] => Some(aview2(&gate_matrix::RCCX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::C3X => match params {
                [] => Some(aview2(&gate_matrix::C3X_GATE).to_owned()),
                _ => None,
            },
            StandardGate::C3SX => match params {
                [] => Some(aview2(&gate_matrix::C3SX_GATE).to_owned()),
                _ => None,
            },
            StandardGate::RC3X => match params {
                [] => Some(aview2(&gate_matrix::RC3X_GATE).to_owned()),
                _ => None,
            },
        }
    }
}

impl<'a> AsCircuit for StandardGateRef<'a> {
    fn definition(&self) -> Option<CircuitData> {
        let params = self.params;
        match self.gate {
            StandardGate::GlobalPhase => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(py, 0, [], params[0].clone())
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::H => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            smallvec![
                                NumericParam::Float(PI / 2.),
                                FLOAT_ZERO,
                                NumericParam::Float(PI)
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::I => None,
            StandardGate::X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            smallvec![NumericParam::Float(PI), FLOAT_ZERO, NumericParam::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::Y => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            smallvec![
                                NumericParam::Float(PI),
                                NumericParam::Float(PI / 2.),
                                NumericParam::Float(PI / 2.),
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),

            StandardGate::Z => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![NumericParam::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::Phase => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            smallvec![FLOAT_ZERO, FLOAT_ZERO, params[0].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::R => Python::with_gil(|py| -> Option<CircuitData> {
                let theta_expr = clone_param(&params[0], py);
                let phi_expr1 = add_param(&params[1], -PI / 2., py);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0, py);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(StandardGate::U, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RX => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::R,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RY => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::R,
                            smallvec![theta.clone(), NumericParam::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RZ => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![theta.clone()],
                            smallvec![Qubit(0)],
                        )],
                        multiply_param(theta, -0.5, py),
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::S => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![NumericParam::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::Sdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![NumericParam::Float(-PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::SX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (StandardGate::Sdg, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::Sdg, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::SXdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (StandardGate::S, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::S, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::T => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![NumericParam::Float(PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::Tdg => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            smallvec![NumericParam::Float(-PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::U => None,
            StandardGate::U1 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::Phase,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::U2 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            smallvec![
                                NumericParam::Float(PI / 2.),
                                params[0].clone(),
                                params[1].clone()
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::U3 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            StandardGate::U,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CH => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::S, smallvec![], q1.clone()),
                            (StandardGate::H, smallvec![], q1.clone()),
                            (StandardGate::T, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_1),
                            (StandardGate::Tdg, smallvec![], q1.clone()),
                            (StandardGate::H, smallvec![], q1.clone()),
                            (StandardGate::Sdg, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),

            StandardGate::CX => None,
            StandardGate::CY => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::Sdg, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_1),
                            (StandardGate::S, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CZ => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::H, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_1),
                            (StandardGate::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::DCX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::ECR => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::RZX,
                                smallvec![NumericParam::Float(PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                            (StandardGate::X, smallvec![], smallvec![Qubit(0)]),
                            (
                                StandardGate::RZX,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::Swap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::ISwap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::S, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::S, smallvec![], smallvec![Qubit(1)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(0)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CPhase => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::Phase,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q0,
                            ),
                            (StandardGate::CX, smallvec![], q0_1.clone()),
                            (
                                StandardGate::Phase,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                q1.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_1),
                            (
                                StandardGate::Phase,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CRX => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 2.)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U,
                                smallvec![
                                    multiply_param(theta, -0.5, py),
                                    NumericParam::Float(0.0),
                                    NumericParam::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U,
                                smallvec![
                                    multiply_param(theta, 0.5, py),
                                    NumericParam::Float(-PI / 2.),
                                    NumericParam::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        NumericParam::Float(0.0),
                    )
                        .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            StandardGate::CRY => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        NumericParam::Float(0.0),
                    )
                        .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            StandardGate::CRZ => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::RZ,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::RZ,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        NumericParam::Float(0.0),
                    )
                        .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            StandardGate::CS => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 4.)],
                                q0,
                            ),
                            (StandardGate::CX, smallvec![], q0_1.clone()),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                q1.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_1),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 4.)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CSdg => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                q0,
                            ),
                            (StandardGate::CX, smallvec![], q0_1.clone()),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 4.)],
                                q1.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_1),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CSX => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::H, smallvec![], q1.clone()),
                            (
                                StandardGate::CPhase,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q0_1,
                            ),
                            (StandardGate::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CU => Python::with_gil(|py| -> Option<CircuitData> {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::Phase,
                                smallvec![params[3].clone()],
                                smallvec![Qubit(0)],
                            ),
                            (
                                StandardGate::Phase,
                                smallvec![param_second_p],
                                smallvec![Qubit(0)],
                            ),
                            (
                                StandardGate::Phase,
                                smallvec![param_third_p],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CU1 => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::U1,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(0)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U1,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U1,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CU3 => Python::with_gil(|py| -> Option<CircuitData> {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::U1,
                                smallvec![param_first_u1],
                                smallvec![Qubit(0)],
                            ),
                            (
                                StandardGate::U1,
                                smallvec![param_second_u1],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U3,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::U3,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RXX => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::H, smallvec![], q0.clone()),
                            (StandardGate::H, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_q1.clone()),
                            (StandardGate::RZ, smallvec![theta.clone()], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_q1),
                            (StandardGate::H, smallvec![], q1),
                            (StandardGate::H, smallvec![], q0),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::RX,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q0.clone(),
                            ),
                            (
                                StandardGate::RX,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q1.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_q1.clone()),
                            (StandardGate::RZ, smallvec![theta.clone()], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_q1),
                            (
                                StandardGate::RX,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q0,
                            ),
                            (
                                StandardGate::RX,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RZZ => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::CX, smallvec![], q0_q1.clone()),
                            (StandardGate::RZ, smallvec![theta.clone()], q1),
                            (StandardGate::CX, smallvec![], q0_q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RZX => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::H, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_q1.clone()),
                            (StandardGate::RZ, smallvec![theta.clone()], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_q1),
                            (StandardGate::H, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::XXMinusYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                StandardGate::RZ,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q1.clone(),
                            ),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q0.clone(),
                            ),
                            (StandardGate::SX, smallvec![], q0.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q0.clone(),
                            ),
                            (StandardGate::S, smallvec![], q1.clone()),
                            (StandardGate::CX, smallvec![], q0_1.clone()),
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, 0.5, py)],
                                q0.clone(),
                            ),
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_1),
                            (StandardGate::Sdg, smallvec![], q1.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q0.clone(),
                            ),
                            (StandardGate::SXdg, smallvec![], q0.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q0,
                            ),
                            (StandardGate::RZ, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::XXPlusYY => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (StandardGate::RZ, smallvec![beta.clone()], q0.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q1.clone(),
                            ),
                            (StandardGate::SX, smallvec![], q1.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q1.clone(),
                            ),
                            (StandardGate::S, smallvec![], q0.clone()),
                            (StandardGate::CX, smallvec![], q1_0.clone()),
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (
                                StandardGate::RY,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q0.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q1_0),
                            (StandardGate::Sdg, smallvec![], q0.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(-PI / 2.)],
                                q1.clone(),
                            ),
                            (StandardGate::SXdg, smallvec![], q1.clone()),
                            (
                                StandardGate::RZ,
                                smallvec![NumericParam::Float(PI / 2.)],
                                q1,
                            ),
                            (
                                StandardGate::RZ,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q0,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CCX => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (StandardGate::H, smallvec![], q2.clone()),
                            (StandardGate::CX, smallvec![], q1_2.clone()),
                            (StandardGate::Tdg, smallvec![], q2.clone()),
                            (StandardGate::CX, smallvec![], q0_2.clone()),
                            (StandardGate::T, smallvec![], q2.clone()),
                            (StandardGate::CX, smallvec![], q1_2),
                            (StandardGate::Tdg, smallvec![], q2.clone()),
                            (StandardGate::CX, smallvec![], q0_2),
                            (StandardGate::T, smallvec![], q1.clone()),
                            (StandardGate::T, smallvec![], q2.clone()),
                            (StandardGate::H, smallvec![], q2),
                            (StandardGate::CX, smallvec![], q0_1.clone()),
                            (StandardGate::T, smallvec![], q0),
                            (StandardGate::Tdg, smallvec![], q1),
                            (StandardGate::CX, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),

            StandardGate::CCZ => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (StandardGate::H, smallvec![], smallvec![Qubit(2)]),
                            (
                                StandardGate::CCX,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(2)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::CSwap => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                            (
                                StandardGate::CCX,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),

            StandardGate::RCCX => Python::with_gil(|py| -> Option<CircuitData> {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                q2.clone(),
                            ),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                q2.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q1_2.clone()),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                q2.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q0_2),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                q2.clone(),
                            ),
                            (StandardGate::CX, smallvec![], q1_2),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                q2.clone(),
                            ),
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                q2,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::C3X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(0)],
                            ),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::Phase,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),

            StandardGate::C3SX => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(0), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                            (
                                StandardGate::CU1,
                                smallvec![NumericParam::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (StandardGate::H, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            StandardGate::RC3X => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (StandardGate::CX, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                StandardGate::U1,
                                smallvec![NumericParam::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                StandardGate::U2,
                                smallvec![FLOAT_ZERO, NumericParam::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
        }
    }
}

impl<'a> ParameterizedOperation for StandardGateRef<'a> {
    type ParamType = &'a NumericParam;
    type Parameters = &'a [NumericParam];

    fn params(&self) -> Self::Parameters {
        self.params
    }
}

#[derive(Debug)]
pub struct StandardInstructionRef<'a> {
    instruction: StandardInstruction,
    params: &'a [Param],
}

impl<'a> StandardInstructionRef<'a> {
    pub fn new(instruction: StandardInstruction, params: &'a [Param]) -> Self {
        Self {
            instruction,
            params,
        }
    }
}

impl<'a> ParameterizedOperation for StandardInstructionRef<'a> {
    type ParamType = &'a Param;
    type Parameters = &'a [Param];

    fn params(&self) -> Self::Parameters {
        self.params
    }
}

impl<'a> StandardGateRef<'a> {
    pub fn new(gate: StandardGate, params: &'a [NumericParam]) -> Self {
        Self { gate, params }
    }

    #[inline]
    pub fn gate(&self) -> StandardGate {
        self.gate
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
#[pyclass(module = "qiskit._accelerate.circuit", eq, eq_int)]
pub enum StandardGate {
    GlobalPhase = 0,
    H = 1,
    I = 2,
    X = 3,
    Y = 4,
    Z = 5,
    Phase = 6,
    R = 7,
    RX = 8,
    RY = 9,
    RZ = 10,
    S = 11,
    Sdg = 12,
    SX = 13,
    SXdg = 14,
    T = 15,
    Tdg = 16,
    U = 17,
    U1 = 18,
    U2 = 19,
    U3 = 20,
    CH = 21,
    CX = 22,
    CY = 23,
    CZ = 24,
    DCX = 25,
    ECR = 26,
    Swap = 27,
    ISwap = 28,
    CPhase = 29,
    CRX = 30,
    CRY = 31,
    CRZ = 32,
    CS = 33,
    CSdg = 34,
    CSX = 35,
    CU = 36,
    CU1 = 37,
    CU3 = 38,
    RXX = 39,
    RYY = 40,
    RZZ = 41,
    RZX = 42,
    XXMinusYY = 43,
    XXPlusYY = 44,
    CCX = 45,
    CCZ = 46,
    CSwap = 47,
    RCCX = 48,
    C3X = 49,
    C3SX = 50,
    RC3X = 51,
    // Remember to update StandardGate::is_valid_bit_pattern below
    // if you add or remove this enum's variants!
}
impl_intopyobject_for_copy_pyclass!(StandardGate);

unsafe impl ::bytemuck::CheckedBitPattern for StandardGate {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 52
    }
}
unsafe impl ::bytemuck::NoUninit for StandardGate {}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0-9
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10-19
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 20-29
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 30-39
    2, 2, 2, 2, 2, 3, 3, 3, 3, 4, // 40-49
    4, 4, // 50-51
];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    1, 0, 0, 0, 0, 0, 1, 2, 1, 1, // 0-9
    1, 0, 0, 0, 0, 0, 0, 3, 1, 2, // 10-19
    3, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 0, 0, 0, 4, 1, 3, 1, // 30-39
    1, 1, 1, 2, 2, 0, 0, 0, 0, 0, // 40-49
    0, 0, // 50-51
];

static STANDARD_GATE_NUM_CTRL_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10-19
    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 30-39
    0, 0, 0, 0, 0, 2, 2, 1, 0, 3, // 40-49
    3, 0, // 50-51
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "global_phase", // 0
    "h",            // 1
    "id",           // 2
    "x",            // 3
    "y",            // 4
    "z",            // 5
    "p",            // 6
    "r",            // 7
    "rx",           // 8
    "ry",           // 9
    "rz",           // 10
    "s",            // 11
    "sdg",          // 12
    "sx",           // 13
    "sxdg",         // 14
    "t",            // 15
    "tdg",          // 16
    "u",            // 17
    "u1",           // 18
    "u2",           // 19
    "u3",           // 20
    "ch",           // 21
    "cx",           // 22
    "cy",           // 23
    "cz",           // 24
    "dcx",          // 25
    "ecr",          // 26
    "swap",         // 27
    "iswap",        // 28
    "cp",           // 29
    "crx",          // 30
    "cry",          // 31
    "crz",          // 32
    "cs",           // 33
    "csdg",         // 34
    "csx",          // 35
    "cu",           // 36
    "cu1",          // 37
    "cu3",          // 38
    "rxx",          // 39
    "ryy",          // 40
    "rzz",          // 41
    "rzx",          // 42
    "xx_minus_yy",  // 43
    "xx_plus_yy",   // 44
    "ccx",          // 45
    "ccz",          // 46
    "cswap",        // 47
    "rccx",         // 48
    "mcx",          // 49 ("c3x")
    "c3sx",         // 50
    "rcccx",        // 51 ("rc3x")
];

/// Get a slice of all standard gate names.
pub fn get_standard_gate_names() -> &'static [&'static str] {
    &STANDARD_GATE_NAME
}

impl StandardGate {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<&[Param]>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = get_std_gate_class(py, *self)?;
        let args = match params.unwrap_or(&[]) {
            &[] => PyTuple::empty(py),
            params => PyTuple::new(py, params.iter().map(|x| x.into_pyobject(py).unwrap()))?,
        };
        if let Some(label) = label {
            let kwargs = [("label", label.into_pyobject(py)?)].into_py_dict(py)?;
            gate_class.call(py, args, Some(&kwargs))
        } else {
            gate_class.call(py, args, None)
        }
    }

    pub fn num_ctrl_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_CTRL_QUBITS[*self as usize]
    }
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix<'py>(
        &self,
        py: Python<'py>,
        params: Vec<NumericParam>,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        let gate = StandardGateRef::new(*self, params.as_slice());
        gate.matrix().map(|x| x.into_pyarray(py))
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<CircuitData> {
        self.definition(&params)
    }

    pub fn _inverse(&self, params: Vec<Param>) -> Option<(StandardGate, SmallVec<[Param; 3]>)> {
        self.inverse(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_ctrl_qubits(&self) -> u32 {
        self.num_ctrl_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }

    #[getter]
    pub fn is_controlled_gate(&self) -> bool {
        self.num_ctrl_qubits() > 0
    }

    #[getter]
    pub fn get_gate_class(&self, py: Python) -> PyResult<&'static Py<PyAny>> {
        get_std_gate_class(py, *self)
    }

    #[staticmethod]
    pub fn all_gates(py: Python) -> PyResult<Bound<PyList>> {
        PyList::new(
            py,
            (0..STANDARD_GATE_SIZE as u8).map(::bytemuck::checked::cast::<_, Self>),
        )
    }

    pub fn __hash__(&self) -> isize {
        *self as isize
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 52;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }

    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }

    fn directive(&self) -> bool {
        false
    }
}

const FLOAT_ZERO: NumericParam = NumericParam::Float(0.0);

// Return explicitly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &NumericParam, py: Python) -> NumericParam {
    match param {
        NumericParam::Float(theta) => NumericParam::Float(*theta),
        NumericParam::ParameterExpression(theta) => {
            NumericParam::ParameterExpression(theta.clone_ref(py))
        }
    }
}

/// Multiply a ``Param`` with a float.
pub fn multiply_param(param: &NumericParam, mult: f64, py: Python) -> NumericParam {
    match param {
        NumericParam::Float(theta) => NumericParam::Float(theta * mult),
        NumericParam::ParameterExpression(theta) => NumericParam::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__rmul__"), (mult,))
                .expect("Multiplication of Parameter expression by float failed."),
        ),
    }
}

/// Multiply two ``Param``s.
pub fn multiply_params(param1: NumericParam, param2: NumericParam, py: Python) -> NumericParam {
    match (&param1, &param2) {
        (NumericParam::Float(theta), NumericParam::Float(lambda)) => {
            NumericParam::Float(theta * lambda)
        }
        (param, NumericParam::Float(theta)) => multiply_param(param, *theta, py),
        (NumericParam::Float(theta), param) => multiply_param(param, *theta, py),
        (NumericParam::ParameterExpression(p1), NumericParam::ParameterExpression(p2)) => {
            NumericParam::ParameterExpression(
                p1.clone_ref(py)
                    .call_method1(py, intern!(py, "__rmul__"), (p2,))
                    .expect("Parameter expression multiplication failed"),
            )
        }
    }
}

pub fn add_param(param: &NumericParam, summand: f64, py: Python) -> NumericParam {
    match param {
        NumericParam::Float(theta) => NumericParam::Float(*theta + summand),
        NumericParam::ParameterExpression(theta) => NumericParam::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (summand,))
                .expect("Sum of Parameter expression and float failed."),
        ),
    }
}

pub fn radd_param(param1: NumericParam, param2: NumericParam, py: Python) -> NumericParam {
    match [&param1, &param2] {
        [NumericParam::Float(theta), NumericParam::Float(lambda)] => {
            NumericParam::Float(theta + lambda)
        }
        [NumericParam::Float(theta), NumericParam::ParameterExpression(_lambda)] => {
            add_param(&param2, *theta, py)
        }
        [NumericParam::ParameterExpression(_theta), NumericParam::Float(lambda)] => {
            add_param(&param1, *lambda, py)
        }
        [NumericParam::ParameterExpression(theta), NumericParam::ParameterExpression(lambda)] => {
            NumericParam::ParameterExpression(
                theta
                    .clone_ref(py)
                    .call_method1(py, intern!(py, "__radd__"), (lambda,))
                    .expect("Parameter expression addition failed"),
            )
        }
    }
}

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyInstruction {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub control_flow: bool,
    pub instruction: PyObject,
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        self.control_flow
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }
    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.instruction.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

pub struct PyParametersIter {
    params: vec::IntoIter<Py<PyAny>>
}

impl Iterator for PyParametersIter {
    type Item = Py<PyAny>;

    fn next(&mut self) -> Option<Self::Item> {
        self.params.next()
    }
}

impl ParameterizedOperation for PyInstruction {
    type ParamType = Py<PyAny>;
    type Parameters = PyParametersIter;

    fn params(&self) -> Self::Parameters {
        Python::with_gil(|py| -> Vec<CircuitData> {
            let params = self.instruction.bind(py).getattr("params").unwrap();
            PyParametersIter {
                params: params
                    .try_iter()
                    .unwrap()
                    .map(|b| {
                        b.unwrap().unbind()
                    })
                    .collect()
            }
        })
    }
}

impl AsCircuit for PyInstruction {
    fn definition(&self) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.instruction.getattr(py, intern!(py, "definition")) {
                Ok(definition) => definition
                    .getattr(py, intern!(py, "_data"))
                    .ok()?
                    .extract::<CircuitData>(py)
                    .ok(),
                Err(_) => None,
            }
        })
    }
}

impl IntoBlockReferences for PyInstruction {
    type BlockRef = CircuitData;
    type BlockReferences = vec::IntoIter<Self::BlockRef>;

    fn blocks(&self) -> Self::BlockReferences {
        if !self.control_flow {
            return vec::IntoIter::default();
        }
        Python::with_gil(|py| -> Self::BlockReferences {
            // We expect that if PyInstruction::control_flow is true then the operation WILL
            // have a 'blocks' attribute which is a tuple of the Python QuantumCircuit.
            let raw_blocks = self.instruction.getattr(py, "blocks").unwrap();
            let blocks: &Bound<PyTuple> = raw_blocks.downcast_bound::<PyTuple>(py).unwrap();
            blocks
                .iter()
                .map(|b| {
                    b.getattr(intern!(py, "_data"))
                        .unwrap()
                        .extract::<CircuitData>()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
    }
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: PyObject,
}

impl Operation for PyGate {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        Python::with_gil(|py| -> Option<StandardGate> {
            match self.gate.getattr(py, intern!(py, "_standard_gate")) {
                Ok(stdgate) => stdgate.extract(py).unwrap_or_default(),
                Err(_) => None,
            }
        })
    }
    fn directive(&self) -> bool {
        false
    }
}

impl AsCircuit for PyGate {
    fn definition(&self) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.gate.getattr(py, intern!(py, "definition")) {
                Ok(definition) => definition
                    .getattr(py, intern!(py, "_data"))
                    .ok()?
                    .extract::<CircuitData>(py)
                    .ok(),
                Err(_) => None,
            }
        })
    }
}

impl ParameterizedOperation for PyGate {
    type ParamType = Py<PyAny>;
    type Parameters = PyParametersIter;

    fn params(&self) -> &[Self::ParamType] {
        todo!()
    }
}

impl AsMatrix for PyGate {
    type Matrix = Option<Array2<Complex64>>;

    fn matrix(&self) -> Self::Matrix {
        Python::with_gil(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<PyObject> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyOperation {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub operation: PyObject,
}

impl Operation for PyOperation {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.operation.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

impl ParameterizedOperation for PyOperation {
    type ParamType = Py<PyAny>;
    type Parameters = PyParametersIter;

    fn params(&self) -> &[Self::ParamType] {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub enum ArrayType {
    NDArray(Array2<Complex64>),
    OneQ(Matrix2<Complex64>),
    TwoQ(Matrix4<Complex64>),
}

/// This class is a rust representation of a UnitaryGate in Python,
/// a gate represented solely by it's unitary matrix.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct UnitaryGate {
    pub array: ArrayType,
}

impl PartialEq for UnitaryGate {
    fn eq(&self, other: &Self) -> bool {
        match (&self.array, &other.array) {
            (ArrayType::OneQ(mat1), ArrayType::OneQ(mat2)) => mat1 == mat2,
            (ArrayType::TwoQ(mat1), ArrayType::TwoQ(mat2)) => mat1 == mat2,
            // we could also slightly optimize comparisons between NDArray and OneQ/TwoQ if
            // this becomes performance critical
            _ => {
                self.matrix() == other.matrix()
            },
        }
    }
}

impl Operation for UnitaryGate {
    fn name(&self) -> &str {
        "unitary"
    }
    fn num_qubits(&self) -> u32 {
        match &self.array {
            ArrayType::NDArray(arr) => arr.shape()[0].ilog2(),
            ArrayType::OneQ(_) => 1,
            ArrayType::TwoQ(_) => 2,
        }
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        0
    }
    fn control_flow(&self) -> bool {
        false
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        false
    }
}

impl AsMatrix for UnitaryGate {
    type Matrix = Array2<Complex64>;

    fn matrix(&self) -> Self::Matrix {
        match &self.array {
            ArrayType::NDArray(arr) => arr.clone(),
            ArrayType::OneQ(mat) => array!(
                [mat[(0, 0)], mat[(0, 1)]],
                [mat[(1, 0)], mat[(1, 1)]],
            ),
            ArrayType::TwoQ(mat) => array!(
                [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)], mat[(0, 3)]],
                [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)], mat[(1, 3)]],
                [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)], mat[(2, 3)]],
                [mat[(3, 0)], mat[(3, 1)], mat[(3, 2)], mat[(3, 3)]],
            ),
        }
    }
}

#[derive(Debug)]
pub struct UnitaryGateRef<'a> {
    unitary: &'a UnitaryGate,
    params: &'a [NumericParam],
}

impl<'a> Deref for UnitaryGateRef<'a> {
    type Target = UnitaryGate;

    fn deref(&self) -> &Self::Target {
        self.unitary
    }
}

impl<'a> ParameterizedOperation for UnitaryGateRef<'a> {
    type ParamType = &'a NumericParam;
    type Parameters = &'a [NumericParam];

    fn params(&self) -> Self::Parameters {
        self.params
    }
}

impl UnitaryGate {
    pub fn create_py_op(&self, py: Python, label: Option<&str>) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        if let Some(label) = label {
            kwargs.set_item(intern!(py, "label"), label.into_py_any(py)?)?;
        }
        let out_array = match &self.array {
            ArrayType::NDArray(arr) => arr.to_pyarray(py),
            ArrayType::OneQ(arr) => arr.to_pyarray(py),
            ArrayType::TwoQ(arr) => arr.to_pyarray(py),
        };
        kwargs.set_item(intern!(py, "check_input"), false)?;
        kwargs.set_item(intern!(py, "num_qubits"), self.num_qubits())?;
        let gate = UNITARY_GATE
            .get_bound(py)
            .call((out_array,), Some(&kwargs))?;
        Ok(gate.unbind())
    }
}
