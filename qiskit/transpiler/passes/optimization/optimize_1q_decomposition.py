# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimize chains of single-qubit gates using Euler 1q decomposer"""

import logging
import math

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
    UGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RXGate,
    RYGate,
    RZGate,
    RGate,
    SXGate,
    XGate,
)
from qiskit.circuit import Qubit
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


logger = logging.getLogger(__name__)

# When expanding the list of supported gates this needs to updated in
# lockstep with the VALID_BASES constant in src/euler_one_qubit_decomposer.rs
# and the global variables in one_qubit_decompose.py
NAME_MAP = {
    "u": UGate,
    "u1": U1Gate,
    "u2": U2Gate,
    "u3": U3Gate,
    "p": PhaseGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "r": RGate,
    "sx": SXGate,
    "x": XGate,
}


class Optimize1qGatesDecomposition(TransformationPass):
    """Optimize chains of single-qubit gates by combining them into a single gate.

    The decision to replace the original chain with a new re-synthesis depends on:
     - whether the original chain was out of basis: replace
     - whether the original chain was in basis but re-synthesis is lower error: replace
     - whether the original chain contains a pulse gate: do not replace
     - whether the original chain amounts to identity: replace with null

     Error is computed as a multiplication of the errors of individual gates on that qubit.
    """

    def __init__(self, basis=None, target=None):
        """Optimize1qGatesDecomposition initializer.

        Args:
            basis (list[str]): Basis gates to consider, e.g. `['u3', 'cx']`. For the effects
                of this pass, the basis is the set intersection between the `basis` parameter
                and the Euler basis. Ignored if ``target`` is also specified.
            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation
                target. When specified, any argument specified for ``basis_gates`` is ignored.
        """
        super().__init__()

        self._basis_gates = basis
        self._target = target
        self._global_decomposers = []
        self._local_decomposers_cache = {}

        if basis:
            self._global_decomposers = _possible_decomposers(set(basis))
        elif target is None:
            self._global_decomposers = _possible_decomposers(None)
            self._basis_gates = None

        self.error_map = self._build_error_map()

    def _build_error_map(self):
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if self._target is not None and self._target.num_qubits is not None:
            error_map = euler_one_qubit_decomposer.OneQubitGateErrorMap(self._target.num_qubits)
            for qubit in range(self._target.num_qubits):
                gate_error = {}
                for gate, gate_props in self._target.items():
                    if gate_props is not None:
                        props = gate_props.get((qubit,), None)
                        if props is not None and props.error is not None:
                            gate_error[gate] = props.error
                error_map.add_qubit(gate_error)
            return error_map
        else:
            return None

    def _get_decomposer(self, qubit=None):
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if self._target is not None and self._target.num_qubits is not None:
            if qubit is not None:
                qubits_tuple = (qubit,)
            else:
                qubits_tuple = None
            if qubits_tuple in self._local_decomposers_cache:
                decomposers = self._local_decomposers_cache[qubits_tuple]
            else:
                available_1q_basis = set(self._target.operation_names_for_qargs(qubits_tuple))
                decomposers = _possible_decomposers(available_1q_basis)
        else:
            decomposers = self._global_decomposers
        return decomposers

    def _resynthesize_run(self, matrix, qubit=None):
        """
        Re-synthesizes one 2x2 `matrix`, typically extracted via `dag.collect_1q_runs`.

        Returns the newly synthesized circuit in the indicated basis, or None
        if no synthesis routine applied.

        When multiple synthesis options are available, it prefers the one with the lowest
        error when the circuit is applied to `qubit`.
        """
        decomposers = self._get_decomposer(qubit)

        best_synth_circuit = euler_one_qubit_decomposer.unitary_to_gate_sequence(
            matrix,
            decomposers,
            qubit,
            self.error_map,
        )
        return best_synth_circuit

    def _gate_sequence_to_dag(self, best_synth_circuit):
        qubits = (Qubit(),)
        out_dag = DAGCircuit()
        out_dag.add_qubits(qubits)
        out_dag.global_phase = best_synth_circuit.global_phase

        for gate_name, angles in best_synth_circuit:
            op = CircuitInstruction(gate_name, qubits=qubits, params=angles)
            out_dag.apply_operation_back(op.operation, qubits, check=False)
        return out_dag

    def _substitution_checks(
        self, dag, old_run, new_circ, basis, qubit, old_error=None, new_error=None
    ):
        """
        Returns `True` when it is recommended to replace `old_run` with `new_circ` over `basis`.
        """
        if new_circ is None:
            return False

        # do we even have calibrations?
        has_cals_p = dag.calibrations is not None and len(dag.calibrations) > 0
        # does this run have uncalibrated gates?
        uncalibrated_p = not has_cals_p or any(not dag.has_calibration_for(g) for g in old_run)
        # does this run have gates not in the image of ._decomposers _and_ uncalibrated?
        if basis is not None:
            uncalibrated_and_not_basis_p = any(
                g.name not in basis and (not has_cals_p or not dag.has_calibration_for(g))
                for g in old_run
            )
        else:
            # If no basis is specified then we're always in the basis
            uncalibrated_and_not_basis_p = False

        # if we're outside of the basis set, we're obligated to logically decompose.
        # if we're outside of the set of gates for which we have physical definitions,
        #    then we _try_ to decompose, using the results if we see improvement.
        if not uncalibrated_and_not_basis_p:
            if new_error is None:
                new_error = self._error(new_circ, qubit)
            if old_error is None:
                old_error = self._error(old_run, qubit)
        else:
            new_error = 0.0
            old_error = 0.0

        return (
            uncalibrated_and_not_basis_p
            or (uncalibrated_p and new_error < old_error)
            or (math.isclose(new_error[0], 0) and not math.isclose(old_error[0], 0))
        )

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the Optimize1qGatesDecomposition pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        runs = []
        qubits = []
        bases = []
        for run in dag.collect_1q_runs():
            qubit = dag.find_bit(run[0].qargs[0]).index
            runs.append(run)
            qubits.append(qubit)
            bases.append(self._get_decomposer(qubit))
        best_sequences = euler_one_qubit_decomposer.optimize_1q_gates_decomposition(
            runs, qubits, bases, simplify=True, error_map=self.error_map
        )
        for index, best_circuit_sequence in enumerate(best_sequences):
            run = runs[index]
            qubit = qubits[index]
            if self._target is None:
                basis = self._basis_gates
            else:
                basis = self._target.operation_names_for_qargs((qubit,))
            if best_circuit_sequence is not None:
                (old_error, new_error, best_circuit_sequence) = best_circuit_sequence
                if self._substitution_checks(
                    dag,
                    run,
                    best_circuit_sequence,
                    basis,
                    qubit,
                    old_error=old_error,
                    new_error=new_error,
                ):
                    first_node_id = run[0]._node_id
                    qubit = run[0].qargs
                    for gate, angles in best_circuit_sequence:
                        op = CircuitInstruction(gate, qubits=qubit, params=angles)
                        node = DAGOpNode.from_instruction(op, dag=dag)
                        node._node_id = dag._multi_graph.add_node(node)
                        dag._increment_op(gate.name)
                        dag._multi_graph.insert_node_on_in_edges(node._node_id, first_node_id)
                    dag.global_phase += best_circuit_sequence.global_phase
                    # Delete the other nodes in the run
                    for current_node in run:
                        dag.remove_op_node(current_node)

        return dag

    def _error(self, circuit, qubit):
        """
        Calculate a rough error for a `circuit` that runs on a specific
        `qubit` of `target` (`circuit` can either be an OneQubitGateSequence
        from Rust or a list of DAGOPNodes).

        Use basis errors from target if available, otherwise use length
        of circuit as a weak proxy for error.
        """
        if isinstance(circuit, euler_one_qubit_decomposer.OneQubitGateSequence):
            return euler_one_qubit_decomposer.compute_error_one_qubit_sequence(
                circuit, qubit, self.error_map
            )
        else:
            return euler_one_qubit_decomposer.compute_error_list(circuit, qubit, self.error_map)


def _possible_decomposers(basis_set):
    decomposers = []
    if basis_set is None:
        decomposers = list(one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES)
    else:
        euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
        for euler_basis_name, gates in euler_basis_gates.items():
            if set(gates).issubset(basis_set):
                decomposers.append(euler_basis_name)
        # If both U3 and U321 are in decomposer list only run U321 because
        # in worst case it will produce the same U3 output, but in the general
        # case it will use U2 and U1 which will be more efficient.
        if "U3" in decomposers and "U321" in decomposers:
            decomposers.remove("U3")
        # If both ZSX and ZSXX are in decomposer list only run ZSXX because
        # in the worst case it will produce the same output, but in the general
        # case it will simplify X rotations to use X gate instead of multiple
        # SX gates and be more efficient. Running multiple decomposers in this
        # case will just waste time.
        if "ZSX" in decomposers and "ZSXX" in decomposers:
            decomposers.remove("ZSX")
    return decomposers
