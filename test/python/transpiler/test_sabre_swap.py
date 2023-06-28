# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Sabre Swap pass"""

import unittest

import ddt

from qiskit.circuit import IfElseOp
from qiskit.circuit.library import CCXGate, HGate, Measure, SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import SabreSwap, TrivialLayout, CheckMap
from qiskit.transpiler import CouplingMap, PassManager, Target, TranspilerError
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals


def looping_circuit(uphill_swaps=1, additional_local_minimum_gates=0):
    """A circuit that causes SabreSwap to loop infinitely.

    This looks like (using cz gates to show the symmetry, though we actually output cx for testing
    purposes):

    .. parsed-literal::

         q_0: ─■────────────────
               │
         q_1: ─┼──■─────────────
               │  │
         q_2: ─┼──┼──■──────────
               │  │  │
         q_3: ─┼──┼──┼──■───────
               │  │  │  │
         q_4: ─┼──┼──┼──┼─────■─
               │  │  │  │     │
         q_5: ─┼──┼──┼──┼──■──■─
               │  │  │  │  │
         q_6: ─┼──┼──┼──┼──┼────
               │  │  │  │  │
         q_7: ─┼──┼──┼──┼──■──■─
               │  │  │  │     │
         q_8: ─┼──┼──┼──┼─────■─
               │  │  │  │
         q_9: ─┼──┼──┼──■───────
               │  │  │
        q_10: ─┼──┼──■──────────
               │  │
        q_11: ─┼──■─────────────
               │
        q_12: ─■────────────────

    where `uphill_swaps` is the number of qubits separating the inner-most gate (representing how
    many swaps need to be made that all increase the heuristics), and
    `additional_local_minimum_gates` is how many extra gates to add on the outside (these increase
    the size of the region of stability).
    """
    outers = 4 + additional_local_minimum_gates
    n_qubits = 2 * outers + 4 + uphill_swaps
    # This is (most of) the front layer, which is a bunch of outer qubits in the
    # coupling map.
    outer_pairs = [(i, n_qubits - i - 1) for i in range(outers)]
    inner_heuristic_peak = [
        # This gate is completely "inside" all the others in the front layer in
        # terms of the coupling map, so it's the only one that we can in theory
        # make progress towards without making the others worse.
        (outers + 1, outers + 2 + uphill_swaps),
        # These are the only two gates in the extended set, and they both get
        # further apart if you make a swap to bring the above gate closer
        # together, which is the trick that creates the "heuristic hill".
        (outers, outers + 1),
        (outers + 2 + uphill_swaps, outers + 3 + uphill_swaps),
    ]
    qc = QuantumCircuit(n_qubits)
    for pair in outer_pairs + inner_heuristic_peak:
        qc.cx(*pair)
    return qc


@ddt.ddt
class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_trivial_case(self):
        """Test that an already mapped circuit is unchanged.
                  ┌───┐┌───┐
        q_0: ──■──┤ H ├┤ X ├──■──
             ┌─┴─┐└───┘└─┬─┘  │
        q_1: ┤ X ├──■────■────┼──
             └───┘┌─┴─┐       │
        q_2: ──■──┤ X ├───────┼──
             ┌─┴─┐├───┤       │
        q_3: ┤ X ├┤ X ├───────┼──
             └───┘└─┬─┘     ┌─┴─┐
        q_4: ───────■───────┤ X ├
                            └───┘
        """
        coupling = CouplingMap.from_ring(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_trivial_with_target(self):
        """Test that an already mapped circuit is unchanged with target."""
        coupling = CouplingMap.from_ring(5)
        target = Target(num_qubits=5)
        target.add_instruction(SwapGate(), {edge: None for edge in coupling.get_edges()})

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(target, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        """Test lookahead mode's lookahead finds single SWAP gate.
                  ┌───┐
        q_0: ──■──┤ H ├───────────────
             ┌─┴─┐└───┘
        q_1: ┤ X ├──■────■─────────■──
             └───┘┌─┴─┐  │         │
        q_2: ──■──┤ X ├──┼────■────┼──
             ┌─┴─┐└───┘┌─┴─┐┌─┴─┐┌─┴─┐
        q_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├
             └───┘     └───┘└───┘└───┘
        q_4: ─────────────────────────

        """
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        pm = PassManager(SabreSwap(coupling, "lookahead"))
        new_qc = pm.run(qc)

        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_do_not_change_cm(self):
        """Coupling map should not change.
        See https://github.com/Qiskit/qiskit-terra/issues/5675"""
        cm_edges = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
        coupling = CouplingMap(cm_edges)

        passmanager = PassManager(SabreSwap(coupling))
        _ = passmanager.run(QuantumCircuit(coupling.size()))

        self.assertEqual(set(cm_edges), set(coupling.get_edges()))

    def test_do_not_reorder_measurements(self):
        """Test that SabreSwap doesn't reorder measurements to the same classical bit.

        With the particular coupling map used in this test and the 3q ccx gate, the routing would
        invariably the measurements if the classical successors are not accurately tracked.
        Regression test of gh-7950."""
        coupling = CouplingMap([(0, 2), (2, 0), (1, 2), (2, 1)])
        qc = QuantumCircuit(3, 1)
        qc.compose(CCXGate().definition, [0, 1, 2], [])  # Unroll CCX to 2q operations.
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)  # This measure is 50/50 between the Z states.
        qc.measure(1, 0)  # This measure always overwrites with 0.
        passmanager = PassManager(SabreSwap(coupling))
        transpiled = passmanager.run(qc)

        last_h = transpiled.data[-4]
        self.assertIsInstance(last_h.operation, HGate)
        first_measure = transpiled.data[-2]
        second_measure = transpiled.data[-1]
        self.assertIsInstance(first_measure.operation, Measure)
        self.assertIsInstance(second_measure.operation, Measure)
        # Assert that the first measure is on the same qubit that the HGate was applied to, and the
        # second measurement is on a different qubit (though we don't care which exactly - that
        # depends a little on the randomisation of the pass).
        self.assertEqual(last_h.qubits, first_measure.qubits)
        self.assertNotEqual(last_h.qubits, second_measure.qubits)

    # The 'basic' method can't get stuck in the same way.
    @ddt.data("lookahead", "decay")
    def test_no_infinite_loop(self, method):
        """Test that the 'release value' mechanisms allow SabreSwap to make progress even on
        circuits that get stuck in a stable local minimum of the lookahead parameters."""
        qc = looping_circuit(3, 1)
        qc.measure_all()
        coupling_map = CouplingMap.from_line(qc.num_qubits)
        routing_pass = PassManager(SabreSwap(coupling_map, method))

        n_swap_gates = 0

        def leak_number_of_swaps(cls, *args, **kwargs):
            nonlocal n_swap_gates
            n_swap_gates += 1
            if n_swap_gates > 1_000:
                raise Exception("SabreSwap seems to be stuck in a loop")
            # pylint: disable=bad-super-call
            return super(SwapGate, cls).__new__(cls, *args, **kwargs)

        with unittest.mock.patch.object(SwapGate, "__new__", leak_number_of_swaps):
            routed = routing_pass.run(qc)

        routed_ops = routed.count_ops()
        del routed_ops["swap"]
        self.assertEqual(routed_ops, qc.count_ops())
        couplings = {
            tuple(routed.find_bit(bit).index for bit in instruction.qubits)
            for instruction in routed.data
            if len(instruction.qubits) == 2
        }
        # Asserting equality to the empty set gives better errors on failure than asserting that
        # `couplings <= coupling_map`.
        self.assertEqual(couplings - set(coupling_map.get_edges()), set())

        # Assert that the same keys are produced by a simulation - this is a test that the inserted
        # swaps route the qubits correctly.
        if not optionals.HAS_AER:
            return

        from qiskit import Aer

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(routed, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_classical_condition(self):
        """Test that :class:`.SabreSwap` correctly accounts for classical conditions in its
        reckoning on whether a node is resolved or not.  If it is not handled correctly, the second
        gate might not appear in the output.

        Regression test of gh-8040."""
        with self.subTest("1 bit in register"):
            qc = QuantumCircuit(2, 1)
            qc.z(0)
            qc.z(0).c_if(qc.cregs[0], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)
        with self.subTest("multiple registers"):
            cregs = [ClassicalRegister(3), ClassicalRegister(4)]
            qc = QuantumCircuit(QuantumRegister(2, name="q"), *cregs)
            qc.z(0)
            qc.z(0).c_if(cregs[0], 0)
            qc.z(0).c_if(cregs[1], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)

    def test_classical_condition_cargs(self):
        """Test that classical conditions are preserved even if missing from cargs DAGNode field.

        Created from reproduction in https://github.com/Qiskit/qiskit-terra/issues/8675
        """
        with self.subTest("missing measurement"):
            qc = QuantumCircuit(3, 1)
            qc.cx(0, 2).c_if(0, 0)
            qc.measure(1, 0)
            qc.h(2).c_if(0, 0)
            expected = QuantumCircuit(3, 1)
            expected.swap(1, 2)
            expected.cx(0, 1).c_if(0, 0)
            expected.measure(2, 0)
            expected.h(1).c_if(0, 0)
            result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
            self.assertEqual(result, expected)
        with self.subTest("reordered measurement"):
            qc = QuantumCircuit(3, 1)
            qc.cx(0, 1).c_if(0, 0)
            qc.measure(1, 0)
            qc.h(0).c_if(0, 0)
            expected = QuantumCircuit(3, 1)
            expected.cx(0, 1).c_if(0, 0)
            expected.measure(1, 0)
            expected.h(0).c_if(0, 0)
            result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
            self.assertEqual(result, expected)

    def test_conditional_measurement(self):
        """Test that instructions with cargs and conditions are handled correctly."""
        qc = QuantumCircuit(3, 2)
        qc.cx(0, 2).c_if(0, 0)
        qc.measure(2, 0).c_if(1, 0)
        qc.h(2).c_if(0, 0)
        qc.measure(1, 1)
        expected = QuantumCircuit(3, 2)
        expected.swap(1, 2)
        expected.cx(0, 1).c_if(0, 0)
        expected.measure(1, 0).c_if(1, 0)
        expected.h(1).c_if(0, 0)
        expected.measure(2, 1)
        result = SabreSwap(CouplingMap.from_line(3), seed=12345)(qc)
        self.assertEqual(result, expected)

    @ddt.data("basic", "lookahead", "decay")
    def test_deterministic(self, heuristic):
        """Test that the output of the SabreSwap pass is deterministic for a given random seed."""
        width = 40

        # The actual circuit is unimportant, we just need one with lots of scoring degeneracy.
        qc = QuantumCircuit(width)
        for i in range(width // 2):
            qc.cx(i, i + (width // 2))
        for i in range(0, width, 2):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)

        coupling = CouplingMap.from_line(width)
        pass_0 = SabreSwap(coupling, heuristic, seed=0, trials=1)
        pass_1 = SabreSwap(coupling, heuristic, seed=1, trials=1)
        dag_0 = pass_0.run(dag)
        dag_1 = pass_1.run(dag)

        # This deliberately avoids using a topological order, because that introduces an opportunity
        # for the re-ordering to sort the swaps back into a canonical order.
        def normalize_nodes(dag):
            return [(node.op.name, node.qargs, node.cargs) for node in dag.op_nodes()]

        # A sanity check for the test - if unequal seeds don't produce different outputs for this
        # degenerate circuit, then the test probably needs fixing (or Sabre is ignoring the seed).
        self.assertNotEqual(normalize_nodes(dag_0), normalize_nodes(dag_1))

        # Check that a re-run with the same seed produces the same circuit in the exact same order.
        self.assertEqual(normalize_nodes(dag_0), normalize_nodes(pass_0.run(dag)))

    def test_rejects_too_many_qubits(self):
        """Test that a sensible Python-space error message is emitted if the DAG has an incorrect
        number of qubits."""
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(5, "q"))
        with self.assertRaisesRegex(TranspilerError, "More qubits in the circuit"):
            pass_(qc)

    def test_rejects_too_few_qubits(self):
        """Test that a sensible Python-space error message is emitted if the DAG has an incorrect
        number of qubits."""
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(3, "q"))
        with self.assertRaisesRegex(TranspilerError, "Fewer qubits in the circuit"):
            pass_(qc)


class TestSabreSwapControlFlow(QiskitTestCase):
    """Tests for control flow in sabre swap."""

    def test_pre_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[2]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[2]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[2]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[2]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[2]])
        efalse_body.x(1)
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[2]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else_route_post_x(self):
        """test swap with if else controlflow construct; pre-cx and post x"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[0]])
        qc.x(1)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        new_order = [0, 2, 1, 3, 4]
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        efalse_body.x(1)
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[0]])
        expected.x(2)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_post_if_else_route(self):
        """test swap with if else controlflow construct; post cx"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.barrier(qreg)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.cx(0, 2)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        efalse_body.x(1)
        expected.barrier(qreg)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[0]])
        expected.barrier(qreg)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else2(self):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(0)
        false_body = QuantumCircuit(qreg, creg[[0]])
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[0]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[0]], creg[[0]])
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[0]], creg[[0]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_intra_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.swap(0, 1)
        etrue_body.cx(1, 2)
        etrue_body.swap(1, 2)
        etrue_body.swap(3, 4)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.swap(0, 1)
        efalse_body.swap(1, 2)
        efalse_body.swap(3, 4)
        efalse_body.cx(2, 3)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        new_order = [1, 2, 0, 4, 3]
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_nested_inner_cnot(self):
        """test swap in nested if else controlflow construct; swap in inner"""
        seed = 1
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(0)

        for_body = QuantumCircuit(qreg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, [])

        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.x(0)

        efor_body = QuantumCircuit(qreg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg, [])

        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg)

        self.assertEqual(dag_to_circuit(cdag), expected)


if __name__ == "__main__":
    unittest.main()
