# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generic fake BackendV2 class"""

from __future__ import annotations
import statistics
import warnings

from collections.abc import Iterable
import numpy as np

from qiskit import pulse
from qiskit.circuit import Measure, Parameter, Delay, Reset, QuantumCircuit
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    BreakLoopOp,
    ContinueLoopOp,
)
from qiskit.circuit.library import XGate, RZGate, SXGate, CXGate, ECRGate, IGate
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
    PulseDefaults,
    Command,
)
from qiskit.pulse import (
    InstructionScheduleMap,
    AcquireChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
)
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals


class FakeGeneric(BackendV2):
    """
    Configurable fake BackendV2 generator for unit testing purposes. This class will
    generate a fake backend from a combination of random defaults
    (with a fixable `seed`) and properties and configuration driven from a series of
    optional input arguments.

    Arguments:
        num_qubits (int): Minimum number of qubits in the fake backend.
                        The final number of qubits will be updated to be coherent
                        with the specified coupling map/ coupling map type.

        coupling_map (list[tuple[int]]): Backend's coupling map as a list of tuples.
                     Example: [(1, 2), (2, 3), (3, 4), (4, 5)]. If None is passed then the
                     coupling map will be generated randomly, in
                     accordance with the coupling_map_type argument.

        coupling_map_type (str): Type of coupling map to be generated. If coupling map
                    is passed, then this option will be overriden. Valid types of coupling
                    map: 'grid' (default), 'heavy_hex'.

        basis_gates (list[str]): Basis gates supported by the backend as list of strings.
                                If None is passed, this parameter will be set internally
                                to ['cx', 'id', 'rz', 'sx', 'x'].

        dynamic (bool): If True, enable dynamic circuits on this backend.
                        Defaults to False.

        bidirectional_cmap (bool): If True, enable bi-directional coupling map.
                                    Defaults to False.
        replace_cx_with_ecr (bool): If True, replace every occurrence of 'cx' with 'ecr'.
                                    Defaults to True.

        enable_reset (bool): If True, enable reset on the backend.
                             Defaults to True.

        add_cal_entries (bool): If True, pulse calibration defaults will be added to
                            the target. Defaults to False.
        empty_cal_gates (list[str): List of gates defined a strings which should have
                                        calibration entries with an empty schedule.

        instruction_schedule_map (InstructionScheduleMap): An optional instruction schedule map
                                               to replace the randomly generated pulse defaults.

        dt (float): The system time resolution of input signals in seconds.
                    Default is 0.2222ns

        seed (int): Optional seed for error and duration value generation.

    Returns:
            None

    Raises:
            QiskitError: If argument basis_gates has a gate which is not a valid basis gate.
    """

    def __init__(
        self,
        num_qubits: int = None,
        *,
        coupling_map: list[tuple[int]] | None = None,
        coupling_map_type: str = "grid",
        basis_gates: list[str] | None = None,
        dynamic: bool = False,
        bidirectional_cmap: bool = False,
        replace_cx_with_ecr: bool = True,
        enable_reset: bool = True,
        add_cal_entries: bool = False,
        empty_cal_gates: list[str] | None = None,
        instruction_schedule_map: InstructionScheduleMap = None,
        dt: float = 0.222e-9,
        seed: int = 42,
    ):

        super().__init__(
            provider=None,
            name="fake_generic",
            description=f"This is a fake device with at least {num_qubits} "
            f"and generic settings. It has been generated right now!",
            backend_version="",
        )
        self.sim = None

        self._rng = np.random.default_rng(seed=seed)

        self._set_basis_gates(basis_gates, replace_cx_with_ecr)
        self._set_coupling_map(num_qubits, coupling_map, coupling_map_type, bidirectional_cmap)

        self._target = Target(
            description="Fake Generic Backend",
            num_qubits=self._num_qubits,
            dt=dt,
            qubit_properties=[
                QubitProperties(
                    t1=self._rng.uniform(100e-6, 200e-6),
                    t2=self._rng.uniform(100e-6, 200e-6),
                    frequency=self._rng.uniform(5e9, 5.5e9),
                )
                for _ in range(self._num_qubits)
            ],
            concurrent_measurements=[list(range(self._num_qubits))],
        )

        self._add_gate_instructions_to_target(dynamic, enable_reset)

        if add_cal_entries:
            self._add_calibration_defaults_to_target(instruction_schedule_map, empty_cal_gates)
            self._build_default_channels()

    @property
    def target(self) -> Target:
        """Return the target.

        Returns:
            Target: The backend's target.
        """
        return self._target

    @property
    def coupling_map(self) -> CouplingMap:
        """Return the coupling map.

        Returns:
            CouplingMap: The backend's coupling map.
        """
        return self._coupling_map

    @property
    def basis_gates(self) -> list[str]:
        """Return the basis gates.

        Returns:
            list[str]: The list of accepted basis gates
        """
        return self._basis_gates

    @property
    def max_circuits(self) -> None:
        """Placeholder to be able to query max_circuits. Set to None.

        Returns: None
        """
        return None

    @property
    def meas_map(self) -> list[list[int]]:
        """Return the measurement map in a V1-compatible way.

        Returns:
            list[list[int]]: The list of concurrent measurements/measurement map
        """
        return self._target.concurrent_measurements

    def drive_channel(self, qubit: int) -> DriveChannel | None:
        """Return the drive channel for the given qubit.

        Returns:
            DriveChannel: The Qubit drive channel
        """
        drive_channels_map = getattr(self, "channels_map", {}).get("drive", {})
        qubits = (qubit,)
        if qubits in drive_channels_map:
            return drive_channels_map[qubits][0]
        return None

    def measure_channel(self, qubit: int) -> MeasureChannel | None:
        """Return the measure stimulus channel for the given qubit.

        Returns:
            MeasureChannel: The Qubit measurement stimulus line
        """
        measure_channels_map = getattr(self, "channels_map", {}).get("measure", {})
        qubits = (qubit,)
        if qubits in measure_channels_map:
            return measure_channels_map[qubits][0]
        return None

    def acquire_channel(self, qubit: int) -> AcquireChannel | None:
        """Return the acquisition channel for the given qubit.

        Returns:
            AcquireChannel: The Qubit measurement acquisition line.
        """
        acquire_channels_map = getattr(self, "channels_map", {}).get("acquire", {})
        qubits = (qubit,)
        if qubits in acquire_channels_map:
            return acquire_channels_map[qubits][0]
        return None

    def control_channel(self, qubits: Iterable[int]) -> list[ControlChannel]:
        """Return the secondary drive channel for the given qubit

        This is typically utilized for controlling multiqubit interactions.
        This channel is derived from other channels.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            list[ControlChannel]: The multi qubit control line.
        """
        control_channels_map = getattr(self, "channels_map", {}).get("control", {})
        qubits = tuple(qubits)
        if qubits in control_channels_map:
            return control_channels_map[qubits]
        return []

    def run(self, run_input, **options):
        """Run on the fake backend using a simulator.

        This method runs circuit jobs (an individual or a list of QuantumCircuit
        ) and pulse jobs (an individual or a list of Schedule or ScheduleBlock)
        using BasicAer or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using AerSimulator with
        noise model of the fake backend. Otherwise, jobs will be run using
        BasicAer simulator without noise.

        Currently noisy simulation of a pulse job is not supported yet in
        FakeBackendV2.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit-aer is not installed.
        """
        circuits = run_input
        pulse_job = None
        if isinstance(circuits, (pulse.Schedule, pulse.ScheduleBlock)):
            pulse_job = True
        elif isinstance(circuits, QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all(isinstance(x, (pulse.Schedule, pulse.ScheduleBlock)) for x in circuits):
                    pulse_job = True
                elif all(isinstance(x, QuantumCircuit) for x in circuits):
                    pulse_job = False
        if pulse_job is None:  # submitted job is invalid
            raise QiskitError(
                "Invalid input object %s, must be either a "
                "QuantumCircuit, Schedule, or a list of either" % circuits
            )
        if pulse_job:  # pulse job
            raise QiskitError("Pulse simulation is currently not supported for V2 fake backends.")
        # circuit job
        if not _optionals.HAS_AER:
            warnings.warn("Aer not found using BasicAer and no noise", RuntimeWarning)
        if self.sim is None:
            self._setup_sim()
        self.sim._options = self._options
        job = self.sim.run(circuits, **options)
        return job

    def _setup_sim(self) -> None:
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            self.sim = AerSimulator()
            noise_model = NoiseModel.from_backend(self)
            self.sim.set_options(noise_model=noise_model)
            # Update fake backend default too to avoid overwriting
            # it when run() is called
            self.set_options(noise_model=noise_model)

        else:
            self.sim = BasicAer.get_backend("qasm_simulator")

    @classmethod
    def _default_options(cls) -> Options:
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: An options object with
                default values set
        """
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator

            return AerSimulator._default_options()
        else:
            return BasicAer.get_backend("qasm_simulator")._default_options()

    def _set_basis_gates(self, basis_gates: list[str], replace_cx_with_ecr: bool) -> None:

        default_gates = ["cx", "id", "rz", "sx", "x"]

        if basis_gates is not None:
            self._basis_gates = basis_gates
        else:
            self._basis_gates = default_gates

        if replace_cx_with_ecr:
            self._basis_gates = [gate.replace("cx", "ecr") for gate in self._basis_gates]

        if "delay" not in self._basis_gates:
            self._basis_gates.append("delay")
        if "measure" not in self._basis_gates:
            self._basis_gates.append("measure")

    def _set_coupling_map(
        self,
        num_qubits: int | None,
        coupling_map: list[tuple[int]],
        coupling_map_type: str,
        bidirectional_cmap: bool,
    ) -> None:

        if not coupling_map:
            if not num_qubits:
                raise QiskitError(
                    "Please provide either ``num_qubits`` or ``coupling_map`` "
                    "to generate a new fake backend."
                )

            if coupling_map_type == "heavy_hex":
                distance = self._get_cmap_args(coupling_map_type, num_qubits)
                self._coupling_map = CouplingMap().from_heavy_hex(
                    distance=distance, bidirectional=bidirectional_cmap
                )
            elif coupling_map_type == "grid":
                num_rows, num_columns = self._get_cmap_args(coupling_map_type, num_qubits)
                self._coupling_map = CouplingMap().from_grid(
                    num_rows=num_rows, num_columns=num_columns, bidirectional=bidirectional_cmap
                )
            else:
                raise QiskitError("Provided coupling map type not recognized")
        else:
            self._coupling_map = CouplingMap(coupling_map)

        self._num_qubits = self._coupling_map.size()

    def _get_cmap_args(self, coupling_map_type: str, num_qubits: int) -> int | tuple[int] | None:
        if coupling_map_type == "heavy_hex":
            for d in range(3, 20, 2):
                # The description of the formula: 5*d**2 - 2*d -1 is explained in
                # https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.011022 Page 011022-4
                n = (5 * (d**2) - (2 * d) - 1) / 2
                if n >= num_qubits:
                    return d

        elif coupling_map_type == "grid":
            factors = [x for x in range(2, num_qubits + 1) if num_qubits % x == 0]
            first_factor = statistics.median_high(factors)
            second_factor = int(num_qubits / first_factor)
            return (first_factor, second_factor)

        return None

    def _add_gate_instructions_to_target(self, dynamic: bool, enable_reset: bool) -> None:

        instruction_dict = self._get_default_instruction_dict()

        for gate in self._basis_gates:
            try:
                self._target.add_instruction(*instruction_dict[gate])
            except Exception as exc:
                raise QiskitError(f"{gate} is not a valid basis gate") from exc

        if dynamic:
            self._target.add_instruction(IfElseOp, name="if_else")
            self._target.add_instruction(WhileLoopOp, name="while_loop")
            self._target.add_instruction(ForLoopOp, name="for_loop")
            self._target.add_instruction(SwitchCaseOp, name="switch_case")
            self._target.add_instruction(BreakLoopOp, name="break")
            self._target.add_instruction(ContinueLoopOp, name="continue")

        if enable_reset:
            self._target.add_instruction(
                Reset(), {(qubit_idx,): None for qubit_idx in range(self._num_qubits)}
            )

    def _get_default_instruction_dict(self) -> dict[str, tuple]:

        instruction_dict = {
            "ecr": (
                ECRGate(),
                {
                    edge: InstructionProperties(
                        error=self._rng.uniform(1e-5, 5e-3),
                        duration=self._rng.uniform(1e-8, 9e-7),
                    )
                    for edge in self.coupling_map
                },
            ),
            "cx": (
                CXGate(),
                {
                    edge: InstructionProperties(
                        error=self._rng.uniform(1e-3, 5e-2),
                        duration=self._rng.uniform(2e-7, 8e-7),
                    )
                    for edge in self.coupling_map
                },
            ),
            "id": (
                IGate(),
                {
                    (qubit_idx,): InstructionProperties(error=0.0, duration=0.0)
                    for qubit_idx in range(self._num_qubits)
                },
            ),
            "rz": (
                RZGate(Parameter("theta")),
                {
                    (qubit_idx,): InstructionProperties(error=0.0, duration=0.0)
                    for qubit_idx in range(self._num_qubits)
                },
            ),
            "x": (
                XGate(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self._rng.uniform(1e-6, 1e-4),
                        duration=self._rng.uniform(2e-8, 4e-8),
                    )
                    for qubit_idx in range(self._num_qubits)
                },
            ),
            "sx": (
                SXGate(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self._rng.uniform(1e-6, 1e-4),
                        duration=self._rng.uniform(1e-8, 2e-8),
                    )
                    for qubit_idx in range(self._num_qubits)
                },
            ),
            "measure": (
                Measure(),
                {
                    (qubit_idx,): InstructionProperties(
                        error=self._rng.uniform(1e-3, 1e-1),
                        duration=self._rng.uniform(1e-8, 9e-7),
                    )
                    for qubit_idx in range(self._num_qubits)
                },
            ),
            "delay": (
                Delay(Parameter("Time")),
                {(qubit_idx,): None for qubit_idx in range(self._num_qubits)},
            ),
        }
        return instruction_dict

    def _add_calibration_defaults_to_target(
        self, instruction_schedule_map: InstructionScheduleMap, empty_cal_gates: list[str] | None
    ) -> None:

        if not instruction_schedule_map:
            defaults = self._build_calibration_defaults(empty_cal_gates)
            inst_map = defaults.instruction_schedule_map
        else:
            inst_map = instruction_schedule_map

        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qpbj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in self._target:
                    if inst == "measure":
                        for qubit in qargs:
                            self._target[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in self._target[inst] and inst != "delay":
                        self._target[inst][qargs].calibration = calibration_entry

    def _build_calibration_defaults(self, empty_cal_gates: list[str] | None) -> PulseDefaults:
        """Build calibration defaults."""

        if empty_cal_gates is None:
            empty_cal_gates = []

        measure_command_sequence = [
            PulseQobjInstruction(
                name="acquire",
                duration=1792,
                t0=0,
                qubits=list(range(self.num_qubits)),
                memory_slot=list(range(self.num_qubits)),
            )
        ]

        measure_command_sequence += [
            PulseQobjInstruction(name="pulse_1", ch=f"m{i}", duration=1792, t0=0)
            for i in range(self.num_qubits)
        ]

        measure_command = Command(
            name="measure",
            qubits=list(range(self.num_qubits)),
            sequence=measure_command_sequence,
        )

        cmd_def = [measure_command]

        for gate in self._basis_gates:
            for i in range(self.num_qubits):
                sequence = []
                if gate not in empty_cal_gates:
                    sequence = [
                        PulseQobjInstruction(name="fc", ch=f"d{i}", t0=0, phase="-P0"),
                        PulseQobjInstruction(name="pulse_3", ch=f"d{i}", t0=0),
                    ]
                cmd_def.append(
                    Command(
                        name=gate,
                        qubits=[i],
                        sequence=sequence,
                    )
                )

        for qubit1, qubit2 in self.coupling_map:
            sequence = [
                PulseQobjInstruction(name="pulse_1", ch=f"d{qubit1}", t0=0),
                PulseQobjInstruction(name="pulse_2", ch=f"u{qubit1}", t0=10),
                PulseQobjInstruction(name="pulse_1", ch=f"d{qubit2}", t0=20),
                PulseQobjInstruction(name="fc", ch=f"d{qubit2}", t0=20, phase=2.1),
            ]

            if "cx" in self._basis_gates:
                if "cx" in empty_cal_gates:
                    sequence = []
                cmd_def += [
                    Command(
                        name="cx",
                        qubits=[qubit1, qubit2],
                        sequence=sequence,
                    )
                ]
            if "ecr" in self._basis_gates:
                if "ecr" in empty_cal_gates:
                    sequence = []
                cmd_def += [
                    Command(
                        name="ecr",
                        qubits=[qubit1, qubit2],
                        sequence=sequence,
                    )
                ]

        qubit_freq_est = np.random.normal(4.8, scale=0.01, size=self.num_qubits).tolist()
        meas_freq_est = np.linspace(6.4, 6.6, self.num_qubits).tolist()
        pulse_library = [
            PulseLibraryItem(name="pulse_1", samples=[[0.0, 0.0], [0.0, 0.1]]),
            PulseLibraryItem(name="pulse_2", samples=[[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]),
            PulseLibraryItem(
                name="pulse_3", samples=[[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [0.5, 0.0]]
            ),
        ]

        return PulseDefaults(
            qubit_freq_est=qubit_freq_est,
            meas_freq_est=meas_freq_est,
            buffer=0,
            pulse_library=pulse_library,
            cmd_def=cmd_def,
        )

    def _build_default_channels(self) -> None:
        """Create default channel map and set "channels_map" attribute"""
        channels_map = {
            "acquire": {(i,): [pulse.AcquireChannel(i)] for i in range(self.num_qubits)},
            "drive": {(i,): [pulse.DriveChannel(i)] for i in range(self.num_qubits)},
            "measure": {(i,): [pulse.MeasureChannel(i)] for i in range(self.num_qubits)},
            "control": {
                (edge): [pulse.ControlChannel(i)] for i, edge in enumerate(self.coupling_map)
            },
        }
        setattr(self, "channels_map", channels_map)
