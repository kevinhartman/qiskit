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
import warnings

from enum import Enum
from collections.abc import Iterable
import numpy as np

from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    BreakLoopOp,
    ContinueLoopOp,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
    PulseDefaults,
    Command,
)
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals


def _get_noise_nisq_2024(inst: str) -> tuple:
    default_dict = {
        "cx": (1e-8, 9e-7, 1e-5, 5e-3),
        "ecr": (1e-8, 9e-7, 1e-5, 5e-3),
        "cz": (1e-8, 9e-7, 1e-5, 5e-3),
        "id": (3e-8, 4e-8, 9e-5, 1e-4),
        "rz": (0.0, 0.0),
        "sx": (1e-8, 9e-7, 1e-5, 5e-3),
        "x": (1e-8, 9e-7, 1e-5, 5e-3),
        "measure": (1e-8, 9e-7, 1e-5, 5e-3),
        "delay": (None, None),
        "reset": (None, None),
    }
    return default_dict.get(inst, (1e-8, 9e-7, 1e-5, 5e-3))


class _NoiseDefaults(Enum):
    NISQ_2024_1 = (_get_noise_nisq_2024,)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)


def _get_sequence_2024(
    inst: str, num_qubits: int, qargs: tuple[int], pulse_library: list[PulseLibraryItem]
) -> list[PulseQobjInstruction]:
    if len(pulse_library) < 3:
        raise ValueError(
            f"This calibration sequence requires at least 3 pulses in "
            f"the pulse library, found {len(pulse_library)}."
        )
    # Note that the calibration pulses are different for
    # 1q gates vs 2q gates vs measurement instructions.
    if inst == "measure":
        sequence = [
            PulseQobjInstruction(
                name="acquire",
                duration=1792,
                t0=0,
                qubits=list(range(num_qubits)),
                memory_slot=list(range(num_qubits)),
            )
        ] + [PulseQobjInstruction(name=pulse_library[1], ch=f"m{i}", t0=0) for i in qargs]
        return sequence
    if num_qubits == 1:
        return [
            PulseQobjInstruction(name="fc", ch=f"u{qargs}", t0=0, phase="-P0"),
            PulseQobjInstruction(name=pulse_library[0].name, ch=f"d{qargs}", t0=0),
        ]
    return [
        PulseQobjInstruction(name=pulse_library[1].name, ch=f"d{qargs[0]}", t0=0),
        PulseQobjInstruction(name=pulse_library[2].name, ch=f"u{qargs[0]}", t0=0),
        PulseQobjInstruction(name=pulse_library[1].name, ch=f"d{qargs[1]}", t0=0),
        PulseQobjInstruction(name="fc", ch=f"d{qargs[1]}", t0=0, phase=2.1),
    ]


def _get_pulse_config_2024(num_qubits: int) -> dict:
    # The number of samples determines the pulse durations of the corresponding
    # instructions. This function defines pulses with durations in multiples of
    # 16 for consistency with the pulse granularity of real IBM devices, but
    # keeps the number smaller than what would be realistic for
    # manageability. If needed, more realistic durations could be added in the
    # future (order of 160dt for 1q gates, 1760dt for 2q gates and measure).
    samples_1 = np.linspace(0, 1.0, 16, dtype=np.complex128)  # 16dt
    samples_2 = np.linspace(0, 1.0, 32, dtype=np.complex128)  # 32dt
    samples_3 = np.linspace(0, 1.0, 64, dtype=np.complex128)  # 64dt
    pulse_library = [
        PulseLibraryItem(name="pulse_1", samples=samples_1),
        PulseLibraryItem(name="pulse_2", samples=samples_2),
        PulseLibraryItem(name="pulse_3", samples=samples_3),
    ]
    qubit_freq_est = np.random.normal(4.8, scale=0.01, size=num_qubits).tolist()
    meas_freq_est = np.linspace(6.4, 6.6, num_qubits).tolist()
    config = {
        "pulse_library": pulse_library,
        "qubit_freq_est": qubit_freq_est,
        "meas_freq_est": meas_freq_est,
    }
    return config


class _PulseCalibrationDefaults(Enum):
    PULSE_CONFIG_2024_1 = (_get_pulse_config_2024,)
    SEQUENCE_2024_1 = (_get_sequence_2024,)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)


class _BasisDefaults(Enum):
    CX = ["cx", "id", "rz", "sx", "x"]
    CZ = ["cz", "id", "rz", "sx", "x"]
    ECR = ["ecr", "id", "rz", "sx", "x"]

    def __get__(self, *args, **kwargs):
        return self.value


class _QubitDefaults(Enum):
    NISQ_2024_1 = {
        "dt": 0.222e-9,
        "t1": (100e-6, 200e-6),
        "t2": (100e-6, 200e-6),
        "frequency": (5e9, 5.5e9),
    }

    def __get__(self, *args, **kwargs):
        return self.value


class GenericFakeBackend(BackendV2):
    """
    Configurable :class:`~.BackendV2` fake backend. Users can directly configure the number
    of qubits, basis gates, coupling map, ability to run dynamic circuits (control flow
    instructions), pulse calibration sequences and dtm of the backend instance without having
    to manually build a target or deal with backend properties. Qubit, instruction and pulse
    properties are randomly generated from a series of default ranges that can also be configured
    through the ``noise_defaults``, ``calibration_sequence``, ``pulse_config`` and ``qubit_defaults``
    inputs.
    The remainder of the backend properties are generated by randomly sampling from default ranges
    extracted from historical IBM backend data. The seed for this random generation is always fixed
    to ensure the reproducibility of the backend output, and can also be configured by the user.
    """

    # Added backend_name for compatibility with
    # snapshot-based fake backends.
    backend_name = "GenericFakeBackend"

    def __init__(
        self,
        num_qubits: int,
        *,
        basis_gates: list[str] | None = None,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_instructions: bool | InstructionScheduleMap | None = None,
        dtm: float | None = None,
        seed: int = 42,
    ):
        """
        Args:
            num_qubits: Number of qubits that will be used to construct the backend's target.
                Note that, while there is no limit in the size of the target that can be
                constructed, fake backends run on local noisy simulators, and these might
                present limitations in the number of qubits that can be simulated.

            basis_gates: List of basis gate names to be supported by
                the target. These must be part of the standard qiskit circuit library.
                The default set of basis gates is ``["id", "rz", "sx", "x", "cx"]``
                The ``"reset"``,  ``"delay"``, and ``"measure"`` instructions are
                always supported by default, even if not specified via ``basis_gates``.

            coupling_map: Optional coupling map
                for the fake backend. Multiple formats are supported:

                #. :class:`~.CouplingMap` instance
                #. List, must be given as an adjacency matrix, where each entry
                   specifies all directed two-qubit interactions supported by the backend,
                   e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

                If ``coupling_map`` is specified, it must match the number of qubits
                specified in ``num_qubits``. If ``coupling_map`` is not specified,
                a fully connected coupling map will be generated with ``num_qubits``
                qubits.

            control_flow: Flag to enable control flow directives on the target
                (defaults to False).

            calibrate_instructions: Instruction calibration settings, this argument
                supports both boolean and :class:`.InstructionScheduleMap` as
                input types, and is ``None`` by default:

                #. If ``calibrate_instructions==None``, no calibrations will be added to the target.
                #. If ``calibrate_instructions==True``, all gates will be calibrated for all
                    qubits using the default pulse schedules generated internally.
                #. If ``calibrate_instructions==False``, all gates will be "calibrated" for
                    all qubits with an empty pulse schedule.
                #. If an :class:`.InstructionScheduleMap` instance is given, this calibrations
                    present in this instruction schedule map will be appended to the target
                    instead of the default pulse schedules (this allows for custom calibrations).

            dtm: System time resolution of output signals in nanoseconds.
                None by default.

            seed: Optional seed for generation of default values.
        """

        super().__init__(
            provider=None,
            name=f"fake_generic_{num_qubits}q",
            description=f"This is a fake device with {num_qubits} qubits and generic settings.",
            backend_version="",
        )

        self.sim = None
        self._rng = np.random.default_rng(seed=seed)
        self._dtm = dtm
        self._num_qubits = num_qubits
        self._control_flow = control_flow
        self._calibrate_instructions = calibrate_instructions
        self._supported_gates = get_standard_gate_name_mapping()

        self._basis_gates = basis_gates if basis_gates is not None else _BasisDefaults.CX
        for name in ["reset", "delay", "measure"]:
            if name not in self._basis_gates:
                self._basis_gates.append(name)

        if coupling_map is None:
            self._coupling_map = CouplingMap().from_full(num_qubits)
        else:
            if isinstance(coupling_map, CouplingMap):
                self._coupling_map = coupling_map
            else:
                self._coupling_map = CouplingMap(coupling_map)

            if num_qubits != self._coupling_map.size():
                raise QiskitError(
                    f"The number of qubits (got {num_qubits}) must match "
                    f"the size of the provided coupling map (got {coupling_map.size()})."
                )

        self._qubit_defaults = _QubitDefaults.NISQ_2024_1
        self._noise_defaults = _NoiseDefaults.NISQ_2024_1
        self._calibration_sequence = _PulseCalibrationDefaults.SEQUENCE_2024_1
        self._pulse_config = _PulseCalibrationDefaults.PULSE_CONFIG_2024_1

        self._build_generic_target()
        self._build_default_channels()

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            The output signal timestep in seconds.
        """
        if self._dtm is not None:
            # converting `dtm` from nanoseconds to seconds
            return self._dtm * 1e-9
        else:
            return None

    @property
    def meas_map(self) -> list[list[int]]:
        return self._target.concurrent_measurements

    def _build_generic_target(self):
        """
        This method generates a :class:`~.Target` instance with
        default qubit, instruction and calibration properties.
        """
        # The qubit properties are currently not configurable
        self._target = Target(
            description=f"Generic Target with {self._num_qubits} qubits",
            num_qubits=self._num_qubits,
            dt=self._qubit_defaults["dt"],
            qubit_properties=[
                QubitProperties(
                    t1=self._rng.uniform(
                        self._qubit_defaults["t1"][0], self._qubit_defaults["t1"][1]
                    ),
                    t2=self._rng.uniform(
                        self._qubit_defaults["t2"][0], self._qubit_defaults["t2"][1]
                    ),
                    frequency=self._qubit_defaults["frequency"],
                )
                for _ in range(self._num_qubits)
            ],
            concurrent_measurements=[list(range(self._num_qubits))],
        )

        # Iterate over gates, generate noise params from defaults,
        # and add instructions to target.
        for name in self._basis_gates:
            if name not in self._supported_gates:
                raise QiskitError(
                    f"Provided basis gate {name} is not an instruction "
                    f"in the standard qiskit circuit library."
                )
            gate = self._supported_gates[name]
            noise_params = self._noise_defaults(name)
            self._add_noisy_instruction_to_target(gate, noise_params)

        if self._control_flow:
            self._target.add_instruction(IfElseOp, name="if_else")
            self._target.add_instruction(WhileLoopOp, name="while_loop")
            self._target.add_instruction(ForLoopOp, name="for_loop")
            self._target.add_instruction(SwitchCaseOp, name="switch_case")
            self._target.add_instruction(BreakLoopOp, name="break")
            self._target.add_instruction(ContinueLoopOp, name="continue")

        # Generate block of calibration defaults and add to target.
        # Note: this could be improved if we could generate and add
        # calibration defaults per-gate, and not as a block, but we
        # currently rely on the `PulseDefaults` class
        # from qiskit.providers.models
        if self._calibrate_instructions is not None:
            if isinstance(self._calibrate_instructions, InstructionScheduleMap):
                inst_map = self._calibrate_instructions
            else:
                defaults = self._generate_calibration_defaults()
                inst_map = defaults.instruction_schedule_map
            self._add_calibrations_to_target(inst_map)

    def _add_noisy_instruction_to_target(
        self, instruction: Instruction, noise_params: tuple[float, ...] | None
    ) -> None:
        """Add instruction properties to target for specified instruction.

        Args:
            instruction (qiskit.circuit.Instruction): Instance of instruction to be added to the target
            noise_params (tuple[float, ...] | None): error and duration noise values/ranges to
                include in instruction properties.
        """
        qarg_set = self._coupling_map if instruction.num_qubits > 1 else range(self._num_qubits)
        props = {}
        for qarg in qarg_set:
            try:
                qargs = tuple(qarg)
            except TypeError:
                qargs = (qarg,)
            duration, error = (
                noise_params
                if len(noise_params) == 2
                else (self._rng.uniform(*noise_params[:2]), self._rng.uniform(*noise_params[2:]))
            )
            props.update({qargs: InstructionProperties(duration, error)})

        self._target.add_instruction(instruction, props)

    def _generate_calibration_defaults(self) -> PulseDefaults:
        """Generate pulse calibration defaults if specified via ``calibrate_instructions``."""
        pulse_config = self._pulse_config(self._num_qubits)
        pulse_library = pulse_config["pulse_library"]
        # If self.calibrate_instructions==True, this method
        # will generate default pulse schedules for all gates in self._basis_gates,
        # except for `delay` and `reset`.
        calibration_buffer = self._basis_gates.copy()
        for inst in ["delay", "reset"]:
            calibration_buffer.remove(inst)
        # List of calibration commands (generated from sequences of PulseQobjInstructions)
        # corresponding to each calibrated instruction.
        cmd_def = []
        for inst in calibration_buffer:
            num_qubits = self._supported_gates[inst].num_qubits
            qarg_set = self._coupling_map if num_qubits > 1 else list(range(self.num_qubits))
            if inst == "measure":
                cmd_def.append(
                    Command(
                        name=inst,
                        qubits=qarg_set,
                        sequence=(
                            self._calibration_sequence(inst, num_qubits, qarg_set, pulse_library)
                            if self._calibrate_instructions
                            else []
                        ),
                    )
                )
            else:
                for qarg in qarg_set:
                    qubits = [qarg] if num_qubits == 1 else qarg
                    cmd_def.append(
                        Command(
                            name=inst,
                            qubits=qubits,
                            sequence=(
                                self._calibration_sequence(inst, num_qubits, qubits, pulse_library)
                                if self._calibrate_instructions
                                else []
                            ),
                        )
                    )

        return PulseDefaults(
            qubit_freq_est=pulse_config["qubit_freq_est"],
            meas_freq_est=pulse_config["meas_freq_est"],
            buffer=0,
            pulse_library=pulse_library,
            cmd_def=cmd_def,
        )

    def _add_calibrations_to_target(self, inst_map: InstructionScheduleMap) -> None:
        """Add calibration entries from provided pulse defaults to target.

        Args:
            inst_map (InstructionScheduleMap): pulse defaults with instruction schedule map
        """
        # The calibration entries are directly injected into the gate map to
        # avoid then being labeled as "user_provided".
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qobj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = inst_map._get_calibration_entry(inst, qargs)
                if inst in self._target._gate_map:
                    if inst == "measure":
                        for qubit in qargs:
                            self._target._gate_map[inst][(qubit,)].calibration = calibration_entry
                    elif qargs in self._target._gate_map[inst] and inst not in ["delay", "reset"]:
                        self._target._gate_map[inst][qargs].calibration = calibration_entry

    def run(self, run_input, **options):
        """Run on the fake backend using a simulator.

        This method runs circuit jobs (an individual or a list of :class:`~.QuantumCircuit`
        ) and pulse jobs (an individual or a list of :class:`~.Schedule` or
        :class:`~.ScheduleBlock`) using :class:`~.BasicAer` or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using the ``AerSimulator`` with
        noise model of the fake backend. Otherwise, jobs will be run using the
        ``BasicAer`` simulator without noise.

        Noisy simulations of pulse jobs are not yet supported in :class:`~.GenericFakeBackend`.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object, then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit_aer is not installed.
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
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator

            return AerSimulator._default_options()
        else:
            return BasicAer.get_backend("qasm_simulator")._default_options()

    def _build_default_channels(self) -> None:
        channels_map = {
            "acquire": {(i,): [pulse.AcquireChannel(i)] for i in range(self.num_qubits)},
            "drive": {(i,): [pulse.DriveChannel(i)] for i in range(self.num_qubits)},
            "measure": {(i,): [pulse.MeasureChannel(i)] for i in range(self.num_qubits)},
            "control": {
                (edge): [pulse.ControlChannel(i)] for i, edge in enumerate(self._coupling_map)
            },
        }
        setattr(self, "channels_map", channels_map)

    def drive_channel(self, qubit: int):
        drive_channels_map = getattr(self, "channels_map", {}).get("drive", {})
        qubits = (qubit,)
        if qubits in drive_channels_map:
            return drive_channels_map[qubits][0]
        return None

    def measure_channel(self, qubit: int):
        measure_channels_map = getattr(self, "channels_map", {}).get("measure", {})
        qubits = (qubit,)
        if qubits in measure_channels_map:
            return measure_channels_map[qubits][0]
        return None

    def acquire_channel(self, qubit: int):
        acquire_channels_map = getattr(self, "channels_map", {}).get("acquire", {})
        qubits = (qubit,)
        if qubits in acquire_channels_map:
            return acquire_channels_map[qubits][0]
        return None

    def control_channel(self, qubits: Iterable[int]):
        control_channels_map = getattr(self, "channels_map", {}).get("control", {})
        qubits = tuple(qubits)
        if qubits in control_channels_map:
            return control_channels_map[qubits]
        return []
