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

"""Functions for constructing fake (simulated) backends with customizable generic targets"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .simulator_backend import _SimulatorBackend

from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import Instruction
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
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
    PulseDefaults,
    Command,
)
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem

# Noise default values/ranges for duration and error of supported
# instructions. There are two possible formats:
# - (min_duration, max_duration, min_error, max_error),
#   if the defaults are ranges.
# - (duration, error), if the defaults are fixed values.
_NOISE_DEFAULTS = {
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

# Ranges to sample qubit properties from.
_QUBIT_PROPERTIES = {
    "dt": 0.222e-9,
    "t1": (100e-6, 200e-6),
    "t2": (100e-6, 200e-6),
    "frequency": (5e9, 5.5e9),
}

# The number of samples determines the pulse durations of the corresponding
# instructions. This default defines pulses with durations in multiples of
# 16 for consistency with the pulse granularity of real IBM devices, but
# keeps the number smaller than what would be realistic for
# manageability. If needed, more realistic durations could be added in the
# future (order of 160dt for 1q gates, 1760dt for 2q gates and measure).
_PULSE_LIBRARY = [
    PulseLibraryItem(name="pulse_1", samples=np.linspace(0, 1.0, 16, dtype=np.complex128)),  # 16dt
    PulseLibraryItem(name="pulse_2", samples=np.linspace(0, 1.0, 32, dtype=np.complex128)),  # 32dt
    PulseLibraryItem(name="pulse_3", samples=np.linspace(0, 1.0, 64, dtype=np.complex128)),  # 64dt
]


def make_generic_backend(
    num_qubits: int,
    basis_gates: list[str] | None = None,
    *,
    coupling_map: list[list[int]] | CouplingMap | None = None,
    control_flow: bool = False,
    calibrate_instructions: bool | InstructionScheduleMap | None = None,
    dtm: float = None,
    seed: int = 42,
) -> BackendV2:
    """
    Instantiates a partially configurable, locally runnable, fake backend.
    When :meth:`~BackendV2.run` is invoked on the returned backend, a simulator
    is used for execution.

    Device characterization properties (such as noise, gate durations, fidelity, etc.) are generated
    by randomly sampling from default ranges extracted from historical IBM backend data.
    The provided ``seed`` is used for random generation to ensure the reproducibility of
    the generated backend.

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
    rng = np.random.default_rng(seed=seed)

    if coupling_map is None:
        coupling_map = CouplingMap().from_full(num_qubits)
    else:
        if not isinstance(coupling_map, CouplingMap):
            coupling_map = CouplingMap(coupling_map)

        if num_qubits != coupling_map.size():
            raise QiskitError(
                f"The number of qubits (got {num_qubits}) must match "
                f"the size of the provided coupling map (got {coupling_map.size()})."
            )

    basis_gates = basis_gates if basis_gates is not None else ["cx", "id", "rz", "sx", "x"]

    # the qubit properties are sampled from default ranges
    target = Target(
        description=f"Generic target with {num_qubits} qubits",
        num_qubits=num_qubits,
        dt=_QUBIT_PROPERTIES["dt"],
        qubit_properties=[
            QubitProperties(
                t1=rng.uniform(_QUBIT_PROPERTIES["t1"][0], _QUBIT_PROPERTIES["t1"][1]),
                t2=rng.uniform(_QUBIT_PROPERTIES["t2"][0], _QUBIT_PROPERTIES["t2"][1]),
                frequency=rng.uniform(
                    _QUBIT_PROPERTIES["frequency"][0], _QUBIT_PROPERTIES["frequency"][1]
                ),
            )
            for _ in range(num_qubits)
        ],
        concurrent_measurements=[list(range(num_qubits))],
    )

    standard_gates = get_standard_gate_name_mapping()
    for name in basis_gates:
        if name not in standard_gates:
            raise QiskitError(
                f"Provided basis gate {name} is not an instruction "
                f"in the standard qiskit circuit library."
            )
        gate = standard_gates[name]
        noise_params = _NOISE_DEFAULTS.get(name, (1e-8, 9e-7, 1e-5, 5e-3))
        _add_noisy_instruction_to_target(target, coupling_map, gate, noise_params, rng)

    # Add directives if they're missing.
    for name in ["reset", "delay", "measure"]:
        if name not in target.operation_names:
            _add_noisy_instruction_to_target(
                target, coupling_map, standard_gates[name], _NOISE_DEFAULTS[name], rng
            )

    if control_flow:
        target.add_instruction(IfElseOp, name="if_else")
        target.add_instruction(WhileLoopOp, name="while_loop")
        target.add_instruction(ForLoopOp, name="for_loop")
        target.add_instruction(SwitchCaseOp, name="switch_case")
        target.add_instruction(BreakLoopOp, name="break")
        target.add_instruction(ContinueLoopOp, name="continue")

    # Generate block of calibration defaults and add to target.
    # Note: this could be improved if we could generate and add
    # calibration defaults per-gate, and not as a block.
    if calibrate_instructions is not None:
        if isinstance(calibrate_instructions, InstructionScheduleMap):
            inst_map = calibrate_instructions
        else:
            defaults = _generate_calibration_defaults(
                target, coupling_map, calibrate_instructions, standard_gates
            )
            inst_map = defaults.instruction_schedule_map
        _add_calibrations_to_target(target, inst_map)

    return _SimulatorBackend(target, dtm=dtm, seed=seed)


def _add_noisy_instruction_to_target(
    target: Target,
    coupling_map: CouplingMap,
    instruction: Instruction,
    noise_params: tuple[float, ...] | None,
    rng: Generator,
) -> None:
    """Add instruction properties to target for specified instruction.

    Args:
        instruction (qiskit.circuit.Instruction): Instance of instruction to be added to the target
        noise_params (tuple[float, ...] | None): error and duration noise values/ranges to
            include in instruction properties.

    Returns:
        None
    """
    qarg_set = coupling_map if instruction.num_qubits > 1 else range(coupling_map.size())
    props = {}
    for qarg in qarg_set:
        try:
            qargs = tuple(qarg)
        except TypeError:
            qargs = (qarg,)
        duration, error = (
            noise_params
            if len(noise_params) == 2
            else (rng.uniform(*noise_params[:2]), rng.uniform(*noise_params[2:]))
        )
        props.update({qargs: InstructionProperties(duration, error)})

    target.add_instruction(instruction, props)


def _get_calibration_sequence(
    inst: str, num_qubits: int, qargs: tuple[int]
) -> list[PulseQobjInstruction]:
    """Return calibration sequence for given instruction (defined by name and num_qubits)
    acting on qargs."""

    pulse_library = _PULSE_LIBRARY
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


def _generate_calibration_defaults(
    target: Target, coupling_map: CouplingMap, calibrate_instructions, supported_gates
) -> PulseDefaults:
    """Generate pulse calibration defaults if specified via `calibrate_instructions`."""

    # If calibrate_instructions==True, this method
    # will generate default pulse schedules for all gates in the target
    # except for `delay` and `reset`.

    # List of calibration commands (generated from sequences of PulseQobjInstructions)
    # corresponding to each calibrated instruction. Note that the calibration pulses
    # are different for 1q gates vs 2q gates vs measurement instructions.
    cmd_def = []
    for inst in target.operation_names:
        if inst in {"delay", "reset"}:
            continue
        num_qubits = supported_gates[inst].num_qubits
        qarg_set = coupling_map if num_qubits > 1 else list(range(coupling_map.size()))
        if inst == "measure":
            cmd_def.append(
                Command(
                    name=inst,
                    qubits=qarg_set,
                    sequence=(
                        _get_calibration_sequence(inst, num_qubits, qarg_set)
                        if calibrate_instructions
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
                            _get_calibration_sequence(inst, num_qubits, qubits)
                            if calibrate_instructions
                            else []
                        ),
                    )
                )

    qubit_freq_est = np.random.normal(4.8, scale=0.01, size=coupling_map.size()).tolist()
    meas_freq_est = np.linspace(6.4, 6.6, coupling_map.size()).tolist()
    return PulseDefaults(
        qubit_freq_est=qubit_freq_est,
        meas_freq_est=meas_freq_est,
        buffer=0,
        pulse_library=_PULSE_LIBRARY,
        cmd_def=cmd_def,
    )


def _add_calibrations_to_target(target: Target, inst_map: InstructionScheduleMap) -> None:
    """Add calibration entries from provided pulse defaults to target.

    Args:
        inst_map (InstructionScheduleMap): pulse defaults with instruction schedule map

    Returns:
        None
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
            if inst in target._gate_map:
                if inst == "measure":
                    for qubit in qargs:
                        target._gate_map[inst][(qubit,)].calibration = calibration_entry
                elif qargs in target._gate_map[inst] and inst not in ["delay", "reset"]:
                    target._gate_map[inst][qargs].calibration = calibration_entry
