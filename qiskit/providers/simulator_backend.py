# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simulator-based backend for running against a target locally."""

from __future__ import annotations
import warnings

from collections.abc import Iterable
import numpy as np

from qiskit import pulse
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.transpiler import Target
from qiskit.providers import Options
from qiskit.providers.basicaer import BasicAer
from qiskit.providers.backend import BackendV2
from qiskit.utils import optionals as _optionals


class _SimulatorBackend(BackendV2):
    """A simulator-based backend for running against a target locally.

    This class runs circuit and pulse jobs using the Aer simulator if qiskit-aer
    is installed, and otherwise falls back to using :class:`~.BasicAer` which
    does NOT support noise (the target's noise model is ignored).

    Noisy simulations of pulse jobs are not yet supported.
    """

    def __init__(
        self,
        target: Target,
        *,
        dtm: float = None,
        seed: int = 42,
    ):
        """
        Args:
            target: The target system the simulator will run against.

            dtm: System time resolution of output signals in nanoseconds.
                None by default.

            seed: Optional seed for generation of default values.
        """
        num_qubits = target.num_qubits

        super().__init__(
            provider=None,
            name=f"simulated_{num_qubits}q",
            description=f"This is a simulator-based device with {num_qubits} qubits. Target description: '{target.description}'.",
            backend_version="",
        )

        self.sim = None
        self._target = target
        self._coupling_map = target.build_coupling_map()
        self._rng = np.random.default_rng(seed=seed)
        self._dtm = dtm
        self._build_default_channels()

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals"""
        # converting `dtm` from nanoseconds to seconds
        return self._dtm * 1e-9 if self._dtm is not None else None

    @property
    def meas_map(self) -> list[list[int]]:
        return self._target.concurrent_measurements

    def _build_default_channels(self) -> None:
        channels_map = {
            "acquire": {(i,): [pulse.AcquireChannel(i)] for i in range(self.target.num_qubits)},
            "drive": {(i,): [pulse.DriveChannel(i)] for i in range(self.target.num_qubits)},
            "measure": {(i,): [pulse.MeasureChannel(i)] for i in range(self.target.num_qubits)},
            "control": {
                (edge): [pulse.ControlChannel(i)] for i, edge in enumerate(self._coupling_map)
            },
        }
        setattr(self, "channels_map", channels_map)

    def run(self, run_input, **options):
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
