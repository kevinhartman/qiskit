from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_gate_map

service = QiskitRuntimeService(channel="ibm_quantum")

backend = service.get_backend("torino")

plot_gate_map(backend)