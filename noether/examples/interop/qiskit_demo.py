"""Runs a Qiskit circuit with mid-circuit feed-forward on the state machine and checks it.

    pip install ./noether/python[qiskit]      # or: uv pip install ./noether/python[qiskit]
    NOETHER_BIN=build/release/noether/noether python noether/examples/interop/qiskit_demo.py
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

import noether_interop as ni

# Unitary part: amplitudes agree with Qiskit's own simulator.
qc = QuantumCircuit(4)
qc.h(range(4))
qc.cu(0.3, 0.4, 0.5, 0.6, 0, 1)
qc.append(QFTGate(4), range(4))
fidelity = abs(np.vdot(ni.statevector(qc), Statevector(qc).data)) ** 2
print(f"fidelity with qiskit.quantum_info.Statevector: {fidelity:.12f}")
assert fidelity > 1 - 1e-10

# Dynamic circuit: teleport ry(1.1)|0⟩ and read it back on qubit 2.
tp = QuantumCircuit(3, 3)
tp.ry(1.1, 0)
tp.h(1)
tp.cx(1, 2)
tp.cx(0, 1)
tp.h(0)
tp.measure([0, 1], [0, 1])
with tp.if_test((tp.clbits[1], 1)):
    tp.x(2)
with tp.if_test((tp.clbits[0], 1)):
    tp.z(2)
tp.measure(2, 2)
shots = 20000
result = ni.run(tp, shots=shots, seed=1)
p1 = sum(n for key, n in result["counts"].items() if key[0] == "1") / shots  # Qiskit keys: clbit 0 rightmost
print(f"backend {result['backend']}; P(c2 = 1) = {p1:.4f}, expected {math.sin(0.55) ** 2:.4f}")
assert abs(p1 - math.sin(0.55) ** 2) < 4 * math.sqrt(0.25 / shots)
