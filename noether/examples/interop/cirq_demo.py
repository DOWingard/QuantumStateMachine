"""Runs a noisy Cirq circuit on the state machine and checks it against Cirq's density matrix.

    pip install ./noether/python[cirq]        # or: uv pip install ./noether/python[cirq]
    NOETHER_BIN=build/release/noether/noether python noether/examples/interop/cirq_demo.py
"""

import math

import cirq
import numpy as np

import noether_interop as ni

a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(a),
    cirq.CNOT(a, b),
    cirq.depolarize(0.1).on(a),    # becomes a Pauli channel
    cirq.amplitude_damp(0.2).on(b),  # becomes a Kraus channel
    cirq.measure(a, b, key="m"),
)

shots = 40000
hist = ni.run(circuit, shots=shots, seed=1)["counts"]["m"]  # key value: a is the most significant bit
rho = cirq.DensityMatrixSimulator().simulate(circuit[:-1]).final_density_matrix
for k, p in enumerate(np.real(np.diag(rho))):
    got = hist.get(k, 0) / shots
    print(f"{k:02b}: {got:.4f} (density matrix {p:.4f})")
    assert abs(got - p) < 4 * math.sqrt(max(p * (1 - p), 1e-9) / shots) + 1e-9
