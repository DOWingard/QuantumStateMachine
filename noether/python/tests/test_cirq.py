from __future__ import annotations

import math

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

import noether_interop as ni  # noqa: E402
from conftest import amplitudes, assert_rates_close, bit_reversed_state, cli_json, fidelity  # noqa: E402

TOL = 1e-10

GATES = {
    cirq.X: 1, cirq.Y: 1, cirq.Z: 1, cirq.H: 1, cirq.S: 1, cirq.T: 1, cirq.X**0.31: 1, cirq.Y**-0.7: 1,
    cirq.Z**0.13: 1, cirq.H**0.4: 1, cirq.rx(0.9): 1, cirq.PhasedXZGate(x_exponent=0.3, z_exponent=0.1, axis_phase_exponent=0.7): 1,
    cirq.XPowGate(exponent=0.2, global_shift=0.3): 1, cirq.CNOT: 2, cirq.CZ: 2, cirq.CZ**0.37: 2, cirq.SWAP: 2,
    cirq.ISWAP: 2, cirq.ISWAP**0.3: 2, cirq.FSimGate(0.4, 0.9): 2, cirq.XX**0.2: 2, cirq.ZZ**0.6: 2, cirq.CCX: 3,
    cirq.CCZ: 3, cirq.CSWAP: 3, cirq.ControlledGate(cirq.Y**0.3, control_values=[0]): 2,
}


def state(circuit):
    """Cirq's final state in core order (index bit q = q-th sorted qubit)."""
    qubits = sorted(circuit.all_qubits())
    return bit_reversed_state(cirq.final_state_vector(circuit, qubit_order=qubits), len(qubits))


@pytest.mark.parametrize("seed", range(24))
def test_random_circuits_match_final_state(seed):
    c = cirq.testing.random_circuit(qubits=3 + seed % 8, n_moments=20, op_density=0.9, gate_domain=GATES, random_state=seed)
    if not c.all_qubits():
        pytest.skip("empty circuit")
    assert fidelity(ni.statevector(c), state(c)) >= 1 - TOL


def test_controlled_global_phase_and_qft():
    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.H.on_each(*q), cirq.GlobalPhaseGate(np.exp(0.7j)).on().controlled_by(q[1]), cirq.qft(*q),
                     (cirq.Z**0.3)(q[2]).controlled_by(q[0], q[1]))
    assert fidelity(ni.statevector(c), state(c)) >= 1 - TOL


@pytest.mark.parametrize("seed", range(8))
def test_cirq_qasm_agrees_with_circuit_json(seed, tmp_path):
    gates = {g: n for g, n in GATES.items() if not isinstance(g, cirq.ControlledGate)}
    c = cirq.testing.random_circuit(qubits=4, n_moments=6, op_density=0.9, gate_domain=gates, random_state=50 + seed)
    path = tmp_path / "c.qasm"
    path.write_text(cirq.qasm(c, args=cirq.QasmArgs(version="2.0")))
    # cirq.qasm declares one register over the sorted qubits, the order the JSON route uses.
    assert fidelity(amplitudes(cli_json(path, "--emit", "statevector")), ni.statevector(c)) >= 1 - TOL


def test_measurement_keys_are_per_key_histograms():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(a), cirq.X(c), cirq.measure(a, b, key="m"), cirq.measure(c, key="z", invert_mask=(True,)))
    out = ni.run(circuit, shots=20, seed=1)["counts"]
    assert out == {"m": {0b10: 20}, "z": {0: 20}}


def test_classical_control_feeds_forward():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.measure(a, key="a"), cirq.X(b).with_classical_controls("a"), cirq.measure(b, key="b"))
    shots = 400
    doc = ni.run(circuit, shots=shots, seed=2)["document"]
    for key, n in doc["counts"].items():
        assert key[0] == key[1], (key, n)


def test_noise_matches_the_density_matrix():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(q[0]), cirq.X(q[1]), cirq.depolarize(0.2).on(q[0]), cirq.asymmetric_depolarize(0.1, 0.05, 0.2).on(q[1]),
                     cirq.amplitude_damp(0.3).on(q[1]), cirq.H(q[0]), cirq.measure(*q, key="m"))
    rho = cirq.DensityMatrixSimulator().simulate(c[:-1]).final_density_matrix
    probs = np.real(np.diag(rho))  # big-endian over (q0, q1)
    shots = 40000
    hist = ni.run(c, shots=shots, seed=3)["counts"]["m"]
    for k in range(4):
        assert_rates_close(hist.get(k, 0) / shots, shots, probs[k], 10**9, f"outcome {k}")
