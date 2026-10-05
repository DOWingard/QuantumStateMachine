from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2, qasm3  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402
from qiskit.circuit.random import random_circuit  # noqa: E402
from qiskit.quantum_info import Kraus, Statevector, random_unitary  # noqa: E402

import noether_interop as ni  # noqa: E402
from conftest import amplitudes, assert_rates_close, cli_json, fidelity  # noqa: E402

TOL = 1e-10


@pytest.mark.parametrize("seed", range(24))
def test_random_circuits_match_statevector(seed):
    qc = random_circuit(1 + seed % 10, 20, max_operands=3, seed=seed)
    assert fidelity(ni.statevector(qc), Statevector(qc).data) >= 1 - TOL


def test_controlled_gates_keep_relative_phases():
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cu(0.3, 0.4, 0.5, 0.6, 0, 1)
    qc.append(UnitaryGate(random_unitary(4, seed=3)).control(2, ctrl_state=1), [2, 0, 1, 3])
    qc.mcp(0.7, [0, 1, 2], 3)
    qc.crx(1.1, 3, 0)
    qc.ch(1, 2, ctrl_state=0)
    assert fidelity(ni.statevector(qc), Statevector(qc).data) >= 1 - TOL


@pytest.mark.parametrize("seed", range(10))
def test_qasm_dumps_agree_with_circuit_json(seed, tmp_path):
    qc = random_circuit(4, 8, max_operands=3, seed=100 + seed)
    want = Statevector(qc).data
    routes = 0
    for name, dump in (("v2.qasm", qasm2.dumps), ("v3.qasm", qasm3.dumps)):
        try:
            text = dump(qc)
        except Exception:  # the exporter cannot write every gate in every version
            continue
        path = tmp_path / name
        path.write_text(text)
        assert fidelity(amplitudes(cli_json(path, "--emit", "statevector")), want) >= 1 - TOL, text
        routes += 1
    assert routes >= 1


def test_counts_are_keyed_like_qiskit():
    q = QuantumRegister(3, "q")
    a, b = ClassicalRegister(2, "a"), ClassicalRegister(1, "b")
    qc = QuantumCircuit(q, a, b)
    qc.x(0)
    qc.x(2)
    qc.measure(0, a[0])
    qc.measure(1, a[1])
    qc.measure(2, b[0])
    assert ni.run(qc, shots=10, seed=1)["counts"] == {"1 01": 10}


def test_parameters_bind_and_unbound_ones_fail():
    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(t, 0)
    want = Statevector(qc.assign_parameters({t: 0.3})).data
    assert fidelity(ni.statevector(qc, params={t: 0.3}), want) >= 1 - TOL
    with pytest.raises(ni.ExportError) as e:
        ni.to_circuit(qc)
    assert e.value.code == "E9005"


def test_dynamic_circuit_matches_aer():
    aer = pytest.importorskip("qiskit_aer")
    qc = QuantumCircuit(3, 3)
    qc.ry(1.1, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    qc.reset(0)
    qc.measure(2, 2)
    shots = 20000
    ours = ni.run(qc, shots=shots, seed=2)["counts"]
    theirs = aer.AerSimulator(seed_simulator=2).run(qc, shots=shots).result().get_counts()
    for key in set(ours) | set(theirs):
        assert_rates_close(ours.get(key, 0) / shots, shots, theirs.get(key, 0) / shots, shots, key)
    p1 = sum(n for k, n in ours.items() if k[0] == "1") / shots
    assert abs(p1 - math.sin(0.55) ** 2) < 4 * math.sqrt(0.25 / shots)


def test_channels_become_pauli_or_kraus():
    noise = pytest.importorskip("qiskit_aer.noise")
    shots = 20000
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.append(noise.pauli_error([("X", 0.3), ("I", 0.7)]).to_instruction(), [0])
    g = 0.25
    qc.append(Kraus([np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])]), [1])
    qc.measure([0, 1], [0, 1])
    doc = ni.to_circuit(qc)
    assert [op["op"] for op in doc["ops"]][1:3] == ["pauli_channel", "kraus"]
    counts = ni.run(qc, shots=shots, seed=3)["counts"]
    p0 = sum(n for k, n in counts.items() if k[-1] == "1") / shots
    p1 = sum(n for k, n in counts.items() if k[-2] == "1") / shots
    assert abs(p0 - 0.3) < 4 * math.sqrt(0.21 / shots)
    assert abs(p1 - (1 - g)) < 4 * math.sqrt(g * (1 - g) / shots)
