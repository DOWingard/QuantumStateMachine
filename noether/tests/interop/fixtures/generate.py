"""Regenerates gates.json, the reference unitaries the C++ interop tests compare against.

Needs qiskit, cirq-core and stim. Run from anywhere:

    python noether/tests/interop/fixtures/generate.py

Every matrix is stored in core index order: index bit q is qubit q (Qiskit's and Stim's little-endian
order; Cirq's big-endian matrices are bit-reversed). Entries are [re, im].
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import cirq
import numpy as np
import qiskit
import stim
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import UnitaryGate, get_standard_gate_name_mapping
from qiskit.quantum_info import Operator, random_unitary

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))
from noether_interop import to_circuit  # noqa: E402
from noether_interop._common import matrix_json  # noqa: E402

rng = np.random.default_rng(20261004)


def angles(k):
    return [float(x) for x in rng.uniform(-math.pi, math.pi, size=k)]


def bit_reversed(u: np.ndarray, n: int) -> np.ndarray:
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]
    return u[np.ix_(perm, perm)]


def snapped(u: np.ndarray) -> np.ndarray:
    """Stim's unitaries are complex64; Clifford entries have parts in {0, ±1/2, ±1/√2, ±1}, so restore
    them to double precision."""
    levels = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2**-0.5, -(2**-0.5)])

    def snap(x):
        k = np.argmin(np.abs(levels[:, None] - x.ravel()[None, :]), axis=0)
        out = levels[k].reshape(x.shape)
        assert np.max(np.abs(out - x)) < 1e-6
        return out

    u = np.asarray(u, dtype=complex)
    return snap(u.real) + 1j * snap(u.imag)


def stim_entries():
    out = []
    for gate in sorted(stim.gate_data().values(), key=lambda g: g.name):
        if not gate.is_unitary or gate.name in ("SPP", "SPP_DAG"):
            continue
        for name in sorted(set(gate.aliases)):
            for text, n in ([(f"{name} 0 1", 2), (f"{name} 1 0", 2)] if gate.is_two_qubit_gate else [(f"{name} 0", 1)]):
                u = snapped(stim.Circuit(text).to_tableau().to_unitary_matrix(endian="little"))
                out.append({"text": text, "qubits": n, "matrix": matrix_json(u)})
    for text, n in [("SPP X0*Y1", 2), ("SPP_DAG Z0*X1*Y2", 3), ("SPP !X0*Z1", 2), ("SPP Y0 Z1", 2), ("SPP_DAG !Y0", 1),
                    ("SPP X2*Z0*Y1", 3), ("SQRT_XX 1 0", 2)]:
        u = snapped(stim.Circuit(text).to_tableau().to_unitary_matrix(endian="little"))
        out.append({"text": text, "qubits": n, "matrix": matrix_json(u)})
    return out


def qiskit_entries():
    skip = {"measure", "reset", "delay", "barrier", "global_phase"}
    gates = []
    for name, gate in sorted(get_standard_gate_name_mapping().items()):
        if name in skip or gate.num_qubits > 5 or not isinstance(gate, qiskit.circuit.Gate):
            continue
        if gate.params:
            gate = type(gate)(*angles(len(gate.params)))
        gates.append((name, gate))
    gates += [
        ("cx_o0", qiskit.circuit.library.CXGate(ctrl_state=0)),
        ("ccx_o1", qiskit.circuit.library.CCXGate(ctrl_state=1)),
        ("crz_o0", qiskit.circuit.library.RZGate(angles(1)[0]).control(1, ctrl_state=0)),
        ("ch2", qiskit.circuit.library.HGate().control(2)),
        ("cswap2", qiskit.circuit.library.SwapGate().control(2, ctrl_state=2)),
        ("mcp3", qiskit.circuit.library.PhaseGate(angles(1)[0]).control(3)),
        ("unitary2", UnitaryGate(random_unitary(4, seed=7))),
        ("cunitary", UnitaryGate(random_unitary(4, seed=8)).control(1)),
    ]
    out = []
    for name, gate in gates:
        n = gate.num_qubits
        qc = QuantumCircuit(n)
        qc.append(gate, range(n))
        entry = {"name": name, "qubits": n, "circuit": to_circuit(qc), "matrix": matrix_json(Operator(qc).data)}
        for key, dump in (("qasm2", qasm2.dumps), ("qasm3", qasm3.dumps)):
            try:
                entry[key] = dump(qc)
            except Exception:
                pass
        out.append(entry)
    return out


def cirq_entries():
    q = cirq.LineQubit.range(3)
    t = angles(8)
    ops = {
        "X": cirq.X(q[0]), "Y": cirq.Y(q[0]), "Z": cirq.Z(q[0]), "H": cirq.H(q[0]), "S": cirq.S(q[0]), "T": cirq.T(q[0]),
        "S**-1": (cirq.S**-1)(q[0]), "T**-1": (cirq.T**-1)(q[0]),
        "X**t": (cirq.X ** (t[0] / math.pi))(q[0]), "Y**t": (cirq.Y ** (t[1] / math.pi))(q[0]),
        "Z**t": (cirq.Z ** (t[2] / math.pi))(q[0]), "X**0.5": (cirq.X**0.5)(q[0]), "X**-0.5": (cirq.X**-0.5)(q[0]),
        "XPow shifted": cirq.XPowGate(exponent=0.3, global_shift=0.2)(q[0]),
        "Rx": cirq.rx(t[3])(q[0]), "Ry": cirq.ry(t[4])(q[0]), "Rz": cirq.rz(t[5])(q[0]),
        "H**0.3": (cirq.H**0.3)(q[0]), "PhasedXZ": cirq.PhasedXZGate(x_exponent=0.3, z_exponent=0.1, axis_phase_exponent=0.7)(q[0]),
        "CNOT": cirq.CNOT(q[0], q[1]), "CNOT reversed": cirq.CNOT(q[1], q[0]), "CZ": cirq.CZ(q[0], q[1]),
        "CZ**t": (cirq.CZ**0.37)(q[0], q[1]), "CX**t": (cirq.CX**0.41)(q[0], q[1]), "SWAP": cirq.SWAP(q[0], q[1]),
        "ISWAP": cirq.ISWAP(q[0], q[1]), "ISWAP**t": (cirq.ISWAP**0.3)(q[0], q[1]), "FSim": cirq.FSimGate(0.4, 0.9)(q[0], q[1]),
        "XX**t": (cirq.XX**0.2)(q[0], q[1]), "ZZ**t": (cirq.ZZ**0.6)(q[0], q[1]),
        "CCX": cirq.CCX(q[0], q[1], q[2]), "CCZ": cirq.CCZ(q[0], q[1], q[2]), "CSWAP": cirq.CSWAP(q[0], q[1], q[2]),
        "C0Y": cirq.ControlledGate(cirq.Y, control_values=[0])(q[0], q[2]),
        "C(H)": cirq.H(q[2]).controlled_by(q[0]), "CC(Z**t)": (cirq.Z**0.3)(q[2]).controlled_by(q[0], q[1]),
        "C(X**t) controlled_by": (cirq.X**0.7)(q[1]).controlled_by(q[2]),
        "C(global phase)": cirq.GlobalPhaseGate(np.exp(0.7j)).on().controlled_by(q[1]),
        "Matrix": cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=3))(q[2], q[0]),
        "QFT": cirq.qft(*q),
    }
    out = []
    for name, op in ops.items():
        circuit = cirq.Circuit(op)
        qubits = sorted(circuit.all_qubits())
        n = len(qubits)
        u = cirq.unitary(circuit)
        out.append({"name": name, "qubits": n, "circuit": to_circuit(circuit), "matrix": matrix_json(bit_reversed(u, n))})
    return out


def main():
    doc = {
        "generator": "noether/tests/interop/fixtures/generate.py",
        "versions": {"qiskit": qiskit.__version__, "cirq": cirq.__version__, "stim": stim.__version__},
        "stim": stim_entries(),
        "qiskit": qiskit_entries(),
        "cirq": cirq_entries(),
    }
    (HERE / "gates.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(doc['stim'])} Stim, {len(doc['qiskit'])} Qiskit and {len(doc['cirq'])} Cirq entries")


if __name__ == "__main__":
    main()
