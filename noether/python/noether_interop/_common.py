"""Pieces shared by the exporters: the document format, conditions and Pauli-channel detection."""

from __future__ import annotations

import itertools

import numpy as np

SCHEMA = "noether.circuit/1"
MAX_DENSE = 10  # QuantumGate::kMaxDenseTargets

# A classical condition that can never hold; its operations are dropped.
NEVER = None


class ExportError(Exception):
    """A circuit construct the state machine cannot represent. `code` is the noether diagnostic code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def matrix_json(m) -> list:
    m = np.asarray(m, dtype=complex)
    return [[[float(z.real), float(z.imag)] for z in row] for row in m]


def operation(name: str, targets, controls=(), params=(), matrix=None, kraus=None, clbit=None, condition=None) -> dict:
    op = {"op": name, "targets": [int(q) for q in targets]}
    if controls:
        op["controls"] = [int(q) for q in controls]
    if params:
        op["params"] = [float(p) for p in params]
    if matrix is not None:
        if len(targets) > MAX_DENSE:
            raise ExportError("E9004", f"a dense matrix on {len(targets)} qubits; at most {MAX_DENSE}")
        op["matrix"] = matrix_json(matrix)
    if kraus is not None:
        if len(targets) > MAX_DENSE:
            raise ExportError("E9004", f"a Kraus channel on {len(targets)} qubits; at most {MAX_DENSE}")
        op["kraus"] = [matrix_json(k) for k in kraus]
    if clbit is not None:
        op["clbit"] = int(clbit)
    if condition:
        bits = sorted(condition)
        op["condition"] = {"clbits": bits, "values": [int(condition[b]) for b in bits]}
    return op


def merge(a, b):
    """Conjunction of two {clbit: value} conditions; NEVER when they contradict."""
    if a is NEVER or b is NEVER:
        return NEVER
    out = dict(a)
    for bit, value in b.items():
        if out.get(bit, value) != value:
            return NEVER
        out[bit] = value
    return out


_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def pauli_probabilities(kraus, n: int):
    """Probabilities in core order (I X Y Z digits, the first letter on targets[0], identity omitted)
    when the channel with these Kraus operators (targets[0] = most significant bit) is a Pauli channel,
    otherwise None. The channel is Pauli iff its chi matrix in the Pauli basis is diagonal."""
    if n not in (1, 2):
        return None
    d = 2**n
    strings = []
    for letters in itertools.product(range(4), repeat=n):
        p = np.ones((1, 1), dtype=complex)
        for x in letters:
            p = np.kron(p, _PAULI[x])
        strings.append(p)
    coeffs = np.array([[np.trace(p.conj().T @ np.asarray(k, dtype=complex)) / d for p in strings] for k in kraus])
    chi = coeffs.T @ coeffs.conj()
    if np.max(np.abs(chi - np.diag(np.diag(chi)))) > 1e-12:
        return None
    probs = np.real(np.diag(chi))
    return [float(max(p, 0.0)) for p in probs[1:]]


def document(num_qubits, num_clbits, labels, registers, framework, version, ops, notes) -> dict:
    return {
        "schema": SCHEMA,
        "num_qubits": max(int(num_qubits), 1),
        "num_clbits": int(num_clbits),
        "qubit_labels": labels if num_qubits else ["(unused)"],
        "clbit_registers": registers,
        "source": {"framework": framework, "version": version},
        "ops": ops,
        "notes": notes,
    }
