"""Running circuits through the noether CLI and re-keying its results to each framework's convention."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from ._common import ExportError


class NoetherError(Exception):
    """The CLI rejected the circuit; `diagnostics` holds its diagnostic records."""

    def __init__(self, message: str, diagnostics=()):
        super().__init__(message)
        self.diagnostics = list(diagnostics)


def _kind(obj) -> str:
    module = type(obj).__module__
    if module.startswith("qiskit"):
        return "qiskit"
    if module.startswith("cirq"):
        return "cirq"
    if module.startswith("stim"):
        return "stim"
    raise TypeError(f"expected a qiskit.QuantumCircuit, cirq.Circuit or stim.Circuit, got {type(obj).__name__}")


def to_circuit(obj, params=None) -> dict:
    """The noether.circuit/1 document of a Qiskit or Cirq circuit."""
    kind = _kind(obj)
    if kind == "qiskit":
        from .qiskit import to_circuit as convert
    elif kind == "cirq":
        from .cirq import to_circuit as convert
    else:
        raise TypeError("a stim.Circuit is imported from its text; use save() or run()")
    return convert(obj, params)


def save(obj, path, params=None) -> None:
    """Writes `obj` for the CLI: a .stim text file for Stim, circuit JSON otherwise."""
    path = Path(path)
    if _kind(obj) == "stim":
        path.write_text(str(obj) + "\n")
    else:
        path.write_text(json.dumps(to_circuit(obj, params)))


def _binary() -> str:
    configured = os.environ.get("NOETHER_BIN")
    if configured:
        if not os.access(configured, os.X_OK):
            raise RuntimeError(f"NOETHER_BIN={configured} is not an executable file")
        return configured
    found = shutil.which("noether")
    if not found:
        raise RuntimeError("the noether CLI was not found: set NOETHER_BIN or put noether on PATH")
    return found


def _invoke(obj, params, args) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ("circuit.stim" if _kind(obj) == "stim" else "circuit.json")
        save(obj, path, params)
        proc = subprocess.run([_binary(), "run", str(path), "--json", "--no-timing", *args], capture_output=True, text=True)
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise NoetherError(f"noether exited with {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}") from None
    if not doc.get("ok"):
        diags = doc.get("diagnostics", [])
        detail = "; ".join(f"{d['code']}: {d['message']}" for d in diags) or doc.get("error", "unknown error")
        raise NoetherError(detail, diags)
    return doc


def run(obj, shots: int = 1024, seed=None, backend: str = "auto", params=None) -> dict:
    """Runs `obj` and returns its results keyed in the framework's own convention.

    Qiskit: counts keyed like `Result.get_counts()` (clbit 0 rightmost, registers space-separated,
    the last register leftmost). Cirq: per-key histograms of ints with the key's first qubit as the
    most significant bit. Stim: measurement records (rows grouped by outcome, not in shot order) and
    detector / observable rates. `document` holds the CLI's noether.import-run/1 output.
    """
    args = ["--shots", str(shots), "--backend", backend]
    if seed is not None:
        args += ["--seed", str(seed)]
    doc = _invoke(obj, params, args)
    out = {"shots": doc["shots"], "seed": doc["seed"], "backend": doc["backend"], "document": doc}
    counts = doc["counts"]  # clbit 0 leftmost
    kind = _kind(obj)
    if kind == "qiskit":
        merged = Counter()
        for k, n in counts.items():
            merged[_qiskit_key(k, obj)] += n
        out["counts"] = dict(merged)
    elif kind == "cirq":
        per_key = {}
        for reg in doc["clbitRegisters"]:
            hist = Counter()
            for k, n in counts.items():
                value = 0
                for bit in k[reg["offset"] : reg["offset"] + reg["size"]]:
                    value = value * 2 + (bit == "1")
                hist[value] += n
            per_key[reg["name"]] = dict(hist)
        out["counts"] = per_key
    else:
        rows = []
        for k, n in counts.items():
            rows += [[c == "1" for c in k]] * n
        out["records"] = np.array(rows, dtype=bool).reshape(len(rows), doc["numClbits"])
        out["detectors"] = [d["rate"] for d in doc.get("detectors", [])]
        out["observables"] = [o["rate"] for o in doc.get("observables", [])]
    return out


def _qiskit_key(key: str, qc) -> str:
    if not qc.cregs:
        return key[::-1]
    parts = []
    for reg in reversed(qc.cregs):
        parts.append("".join(key[qc.find_bit(b).index] for b in reversed(list(reg))))
    return " ".join(parts)


def statevector(obj, params=None) -> np.ndarray:
    """Amplitudes before the terminal measurements, index bit q = qubit q (Qiskit's order; reverse the
    bits for Cirq's)."""
    doc = _invoke(obj, params, ["--emit", "statevector"])
    return np.array([complex(re, im) for re, im in doc["amplitudes"]])


def probabilities(obj, params=None) -> np.ndarray:
    doc = _invoke(obj, params, ["--emit", "probabilities"])
    return np.array(doc["probabilities"])


__all__ = ["ExportError", "NoetherError", "probabilities", "run", "save", "statevector", "to_circuit"]
