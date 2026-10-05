"""Shared helpers. The suite drives the noether CLI: NOETHER_BIN, else `noether` on PATH, else the
repository's build/release (or build/debug) binary. Without one the session stops at once."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def pytest_configure(config):
    if os.environ.get("NOETHER_BIN") or shutil.which("noether"):
        return
    for build in ("release", "debug"):
        candidate = ROOT / "build" / build / "noether" / "noether"
        if candidate.exists():
            os.environ["NOETHER_BIN"] = str(candidate)
            return
    raise pytest.UsageError("no noether binary: set NOETHER_BIN, put noether on PATH, or build the project first")


def cli_json(path, *args) -> dict:
    """`noether run <path> --json --no-timing <args>`, parsed; fails the test on a non-zero exit."""
    proc = subprocess.run([os.environ.get("NOETHER_BIN") or shutil.which("noether"), "run", str(path), "--json", "--no-timing", *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def amplitudes(doc) -> np.ndarray:
    return np.array([complex(re, im) for re, im in doc["amplitudes"]])


def fidelity(a, b) -> float:
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def bit_reversed_state(v, n) -> np.ndarray:
    """Reorders amplitudes between little-endian (index bit q = qubit q) and big-endian."""
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)] if n else [0]
    return np.asarray(v)[perm]


def assert_rates_close(a, n, b, m, what=""):
    """Two binomial proportions (a over n trials, b over m) agree within 4σ of their difference."""
    p = (a * n + b * m) / (n + m)
    sigma = np.sqrt(max(p * (1 - p), 1e-12) * (1 / n + 1 / m))
    assert abs(a - b) <= 4 * sigma + 1e-9, f"{what}: {a} vs {b} (σ = {sigma})"
