"""The Python examples in noether/examples/interop assert their own results."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples" / "interop"


@pytest.mark.parametrize("script,needs", [("qiskit_demo.py", "qiskit"), ("cirq_demo.py", "cirq")])
def test_example_runs(script, needs):
    pytest.importorskip(needs)
    proc = subprocess.run([sys.executable, str(EXAMPLES / script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
