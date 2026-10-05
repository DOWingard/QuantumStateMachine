"""Cirq Circuit → noether.circuit/1.

Qubit k is the k-th of `sorted(circuit.all_qubits())`. Cirq matrices are big-endian in the
operation's qubits, as are the core's (targets[0] = most significant bit), so dense operators keep
their qubit order. Each measurement key takes a contiguous clbit range in order of first appearance;
the key's j-th qubit is offset + j.
"""

from __future__ import annotations

import math

import cirq
import numpy as np

from ._common import MAX_DENSE, ExportError, document, operation, pauli_probabilities


def to_circuit(circuit: cirq.AbstractCircuit, params=None) -> dict:
    """The noether.circuit/1 document of `circuit`, with `params` resolved first."""
    if params is not None:
        circuit = cirq.resolve_parameters(circuit, params)
    if cirq.is_parameterized(circuit):
        names = ", ".join(sorted(str(s) for s in cirq.parameter_names(circuit)))
        raise ExportError("E9005", f"unresolved symbols: {names}; pass params= to resolve them")
    qubits = sorted(circuit.all_qubits())
    for q in qubits:
        if q.dimension != 2:
            raise ExportError("E9001", f"{q} has dimension {q.dimension}; only qubits are supported")
    return _Exporter(qubits).run(circuit)


def _is(gate, cls, exponent=None, shift=0.0) -> bool:
    if not isinstance(gate, cls):
        return False
    if exponent is not None and float(gate.exponent) != exponent:
        return False
    return shift is None or float(getattr(gate, "global_shift", 0.0)) == shift


def _keep(op) -> bool:
    """Operations mapped as they stand; everything else is decomposed first."""
    if isinstance(op, cirq.ClassicallyControlledOperation):
        return _keep(op.without_classical_controls())
    if isinstance(op, cirq.ControlledOperation) or isinstance(op.gate, cirq.ControlledGate):
        return True
    g = op.gate
    if g is None:
        return False
    if isinstance(g, (cirq.MeasurementGate, cirq.ResetChannel, cirq.GlobalPhaseGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,
                      cirq.CZPowGate, cirq.CSwapGate)):
        return True
    if _is(g, cirq.HPowGate, 1) or _is(g, cirq.CXPowGate, 1) or _is(g, cirq.SwapPowGate, 1):
        return True
    if _is(g, cirq.CCXPowGate, 1) or _is(g, cirq.CCZPowGate, 1):
        return True
    return cirq.has_kraus(op) and not cirq.has_unitary(op)


class _Exporter:
    def __init__(self, qubits):
        self.qubits = qubits
        self.index = {q: i for i, q in enumerate(qubits)}
        self.ops = []
        self.notes = []
        self.keys = {}  # MeasurementKey → (offset, size)
        self.registers = []
        self.clbits = 0
        self.phase_noted = False

    def run(self, circuit) -> dict:
        flat = cirq.decompose(list(circuit.all_operations()), keep=_keep, on_stuck_raise=None)
        for k, op in enumerate(flat):
            try:
                self.op(op, {})
            except ExportError as e:
                raise ExportError(e.code, f"operation {k} ({op}): {str(e).split(': ', 1)[-1]}") from None
        return document(len(self.qubits), self.clbits, [str(q) for q in self.qubits], self.registers, "cirq", cirq.__version__,
                        self.ops, self.notes)

    def emit(self, name, targets, condition, **kw):
        self.ops.append(operation(name, targets, condition=condition, **kw))

    def qs(self, qubits):
        return [self.index[q] for q in qubits]

    def op(self, op, cond):
        if isinstance(op, cirq.ClassicallyControlledOperation):
            self.classically_controlled(op, cond)
            return
        if isinstance(op, cirq.ControlledOperation):
            self.controlled(list(op.controls), op.control_values, op.sub_operation, cond)
            return
        g = op.gate
        if isinstance(g, cirq.ControlledGate):
            n = g.num_controls()
            self.controlled(list(op.qubits[:n]), g.control_values, g.sub_gate.on(*op.qubits[n:]), cond)
            return
        qs = self.qs(op.qubits)
        if isinstance(g, cirq.MeasurementGate):
            self.measure(op, g, qs, cond)
            return
        if isinstance(g, cirq.ResetChannel):
            self.emit("reset", qs, cond)
            return
        if isinstance(g, cirq.GlobalPhaseGate):
            if not self.phase_noted:
                self.notes.append({"code": "W9001", "message": "an uncontrolled cirq.GlobalPhaseGate is unobservable and is dropped"})
                self.phase_noted = True
            return
        if self.gate(g, qs, cond) or self.channel(op, g, qs, cond):
            return
        if cirq.has_unitary(op):
            if len(qs) > MAX_DENSE:
                raise ExportError("E9004", f"a dense matrix on {len(qs)} qubits; at most {MAX_DENSE}")
            self.emit("unitary", qs, cond, matrix=cirq.unitary(op))
            return
        raise ExportError("E9001", "no mapping, decomposition or unitary")

    def gate(self, g, qs, cond) -> bool:
        """Exact maps of uncontrolled gates; a global shift only changes the unobservable global phase."""
        if isinstance(g, cirq.XPowGate):
            t = float(g.exponent)
            if _is(g, cirq.XPowGate, 1):
                self.emit("x", qs, cond)
            elif _is(g, cirq.XPowGate, 0.5):
                self.emit("sx", qs, cond)
            elif _is(g, cirq.XPowGate, -0.5):
                self.emit("sx", qs, cond)
                self.emit("x", qs, cond)
            else:
                self.emit("rx", qs, cond, params=[math.pi * t])
            return True
        if isinstance(g, cirq.YPowGate):
            if _is(g, cirq.YPowGate, 1):
                self.emit("y", qs, cond)
            else:
                self.emit("ry", qs, cond, params=[math.pi * float(g.exponent)])
            return True
        if isinstance(g, cirq.ZPowGate):
            t = float(g.exponent)
            named = {1.0: "z", 0.5: "s", -0.5: "sdg", 0.25: "t", -0.25: "tdg"}
            if float(g.global_shift) == 0.0 and t in named:
                self.emit(named[t], qs, cond)
            elif float(g.global_shift) == -0.5:
                self.emit("rz", qs, cond, params=[math.pi * t])
            else:
                self.emit("phase", qs, cond, params=[math.pi * t])
            return True
        if _is(g, cirq.HPowGate, 1):
            self.emit("h", qs, cond)
            return True
        if _is(g, cirq.CXPowGate, 1):
            self.emit("cnot", [qs[1]], cond, controls=[qs[0]])
            return True
        if isinstance(g, cirq.CZPowGate):
            if _is(g, cirq.CZPowGate, 1):
                self.emit("cz", [qs[1]], cond, controls=[qs[0]])
            else:
                self.emit("cphase", [qs[1]], cond, controls=[qs[0]], params=[math.pi * float(g.exponent)])
            return True
        if _is(g, cirq.SwapPowGate, 1):
            self.emit("swap", qs, cond)
            return True
        if _is(g, cirq.CCXPowGate, 1):
            self.emit("toffoli", [qs[2]], cond, controls=qs[:2])
            return True
        if _is(g, cirq.CCZPowGate, 1):
            self.emit("mcz", qs, cond)
            return True
        if isinstance(g, cirq.CSwapGate):
            self.emit("fredkin", qs[1:], cond, controls=[qs[0]])
            return True
        return False

    def channel(self, op, g, qs, cond) -> bool:
        if not cirq.has_kraus(op) or cirq.has_unitary(op):
            return False
        n = len(qs)
        probs = None
        if isinstance(g, cirq.DepolarizingChannel) and n in (1, 2):
            probs = [g.p / (4**n - 1)] * (4**n - 1)
        elif isinstance(g, cirq.BitFlipChannel):
            probs = [g.p, 0.0, 0.0]
        elif isinstance(g, cirq.PhaseFlipChannel):
            probs = [0.0, 0.0, g.p]
        elif isinstance(g, cirq.AsymmetricDepolarizingChannel) and n in (1, 2):
            # Letter k of a string acts on op.qubits[k] = targets[k]; the core orders terms the same way.
            letters = "IXYZ"
            probs = [0.0] * (4**n - 1)
            for string, p in g.error_probabilities.items():
                index = 0
                for ch in string:
                    index = index * 4 + letters.index(ch)
                if index:
                    probs[index - 1] += float(p)
        if probs is None:
            kraus = cirq.kraus(op)
            probs = pauli_probabilities(kraus, n)
            if probs is None:
                self.emit("kraus", qs, cond, kraus=kraus)
                return True
        self.emit("pauli_channel", qs, cond, params=probs)
        return True

    def measure(self, op, g, qs, cond):
        if cond:
            raise ExportError("E9003", "a conditional measurement is not supported")
        if g.confusion_map:
            raise ExportError("E9003", "measurement confusion maps are classical noise, which the state machine lacks")
        key = g.mkey
        if key in self.keys:
            raise ExportError("E9003", f"measurement key `{key}` is used twice")
        self.keys[key] = (self.clbits, len(qs))
        self.registers.append({"name": str(key), "offset": self.clbits, "size": len(qs)})
        invert = g.full_invert_mask()
        for j, q in enumerate(qs):
            if invert[j]:
                self.emit("x", [q], None)
            self.emit("measure", [q], None, clbit=self.clbits + j)
            if invert[j]:
                self.emit("x", [q], None)
        self.clbits += len(qs)

    def classically_controlled(self, op, cond):
        c = dict(cond)
        for control in op.classical_controls:
            if not isinstance(control, cirq.KeyCondition):
                raise ExportError("E9003", f"condition {control} is not a single-bit test")
            if control.key not in self.keys:
                raise ExportError("E9001", f"condition on key `{control.key}` before it is measured")
            offset, size = self.keys[control.key]
            if size != 1:
                raise ExportError("E9003", f"key `{control.key}` covers {size} qubits; `key ≠ 0` is not a {{mask, value}} test")
            c[offset] = 1
        inner = op.without_classical_controls()
        for part in cirq.decompose(inner, keep=_keep, on_stuck_raise=None):
            self.op(part, c)

    def controlled(self, controls, values, sub, cond):
        flips = []
        for q, v in zip(controls, values):
            if len(v) != 1 or v[0] not in (0, 1):
                raise ExportError("E9003", f"control values {values} are not a product of single 0/1 values")
            if v[0] == 0:
                flips.append(self.index[q])
        cs = self.qs(controls)
        ts = self.qs(sub.qubits)
        for q in flips:
            self.emit("x", [q], cond)
        g = sub.gate
        n = len(cs)
        if _is(g, cirq.XPowGate, 1):
            self.emit({1: "cnot", 2: "toffoli"}.get(n, "mcx"), ts, cond, controls=cs)
        elif _is(g, cirq.ZPowGate, 1):
            if n == 1:
                self.emit("cz", ts, cond, controls=cs)
            else:
                self.emit("mcz", cs + ts, cond)
        elif _is(g, cirq.ZPowGate):
            if n == 1:
                self.emit("cphase", ts, cond, controls=cs, params=[math.pi * float(g.exponent)])
            else:
                self.emit("mcphase", cs + ts, cond, params=[math.pi * float(g.exponent)])
        elif _is(g, cirq.SwapPowGate, 1) and n == 1:
            self.emit("fredkin", ts, cond, controls=cs)
        elif not ts:
            # A controlled global phase is a phase gate on the controls.
            angle = float(np.angle(cirq.unitary(sub)[0, 0]))
            if n == 1:
                self.emit("phase", cs, cond, params=[angle])
            else:
                self.emit("mcphase", cs, cond, params=[angle])
        else:
            if len(ts) > MAX_DENSE:
                raise ExportError("E9004", f"a controlled gate on {len(ts)} targets; at most {MAX_DENSE}")
            self.emit("unitary", ts, cond, controls=cs, matrix=cirq.unitary(sub))
        for q in flips:
            self.emit("x", [q], cond)
