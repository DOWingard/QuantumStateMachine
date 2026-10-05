"""Qiskit QuantumCircuit → noether.circuit/1.

Qubit q is `qc.find_bit(q).index` and clbit c is `qc.find_bit(c).index`. Qiskit matrices are
little-endian in their qargs while the core puts targets[0] on the most significant bit, so a dense
operator on qargs (a, b, …) is written with targets reversed.
"""

from __future__ import annotations

import math

import numpy as np
import qiskit
from qiskit.circuit import (
    BreakLoopOp,
    ClassicalRegister,
    Clbit,
    ContinueLoopOp,
    ControlledGate,
    ForLoopOp,
    IfElseOp,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.circuit.classical import expr
from qiskit.quantum_info import Kraus, Operator

from ._common import MAX_DENSE, NEVER, ExportError, document, merge, operation, pauli_probabilities

try:
    from qiskit.circuit import BoxOp
except ImportError:  # Qiskit < 2.0
    BoxOp = None

_SKIP = {"barrier", "delay", "id"}
_FIXED = {"x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"}


def to_circuit(qc: qiskit.QuantumCircuit, params=None) -> dict:
    """The noether.circuit/1 document of `qc`, with `params` bound first."""
    if params:
        qc = qc.assign_parameters(params, inplace=False)
    if qc.parameters:
        names = ", ".join(sorted(p.name for p in qc.parameters))
        raise ExportError("E9005", f"unbound parameters: {names}; pass params= to bind them")
    return _Exporter(qc).run()


def _active_block(op, n, base):
    """The target block of `op` with its controls in their active state, little-endian in the targets.

    Read from the full operator because base_gate can omit part of the gate: CUGate's base is
    U(θ, φ, λ), without the e^{iγ} that the control turns into a relative phase.
    """
    if op.num_qubits <= 12:
        full = Operator(op).data  # qarg k is index bit k; the controls come first
        idx = [op.ctrl_state | (j << n) for j in range(2 ** (op.num_qubits - n))]
        return full[np.ix_(idx, idx)]
    if len(op.params) != len(base.params):
        raise ExportError("E9004", f"{op.name} on {op.num_qubits} qubits is too wide to read its controlled block")
    return Operator(base).data


class _Exporter:
    def __init__(self, qc):
        self.qc = qc
        self.ops = []
        self.notes = []

    def run(self) -> dict:
        qc = self.qc
        phase = float(qc.global_phase)
        if abs(math.remainder(phase, 2 * math.pi)) > 1e-12:
            self.notes.append({"code": "W9001", "message": f"circuit global phase {phase} dropped (unobservable)"})
        self.walk(qc, list(range(qc.num_qubits)), list(range(qc.num_clbits)), {}, "")
        labels = []
        for q in qc.qubits:
            loc = qc.find_bit(q)
            labels.append(f"{loc.registers[0][0].name}[{loc.registers[0][1]}]" if loc.registers else f"q{loc.index}")
        registers = []
        for reg in qc.cregs:
            idx = [qc.find_bit(b).index for b in reg]
            if idx and idx == list(range(idx[0], idx[0] + len(idx))):
                registers.append({"name": reg.name, "offset": idx[0], "size": len(idx)})
        return document(qc.num_qubits, qc.num_clbits, labels, registers, "qiskit", qiskit.__version__, self.ops, self.notes)

    def emit(self, name, targets, condition, **kw):
        self.ops.append(operation(name, targets, condition=condition, **kw))

    def walk(self, circ, qmap, cmap, cond, path):
        for i, ins in enumerate(circ.data):
            op = ins.operation
            where = f"instruction {path}{i} ({op.name})"
            qs = [qmap[circ.find_bit(q).index] for q in ins.qubits]
            cs = [cmap[circ.find_bit(c).index] for c in ins.clbits]
            c = cond
            legacy = getattr(op, "condition", None)  # c_if, Qiskit < 2.0
            if legacy is not None:
                c = merge(c, self.condition(legacy, circ, cmap, where))
                if c is NEVER:
                    self.never(where)
                    continue
            try:
                self.instruction(op, qs, cs, c, circ, cmap, where, f"{path}{i}.")
            except ExportError as e:
                if "instruction " in str(e):
                    raise
                raise ExportError(e.code, f"{where}: {str(e).split(': ', 1)[-1]}") from None

    def never(self, where):
        self.notes.append({"code": "W9002", "message": f"{where}: the condition can never hold; skipped"})

    def instruction(self, op, qs, cs, cond, circ, cmap, where, path):
        name = op.name
        if name in _SKIP:
            return
        if name == "measure":
            if cond:
                raise ExportError("E9003", f"{where}: a conditional measurement is not supported")
            self.emit("measure", [qs[0]], None, clbit=cs[0])
            return
        if name == "reset":
            self.emit("reset", [qs[0]], cond)
            return
        if isinstance(op, IfElseOp):
            self.if_else(op, qs, cs, cond, circ, cmap, where, path)
            return
        if isinstance(op, ForLoopOp):
            indexset, loop_param, body = op.params
            for value in indexset:
                block = body.assign_parameters({loop_param: value}, inplace=False) if loop_param is not None else body
                self.walk(block, qs, cs, cond, path)
            return
        if BoxOp is not None and isinstance(op, BoxOp):
            self.walk(op.blocks[0], qs, cs, cond, path)
            return
        if isinstance(op, (WhileLoopOp, SwitchCaseOp, BreakLoopOp, ContinueLoopOp)) or name == "store":
            raise ExportError("E9003", f"{where}: `{name}` needs run-time classical control the state machine lacks")
        if self.direct(op, qs, cond):
            return
        if isinstance(op, ControlledGate) and self.controlled(op, qs, cond):
            return
        if self.channel(op, qs, cond):
            return
        if op.definition is not None:
            # Recursing keeps Clifford definitions Clifford, so the tableau stays eligible. The
            # definition's global phase is unobservable here: controlled gates never reach this rule.
            self.walk(op.definition, qs, cs, cond, path)
            return
        try:
            m = Operator(op).data
        except Exception as e:
            raise ExportError("E9001", f"{where}: no definition and no matrix ({e})") from None
        if len(qs) > MAX_DENSE:
            raise ExportError("E9004", f"{where}: a dense matrix on {len(qs)} qubits; at most {MAX_DENSE}")
        self.emit("unitary", list(reversed(qs)), cond, matrix=m)

    def direct(self, op, qs, cond) -> bool:
        name = op.name
        if isinstance(op, ControlledGate) and op.ctrl_state != 2**op.num_ctrl_qubits - 1:
            return False
        def p():
            return [float(x) for x in op.params]

        if name in _FIXED:
            self.emit(name, [qs[0]], cond)
        elif name in ("rx", "ry", "rz"):
            self.emit(name, [qs[0]], cond, params=p())
        elif name in ("p", "u1"):
            self.emit("phase", [qs[0]], cond, params=p())
        elif name in ("u", "u3"):
            self.emit("u3", [qs[0]], cond, params=p())
        elif name == "u2":
            self.emit("u3", [qs[0]], cond, params=[math.pi / 2, *p()])
        elif name == "sxdg":
            self.emit("sx", [qs[0]], cond)
            self.emit("x", [qs[0]], cond)
        elif name == "cx":
            self.emit("cnot", [qs[1]], cond, controls=[qs[0]])
        elif name == "cz":
            self.emit("cz", [qs[1]], cond, controls=[qs[0]])
        elif name in ("cp", "cu1"):
            self.emit("cphase", [qs[1]], cond, controls=[qs[0]], params=p())
        elif name == "swap":
            self.emit("swap", qs, cond)
        elif name == "ccx":
            self.emit("toffoli", [qs[2]], cond, controls=qs[:2])
        elif name == "cswap":
            self.emit("fredkin", qs[1:], cond, controls=[qs[0]])
        elif name in ("mcx", "mcx_gray") and len(qs) == op.num_ctrl_qubits + 1:
            self.emit("mcx", [qs[-1]], cond, controls=qs[:-1])
        elif name == "ccz":
            self.emit("mcz", qs, cond)
        elif name == "mcphase":
            self.emit("mcphase", qs, cond, params=p())
        else:
            return False
        return True

    def controlled(self, op, qs, cond) -> bool:
        n = op.num_ctrl_qubits
        base = op.base_gate
        if len(qs) != n + base.num_qubits:
            return False  # ancilla-using variants: fall back to the definition
        controls, targets = qs[:n], qs[n:]
        flips = [controls[k] for k in range(n) if not (op.ctrl_state >> k) & 1]
        for q in flips:
            self.emit("x", [q], cond)
        bname = base.name
        bp = [float(x) for x in base.params] if bname in ("p", "u1") else []
        if bname == "x":
            if n == 1:
                self.emit("cnot", targets, cond, controls=controls)
            elif n == 2:
                self.emit("toffoli", targets, cond, controls=controls)
            else:
                self.emit("mcx", targets, cond, controls=controls)
        elif bname == "z":
            if n == 1:
                self.emit("cz", targets, cond, controls=controls)
            else:
                self.emit("mcz", controls + targets, cond)
        elif bname in ("p", "u1"):
            if n == 1:
                self.emit("cphase", targets, cond, controls=controls, params=bp)
            else:
                self.emit("mcphase", controls + targets, cond, params=bp)
        elif bname == "swap" and n == 1:
            self.emit("fredkin", targets, cond, controls=controls)
        else:
            if len(targets) > MAX_DENSE:
                raise ExportError("E9004", f"a controlled gate on {len(targets)} targets; at most {MAX_DENSE}")
            self.emit("unitary", list(reversed(targets)), cond, controls=controls, matrix=_active_block(op, n, base))
        for q in flips:
            self.emit("x", [q], cond)
        return True

    def channel(self, op, qs, cond) -> bool:
        if op.name == "kraus":
            kraus = [np.asarray(k, dtype=complex) for k in op.params]
        elif hasattr(op, "_quantum_error") or type(op).__name__ == "QuantumChannelInstruction":
            kraus = Kraus(op).data
        else:
            return False
        targets = list(reversed(qs))  # Qiskit's Kraus matrices are little-endian in the qargs
        probs = pauli_probabilities(kraus, len(qs))
        if probs is not None:
            self.emit("pauli_channel", targets, cond, params=probs)
        else:
            self.emit("kraus", targets, cond, kraus=kraus)
        return True

    def if_else(self, op, qs, cs, cond, circ, cmap, where, path):
        c = self.condition(op.condition, circ, cmap, where)
        true_body = op.blocks[0]
        false_body = op.blocks[1] if len(op.blocks) > 1 else None
        if false_body is not None and c is not NEVER and len(c) != 1:
            raise ExportError("E9003", f"{where}: an else branch needs a single-bit condition")
        then = merge(cond, c)
        if then is NEVER:
            self.never(where)
        else:
            self.walk(true_body, qs, cs, then, path)
        if false_body is not None:
            if c is NEVER:
                self.walk(false_body, qs, cs, cond, path)
                return
            negated = {bit: 1 - v for bit, v in c.items()}
            orelse = merge(cond, negated)
            if orelse is NEVER:
                self.never(where)
            else:
                self.walk(false_body, qs, cs, orelse, path)

    def condition(self, c, circ, cmap, where):
        def bit(b):
            return cmap[circ.find_bit(b).index]

        def register(reg, value):
            bits = [bit(b) for b in reg]
            value = int(value)
            if value < 0 or value >> len(bits):
                return NEVER
            return {b: (value >> j) & 1 for j, b in enumerate(bits)}

        if isinstance(c, tuple) and len(c) == 2:
            target, value = c
            if isinstance(target, Clbit):
                return {bit(target): int(bool(value))}
            if isinstance(target, ClassicalRegister):
                return register(target, value)
        if isinstance(c, expr.Expr):
            return self.expr_condition(c, bit, register, where)
        raise ExportError("E9003", f"{where}: unsupported condition {c!r}")

    def expr_condition(self, e, bit, register, where):
        def unwrap(x):
            while isinstance(x, expr.Cast) and x.implicit:
                x = x.operand
            return x

        e = unwrap(e)
        if isinstance(e, expr.Var) and isinstance(e.var, Clbit):
            return {bit(e.var): 1}
        if isinstance(e, expr.Unary) and e.op == expr.Unary.Op.LOGIC_NOT:
            inner = unwrap(e.operand)
            if isinstance(inner, expr.Var) and isinstance(inner.var, Clbit):
                return {bit(inner.var): 0}
        if isinstance(e, expr.Binary):
            if e.op == expr.Binary.Op.LOGIC_AND:
                return merge(self.expr_condition(e.left, bit, register, where), self.expr_condition(e.right, bit, register, where))
            if e.op in (expr.Binary.Op.EQUAL, expr.Binary.Op.NOT_EQUAL):
                left, right = unwrap(e.left), unwrap(e.right)
                if isinstance(left, expr.Value):
                    left, right = right, left
                if isinstance(left, expr.Var) and isinstance(right, expr.Value):
                    equal = e.op == expr.Binary.Op.EQUAL
                    if isinstance(left.var, Clbit):
                        v = int(bool(right.value))
                        return {bit(left.var): v if equal else 1 - v}
                    if isinstance(left.var, ClassicalRegister) and equal:
                        return register(left.var, right.value)
        raise ExportError("E9003", f"{where}: condition {e} is not a single {{mask, value}} test of classical bits")
