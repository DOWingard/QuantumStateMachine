---
name: qsm
description: Write, check, run and debug Noether (.ntr) quantum programs with the noether CLI, which drives the Qputer quantum state machine (state-vector and stabilizer backends). Use when asked to write or simulate a quantum algorithm or circuit, compute expectation values, entanglement or measurement statistics, add noise, optimise a parameterised circuit, check that two circuits are equivalent, or when a .ntr file is involved.
---

# Noether

Noether is a physics-notation quantum scripting language. You may type ASCII or LaTeX; every
compiling command rewrites the file to canonical Unicode first (pass `--no-format` to leave it
alone). Every command prints JSON with `--json`. `reference.md` next to this file holds the
full grammar, token table, built-ins and diagnostic codes (`noether grammar`, `noether tokens`
and `noether explain CODE` print the same material from the binary).

## Workflow (always in this order)

1. Write the program. The first line is `noether 0.1`.
2. `noether check f.ntr --json`. Fix every diagnostic. Each has a `code`, a `span` and often
   `fixes`; apply them with `noether fix f.ntr --apply`. If a cause is unclear, run
   `noether explain <CODE>`.
3. `noether draw f.ntr`. Confirm the time order (left → right) is what you meant.
4. `noether estimate f.ntr --json`. Check `backend`, `estimatedBytes` and `estimatedSeconds`
   before running anything large.
5. `noether run f.ntr --json [--set name=value] [--params params.json] [--seed n]`. Read
   `prints`, `asserts` and `runs`.
6. Encode the expected physics as `assert` statements so the tool checks correctness for you.
   Exit code 3 means an assertion failed.
7. Rewrites: `noether equiv old.ntr new.ntr --json`. Params: `noether opt f.ntr --minimize <label>
   --json` (writes `params.json`); gradients: `noether grad f.ntr --of <label> --json`.

## Syntax (ASCII input; the formatter shows the Unicode form)

```
noether 0.1
let N = 8, theta = pi/4                     # immutable; a name must be bound before it is used
qubits q[N]; bits c[M]; seed 42             # <= 25 qubits state vector; Clifford-only up to 65536
param phi[4] in [-pi, pi] = 0               # optimisable; affine inside rotation angles
prepare |0>^{\otimes N}                     # or |0101>, |+>, (|00> + |11>)/sqrt(2)
H_0   X_{q[0..3]}   Rz(theta)_2   S\dagger_0   CNOT_{0->1}   Toffoli_{0,1->2}
C_{0,1}(Z_2)   C_{not 0}(X_1)   U3(a,b,c)_0   QFT\dagger_q   U^(2^k)_t
exp(-i theta/2 X_0 X_1)   trotter(Ham, t, steps=50, order=2)
let Ham = -J \sum_{j=0}^{N-2} Z_j Z_{j+1} - g \sum_{j=0}^{N-1} X_j
def Name(a)_{x, r[m]}:                      # unitary gate: can be daggered, powered, controlled
proc Name_{r[m]}:                           # may measure, reset, branch, add noise
Name(0.3)_{q[0], q[1..m]}                   # apply: classical args in (), qubits in _{…}
c[0] <- measure Z_0 Z_1                     # ONE parity outcome
c <- measure_q                              # Z on each qubit of q, into c (same size)
reset_q
if c[0] and not c[1]: X_2
for k in 0..N-2:                            # inclusive, unrolled; or a list of integers [0, 3, 5]
flip(0.01)_q   depolarize(1e-3)_0           # channels (Monte Carlo trajectories)
noise:                                      # top level; applies to every op after it, to EOF
    after gate2: depolarize2(1e-3)          # events: gate1 gate2 measure reset (before/after), after prepare
trajectories 2000                           # print/assert average over 2000 trajectories
print \expval{Ham} as energy, |<0101|psi>|^2, entropy(rho_{q[0..3]})
assert \expval{Z_0} ~= 1 +- 1e-9
counts <- run 1000
assert counts["00"] + counts["11"] == 1000
let k = marginal(counts, c)                 # let also binds run-time results
import "std/qft.ntr"                        # quoted path
```

Standard library (`import "std/<file>.ntr"`; registers have r[0] as the most significant bit):

- `qft.ntr`: `QFT_{r[n]}` (Nielsen & Chuang, with the final swaps), `QFTNoSwap_{r[n]}`;
- `grover.ntr`: `MarkOnes_{r[m]}` (phase oracle on |1…1⟩), `Diffuse_{r[m]}`, `GroverStep_{r[m]}`;
- `arith.ntr`: `PhaseAdd(k)_{r[n]}` (Draper adder, needs QFT around it), `Increment_{r[n]}`,
  `Maj_{c,b,a}`, `Uma_{c,b,a}`;
- `gates.ntr`: `Rxx(θ)_{a,b}`, `Ryy(θ)_{a,b}`, `Rzz(θ)_{a,b}`, `Bell_{a,b}`, `GHZ_{r[n]}`,
  `CY_{c→t}`, `ISWAP_{a,b}`.

## Running the state machine

- **Backend.** `backend auto` (default) picks the stabilizer tableau when every op is Clifford
  (or a Pauli channel) and no readout needs amplitudes. Otherwise it picks the state vector,
  which holds at most 25 qubits. The choice and the reason are in `backend` / `backendReason`.
  Force one with `backend statevector|stabilizer` or `--backend`.
- **Live pass.** Statements run top to bottom on one live register. `print` and `assert` read
  the live state and never collapse it. `measure` collapses it, drawing from the seeded RNG.
- **run n.** Replays everything since the last `prepare`, n times, and tallies the classical
  bits. Counts keys list every clbit with c0 leftmost and `|` between registers (`"01|111"`).
  `marginal(counts, reg)` re-keys to one register; a missing key reads as 0.
- **Readout cost.** ⟨P⟩, ⟨Σ cP⟩, |⟨b|ψ⟩|², `fidelity(|0110⟩)` against a basis ket, entropy
  and counts work on both backends. ⟨φ|ψ⟩, `fidelity` against a superposition, ρ_A, `|ψ⟩`
  amplitudes and dense observables need the state vector.
- **Noise.** Channels run as Monte Carlo trajectories: each `run` shot samples them, and the
  live pass samples one branch. Set `trajectories K` so that `print` and `assert` average over
  K trajectories and report `stderr`. `assert a ~= b +- tol` then passes when
  |mean − b| ≤ tol + 2·stderr; each comparison of an `and`/`or` is averaged on its own.
- **Determinism.** The output depends only on the source hash, params, seed and backend. Add
  `--no-timing` for byte-stable JSON. Without a `seed` statement or `--seed`, a seed is drawn
  and reported as `seed`; pass it back with `--seed n` to reproduce the run.
- **Params.** `--set theta=0.3`, `--set theta[2]=0.1` or `--set theta=[0.1,0.2]` (plain
  numbers; ASCII Greek names work) rebind `param`s only, since `let` values are compiled in.
  `--params params.json` loads what `opt` or `eval` wrote.
- **Limits.** `--timeout 60s`, `--max-mem 8G` and `--top k` (which truncates counts and
  amplitudes) apply to every command.

## Traps

- Products act RIGHT TO LEFT: `CNOT_{0->1} H_0` applies H first. Lines run top to bottom.
- Kets, counts keys and bitstrings put q0/c0 LEFTMOST. This is not Qiskit order.
- `_` is always a subscript, so names cannot contain `_`; use camelCase.
- `i`, `e`, `pi`, `psi`, `rho` and the gate names are reserved. Never write `for i in …`; use
  `k` or `j`. Greek words become letters everywhere, labels included: `print x as psi` is
  labelled `ψ` in the JSON.
- Ket labels are a bit string (`|0110⟩`), one of `+ - +i -i`, or a `let`-bound name. Mix them
  with ⊗: `|+⟩ ⊗ |0⟩`, not `|+0⟩`.
- `H` is the Hadamard. Name Hamiltonians `Ham`.
- Ranges `a..b` are inclusive, and empty when b < a.
- `/` binds tighter than juxtaposition: `theta/2 X_0` is (θ/2)X₀, but `1/2pi` is π/2. Write
  `1/(2pi)`.
- The postfix operators `_ ( [ ^ \dagger` must touch their operand. `f (x)` with a space is a
  product, and a touching `(` is a call even after a power: `3p^2(1-p)` calls `p^2`. Write
  `3p^2 (1-p)` or `3p^2·(1-p)`.
- `√X` is one gate, so `√X\dagger` is (√X)†. `H^{\otimes n}_q` is n parallel H gates.
- `CNOT` needs the arrow: `CNOT_{c->t}`. Symmetric gates (`CZ`, `SWAP`, `CP`) take commas.
- Write exact angles (`pi/4`), never `0.785398`. Exact angles drive Clifford detection and the
  T-count.
- `run` needs at least one `measure` since the last `prepare`. There are at most 64 classical
  bits. `print`, `assert`, `run` and `prepare` cannot sit inside a run-time `if`.
- LaTeX commands take ONE backslash: `\dagger`, `\otimes`, `\expval{…}`.

## Rules

- Never guess output: run it and read the JSON.
- Fix diagnostics by code. Do not suppress lints (`# noether: allow W0004`) unless the user
  asks.
- Run `estimate` before anything over 20 qubits or with `trajectories`.
