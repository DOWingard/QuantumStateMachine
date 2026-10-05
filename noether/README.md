# Noether

Noether is a quantum scripting language written in the notation of physics papers: kets,
operator products, subscripts for qubits, `⟨…⟩` for expectation values, Σ and ∏. Programs
(`.ntr` files) compile to the operations of the Qputer quantum state machine
(`qputer_lib`, a state-vector and stabilizer-tableau simulator) and run on it through the
`noether` CLI.

```
noether 0.1
qubits q[3]; bits c[2]; seed 7
let θ = π/3, φ = π/5
prepare (cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩) ⊗ |00⟩
CNOT_{1→2} H_1                  # Bell pair on (1, 2): H acts first
H_0 CNOT_{0→1}                  # Bell-basis rotation: CNOT acts first
c[0] ← measure Z_0
c[1] ← measure Z_1
if c[1]: X_2
if c[0]: Z_2
assert ⟨X_2⟩ ≈ sin(θ) cos(φ)    # the state arrived on qubit 2
assert ⟨Z_2⟩ ≈ cos(θ)
```

You can type ASCII or LaTeX (`CNOT_{1->2}`, `\expval{Z_2}`, `S\dagger_0`, `|0>^{\otimes 3}`).
Every compiling command first rewrites the file into one canonical Unicode form, so all
files read alike.

---

## Contents

- [Building](#building)
- [Command line](#command-line)
- [The language](#the-language)
- [Execution](#execution)
- [Diagnostics](#diagnostics)
- [Tools](#tools)
- [Importing circuits](#importing-circuits)
- [Research loop](#research-loop)
- [Agent skills](#agent-skills)
- [Examples](#examples)
- [Tests](#tests)
- [Files](#files)

---

## Building

Noether is a subdirectory of the Qputer CMake project and links `qputer_lib`. It needs
C++23, and nothing beyond what the simulator needs. From the repository root:

```sh
cmake --preset release && cmake --build --preset release   # builds build/release/noether/noether
ctest --preset release -L noether                          # Noether tests only
```

`QPUTER_BUILD_NOETHER` (default on for a top-level build) switches it on or off. The
standard library (`std/*.ntr`) and the agent skills (`skills/` at the repository root) are
embedded in the binary at build time, so the executable is self-contained.

---

## Command line

```
noether check f.ntr            parse and analyse; report every diagnostic
noether run f.ntr              execute on the quantum state machine
noether fmt f.ntr [-w] [--check] [--ascii]
noether fix f.ntr [--apply]    list or apply machine-applicable fix-its
noether draw f.ntr             circuit diagram, one wire per qubit
noether ir f.ntr               lowered IR (ops, spans, Clifford tags, params)
noether estimate f.ntr         qubits, depth, T-count, memory and time, without running
noether equiv a.ntr b.ntr      semantic equivalence (exact, up to global phase by default)
noether opt f.ntr --minimize LABEL [--method nelder-mead|spsa|adam]
noether grad f.ntr --of LABEL  parameter-shift gradient of a labelled print
noether qasm export f.ntr      OpenQASM 3
noether run f.qasm|f.stim|f.json    import an OpenQASM 2/3, Stim or circuit JSON file and run it
noether check / import …       import only: diagnostics, or the circuit as JSON (below)
noether eval / research …      autonomous research loop (below)
noether explain CODE           long explanation of a diagnostic
noether grammar | tokens       EBNF grammar; token and alias table
noether skill [--install DIR]  print or install the agent skills
noether repl                   interactive session
```

Shared options: `--json`, `--no-timing` (byte-stable JSON), `--timeout 60s`,
`--max-mem 8G`, `--top k`, `--seed n`, `--set name=value`, `--params params.json`,
`--backend auto|statevector|stabilizer`, `--deny warnings`, `--no-format`.

With `--json`, every command prints one document carrying a `schema` field
(`noether.run/1`, `noether.check/1`, …).

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | compile error (or a failed candidate in `eval`) |
| 2 | usage error (bad flag, missing file) |
| 3 | an `assert` failed |
| 4 | resource limit or timeout |
| 5 | internal error |
| 6 | a research spec was modified after its lock (E8001) |

---

## The language

### Statements

```
noether 0.1                                  # header, first line
let N = 8, J = 1.0                           # immutable bindings, used after they are bound
qubits q[N]; bits c[N]; seed 42              # registers; ';' separates statements on a line
param θ[4] ∈ [-π, π] = 0                     # optimisable, affine inside rotation angles
backend stabilizer                           # or statevector; default auto
import "std/qft.ntr"
prepare |0⟩^⊗N                               # or |0101⟩, |+⟩ ⊗ |−i⟩, (|00⟩ + |11⟩)/√2
H_0  X_{q[0..3]}  Rz(θ[0])_2  S†_0  CNOT_{0→1}  Toffoli_{0,1→2}  C_{0, ¬1}(Z_2)
exp(-i θ[1]/2 X_0 X_1)                       # Pauli exponentials
let Ham = -J Σ_{j=0}^{N-2} Z_j Z_{j+1} - Σ_{j=0}^{N-1} X_j
trotter(Ham, 2.0, steps=100, order=2)
def Layer(t)_{r[n]}:                         # unitary: can be daggered, powered, controlled
    for j ∈ 0..n-1: Rx(t)_{r[j]}
proc Correct_{r[3]}:                         # may measure, reset, branch and add noise
    …
Layer(0.3)_q   Layer(0.3)†_q   C_{a}(Layer(0.3)_q)
c[0] ← measure Z_0 Z_1                       # one parity outcome
c ← measure_q                                # Z on every qubit of q
reset_q
if c[0] and ¬c[1]: X_2
else: Z_2
for k ∈ 0..N-2: CNOT_{k→k+1}                 # inclusive ranges, unrolled; also [0, 3, 5] and `by`
flip(0.01)_q   depolarize(1e-3)_0            # channels
noise:                                       # rules applied to every later op
    after gate2: depolarize2(1e-3)
trajectories 2000                            # readouts average over trajectories
print ⟨Ham⟩ as energy, |⟨0101|ψ⟩|^2, entropy(ρ_{q[0..3]})
assert ⟨Z_0⟩ ≈ 1 ± 1e-9
counts ← run 1000
assert counts["00"] + counts["11"] == 1000
```

### Operators and conventions

- **Time order.** A product acts right to left, as in the algebra: `CNOT_{0→1} H_0` applies H
  first. Lines run top to bottom. `∏_{j=a}^{b} U_j = U_b ⋯ U_a`, so index order is time order.
- **Bit order.** Qubit q is bit q of a basis index. Kets, counts keys and printed bitstrings
  put q0 (c0) leftmost: `|10⟩` has qubit 0 set. For `U_{a,b}` with a written matrix U, a is
  the most significant bit of U's row index.
- **Gates.** `I X Y Z H S T`, `√X`, `Rx Ry Rz P U3`, `CNOT CZ CP SWAP Toffoli Fredkin`, and
  `C_{controls}(U)` with `¬c` for a negative control. Aliases (`CX`, `Sdg`, `SX`, `CCX`, …)
  canonicalize to these. Any unitary matrix literal `[[…]]` is a gate on the qubits it is
  subscripted with.
- **Roles.** `CNOT_{c→t}`, `Toffoli_{c,c→t}` and `Fredkin_{c→a,b}` need the arrow; the
  symmetric `CZ`, `SWAP` and `CP` take commas. A register in a single-qubit slot broadcasts.
- **Postfix binds tightest.** `_ ( [ ^ ^⊗ †` must touch their operand. `√X` is one gate, so
  `√X†` is (√X)†. `G^⊗n` of a one-qubit gate is n parallel copies.
- **Arithmetic.** `/` binds tighter than juxtaposition: `θ/2 X_0` is (θ/2)X₀ and `1/2π` is π/2.
  `^` is right-associative and `-2^2 = -4`. Angles are radians; time evolution uses ħ = 1.
- **Reserved names.** `i`, `e`, `π`, the gate names, `ψ` (the live state) and `ρ` (reduced
  states). Greek words spell their letters everywhere (`theta` is θ). `_` always means a
  subscript, so names are camelCase.

### Canonical form

The formatter is a pure function of the syntax tree. It writes one spelling per meaning
(`≈`, `→`, `←`, `†`, `⊗`, `⟨⟩`, kets in Unicode, `|x|` for `abs`, `e^{x}` for `exp` of a
number) and is idempotent. A rewrite that would change the tree is refused. `fmt --ascii`
prints the ASCII form, which formats back to the same file. `--no-format` leaves a file
untouched, and so does a file that does not parse.

LaTeX commands take one backslash (`\dagger`, `\otimes`, `\expval{…}`). A doubled backslash
is one E1004 with a fix-it.

---

## Execution

**Backends.** `backend auto` picks the stabilizer tableau when every operation is Clifford
(including Clifford angles such as `Rz(π/2)`) or a Pauli channel, and no readout needs
amplitudes. Otherwise it picks the state vector (≤ 25 qubits). The tableau handles up to
65 536 qubits in N²/2 bytes. The JSON reports `backend` and `backendReason`.

**Live pass.** Statements run once, top to bottom, on one live register. `print` and
`assert` read it without collapsing it. `measure` collapses it with the seeded RNG.

**`run n`** replays everything since the last `prepare` n times and tallies the classical
bits. Counts keys list every clbit, c0 leftmost, with `|` between registers (`"01|111"`).
`marginal(counts, reg)` re-keys them to one register, and a missing key reads as 0.

**Readouts.**

| Readout | Backends |
|---|---|
| `⟨P⟩`, `⟨Σ c_k P_k⟩` for Pauli strings P | both |
| `\|⟨b\|ψ⟩\|²`, `fidelity(\|b⟩)` for a basis ket b | both |
| `entropy(ρ_A)` (von Neumann, bits) | both; exact rank formula on the tableau |
| `counts`, `marginal(counts, reg)` | both |
| `⟨φ\|ψ⟩`, `fidelity` against a superposition, `ρ_A`, `\|ψ⟩` amplitudes, `⟨M⟩` for a dense M | state vector |

`count(G)` and `depth()` read structural metrics of the circuit recorded so far.

**Noise.** Channels (`depolarize`, `depolarize2`, `dephase`, `flip`, `pauli`, `ampdamp`,
`kraus`) are Monte Carlo trajectories: each `run` shot samples them, and the live pass
samples one branch. With `trajectories K`, `print` and `assert` average over K
trajectories and report the standard error, and `assert a ≈ b ± tol` passes when
|mean − b| ≤ tol + 2·stderr. Each comparison of an `and`/`or` is averaged separately.

**Determinism.** Output depends only on the source hash, params, seed and backend. A
program without `seed` draws one and reports it, and `--seed n` replays the run.

---

## Diagnostics

Every diagnostic has a code, a severity, a source span, notes and often machine-applicable
fix-its. `noether explain CODE` prints the long form, with a wrong and a right example.

| Range | Area |
|---|---|
| E1xxx | lexing: invalid UTF-8, tabs, confusable identifiers, unknown LaTeX commands |
| E2xxx | syntax: unexpected tokens, header, `±` without `≈`, gate role separators, Qiskit-style `qc.cx(…)` |
| E3xxx | names: unknown, reserved, rebound, used before definition, `_` inside a name |
| E4xxx | types and stages: mismatches, non-integer indices, run-time values at compile time, non-affine params |
| E5xxx | quantum semantics: repeated or overlapping qubits, non-unitary or non-Hermitian operators, unnormalised kets, `run` without `measure` |
| E6xxx | backends and resources: operations or readouts a backend cannot run, qubit and memory limits |
| E7xxx | run time: assertion failed, timeout, numerical drift |
| E8xxx | research specs: lock mismatch, forbidden statements or gates, basis, coupling, signature |
| E9xxx | imports: unsupported or malformed input, unknown gates, constructs the state machine cannot represent, resource limits, unbound inputs, argument mismatches, malformed circuit JSON |
| W0xxx | lints: `1/2π`, decimal angles near multiples of π/4, shadowing, unused bindings, readouts on one noisy trajectory, `f (x)` with a space |
| W9xxx | imports: a global phase dropped, a statement with no simulated effect skipped, declared qubits never used |

A comment `# noether: allow W0004` suppresses that lint in its file; `--deny warnings` turns lints
into errors.

---

## Tools

- **`draw`**: a text circuit, one wire per qubit, left to right in time.
- **`estimate`**: qubits, op counts, 2-qubit count, depth, T-count and T-depth, Clifford
  status, peak memory and predicted time; E6004 if over `--max-mem`.
- **`equiv`**: compares the operations of two programs: full unitaries up to 12 qubits,
  the Choi state on the tableau for Clifford circuits up to 4096 qubits, and random input
  states otherwise (up to 25 qubits). Global phase is ignored unless `--phase exact`.
- **`opt`**: minimises (or `--maximize`s) a labelled `print` over the `param`s with
  Nelder–Mead, SPSA or Adam, and writes `params.json` (`noether.params/1`), which
  `--params` loads back.
- **`grad`**: exact parameter-shift gradients, `∂f/∂θ = [f(θ + π/2) − f(θ − π/2)]/2` per
  rotation, falling back to finite differences where a parameter enters non-linearly.
- **`qasm export`**: OpenQASM 3 for circuits without Noether-only constructs.
- **`repl`**: accepted lines accumulate into one program, so bindings and the state carry
  over; a line with an error is reported and dropped, and `:source` prints the session.

---

## Importing circuits

`noether run`, `check` and `import` also take circuits written for other tools. The format
comes from the extension, or from `--format qasm|qasm2|qasm3|stim|circuit`:

| Extension | Format |
|---|---|
| `.qasm` | OpenQASM 2 or 3, from the `OPENQASM` header (3 without one) |
| `.qasm2`, `.qasm3` | OpenQASM 2 / 3 |
| `.stim` | Stim |
| `.json` | `noether.circuit/1`, written by the Python package from Qiskit and Cirq circuits |

```sh
noether run bell.qasm --shots 4000 --seed 1
noether run teleport.qasm --param theta=1.1          # values for OpenQASM 3 `input`s
noether run qft.qasm --emit statevector              # or probabilities: the state before the final measurements
noether run rep_code.stim --shots 100000 --json      # with detector and observable rates
noether import circuit.qasm > circuit.json           # the imported operations as noether.circuit/1
```

Every importer lowers to one flat list of state-machine operations. Gates map to native
operations where one exists and to dense controlled unitaries otherwise (at most 10 targets);
the lowering is exact, global phase aside, so `--emit statevector` agrees with the source
framework's simulator to rounding error.

**Conventions.** Qubit q is bit q of an amplitude index, so `amplitudes` and `probabilities`
are little-endian (`"basisOrder": "q0-lsb"`, Qiskit's order). Counts are keyed by bit strings
with classical bit 0 leftmost (`"bitOrder": "c0-left"`); the Python package re-keys them to each
framework's convention. Classical bits are one 64-bit word, so a circuit has at most 64.

**Backend.** `--backend auto` (the default) takes the stabilizer tableau when every operation is
Clifford — rotations by multiples of π/2 and controlled phases by multiples of π count, rewritten
exactly up to global phase — and the state vector (at most 25 qubits) otherwise. When every
measurement is terminal the state is simulated once and the shots are sampled from it; mid-circuit
measurement, feed-forward, reset and noise replay the circuit once per shot.

**OpenQASM 2 and 3.** Supported: `qreg`/`creg` and `qubit`/`bit` registers, physical qubits
`$n`, `const` and `input` (bound with `--param`), gate definitions, the modifiers `ctrl`,
`negctrl`, `inv` and `pow`, every measure form, `reset`, `if`/`else` on bits and registers, `for`
over constant ranges and sets, and `box`. `qelib1.inc` and `stdgates.inc` are built in; other
includes resolve relative to the including file. `barrier` is skipped silently; `delay`,
`duration`, `stretch`, pragmas and annotations are skipped with W9002. Subroutines (`def`),
`while`, `switch`, run-time classical variables and arithmetic, `extern` and pulse-level
calibration are rejected with E9001. A condition must reduce to one mask-and-value test of the
classical bits — `c[0]`, `!c[0]`, `c == 5`, and `&&` of those — and `else` needs a single-bit
condition.

OpenQASM 3 defines `U(θ, φ, λ) = e^{iθ/2}·u3(θ, φ, λ)`, which matters under `ctrl @`. With that
definition every `stdgates.inc` gate equals its definition exactly, except `CX`: defined as
`ctrl @ U(π, 0, π)`, which is controlled-(iX); it runs as CNOT, as every producer intends.
OpenQASM 2's `U` is `u3`, and `qelib1.inc` fixes its gates up to global phase.

**Stim.** Every gate, noise channel and annotation of Stim 1.16 except `MPAD` and the heralded
errors. Qubit ids are compacted in ascending order and labelled `q<id>`; the k-th measurement
result is classical bit k. Cliffords stay Clifford, so Stim circuits run on the tableau at any
size. Pauli noise becomes `pauli_channel` (Stim's term order); an `E`/`ELSE_CORRELATED_ERROR`
chain becomes one channel with term probabilities qⱼ = pⱼ ∏ᵢ<ⱼ (1 − pᵢ) — a Pauli channel on
up to 2 qubits, a Kraus channel on up to 10. `MPP` and `SPP` are a basis change, a CNOT chain
and a Z measurement or S gate, undone afterwards. A noisy result (`M(p)`) is an X error before
the measurement, which is exact only if the qubit is idle afterwards; otherwise E9003.
`REPEAT` blocks are unrolled (at most 10⁷ instructions). Detectors and observables are
reported as Stim reports them: the shots in which their parity differs from a noiseless
reference sample. `sweep[k]` controls read 0, Stim's default, with W9002.

**JSON output.** `run --json` prints `noether.import-run/1`: `format`, `sourceSha256`,
`numQubits`, `numClbits`, `qubitLabels`, `clbitRegisters` (`name`, `offset`, `size`),
`backend`, `backendReason`, `method` (`sampled`, `trajectories` or `state`), `seed`, then
`shots`, `bitOrder` and `counts` (with `detectors` and `observables`, each `{index, fires|flips,
rate}`), or `basisOrder` with `amplitudes` (`[re, im]` pairs) or `probabilities`; `timing`
unless `--no-timing`. A run without `--seed` draws one below 2⁵³ and reports it.

| Limit | Value | Code |
|---|---|---|
| state-vector qubits | 25 | E9004, before allocating |
| tableau qubits | 65 536 | E9004 |
| classical bits (measurement results) | 64 | E9004 |
| dense gate targets | 10 | E9004 |
| operations after inlining and unrolling | 10⁷ | E9004 |
| integer `pow` repeated before a dense power | 4096 operations | — |

### From Python

`noether/python` is the `noether-interop` package. It converts Qiskit and Cirq circuits to
`noether.circuit/1`, hands Stim circuits over as text, runs them through the CLI and returns
results in the source framework's terms.

```sh
pip install './noether/python[qiskit,cirq,stim]'     # or: uv pip install …
export NOETHER_BIN=build/release/noether/noether     # else `noether` on PATH
```

```python
import noether_interop as ni

ni.run(qiskit_circuit, shots=1000, seed=1)["counts"]    # {"1 01": 1000}: Qiskit's keys
ni.run(cirq_circuit, shots=1000)["counts"]["m"]         # {value: n} per measurement key
ni.run(stim_circuit, shots=1000)["records"]             # shots × measurements, plus "detectors"
ni.statevector(circuit)                                 # little-endian amplitudes
ni.to_circuit(circuit)                                  # the noether.circuit/1 document
```

Qiskit: standard gates, controlled gates with any control state, `UnitaryGate`, `Kraus` and Aer
noise instructions, `if_test` on a bit or register, `for_loop`, `box`, `reset`, parameters bound
with `params=`. Cirq: every gate with a unitary (exact maps for the common ones, dense matrices
otherwise), controlled operations with 0/1 control values, measurement keys with invert masks,
classical control on single-bit keys, and noise channels. What cannot be represented raises
`ExportError` with an E9xxx code; the CLI's rejections raise `NoetherError` with its
diagnostics.

---

## Research loop

A task spec states what to search for. The agent edits only the candidate and the CLI
scores it:

```
task "toffoli-tdepth"
qubits q[3]
candidate def Toffoli3_{q[3]}
target unitary Toffoli_{q[0], q[1]→q[2]}
gates {H, S, S†, T, T†, CNOT}
coupling all
require fidelity ≥ 0.999999
minimize tdepth
minimize tcount
goal tdepth ≤ 3
budget experiments=40, wall=30m
```

- `noether check algo.ntr --spec s.ntr` lowers the candidate into `gates` and checks the
  signature, the forbidden statements and the `coupling`.
- `noether eval --spec s.ntr --candidate algo.ntr` fits the candidate's params, then reports
  feasibility, the metrics (`fidelity`, `count2q`, `depth`, `depth2q`, `nops`, `nparams`,
  `tcount`, `tdepth`) and whether the goal is met. For a unitary target, fidelity is the
  average gate fidelity up to global phase. Objectives compare lexicographically, in file
  order.
- `noether research init TAG --spec s.ntr [--from start.ntr]` creates a workspace: the frozen
  spec and its SHA-256 lock, the candidate, a ledger (`results.tsv`), notes and a brief.
  `eval --dir D --record "hypothesis"` appends a ledger row with a verdict (`improved`,
  `equal`, `worse`, `infeasible`, `error`). `research status` reports the best row and
  whether to stop; `research report` renders Markdown and evaluates holdout instances.
- Editing the spec after `init` fails every later `eval` with E8001 (exit 6) and records
  nothing.

---

## Agent skills

Three skills for coding agents live in `skills/` at the repository root. They are embedded
in the binary, and `noether skill --install DIR` writes them into any skills directory:

| Skill | Purpose |
|---|---|
| `qsm` | Writing, checking, running and debugging Noether programs: workflow, syntax, backend choice, readout costs, the common traps. `reference.md` beside it holds the grammar, tokens, built-ins and diagnostic codes; it is generated from the binary, and a test fails when it is stale (`NOETHER_UPDATE_GOLDENS=1` rewrites it). |
| `qsm-research` | Running the research loop: one hypothesis per experiment, keep improvements in git, revert the rest, stop at the goal or budget. |
| `qsm-tune` | Tuning the simulator for the host: fingerprint, baseline, interleaved A/B measurement, one change at a time, results proven unchanged. |

`noether skill --full` prints the `qsm` skill with its reference.

---

## Examples

Every example checks its own result with `assert`s, and the test suite runs all of them.

| File | Shows |
|---|---|
| `teleport.ntr` | teleportation with feed-forward corrections |
| `superdense.ntr` | two bits through one qubit of a Bell pair |
| `entanglement_swapping.ntr` | qubits that never interact end up entangled |
| `chsh.ntr` | the CHSH value 2√2 and the CHSH game |
| `deutsch_jozsa.ntr`, `bernstein_vazirani.ntr` | one-query oracle algorithms (a 40-bit hidden string) |
| `grover.ntr` | search among 64 items, success probability against sin²((2k+1)·asin(1/8)) |
| `phase_estimation.ntr` | QPE with the standard-library QFT |
| `vqe.ntr` | a 12-parameter ansatz for the 4-site transverse-field Ising chain; asserts the variational bounds, and `opt` lowers the energy towards E₀ |
| `tfim_quench.ntr` | Trotterised quench; ⟨H⟩ and ⟨H²⟩ conserved up to the Trotter error |
| `ghz_large.ntr` | a 10 000-qubit GHZ state on the tableau |
| `noise_channels.ntr` | every channel against its analytic decay law |
| `repetition_code.ntr` | a bit-flip code with a measuring `proc`, logical error rate 3p²(1−p) + p³ |
| `research/` | the Toffoli T-depth task spec and a T-depth-3 candidate |
| `interop/bell.qasm`, `interop/teleport.qasm` | OpenQASM 3 on the tableau, and teleportation with `input` and feed-forward |
| `interop/rep_code.stim` | a noisy Stim repetition code with detectors and an observable |
| `interop/qiskit_demo.py`, `interop/cirq_demo.py` | Qiskit and Cirq circuits through the Python package, checked against their own simulators |

```sh
build/release/noether/noether run noether/examples/grover.ntr
```

The standard library: `std/qft.ntr` (`QFT`, `QFTNoSwap`), `std/grover.ntr` (`MarkOnes`,
`Diffuse`, `GroverStep`), `std/arith.ntr` (`PhaseAdd`, `Increment`, `Maj`, `Uma`) and
`std/gates.ntr` (`Rxx`, `Ryy`, `Rzz`, `Bell`, `GHZ`, `CY`, `ISWAP`). Registers put r[0] as
the most significant bit.

---

## Tests

`noether_tests` (GoogleTest, label `noether`):

| File | Covers |
|---|---|
| `unit/test_lexer.cpp` | Every spelling in the token table (canonical, ASCII, Unicode alternative, LaTeX with one backslash) lexes to its token; a doubled backslash is one diagnostic; Dirac forms; Greek words. |
| `unit/test_parser.cpp` | Precedence, time order, postfix adjacency, one-line suites, error recovery. |
| `unit/test_formatter.cpp` | Canonical spellings; on every program, example and standard-library module: idempotence, tree preservation, ASCII ↔ Unicode round trip, comments kept; auto-format on run. |
| `unit/test_operators.cpp` | Every gate and alias: canonical spelling, action on a generic 3-qubit state against its matrix, Clifford gates against the tableau on all 63 Pauli expectations. Every punctuation token in every spelling transpiles and runs; every function, channel and keyword drives the state machine as specified; readouts return their exact values. |
| `unit/test_golden.cpp` | Seven spec programs against checked-in `.expected.json` (numbers to 1e-9 relative), determinism, canonical form; every example runs warning-free with all asserts passing. |
| `unit/test_diagnostics.cpp` | One fixture per code in `tests/diag/` triggers exactly that code at its span; fix-its remove their diagnostic; every catalog entry has a fixture and an explanation. |
| `unit/test_tools.cpp` | Decomposition into 8 gate bases, equivalence, optimisers and gradients, estimates, QASM, draw/IR schemas, skills, REPL, exit codes. |
| `unit/test_research.cpp` | Spec checks, metrics, verdicts, the ledger, the spec lock, coupling violations. |
| `interop/test_interop_gates.cpp` | Every Stim unitary gate and alias, every Qiskit standard gate (as circuit JSON and as its OpenQASM 2/3 dumps) and a set of Cirq gates against reference unitaries in `interop/fixtures/gates.json`; built-in include files against their own definitions. |
| `interop/test_interop_lowering.cpp` | Control, inverse and power against block, adjoint and root identities; emitted operations against the algebra; Pauli-product measurement and rotation; Clifford rewrites; operation validation. |
| `interop/test_interop_formats.cpp` | Golden programs in `interop/programs/` with physical checks (Bell, GHZ, teleportation, QFT, Grover, a noisy repetition code); Stim and OpenQASM semantics; circuit JSON round trips; limits; the interop examples. |
| `interop/test_interop_runner.cpp` | Backend choice, sampled against replayed shots, state readout, seeds, refusals before allocation. |
| `interop/test_interop_fuzz.cpp` | Mutated OpenQASM, Stim and JSON sources end in a circuit or diagnostics, never a crash. |

`interop/fixtures/generate.py` regenerates the reference unitaries (needs Qiskit, Cirq and Stim).
The Python package has its own suite, which drives the CLI against Qiskit, Qiskit Aer, Cirq and
Stim: random circuits to fidelity 1 − 10⁻¹⁰, OpenQASM routes against the JSON route, Stim noise
term order, and sampled distributions, detector and observable rates within 4σ of the
frameworks' own samplers:

```sh
pip install -e './noether/python[test]'
NOETHER_BIN=build/release/noether/noether pytest noether/python
```

Golden outputs are rewritten with `NOETHER_UPDATE_GOLDENS=1` after a deliberate change;
review the diff before keeping it.

---

## Files

```
noether/
├── CMakeLists.txt        noether_lib, the noether CLI, embedded std library and skills
├── apps/main.cpp         CLI entry point
├── cmake/embed.cmake     embeds std/ and the repository's skills/ into the binary
├── src/
│   ├── tokens.def        the token table: every spelling, alias, keyword and gate
│   ├── Lexer, Parser     tokens and syntax tree (Ast.hpp)
│   ├── Compiler, Eval    binding-time analysis, values, lowering to IR (Ir.hpp, Values.hpp)
│   ├── Backend, Lowering backend choice, Clifford tagging, IR → state-machine operations
│   ├── Executor          live pass, readouts, trajectories, run, JSON results
│   ├── Formatter         canonical form
│   ├── Diagnostics       codes, catalog, rendering, fix-its
│   ├── Estimate, Draw, Equiv, Optimize, Decompose, Qasm   tools
│   ├── Research          specs, metrics, eval, workspaces
│   ├── Cli, Repl, Skills, StdLib, Json, Sha256, Source
│   └── interop/          importers (noether_interop): Qasm, Stim, CircuitJson, the gate algebra
│                         (Lowering), Run, ImportCli, and the built-in qelib1.inc / stdgates.inc
├── python/               the noether-interop package (Qiskit, Cirq and Stim front ends) and its tests
├── std/                  standard library modules
├── examples/             self-checking programs, research/ task, interop/ examples
└── tests/                unit/, programs/ (goldens), diag/ (fixtures), interop/
```
