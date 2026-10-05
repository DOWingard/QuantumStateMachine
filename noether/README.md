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
| W0xxx | lints: `1/2π`, decimal angles near multiples of π/4, shadowing, unused bindings, readouts on one noisy trajectory, `f (x)` with a space |

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
│   └── Cli, Repl, Skills, StdLib, Json, Sha256, Source
├── std/                  standard library modules
├── examples/             self-checking programs, research/ task
└── tests/                unit/, programs/ (goldens), diag/ (fixtures)
```
