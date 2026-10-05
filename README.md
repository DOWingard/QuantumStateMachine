# Quantum State Machine

A C++23 quantum simulator with two backends behind one API:

- **State vector** (N ≤ 25): stores the full register |ψ⟩ ∈ ℂ^(2^N), runs every gate in
  place with OpenMP-parallel kernels.
- **Stabilizer tableau** (N ≤ 65 536): runs Clifford circuits (H, S, CNOT, Paulis, …) in
  O(N) per gate and O(N²/64) per measurement, using N²/2 bytes.

Registers above 25 qubits use the tableau automatically. Smaller registers can opt into
it. Both backends handle measurement, classical feed-forward, Pauli noise channels, and
multi-shot circuit execution. Seeded runs give bitwise-identical results for any thread count, and for a
Clifford circuit the same seed gives the same outcomes on either backend.

```cpp
#include <QuantumStateMachine.hpp>

Qputer::QuantumStateMachine m{2, 2, /*seed=*/2026};   // 2 qubits, 2 classical bits
m.h(0).cnot(0, 1);                                     // Bell pair (|00> + |11>)/√2
double zz = m.expectation("ZZ", {0, 1});               // +1, state unchanged
m.measure_all();                                       // collapses, recorded in the circuit
Qputer::Counts counts = m.run(10000);                  // {00: ~5000, 11: ~5000}

Qputer::QuantumStateMachine big{1000, 2};              // > 25 qubits: stabilizer backend
big.h(0);
for (Qputer::Qubit q = 1; q < 1000; ++q) big.cnot(q - 1, q); // 1000-qubit GHZ
big.t(0);                                              // throws: T is not Clifford
```

The same machine can be driven from **Noether**, a physics-notation scripting language
(`.ntr` files) with its own CLI, formatter and diagnostics, built in `noether/`, with agent
skills for it in `skills/`:

```
noether 0.1
qubits q[2]; bits c[2]; seed 2026
CNOT_{0→1} H_0                 # products act right to left: H first
assert ⟨Z_0 Z_1⟩ ≈ 1
c ← measure_q
counts ← run 10000
```

---

## Contents

- [Building](#building)
- [Conventions](#conventions)
- [Architecture](#architecture)
- [API overview](#api-overview)
- [Writing your own algorithms](#writing-your-own-algorithms)
- [Execution model](#execution-model)
- [Stabilizer backend](#stabilizer-backend)
- [Performance design](#performance-design)
- [Determinism](#determinism)
- [Limits and error handling](#limits-and-error-handling)
- [Tests](#tests)
- [Noether](#noether)
- [File reference](#file-reference)

---

## Building

**Requirements:** CMake ≥ 3.25, a C++23 compiler (GCC or Clang), and optionally OpenMP.
CMake's `FetchContent` downloads [Eigen 3.4.0](https://eigen.tuxfamily.org) and
[GoogleTest 1.15.2](https://github.com/google/googletest). An installed GoogleTest is
used if one is found.

The `Makefile` is a thin wrapper over `CMakePresets.json`. Each build type gets its own
tree under `build/<preset>/`.

```sh
make                       # configure + build (Release by default)
make test                  # build, then run the GoogleTest suite and the examples through ctest
make run                   # build, then run the demo app (build/release/qputer)
make examples              # build, then run the three example algorithms
make BUILD_TYPE=Debug test # Debug build with AddressSanitizer + UBSan
make clean                 # clean the current build tree
make distclean             # rm -rf build/
```

You can also call CMake directly:

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
```

| CMake option         | Default            | Effect                                                    |
|----------------------|--------------------|-----------------------------------------------------------|
| `QPUTER_NATIVE`      | `ON`               | Adds `-march=native` (PUBLIC, so every TU agrees on Eigen alignment). |
| `QPUTER_BUILD_TESTS` | top-level project  | Builds `qputer_tests`.                                    |
| `QPUTER_BUILD_EXAMPLES` | top-level project | Builds the programs in `examples/` (registered with ctest when tests are built). |
| `QPUTER_BUILD_NOETHER` | top-level project | Builds the Noether language library, the `noether` CLI and (with tests) `noether_tests`. |

Every first-party target is built with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow
-Wnon-virtual-dtor -Wold-style-cast -Werror`. Debug builds also add
`-fsanitize=address,undefined`. The library is compiled as position-independent code,
so it can be linked into a shared module. Without OpenMP, the build prints a warning and
every kernel runs single-threaded.

### Build targets

| Target          | Kind           | Contents                                                    |
|-----------------|----------------|-------------------------------------------------------------|
| `qputer_lib`    | static library | State vector, gate kernels, readout kernels, stabilizer tableau, state machine. |
| `qputer`        | executable     | Demo: Bell pair, 5-qubit GHZ marginal, teleportation, 1000-qubit GHZ. |
| `qputer_tests`  | executable     | GoogleTest suite (131 tests).                               |
| `noether_lib`, `noether` | static library, executable | The Noether compiler, executor and tools; the `noether` CLI (`build/<preset>/noether/noether`). |
| `noether_tests` | executable     | Noether GoogleTest suite (253 test cases, many parameterised over programs and fixtures). |
| `phase_estimation`, `variational`, `repetition_code` | executables | Example algorithms in `build/<preset>/examples/`; see [Writing your own algorithms](#writing-your-own-algorithms). |

### Demo output (`make run`)

```
Bell state probabilities:
  |00>  p = 0.500000
  |11>  p = 0.500000
  <ZZ> = +1.000000  <XX> = +1.000000  <YY> = -1.000000
Bell state, 10000 shots:
  00  5070
  11  4930
GHZ marginal over (q0, q4): [0.500, 0.000, 0.000, 0.500]
Teleportation, 10000 shots (bit 2 must be 0):
  000  2499
  001  2588
  010  2432
  011  2481
GHZ on 1000 qubits (stabilizer backend): <Z0 Z999> = +1.0  <Z500> = +0.0
GHZ-1000 qubits (0, 500, 999), 10000 shots:
  000  4931
  111  5069
Seed 2026 reproduces every result above.
```

---

## Conventions

These conventions apply across the whole library.

**Qubit ordering (little-endian).** Qubit q is bit q of the amplitude index:

```
|ψ⟩ = Σ_i a_i |i⟩,   i = Σ_q b_q · 2^q,   b_q = state of qubit q
```

**Dense matrix target ordering (big-endian within U).** For a 2^M × 2^M matrix U applied
to targets (t₀, …, t_{M−1}), t₀ is the *most* significant bit of U's row/column index.
So `unitary({a, b}, kron(A, B))` applies A to qubit a and B to qubit b.

**Controls** are active-high. U acts only on the subspace where every control is |1⟩.

**Readout ordering.** A readout over qubits (q₀, …, q_{k−1}) reports outcome o, where
bit j of o is the value of q_j. Reading out (0, …, N−1) therefore gives the basis index
itself.

**Classical register.** Clbit c is bit c of `classical_register()` and of every `Counts`
key. `bitstring()` prints clbit 0 rightmost, so `bitstring(0b011, 3) == "011"`.

**Stabilizer strings.** `stabilizers()` prints each generator as a sign followed by one
letter per qubit, qubit 0 first: `"+XZI"` is +X₀Z₁. This is the opposite direction from
`bitstring()`.

**Gate definitions:**

| Gate              | Matrix / action                                         |
|-------------------|---------------------------------------------------------|
| `rx/ry/rz(θ)`     | exp(−iθσ/2)                                             |
| `phase(λ)`        | diag(1, e^{iλ})                                         |
| `u3(θ, φ, λ)`     | [[cos θ/2, −e^{iλ} sin θ/2], [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]] |
| `sx`              | √X = ½[[1+i, 1−i], [1−i, 1+i]]                          |
| `cz`, `cphase(λ)` | diag(1, 1, 1, −1) / diag(1, 1, 1, e^{iλ}), symmetric in both qubits |
| `mcz`, `mcphase`  | −1 / e^{iλ} on \|1…1⟩ over the listed qubits             |
| `fredkin(c, a, b)`| controlled SWAP                                         |

---

## Architecture

There are three layers. Each layer adds state and validation on top of the one below it.
The state machine holds one of two registers (`std::variant`) and dispatches every
operation to the backend that holds it.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ QuantumStateMachine                    include/QuantumStateMachine.hpp          │
│  live register (state vector | tableau) · classical register · seeded RNG ·     │
│  recorded circuit · prepare · gates · measure/reset · readout · run(shots)      │
└────────┬──────────────────────────┬──────────────────────────┬──────────────────┘
         │ state-vector backend     │                          │ stabilizer backend
┌────────▼─────────────────┐ ┌──────▼───────────────────┐ ┌────▼─────────────────────┐
│ QuantumGate (static)     │ │ ReadoutKernels (detail)  │ │ StabilizerState          │
│ include/QuantumGates.hpp │ │ src/ReadoutKernels.hpp   │ │ include/StabilizerState  │
│ in-place unitary kernels │ │ |a_i|², marginals, ⟨P⟩,  │ │ .hpp                     │
│ + DenseGate (prepared U) │ │ collapse, CDF search     │ │ column-packed tableau,   │
└────────┬─────────────────┘ └──────┬───────────────────┘ │ Clifford updates, Z      │
         │                          │                     │ measurement, outcome     │
┌────────▼──────────────────────────▼─────────────────┐   │ support, ⟨P⟩, → |ψ⟩      │
│ QuantumStateVector        include/QuantumState.hpp  │◄──┤ (to_state_vector)        │
│ Eigen::VectorXcd of 2^N complex<double>, contiguous │   └──────────────────────────┘
└─────────────────────────────────────────────────────┘
               Parallel.hpp: OpenMP macro + thresholds (all kernels)
```

- **`QuantumStateVector`**: owns the amplitudes. It does no physics: you get
  construction, element access, `norm()`, and `normalize()`.
- **`QuantumGate`**: stateless static functions, each mutating a `QuantumStateVector` in
  place. Every function validates its input first, so on invalid input it throws and
  leaves the state untouched.
- **`StabilizerState`**: the Aaronson–Gottesman tableau. It is public, so you can use it
  directly without the state machine. Like `QuantumGate`, it validates input before
  changing anything.
- **`QuantumStateMachine`**: the stateful interface you program against. It records every
  operation and, on the state-vector backend, maintains the invariant ‖ψ‖ = 1 (to rounding).
- **`detail::` kernels**: internal headers under `src/`. They are not part of the public
  include path.

---

## API overview

All public symbols live in namespace `Qputer`.

### `QuantumStateVector`

```cpp
QuantumStateVector s{n};                 // |0…0>, 1 ≤ n ≤ 25
QuantumStateVector s{amplitudes};        // size must be 2^n; copied as-is, not normalized
s.num_qubits(); s.size(); s.norm(); s.normalize();
s[i]; s.data(); s.vector();              // amplitude access
```

### `QuantumGate` (low-level, stateless)

```cpp
QuantumGate::h(s, 0);
QuantumGate::cnot(s, 0, 1);
QuantumGate::mcx(s, {0, 1, 2}, 3);
QuantumGate::apply(s, {2, 0}, U);                // dense U, M ≤ 10 targets
QuantumGate::controlled(s, {1}, {2, 0}, U);      // controlled dense U
QuantumGate::require_unitary(U, M, "ctx");       // throws unless ‖U†U − I‖_max ≤ 1e-10
```

Available gates:

- **1 qubit:** `x y z h s sdg t tdg sx rx ry rz phase u3`
- **2 qubit:** `cnot cz cphase swap`
- **3 qubit:** `toffoli fredkin`
- **N qubit:** `mcx mcz mcphase`
- **Dense:** `apply controlled`

### `StabilizerState` (low-level tableau)

```cpp
StabilizerState t{n};                    // |0…0>, 1 ≤ n ≤ 65536
t.set_basis(index);                      // |index>; qubits ≥ 64 start in |0>
t.h(0); t.s(1); t.cnot(0, 1); t.cz(1, 2); t.swap(0, 2);   // also x y z sdg sx
int r = t.measure(q, outcomeIfRandom);   // certain outcome, or the given 0/1 if random
OutcomeSupport sup = t.outcome_support({0, 2});  // joint Z distribution, state unchanged
int e = t.expectation("XZ", {0, 1});     // exactly -1, 0 or +1
std::size_t S = t.entanglement_entropy(std::vector<Qubit>{0, 1});  // bits; GF(2) rank of the restricted generators
std::vector<std::string> g = t.stabilizers();    // e.g. {"+XX", "+ZZ"}
QuantumStateVector psi = t.to_state_vector();    // n ≤ 25
```

The caller supplies `outcomeIfRandom`, so the tableau never touches an RNG. The state
machine passes the top bit of one RNG draw.

### `QuantumStateMachine`

```cpp
QuantumStateMachine m{numQubits, numClbits = 0, seed = std::nullopt, backend = Backend::Auto};
```

If you don't pass a seed, one is drawn from `std::random_device`. `seed()` always reports
the seed in use, so you can reproduce any run.

| `Backend`     | Register                       | Operations                                        | Qubits       |
|---------------|--------------------------------|---------------------------------------------------|--------------|
| `Auto`        | `StateVector` if N ≤ 25, else `Stabilizer` | —                                     | 1 … 65 536   |
| `StateVector` | 2^N amplitudes                 | all                                               | 1 … 25       |
| `Stabilizer`  | tableau                        | `x y z h s sdg sx cnot cz swap`, `pauli_channel`, measure, reset, `when` | 1 … 65 536 |

To use the tableau on a small register, request it explicitly:
`QuantumStateMachine m{10, 10, seed, Backend::Stabilizer}`. `stabilizerSupports(kind)`
tells you whether an `OpKind` runs on the tableau.

| Group           | Members                                                                                 |
|-----------------|-----------------------------------------------------------------------------------------|
| System          | `num_qubits()` `num_clbits()` `seed()` `reseed(s)` `backend()` `classical_register()` `clbit(c)` `circuit()` |
| State access    | `state()` (state vector only), `stabilizer_state()` (tableau only), `state_vector()` (either backend: a copy, or the tableau converted for N ≤ 25) |
| Preparation     | `prepare()` → \|0…0⟩, `prepare_basis(i)` → \|i⟩, `prepare_state(amps)` (state vector only; must satisfy \|‖ψ‖² − 1\| ≤ 1e-10). Each one clears the circuit and the classical register. |
| Gates (chainable) | every `QuantumGate` gate as a method returning `*this`, plus `unitary(targets, U)` and `controlled_unitary(controls, targets, U)` |
| Generic         | `append(Operation)`: the same validation path as the named methods, intended for deserialized circuits |
| Feed-forward    | `when(clbit, value = true)`: the *next* gate, reset or channel runs only if `clbit == value`; `when_bits(mask, value)`: only if `(classical_register() & mask) == value` |
| Noise           | `pauli_channel(targets, probabilities)`: one target {pX, pY, pZ}, two targets 15 probabilities (order IX IY … ZZ, first letter on targets[0]), sum ≤ 1, both backends. `kraus(targets, {K…})`: Σ K†K = I within 1e-10, state vector only. The live system samples one branch; each replayed shot samples its own. |
| Measurement     | `measure(q, clbit?)` → 0/1, `measure_all()` → basis index (needs `num_clbits ≥ num_qubits`), `reset(q)` |
| Readout         | `probabilities()`, `probability(i)`, `marginal_probabilities(qubits)`, `expectation("XZY", qubits)`, `sample(qubits, shots)`, `sample_counts(qubits, shots)`, `entropy(qubits)` (von Neumann, bits), `reduced_density_matrix(qubits)` (state vector only, k ≤ 13, qubits[0] = MSB) |
| Execution       | `run(shots)` → `Counts`, `terminal_measurements_only()`, `bitstring(value, width)` |

**Readout never collapses the live state.** `sample*` does advance the RNG.
`expectation` computes ⟨ψ|P|ψ⟩ for a Pauli string P = ⊗_k σ_k on the listed qubits,
using the letters `I X Y Z`. Every readout except `reduced_density_matrix` works on both
backends; `entropy` is exact on the tableau and uses the smaller side of the cut on the
state vector. Vector results
(`probabilities`, `marginal_probabilities`) have 2^k entries, so they need k ≤ 25.
`sample` reads at most 64 qubits, because an `Outcome` is 64 bits wide.

**Example: teleportation with feed-forward**

```cpp
QuantumStateMachine tele{3, 3, 2026};
tele.ry(0, 1.1);                       // state to teleport
tele.h(1).cnot(1, 2);                  // Bell pair on (1, 2)
tele.cnot(0, 1).h(0);
tele.measure(0, 0);
tele.measure(1, 1);
tele.when(1).x(2);                     // X correction if clbit 1 == 1
tele.when(0).z(2);                     // Z correction if clbit 0 == 1
tele.ry(2, -1.1);                      // undo the input rotation
tele.measure(2, 2);                    // always 0
Counts c = tele.run(10000);            // clbit 2 is 0 in every shot
```

### `Operation` / `OpKind`

`circuit()` returns a `std::vector<Operation>`. Each `Operation` has these fields:

- `kind`
- `controls`
- `targets`
- `params`
- `matrix` (unitary only)
- `kraus` (kraus only)
- `clbit` (measure only)
- `condition` (an optional `Condition{mask, value}`: runs only while `(classical register & mask) == value`)

`opName(kind)` and `opKindFromName(name)` convert between `OpKind` values and the
lowercase method names, which makes the circuit straightforward to serialize.

---

## Writing your own algorithms

An algorithm is a C++ function that appends operations to a `QuantumStateMachine&`. The
programs in `examples/` are working templates. Each one checks its result against theory
and exits nonzero on a mismatch, and ctest runs all of them.

| Example                | Algorithm                                                            | Shows |
|------------------------|----------------------------------------------------------------------|-------|
| `phase_estimation.cpp` | Quantum phase estimation of any 2^M × 2^M unitary U                  | Subroutines as functions (inverse QFT), `controlled_unitary` with U^(2^k) by repeated squaring, `run(shots)`, comparison with the peak probability (sin πδ / (2^t sin(πδ/2^t)))² |
| `variational.cpp`      | VQE for an n-qubit transverse-field Ising chain                      | A Hamiltonian as weighted Pauli strings, exact energies from `expectation`, parameter-shift gradients ∂E/∂θ = [E(θ+π/2) − E(θ−π/2)]/2, Adam, a check against exact diagonalization |
| `repetition_code.cpp`  | Bit-flip repetition code memory, d = 7 and d = 500                   | Ancilla parity measurement, classical decoding of live outcomes, the same code on both backends (13 qubits on the state vector, 999 on the tableau) |

**Patterns:**

- **Subroutines** are functions that take the machine and a `QubitList`, so they compose by
  calling each other. To invert one, write the gates in reverse order with negated angles.
  The library does not build adjoints or controlled versions for you.
- **Readout while developing.** `probabilities()`, `expectation()` and `state()` are exact
  and do not collapse anything, which no hardware offers. Use them to debug, and for
  variational objectives. Use `run(shots)` to see what hardware would report.
- **Classical control.** A live `measure()` returns the outcome, so C++ can branch on it.
  But `run()` replays the recorded circuit, and gates added inside a C++ branch are
  recorded unconditionally. For logic that differs per shot, use `when(clbit)`, which
  tests one bit. Decode more complex logic from the counts afterwards.
- **Large Clifford circuits.** Outcomes that would need more than 64 classical bits can be
  read live from `measure()`, as `repetition_code.cpp` does.

**Adding one:** copy an example to `examples/<name>.cpp`, add `qputer_example(<name>)` to
`examples/CMakeLists.txt`, then `make` and run `build/release/examples/<name>`.

---

## Execution model

**Eager execution.** Every operation goes through validate → apply to the live state →
record. Validation throws before anything changes, so on invalid input the live state,
the classical register, and the circuit are all left as they were.

**`run(shots)`** replays the recorded circuit from the recorded preparation on scratch
registers. The live system is not modified. The circuit must contain at least one
measurement into a classical bit. `run` picks one of two strategies:

1. **Sampled** (`terminal_measurements_only() == true`): used when there is no reset, no
   channel, no condition, and no gate after a measurement on the measured qubit. The unitary part is
   simulated once. The joint marginal over the measured qubits is built as a CDF, and S
   outcomes are drawn from it.
   Cost: one circuit pass + O(2^k) + O(S log 2^k).
2. **Trajectories** (all other circuits): each shot is an independent stochastic run with
   real mid-circuit collapse, feed-forward and channel sampling. A Pauli channel takes one
   draw on either backend; a Kraus channel picks branch k with probability ‖K_k ψ‖².
   Cost: S × circuit cost. Registers of up to 2^17 amplitudes (2 MiB, so one register
   per thread still fits a typical last-level cache) run one shot per thread with serial
   gates. Larger registers run shots one after another, outside any parallel region, and
   each gate kernel uses the thread pool.

`measure_all()` does one joint draw over all 2^N outcomes instead of N sequential
single-qubit measurements. The two have the same distribution, but the joint draw takes
two passes over the state instead of 2N.

On the stabilizer backend, both strategies work the same way. The sampled strategy runs
the unitary part once, computes the joint outcome support of the measured qubits, and
selects one outcome per shot. Trajectories always run in parallel across shots, because
tableau gates cost only O(N/64).

---

## Stabilizer backend

By the Gottesman–Knill theorem, circuits of H, S and CNOT (and therefore X, Y, Z, S†, √X,
CZ, SWAP), plus Z measurements and feed-forward, can be simulated in polynomial time. The
backend uses the Aaronson–Gottesman tableau (Phys. Rev. A 70, 052328, 2004).

**Representation.** The state |ψ⟩ is the unique state with S_i|ψ⟩ = |ψ⟩ for N commuting
stabilizer generators. The tableau also keeps N destabilizers D_i, where D_i anticommutes
with S_i only. Each of the 2N rows is a signed Pauli string (−1)^r ⊗_q P_q, encoded as
bits (x_q, z_q):

```
I = (0,0)   X = (1,0)   Z = (0,1)   Y = (1,1)
```

The tableau is stored **column-major**. For each qubit q, the x bits of all 2N generators
form one bit column and the z bits another, packed 64 generators per word. Each column
has two word-aligned halves: destabilizer i is bit i of the first ⌈N/64⌉ words, and
stabilizer i is bit i of the second half. A generator and its destabilizer partner
therefore share a bit position, and a mask over one half indexes the other directly.
Columns are padded to whole 64-byte cache lines, stored 64-byte aligned, and get one
extra line when their stride would be a multiple of 512 bytes, so that columns do not all
map to the same cache sets. The signs form one more bit column. Memory is about
2N × 2N bits ≈ N²/2 bytes, or 2 GiB at the 65 536-qubit cap. The tableau has no global
phase.

**Gates** conjugate every generator, U P U†. Each rule below is a few bitwise operations,
applied to 64 generators per word over one or two columns, so a gate costs O(N/64) word
operations in loops the compiler vectorizes. All right-hand sides use the values from
before the update:

| Gate        | Row update                                              |
|-------------|---------------------------------------------------------|
| `h`         | r ⊕= x·z;  x ↔ z                                         |
| `s`         | r ⊕= x·z;  z ⊕= x                                        |
| `sdg`       | r ⊕= x·¬z;  z ⊕= x                                       |
| `sx`        | r ⊕= z·¬x;  x ⊕= z  (√X = H S H)                         |
| `x`/`y`/`z` | r ⊕= z / x ⊕ z / x                                      |
| `cnot(c,t)` | r ⊕= x_c·z_t·(x_t ⊕ z_c ⊕ 1);  x_t ⊕= x_c;  z_c ⊕= z_t  |
| `cz(a,b)`   | r ⊕= x_a·x_b·(z_a ⊕ z_b);  z_a ⊕= x_b;  z_b ⊕= x_a       |
| `swap(a,b)` | swap columns a and b                                    |

**Measuring Z_q** costs at most O(N²/64):

- If some stabilizer S_p anticommutes with Z_q (x_{p,q} = 1), the outcome is random.
  Every other anticommuting generator is multiplied by S_p, S_p moves into D_p, and S_p
  becomes (−1)^m Z_q. The generators to update are x column q as a bit mask. The
  multiplication visits only the columns where S_p is not the identity, and streams each
  of them block by block. Cost: O(wt(S_p) · N/64).
- Otherwise the outcome is certain. Z_q = ±∏ S_i over the i whose D_i anticommutes with
  Z_q (the first half of x column q), and the sign of that product is the outcome.

Products track the i-exponent Σ_q g(P_q, P'_q) mod 4 with bit-sliced arithmetic. For
single-qubit Paulis (x₁, z₁)·(x₂, z₂), the pair anticommutes where
a = x₁z₂ ⊕ z₁x₂, and among those g = −1 exactly where x₁ ⊕ z₁ ⊕ x₂ ⊕ z₂ ⊕ x₁z₂. A pair of
words (lo, hi) holds a mod-4 counter per bit lane, updated as hi ⊕= (lo ⊕ [g = −1]) · a,
lo ⊕= a.

- When multiplying by the pivot, the lanes are the updated generators. New sign =
  r_i ⊕ r_p ⊕ hi_i (lo is 0 because every updated generator commutes with the pivot).
- For the certain-outcome product S_{i₁} S_{i₂} …, the running product's bits at qubit c
  are the exclusive prefix XOR of the chosen generators' bits in column c. That is one
  carry-less multiply (PCLMULQDQ) per word, or six shift-XORs without it. The sign is the
  XOR of the chosen signs and bit 1 of Σ_c [popcount(lo_c) + 2·popcount(hi_c)].

The column layout trades certain outcomes for gates. A product over a sparse set of
generators has to read every column's word holding them, O(N · words) instead of the
row-major O(|set| · N/64). On dense random states that makes a certain outcome up to ~3×
slower than with rows, while gates are two to three orders of magnitude faster.

**Joint outcome distribution** (`OutcomeSupport`). The Z outcomes over k qubits are uniform
over an affine subspace

```
o₀ ⊕ span(B) ⊆ GF(2)^k,   |B| = d,   P(o) = 2^−d for each of the 2^d outcomes
```

The backend computes it in one pass of sequential measurements on a copy:

1. Each random outcome is fixed to 0 and kept as a free bit. The free bit lives only in
   the sign of the ±Z row its measurement creates. Later measurements never multiply that
   row into others, because it has no X part.
2. Each certain outcome is therefore a constant XOR the free bits of the rows in its
   product. The constants form o₀, and the free-bit dependencies form B.
3. B is reduced to echelon form with descending leading bits.

After step 3, o₀ is the smallest outcome, and the outcome of rank j is
o₀ ⊕ ⨁{B_m : bit d−1−m of j}. Sampling S shots therefore costs O(k·N²/64) once plus
O(d·k/64) per shot, instead of copying the tableau and measuring k qubits on every shot.
Probabilities and marginals enumerate the support. `probability(i)` tests whether i is in
the support.

**Expectation values** are exact. If a Pauli string P anticommutes with some S_i,
⟨P⟩ = ⟨P S_i⟩ = −⟨S_i P⟩ = −⟨P⟩, so ⟨P⟩ = 0. Otherwise ±P is in the stabilizer group
(the product of the S_i whose D_i anticommute with P), and ⟨P⟩ is its sign. Finding the
generators that anticommute with a k-qubit P costs k column XORs, O(k·N/64): an X at
qubit c meets z column c, and a Z meets x column c.

**Conversion to amplitudes** (`to_state_vector`, `state_vector()`, N ≤ 25; O(N·2^N)) uses
|ψ⟩⟨ψ| = ∏_i (I + S_i)/2. Applying the product to a basis state |b⟩ leaves ⟨ψ|b⟩ |ψ⟩:

- b must overlap |ψ⟩. Starting from |0…0⟩ fails for states orthogonal to it, such as |1⟩.
  The backend starts from b = o₀, the smallest outcome in the support.
- The amplitude at b, |⟨ψ|b⟩|², is real and positive. This fixes the global phase: the
  lowest-index nonzero amplitude is real and positive.
- Each step adds unit multiples (±1, ±i) and halves, so amplitudes stay exact until the
  final normalization.

**What the tableau cannot do.** Non-Clifford operations (`t`, rotations, `phase`, `u3`,
`cphase`, `toffoli`, `fredkin`, `mc*`, `unitary`) throw `std::invalid_argument`. This
includes rotations at Clifford angles such as `rz(π/2)`; use `s`, `sdg`, `z` instead.
`prepare_state` also throws, because arbitrary amplitudes are generally not a stabilizer
state. Nothing converts a tableau into a state vector automatically mid-circuit. To
continue a Clifford prefix with non-Clifford gates, load `state_vector()` into a
state-vector machine with `prepare_state`.

---

## Performance design

**Memory.** 16 · 2^N bytes. N = 25 → 512 MiB. `kMaxQubits` in
`include/QuantumState.hpp` caps the state vector and sets where `Backend::Auto` switches to
the tableau. Raise it to allow larger state vectors. Construction and copies write the
amplitudes in parallel with the same static partition as the readout kernels. On
multi-socket hosts each page is therefore first touched, and placed, on the NUMA node of
the thread that later streams it. `prepare_state` overwrites the live register in place,
so it holds three 2^N vectors at its peak (the caller's, the live register, the replay
copy) rather than four.

**Gate kernels** (`src/QuantumGates.cpp`). A gate with k active (control + target) qubits
acts independently on 2^(N−k) disjoint subspaces. Kernels enumerate those subspaces
directly:

- A loop counter p ∈ [0, 2^(N−k)) gets zero bits inserted at the active positions to
  form each subspace base index, with control bits OR'd in. Only amplitudes the gate can
  change are touched.
- Bits of p below the lowest active qubit pass through unchanged, so consecutive p
  values map to contiguous runs of amplitudes. If the lowest g active qubits are
  consecutive, consecutive runs are 2^g runs apart. That holds for as long as the p bits
  from the next active qubit up stay fixed (a *window*). One index computation therefore
  serves a whole window, and gates on low qubits (runs of 1, 2 or 4 amplitudes) no longer
  pay an index computation per element.
- With AVX2 + FMA (`-march=native` on a capable host), the streams use explicit
  intrinsics on the interleaved layout, two amplitudes per 256-bit register. A complex
  2×2 is m·x = addsub(Re m · x, Im m · swap(x)) with fused multiply-adds. When qubit 0 is
  the target, each pair (a₂ᵣ, a₂ᵣ₊₁) fills one register and is transformed in place.
  When qubit 0 is a phase qubit, the odd amplitude is scaled and blended. Other builds use
  plain loops that the compiler vectorizes. The choice is made at compile time, so a binary
  built with `QPUTER_NATIVE=OFF` for distribution always takes the plain loops.
- Each gate shape has its own kernel:

  | Kernel    | Used for                    | Arithmetic                       |
  |-----------|-----------------------------|----------------------------------|
  | swap      | X / SWAP                    | pure permutation, no arithmetic  |
  | phase     | Z, S, T, CZ, MCZ, …         | scales 2^(N−k) amplitudes only   |
  | diagonal  | RZ                          | two scalings                     |
  | real 2×2  | H, RY                       | half the multiplies of a complex 2×2 |
  | complex 2×2 | Y, SX, RX, U3             | general case                     |
  | dense     | M-target U                  | compile-time unrolled for M = 2, 3; runtime-sized above that |

- Complex multiplication is hand-written. `std::complex::operator*` carries a NaN/Inf
  recovery branch (`__muldc3`) that blocks vectorization, and unitary gates on finite
  states never need it.
- Multi-controlled gates (`mcx`, `mcz`, `mcphase`) cost O(2^N) however many qubits they
  involve. They are never decomposed into smaller gates.
- At N ≥ 22 every full-state gate streams the vector through DRAM. The kernels run
  within about 25% of the in-place read+write bandwidth floor, so above that size layout
  changes cannot buy much.

**Amplitude layout.** Amplitudes stay interleaved (`std::complex<double>`, the layout
`data()` exposes). A split real/imaginary layout was measured against it. With explicit
AVX2 on the interleaved layout, the split layout gained about 1.2× for complex 2×2 gates
on cache-resident states, nothing at DRAM sizes, and lost on qubit 1 without extra
shuffles. That did not justify changing the public layout. Strided gather/scatter loads
are not used: every target is reached through contiguous runs or in-register pairs.

**Dense gates** (`src/DenseGate.hpp`). Validating U costs O(8^M), and applying it costs
O(2^(N+M)). `prepareDense` converts U once into row-major form plus per-basis-state
amplitude offsets. The state machine stores that prepared form next to each recorded
unitary and shares it between copies of the machine (`shared_ptr<const>`), so replays
across thousands of shots never re-validate or re-lay-out U.

**Readout kernels** (`src/ReadoutKernels.cpp`). Each is one O(2^N) pass:

- **Marginals:** while 2^k ≤ 2^16, per-block histograms stay cache-resident. A byte-wise
  lookup table maps each amplitude index to its outcome, one load per 256 amplitudes for
  the high bytes plus one per amplitude for the low byte, instead of one step per
  measured qubit. Block histograms are merged histogram by histogram, so the merge
  streams sequentially. Above 2^16 outcomes, each outcome is summed independently
  (outcome-major order).
- **Pauli expectation:** P = i^{#Y} X^x Z^z, so

  ```
  ⟨ψ|P|ψ⟩ = Re( i^{#Y} Σ_i conj(a_{i⊕x}) a_i (−1)^{popcount(i ∧ z)} )
  ```

  Terms i and i ⊕ x pair up. With w = conj(a_{k⊕x}) a_k and σ = (−1)^{popcount(x ∧ z)},
  a pair contributes s(k)(w + σ w̄), which is 2 s(k) Re w or 2i s(k) Im w. The kernel
  enumerates k with x's top bit clear, so every amplitude is read once. The library never
  builds a 2^N × 2^N matrix.
- **Sampling:** for ≤ 2^16 outcomes, each shot does a binary search on the CDF. For more
  outcomes, the S draws are sorted and matched against the CDF in a single sequential
  sweep: O(S log S + 2^k) instead of O(S · k) cache misses.

**Tableau kernels** (`src/StabilizerState.cpp`). Gates are vectorized word loops over one
or two columns. A random-outcome collapse streams each non-identity pivot column in 8-word
blocks, each a cache line holding 512 generators, accumulating the phase counters
alongside. Measurement updates stay on one thread below 2^22 column-words (about 1 ms on
one core). Smaller tableaus are cache-resident, and splitting them across cores costs
more in fork/join and cache-to-cache traffic than it saves. Above that, threads split the
columns and add their bit-sliced counters exactly.

**Parallelism.** OpenMP regions only fork at ≥ 2^14 iterations (`kParallelThreshold`,
`src/Parallel.hpp`). Below that, fork/join overhead exceeds the work. A gate's team gets
one thread per 2^13 units of work, up to the OpenMP maximum (`teamSize`). A unit is one
amplitude pair of a single-qubit gate. A dense m-target subspace counts as 4^m / 4 units,
because its 2^m × 2^m mat-vec is compute-bound. Mid-sized streaming gates therefore fork
fewer threads, and each barrier waits on fewer of them. That matters when
other processes occupy some cores: a single descheduled team member stalls the whole
region. The parallel shot sampler stays serial below 4096 shots. Bandwidth-bound kernels
rarely gain from SMT siblings, so on a shared machine `OMP_NUM_THREADS` = physical cores is
usually faster than the default of one thread per hardware thread.

**Floating-point contraction.** The scalar tails of the AVX2 kernels call `std::fma` in the
same order as the vector bodies. Every product left unfused is passed as an argument to an
`fma`, so the compiler has nothing more to fuse. Implicit contraction (GCC's default in C++,
Clang's within an expression) therefore cannot make the two paths round differently. It stays
enabled everywhere else, because dense gates and readout have no hand-written vector path to
match and switching it off roughly doubles the cost of the dense mat-vec.

---

## Determinism

Given a seed, every result is reproducible and **bitwise-identical regardless of the
OpenMP thread count**. For Clifford circuits it is also **the same on both backends**:

- **RNG:** xoshiro256** seeded through SplitMix64. Each shot owns its own stream
  `Rng(base, shot)`, where `base` is one draw from the machine's RNG per `run()`. The
  random numbers therefore don't depend on how shots are distributed across threads.
  Repeated `run()` calls give different results, but all of them follow from `seed()`.
- **Reductions:** floating-point reductions (norms, marginal weights, ⟨P⟩) use a fixed
  decomposition into 256 blocks determined only by the problem size. Partial sums are
  combined in block order, so the rounding is the same at 1 thread or 64.
- **Gate kernels:** each amplitude's new value is computed by the same operation
  sequence wherever a thread's slice begins or ends. Vector bodies and scalar tails
  perform identical fused multiply-adds, so results are bitwise identical at any thread
  count and team size.
- **Tableau:** all updates are integer bit operations, and per-thread phase counters add
  exactly, so tableau results never depend on threading.
- **Rounding at the CDF tail:** draws that rounding pushes past the end of the CDF
  resolve to the last outcome with nonzero weight. A zero-probability outcome is never
  returned.
- **Across backends:** a stabilizer state's outcomes are 2^d equally likely values. The
  state vector's CDF search with u = draw · 2^−64 (53 bits) picks the outcome of rank
  ⌊u·2^d⌋ in index order. The tableau picks the same rank from the top d bits of the same
  draw. Single-qubit measurements are the d = 1 case: outcome 1 exactly when the top bit
  is set. The two backends can only diverge when a draw lands within rounding error of a
  state-vector CDF boundary, with probability about 2^(d−53) per draw.

---

## Limits and error handling

| Limit                         | Value       | Constant                                   |
|-------------------------------|-------------|--------------------------------------------|
| Qubits, state vector          | 1 … 25      | `kMaxQubits` (also the `Auto` cutoff)      |
| Qubits, stabilizer tableau    | 1 … 65 536 (≈ N²/2 bytes) | `kMaxStabilizerQubits`       |
| Readout vector length         | 2^k, k ≤ 25 | `kMaxQubits`                               |
| Qubits per `sample` outcome   | ≤ 64        | width of `Outcome`                         |
| Classical bits                | 0 … 64      | `QuantumStateMachine::kMaxClbits`          |
| Dense gate targets            | 1 … 10 (16 MiB matrix) | `QuantumGate::kMaxDenseTargets` |
| Unitarity tolerance           | max\|U†U − I\| ≤ 1e-10 | `QuantumGate::kUnitaryTolerance` |
| `prepare_state` norm tolerance| \|‖ψ‖² − 1\| ≤ 1e-10 | `QuantumStateMachine::kNormTolerance` |

The library reports errors by throwing standard exceptions. Each message is prefixed with
the calling function's name:

| Exception                | Thrown for                                                   |
|--------------------------|--------------------------------------------------------------|
| `std::length_error`      | register size out of range for the backend; readout vector or conversion over 2^25 entries |
| `std::out_of_range`      | qubit, clbit, or basis index out of range                    |
| `std::invalid_argument`  | duplicate qubits, wrong arity, non-finite parameters, non-unitary matrix, channel probabilities outside [0, 1] or summing above 1, Kraus operators that are not trace preserving, empty or out-of-mask conditions, conditioned measurement, run with no measurement, non-Clifford operation, `kraus` or `prepare_state` on the stabilizer backend |
| `std::logic_error`       | `state()` on the stabilizer backend, `stabilizer_state()` on the state-vector backend |
| `std::domain_error`      | sampling or measuring a zero-norm or non-finite state        |

Every operation either throws before mutating anything or completes. Nothing is left
partially applied.

---

## Tests

`make test` runs 131 GoogleTest cases, the three examples and the 253 Noether test cases
(387 ctest entries; `ctest -L noether` selects the Noether ones). In Debug mode they all run
under ASan and UBSan.

| File                          | Covers                                                                 |
|-------------------------------|------------------------------------------------------------------------|
| `test_state.cpp`              | Construction, size limits, accessor aliasing, normalization.           |
| `test_single_qubit_gates.cpp` | Basis images vs textbook matrices, known states, algebraic identities (e.g. HZH = X), bit mapping, input rejection. |
| `test_multi_qubit_gates.cpp`  | Truth tables (CNOT, SWAP, Toffoli, Fredkin), phase gates, MCX, Bell/GHZ preparation, decomposition identities. |
| `test_dense_gates.cpp`        | MSB target convention, agreement with named gates, Haar-random unitaries vs reference, input rejection. |
| `test_reference.cpp`          | Every named gate on every qubit placement vs an independently built 2^N × 2^N operator. Norm preservation. U·U† = I on random circuits. |
| `test_algorithms.cpp`         | QFT, Grover (closed-form success probability), Bernstein–Vazirani, Deutsch–Jozsa, teleportation, superdense coding. |
| `test_parallel.cpp`           | A 17-qubit circuit (above the parallel threshold): the multi-threaded result matches the single-threaded one. |
| `test_state_machine.cpp`      | Circuit recording, `append` ≡ named methods, name round-trip, atomic failure, preparation, `when`, seed reproducibility. |
| `test_readout.cpp`            | Probabilities, marginals (both kernel paths, any qubit order), Pauli expectations, Born-rule sampling without collapse, zero-weight exclusion, sorted-sweep path. |
| `test_measurement.cpp`        | Collapse = normalized projection, Born-rule frequencies, reset, both `run` strategies vs the exact distribution, replay from the prepared state, live state untouched, thread-count independence. |
| `test_stabilizer.cpp`         | Tableau alone: textbook conjugation rules, random Clifford circuits vs the reference operator, exact basis-state conversion, global-phase convention, ⟨P⟩ for every Pauli string, generators stabilize the state, outcome support = Born distribution in rank order, collapse = projection, qubits across word boundaries, 1000-qubit GHZ, input rejection. |
| `test_channels.cpp`           | `when_bits` on several clbits (live pass = `run`, both backends, validation). Pauli channels: certain Paulis act as their gates, the documented two-qubit order, flip frequencies, forced trajectories, identical seeded outcomes on both backends, conditioned channels, validation. Kraus channels: full amplitude damping, Born-rule branch frequencies, target convention, normalized post-branch state, validation, thread-count independence. |
| `test_entropy.cpp`            | Entropy of Bell, product, GHZ, partially entangled (binary entropy) and random states vs the reference on either side of the cut, tableau entropy vs the state vector on random Clifford circuits, a 20 000-qubit GHZ half cut, validation. Reduced density matrices vs the reference partial trace, thread-count independence, validation. |
| `test_backend_comparison.cpp` | Backend selection at the cutoff, Clifford-only enforcement above it, 1000-qubit GHZ through the state machine. The same circuit on both backends, checked against each other and the expected answer: random Clifford circuits (state, probabilities, marginals, ⟨P⟩ vs the reference operator), GHZ, superdense coding, teleportation with feed-forward, reset + conditioned gates, live mid-circuit measurement, sampling. Identical seeded counts across backends, thread-count independence. |

`tests/TestSupport.hpp` provides the oracle. `embed()` builds the full operator column by
column from the documented conventions and shares no code with the kernels, so agreement
with it is independent evidence of correctness. The header also provides Haar-random
unitaries (QR of a Ginibre matrix with phase correction), random states and circuits,
random Clifford circuits with their reference states and Pauli expectations, textbook
matrices, and `statesNear` / `statesNearUpToPhase` assertions that report the
worst-mismatching index.

---

## Noether

Noether writes circuits the way physics papers do: kets, operator products acting right to
left, subscripts for qubits, `⟨…⟩` for expectation values, Σ and ∏ for sums and products. It
compiles to the operations above and runs them on a `QuantumStateMachine`. ASCII and LaTeX
spellings are accepted (`CNOT_{0->1}`, `\expval{Z_0}`, `S\dagger_0`), and every compiling
command first rewrites the file to one canonical Unicode form.

```sh
build/release/noether/noether run noether/examples/teleport.ntr --json
build/release/noether/noether check f.ntr --json      # diagnostics with codes, spans and fix-its
```

The CLI also draws circuits, estimates resources, checks equivalence, optimises parameters,
exports OpenQASM 3 and runs an autonomous research loop against a task spec. The full
description, the examples and the test layout are in `noether/README.md`.

`skills/` holds three skills for coding agents: `qsm` (writing and running Noether
programs), `qsm-research` (the research loop) and `qsm-tune` (tuning the simulator for the
host). They are embedded in the `noether` binary, and `noether skill --install DIR` writes
them into any skills directory an agent reads.

---

## File reference

```
.
├── CMakeLists.txt              Library, app, examples, warning/sanitizer flags, Eigen + OpenMP
├── CMakePresets.json           debug (ASan+UBSan) / release presets → build/<preset>/
├── Makefile                    make {build,test,run,examples,clean,distclean}, BUILD_TYPE=Release|Debug
├── apps/
│   └── main.cpp                Demo: Bell, GHZ marginal, teleportation, 1000-qubit GHZ
├── examples/                   Self-checking algorithm templates, also run by ctest
│   ├── CMakeLists.txt          qputer_example(<name>): executable + ctest entry
│   ├── phase_estimation.cpp    QPE of an arbitrary unitary, inverse QFT subroutine
│   ├── variational.cpp         VQE: Pauli-sum Hamiltonian, parameter shift, Adam, exact check
│   └── repetition_code.cpp     Syndrome extraction + decoding on both backends
├── include/                    Public API
│   ├── QuantumState.hpp        QuantumStateVector, kMaxQubits
│   ├── QuantumGates.hpp        QuantumGate static kernels, Qubit/QubitList
│   ├── StabilizerState.hpp     StabilizerState tableau, OutcomeSupport, kMaxStabilizerQubits
│   └── QuantumStateMachine.hpp QuantumStateMachine, Backend, Rng, Operation, OpKind, Counts
├── src/                        Implementation (src/*.hpp are internal)
│   ├── QuantumState.cpp        State vector construction + validation, parallel first-touch init and copies
│   ├── QuantumGates.cpp        Subspace layout + windowed runs, AVX2/scalar kernels, dense prepare/apply
│   ├── DenseGate.hpp           Prepared dense-gate representation
│   ├── ReadoutKernels.hpp/.cpp Probabilities, marginals, Pauli sums, collapse, CDF search
│   ├── StabilizerState.cpp     Column-major Clifford updates, bit-sliced measurement, outcome support, conversion
│   ├── QuantumStateMachine.cpp Backend dispatch, validation, execution, sampling, run()
│   └── Parallel.hpp            QPUTER_OMP macro, thread helpers, kParallelThreshold, teamSize
├── noether/                    Noether language, `noether` CLI, examples and tests
├── skills/                     Agent skills: qsm, qsm-research, qsm-tune
└── tests/
    ├── CMakeLists.txt          qputer_tests + GoogleTest discovery
    ├── TestSupport.hpp         Reference operator, random states/unitaries, assertions
    └── test_*.cpp              14 suites, see Tests above
```
