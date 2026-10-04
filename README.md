# Quantum State Machine

A C++23 state-vector quantum simulator for up to 25 qubits. It stores the full
register |ψ⟩ ∈ ℂ^(2^N), applies gates in place with OpenMP-parallel kernels, and
handles measurement, classical feed-forward, and multi-shot circuit execution.
Seeded runs give bitwise-identical results for any thread count.

```cpp
#include <QuantumStateMachine.hpp>

Qputer::QuantumStateMachine m{2, 2, /*seed=*/2026};   // 2 qubits, 2 classical bits
m.h(0).cnot(0, 1);                                     // Bell pair (|00> + |11>)/√2
double zz = m.expectation("ZZ", {0, 1});               // +1, state unchanged
m.measure_all();                                       // collapses, recorded in the circuit
Qputer::Counts counts = m.run(10000);                  // {00: ~5000, 11: ~5000}
```

---

## Contents

- [Building](#building)
- [Conventions](#conventions)
- [Architecture](#architecture)
- [API overview](#api-overview)
- [Execution model](#execution-model)
- [Performance design](#performance-design)
- [Determinism](#determinism)
- [Limits and error handling](#limits-and-error-handling)
- [Tests](#tests)
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
make test                  # build, then run the GoogleTest suite through ctest
make run                   # build, then run the demo app (build/release/qputer)
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

Every first-party target is built with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow
-Wnon-virtual-dtor -Wold-style-cast -Werror`. Debug builds also add
`-fsanitize=address,undefined`. The library is compiled as position-independent code,
so it can be linked into a shared module. Without OpenMP, the build prints a warning and
every kernel runs single-threaded.

### Build targets

| Target          | Kind           | Contents                                                    |
|-----------------|----------------|-------------------------------------------------------------|
| `qputer_lib`    | static library | State vector, gate kernels, readout kernels, state machine. |
| `qputer`        | executable     | Demo: Bell pair, 5-qubit GHZ marginal, teleportation.       |
| `qputer_tests`  | executable     | GoogleTest suite (75 tests).                                |

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

```
┌────────────────────────────────────────────────────────────────────┐
│ QuantumStateMachine          include/QuantumStateMachine.hpp       │
│  live register · classical register · seeded RNG · recorded circuit│
│  prepare · gates · measure/reset · readout · run(shots)            │
└──────────────┬──────────────────────────────┬──────────────────────┘
               │                              │
┌──────────────▼───────────────┐  ┌───────────▼──────────────────────┐
│ QuantumGate (static)         │  │ ReadoutKernels (detail)          │
│ include/QuantumGates.hpp     │  │ src/ReadoutKernels.hpp           │
│ in-place unitary kernels     │  │ |a_i|², marginals, ⟨P⟩, collapse,│
│ + DenseGate (prepared U)     │  │ CDF search, basis set, copy      │
└──────────────┬───────────────┘  └───────────┬──────────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼──────────────────────┐
│ QuantumStateVector           include/QuantumState.hpp              │
│ N qubits, Eigen::VectorXcd of 2^N complex<double>, contiguous      │
└────────────────────────────────────────────────────────────────────┘
               Parallel.hpp: OpenMP macro + thresholds (all kernels)
```

- **`QuantumStateVector`**: owns the amplitudes. It does no physics: you get
  construction, element access, `norm()`, and `normalize()`.
- **`QuantumGate`**: stateless static functions, each mutating a `QuantumStateVector` in
  place. Every function validates its input first, so on invalid input it throws and
  leaves the state untouched.
- **`QuantumStateMachine`**: the stateful interface you program against. It records every
  operation and maintains the invariant ‖ψ‖ = 1 (to rounding).
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

### `QuantumStateMachine`

```cpp
QuantumStateMachine m{numQubits, numClbits = 0, seed = std::nullopt};
```

If you don't pass a seed, one is drawn from `std::random_device`. `seed()` always reports
the seed in use, so you can reproduce any run.

| Group           | Members                                                                                 |
|-----------------|-----------------------------------------------------------------------------------------|
| System          | `num_qubits()` `num_clbits()` `seed()` `reseed(s)` `state()` `classical_register()` `clbit(c)` `circuit()` |
| Preparation     | `prepare()` → \|0…0⟩, `prepare_basis(i)` → \|i⟩, `prepare_state(amps)` (must satisfy \|‖ψ‖² − 1\| ≤ 1e-10). Each one clears the circuit and the classical register. |
| Gates (chainable) | every `QuantumGate` gate as a method returning `*this`, plus `unitary(targets, U)` and `controlled_unitary(controls, targets, U)` |
| Generic         | `append(Operation)`: the same validation path as the named methods, intended for deserialized circuits |
| Feed-forward    | `when(clbit, value = true)`: the *next* gate or reset runs only if `clbit == value` |
| Measurement     | `measure(q, clbit?)` → 0/1, `measure_all()` → basis index (needs `num_clbits ≥ num_qubits`), `reset(q)` |
| Readout         | `probabilities()`, `probability(i)`, `marginal_probabilities(qubits)`, `expectation("XZY", qubits)`, `sample(qubits, shots)`, `sample_counts(qubits, shots)` |
| Execution       | `run(shots)` → `Counts`, `terminal_measurements_only()`, `bitstring(value, width)` |

**Readout never collapses the live state.** `sample*` does advance the RNG.
`expectation` computes ⟨ψ|P|ψ⟩ for a Pauli string P = ⊗_k σ_k on the listed qubits,
using the letters `I X Y Z`.

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
- `clbit` (measure only)
- `condition` (an optional `Condition{clbit, value}`)

`opName(kind)` and `opKindFromName(name)` convert between `OpKind` values and the
lowercase method names, which makes the circuit straightforward to serialize.

---

## Execution model

**Eager execution.** Every operation goes through validate → apply to the live state →
record. Validation throws before anything changes, so on invalid input the live state,
the classical register, and the circuit are all left as they were.

**`run(shots)`** replays the recorded circuit from the recorded preparation on scratch
registers. The live system is not modified. The circuit must contain at least one
measurement into a classical bit. `run` picks one of two strategies:

1. **Sampled** (`terminal_measurements_only() == true`): used when there is no reset, no
   condition, and no gate after a measurement on the measured qubit. The unitary part is
   simulated once. The joint marginal over the measured qubits is built as a CDF, and S
   outcomes are drawn from it.
   Cost: one circuit pass + O(2^k) + O(S log 2^k).
2. **Trajectories** (all other circuits): each shot is an independent stochastic run with
   real mid-circuit collapse and feed-forward.
   Cost: S × circuit cost. For small registers (2^N < 2^14), shots run in parallel across
   threads. For large registers, shots run one after another and each gate kernel uses
   all threads.

`measure_all()` does one joint draw over all 2^N outcomes instead of N sequential
single-qubit measurements. The two have the same distribution, but the joint draw takes
two passes over the state instead of 2N.

---

## Performance design

**Memory.** 16 · 2^N bytes. N = 25 → 512 MiB. Raise `kMaxQubits` in
`include/QuantumState.hpp` to go larger.

**Gate kernels** (`src/QuantumGates.cpp`). A gate with k active (control + target) qubits
acts independently on 2^(N−k) disjoint subspaces. Kernels enumerate those subspaces
directly:

- A loop counter p ∈ [0, 2^(N−k)) gets zero bits inserted at the active positions to
  form each subspace base index, with control bits OR'd in. Only amplitudes the gate can
  change are touched.
- Bits of p below the lowest active qubit pass through unchanged. Consecutive p values
  therefore map to contiguous runs of amplitudes, which stream with unit stride (SIMD,
  prefetch, `__restrict`) instead of computing one index per element.
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

**Dense gates** (`src/DenseGate.hpp`). Validating U costs O(8^M), and applying it costs
O(2^(N+M)). `prepareDense` converts U once into row-major form plus per-basis-state
amplitude offsets. The state machine stores that prepared form next to each recorded
unitary and shares it between copies of the machine (`shared_ptr<const>`), so replays
across thousands of shots never re-validate or re-lay-out U.

**Readout kernels** (`src/ReadoutKernels.cpp`). Each is one O(2^N) pass:

- **Marginals:** while 2^k ≤ 4096, per-block histograms stay cache-resident. Above that,
  each outcome is summed independently (outcome-major order).
- **Pauli expectation:** P = i^{#Y} X^x Z^z, so

  ```
  ⟨ψ|P|ψ⟩ = Re( i^{#Y} Σ_i conj(a_{i⊕x}) a_i (−1)^{popcount(i ∧ z)} )
  ```

  One pass computes this. The library never builds a 2^N × 2^N matrix.
- **Sampling:** for ≤ 2^16 outcomes, each shot does a binary search on the CDF. For more
  outcomes, the S draws are sorted and matched against the CDF in a single sequential
  sweep: O(S log S + 2^k) instead of O(S · k) cache misses.

**Parallelism threshold.** OpenMP regions only fork at ≥ 2^14 iterations
(`kParallelThreshold`, `src/Parallel.hpp`). Below that, fork/join overhead exceeds the
work. The parallel shot sampler also stays serial below 4096 shots.

---

## Determinism

Given a seed, every result is reproducible and **bitwise-identical regardless of the
OpenMP thread count**:

- **RNG:** xoshiro256** seeded through SplitMix64. Each shot owns its own stream
  `Rng(base, shot)`, where `base` is one draw from the machine's RNG per `run()`. The
  random numbers therefore don't depend on how shots are distributed across threads.
  Repeated `run()` calls give different results, but all of them follow from `seed()`.
- **Reductions:** floating-point reductions (norms, marginal weights, ⟨P⟩) use a fixed
  decomposition into 256 blocks determined only by the problem size. Partial sums are
  combined in block order, so the rounding is the same at 1 thread or 64.
- **Rounding at the CDF tail:** draws that rounding pushes past the end of the CDF
  resolve to the last outcome with nonzero weight. A zero-probability outcome is never
  returned.

---

## Limits and error handling

| Limit                         | Value       | Constant                                   |
|-------------------------------|-------------|--------------------------------------------|
| Qubits                        | 1 … 25      | `kMaxQubits`                               |
| Classical bits                | 0 … 64      | `QuantumStateMachine::kMaxClbits`          |
| Dense gate targets            | 1 … 10 (16 MiB matrix) | `QuantumGate::kMaxDenseTargets` |
| Unitarity tolerance           | max\|U†U − I\| ≤ 1e-10 | `QuantumGate::kUnitaryTolerance` |
| `prepare_state` norm tolerance| \|‖ψ‖² − 1\| ≤ 1e-10 | `QuantumStateMachine::kNormTolerance` |

The library reports errors by throwing standard exceptions. Each message is prefixed with
the calling function's name:

| Exception                | Thrown for                                                   |
|--------------------------|--------------------------------------------------------------|
| `std::length_error`      | register size out of range                                   |
| `std::out_of_range`      | qubit, clbit, or basis index out of range                    |
| `std::invalid_argument`  | duplicate qubits, wrong arity, non-finite parameters, non-unitary matrix, conditioned measurement, run with no measurement |
| `std::domain_error`      | sampling or measuring a zero-norm or non-finite state        |

Every operation either throws before mutating anything or completes. Nothing is left
partially applied.

---

## Tests

`make test` runs 75 GoogleTest cases. In Debug mode they run under ASan and UBSan.

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

`tests/TestSupport.hpp` provides the oracle. `embed()` builds the full operator column by
column from the documented conventions and shares no code with the kernels, so agreement
with it is independent evidence of correctness. The header also provides Haar-random
unitaries (QR of a Ginibre matrix with phase correction), random states and circuits,
textbook matrices, and a `statesNear` assertion that reports the worst-mismatching index.

---

## File reference

```
.
├── CMakeLists.txt              Library, app, warning/sanitizer flags, Eigen + OpenMP
├── CMakePresets.json           debug (ASan+UBSan) / release presets → build/<preset>/
├── Makefile                    make {build,test,run,clean,distclean}, BUILD_TYPE=Release|Debug
├── apps/
│   └── main.cpp                Demo: Bell, GHZ marginal, teleportation
├── include/                    Public API
│   ├── QuantumState.hpp        QuantumStateVector, kMaxQubits
│   ├── QuantumGates.hpp        QuantumGate static kernels, Qubit/QubitList
│   └── QuantumStateMachine.hpp QuantumStateMachine, Rng, Operation, OpKind, Counts
├── src/                        Implementation (src/*.hpp are internal)
│   ├── QuantumState.cpp        State vector construction + validation
│   ├── QuantumGates.cpp        Subspace layout, streaming kernels, dense prepare/apply
│   ├── DenseGate.hpp           Prepared dense-gate representation
│   ├── ReadoutKernels.hpp/.cpp Probabilities, marginals, Pauli sums, collapse, CDF search
│   ├── QuantumStateMachine.cpp Validation, execution, measurement, sampling, run()
│   └── Parallel.hpp            QPUTER_OMP macro, thread helpers, kParallelThreshold
└── tests/
    ├── CMakeLists.txt          qputer_tests + GoogleTest discovery
    ├── TestSupport.hpp         Reference operator, random states/unitaries, assertions
    └── test_*.cpp              10 suites, see Tests above
```
