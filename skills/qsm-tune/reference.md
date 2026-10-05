# Qputer host tuning: reference

## 1. Where speed depends on the host

The host decides how fast the simulator runs:

- cache sizes set the crossover points between strategies;
- core count and fork/join cost decide when parallelism pays;
- the instruction set decides which kernels exist;
- memory bandwidth sets a floor that no kernel change beats.

The shipped constants were tuned on a Ryzen 5 3600 with:

- 6 cores / 12 threads;
- 512 KiB L2 per core and 2 × 16 MiB L3;
- AVX2 + FMA;
- one NUMA node at ≈ 23.6 GB/s in-place read-modify-write.

**Roofline first.** Arithmetic intensity I is flops per byte moved, and machine balance is
β = F_peak / B_mem. A kernel with I < β is bandwidth-bound, so only thread placement helps it.
A workload already at its bandwidth floor is done: report it and move on.

| Kernel | I (flop/B) |
|---|---|
| 1-qubit gate | 28 / 64 ≈ 0.44 |
| dense m-qubit gate | 2^m / 4 |

**Cutoffs.** Each constant is the crossover between two strategies. The prior comes from the
hardware, and the measured crossover replaces it. P is the thread count; τ_fork, τ_barrier,
t_pair and t_draw come from the microbenchmarks.

| Constant, location | Chooses between | Prior | Sweep |
|---|---|---|---|
| `kParallelThreshold`, `src/Parallel.hpp` | serial or forked region | n* = τ_fork / (t_pair (1 − 1/P)) | 1-qubit gate, N = 10…18 |
| `kThreadGrain`, `src/Parallel.hpp` | team size of a region | g · t_pair ≈ 10 τ_barrier | N = 14…20 |
| `kShotParallelMaxAmplitudes`, `src/QuantumStateMachine.cpp` | one trajectory per thread, or all threads on each gate | largest 2^N with P · 16 · 2^N ≤ L3 | noisy `run`, N = 13…20 |
| `kParallelSampleShots`, same file | serial or parallel shot draws | s* = τ_fork / (t_draw (1 − 1/P)) | 2^8…2^16 shots |
| `kSearchMaxOutcomes`, same file | per-shot binary search, or one sorted sweep | 8 · 2^k ≤ L2 per core | sampling on k = 12…22 qubits |
| `kParallelWords`, `src/StabilizerState.cpp` | serial or parallel tableau collapse | 8 · W ≥ L3 | measurement at N = 10³…2·10⁴ |
| `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND` | thread count and placement | physical cores; `close` on one socket, `spread` on several | whole suite |

**Which constants keep results bitwise identical.**

- **These cutoffs:**
  - Gate kernels give the same bits for any slicing and team size.
  - Sampling and trajectories draw from one RNG stream per shot.
  - Both sampling strategies select the first j with cum[j] > u.
  - Tableau phases are summed mod 4, which is exact in any order.
- **Exception, `kParallelThreshold`:** through `blockCount` in `src/ReadoutKernels.cpp`, it
  also decides whether a readout reduction sums 1 partial or `kBlocks` partials. Give
  `blockCount` its own fixed cutoff before sweeping it.
- **Not cutoffs; don't tune them:** `kBlocks`, `kHistogramMaxOutcomes` and `kHistogramBudget`
  fix the summation order by design. The same holds for the fixed blocks of the Kraus
  branch-weight reduction in `src/ReadoutKernels.cpp`.

**Kernels and layout.**

| Target | Location | When it pays | Constraint |
|---|---|---|---|
| team size by work | `QPUTER_OMP_PARALLEL(work)`, `src/QuantumGates.cpp` | any compute-bound kernel | a 2^m-target dense subspace counts as 4^m / 4 pair units |
| SIMD stream kernels | the `__AVX2__ && __FMA__` block, `src/QuantumGates.cpp` | AVX-512; Neon | the vector body and its scalar tail use one FMA order |
| prefix XOR | `exclusivePrefixXor`, `src/StabilizerState.cpp` | ARM: PMULL (`vmull_p64`) | integer: must match bit for bit |
| dense mat-vec | `kernelDense`, `src/QuantumGates.cpp` | hosts with high β | register blocking; SIMD across subspaces |
| cache-line words `kBlock`, alignment `kAlign` | `src/StabilizerState.cpp`, `include/StabilizerState.hpp` | 128-byte lines (Apple M-series) | integer layout, bitwise identical |
| set-aliasing pad `kSetStride` | `src/StabilizerState.cpp` | different L1 geometry | keep only if L1 misses drop (`perf stat`) |

**Build.** Try the compiler, `-march` (the `QPUTER_NATIVE` option), LTO and PGO. Before adding
any flag, compile a small test to see what it actually changes.

## 2. Harness

Build the harness once, in a gitignored scratch directory, and reuse it on later runs.

| Part | Purpose |
|---|---|
| baseline worktree | `git worktree add --detach <dir> <start commit>`, built with the same compiler and flags |
| scenario benchmark | one process per scenario, printing best and median ms, peak RSS and a result checksum as JSON |
| interleaved A/B driver | runs base and candidate alternately and keeps the best of each; prints the speedup, peak RSS, and whether the checksums match |
| state-vector dump and compare | applies a seeded random circuit of every gate kind (including dense and controlled unitaries, Pauli channels and Kraus channels) and writes the amplitudes; the comparer reports the bitwise-differing count and the max \|Δa\| |
| tableau reference | the baseline tableau under another class name, fuzzed bit for bit against the current one on random Clifford circuits |
| microbenchmarks | in-place bandwidth at 2^24 amplitudes; τ_fork and τ_barrier; t_pair, t_draw; a per-qubit gate profile |
| Noether scenarios | `.ntr` programs run with `noether run --json --no-format`: `timing.executeMs` is the time, and the `prints`/`runs` blocks (with `--no-timing`) are the checksum |

The scenarios cover every constant above:

- gate sweeps at N = 14, 18 and 24;
- gates on qubits 0–2;
- dense gates on 2–4 targets;
- readout, allocation and QFT;
- noisy trajectories at N = 10–18;
- tableau gates, measurement and readout at N = 10³–10⁴;
- tableau trajectories;
- entropy and reduced density matrices;
- the user's own workloads, when they name any.

## 3. Correctness gate

Every change must pass all of these before its timing counts:

1. The Release tests (`cmake --preset release && cmake --build build/release && ctest --preset
   release`) and the Debug tests under ASan + UBSan (`--preset debug`, with a lower `-j`).
2. **Thread invariance:** amplitude dumps are bitwise equal at 1, ⌊P/2⌋ + 1 and P threads.
3. **Against the baseline:** max |Δa| ≤ 10⁻¹³ on the dump workload, and every benchmark and
   Noether checksum is equal.
4. **Tableau fuzz:** bit-exact against the reference.
5. **Instruction-set-specific code only:** gates 1–4 also on a generic build
   (`-DQPUTER_NATIVE=OFF`), and with a second compiler when one is available.
6. **Every change:** the build and tests without OpenMP
   (`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON`), where every `QPUTER_OMP` clause disappears.

## 4. Measurement protocol

On a loaded desktop, a 12-thread barrier stalls whenever the OS deschedules one of its threads.
Identical runs have ranged from 17 ms to 3.2 s. So:

- Don't time while the load average exceeds 0.5 per core.
- Interleave base and candidate (B, C, B, C, …) with k ≥ 5 repeats, and keep the best of each.
- Count a change as a gain only above max(3%, the base's own (max − min)/min spread).
- Time algorithmic changes on 1 pinned thread (`taskset`), and placement changes at full width.
- Re-check every large gain with pinned single-thread runs. Outliers in the baseline have
  inflated gains before: a 9.5× was really 2.1×.
- Compare against the baseline at its *best* thread count, not its default.

## 5. Loop

1. **Fingerprint the host.**
2. **Measure the floors:** bandwidth; FMA peak = cores × clock × flops per cycle; β.
3. **Measure the noise band:** run the baseline against itself.
4. **Classify each scenario** as one of:
   - at its bandwidth floor: done;
   - compute-bound below peak: kernel work;
   - overhead-bound: cutoffs, team size, fork/join.

   Run `perf record -g` on the scenario furthest from its floor.
5. **Change one thing.** In order of typical payoff for effort:
   1. thread placement;
   2. cutoffs: start at the prior, sweep powers of two, and put the cutoff where the two
      strategies' timings cross;
   3. team sizing by work;
   4. traversal and data layout;
   5. instruction-set kernels.
6. **Gate, then measure** with an interleaved A/B. Keep a change that gains (one commit per
   change on `tune/<host>`); revert one that doesn't.
7. **Re-run the whole suite** after any kept change to shared code.
8. **Finish** with a full A/B of the branch against the start commit.

## 6. Traps

| Trap | What happened | Rule |
|---|---|---|
| compiler defaults | `-ffp-contract=off` switched off GCC's default FMA contraction, and dense gates ran 2× slower | compile a small test to see what a flag changes before adding it |
| team size by iteration count | a compute-bound 4-qubit dense gate at N = 18 got 2 threads | size teams by work, not by loop count |
| wrong diagnosis | a loop was restructured to fix a regression whose real cause was a flag | confirm a cause by measurement; revert fixes that did not help |
| noisy baseline | a background process on one core stalled every 12-thread barrier | check `uptime` and `top`; compare only interleaved runs |
| argument evaluation order in harness code | `u3(q, ang(rng), ang(rng), ang(rng))` draws in a different order under GCC and Clang | sequence every RNG draw in harness code |
| SMT | memory-bound kernels ran slower on 12 threads than on 6 | sweep `OMP_NUM_THREADS` with `OMP_PLACES=cores` |
| build memory | a Debug sanitizer build at `-j6` ran out of memory | lower `-j` for Debug |
| untested configuration | a variable used only inside an OpenMP clause broke the build without OpenMP | build every configuration |
| techniques that do not pay | AVX-512 asked for on a host without it; SoA (≈ 1.2× in cache, nothing at DRAM sizes, and it breaks the API); gather/scatter | prototype and measure before adopting; push back with numbers |
