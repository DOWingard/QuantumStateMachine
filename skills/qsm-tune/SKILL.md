---
name: qsm-tune
description: Optimise the Qputer quantum state machine (the simulator behind Noether) for the machine it is running on. Fingerprint the host, benchmark against a baseline, find what limits each workload (bandwidth, compute or fork/join), change one cutoff, thread setting or kernel at a time, prove results unchanged, and keep only measured improvements. Use when asked to tune, profile or benchmark the simulator or noether, or to "make it faster on this machine".
---

# Qputer host tuning

Repeat the optimisation pass that produced the current kernels, on this host. Speed and memory
may change; results may not. `reference.md` holds the cutoff and kernel tables, the harness,
the correctness gate, the measurement protocol and the known traps.

## Setup (once)

1. `git status` must be clean. Create branch `tune/<host>` from HEAD; the start commit is the
   baseline.
2. Fingerprint the host and record it at the top of `notes.md` in a gitignored scratch directory
   of the repository:
   - `lscpu` and the cache sizes in `/sys/devices/system/cpu/cpu*/cache/`;
   - `numactl -H`;
   - the compiler and its version;
   - `uptime`.
3. Build the harness, or reuse it if the scratch directory already holds one (reference.md,
   Harness). Build the baseline worktree with the same compiler and flags.
4. Run the gate on the untouched tree. If it fails, stop and report it: tuning cannot fix a
   broken build.
5. Measure the floors (bandwidth, FMA peak, β) and the noise band (the baseline against itself).
   Compute each cutoff's prior.

## Loop

1. Pick the scenario furthest from its floor that is not bandwidth-bound. Write the hypothesis
   in `notes.md` BEFORE editing: what limits it, what you will change, and the gain you expect.
2. Change ONE thing: a cutoff, a thread setting, a team size, a traversal or a kernel.
3. Rebuild and run the gate. If the gate fails, revert, whatever the timing.
4. Run an interleaved A/B on the affected scenarios, then on the whole suite.
5. Commit only if it gains above the noise band and no scenario slows by more than 3%. The
   commit message says what changed and the measured gain. Otherwise run `git checkout -- .`
   and write down the lesson.
6. Every 5 experiments, re-read `notes.md`. If none was kept, move to the next class of change
   (reference.md, Loop step 5).

Noether programs make good end-to-end scenarios. `noether run f.ntr --json --no-format` reports
`timing.executeMs` and `resources.peakBytes`, and the `prints` and `runs` blocks act as the
result checksum: they must be byte-identical between base and candidate with `--no-timing`.

## Rules

- NEVER edit or skip a test to make a change pass. NEVER loosen a gate tolerance.
- NEVER change system state: CPU governor, turbo, huge pages, IRQ affinity, or other people's
  processes. Suggest such changes to the user instead.
- Never trust a single run, and never compare runs taken minutes apart. Pin threads for
  algorithmic changes; use full width for placement changes.
- Commit only on `tune/<host>`. Never commit the scratch directory, and never push.
- Don't adopt a technique because it is standard HPC practice. Prototype it, measure it, and
  push back with numbers when it does not pay.

## Finish

1. Run a full interleaved A/B of `tune/<host>` against the start commit, then the gate.
2. Report:
   - base → tuned time and peak RSS per scenario;
   - every constant changed, with its prior, measured crossover and new value;
   - which workloads are at the bandwidth floor;
   - any scenario that got slower;
   - caveats: noise, and platforms you could not test.
3. Ask before merging into the user's branch.
