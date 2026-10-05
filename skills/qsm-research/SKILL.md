---
name: qsm-research
description: Run an autonomous algorithm-research loop on a Noether task spec. Edit the candidate algo.ntr, evaluate it with noether eval, keep improvements in git and discard the rest, until the spec's goal or budget is reached. Use when asked to "research", "optimise", "search for" or "iterate on" a quantum algorithm against a target, or when pointed at a research/<tag>/ directory.
---

# Noether research loop

Requires the `qsm` skill for syntax and traps. A workspace holds:

- `spec.ntr`: the frozen task;
- `spec.lock`: its hash;
- `program.md`: the human's brief;
- `algo.ntr`: the candidate, the ONLY file you edit;
- `params.json`: the best params, written by eval;
- `results.tsv`: the ledger;
- `notes.md`: your hypotheses and lessons;
- `runs/<n>.json`: one file per evaluation.

## Reading the spec

```
task "toffoli-tdepth"
qubits q[3]
candidate def Toffoli3_{q[3]}            # what algo.ntr must define; any register name works there
target unitary Toffoli_{q[0], q[1]→q[2]} # or a target state
gates {H, S, S†, T, T†, CNOT}            # the candidate is lowered into this basis
coupling all                             # or line, ring, grid, a set of pairs: allowed 2-qubit gates
require fidelity ≥ 0.999999              # feasibility
minimize tdepth                          # objectives, compared lexicographically in file order
minimize tcount
goal tdepth ≤ 3                          # status.stop once met by a feasible candidate
budget experiments=40, wall=30m
```

- Metrics: `fidelity`, `count2q`, `depth`, `depth2q`, `nops`, `nparams`, `tcount`, `tdepth`.
- For a unitary target, `fidelity` is the average gate fidelity, up to global phase.
- `require` failures make a candidate infeasible; it can never meet the goal.

## Setup (once)

1. If the workspace does not exist yet, the human runs
   `noether research init <tag> --spec specs/<task>.ntr [--from start.ntr]`. Then create the
   branch: `git switch -c research/<tag>`.
2. `cd research/<tag>`. Read `program.md`, `spec.ntr`, `algo.ntr`, `notes.md` and the last 20
   lines of `results.tsv`.
3. `git status`: the branch must be `research/<tag>` and the tree clean.
4. `noether research status --json`. If `stop` is true, go to Finish.
5. If `results.tsv` has only the baseline, first write the simplest correct candidate and
   record it. A feasible baseline matters more than a clever first idea.

## Loop (repeat until `status.stop`)

1. Choose ONE hypothesis. Draw on the hints in `program.md`, the `pareto` rows, and what
   `notes.md` says has not been tried. Write the hypothesis in `notes.md` BEFORE editing.
2. Edit only `algo.ntr`. It may contain only the header, `import`, `let`, `param`, `def` and
   `proc`. Make continuous quantities `param`s: the inner optimiser fits numbers, and your
   experiments are for structure.
3. `noether check algo.ntr --spec spec.ntr --json`. Fix every diagnostic (E8002 forbidden
   statement, E8005 signature, E8007 forbidden gate, E8003 not decomposable). Compile fixes do
   not count as experiments.
4. Optional, cheap: `noether draw algo.ntr`.
5. `noether eval --dir . --record "<one-line hypothesis>" --json`
6. If the verdict is `improved`:
   `git add algo.ntr params.json && git commit -m "<tag>: <hypothesis> (<key metrics>)"`.
   For any other verdict: `git checkout -- algo.ntr params.json`.
7. Append the outcome and the lesson to `notes.md`: why it helped or didn't.
8. Every 10 experiments, re-read `results.tsv` and `notes.md`. If none of the last 10 improved,
   switch to a different strategy class rather than micro-tweaking:
   - a different known construction;
   - a different symmetry;
   - a different ansatz family;
   - a reordering that exploits commutation.

## Reading eval

- `key` = (0 if feasible else 1, violation, objectives…, nops, candidate tokens). Lower is
  better. `improved` means strictly below the best recorded key, so infeasible candidates still
  improve by reducing violation, and a simpler program with the same score counts as an
  improvement.
- Metrics are computed on the circuit lowered into the spec's `gates`. A gate hidden in a
  `def` still counts.
- `violations` lists coupling breaks (E8004), which make the candidate infeasible. There is no
  automatic SWAP insertion.

## Rules

- NEVER edit `spec.ntr`, `spec.lock`, `results.tsv` or `.gitignore`. Never run
  `noether research report`: it reads the holdout instances.
- One idea per experiment, so every ledger row can be attributed.
- An error, timeout or crash is logged by eval: revert and move on. If the same crash repeats
  3 times, record it in `notes.md` and avoid that construct.
- Do not stop to ask the human mid-loop. Stop only when `status.stop` is true: the goal is met,
  or the experiment or wall-clock budget is used up.
- Prefer simpler candidates.

## Finish

1. `noether research status --json`.
2. Add a summary to `notes.md`:
   - the best key;
   - whether the goal was met;
   - the 3–5 lessons that moved the score;
   - what to try next.
3. Report to the user:
   - the best metrics against the goal;
   - the number of experiments;
   - the best commit;
   - `noether research report` as their next step (it runs the holdout).
