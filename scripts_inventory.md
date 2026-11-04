# Verification-script inventory

Scope: the 11 scripts requested in the verification-hygiene task (4 November 2026).
V means verification; G means data/figure generation without asserted scientific claims.
All scientific checks below concern project-derived claims **(c)** in the existing
write-ups; numerical checks establish only the stated finite samples and tolerances.
They do not establish the unresolved physical quantum frame dictionary **(d)**.

Read: `AGENTS.md`, `README.md`, `phase0/literature_map.md`, `phase1/review.tex`,
`phase2/toda_route.tex`, and `phase2/toda_edge.tex`. No new literature references
were introduced or verified independently.

All development and execution took place in `/tmp/hygiene_clone`, created with
`git clone --no-hardlinks /home/gustav/Git/quantum_boussinesq_solitons /tmp/hygiene_clone`.
Each script ran as `python3 -u <filename>` from its own script directory, with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MPLCONFIGDIR=/tmp/hygiene_mpl`. Runtimes are measured wall-clock seconds, including
imports, the existing computation/output generation, and added checks. Heavy runs
were sequential. Python 3.12.3; NumPy 1.26.4; SciPy 1.11.4; SymPy 1.12;
mpmath 1.2.1. Logs and timing records are retained under `/tmp/hygiene_logs`.

| Script | Class | Claims now asserted / generator purpose | Runtime (s) | Outcome |
|---|---|---|---:|---|
| `phase0/checks/convention_check.py` | V | Nonlinearity coefficient is exactly `c=3s` (good: −3; bad: +3); existing symbolic PDE residual vanishes for all theta. | 1.88 | PASS (exit 0) |
| `phase1/scripts/hirota_solitons.py` | V | Replaces disabled one-soliton sech identity by exact exponential rewriting; additionally asserts the write-up's two-soliton residual < 1e-38 at unchanged 40-digit precision, complex N=3 Hirota residual < 1e-12, symbolic interaction-factor identity and sampled agreement < 1e-12. Existing rational N=2,3,4 direction checks remain. | 9.19 | PASS (exit 0) |
| `phase1/scripts/clarkson_dowie_map.py` | V | The specific real map `(a,b,c,d)=(3,1/sqrt(3),1/3,1/sqrt(3))` occurs among candidates and transforms the equation exactly to nine times the target equation. | 1.28 | PASS (exit 0) |
| `phase1/scripts/figures.py` | G | Generates the review's interaction-regime and dispersion figure; no checks invented. | 2.94 | Generated (exit 0) |
| `phase2/scripts/toda_bethe_quantum.py` | V | Density normalization and discrete integral equation residual < 1e-10; finite errors in edge q, dressed momentum, and edge-relative energy decrease at each halving of h; finest error ratios are within 0.1 of 1/2, testing the reported leading linear convergence. | 46.36 | PASS (exit 0) |
| `phase2/scripts/toda_edge_ba.py` | V | Standalone test at the existing nodecheck configuration `(a,h,N)=(1,0.04,700)`: positive density, density normalization and discrete equation residual < 1e-10; physical momentum inversion on both sides < 1e-12; four-term phase vanishes when either soliton is at its edge, in both channels, within 1e-10. Imports remain module-only. | 2.25 | PASS (exit 0) |
| `phase2/scripts/toda_edge_grid.py` | G | Generates the exploratory overtaking four-term grid, explicitly superseded in toda_edge.tex; no candidate-agreement checks invented. | 19.82 | Generated (exit 0) |
| `phase2/scripts/toda_edge_kappa.py` | V | Finite linear extrapolations; kappa−eta and original/V1/V2 residual ranges quoted in conclusion 3 of toda_edge.tex. Two-decimal range endpoints allow 0.005 rounding; original residual bound remains 0.11. | 824.13 | PASS (exit 0) |
| `phase2/scripts/toda_edge_nodecheck.py` | V | All results finite; full spread over N=700,1000,1400 < 1e-10 for q and < 1e-8 for both phases (one last displayed digit). | 17.31 | PASS (exit 0) |
| `phase2/scripts/toda_edge_twoterm.py` | V | All eight extrapolated Toda/candidate/semiclassical/residual table rows within 1e-4; residual depends on y1 at fixed y2 and changes sign; successive raw Toda phase changes < 0.004, exactly as claimed in the write-up. | 822.00 | FAIL: stated step bound (exit 1) |
| `phase2/scripts/toda_edge_scan.py` | V | For all three head-on pairs, linearly extrapolated Toda, candidate, and semiclassical phases agree with `4*sqrt(y1*y2)` within the write-up's 1e-4 bound. Superseded overtaking results remain exploratory. | 124.89 | PASS (exit 0) |

Tolerances: symbolic comparisons are exact. The two-soliton 1e-38, head-on
1e-4, and successive-step 0.004 limits come directly from the write-ups.
Table values have four decimals; 1e-4 covers rounding of both values and their
reported difference. Kappa ranges have two decimals, hence half a last-place
unit at their rounded endpoints. Mesh tolerances follow the displayed precision.
Complex Hirota/sample-factor checks use 1e-12 for cancellation in double
precision (the three seeded complex residuals were 2.39e-15, 2.22e-16,
and 6.75e-16 in a focused clone preflight); they do not purport to be exact complex proofs. Solver residual
bounds allow floating-point cancellation and the existing root stopping criteria.
The convergence-ratio test is a finite-h leading-order check, not a proof of
an asymptotic series or of the dispersion coefficients tested by other scripts.

## Failures recorded

- **(c) `toda_edge_twoterm.py`: FAILED**, exit code 1 after 822.00 s. All eight
  extrapolated table rows, residual dependence at fixed y2, and residual sign
  checks passed first. The literal `toda_edge.tex` claim that successive raw Toda
  values change by less than 0.004 is false for these three steps:

  | (y1,y2) | h before | h after | Absolute Toda phase change |
  |---|---:|---:|---:|
  | (3,1) | 0.01 | 0.005 | 0.006326021899488055 |
  | (6,1) | 0.01 | 0.005 | 0.009016412921681649 |
  | (6,1) | 0.005 | 0.0025 | 0.004511945128394146 |

  The bound was not relaxed. The script finishes its original outputs and cache
  writes before asserting this claim, then exits nonzero without a success marker.
  This contradicts the stated finite-step bound, not the separate fact that the
  extrapolated table is reproduced. The same violations were also visible in the
  committed cache before rerunning.

- No other assertions failed in the full executions or the focused complex-sample
  preflight. No failing check was weakened or removed. Eight V scripts passed;
  two G scripts generated data. The one failed V script is intentionally retained
  as a failing verification of the write-up's bound.

## Numerical evidence for the tolerances

- **(c)** The two-soliton residuals at the unchanged 40-digit precision and the
  symbolic profile/interaction identities passed in the full Hirota execution.
  The measured PDE residuals were 3.8869e-40 and 5.8492e-39.
- **(c)** All three node counts print the same q to ten decimals and both phases
  to eight decimals; the added full-spread assertions pass at those precisions.
- **(c)** From the printed finite-hbar Bethe data, the finest error-halving
  ratios are approximately 0.543 (q), 0.500 (momentum), and 0.500 (energy),
  inside the stated 0.5 ± 0.1 leading-order test window.
- **(c)** The kappa run gives `kappa-eta` in [-0.453895305, -0.028636980].
  Absolute residual ranges are V0 [0.001282963, 0.109613414],
  V1 [0.165823958, 0.862394184], and V2 [0.280921044, 0.489922580].
  These explain why rounded two-decimal endpoints need a 0.005 rounding allowance;
  the unrounded original-candidate upper bound 0.11 remains satisfied.
- **(c)** Head-on extrapolated Toda phases print as 2.00000, 4.00002, and 4.00000.
  Independently, each extrapolated candidate and semiclassical phase passes the
  1e-4 comparison with `4*sqrt(y1*y2)`. These are fresh executions of the current
  source, not tests of the stale head-on candidate values in the committed cache.

## Preservation and markers

Only assertions, their necessary test bookkeeping, comments, and terminal prints
were added, except for the explicitly requested replacement of the disabled
Hirota assertion. A structural AST comparison verified that all other original
statements remain in order and unchanged. No solver parameters, precision settings,
formula definitions, random seeds, cache schemas, or figure-generation calls changed.
V scripts print `ALL CHECKS PASSED` only after all assertions succeed; G scripts
end with `DATA GENERATED (no checks)`. An assertion failure intentionally prevents
the success marker and returns a nonzero exit code.

Only the 11 edited Python files and this inventory are delivered to the original
repository. Regenerated JSON caches, figures, bytecode, and logs stay in scratch.
No pull, commit, stash, branch switch, index update, or push was performed in the
original checkout. This task's explicit constraints override the general README,
build, and Git-update instructions in AGENTS.md.
