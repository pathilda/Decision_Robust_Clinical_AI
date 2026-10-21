# Uncertainty-aware proxy analysis: implementation and validation report

## Outcome and architecture

Implemented stochastic SAA and exact discrete budgeted robust optimization alongside
existing deterministic proxy analysis. The legacy runner and deterministic solver
were preserved. The extension separates input/grid validation, policy optimization,
checkpointed orchestration, and offline plotting, so uncertainty logic does not enter
inference or alter the original deterministic objective.

No production patient data or external prediction paths were opened for validation.
All implementation fixtures and generated reports are synthetic and repository-local.

## Files

| File | Change |
| --- | --- |
| `patient_assignment/proxy_uncertainty.py` | Versioned strict YAML grids, probability loading and separate-file ID joins, alias/column validation, source hashes and diagnostics. |
| `patient_assignment/proxy_policy_solver.py` | Stable scenario generation, exact SAA MILP, confidence-threshold discrete supports, exact robust separation/cut generation, policy-optimal random selection, and holdout statistics. |
| `patient_assignment/proxy_policy_analysis.py` | Shared point benchmark, policy grids, certified cache/checkpoints, independent verification, failed-attempt rows, atomic JSON publication, exports and stratified summaries. |
| `patient_assignment/proxy_policy_plots.py` | Offline saved-result validation, four legacy-named views, parameter sweeps, matched policy/configuration controls, diversity, training-replicate and holdout panels. |
| `patient_assignment/proxy_cli.py` | Optional policy, uncertainty-YAML and configuration-ID flags; dispatch to the extension only when requested. |
| `patient_assignment/proxy_inputs.py` | Accept probability-source metadata while keeping categorical loading independent of probabilities. |
| `patient_assignment/proxy_plots.py` | Detect extended saved runs during `proxy-plots`; preserve the original deterministic renderer. |
| `patient_assignment/tests/test_proxy_uncertainty.py` | Pure and solver-backed tests, independent enumeration oracles, explicit solver-unavailable skips, parsing, cache, failures, interruption and plotting checks. |
| `configs/proxy_uncertainty.example.yaml` | Full requested scenario-count/seed and Gamma/threshold grid. |
| `configs/proxy_uncertainty_demo/` | Eight entirely synthetic patients, two chunks, three profiles, three model IDs, a fixed proxy, custom probability joins and a small pilot grid. |
| `patient_assignment/PROXY_ANALYSIS.md` | Runnable commands, equations, schema/precedence, uncertainty assumptions, output interpretation and testing guidance. |
| `README.md` | Link to the extension guide and synthetic pilot. |
| `.gitignore` | Ignore repository-local test temporary files; `.venv` and `outputs` were already ignored. |

## Commands

The original `python run.py proxy-analysis ...` remains valid. Without new options,
it follows the unchanged deterministic runner and preserves the legacy output schema.

New options:

- `--policies deterministic stochastic robust`
- `--uncertainty-config configs/proxy_uncertainty.example.yaml`
- `--policy-configuration-ids ID ...` (IDs are printed by dry-run and saved in the grid)

The extension retains `--objective complexity-range|lexicographic`, model/chunk/profile
filters, explicit or generated selection seeds, independent benchmark/chunk/solver
seeds, solver tolerances/time limits, `--dry-run`, `--resume`, and `--cache-dir`.
`proxy-plots` regenerates saved extended runs with `--show_type Point|Box`, existing
filters, policy/configuration filters and `--facet-cols`, without invoking Gurobi.

See [the complete pilot command](PROXY_ANALYSIS.md#synthetic-pilot). On this Windows
checkout the test interpreter is `.venv/Scripts/python.exe`; there is no dependency
on its absolute location in the implementation.

## Mathematical and reproducibility guarantees

- M0 minimizes final complexity range at categorical point predictions.
- M2 minimizes empirical expected final complexity range, using **one assignment**
  shared across every training scenario. Identical scenarios retain multiplicity.
- M3 minimizes exact worst-case range over probability-threshold severity supports
  with at most G patients deviating from nominal weights. Nominal weights are always
  in support. Gamma fractions are converted with decimal `ceil(fraction * n)`.
- Robust separation enumerates clinician ordered pairs and takes the largest
  positive feasible patient gains. It is repeated for master solutions **and** random
  tertiary selections. No sampled-scenario approximation is called robust.
- Lexicographic mode first certifies count range and imposes that optimum in every
  subsequent stage. All reported policy objectives are independently recomputed.
- Policy objectives have zero intended degradation. Original objective versus global
  bound, selected policy objective and count priority must satisfy certification.
- The benchmark optimum and seeded reference allocation are fixed per chunk/profile
  and shared across every policy. Proxy regret remains point-proxy regret.
- Selection coefficients are paired across policies. Training-scenario, selection,
  algorithm, benchmark and holdout seeds have distinct roles and saved identities.
- Training scenarios depend on model/chunk/ordered patient IDs and training seed,
  not starting profile or selection seed. Holdout uses a separate RNG role and the
  original probabilities, shared across all parameters for a model/chunk.
- Duplicate optimal assignments are preserved. Random linear objectives are not
  uniform samples; observed spread is not an exact optimal-set regret envelope.
- Cache keys include source/version, model/instance, full policy/scenario/support,
  probability-source and normalized-probability hashes, solver settings and tolerance.
  Resume rejects incompatible requests. Interrupted or failed solves are not cached.

The stochastic model assumes independent categorical patient marginals. The robust
set is a discrete cardinality budget, not a probability guarantee or a claim to be
the original continuous Bertsimas-Sim model. Neither model probabilities nor the
proxy labels are asserted to be calibrated clinical ground truth.

## Tests actually executed

All solver tests below ran with Gurobi; they were not skipped.

1. `python -m pytest patient_assignment/tests clinical_target_extraction/tests/test_probabilities.py -q`
   **216 passed**, with 10 existing Matplotlib pending-deprecation warnings.
2. After final decimal-rounding, empty-failure-schema and atomic-publication refinements:
   `python -m pytest patient_assignment/tests/test_proxy_uncertainty.py patient_assignment/tests/test_proxy_analysis.py -q`
   **70 passed**. This overlaps the combined suite; the counts should not be added.
3. Ruff checks on all new/modified Python implementation files and the new tests:
   **passed**. `git diff --check`: **passed**.
4. `proxy-analysis --help`: verified the new CLI arguments.

Tests used a repository-local pytest `--basetemp`. Probability-schema tests used
Hugging Face offline mode; no models were downloaded or inference run. The environment
used Python 3.12.10, Gurobi 13.0.3, pandas 3.0.6 and Plotly 7.1.0.

Coverage includes independent full enumeration of tiny assignments and discrete
uncertainty sets; nonuniform weights and initial loads; Gamma=0 and monotonicity;
single-scenario SAA; lexicographic priorities; separate seed roles; different optimal
assignments with different proxy loss; policies that must leave the nominal-optimal
set; empirical CVaR boundary mass; invalid/aliased/separate-file probabilities;
source-hash changes; cached-result verification; forced interruption and resume;
failed-only exports; solver API/license error conversion; and offline Point/Box plots.

## Synthetic experiments

- Three model fixtures x two chunks x three profiles x four policy cells x three
  selection seeds: **216/216 certified allocations**.
- One model/chunk/profile, all 23 full-grid policy cells, 30 selection seeds:
  **690/690 certified allocations** (30 deterministic, 360 stochastic, 300 robust).
- The sweep covers scenario counts 20/50/100/200, training seeds 42/123/2026,
  Gamma fractions 0/.1/.25/.5/1, and thresholds .05/.2.
- Export audit confirmed one common benchmark reference, one common holdout set,
  and nonnegative proxy regret throughout that sweep.

Final-source reports are written under `outputs/policy_comparison_demo/` and
`outputs/policy_parameter_sweep_demo/`; open each `index.html`. Generated HTML embeds
Plotly, and these output directories are ignored by Git. Earlier development test
artifacts are also confined to ignored repository-local directories.

## Remaining limitations

- No production clinical dataset, production-scale timing, or scientific calibration
  validation was performed. Large exact robust grids can need many master solves and
  a full Gurobi license. Time limits apply per optimize call, not per entire experiment.
- Only independent categorical sampling, empirical scenario weights, probability-
  threshold supports with nominal inclusion, and ceil budget rounding are supported.
  Unsupported schemes and invalid probability rows fail explicitly; scalar confidence
  is never silently expanded to a distribution.
- No exact regret envelopes, clinical statistical significance, calibrated risk
  guarantees, or human-label probability scores are claimed. Paired differences and
  training-replicate summaries are descriptive.
- Output charts were verified through generated artifacts and offline regeneration
  tests; there was no browser automation or visual screenshot review in this environment.
