# Assignment robustness experiments

These commands consume existing prediction tables. They do not load inference engines,
change prompts, or run extraction. Legacy `chunk`, `solve`, `compare-assignments`, and
`compare-all` retain their defaults and output schemas. In particular, legacy random
selection remains solver-seed sensitivity; the new command uses independent random
linear objectives restricted to a certified optimal set.

## First experiment (PowerShell, from the repository root)

The checked-in example contains **synthetic** scores for eight patients, three input
configurations, two chunks, three conserved-load profiles, construction scenarios,
and separate held-out scenarios. It is small enough for a restricted Gurobi license.
Use the existing virtual environment; the new dependency is `PyYAML>=6` (also declared
in `pyproject.toml`). On a new analysis-only environment:

```powershell
python -m pip install pandas openpyxl xlrd PyYAML matplotlib scikit-learn gurobipy pytest
```

Validate, run all three strategies and both objectives, then plot:

```powershell
.venv\Scripts\python.exe run.py assignment-experiment --config configs/assignment_demo/experiment.yaml --dry-run
.venv\Scripts\python.exe run.py assignment-experiment --config configs/assignment_demo/experiment.yaml --output outputs/assignment_first
.venv\Scripts\python.exe run.py assignment-plots --results outputs/assignment_first --plots regret-envelope profile-effect assignment-regret robustness --format png pdf --output outputs/assignment_first/figures
```

The example requests 30 random selections, one matched pair, one robust selection,
and exact regret bounds per candidate/instance/objective. For a shorter first pass add
`--samples 5 --profile-ids balanced` to the experiment command. Choose a new output
directory for a different experiment. To reuse an identical completed run:

```powershell
.venv\Scripts\python.exe run.py assignment-experiment --config configs/assignment_demo/experiment.yaml --output outputs/assignment_first --resume
```

Dry-run reports memberships, exclusions, feasibility, missing robust inputs, and planned
optimization-stage counts before caching. It does not create output or initialize Gurobi.
Plotting reads saved CSV/JSON only and also works without a Gurobi installation/license.

## Use existing extractions

Copy `configs/assignment_existing.example.yaml`, replace its input and chunk paths, and
supply an operational profile with adequate capacity. The paths in that template are
placeholders, not files expected to exist in this checkout. Existing prediction columns
are `Client AlayaCare Client ID` and `treatment_category`; the repository spells categories
`Easy to treatment`, `Moderate to treatment`, and `severe to treatment`. CSV and Excel
files are accepted; Excel `sheet` is optional (first sheet by default).

Use an existing `chunks.csv` from `run.py chunk`, its `chunk_config.json`, or a JSON pointer
`{"schema_version": 1, "membership": "chunks.csv"}`. Optional discipline filtering stays
in the existing `chunk --discipline PT` / `OT` / `SLP` workflow, so all subsequent models
share exactly that frozen membership. Patient IDs use the existing normalization, remain
strings, and preserve text leading zeros. Membership is sorted, never reshuffled.

YAML paths resolve relative to the YAML file that contains them; CLI paths resolve relative
to the current working directory. Scalar CLI options override YAML. Selection lists from
the CLI replace YAML lists (multiple CLI occurrences concatenate); duplicate selected IDs
are errors. All substantive experiment options are also top-level YAML keys with underscores,
for example `optimality_tolerance`, `sample_seed`, `nominal_slack`, `regret_bounds`, and
`stress_test_manifest`. Benchmark overrides use `--benchmark` and `--benchmark-kind`, with
YAML representation `benchmark: {id: ..., kind: proxy}`. Unknown keys and duplicate YAML keys
are errors. Run each subcommand with `--help` for the full option list.

Each registered input requires `id`, `path`, `id_column`, `value_column`, and
`value_type: category|numeric`. `category_aliases` under `scores` explicitly maps alternative
spellings to category-weight keys; there is no inferred spelling/meaning conversion.
Numeric scores must be finite and nonnegative. The default common scale/bounds are the
repository's 1--3 weights. Set `scores.score_min`, `scores.score_max`, and `scores.scale`
explicitly for other units; all inputs and initial loads must use those units. An input's
optional `scale` must equal the common scale. Optional numeric
`normalization: {multiplier: 1, offset: 0}` is applied before checking common bounds and is
saved in provenance. Category normalization instead uses explicit shared category weights.

Duplicate normalized prediction IDs are always rejected, including collisions outside
selected chunks. Strict coverage rejects selected missing/invalid/failed records.
`--coverage intersection` removes each affected patient from **every** compared input,
including requested scenario files, and exports exclusions and effective memberships.
Empty/reduced chunks are flagged, not repacked. Mixed `model`, `prompt_style`, `run_id`,
`configuration_id`, or `request_id` values within a file are rejected when those columns
exist. Custom identity columns can be declared with `request_id_column` and `request_id`.
Shared note hashes are compared across inputs/scenarios. File provenance cannot establish
request identity when the original export did not record it; use one verified extraction
per file, not concatenated stale results.

The benchmark is reoptimized for each profile, chunk, and objective. Switching it does
not require extraction:

```powershell
.venv\Scripts\python.exe run.py assignment-experiment --config configs/assignment_demo/experiment.yaml --benchmark synthetic_anchor --output outputs/assignment_alternate_benchmark
```

AI references are labeled **proxy-benchmark regret** and **agreement with proxy benchmark**.
They do not establish clinical accuracy or quality. A clinical reference requires actual
human reference data and provenance on its registry input:

```yaml
provenance:
  kind: human
  source: "Reference dataset or adjudication record identifier"
  description: "Who assessed the patients, with which rubric and adjudication procedure"
```

The software requires this declaration; it cannot independently authenticate its truth.
Use `benchmark: {id: human_reference, kind: clinical}` only for such data. Category disagreement
and ordinal quadratic weighted kappa are reported for categorical comparisons only; kappa
is blank when mathematically undefined. Numeric scores are never silently binned.

## Operational profiles

Explicit profile files are authoritative. Each clinician has an ID, initial count, initial
aggregate complexity, and its own final count capacity. Initial patients exclude the chunk.
Counts must be nonnegative integers within capacity; zero counts require zero load. Loads
must respect the shared additive score bounds. An insufficient-capacity instance is recorded
as infeasible and excluded from metric summaries; the run exits nonzero and discloses counts.
No patients are silently truncated. No clinician eligibility constraints beyond capacity are
added; the existing system also assumes all-to-all eligibility.

Generate conserved-load experimental profiles:

```powershell
.venv\Scripts\python.exe run.py make-profiles --clinician A 10 20 --clinician B 10 20 --clinician C 10 20 --clinician-capacity 25 --imbalance-factors 0 0.5 1 --profile-seed 42 --score-min 1 --score-max 3 --output configs/generated_profiles.yaml
```

The generator preserves counts, capacities, and total initial load. It uses evenly spaced
centered directions, a seeded permutation, and the largest amplitude consistent with the
bounds. Equal initial counts and a feasible equal-load baseline are required. Bounds default
to 1--3; supply clinically intended finite bounds for another scale. Zero amplitude is
explicitly flagged as identical profiles. Values are not rounded or independently clipped.
The file is created exclusively and an existing file is never overwritten. Fractional initial
aggregates are allowed; integer-realizable projection is not implemented. Use explicit profiles
if initial patient categories/observed operations impose additional constraints. Changing score
units requires compatible initial aggregates. Generated labels are experimental, not clinical
definitions of balance.

## Objectives, selection, and numerical certification

`complexity-range` minimizes the range of final clinician aggregate loads, with count capacity
constraints only. `lexicographic` first certifies the minimum count range, then minimizes load
range without degrading count balance. The count restriction remains in every downstream solve.
The sum of loads is invariant and is never used as the balancing objective.

Every original optimum must have Gurobi `OPTIMAL` status **and** an objective/bound difference
at most `optimality_tolerance` (default `1e-6`). Status `OPTIMAL` from a loose `mip_gap` alone
is insufficient. Defaults are `MIPGap=0`, `MIPGapAbs=0`, feasibility/integrality/solver-optimality
tolerances `1e-9`, one thread, unlimited time, solver seed zero. Settings, bounds, achieved gaps,
statuses and timings are saved. `--time-limit` is per solve stage. Gurobi uses the existing license
configuration, including `GRB_LICENSE_FILE`; the code does not inspect or rewrite license files.

The nominal set uses `range <= certified_range + optimality_tolerance`. Ranges are recomputed
from assignments for all exports. Integer score/load levels are separated by the small absolute
tolerance without rescaling. Fractional scores use that documented absolute tolerance; choose
units with a numerically meaningful scale. There is no automatic score quantization. Numerical
tolerance is distinct from intentional `--nominal-slack`, which only relaxes robust selection's
complexity restriction. Positive slack is labeled **near-optimal robust** and never relaxes the
lexicographic count optimum. Exact regret envelopes always concern the original zero-slack set.

* `random`: independently draw Uniform(0,1) assignment coefficients and minimize their weighted
  sum inside the optimal set. SHA-256-derived seeds use the master seed, chunk/profile IDs,
  objective, candidate ID/canonical content, strategy role and sample index. They are independent
  of model-list order and benchmark choice. All draws, including duplicates, are retained. This
  is **not** uniform sampling of the optimal set. Benchmark samples use an independent role,
  are cached once, and sample `r` is paired with candidate `r` without minimizing disagreement.
* `matched`: jointly optimize two original-optimal assignment copies and same-clinician AND
  variables to minimize Hamming disagreement with fixed clinician identities. This is a separate
  benchmark-informed diagnostic, not a deployable policy or minimum-regret rule.
* `robust`: minimize worst scenario load range over the entire nominal-optimal feasible set,
  then draw seeded random tertiary objectives within the certified robust optimum. It is not a
  search over sampled allocations. `--robust-samples` explores residual robust ties. Failed robust
  solves never fall back to random selection.

Regret is the candidate assignment's benchmark-evaluated range minus the benchmark optimum
for that same objective/domain. Raw tiny negative values are retained and display values clamp
to zero; larger negatives fail verification. Regret does not depend on the benchmark assignment
chosen, while assignment disagreement does. The self-proxy case has zero optimal-set regret
up to tolerance but independent samples can disagree on patient assignments.

`--regret-bounds exact` minimizes benchmark range over the original candidate-optimal set, then
maximizes each distinct ordered clinician load difference to obtain the exact maximum. One
clinician has zero range. Witness assignments and every solve status/bound are saved. Incomplete
envelopes have `complete=false`; their extrema are incumbents, not exact endpoints. They are not
confidence intervals and are never silently replaced with sampled extrema.

## Uncertainty and robustness

See the example `uncertainty.yaml` for per-candidate file scenarios. All vectors cover the same
effective patients on the common scale. The nominal vector is always included; identical vectors
are deduplicated for solving, with multiplicities and source provenance retained. Initial loads
stay fixed. Random, matched, and robust allocations are evaluated without reoptimization on
exactly the same candidate-specific vectors. Optional `stress_test_manifest` uses the same schema
for independent, held-out evaluation only. `stress_w` and `heldout_w` are exported separately.

For an explicit synthetic stress test, use a configuration with no uncertainty file manifest:

```yaml
uncertainty_manifest: null
uncertainty_source: perturbations
perturb_fraction: 0.10
uncertainty_samples: 30
uncertainty_seed: 42
```

Select only categorical models in this mode. Each vector changes exactly `ceil(fraction*N)`
patients, sampled without replacement, by one legal neighboring ordinal category (uniformly
between two neighbors where available). Numeric inputs require file scenarios. Generated vectors,
actual changes, seeds, and multiplicities are persisted. These are transparent stress tests,
not calibrated uncertainty probabilities; repeated model outputs may share systematic errors.

Construction scenarios matching the benchmark or declared human references by resolved path,
file hash, or canonical content/score fingerprint are rejected. This conservative check also
rejects independently exported files with identical complete score vectors. The explicit
`--allow-benchmark-in-uncertainty` override labels affected candidates **ORACLE diagnostic** and
excludes them from `deployable_summaries.csv`. Matching is also excluded from that table.
Held-out references may be used for evaluation; they never enter the robust objective.
Robust selection optimizes in-set worst range; reduced benchmark regret is an empirical outcome,
not a guarantee. Directly compare W across candidates only on a shared evaluation stress set.

## Output, resume, and plots

Runs create `resolved_config.json`, `instances.csv`, `solutions.csv`, `assignments.csv`,
`clinician_loads.csv`, `regret_bounds.csv`, `regret_witnesses.csv`, `summaries.csv`,
`deployable_summaries.csv`, `benchmark_samples.json`, `uncertainty_manifest.json`,
`uncertainty_scores.csv`, `effective_membership.csv`, `exclusions.csv`, `validation_report.json`,
and `run_status.json`. Each solution has model, objective, chunk, profile, strategy, and sample IDs,
an assignment hash, nominal/benchmark/count ranges, regret, disagreement, W, certification and
timing. Witness rows are separate from selection rows. Scenario evaluations are not counted as
additional assignments. Package/solver versions, source hashes, all options, effective memberships,
resolved profiles, and uncertainty vectors are part of the immutable run manifest.

Within-instance draw summaries precede across-chunk means. Across-chunk means give equal weight
to chunks and stratify by effective chunk size; profiles and strategies remain separate. Unique
allocation counts retain duplicate-draw frequencies. W is summarized within a fixed instance and
stress set, not pooled across different sets. No bootstrap intervals or claims of independent
clinical replication from draws/profiles are made. Alternative overlapping chunk partitions are
sensitivity analyses.

Existing nonempty output directories are refused by default. Resume requires the same fingerprint
and verifies manifest/artifact checksums. An incomplete compatible run retries failed components
and regenerates derived tables; only completed content-addressed components are reused. The
default component cache is `.assignment_cache` next to the output directory; `--cache-dir` can
choose a shared location. Cache keys include implementation/package versions, scores, conditions,
selection parameters and tolerances. Nominal/random/robust keys omit benchmark dependencies;
matching keys include them. Benchmark-dependent metrics and regret envelopes are recomputed on
a benchmark switch. Historical inference outputs are never deleted or replaced. Cached solve
timings describe the original solve, and run status discloses cache hits/misses.

All four plot families are implemented: exact regret envelopes with sample/robust markers,
paired profile effects (regret, actual benchmark range, count range), assignment disagreement
versus regret, and paired random-mean versus robust W/regret (held-out W separately). Matched
comparisons are separate. Near-optimal robust plots omit original-optimal envelope overlays.
Plot filters are `--models`, `--profile-ids`, `--objectives`, `--selection`; pagination uses
`--max-instances-per-figure`. No dots are subsampled, so `plot_seed` is recorded but inactive.
Threshold lines appear only for explicit `--regret-threshold` or `--disagreement-threshold`.
Plot directories must be new/empty. `plot_manifest.json` records source hashes and plot options.

## Verification and current limits

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.venv\Scripts\python.exe -m pytest patient_assignment/tests -q
.venv\Scripts\python.exe -m pytest -q
```

The environment variable avoids unrelated, globally installed pytest plugins. Real solver tests
require an available Gurobi license and otherwise explicitly skip. New tests enumerate tiny
feasible sets for both objectives, random/matched/robust restrictions, exact regret extrema,
one-clinician/identical-score/fractional cases, fixed-label joint Hamming distance, full-set minimax,
and fixed-assignment stress arithmetic. They also cover self-proxy ties, conserved profiles, row
and model ordering, benchmark independence, invalid inputs, leakage, infeasibility, interrupted
certification, stale cache/resume, sample aggregation, and plotting without solver initialization.

Not implemented: integer projection of generated initial totals, uncertain initial loads,
patient-specific eligibility beyond existing capacity rules, automatic clinical scale inference,
calibrated scenario probabilities, optional large per-scenario clinician-load tables, and bootstrap
intervals. Large exact envelopes need one minimization plus J*(J-1) maximizations per candidate
instance/objective and may be expensive. Outputs are accumulated in memory; very large experiments
should be split using chunk/profile filters. Profile/scenario metadata is user supplied, not verified
clinical evidence.
