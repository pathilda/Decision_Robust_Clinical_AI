# Proxy benchmark analysis of saved LLM assessments

This workflow compares categorical assessments and their patient allocations with an
explicitly selected **proxy benchmark**, not human ground truth. It never runs LLM
inference. Existing classification and assignment commands keep their behavior.

Install the optional analysis dependencies using the same environment and Gurobi
license configuration used by the existing allocation tools:

```bash
python -m pip install -e ".[proxy]"
```

## Runnable synthetic demonstration

The checked-in `configs/proxy_demo/` contains 12 invented patient assessments,
two fixed, nonoverlapping chunks, two candidates, a proxy benchmark, and the three
specified starting profiles. Initial counts are 10, capacities 25, and total
initial complexity 60 in every profile; only its distribution changes. These are
experimental labels and synthetic values, not clinical findings or defaults for
an actual experiment. Use a new output directory for each new run.

```bash
python run.py proxy-analysis --config configs/proxy_demo/inputs.yaml --models qwen_base qwen_fewshot --benchmark medgemma_base --chunks configs/proxy_demo/manifest.json --profiles configs/proxy_demo/profiles.yaml --n-samples 30 --sample-seed 42 --benchmark-seed 1001 --chunk-seed 42 --show_type Point --output outputs/proxy_demo_point
python run.py proxy-plots --results outputs/proxy_demo_point --show_type Box --output outputs/proxy_demo_box
```

Open either directory's `index.html` locally. Each report links to all four figures;
each figure embeds Plotly JS and works offline without a server or Dash. The Point
run contains 360 allocation rows; the fixed-seed figures contain 12 observations.
Repeated allocations are retained. `validation_report.json` records distinct
allocation counts separately. `--dry-run` validates and prints expected solve-stage
counts before cache reuse without creating outputs or initializing Gurobi.

## First actual-data run

Copy and edit `configs/proxy_inputs.example.yaml` to point to your existing exports.
Use the exact configuration IDs in that registry; the benchmark cannot be a candidate.
Provide your explicit initial clinician state in a profile file with the three IDs
`balanced`, `unbalanced`, `highly_unbalanced`. Do not use the synthetic profile values
as actual starting conditions without making that experimental choice explicitly.

```bash
python run.py proxy-analysis --config configs/proxy_inputs.example.yaml --models qwen_base qwen_fewshot --benchmark medgemma_base --chunks data/chunks/chunks.csv --profiles configs/proxy_profiles.yaml --objective complexity-range --n-samples 30 --sample-seed 42 --benchmark-seed 1001 --chunk-seed 42 --threads 1 --solver-seed 0 --show_type Point --output outputs/proxy_real_point --dry-run
python run.py proxy-analysis --config configs/proxy_inputs.example.yaml --models qwen_base qwen_fewshot --benchmark medgemma_base --chunks data/chunks/chunks.csv --profiles configs/proxy_profiles.yaml --objective complexity-range --n-samples 30 --sample-seed 42 --benchmark-seed 1001 --chunk-seed 42 --threads 1 --solver-seed 0 --show_type Point --output outputs/proxy_real_point
python run.py proxy-plots --results outputs/proxy_real_point --show_type Box --output outputs/proxy_real_box
python run.py proxy-plots --results outputs/proxy_real_point --show_type Point --models qwen_base --starting-parameters balanced --chunk-ids chunk_0001 --facet-cols 1 --output outputs/proxy_real_filtered
```

`configs/proxy_profiles.yaml` and the actual prediction/chunk paths in these commands
are user-supplied files. The demo commands above run without editing input files.
Registry paths resolve relative to the registry; CLI paths resolve from the working
directory. CSV, Excel and Parquet predictions use explicit `id_column` and `category_column`;
Excel supports `sheet`. Text IDs retain leading zeros. Store IDs as text in Excel
when zeros matter. The category alias mapping accepts the repository's exact category
spellings. Numeric/continuous assessments cannot be silently binned or rounded.
The allocation weights are the explicit category weights, even if a file also
contains continuous scores. Blank, duplicate, missing or failed selected records
cause validation failure; extra unselected patients are not included.

Use the existing `chunk` command's saved `chunks.csv` unchanged. JSON manifests may
contain `{"schema_version":1,"membership":"chunks.csv"}` (optional `sha256`) or
`{"schema_version":1,"chunks":{"group_1":["001","002"]}}`. Overlaps are rejected,
including overlaps outside a requested subset. No implicit partitioning occurs.
`--chunk-ids` selects existing chunks. `--starting-parameters` selects profile IDs
for either analysis or plotting. The profile file must still define all three
profiles, using a mapping as in the demo or the existing generator's list format.
Initial loads must lie between initial count times minimum/maximum category weight.
All profiles use the same clinician identities; capacity is checked per instance.

## Objectives, randomness and interpretation

For each clinician, final complexity is initial complexity plus assigned assessment
weights. The default `--objective complexity-range` minimizes the maximum minus
minimum final complexity under final count capacities. `lexicographic` first
minimizes final count range, then complexity range, and retains the count restriction
through every subsequent selection.

For each nominally optimal set, a selection minimizes independent Uniform(0,1)
coefficients times assignment variables. Patient and clinician IDs are sorted.
SHA256 of the canonical instance, selection seed and purpose derives the local
Python MT19937 stream. Candidates share coefficients within an instance/seed.
Neither candidate IDs/order nor the benchmark enters the candidate RNG stream.
The benchmark has a separate purpose tag and one `--benchmark-seed` selection per
chunk/profile, shared across every candidate and draw. No clinician relabeling,
reference matching or solver-seed substitution occurs.

`--n-samples` defaults to 30 and `--sample-seed` to 42; seeds are consecutive.
Alternatively use `--selection-seeds 12 37 99` without explicitly specifying either
of those options. Selection seeds are unique integers (negative values are supported).
`--chunk-seed` defaults to 42: it reuses that draw if present, or adds one selection
tagged only for cross-chunk evaluation. The sample index is zero-based.
`--solver-seed` controls Gurobi's algorithm, separately from random coefficients.

Proxy regret is the candidate allocation's complexity range **evaluated with proxy
weights**, minus the proxy-optimal complexity range under the same objective mode.
The initial loads are unchanged. Assignment agreement is the percentage of patients
assigned to the same named clinician as the fixed reference. It depends on which
benchmark-optimal reference was selected: zero regret does not imply 100% agreement.
Changing reference seed does not change proxy regret or candidate draws.

Accuracy/QWK use each patient once in the union of the run's selected chunks. Exact
Accuracy means proxy category agreement. QWK uses positions in the full category
order, independently of complexity-weight distances. Undefined QWK is blank/NA with
an explanatory `qwk_status`. Plot-only filters do not recalculate the original table.

Figures 1/2 facet all selected chunks with a shared y scale, displaying each selection
draw. Figures 3/4 pool one draw per chunk at the fixed chunk seed. Point mode preserves
every draw; Box mode keeps model/profile groups separate and uses linear quartiles,
1.5 IQR whiskers and all observation points, including outliers/singletons. Whiskers
are neither confidence intervals nor exact min/max bounds. These charts describe
cross-chunk performance on the observed patient groups. Each figure's Starting
Parameter dropdown controls all facets; hidden models remain hidden after a profile
change. Colors depend on model identity and symbols/offsets identify profiles.

## Saved artifacts, failures and resume

The run saves `accuracy_vs_proxy.csv`, `results.csv`, `allocations.csv`,
`reference_allocations.csv`, `benchmark_optima.json`, `resolved_config.json`,
`validation_report.json`, `artifacts.json`, four `figures/*_{Point|Box}.html`,
and `index.html`. CSV records carry the benchmark ID and proxy label. Complete
solver-stage diagnostics and bounds are also retained in the immutable JSON cache.
Results include one row per requested candidate selection, with null metrics and an
explicit reason for failures. No failure is converted to zero regret. Failed runs
still produce coverage reports/figures and exit with code 2; valid runs exit 0.
All-null groups display an annotation. Missing or modified stored selection data
causes plot-only failure instead of launching any solver.

All nominal and random solves require Gurobi OPTIMAL status and an absolute objective
bound difference within `--optimality-tolerance` (default 1e-7). A time-limit incumbent
is never certified. MIP relative/absolute gaps are set to zero. The optimal-set
constraint adds **zero intended objective degradation**. Numerical feasibility,
integrality and solver optimality tolerances default to 1e-9 and can be controlled
with `--feasibility-tolerance`, `--integer-tolerance`, and
`--solver-optimality-tolerance`; verification tolerance is separately recorded.
These floating-point tolerances remain a numerical limitation. `--time-limit`
applies per solve stage, and `--threads` defaults to one.

Use the original analysis command plus `--resume` for an interrupted matching run.
The run fingerprint must match file contents, membership, profile values, candidates,
seeds, objective, tolerances and versions; candidate order and rendering mode can
change. Changed scientific inputs require a fresh output directory. Successful
components are reused, failures are retried, and corrupt cache entries are rejected.
Optional `--cache-dir PATH` shares immutable solves between fresh runs; nominal
identities depend on source content, scores, instance and solver settings; selection
identities additionally depend on seed and purpose. Reference keys never include
candidate list/order or sampling count. Evaluation is recalculated for each run's
benchmark/reference, so changing benchmark cannot reuse stale regret/agreement.
The default cache lives inside the output directory. Existing outputs are rejected
without explicit `--resume`; plot-only output must be empty or absent.

## Verification

```bash
python -m pytest patient_assignment/tests/test_proxy_analysis.py -q
```

The solver tests require a working Gurobi license and deliberately fail rather than
skip if unavailable. They compare tiny assignments and random objectives against
exhaustive enumeration, check QWK by hand, test strict inputs, caching/seed invariance,
fixed-reference evaluation, failures, sample counts, offline HTML and solver-free
regeneration. When Node is available, the suite also executes the exported JavaScript
controller against a minimal DOM/Plotly adapter to verify combined profile and model
visibility across all facets. This is not a browser screenshot/visual-layout test.
If unrelated globally installed pytest plugins fail at startup, set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the test environment.

## Stochastic and robust policy extension

The original command and deterministic CSV schema are unchanged when neither
`--uncertainty-config` nor non-deterministic `--policies` is supplied. The modular
extension uses `proxy_uncertainty.py` for strict input/grid resolution,
`proxy_policy_solver.py` for exact optimization and verification,
`proxy_policy_analysis.py` for caching/evaluation/exports, and
`proxy_policy_plots.py` for saved-data-only reports. It does not run inference.

### Synthetic pilot

Every input under `configs/proxy_uncertainty_demo/` is synthetic. From the repo root:

```bash
python run.py proxy-analysis --config configs/proxy_uncertainty_demo/inputs.yaml \
  --models Qwen MedGemma OpenDecider --benchmark proxy \
  --chunks configs/proxy_uncertainty_demo/chunks.csv \
  --profiles configs/proxy_uncertainty_demo/profiles.yaml \
  --starting-parameters balanced --chunk-ids 01 \
  --policies deterministic stochastic robust \
  --uncertainty-config configs/proxy_uncertainty_demo/pilot.yaml \
  --n-samples 3 --sample-seed 42 --chunk-seed 42 --benchmark-seed 1001 \
  --solver-seed 0 --threads 1 --output results/policy_pilot --dry-run
```

Remove `--dry-run` to solve; repeat with `--resume` after interruption. Remove the
profile/chunk filters for both chunks and all three starting profiles. To scale up,
use `configs/proxy_uncertainty.example.yaml` (20/50/100/200 scenarios, independent
training seeds 42/123/2026, Gamma fractions 0/.1/.25/.5/1, thresholds .05/.2).
Start with the small pilot: robust cut generation can require many MILP solves.
`--dry-run` validates all inputs without opening Gurobi and prints grid IDs,
feasibility and minimum solve counts. `--policy-configuration-ids ID ...` restricts
that grid for analysis or plotting. `--policies` defaults to deterministic.

```bash
python run.py proxy-plots --results results/policy_pilot \
  --output results/policy_pilot_box --show_type Box \
  --policies deterministic stochastic robust
```

Plot regeneration requires saved artifacts, pandas and Plotly, but no solver or
inference dependencies. Reports embed Plotly JavaScript for offline use. Existing
four plot filenames are retained; additional pages show policy comparisons,
scenario-count and Gamma sensitivity, cumulative solution diversity, between-training-
replicate mean regret, and holdout mean/CVaR. Controls filter model, policy,
configuration, profile, chunk, scenario count/seed, Gamma, threshold and temperature.
Failure counts remain in coverage denominators; only certified metrics are plotted.
Point/Box, `--facet-cols`, `--models`, `--chunk-ids` and
`--starting-parameters` remain available. Parameter plots are descriptive, not CIs.

### Probability inputs and configuration precedence

A model entry may specify `probability_columns`, `probability_path`,
`probability_id_column`, and `probability_sheet`. Relative probability paths are
resolved against the model registry, not the uncertainty YAML. Entry-specific
columns override `probabilities.columns`; the default fields are
`probability_easy`, `probability_moderate`, `probability_severe`, in the configured
category order. The synthetic MedGemma fixture demonstrates a separate ID join and
custom columns. These settings also work with Qwen, OpenDecider, or any model ID;
no model-specific distribution is invented. Category aliases normalize column-map
keys, while numerical severity weights remain distinct from category indices.

All included patients need finite probabilities in [0,1], unique IDs and complete
category mappings. Scalar chosen-label confidence is insufficient. Point labels
are always preserved; disagreement with probability argmax is counted (or rejected
with `point_prediction_consistency: error`). Default `bad_row: error` rejects bad
inputs before solving. No skip-policy/report-only fallback is implemented.
A sum differing from 1 by more than machine rounding is rejected by default.
Explicit `renormalize: true` permits correction only within `sum_tolerance` and
records the correction count. Source and normalized-vector hashes are saved.
No clinical accuracy, calibration, log-loss or Brier claims are made from proxy labels.

Unspecified YAML fields use the machine-readable defaults in
`proxy_uncertainty.DEFAULTS`; unknown keys and unsupported schemes fail. The resolved
configuration includes the complete grid and defaults. `gamma_counts: [0,1,2]`
selects integer patient budgets instead of `gamma_fractions`; counts above chunk
size fail. Fractions use `ceil(fraction * chunk_size)` with decimal arithmetic (avoiding binary-float rounding overshoots), and both values are exported.
`temperature_values` affects stochastic training only, via normalized
`p_k ** (1/temperature)`; 1 preserves the supplied distribution. Holdout always uses
the original distribution, making temperature sensitivities comparable.

### Mathematical definitions and certification

For feasible binary assignments, each patient is assigned exactly once and final
patient counts cannot exceed capacity. With existing workload H and counts N:

```text
L_j(x,s) = H_j + sum_i s_i x_ij
D(x,s) = max_j L_j(x,s) - min_j L_j(x,s)
K(x)   = max_j (N_j + sum_i x_ij) - min_j (N_j + sum_i x_ij)
M0: min_x D(x, categorical_point_weights)
M2: min_x sum_w D(x,s_w) / W
M3: min_x max_{s in U} D(x,s)
U = {s_i in S_i, number of i with s_i != nominal_i <= G}
S_i = {nominal_i} union {category_weight_k: p_ik >= threshold}
```

M2 uses one shared assignment across all training scenarios (nonanticipativity).
Independent categorical marginals are an explicit modeling assumption; they do not
supply patient dependence or establish calibrated clinical risk. Identical vectors
are deduplicated with multiplicities, exactly preserving empirical scenario weights.
Expected range generally differs from range at expected weights: two independent
patients each equally likely to have weight 1 or 3, assigned one per clinician,
have expected range 1, while their expected weights have range 0.

M3 is a discrete cardinality-budget uncertainty set, not a probabilistic guarantee
or the original continuous Bertsimas-Sim model. It is not restricted to nominal
optima. A master MILP begins with the nominal vector. Exact separation examines
each ordered clinician pair and selects up to G positive patient deviation gains,
including existing clinician loads. Violated adversarial vectors become new cuts.
G=0 recovers nominal loss; G=n permits the full Cartesian support. For fixed support,
optimal worst-case loss cannot decrease as G increases. Increasing thresholds shrinks
support because the nominal category is always included; proxy regret need not be
monotone in either parameter. The nominal-category inclusion is part of the set.

For `--objective lexicographic`, count range is optimized and certified first; its
optimum is imposed in every later policy solve and selection. No weighted sum is
used. Random linear tertiary objectives restrict the policy objective to its optimum
with zero intended degradation. Robust separation is repeated during tertiary
selection so newly discovered adversaries cannot invalidate certification. Exact
objectives and counts are recomputed from extracted assignments, never inferred
from free epigraph values. Numerical certification uses `--optimality-tolerance`.
Time-limited/uncertified solves produce failure rows and are excluded from optimum
comparisons. `--time-limit` applies to each Gurobi optimize call, including each master
iteration, not to the entire grid. No sampled approximation is labeled robust.

### Seed roles, benchmark and holdout evaluation

Selection coefficient streams use the existing stable instance/seed identity and
are shared across policies. Policy cache identities additionally include policy
parameters, probability hashes, and scenario content. Scenario generation hashes
ordered patient IDs, model ID, chunk ID, training seed, and a generator-version/role
namespace, never the profile or selection seed. Gurobi's algorithm seed and the fixed
benchmark-selection seed are separate. Holdout has an independent role namespace,
even when its numerical seed equals a training seed. Sample counts share prefixes
within a training replicate; these are correlated SAA sensitivity experiments.

The point-proxy benchmark optimum/reference allocation is solved once per
chunk/profile and reused for every candidate cell:

```text
proxy_regret = D(candidate_assignment, proxy_point_weights) - proxy_optimum
exact_assignment_pct = 100 * matching_patient_assignments / chunk_size
```

This is point-proxy regret, not actual clinical regret or expected/robust regret.
Lexicographic candidates and reference must share the certified count optimum.
Holdout scenarios are shared across all policies/parameters for the same model and
chunk, independent of training, using original marginals. Mean, population SD,
median, maximum, q90/q95 and empirical upper-tail CVaR are exported. CVaR integrates
the empirical quantile over [alpha,1], including fractional boundary mass. Different
models have different holdout distributions: prioritize the common point proxy
for cross-model comparisons. Scenario hashes, generator settings and seeds support
regeneration without dumping huge scenario tables.

### Results, resume and interpretation

Extended `results.csv` preserves legacy columns and adds policy identity, scenario
and support fingerprints, probability parameters, objective/certification metadata,
seed roles, exact policy/point objectives and holdout metrics. Nonapplicable parameters
are null. `allocations.csv` includes policy configuration and training seed. Common
`reference_allocations.csv` and `benchmark_optima.json` remain separate. Additional
files include `policy_grid.csv`, `policy_optima.csv`, `policy_summary.csv`,
`holdout_metrics.csv`, `probability_validation.csv`, `run_failures.csv`,
`paired_policy_differences.csv`, and `scenario_replicate_summary.csv`.

Summaries stratify by model, full policy configuration (including training seed),
chunk and profile. They retain attempted/successful/unique counts, duplicate rate,
regret and agreement moments/quantiles, and holdout summaries. Training-replicate
means are reported separately from within-cell selection-seed variation. Paired
policy differences use the same model/chunk/profile/selection seed against M0 and
are descriptive; solver samples are not independent clinical observations. Random
linear objectives are not uniform draws, and observed regret spread is not an exact
envelope over all optima. Duplicate assignments are intentionally retained.

Content-addressed cache keys include implementation/version, full solver settings,
objective/tolerance, instance, model, policy parameters, input probability source and
normalized probabilities, and generated scenarios/support. Configuration, manifests and completed cache entries
are atomically replaced; incomplete temporary files are ignored. Resume validates
the run fingerprint before rebuilding exports from only the requested cells. Changing
inputs, seeds, solver settings or implementation requires a fresh output directory;
a shared `--cache-dir` can reuse compatible completed solves. Exports are checksummed
and plotting refuses corrupt/incomplete artifacts. Source notes are never exported.

### Verification

```bash
python -m pytest patient_assignment/tests/test_proxy_uncertainty.py -q
python -m pytest patient_assignment/tests/test_proxy_analysis.py -q
python -m pytest patient_assignment/tests clinical_target_extraction/tests/test_probabilities.py -q
```

New pure tests use independent enumeration for separation, probability/grid validation,
seed roles and CVaR. Solver-backed tests compare MILPs against full tiny assignment
and uncertainty-set enumeration, check Gamma=0/monotonicity, single-scenario SAA,
lexicographic restrictions, alternative optima, shared benchmarks, complete row counts,
resume and offline regeneration. New solver tests explicitly skip without Gurobi or
an available license; the older proxy suite requires Gurobi and does not skip. Large
grids may require a full Gurobi license; synthetic tests are small. Production input
paths and license credentials are never assumed or created.
