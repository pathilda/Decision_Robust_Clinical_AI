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
