# Patient assignment

For selectable objectives, operational profiles, repeated optimal selections, regret envelopes,
and robust tie-breaking, see [EXPERIMENTS.md](EXPERIMENTS.md). These are opt-in commands;
the legacy interface documented below is preserved.

This package extends the existing classification framework with four commands. The
classification labels, prompts, model settings, and legacy entry points remain available.
New code is separated into `cohort.py`, `inputs.py`, `chunking.py`, `optimization.py`,
`selection.py`, `reporting.py`, and `cli.py`. Nothing invokes inference to allocate patients.

## Installation and command entry points

```bash
python -m pip install -e '.[optimization,dev]'
# Or add only the Gurobi API to an existing environment:
python -m pip install gurobipy
python run.py chunk --help
python run.py solve --help
python run.py compare-assignments --help
python run.py compare-all --help
```

`python -m patient_assignment` and installed `clinical-patient-assignment` accept the same
subcommands. Legacy `python run.py --pipeline inspect|classify`, installed classification
entry points, `python -m opendecider_inference`, and `python -m model_comparison` are preserved.
Chunking and all command help work without Gurobi, a license, LLM weights, or CUDA.

## Source cohort and ID contract

Source inputs support Excel (CSV is also accepted for chunking). Default columns are
`Client AlayaCare Client ID` and `Employee Submission Full Name`. Override them with
`--id-column` and `--employee-column`; `--sheet` accepts a name or zero-based index.
Repeat `--discipline PT`, `--discipline OT`, or `--discipline SLP` to select their union.
Without a filter, the employee column is optional and untagged patients remain eligible.

Only the configured employee cell is inspected. Null has no tags. For other cells, convert
to text, pad each end with one ASCII space, then search for literal ` PT `, ` OT `, and
` SLP `. Matching is case-sensitive: `Jane SLP` matches; `Jane slp`, `Jane SLP,`,
`Jane (PT)`, `ADOPT`, and tab/newline-separated tokens do not. Multiple matching tags do
not duplicate a patient. Source/selected/excluded counts, missing employee counts, and
overlapping recognized-tag counts are reported; employee names are never exported here.

IDs are stripped at their ends. Textual leading zeros, case, and A/B/C suffixes are
preserved. Integral numeric cells normalize to integer strings (`123.0` numeric becomes
`123`); textual `123.0` remains textual. Missing IDs fail. No absent zeros are inferred.
Text such as `NA` is preserved rather than interpreted as an Excel/CSV missing-value code.
Normalization collisions are duplicate normalized IDs: source keeps the first row and
reports discarded rows, while membership and prediction/reference files reject every
duplicate, even identical duplicate values or duplicates outside the membership.

Source canonicalization happens **before** discipline filtering. Notes and tags use the
same retained first row; notes and tags from later duplicates are never merged. This is
a historical compatibility policy, not a clinical endorsement of duplicate records.
In both inference backends, filtering happens before `--client-id` and `--max-clients`.
Explicit excluded IDs error. Employee names are never appended to model prompts.

Discipline filtering selects patients; it does not constrain clinician eligibility.
Run separate PT/OT/SLP manifests when separate professional pools are intended.

## End-to-end workflow

Commands below use Bash continuation syntax. In PowerShell, use one line or replace
trailing backslashes with PowerShell backticks. Each new output directory must be empty
or absent. Use fresh inference output directories when changing input/configuration.

Create authoritative membership without requiring clinical notes or de-identification
acknowledgement:

```bash
python run.py chunk --input patients.xlsx --discipline SLP \
  --chunk-size 50 --seed 42 --output scenarios/slp_50_seed42
```

IDs are sorted lexicographically, shuffled once by local Python `random.Random(seed)`
(MT19937), then divided into consecutive groups; the final smaller group is retained.
No global RNG is mutated. Seed default is 42; schema, Python version, source hash,
normalization/filter/duplicate policy, and membership hash are recorded. Every selected
patient appears exactly once. Files are `chunks.csv` (`patient_id`, `chunk_id`,
`within_chunk_order`), `chunk_summary.csv`, and `chunk_config.json`.
Repeating chunk seeds on the same patients is sensitivity analysis, not independent
clinical sampling. Saved membership is reused across all AI/reference runs.

Reuse existing predictions, or explicitly run inference in your compute environment:

```bash
python run.py --pipeline classify --input patients.xlsx --discipline SLP \
  --model qwen --deidentified-confirmed --output results/qwen_run1
python run.py --pipeline classify --input patients.xlsx --discipline SLP \
  --model medgemma --deidentified-confirmed --output results/medgemma_run1
# OpenDecider supports the same --discipline/--employee-column options:
python -m opendecider_inference --pipeline classify --input patients.xlsx \
  --discipline SLP --adapter-path /models/adapter --base-path /models/base \
  --deidentified-confirmed --output results/od_run1
```

No inference was launched to implement or test this package. Do not use `--max-clients`
on an inference run intended to cover the entire membership unless it still covers
every member. Existing inference export only includes selected IDs, but a historical
validated record for a selected patient can still appear after a changed request fails.
Request-fingerprint reuse checks remain in place. This enhancement does **not** migrate
or establish provenance of old checkpoints; use fresh directories for changed runs.

Solve independently for each chunk:

```bash
python run.py solve --chunks scenarios/slp_50_seed42/chunks.csv \
  --predictions results/qwen_run1/qwen/client_classifications.xlsx --label QS \
  --clinician A 12 24 --clinician B 10 27 --clinician C 14 22 \
  --clinician-capacity 35 --output assignments/QS_run1
```

Compare two AI systems:

```bash
python run.py compare-assignments --chunks scenarios/slp_50_seed42/chunks.csv \
  --predictions-a results/qwen_run1/qwen/client_classifications.xlsx --label-a QS \
  --predictions-b results/medgemma_run1/medgemma/client_classifications.xlsx --label-b MS \
  --clinician A 12 24 --clinician B 10 27 --clinician C 14 22 \
  --clinician-capacity 35 --selection-mode matched --output comparisons/QS_vs_MS_run1
```

For AI versus a reference, replace B's file/label and explicitly declare it:

```bash
python run.py compare-assignments --chunks scenarios/slp_50_seed42/chunks.csv \
  --predictions-a results/qwen_run1/qwen/client_classifications.xlsx --label-a QS \
  --predictions-b reference.xlsx --label-b Reference --reference-side B \
  --clinician A 12 24 --clinician B 10 27 --clinician C 14 22 \
  --clinician-capacity 35 --output comparisons/QS_vs_reference_matched_run1
```

For randomized optimal selection, append `--selection-mode random --solver-seed 0
--selection-repeats 20` and use a new output directory. Reference declaration is optional
in either mode. `--reference-side A` is supported. Never infer reference status from a label.

## Severity, clinicians, and feasibility

### Combined extraction and assignment comparison

`compare-all` compares the extracted treatment categories and runs both matched and
random optimal assignment comparisons in one command:

```bash
python run.py compare-all --chunks scenarios/slp_50_seed42/chunks.csv \
  --predictions-a results/qwen_run1/qwen/client_classifications.xlsx --label-a QS \
  --predictions-b results/medgemma_run1/medgemma/client_classifications.xlsx --label-b MS \
  --clinician A 12 24 --clinician B 10 27 --clinician C 14 22 \
  --clinician-capacity 35 --threads 20 --time-limit 300 \
  --solver-seed 0 --selection-repeats 20 --output comparisons/slp_all_run1
```

All three comparisons use exactly the union of the saved chunk membership. Missing
labels fail before report generation; predictions outside membership are excluded.
Extraction here means treatment-category agreement (distributions, exact agreement,
quadratic weighted kappa), not agreement of free-text evidence spans. Separate
`--id-column-a/b`, `--category-column-a/b`, and `--sheet-a/b` options are supported.
Outputs are grouped into `extraction/`, `matched/`, and `random/`. Both assignment
modes print and save pooled patient agreement and clinician tables. Random repeats
default to 20; matched always uses one selection. `--selection-mode` is unnecessary.
`--time-limit` remains per solver stage. A time-limited matched comparison does not
prevent the random comparison from running. `comparison_bundle_status.json` records
each stage's exit code and completion; overall exit codes are 0 complete, 1
uncertified/incomplete optimization, and 2 input/dependency/configuration failure.
Use a fresh output directory. The component optimization is rerun for each mode.

Prediction/reference tables support `.csv`, `.xlsx`, and `.xls`. Defaults are
`Client AlayaCare Client ID` and `treatment_category`. Solve accepts `--id-column`,
`--category-column`, `--sheet`; comparison accepts separate `--id-column-a/b`,
`--category-column-a/b`, and `--sheet-a/b` (each suffix is a separate flag).
Membership always uses `patient_id`/`chunk_id`, not prediction-column defaults.

| Exact stored category | Weight |
| --- | ---: |
| `Easy to treatment` | 1 |
| `Moderate to treatment` | 2 |
| `severe to treatment` | 3 |

Category whitespace is trimmed; capitalization is not repaired. Every member requires
one valid prediction on every side. Missing/blank/unknown selected categories and explicit
failed records error before optimization. If present, `success`/`successful` must be
true or 1; `status`/`classification_status` must be success, successful, completed,
validated, or ok (status spelling is case-insensitive). Unselected rows are counted and
ignored for category/status validation; all IDs are validated for uniqueness.
Comparison never intersects/shrinks chunks. For incomplete reference labels, explicitly
build a restricted, completely labeled manifest **before** comparison.
When compatible `input_sha256`, `sp_text_sha256`, or `assessment_text_sha256` columns
exist on both sides, selected note hashes are checked. Metadata states how many patients
have fully verifiable note identity; file hashes alone cannot establish note consistency.

`--clinician NAME INITIAL_COUNT INITIAL_SEVERITY` is repeatable and required. Names must
be unique/nonblank, counts nonnegative integers, initial severity finite/nonnegative.
Decimal severity is retained and its original CLI representation is saved. Severity units
must match the fixed patient weights; these are experimental treatment-difficulty weights,
not validated hours of work. A consistency warning is recorded when initial severity falls
outside count-to-three-times-count; the supplied input is retained without imputation.

Capacity is a shared positive integer limiting **final** count. Initial count must not
exceed capacity, and every chunk N must fit `sum(capacity - initial_count)`; all chunks
are preflighted together without truncation. In the examples, the three clinicians have
69 free slots, sufficient for 50 new patients. Each chunk restarts from the same initial
state independently. Initial patients are never reassigned or discharged; their aggregate
state is assumed to describe patients outside the new membership (overlap is unverifiable).
Solve/compare need neither notes nor a source workbook; discipline is already in membership.

## Optimization and selection interpretation

Stage 1 minimizes range of final clinician counts, including initial counts. Stage 2
minimizes final severity range, including initial severity, subject to the certified
optimal count range. These are two separate MIPs, not a weighted objective. New patient
counts need not be equal. Each patient gets exactly one clinician, with all-to-all eligibility.

Both stages require Gurobi's OPTIMAL status for exact selection. Requested relative/absolute
MIP gaps are zero. `FeasibilityTol`, `IntFeasTol`, and `OptimalityTol` are 1e-9; exported
assignment verification tolerance is 1e-6. Decimal inputs and reports retain their precision;
Gurobi optimization itself uses floating-point arithmetic. No epsilon-optimal scientific
set is introduced. `--time-limit` is optional positive seconds **per optimization stage**;
default is unlimited. `--threads` defaults to 1; `--solver-seed` defaults to 0
(nonnegative, at most 2000000000). Effective parameters and versions are saved.

Matched mode (default) solves one joint MIP with both allocations free inside their own
certified count/severity optimal sets, minimizing number of patients with different
clinicians. Clinician identities never permute. Identical severity vectors give minimum
disagreement zero; zero for different vectors means optimal sets intersect, not that they
are identical. A model can have different matched allocations against different partners.
Ties use the solver-selected closest pair: no regret objective, ID tie-break, or uniqueness
promise is added. Seeds/settings/order and exact assignments provide traceability.

Random mode independently reruns the original two-stage lexicographic optimization
for each model and each repeat, using Gurobi `Seed = solver_seed + repeat_id - 1`.
There are no random costs or extra selection objectives. Both models use the same
seed in each repeat, and the schedule is reused across chunks. With `--solver-seed 0`
and `--selection-repeats 20`, seeds are 0 through 19. Changing model labels, partner,
reference designation, or A/B position does not change the seed schedule.
The effective seed and stage diagnostics are saved in `selection_streams.json`;
selected assignments, incumbents, and comparison results also record `solver_seed`.
Seeds change the solver search, not the workload objectives. Repeats may return the
same assignment and are all retained. This is solver-seed sensitivity, not uniform
sampling over optimal allocations. Initial component solves certify the workload
optima; each seeded solve is independently certified and checked against those optima.

`--selection-seed` is retained only for command compatibility and is inactive.
Repeats default to 1 in `compare-assignments` and 20 in `compare-all`.
Matched mode permits only one repeat and continues to minimize disagreement jointly.
The last repeat seed must not exceed 2000000000. Repeats measure conditional solver
sensitivity, not new independent patients/chunks. Identical vectors can have nonzero
random disagreement when solver-selected tied optima differ, while reference regret
remains zero. Random results are observed disagreement, never D_min.

With an explicit reference, evaluate the **saved selected AI assignment** under reference
weights and the same initial loads. Regret is its reference severity range minus the
reference's certified minimum severity range **subject to the optimal final-count range**.
Large negative regret fails validation; tiny negatives within 1e-6 are retained raw and
reported as zero. Without a reference, regret is null. Differences in AI-specific objectives
are not called true regret. Matching with reference is retrospective, matching-conditioned
regret using reference information; it is not deployment performance without reference
labels. Minimum disagreement does not imply minimum regret. Random selection stays
independent of reference and evaluates the actual saved AI allocation.

## Outputs and failure handling

CSV uses UTF-8, lowercase true/false, empty missing fields, unrounded numerical values;
JSON uses explicit null and preserves decimal values as strings. No notes or employee
names appear in allocation outputs. Reports recompute membership, capacities, counts,
loads, ranges, disagreement, and regret directly from assignments before exporting.

Solve writes `assignments.csv`, `clinician_loads.csv`, `chunk_results.csv`,
`run_config.json`, and `run_status.json`. Comparison writes canonical mode-neutral
`comparison_assignments.csv` (selected clinician A/B), `comparison_clinician_loads.csv`,
`comparison_results.csv`, `comparison_config.json`, `comparison_summary.json`, and
`selection_streams.json`. Separately stored `component_assignments.csv`,
`component_clinician_loads.csv`, and `component_results.csv` preserve individual component
incumbents/optima; they are never substituted for the selected comparison pair.
`selection_incumbents.csv` preserves available random tie-selection incumbents per side,
including a side whose partner has no incumbent; only complete pairs get comparison metrics.
Every comparison/selected-allocation row includes mode and repeat ID.

At completion, comparison also prints patient agreement across the union of chunks:
evaluated/total patients, same/different clinician counts, agreement percentage, and
a clinician cross-tabulation (rows = model A, columns = model B). Only certified
assignments contribute; incomplete coverage is explicitly reported. Random repeats
are reported separately, without counting repeated patients as new patients.
These results are saved in `patient_agreement_summary.csv` and
`clinician_agreement_matrix.csv`, and the summary is included under
`patient_agreement_by_repeat` in `comparison_summary.json`. Chunks still optimize
independently; these reports pool their saved assignments.

Statuses, incumbent availability, bounds, gaps, runtimes, and certification flags distinguish
failures from zero disagreement/regret. Uncertified stage 1 never advances to stage 2;
uncertified components never enter optimal-set selection. Interrupted matching candidates
are provisional, and their disagreement is an upper bound on the minimum; their bound is
also saved. Interrupted seeded solves may have nonoptimal workload incumbents. Only incumbents
verified to attain the certified component optima are exported as selected assignments;
these remain provisional until both original stages are certified. Stage diagnostics
are retained even when no eligible selected assignment exists. Missing incumbents yield
no selected pair/regret. Provisional metrics are excluded from completed aggregates.

Summary reports successful/failed/provisional chunks, coverage, within-chunk mean/min/max
and sample standard deviation (null for one repeat), equally weighted complete-chunk
means, and pooled `sum(changed)/sum(N)` over certified chunk/repeat rows. Repeated patients
are not additional clinical samples; modes are never pooled. No p-values/significance
claims are added. Global malformed inputs stop before solving. Per-chunk optimization
failures produce diagnostic rows and other chunks continue. Exit codes: 0 fully certified;
1 incomplete/uncertified optimization; 2 input/config/dependency errors. License/dependency
initialization errors write `failure_status.json`, never a success summary. Interrupted
batches and existing nonempty output directories are not resumable; use a fresh directory.

## Gurobi deployment

The reusable solver uses standard Gurobi license discovery and supports institutional
configurations. It never retrieves/activates licenses, changes license files, requests
credentials as arguments, or stores license secrets. For your reported Linux file license:

```bash
export GRB_LICENSE_FILE=/home/pathilda/gurobi.lic
python -m pip install gurobipy
```

That expected filename, readability, validity, and compute-node compatibility must be
checked in the actual deployment environment. It is absent in this Windows checkout.
`Deployment_NEW.ipynb` supplies an environment-aware file check before Gurobi initialization,
and an optional tiny smoke solve that is never run on import. An already set
`GRB_LICENSE_FILE` is respected. Other users need not use this file-based notebook example.

Installing `gurobipy` installs the Python API, not `grbgetkey`. Named-User activation uses
official license tools/full installation and your exact portal-provided `grbgetkey` command
on the intended machine; Academic Named-User retrieval requires the academic network.
WLS uses its downloaded/generated license file rather than `grbgetkey`, and needs network
access for license tokens. Machine-bound laptop/login-node licenses need not work on
changing HPC compute nodes; confirm the license type/institutional setup locally.
A successful small smoke solve alone does not prove a full license: the Python package
can use a size-limited license. Inspect startup licensing locally and confirm coverage for
the actual allocation model size; application reports deliberately omit license contents.

Official references: [Python installation](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python),
[Academic Named-User](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license),
[WLS](https://support.gurobi.com/hc/en-us/articles/13232844297489-How-do-I-set-up-a-Web-License-Service-WLS-license),
[`grbgetkey` tools](https://support.gurobi.com/hc/en-us/articles/360040113232-How-do-I-resolve-the-error-grbgetkey-command-not-found-or-grbgetkey-is-not-recognized),
and [Python model API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html).

## Tests

```bash
python -m pytest -q
python -m pytest patient_assignment/tests -q -rs
```

All records are synthetic. Pure tests need neither GPU nor Gurobi. Real optimization tests
skip with a clear reason without a usable Gurobi installation/license; skipped tests are
unverified, never reported as passes. Tests independently enumerate tiny lexicographic
optimal sets and their minimum Hamming disagreement, check saved allocations/regret,
random substream independence, filtering, and deterministic interrupted solver outcomes.
