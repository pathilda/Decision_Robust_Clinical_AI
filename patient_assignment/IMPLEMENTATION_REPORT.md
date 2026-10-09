# Implementation and verification

The enhancement is implemented as the separate `patient_assignment` package. New public
commands are `python run.py chunk`, `python run.py solve`, and
`python run.py compare-assignments`; `python -m patient_assignment` and the installed
`clinical-patient-assignment` entry point provide the same interface.

## Modules

- `cohort.py`: shared ID normalization, literal discipline tags, canonical first-row source
  selection, and coverage/duplicate reporting.
- `inputs.py`: CSV/Excel loading, strict membership/severity/status validation, compatible
  note hashes, clinician inputs, and batch capacity preflight.
- `chunking.py`: seeded metadata-only chunks and manifests.
- `optimization.py`: lazy Gurobi session, two lexicographic stages, incumbent diagnostics,
  assignment verification, and precise initial/final load reporting.
- `selection.py`: joint closest-pair MIP, independent random objectives on certified optimal
  sets, stable substream/checksums, and evaluation of the actual selected AI assignment.
- `reporting.py` / `cli.py`: mode-neutral CSV/JSON, component/partial incumbents,
  complete provenance, provisional status, descriptive certified summaries, and exit codes.
- `tests/test_inputs.py`, `test_failures.py`, and `test_optimization.py`: synthetic acceptance
  tests, deterministic interrupted outcomes, and independent exhaustive tiny-instance checks.

Integration updates touch `run.py`, `pyproject.toml`, source `data.py`, both inference
CLIs/pipelines, and classification-comparison `analysis.py`/`cli.py`. Source filter metadata
is recorded in inference run configs; prompt/model settings and fingerprint checks are
preserved. Existing selected-only export behavior is tested for both stores.
README/package guides are updated. The original `Deployment_NEW.ipynb` was absent in this
checkout, so a small unexecuted notebook was added with file-license setup, an optional
synthetic smoke solve, and the full chunk/inference/solve/comparison workflow.
Existing uncommitted checkout edits were preserved.

## Actual checks

Python was not on the sandbox PATH. An installed Anaconda Python 3.11.5 was used to create
an ignored `.venv` with system site packages. Gurobi 13.0.3 and Ruff 0.16.10 were installed
only in that environment; no license was retrieved, activated, or changed.
An unrelated auto-loaded Dash pytest plugin failed to initialize outside Jupyter, so tests
were run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m pytest patient_assignment/tests -q -rs
.\.venv\Scripts\python.exe -m pytest patient_assignment/tests/test_optimization.py -q -rs
.\.venv\Scripts\python.exe -m ruff check patient_assignment run.py `
  clinical_target_extraction/src/data.py clinical_target_extraction/src/cli.py `
  opendecider_inference/cli.py model_comparison/cli.py model_comparison/analysis.py
.\.venv\Scripts\python.exe run.py chunk --help
.\.venv\Scripts\python.exe run.py solve --help
.\.venv\Scripts\python.exe -m opendecider_inference --help
.\.venv\Scripts\python.exe -m model_comparison --help
```

Final full test run: **79 passed, 13 skipped, 1 failed**. The new package contributes
50 passing tests, with 13 real Gurobi optimization tests skipped. Gurobi environment
initialization failed in both sandbox and an approved outside-sandbox attempt; no usable
license was available here. Those optimizer/matching/random-solve tests remain unverified.
The one existing-suite failure is
`opendecider_inference/tests/test_pipeline.py::test_adapter_overlay_uses_local_base_without_modifying_adapter`,
which fails on Windows `symlink_to` with WinError 1314 (missing symlink privilege).
It also failed before this enhancement's tests were added; the overlay implementation
and its test were not changed.

Ruff checks above passed. Checking the complete existing inference pipeline modules also
reports pre-existing broad-exception lint warnings, outside the new allocation implementation.
`git diff --check` passed. All new command help was checked by the pure tests without
importing GPU/solver resources. Notebook JSON/schema validation and Python AST parsing
passed for all eight code cells; all have null execution counts and empty outputs.

## Remaining verification and scope

Run the real optimization tests in the actual compute environment with a usable Gurobi
license. `/home/pathilda/gurobi.lic` does not exist in this Windows checkout; its expected
filename, validity, readability, machine compatibility, and full-license status must be
verified on the intended node. The notebook checks file existence/readability without
displaying contents and respects an existing `GRB_LICENSE_FILE`.

No real inference, new clinical-data processing, or notebook execution was performed.
Historical selected-record checkpoint provenance is not migrated or repaired; use fresh
inference output directories for changed configurations. Matching ties remain solver-selected,
random selection is not uniform sampling, and reference regret is retrospective as explained
in the package guide. No scientific tie policy or significance analysis was added.
