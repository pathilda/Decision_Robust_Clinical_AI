# Sequential Clinical Target Extraction

This repository processes de-identified clinical session notes in chronological order for each
client. Session 1 creates an initial target profile. Every later session receives only the
immediately previous JSON profile plus the current note and emits the next complete profile.
It records verbatim evidence, treatment and observation flags, adjacent-session change, stable
target IDs, carry-forward state, and resumable checkpoints.

Normal use goes through one file: `run.py`. Extraction loads vLLM directly in the Python process;
it does not start an HTTP/API server.

## Input

The Excel workbook must contain these columns:

| Column | Meaning |
| --- | --- |
| `ID` | De-identified client ID |
| `statement2` | Session note |

Rows must already be ordered from session 1 to session n for each client. After the last session
of one client, the next row starts session 1 of the next client. A client ID may not reappear in a
later block.

The pipeline derives session numbers from row order and creates note IDs such as
`C001::S0001`.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python --version
nvidia-smi
python -c "from vllm import LLM; print('vLLM offline API is available')"
python run.py --help
```

The project is intended to load one model at a time on one H100 GPU.

## Download the models

Authenticate without putting a Hugging Face token in this repository:

```bash
hf auth login
hf auth whoami
```

MedGemma is gated. Accept its terms on the
[MedGemma model page](https://huggingface.co/google/medgemma-27b-text-it) before downloading.

Create the model directory used by the CLI:

```bash
export MODEL_ROOT=/scratch/pathilda/Models
mkdir -p "$MODEL_ROOT"
```

Download each checkpoint into the expected directory:

```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir "$MODEL_ROOT/Qwen"

hf download google/medgemma-27b-text-it \
  --local-dir "$MODEL_ROOT/MedGemma"

hf download openai/gpt-oss-120b \
  --exclude "original/*" \
  --exclude "metal/*" \
  --local-dir "$MODEL_ROOT/GPT_OSS"
```

The directory should look like this:

```text
/scratch/pathilda/Models/
├── Qwen/
├── MedGemma/
└── GPT_OSS/
```

## Commands

### 1. Inspect and validate the workbook

This command does not load a model:

```bash
python run.py \
  --pipeline inspect \
  --input /scratch/pathilda/deidentified_notes.xlsx
```

For a named Excel sheet:

```bash
python run.py \
  --pipeline inspect \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --sheet Notes
```

### 2. Run a small Qwen test

```bash
python run.py \
  --pipeline extract \
  --model qwen \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --max-clients 3 \
  --deidentified-confirmed
```

This single command:

1. validates the input;
2. loads Qwen directly with the offline `vllm.LLM` API;
3. displays vLLM logs and model-loading progress;
4. initializes the offline inference engine;
5. submits direct prompt batches with vLLM's native generation progress bar;
6. writes checkpoints and final outputs;
7. releases the offline vLLM engine and GPU.

### 3. Run the complete workbook with one model

Omit `--max-clients` to process every client:

```bash
python run.py \
  --pipeline extract \
  --model qwen \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --deidentified-confirmed
```

Use `--model medgemma` or `--model gpt_oss` for the other checkpoints.

### 4. Run all three models sequentially

```bash
python run.py \
  --pipeline extract \
  --model all \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --deidentified-confirmed
```

The CLI loads only one model at a time. It releases one offline engine before loading the next
model. Outputs are separated into `qwen`, `medgemma`, and `gpt_oss` directories.

### 5. Run selected clients

Repeat `--client-id` for each desired client:

```bash
python run.py \
  --pipeline extract \
  --model qwen \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --client-id C001 \
  --client-id C014 \
  --deidentified-confirmed
```

## Useful options

```text
--sheet Notes                  Excel sheet name
--id-column ID                 Override the client-ID column
--note-column statement2       Override the note-text column
--max-clients 5                Run only the first five clients
--client-id C001               Select one client; may be repeated
--max-model-len 131072         Total input-plus-output context limit
--max-tokens 32768             Maximum generated tokens per model response
--batch-size 8                 Maximum active sequences in direct vLLM generation
--gpu-memory-utilization 0.90  Change vLLM's GPU-memory fraction
```

Run `python run.py --help` for the complete list.

### Running from Jupyter

Use the same CLI through a bash cell; nothing else changes:

```bash
%%bash
python run.py \
  --pipeline extract \
  --model qwen \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --max-clients 3 \
  --deidentified-confirmed
```

The cell stays active while extraction runs and shows the native vLLM loading logs plus the
session progress bar.

### Fast batched execution

The extraction path now uses the same inference structure as the earlier repository: it creates
an in-process `vllm.LLM` object and passes prompt lists directly to
`LLM.generate(..., use_tqdm=True)`. There is no HTTP layer. Each generation wave contains the
next eligible session from every active client. vLLM batches that list on the H100 while the
pipeline preserves chronological dependencies inside each client.

```bash
python run.py \
  --pipeline extract \
  --model qwen \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/deidentified_notes.xlsx \
  --output /scratch/pathilda/clinical_target_results \
  --batch-size 8 \
  --deidentified-confirmed
```

Sessions belonging to one client remain strictly sequential because session n depends on the
JSON profile from session n-1. Raw notes and model responses from older sessions are not repeated
in the prompt. With multiple clients, each wave is generated as one direct batch. A run containing
only one client cannot batch dependent sessions.

There is one normal model call per session. A later-session call handles existing targets,
carry-forward, change classification, and genuinely new targets together. Python assigns new
`T###` identifiers, so there is no separate canonicalization call. Only structurally invalid JSON
can trigger one repair call.
If a model or GPU allocation runs out of memory, retry with `--batch-size 4`, then `2`, then `1`.
The old `--concurrency` spelling remains accepted as an alias for `--batch-size`.

### Alliance clusters without `nvcc`

Before importing vLLM, the pipeline sets `VLLM_USE_FLASHINFER_SAMPLER=0`. This prevents
FlashInfer's sampling warm-up from trying to JIT-compile a CUDA kernel when `nvcc` or
`/usr/local/cuda` is unavailable. vLLM uses its native sampler instead; the model's
FlashAttention attention backend remains available.

## Output

Each model writes into its own directory:

```text
/scratch/pathilda/clinical_target_results/
├── qwen/
├── medgemma/
└── gpt_oss/
```

Each directory contains:

```text
run_config.json
raw_responses.jsonl
validated_sessions.jsonl
failures.jsonl
registries/
session_targets.csv
target_changes.csv
pairwise_comparisons.csv
```

CSV analysis tables are always written. If `pyarrow` or `fastparquet` is available, equivalent
`.parquet` files are written as an additional convenience; Parquet support is not required.

`validated_sessions.jsonl` stores both the raw schema-conforming model output and the complete
`session_profile` passed to the next session. `session_targets.csv` has one row per
client-session-target, including:

```text
verbatim_evidence
evidence_source_session
substantively_treated
performance_observed
change_from_previous
carried_forward
newly_added
```

`target_changes.csv` contains only adjacent transitions: `improved`, `stable`, `worsened`,
`not_assessed`, or `new`. For compatibility, `pairwise_comparisons.csv` contains the same
adjacent-only rows and identifies its scope as `immediately_previous_profile`; it no longer
contains every historical session pair. It also provides derived binary `comparable`, `better`,
and `worse` columns alongside the one-hot change columns.

Evidence mismatches and inconsistent carry-forward combinations are preserved under
`quality_warnings` instead of causing a repair. Hard validation is limited to schema-conforming
JSON, valid identifiers, and exactly one update for every target in the previous profile.

The pipeline checkpoints JSONL, the complete session profile, and the client registry after every
successful session. Rerunning the same command reuses valid checkpoints when the current note,
previous profile, prompts, model settings, and registry have not changed. The derived CSV/Parquet
tables are materialized once at the end of the run.

The rolling-profile contract is versioned as `rolling_snapshot_v1`. Checkpoints from the former
all-history architecture do not match its fingerprints and are recomputed automatically. Use a
fresh `--output` directory when you want the cleanest before/after speed comparison.

## Architecture

```text
run.py
  └── cli.py
      ├── data.py          input validation and session ordering
      ├── settings.py      Qwen, MedGemma, and GPT-OSS presets
      └── run_extraction.py
          ├── prompts.py
          ├── model_client.py  direct vllm.LLM.generate batching
          ├── validate.py
          ├── registry.py
          └── output_store.py
```

For every client, session 1 receives its note and creates the initial profile. Session n receives
only profile n-1 and note n. When a prior target is absent from note n, the model copies its prior
verbatim evidence and source-session number, sets `carried_forward=1`, and records
`change_from_previous="not_assessed"`. When a genuinely new target appears, the same response
describes it and Python assigns the next stable ID.

This avoids repeating all prior notes and all prior structured outputs, and removes the former
all-pairs comparison and canonicalization-call loops. Invalid structured output still gets one
repair attempt. Each batch contains at most one session per client, and vLLM keeps up to
`--batch-size` sequences active while displaying its native TQDM progress.

## Tests

```bash
pytest -q
```

Only de-identified clinical notes may be used. `--deidentified-confirmed` is deliberately
required before inference.
