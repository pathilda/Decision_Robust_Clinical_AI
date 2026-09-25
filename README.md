# Sequential Clinical Target Extraction

This repository processes de-identified clinical session notes in chronological order for each
client. It extracts clinical targets, maintains a stable per-client target registry, compares
later sessions with earlier sessions, validates structured output, and checkpoints results.

Normal use goes through one file: `run.py`. You do not need to manually edit YAML files or start
a separate vLLM server.

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
python -m vllm.entrypoints.openai.api_server --help
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
2. starts the Qwen vLLM server;
3. displays vLLM logs and model-loading progress;
4. waits for the API to become ready;
5. processes the first three clients with a session-level progress bar;
6. writes checkpoints and final outputs;
7. shuts down vLLM and releases the GPU.

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

The CLI loads only one model at a time. It finishes and shuts down one server before loading the
next model. Outputs are separated into `qwen`, `medgemma`, and `gpt_oss` directories.

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

### 6. Start only the model server

This is optional and is useful for manual API testing. The command remains active until
`Ctrl+C`:

```bash
python run.py \
  --pipeline serve \
  --model qwen \
  --model-root /scratch/pathilda/Models
```

Normal extraction does not require this separate step.

## Useful options

```text
--sheet Notes                  Excel sheet name
--id-column ID                 Override the client-ID column
--note-column statement2       Override the note-text column
--max-clients 5                Run only the first five clients
--client-id C001               Select one client; may be repeated
--port 8001                    Use a different local API port
--max-model-len 16384          Reduce context length if GPU memory is tight
--gpu-memory-utilization 0.90  Change vLLM's GPU-memory fraction
--startup-timeout 1800         Model-loading timeout in seconds
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
session_targets.parquet
pairwise_comparisons.parquet
```

The pipeline checkpoints after every successful session. Rerunning the same command reuses
valid checkpoints when the note history, prompts, model settings, registry, and prior structured
records have not changed.

## Architecture

```text
run.py
  └── cli.py
      ├── data.py          input validation and session ordering
      ├── settings.py      Qwen, MedGemma, and GPT-OSS presets
      ├── server.py        vLLM start, readiness, live output, and shutdown
      └── run_extraction.py
          ├── prompts.py
          ├── model_client.py
          ├── validate.py
          ├── registry.py
          └── output_store.py
```

For every client, the pipeline processes only information available through the current
session. It never includes a future note in an earlier prompt. Invalid structured output gets
one repair attempt. Context overflow is recorded instead of silently truncating the history.

## Tests

```bash
pytest -q
```

Only de-identified clinical notes may be used. `--deidentified-confirmed` is deliberately
required before inference.
