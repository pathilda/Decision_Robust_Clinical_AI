# Pre-treatment Client Classification

This repository classifies each client once from de-identified pre-treatment information. There
is no session sequence or rolling profile. Each Excel row represents one client; the pipeline
combines that row's service-planning and assessment text into one source-labeled note and asks a
local LLM to assign exactly one category:

- `Easy to treatment`
- `Moderate to treatment`
- `severe to treatment`

Normal use goes through `run.py`. Inference uses vLLM directly in the Python process and does not
start an HTTP server.

## Input workbook

The Excel workbook must contain these exact column names by default:

| Column | Meaning |
| --- | --- |
| `Client AlayaCare Client ID` | Unique, de-identified client ID |
| `SP text` | Service-planning text |
| `assessment text` | Assessment text |

When a client ID occurs on multiple rows, the pipeline analyzes only that client's first row and
reports the number of repeated clients and later rows it ignored. One of the two text cells may be
blank, but they cannot both be blank on the selected first row. The pipeline keeps the sources distinguishable when
it creates the one combined note:

```text
<SERVICE_PLANNING_TEXT>
...
</SERVICE_PLANNING_TEXT>

<ASSESSMENT_TEXT>
...
</ASSESSMENT_TEXT>
```

Use `--id-column`, `--sp-column`, or `--assessment-column` only when the workbook intentionally
uses different headers.

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

## Models

The built-in model presets expect these checkpoint directories below `MODEL_ROOT`:

```text
Models/
├── Qwen/
├── MedGemma/
└── GPT_OSS/
```

For example:

```bash
export MODEL_ROOT=/scratch/pathilda/Models

hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir "$MODEL_ROOT/Qwen"

hf download google/medgemma-27b-text-it \
  --local-dir "$MODEL_ROOT/MedGemma"

hf download openai/gpt-oss-120b \
  --exclude "original/*" \
  --exclude "metal/*" \
  --local-dir "$MODEL_ROOT/GPT_OSS"
```

MedGemma is gated; accept its model terms before downloading it.

## Commands

Inspect and validate a workbook without loading an LLM:

```bash
python run.py \
  --pipeline inspect \
  --input /scratch/pathilda/pre_treatment_clients.xlsx \
  --sheet Clients
```

Classify every client with Qwen:

```bash
python run.py \
  --pipeline classify \
  --model qwen \
  --model-root /scratch/pathilda/Models \
  --input /scratch/pathilda/pre_treatment_clients.xlsx \
  --sheet Clients \
  --output /scratch/pathilda/client_classification_results \
  --deidentified-confirmed
```

For a small validation run, add `--max-clients 3`. To select particular clients, repeat
`--client-id`:

```bash
python run.py \
  --pipeline classify \
  --model qwen \
  --input /scratch/pathilda/pre_treatment_clients.xlsx \
  --output /scratch/pathilda/client_classification_results \
  --client-id C001 \
  --client-id C014 \
  --deidentified-confirmed
```

Use `--model medgemma`, `--model gpt_oss`, or `--model all`. With `all`, models run sequentially
and their results are kept in separate directories. Independent clients are submitted in direct
batches; adjust `--batch-size` if needed.

### Optional brief-reasoning prompt

The original prompt remains the default and is unchanged. To ask Qwen or MedGemma for a concise,
evidence-based reasoning summary before the category, add:

```text
--prompt-style brief-reasoning
```

For example, in Jupyter:

```python
!python run.py \
  --pipeline classify \
  --model medgemma \
  --input "$EXCEL" \
  --model-root "$MODEL_ROOT" \
  --output "$OUTPUT" \
  --prompt-style brief-reasoning \
  --batch-size 1 \
  --deidentified-confirmed
```

This mode returns `brief_reasoning` followed by `treatment_category` in structured JSON. The
reasoning is limited to a short summary of documented evidence; it is not a request for hidden or
detailed chain-of-thought. Its results are stored separately under
`<output>/<model>_brief_reasoning/`. Omit the option, or use `--prompt-style standard`, to run the
unchanged original prompt under `<output>/<model>/`.

## Output

Each model writes to `<output>/<model>/`:

```text
run_config.json
raw_responses.jsonl
validated_classifications.jsonl
failures.jsonl
client_classifications.csv
client_classifications.xlsx
client_classifications.parquet  # when a Parquet engine is installed
```

The flat CSV and Excel result files contain one row per successfully classified selected client,
including `Client AlayaCare Client ID` and `treatment_category`. JSONL checkpoints retain validation and reproducibility
metadata. Re-running an unchanged client/model/prompt combination reuses its validated result;
changed input is classified again.

Brief-reasoning runs also include `prompt_style` and `brief_reasoning` in the flat result files.

The category is an LLM-generated operational classification, not a diagnosis, causal conclusion,
or substitute for clinical judgment. Inputs must be de-identified, which the classification
command enforces through `--deidentified-confirmed`.

## Local OpenDecider backend

A separate local-only OpenDecider framework is available in
[`opendecider_inference/`](opendecider_inference/README.md). It uses an independently downloaded
Qwen3-30B-A3B base and OpenDecider adapter, accepts the same workbook and category rubric, and
writes a comparable classification table plus confidence and all three category probabilities.
OpenDecider does not have a brief-reasoning option: it is a typed decision model that returns a
category distribution rather than generated explanatory text.

## Compare model configurations

The reusable comparison command accepts any number of named CSV, Excel, or Parquet result files.
It creates the category distribution plus exact-agreement and quadratic-weighted Cohen's kappa
matrices as CSV tables and publication-ready PNG figures.

```bash
python -m model_comparison \
  --input "QS=/path/to/results/qwen/client_classifications.xlsx" \
  --input "QR=/path/to/results/qwen_brief_reasoning/client_classifications.xlsx" \
  --input "MS=/path/to/results/medgemma/client_classifications.xlsx" \
  --input "MR=/path/to/results/medgemma_brief_reasoning/client_classifications.xlsx" \
  --input "OD=/path/to/results/opendecider/client_classifications.xlsx" \
  --output /path/to/results/model_comparison \
  --title "Treatment-difficulty model agreement"
```

By default, all tables use the intersection of client IDs present in every input so every system
is evaluated on the same cohort. See [`model_comparison/`](model_comparison/README.md) for custom
column names, ordinal categories, and pairwise-cohort analysis.

## Useful options

```text
--sheet Clients                 Excel sheet name
--id-column "Client AlayaCare Client ID"
--sp-column "SP text"          Service-planning column
--assessment-column "assessment text"
--prompt-style standard         Original unchanged prompt
--prompt-style brief-reasoning  Concise evidence summary plus category
--max-clients 5                Process only the first five clients
--client-id C001               Select one client; may be repeated
--max-model-len 131072         Total input-plus-output context limit
--max-tokens 256               Maximum generated tokens per classification
--batch-size 8                 Clients per direct inference batch
--gpu-memory-utilization 0.92  vLLM GPU-memory fraction
```
