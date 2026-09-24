# Sequential clinical-target extraction

This repository implements a local research pipeline that reads de-identified clinical notes
one client and one session at a time. It extracts treatment targets and longitudinal evidence;
it does not diagnose, recommend treatment, or make downstream operational decisions.

The intended environment is one interactive Open OnDemand session with one H100 GPU and one
model loaded at a time. The first dry run should use Qwen and only 3–5 deliberately selected
clients.

## Input workbook

The Excel workbook has exactly two required source columns:

| Column | Meaning |
| --- | --- |
| `ID` | De-identified client identifier |
| `statement2` | Clinical note for one session |

Workbook row order is the session order. All notes for a client must form one contiguous block:

| Excel row order | ID | statement2 |
| ---: | --- | --- |
| 1 | C001 | Client C001 session 1 note |
| 2 | C001 | Client C001 session 2 note |
| 3 | C001 | Client C001 session 3 note |
| 4 | C002 | Client C002 session 1 note |
| 5 | C002 | Client C002 session 2 note |

The loader does not sort the workbook. It derives `session_index=1,2,...,n` from row order inside
each client block and generates stable-in-file note IDs such as `C001::S0001`. It rejects blank
IDs, blank notes, missing columns, and any client ID that reappears after the next client block
has started. Because session order comes only from row position, inserting, deleting, or moving
a row changes the derived session and note IDs for affected rows.

Only de-identified notes may be used. The pipeline requires an explicit acknowledgement in the
data configuration before inference.

## End-to-end implementation steps

### 1. Start an OnDemand GPU session

Request one H100 GPU, open a terminal on the allocated compute node, and go to this repository.
No SLURM script, multi-GPU setup, database, or web service is required.

### 2. Create the Python environment

Use the Python and CUDA modules appropriate for the cluster, then create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install --upgrade huggingface_hub
python -m pip install vllm
```

The Hugging Face `hf` command is installed by `huggingface_hub`. Verify the main commands:

```bash
python --version
nvidia-smi
hf --help
vllm --help
```

If the cluster provides a supported vLLM module or container, use that instead of installing a
second copy. Keep the vLLM version fixed across model comparisons and record it with:

```bash
python -m pip freeze > environment-freeze.txt
```

### 3. Authenticate with Hugging Face

Log in without placing a token in this repository:

```bash
hf auth login
hf auth whoami
```

MedGemma is gated. While logged into the same Hugging Face account, open the
[MedGemma model page](https://huggingface.co/google/medgemma-27b-text-it) and accept the Health
AI Developer Foundations terms before downloading it.

### 4. Download the three model checkpoints

Choose a persistent, high-capacity project or scratch directory. Do not use a temporary job
directory if downloaded weights must survive the OnDemand session:

```bash
export MODEL_ROOT=/absolute/persistent/path/to/clinical-target-models
mkdir -p "$MODEL_ROOT"
```

The repositories are large. Use `--dry-run` first to check the required download size and your
storage quota:

```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --dry-run
hf download google/medgemma-27b-text-it --dry-run
hf download openai/gpt-oss-120b --exclude "original/*" --exclude "metal/*" --dry-run
```

Download Qwen:

```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir "$MODEL_ROOT/qwen3-30b-a3b-instruct-2507"
```

Download MedGemma after accepting its terms:

```bash
hf download google/medgemma-27b-text-it \
  --local-dir "$MODEL_ROOT/medgemma-27b-text-it"
```

Download the native MXFP4 GPT-OSS checkpoint used by vLLM. Excluding `original/` and `metal/`
avoids downloading alternate weight formats that this pipeline does not use:

```bash
hf download openai/gpt-oss-120b \
  --exclude "original/*" \
  --exclude "metal/*" \
  --local-dir "$MODEL_ROOT/gpt-oss-120b"
```

The official model pages are:

- [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [MedGemma 27B text](https://huggingface.co/google/medgemma-27b-text-it)
- [GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b)

For a frozen experiment, add `--revision COMMIT_HASH` to each download and record those hashes
in the research log. See the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
for authentication, revisions, local directories, cache placement, and download filtering.

### 5. Configure and inspect the Excel input

Copy the example configuration:

```bash
cp clinical_target_extraction/configs/data.example.yaml \
  clinical_target_extraction/configs/data.yaml
```

Edit `data.yaml`:

```yaml
input_path: /absolute/path/to/deidentified_notes.xlsx
sheet_name: Notes
deidentified_confirmed: true
columns:
  client_id: ID
  note_text: statement2
```

Set `deidentified_confirmed: true` only after verifying the workbook contains no protected or
directly identifying information. Then run the read-only preflight:

```bash
clinical-target-extraction inspect-data \
  --data-config clinical_target_extraction/configs/data.yaml
```

Review the reported source columns, client IDs, row count, client count, and derived-order rule.
Do not start inference if the workbook order or client blocks are incorrect.

### 6. Run the synthetic tests

```bash
pytest -q
```

All tests should pass before real notes are sent to a model.

### 7. Point each model configuration at its local tokenizer

Edit `tokenizer_id` in the three files below so request token counting uses the exact local
checkpoint that vLLM serves:

```text
clinical_target_extraction/configs/qwen.yaml
  tokenizer_id: /absolute/persistent/path/to/clinical-target-models/qwen3-30b-a3b-instruct-2507

clinical_target_extraction/configs/medgemma.yaml
  tokenizer_id: /absolute/persistent/path/to/clinical-target-models/medgemma-27b-text-it

clinical_target_extraction/configs/gpt_oss.yaml
  tokenizer_id: /absolute/persistent/path/to/clinical-target-models/gpt-oss-120b
```

Keep `served_model_name: extractor`, since the extraction client sends that name to the local
OpenAI-compatible endpoint.

### 8. Start exactly one vLLM server

Start with Qwen. Run this in a dedicated terminal and leave it running:

```bash
vllm serve "$MODEL_ROOT/qwen3-30b-a3b-instruct-2507" \
  --served-model-name extractor \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --host 127.0.0.1 \
  --port 8000
```

For MedGemma, stop Qwen and start:

```bash
vllm serve "$MODEL_ROOT/medgemma-27b-text-it" \
  --served-model-name extractor \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --host 127.0.0.1 \
  --port 8000
```

For GPT-OSS, stop the previous server and preserve its native quantized loading with `auto`:

```bash
vllm serve "$MODEL_ROOT/gpt-oss-120b" \
  --served-model-name extractor \
  --dtype auto \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --host 127.0.0.1 \
  --port 8000
```

Check that the active server is reachable:

```bash
curl http://127.0.0.1:8000/v1/models
```

Do not start multiple model servers on the same GPU. The official model cards document vLLM
serving for [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507),
[MedGemma](https://huggingface.co/google/medgemma-27b-text-it), and
[GPT-OSS](https://huggingface.co/openai/gpt-oss-120b). OpenAI also documents GPT-OSS 120B as
an open-weight model designed to fit on one H100-class GPU in its
[official model documentation](https://developers.openai.com/api/docs/models/gpt-oss-120b).

### 9. Run a selected-client Qwen dry run

In a second terminal, activate the same environment and select 3–5 client IDs found by the
`inspect-data` command:

```bash
source .venv/bin/activate

clinical-target-extraction run \
  --data-config clinical_target_extraction/configs/data.yaml \
  --model-config clinical_target_extraction/configs/qwen.yaml \
  --client-id C001 \
  --client-id C002 \
  --client-id C003
```

Alternatively, place one client ID per line in a text file:

```bash
clinical-target-extraction run \
  --data-config clinical_target_extraction/configs/data.yaml \
  --model-config clinical_target_extraction/configs/qwen.yaml \
  --client-file selected_clients.txt
```

The CLI requires an explicit client list and enforces a five-client dry-run limit. It will not
silently launch inference over the full workbook.

### 10. Review the dry run before using another model

Inspect these files under `clinical_target_extraction/outputs/qwen/`:

```text
run_config.json
raw_responses.jsonl
validated_sessions.jsonl
registries/
session_targets.parquet
pairwise_comparisons.parquet
failures.jsonl
```

Manually review target granularity, duplicate targets, treatment and observation labels,
comparison validity, evidence quotations, context length, and JSON failure rate. Do not proceed
to the complete dataset until this review is satisfactory.

After review, repeat the selected-client run with `medgemma.yaml` and `gpt_oss.yaml`, loading
only the corresponding server. Each model writes to its own output directory and builds its own
target registry.

## What the pipeline implements

For each selected client, the pipeline:

1. preserves Excel row order and derives sessions 1…n;
2. sends session 1 alone for initial target discovery;
3. assigns stable target IDs in Python;
4. sends the current registry, all notes through the current session, and prior structured
   records for each later session;
5. compares every registered target with every earlier session separately;
6. canonicalizes new-target candidates only after existing targets are assessed;
7. validates schema, identifiers, logical constraints, and verbatim evidence quotations;
8. retains an invalid response and makes at most one repair call;
9. records a context overflow instead of truncating or summarizing history;
10. checkpoints JSONL, registry, and Parquet outputs after every successful session.

No future note is included in an earlier prompt. A checkpoint is reused only when its complete
request fingerprint—including note history, prompts, model settings, prior records, and registry
state—has not changed.
