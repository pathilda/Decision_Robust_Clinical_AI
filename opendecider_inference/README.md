# Local OpenDecider inference

This folder is a separate, local-only inference path for
`manjunathshiva/opendecider-medium-td`. It uses the same workbook rows, first-row
duplicate policy, combined note, and three category definitions as the Qwen and
MedGemma pipeline. OpenDecider's native confidence and category probabilities are
preserved as additional columns.

OpenDecider does not accept the literal chat prompt used by Qwen and MedGemma. Its
equivalent input is the same combined client note plus a typed `choice` question whose
three criteria reproduce the chat prompt's category definitions. This keeps the task,
input rows, and output labels comparable while using OpenDecider's intended interface.

It never modifies the downloaded adapter. At runtime it makes a temporary symlink
overlay whose `opendecider.json` points to the explicit local base-model path. Both
paths therefore remain separate and no Hub lookup is needed.

## Expected local files

The folder names are examples; pass your actual paths on the command line:

```text
/scratch/pathilda/Models/
├── Qwen/                                  # existing, unrelated project Qwen
├── MedGemma/
├── Qwen3-30B-A3B-Instruct-2507/           # OpenDecider base
└── OpenDecider-medium-td/                  # OpenDecider adapter
```

Required files include:

```text
Qwen3-30B-A3B-Instruct-2507/config.json
OpenDecider-medium-td/adapter_config.json
OpenDecider-medium-td/adapter_model.safetensors
OpenDecider-medium-td/opendecider.json
```

## Exact roadmap

### 1. Create an isolated environment

From the repository root:

```bash
python -m venv .venv-opendecider
source .venv-opendecider/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[opendecider,dev]"
pytest -q opendecider_inference/tests
```

### 2. Inspect the workbook

This uses the same input validation as Qwen and MedGemma and does not load a model:

```bash
python -m opendecider_inference \
  --pipeline inspect \
  --input /path/to/clients.xlsx \
  --sheet Clients
```

### 3. Run a one-client local smoke test

```bash
python -m opendecider_inference \
  --pipeline classify \
  --input /path/to/clients.xlsx \
  --sheet Clients \
  --adapter-path /scratch/pathilda/Models/OpenDecider-medium-td \
  --base-path /scratch/pathilda/Models/Qwen3-30B-A3B-Instruct-2507 \
  --output /scratch/pathilda/client_classification_results \
  --max-clients 1 \
  --batch-size 1 \
  --deidentified-confirmed
```

To run a known client instead of the first client, replace `--max-clients 1` with
`--client-id C001`, using the exact ID from the workbook.

The runner forces these settings before importing OpenDecider:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

Use a compute node or container with outbound networking disabled when an operating-system-level
guarantee is required.

### 4. Review the smoke-test output

```text
/scratch/pathilda/client_classification_results/opendecider/
├── run_config.json
├── raw_responses.jsonl
├── validated_classifications.jsonl
├── failures.jsonl                         # created only when failures occur
├── client_classifications.csv
├── client_classifications.xlsx
└── client_classifications.parquet         # when a Parquet engine is installed
```

The flat table begins with the same comparison fields as other models and adds probabilities:

```text
model
Client AlayaCare Client ID
treatment_category
confidence
probability_easy
probability_moderate
probability_severe
```

### 5. Run a pilot

Run 10–20 clients and inspect classifications, failures, probability sums, GPU memory, and cases
where the two largest probabilities are close:

```bash
python -m opendecider_inference \
  --pipeline classify \
  --input /path/to/clients.xlsx \
  --sheet Clients \
  --adapter-path /scratch/pathilda/Models/OpenDecider-medium-td \
  --base-path /scratch/pathilda/Models/Qwen3-30B-A3B-Instruct-2507 \
  --output /scratch/pathilda/client_classification_results \
  --max-clients 20 \
  --batch-size 4 \
  --deidentified-confirmed
```

### 6. Run the complete cohort

Remove `--max-clients` after the pilot succeeds:

```bash
python -m opendecider_inference \
  --pipeline classify \
  --input /path/to/clients.xlsx \
  --sheet Clients \
  --adapter-path /scratch/pathilda/Models/OpenDecider-medium-td \
  --base-path /scratch/pathilda/Models/Qwen3-30B-A3B-Instruct-2507 \
  --output /scratch/pathilda/client_classification_results \
  --batch-size 8 \
  --deidentified-confirmed
```

Validated unchanged clients are reused on subsequent runs. A changed note, rubric, adapter
configuration, adapter metadata, or base configuration produces a new classification.

### 7. Compare against Qwen and MedGemma

Run all models on the same selected client IDs and compare the common columns:

```text
Client AlayaCare Client ID
treatment_category
```

OpenDecider probabilities are additional evidence; they are not directly comparable to generated
text-model confidence. For a real performance comparison, join all three outputs to adjudicated
clinician labels and calculate macro F1, severe-category recall, quadratic-weighted kappa, and a
confusion matrix. Evaluate OpenDecider's Brier score and calibration separately.

Keep the workbook, sheet, selected client IDs, duplicate-row policy, and rubric fixed across all
three runs. Record the exact local model directories and preserve each `run_config.json`.
