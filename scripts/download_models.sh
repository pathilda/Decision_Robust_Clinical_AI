#!/usr/bin/env bash
# Download ONE model on a LOGIN node (compute nodes usually have no internet).
#
#   bash scripts/download_models.sh <key> --models-dir <MODELS_DIR> --hf-home <HF_HOME>
#
#   bash scripts/download_models.sh qwen35-4b    --models-dir /project/me/models --hf-home /project/me/hf_cache
#   bash scripts/download_models.sh qwen35-27b   --models-dir /project/me/models --hf-home /project/me/hf_cache
#   bash scripts/download_models.sh medgemma-4b  --models-dir /project/me/models --hf-home /project/me/hf_cache
#   bash scripts/download_models.sh medgemma-27b --models-dir /project/me/models --hf-home /project/me/hf_cache
#
# The model is written to <MODELS_DIR>/<key>/ ; <HF_HOME> holds the Hugging Face cache and the
# login token (run `HF_HOME=<HF_HOME> hf auth login` once, or export HF_TOKEN). The two MedGemma
# models are gated: accept the Health AI Developer Foundations terms on their pages first.
#
# The repository's current commit hash is resolved FIRST (this also verifies the ID exists and
# is accessible), the download is pinned to exactly that commit, and the hash is written to
# <MODELS_DIR>/<key>/REVISION (recorded in every output record and in manifest.json).
set -euo pipefail

usage() { echo "usage: $0 {qwen35-4b|qwen35-27b|medgemma-4b|medgemma-27b} --models-dir DIR --hf-home DIR" >&2; exit 2; }

key="${1:-}"; shift || true
models_dir=""; hf_home=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) models_dir="${2:-}"; shift 2 ;;
    --hf-home)    hf_home="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$models_dir" && -n "$hf_home" ]] || usage

case "$key" in
  qwen35-4b)    repo="Qwen/Qwen3.5-4B" ;;
  qwen35-27b)   repo="Qwen/Qwen3.5-27B" ;;
  medgemma-4b)  repo="google/medgemma-4b-it" ;;
  medgemma-27b) repo="google/medgemma-27b-text-it" ;;
  *) usage ;;
esac

export HF_HOME="$hf_home"
target="$models_dir/$key"
mkdir -p "$models_dir" "$hf_home"

revision="$(python - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1]).sha)
PY
)"
echo ">>> $repo resolved to commit $revision"

if command -v hf >/dev/null 2>&1; then
  hf download "$repo" --revision "$revision" --local-dir "$target"
else
  huggingface-cli download "$repo" --revision "$revision" --local-dir "$target"
fi

echo "$revision" > "$target/REVISION"
echo ">>> $key ready at $target (REVISION $revision)"
