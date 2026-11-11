#!/bin/bash
# One job = one model loaded once (spec 12.1). Written for Trillium (Alliance) GPU nodes.
# Submit from the GPU login node (trig-login01), from the repository root. $HOME is read-only on
# compute nodes, so the log goes to scratch: pass --output on the command line, e.g.
#
#   mkdir -p $SCRATCH/dr_logs
#   sbatch --job-name=rate-qwen35-27b --output=$SCRATCH/dr_logs/%x-%j.out slurm/run.sbatch \
#     qwen35-27b --steps rating allocation --batch-size 64 --models-dir $SCRATCH/Models \
#     --data-dir /path/to/data --outputs-dir $SCRATCH/dr_outputs
#
# Optional: --gurobi-license /path/gurobi.lic (candidates step). Runs are resumable: if a job
# hits the 24 h limit, resubmit the same command and finished records are skipped.
#SBATCH --account=def-abouee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dr

set -euo pipefail
module load python/3.11.5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Every cache that vLLM / Triton / torch.compile writes must live on scratch ($HOME is read-only).
CACHE_ROOT="$SCRATCH/.cache/dr"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"
python -m dr.run_model --model "$1" "${@:2}"
