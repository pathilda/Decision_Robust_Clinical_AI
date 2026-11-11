#!/bin/bash
# OR candidate pools (step 5) for one or more models on one Trillium CPU node; no GPU, no LLM.
# Every (model, rung, source, profile) runs as its own single-threaded process in parallel
# (2 models x 6 rungs x 3 sources x 3 profiles = 108 processes on 192 cores).
# Submit from the CPU login node (tri-login01), from the repository root:
#
#   mkdir -p $SCRATCH/dr_logs/or
#   sbatch --output=$SCRATCH/dr_logs/%x-%j.out slurm/or.sbatch /scratch/me/dr_outputs \
#     /path/to/data qwen35-4b medgemma-4b
#
# Resumable: finished pools with the same inputs are reused, so a resubmit only does the rest.
#SBATCH --account=def-abouee
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dr-or

set -uo pipefail
OUTPUTS="$1"; DATA="$2"; shift 2
module load python/3.11.5 gurobi
export PYTHONDONTWRITEBYTECODE=1
cd "$SLURM_SUBMIT_DIR"
LOGS="$SCRATCH/dr_logs/or"
mkdir -p "$LOGS"

for model in "$@"; do
  for rung in 1 2 3 4 5 6; do
    for source in label distribution score; do
      for profile in balanced unbalanced highly_unbalanced; do
        python -m dr.candidates --model "$model" --rung "$rung" --source "$source" \
          --profile "$profile" --outputs-dir "$OUTPUTS" --data-dir "$DATA" \
          > "$LOGS/$model-r$rung-$source-$profile.log" 2>&1 &
      done
    done
  done
done
wait
echo "failed processes (non-empty Traceback):"
grep -l Traceback "$LOGS"/*.log || echo none
