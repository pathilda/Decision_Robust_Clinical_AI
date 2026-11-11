#!/bin/bash
# OR candidate pools (step 5) on one Trillium CPU node (192 cores); no GPU, no LLM.
# Every (model, rung, source, profile) runs as its own single-threaded process in parallel.
# Gurobi: gurobipy uses the license it finds (GRB_LICENSE_FILE, else ~/gurobi.lic). Check first
# that it works on a compute node with this many sessions: slurm/gurobi_check.sbatch.
# Submit from the CPU login node (tri-login01), from the repository root:
#
#   sbatch --output=$SCRATCH/dr_logs/%x-%j.out slurm/or.sbatch <OUTPUTS_DIR> <DATA_DIR> \
#     qwen35-4b medgemma-4b ...
#
# Each solve has the time limit in config/study.yaml; pools that hit it are saved as
# "solve_failed". Resumable: finished pools with the same inputs are reused on resubmit.
#SBATCH --account=def-abouee
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dr-or

set -uo pipefail
OUTPUTS="$1"; DATA="$2"; shift 2
module load python/3.11.5
export PYTHONDONTWRITEBYTECODE=1
cd "$SLURM_SUBMIT_DIR"
LOGS="$SCRATCH/dr_logs/or-$SLURM_JOB_ID"
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
echo "processes that crashed (Traceback in log):"
grep -l Traceback "$LOGS"/*.log || echo none
echo "pool status counts:"
find "$OUTPUTS" -path '*/candidates/*' -name 'chunk*.json' -newer "$LOGS" \
  -exec grep -ho '"status": "[a-z_]*"' {} + | sort | uniq -c
