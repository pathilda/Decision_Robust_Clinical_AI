#!/bin/bash
# OR candidate pools (step 5); no GPU, no LLM. Each (model, rung, source, profile) is one
# single-threaded process; at most PARALLEL run at once (default: the number of CPU cores).
# Gurobi: gurobipy uses the license it finds (GRB_LICENSE_FILE, else ~/gurobi.lic). Check first
# that it works where you run, with this many sessions: slurm/gurobi_check.sbatch.
#
# As a batch job on a Trillium CPU node (submit from tri-login01, repository root):
#   sbatch --output=$SCRATCH/dr_logs/%x-%j.out slurm/or.sbatch <OUTPUTS_DIR> <DATA_DIR> MODEL...
# Directly, e.g. in an OnDemand terminal (repository root):
#   PARALLEL=16 bash slurm/or.sbatch <OUTPUTS_DIR> <DATA_DIR> MODEL...
#
# Each solve has the time limit in config/study.yaml; pools that hit it are saved as
# "solve_failed". Resumable: finished pools with the same inputs are reused on a rerun.
#SBATCH --account=def-abouee
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dr-or

set -uo pipefail
OUTPUTS="$1"; DATA="$2"; shift 2
PARALLEL="${PARALLEL:-$(nproc)}"
module load python/3.11.5 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
# Run from the repository: the script's own parent folder when run with bash (OnDemand sessions
# are Slurm jobs too, so SLURM_SUBMIT_DIR may point elsewhere); under sbatch the script is a
# spool copy, so fall back to the submit directory.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
[ -d "$REPO/dr" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO" || exit 1
[ -d dr ] || { echo "not in the repository ($REPO has no dr/); run from the repository root"; exit 1; }
LOGS="$SCRATCH/dr_logs/or-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGS"
echo "running up to $PARALLEL processes at once; logs in $LOGS"

for model in "$@"; do
  for rung in 1 2 3 4 5 6; do
    for source in label distribution score; do
      for profile in balanced unbalanced highly_unbalanced; do
        echo "$model $rung $source $profile"
      done
    done
  done
done | xargs -P "$PARALLEL" -L 1 bash -c '
  python -m dr.candidates --model "$0" --rung "$1" --source "$2" --profile "$3" \
    --outputs-dir "'"$OUTPUTS"'" --data-dir "'"$DATA"'" \
    > "'"$LOGS"'/$0-r$1-$2-$3.log" 2>&1'

# A finished process prints a {"step": "candidates", ...} summary; any log without one crashed.
echo "processes that did not finish (no summary line in log):"
grep -L '"step": "candidates"' "$LOGS"/*.log || echo none
echo "pool status counts:"
find "$OUTPUTS" -path '*/candidates/*' -name 'chunk*.json' -newer "$LOGS" \
  -exec grep -ho '"status": "[a-z_]*"' {} + | sort | uniq -c
