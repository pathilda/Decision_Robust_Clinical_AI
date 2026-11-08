#!/bin/bash
# One job = one model loaded once (spec 12.1). Every directory is passed on the command line:
#
#   sbatch --job-name=icf-qwen35-27b slurm/run.sbatch qwen35-27b --steps icf \
#     --models-dir /project/me/models --data-dir /project/me/dr_study/data \
#     --outputs-dir /project/me/dr_study/outputs
#
# Optional: --gurobi-license /path/gurobi.lic (candidates step). Submit from the repository
# root; create logs/ there first. Fill in the <...> placeholders below.
#SBATCH --account=<existing account>
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=dr
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load <existing modules>
source <existing venv>/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
python -m dr.run_model --model "$1" "${@:2}"
