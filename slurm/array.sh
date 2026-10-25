#!/bin/bash
# One fit per array task on the SoC Compute Cluster.
#   sbatch --array=0-$(( $(python -m stg.sweep --stage test --count) - 1 )) slurm/array.sh test
# The stage is the first argument (validation, test or synthetic).
#SBATCH --job-name=stg
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -m stg.sweep --stage "${1:-test}" --index "$SLURM_ARRAY_TASK_ID" --device cuda
