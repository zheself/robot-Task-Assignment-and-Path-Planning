#!/bin/bash
#SBATCH --job-name=a4a-validation
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a4a_validation_%A_%a.out
#SBATCH --error=logs/a4a_validation_%A_%a.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
CELLS=(iid_small iid_medium dense_precedence resource_bottleneck tight_windows scale)
cd "$PROJECT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/run_a4_warm_start_pilot.py evaluate --cell "${CELLS[$SLURM_ARRAY_TASK_ID]}"
