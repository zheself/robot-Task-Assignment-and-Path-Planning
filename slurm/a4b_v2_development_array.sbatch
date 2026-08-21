#!/bin/bash
#SBATCH --job-name=a4b-v2-dev
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodelist=sist_gpu59
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/a4b_v2_development_%A_%a.out
#SBATCH --error=logs/a4b_v2_development_%A_%a.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
CELLS=(iid_small iid_medium dense_precedence resource_bottleneck tight_windows scale)
cd "$PROJECT"
export PYTHONPATH=src
scripts/run_a4b_cpu_worker.sh .venv/bin/python scripts/run_a4b_ordinary_lns_development.py run-cell --cell "${CELLS[$SLURM_ARRAY_TASK_ID]}"
