#!/bin/bash
#SBATCH --job-name=a35f-preflight
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a35f-preflight-%j.out
#SBATCH --error=logs/a35f-preflight-%j.err
set -euo pipefail
PROJECT_ROOT="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "$PROJECT_ROOT"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --unbuffered .venv/bin/python scripts/run_a3_5_final_evaluation.py --mode development --per-cell 2 --output outputs/phase1_allocation/a3_5_final_validation_preflight_v1
