#!/bin/bash
#SBATCH --job-name=a35f-evaluate
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/a35f-evaluate-%j.out
#SBATCH --error=logs/a35f-evaluate-%j.err
set -euo pipefail
PROJECT_ROOT="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "$PROJECT_ROOT"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --unbuffered .venv/bin/python scripts/run_a3_5_final_evaluation.py --mode sealed --confirm RUN_A3_5_SEALED_FINAL_ONCE
