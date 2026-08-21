#!/bin/bash
#SBATCH --job-name=a35f-generate
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a35f-generate-%j.out
#SBATCH --error=logs/a35f-generate-%j.err
set -euo pipefail
PROJECT_ROOT="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "$PROJECT_ROOT"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --unbuffered .venv/bin/python scripts/generate_a3_5_final_benchmark.py --confirm GENERATE_A35F1_FROZEN_ONCE
