#!/bin/bash
#SBATCH --job-name=a4b-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/a4b_smoke_%j.out
#SBATCH --error=logs/a4b_smoke_%j.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
cd "$PROJECT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/run_a4b_baseline_development.py smoke
