#!/bin/bash
#SBATCH --job-name=a35-aggregate
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a35-aggregate-%j.out
#SBATCH --error=logs/a35-aggregate-%j.err

set -euo pipefail
PROJECT_ROOT="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "$PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
srun --unbuffered .venv/bin/python scripts/aggregate_a3_5_pointer_pilot.py
