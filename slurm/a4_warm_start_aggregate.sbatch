#!/bin/bash
#SBATCH --job-name=a4a-aggregate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/a4a_aggregate_%j.out
#SBATCH --error=logs/a4a_aggregate_%j.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
cd "$PROJECT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/run_a4_warm_start_pilot.py aggregate
