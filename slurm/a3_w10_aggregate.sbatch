#!/usr/bin/env bash
#SBATCH --job-name=a3-w10-aggregate
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-aggregate-%j.out
#SBATCH --error=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-aggregate-%j.err

set -euo pipefail

PROJECT_DIR="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job_id=${SLURM_JOB_ID:-} node_list=${SLURM_JOB_NODELIST:-}"
echo "partition=${SLURM_JOB_PARTITION:-} start_time=$(date --iso-8601=seconds)"
.venv/bin/python --version
.venv/bin/python -c 'import sys, numpy, scipy, torch, safe_residual_rl; assert not any("/.local/" in p for p in sys.path); print("preflight_imports=ok")'

srun --unbuffered /usr/bin/env .venv/bin/python scripts/aggregate_a3_w10.py

echo "end_time=$(date --iso-8601=seconds)"
