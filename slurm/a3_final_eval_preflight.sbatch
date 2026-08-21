#!/usr/bin/env bash
#SBATCH --job-name=a3-final-preflight
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-final-preflight-%j.out
#SBATCH --error=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-final-preflight-%j.err

set -euo pipefail

PROJECT_DIR="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "${PROJECT_DIR}"
mkdir -p logs
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "job_id=${SLURM_JOB_ID:-} node_list=${SLURM_JOB_NODELIST:-}"
.venv/bin/python -c 'import numpy, scipy, torch, safe_residual_rl; print("imports=ok", numpy.__version__, scipy.__version__, torch.__version__)'
srun --unbuffered /usr/bin/env .venv/bin/python scripts/run_a3_final_evaluation.py \
  --mode development \
  --development-instances 4
