#!/usr/bin/env bash
#SBATCH --job-name=a3-w10
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%9
#SBATCH --output=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-%A_%a.out
#SBATCH --error=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-%A_%a.err

set -euo pipefail

PROJECT_DIR="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "${PROJECT_DIR}"
mkdir -p logs

families=(edge_mlp edge_mlp edge_mlp hetero_gnn hetero_gnn hetero_gnn graph_transformer graph_transformer graph_transformer)
seeds=(17 29 43 17 29 43 17 29 43)
family="${families[${SLURM_ARRAY_TASK_ID}]}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "job_id=${SLURM_JOB_ID:-} array_task=${SLURM_ARRAY_TASK_ID:-}"
echo "node_list=${SLURM_JOB_NODELIST:-} partition=${SLURM_JOB_PARTITION:-}"
echo "family=${family} seed=${seed} start_time=$(date --iso-8601=seconds)"
.venv/bin/python --version
.venv/bin/python -c 'import sys, numpy, scipy, torch, safe_residual_rl; assert not any("/.local/" in p for p in sys.path); print("preflight_imports=ok")'

srun --unbuffered /usr/bin/env .venv/bin/python \
  scripts/run_a3_w10_shard.py \
  --family "${family}" \
  --seed "${seed}"

echo "end_time=$(date --iso-8601=seconds)"
