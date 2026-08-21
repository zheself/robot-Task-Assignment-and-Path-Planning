#!/bin/bash
#SBATCH --job-name=a35-models
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --array=0-14%15
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/a35-model-%A_%a.out
#SBATCH --error=logs/a35-model-%A_%a.err

set -euo pipefail
PROJECT_ROOT="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "$PROJECT_ROOT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
VARIANTS=(edge_mlp_static hetero_gnn_static graph_transformer_static hetero_gnn_pair_pointer graph_transformer_pair_pointer)
SEEDS=(101 211 307)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
srun --unbuffered .venv/bin/python scripts/run_a3_5_pointer_shard.py --variant "${VARIANTS[$VARIANT_INDEX]}" --seed "${SEEDS[$SEED_INDEX]}"
