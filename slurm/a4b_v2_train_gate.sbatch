#!/bin/bash
#SBATCH --job-name=a4b-v2-train
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodelist=sist_gpu59
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/a4b_v2_train_%j.out
#SBATCH --error=logs/a4b_v2_train_%j.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
cd "$PROJECT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src
.venv/bin/python scripts/run_a4b_ordinary_lns_development.py calibrate
.venv/bin/python scripts/run_a4b_ordinary_lns_development.py train-gate
.venv/bin/python scripts/run_a4b_ordinary_lns_development.py labels
