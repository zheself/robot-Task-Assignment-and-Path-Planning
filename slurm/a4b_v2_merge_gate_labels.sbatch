#!/bin/bash
#SBATCH --job-name=a4b-v2-merge
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodelist=sist_gpu59
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a4b_v2_merge_%j.out
#SBATCH --error=logs/a4b_v2_merge_%j.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
cd "$PROJECT"
export PYTHONPATH=src
scripts/run_a4b_cpu_worker.sh .venv/bin/python scripts/run_a4b_ordinary_lns_development.py merge-calibration
scripts/run_a4b_cpu_worker.sh .venv/bin/python scripts/run_a4b_ordinary_lns_development.py train-gate
scripts/run_a4b_cpu_worker.sh .venv/bin/python scripts/run_a4b_ordinary_lns_development.py labels
