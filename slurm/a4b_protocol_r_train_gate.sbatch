#!/bin/bash
#SBATCH --job-name=a4b-r3-train-gate
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/a4b_protocol_r_train_gate_%j.out
#SBATCH --error=logs/a4b_protocol_r_train_gate_%j.err
set -euo pipefail
cd /public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
export PYTHONPATH=src
scripts/run_a4b_protocol_r_worker.sh .venv/bin/python scripts/run_a4b_protocol_r.py merge-train-gate
