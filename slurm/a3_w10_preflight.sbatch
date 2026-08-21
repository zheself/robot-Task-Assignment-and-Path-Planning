#!/usr/bin/env bash
#SBATCH --job-name=a3-w10-preflight
#SBATCH --partition=cpu
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-preflight-%j.out
#SBATCH --error=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/logs/a3-w10-preflight-%j.err

set -euo pipefail

PROJECT_DIR="/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL"
cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "job_id=${SLURM_JOB_ID:-} node_list=${SLURM_JOB_NODELIST:-}"
.venv/bin/python -c '
import sys
import numpy
import scipy
import torch
import safe_residual_rl
from safe_residual_rl.allocation.a3_protocol import prepare_a3_development
from safe_residual_rl.allocation.oracle import load_oracle_context

assert not any("/.local/" in p for p in sys.path), sys.path
context = load_oracle_context("configs/allocation/oracle_proxy_v1.json")
prepared = prepare_a3_development(
    "outputs/phase1_allocation/a3_development_v1/data", context
)
assert len(prepared.train_examples) == 192
assert len(prepared.validation_examples) == 48
print("python=" + sys.version.replace("\n", " "))
print("numpy=" + numpy.__version__ + " scipy=" + scipy.__version__ + " torch=" + torch.__version__)
print("project_import=ok train=192 validation=48 user_site_isolated=true")
'
