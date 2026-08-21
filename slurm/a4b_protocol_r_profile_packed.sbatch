#!/bin/bash
#SBATCH --job-name=a4b-r3-profile
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a4b_protocol_r_profile_%j.out
#SBATCH --error=logs/a4b_protocol_r_profile_%j.err
set -euo pipefail
cd /public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
export PYTHONPATH=src
cells=(iid_small iid_medium dense_precedence resource_bottleneck tight_windows scale)
pids=()
for index in "${!cells[@]}"; do
  cell="${cells[$index]}"
  (ulimit -v 4194304; export A4B_WORKER_INDEX="${index}"; exec scripts/run_a4b_protocol_r_worker.sh .venv/bin/python scripts/run_a4b_protocol_r.py profile-cell --cell "${cell}") >"logs/a4b_protocol_r_profile_${SLURM_JOB_ID}_${cell}.out" 2>"logs/a4b_protocol_r_profile_${SLURM_JOB_ID}_${cell}.err" &
  pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done; exit "${failed}"
