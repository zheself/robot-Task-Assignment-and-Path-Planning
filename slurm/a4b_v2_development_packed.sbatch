#!/bin/bash
#SBATCH --job-name=a4b-v2-dev
#SBATCH --partition=normal
#SBATCH --account=v-chengwy
#SBATCH --nodelist=sist_gpu59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/a4b_v2_development_packed_%j.out
#SBATCH --error=logs/a4b_v2_development_packed_%j.err
set -euo pipefail
PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
CELLS=(iid_small iid_medium dense_precedence resource_bottleneck tight_windows scale)
cd "$PROJECT"
export PYTHONPATH=src

pids=()
for index in "${!CELLS[@]}"; do
  cell="${CELLS[$index]}"
  (
    ulimit -v 4194304
    export A4B_WORKER_INDEX="$index"
    exec scripts/run_a4b_cpu_worker.sh .venv/bin/python \
      scripts/run_a4b_ordinary_lns_development.py run-cell --cell "$cell"
  ) >"logs/a4b_v2_development_${SLURM_JOB_ID}_${cell}.out" \
    2>"logs/a4b_v2_development_${SLURM_JOB_ID}_${cell}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
