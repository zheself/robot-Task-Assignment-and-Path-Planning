#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

allowed="$(awk '$1 == "Cpus_allowed_list:" {print $2}' /proc/self/status)"
worker_index="${A4B_WORKER_INDEX:-0}"
if [[ ! "${worker_index}" =~ ^[0-9]+$ ]]; then
  echo "invalid A4B_WORKER_INDEX=${worker_index}" >&2
  exit 64
fi
cpus=()
IFS=',' read -r -a ranges <<< "${allowed}"
for range in "${ranges[@]}"; do
  if [[ "${range}" == *-* ]]; then
    start="${range%-*}"
    end="${range#*-}"
    for ((cpu=start; cpu<=end; cpu++)); do cpus+=("${cpu}"); done
  else
    cpus+=("${range}")
  fi
done
if (( worker_index >= ${#cpus[@]} )); then
  echo "worker index ${worker_index} exceeds affinity=${allowed}" >&2
  exit 64
fi
worker_cpu="${cpus[$worker_index]}"
echo "worker_start_utc=$(date --iso-8601=seconds)" >&2
echo "worker_node=$(hostname) slurm_job_id=${SLURM_JOB_ID:-}" >&2
echo "worker_original_affinity=${allowed} worker_index=${worker_index} worker_bound_cpu=${worker_cpu}" >&2
echo "worker_loadavg=$(cat /proc/loadavg)" >&2
echo "worker_threads=OMP:${OMP_NUM_THREADS},MKL:${MKL_NUM_THREADS},OPENBLAS:${OPENBLAS_NUM_THREADS},NUMEXPR:${NUMEXPR_NUM_THREADS} cuda='${CUDA_VISIBLE_DEVICES}'" >&2
exec taskset --cpu-list "${worker_cpu}" "$@"
