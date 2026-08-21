#!/usr/bin/env bash
set -euo pipefail

if [[ "${A4B_CONFIRM_PROTOCOL_R_SUBMIT:-}" != "YES" ]]; then
  echo "set A4B_CONFIRM_PROTOCOL_R_SUBMIT=YES only after separate execution authorization" >&2
  exit 64
fi
if [[ "${A4B_PROTOCOL_R_EXECUTION_TOKEN:-}" != "EXECUTE_A4B_PROTOCOL_R_V3" ]]; then
  echo "missing exact Protocol-R execution token" >&2
  exit 64
fi
if squeue -h -u "${USER}" -o '%j' | grep -Eq '^a4b-r3-'; then
  echo "a Protocol-R job is already queued or running" >&2
  exit 65
fi

PROJECT=/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL
cd "${PROJECT}"
export PYTHONPATH=src
.venv/bin/python - <<'PY'
from pathlib import Path
from safe_residual_rl.allocation.search.data_protocol_r import load_protocol_r_config, require_execution_ready
c = load_protocol_r_config(Path('configs/allocation/a4b_protocol_r_freeze_candidate_v1.json'))
require_execution_ready(c)
PY

selected_node=""
for node in sist_gpu58 sist_gpu59 sist_gpu60; do
  read -r allocated total real_memory allocated_memory state < <(
    scontrol show node "${node}" | awk '
      /CPUAlloc=/ {for(i=1;i<=NF;i++){if($i~/^CPUAlloc=/){split($i,a,"=");ca=a[2]} if($i~/^CPUTot=/){split($i,a,"=");ct=a[2]}}}
      /RealMemory=/ {for(i=1;i<=NF;i++){if($i~/^RealMemory=/){split($i,a,"=");rm=a[2]} if($i~/^AllocMem=/){split($i,a,"=");am=a[2]}}}
      /State=/ {for(i=1;i<=NF;i++)if($i~/^State=/){split($i,a,"=");st=a[2]}}
      END{print ca,ct,rm,am,st}'
  )
  if (( total - allocated >= 6 && real_memory - allocated_memory >= 32768 )) && [[ "${state}" != *DOWN* && "${state}" != *DRAIN* ]]; then
    selected_node="${node}"
    break
  fi
done
if [[ -z "${selected_node}" ]]; then
  echo "no preregistered node currently has 6 CPUs and 32GiB Slurm-unallocated memory" >&2
  exit 66
fi

submit() {
  local script="$1" dependency="${2:-}"
  local args=(--parsable --nodelist="${selected_node}" --export=ALL)
  if [[ -n "${dependency}" ]]; then args+=(--dependency="afterok:${dependency}"); fi
  sbatch "${args[@]}" "${script}"
}

preflight_job="$(submit slurm/a4b_protocol_r_preflight.sbatch)"
generate_job="$(submit slurm/a4b_protocol_r_generate.sbatch "${preflight_job}")"
profile_job="$(submit slurm/a4b_protocol_r_profile_packed.sbatch "${generate_job}")"
profile_gate_job="$(submit slurm/a4b_protocol_r_profile_gate.sbatch "${profile_job}")"
calibration_job="$(submit slurm/a4b_protocol_r_calibration_packed.sbatch "${profile_gate_job}")"
train_gate_job="$(submit slurm/a4b_protocol_r_train_gate.sbatch "${calibration_job}")"
smoke_job="$(submit slurm/a4b_protocol_r_smoke.sbatch "${train_gate_job}")"
development_job="$(submit slurm/a4b_protocol_r_development_packed.sbatch "${smoke_job}")"
finalize_job="$(submit slurm/a4b_protocol_r_finalize.sbatch "${development_job}")"

mkdir -p logs
record="logs/a4b_protocol_r_submission_$(date -u +%Y%m%dT%H%M%SZ).json"
.venv/bin/python - "${record}" "${selected_node}" "${preflight_job}" "${generate_job}" "${profile_job}" "${profile_gate_job}" "${calibration_job}" "${train_gate_job}" "${smoke_job}" "${development_job}" "${finalize_job}" <<'PY'
import json, sys
path, node, *jobs = sys.argv[1:]
stages = ['preflight','generate','profile','profile_gate','calibration','train_gate','smoke','development','finalize']
rows=[]
for index,(stage,job) in enumerate(zip(stages,jobs)):
    rows.append({'stage':stage,'job_id':job,'dependency':None if index==0 else f'afterok:{jobs[index-1]}','node':node})
open(path,'w').write(json.dumps({'version':'a4b-protocol-r-submission-v1','jobs':rows},indent=2,sort_keys=True)+'\n')
PY
echo "selected_node=${selected_node}"
echo "submission_record=${record}"
for name in preflight generate profile profile_gate calibration train_gate smoke development finalize; do eval "value=\${${name}_job}"; echo "${name}_job=${value}"; done
