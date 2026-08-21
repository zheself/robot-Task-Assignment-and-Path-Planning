#!/usr/bin/env bash
set -euo pipefail

if [[ "${A4B_CONFIRM_METADATA_RECOVERY_SUBMIT:-}" != "YES" ]]; then
  echo "set A4B_CONFIRM_METADATA_RECOVERY_SUBMIT=YES after preserving/cancelling jobs 984112-984115" >&2
  exit 64
fi
if squeue -h -n a4b-v2-recover,a4b-v2-merge,a4b-v2-smoke,a4b-v2-dev,a4b-v2-finalize | grep -q .; then
  echo "an A4b v2 recovery/development job is already queued or running" >&2
  exit 65
fi

shard_root="outputs/phase1_allocation/a4b_ordinary_lns_dev_v2/train_calibration_shards"
cells=(iid_small iid_medium dense_precedence resource_bottleneck tight_windows scale)
for cell in "${cells[@]}"; do
  [[ -s "${shard_root}/${cell}.jsonl" ]] || {
    echo "missing frozen calibration trace shard: ${cell}" >&2
    exit 66
  }
  [[ ! -e "${shard_root}/${cell}.json" ]] || {
    echo "unexpected pre-existing recovered metadata: ${cell}" >&2
    exit 67
  }
done

recovery_job="$(sbatch --parsable slurm/a4b_v2_recover_merge_gate_labels.sbatch)"
smoke_job="$(sbatch --parsable --dependency="afterok:${recovery_job}" slurm/a4b_v2_development_smoke.sbatch)"
development_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" slurm/a4b_v2_development_packed.sbatch)"
finalize_job="$(sbatch --parsable --dependency="afterok:${development_job}" slurm/a4b_v2_finalize.sbatch)"

echo "recovery_merge_gate_labels_job=${recovery_job}"
echo "smoke_job=${smoke_job}"
echo "development_job=${development_job}"
echo "finalize_job=${finalize_job}"
