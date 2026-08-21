#!/usr/bin/env bash
set -euo pipefail

if [[ "${A4B_CONFIRM_PARALLEL_SUBMIT:-}" != "YES" ]]; then
  echo "set A4B_CONFIRM_PARALLEL_SUBMIT=YES after preserving/cancelling the serial chain" >&2
  exit 64
fi
if squeue -h -n a4b-v2-cal,a4b-v2-train | grep -q .; then
  echo "an A4b calibration job is already queued or running" >&2
  exit 65
fi

calibration_job="$(sbatch --parsable slurm/a4b_v2_calibration_packed.sbatch)"
merge_job="$(sbatch --parsable --dependency="afterok:${calibration_job}" slurm/a4b_v2_merge_gate_labels.sbatch)"
smoke_job="$(sbatch --parsable --dependency="afterok:${merge_job}" slurm/a4b_v2_development_smoke.sbatch)"
development_job="$(sbatch --parsable --dependency="afterok:${smoke_job}" slurm/a4b_v2_development_packed.sbatch)"
finalize_job="$(sbatch --parsable --dependency="afterok:${development_job}" slurm/a4b_v2_finalize.sbatch)"

echo "calibration_job=${calibration_job}"
echo "merge_gate_labels_job=${merge_job}"
echo "smoke_job=${smoke_job}"
echo "development_job=${development_job}"
echo "finalize_job=${finalize_job}"
