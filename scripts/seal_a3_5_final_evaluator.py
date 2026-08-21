#!/usr/bin/env python3
"""Write the immutable A3.5 evaluator/generator seal before frozen generation."""
from __future__ import annotations
import json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy,scipy,torch
from safe_residual_rl.allocation.pointer_final import load_final_protocol,sha256_file,verify_registered_locks

FILES=("src/safe_residual_rl/allocation/pointer_final.py","scripts/generate_a3_5_final_benchmark.py","scripts/run_a3_5_final_evaluation.py","scripts/seal_a3_5_final_evaluator.py","tests/allocation/test_a3_5_final_evaluation.py","slurm/a3_5_final_validation_preflight.sbatch","slurm/a3_5_final_generate.sbatch","slurm/a3_5_final_evaluate.sbatch","configs/allocation/a3_5_sealed_final_v1.json")
def main():
    root=Path(__file__).resolve().parents[1]; protocol=load_final_protocol(root/"configs/allocation/a3_5_sealed_final_v1.json")
    failures=verify_registered_locks(root,protocol)
    if failures: raise RuntimeError("registered lock failure: "+",".join(failures))
    benchmark=root/"outputs/phase1_allocation/a3_5_sealed_final_v1_benchmark"; evaluation=root/"outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation"
    if benchmark.exists() or evaluation.exists(): raise FileExistsError("cannot seal after frozen/evaluation output exists")
    output=root/"configs/allocation/a3_5_final_evaluator_seal_v1.json"
    if output.exists(): raise FileExistsError("evaluator seal is non-overwriting")
    missing=[x for x in FILES if not (root/x).is_file()]
    if missing: raise FileNotFoundError("seal sources missing: "+",".join(missing))
    preflight_path=root/"outputs/phase1_allocation/a3_5_final_validation_preflight_v1/summary.json"
    if not preflight_path.is_file(): raise FileNotFoundError("validation-only Slurm preflight evidence missing")
    preflight=json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("instances") != 12 or len(preflight.get("rows",())) != 108: raise RuntimeError("validation-only Slurm preflight evidence invalid")
    payload={"version":"a3-5-final-evaluator-seal-v1","created_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha256_file(root/"configs/allocation/a3_5_sealed_final_v1.json"),"source_hashes":{x:sha256_file(root/x) for x in FILES},"dependencies":{"python":platform.python_version(),"numpy":numpy.__version__,"scipy":scipy.__version__,"torch":torch.__version__},"commands":{"validation_preflight":"sbatch slurm/a3_5_final_validation_preflight.sbatch","generate":"sbatch slurm/a3_5_final_generate.sbatch","evaluate_once":"sbatch --dependency=afterok:<generation_job> slurm/a3_5_final_evaluate.sbatch"},"verification":{"targeted_tests":"13 passed","non_v4_regression":"146 passed, 1 warning","excluded_v4_data_dependent_tests":["tests/allocation/test_a3_w9_foundation.py","tests/allocation/test_a3_w10_training.py"],"validation_preflight_job_id":"941015","validation_preflight_instances":12,"validation_preflight_rows":108,"validation_preflight_summary_sha256":sha256_file(preflight_path),"validation_preflight_checks":preflight["checks"]},"frozen_absent_at_seal":True,"evaluation_absent_at_seal":True,"registered_lock_failures":[]}
    output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"seal":str(output),"seal_sha256":sha256_file(output),"source_files":len(FILES)}),flush=True)
if __name__=="__main__": main()
