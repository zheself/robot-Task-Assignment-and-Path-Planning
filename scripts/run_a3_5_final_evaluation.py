#!/usr/bin/env python3
"""Validation-only preflight or the unique sealed A3.5 final evaluation."""

from __future__ import annotations
import argparse,csv,json,os,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, scipy, torch
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_final import aggregate_final,audit_witnesses,evaluate_candidates,load_final_items,load_final_protocol,load_fixed_models,sha256_file,validation_items,verify_registered_locks
from safe_residual_rl.allocation.pointer_pilot import load_pointer_pilot_config
from safe_residual_rl.allocation.pointer_training import prepare_pointer_pilot

CONFIRM="RUN_A3_5_SEALED_FINAL_ONCE"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("development","sealed"),required=True); p.add_argument("--confirm",default=""); p.add_argument("--protocol",type=Path,default=Path("configs/allocation/a3_5_sealed_final_v1.json")); p.add_argument("--seal",type=Path,default=Path("configs/allocation/a3_5_final_evaluator_seal_v1.json")); p.add_argument("--benchmark",type=Path,default=Path("outputs/phase1_allocation/a3_5_sealed_final_v1_benchmark")); p.add_argument("--output",type=Path,default=None); p.add_argument("--per-cell",type=int,default=1); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; protocol=load_final_protocol(root/args.protocol)
    failures=verify_registered_locks(root,protocol)
    if failures: raise RuntimeError("registered lock failure: "+",".join(failures))
    context=load_oracle_context(root/"configs/allocation/oracle_proxy_v1.json")
    pilot_cfg=load_pointer_pilot_config(root/"configs/allocation/a3_5_pointer_pilot_v1.json")
    pilot_root=root/"outputs/phase1_allocation/a3_5_pointer_pilot_v1"
    prepared=prepare_pointer_pilot(pilot_root/"data",pilot_root/"manifest.json",context,pilot_cfg)
    models=load_fixed_models(root,protocol,prepared); torch.set_num_threads(1)
    if args.mode=="development":
        output=root/(args.output or Path("outputs/phase1_allocation/a3_5_final_evaluator_development_v1"))
        if output.exists(): raise FileExistsError("development output is non-overwriting")
        items=validation_items(prepared,args.per_cell); rows,raw=evaluate_candidates(items,models,prepared,context,protocol)
        checks={"validation_only":all(x["split"]=="validation" for x in rows),"complete_matrix":len(rows)==len(items)*9,"finite_runtime":all(np.isfinite(x["runtime_s"]) for x in rows),"zero_mask":sum(x["hard_mask_violations"] for x in rows)==0,"zero_atomicity":sum(x["atomicity_violations"] for x in rows)==0}
        output.mkdir(parents=True); _json(output/"summary.json",{"checks":checks,"passed":all(checks.values()),"instances":len(items),"rows":rows}); _json(output/"raw_predictions.json",raw)
        print(json.dumps({"mode":"development","instances":len(items),"rows":len(rows),"passed":all(checks.values())}),flush=True)
        if not all(checks.values()): raise RuntimeError("validation-only preflight failed")
        return
    if args.confirm!=CONFIRM: raise PermissionError("exact one-time sealed confirmation token required")
    seal_path=root/args.seal; seal=json.loads(seal_path.read_text()); _verify_seal(root,seal)
    benchmark=root/args.benchmark; output=root/(args.output or Path("outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation"))
    if output.exists(): raise FileExistsError("sealed evaluation output exists; rerun forbidden")
    output.mkdir(parents=True)
    _json(output/"attempt_started.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"status":"STARTED_BEFORE_FROZEN_ACCESS","slurm_job_id":os.environ.get("SLURM_JOB_ID"),"node":os.environ.get("SLURM_JOB_NODELIST"),"protocol_sha256":sha256_file(root/args.protocol),"seal_sha256":sha256_file(seal_path)})
    try:
        items=load_final_items(root,benchmark); rows,raw=evaluate_candidates(items,models,prepared,context,protocol)
        _json(output/"predictions_before_witness.json",raw)
        witness_failures=audit_witnesses(benchmark,items,raw,context)
        summary=aggregate_final(rows,protocol,witness_failures)
        summary.update({"created_utc":datetime.now(timezone.utc).isoformat(),"evidence_label":"SIM_GEOMETRIC","protocol_sha256":sha256_file(root/args.protocol),"seal_sha256":sha256_file(seal_path),"manifest_file_sha256":sha256_file(benchmark/"manifest.json"),"versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__},"accessed_split":"frozen_test","v4_accessed":False,"repair_used":False})
        _json(output/"summary.json",summary); _json(output/"failure_library.json",[x for x in rows if not x["verified"]]); _csv(output/"rows.csv",rows)
        print(json.dumps({"result_class":summary["result_class"],"difference":summary["primary"]["mean_paired_coverage_difference"],"p":summary["primary"]["one_sided_randomization_p"],"ci":summary["primary"]["cluster_bootstrap_ci95"]}),flush=True)
    except Exception as exc:
        _json(output/"invalid.json",{"result_class":"A3_5_FINAL_INVALID","error":type(exc).__name__,"message":str(exc)}); raise

def _verify_seal(root,seal):
    for relative,expected in seal["source_hashes"].items():
        if sha256_file(root/relative)!=expected: raise RuntimeError(f"sealed source changed: {relative}")
def _json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True),encoding="utf-8")
def _csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__=="__main__": main()
