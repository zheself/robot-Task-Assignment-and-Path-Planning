#!/usr/bin/env python3
"""Generate the preregistered a35f1 benchmark exactly once after source seal."""

from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from safe_residual_rl.allocation.pointer_final import generate_final_benchmark, load_final_protocol, sha256_file, verify_registered_locks

CONFIRM = "GENERATE_A35F1_FROZEN_ONCE"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--confirm",required=True); parser.add_argument("--protocol",type=Path,default=Path("configs/allocation/a3_5_sealed_final_v1.json")); parser.add_argument("--seal",type=Path,default=Path("configs/allocation/a3_5_final_evaluator_seal_v1.json")); parser.add_argument("--output",type=Path,default=Path("outputs/phase1_allocation/a3_5_sealed_final_v1_benchmark")); args=parser.parse_args()
    if args.confirm != CONFIRM: raise PermissionError("exact frozen-generation confirmation token required")
    root=Path(__file__).resolve().parents[1]; protocol=load_final_protocol(root/args.protocol)
    failures=verify_registered_locks(root,protocol)
    if failures: raise RuntimeError("registered lock failure: "+",".join(failures))
    seal_path=root/args.seal; seal=json.loads(seal_path.read_text()); _verify_seal(root,seal)
    output=root/args.output
    if output.exists(): raise FileExistsError("a35f1 benchmark already exists; regeneration forbidden")
    if (root/"outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation").exists(): raise FileExistsError("evaluation output exists before generation")
    manifest=generate_final_benchmark(root,protocol,output,sha256_file(seal_path))
    print(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"instances":len(manifest["records"]),"groups":len({x["task_group_id"] for x in manifest["records"]}),"manifest_sha256":manifest["manifest_sha256"],"manifest_file_sha256":sha256_file(output/"manifest.json")},sort_keys=True),flush=True)

def _verify_seal(root,seal):
    for relative,expected in seal["source_hashes"].items():
        if sha256_file(root/relative)!=expected: raise RuntimeError(f"sealed source changed: {relative}")

if __name__=="__main__": main()
