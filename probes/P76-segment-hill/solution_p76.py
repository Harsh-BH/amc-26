"""P76 SageMaker entry (sm_train.py main contract): run hill.py / fr_hill.py search jobs on every vCPU.

The train channel holds the repo-relative frames (data/work/p76/{F,S}_*.parquet, data/work/p42/loco_*.npz,
data/work/p42/sm/grid.json); the job chdirs there, so the probe modules run unchanged.
params: {"kind": "iu" | "fr", "spec": "arm:restarts:nulls,...", "name": str}   (hyperparameter values cap at 2,500 chars)
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def jobs(kind, spec):
    """[arm, half, restarts, seed, perm] per job: the real fit both halves + label-permuted nulls (as hill.py / fr_hill.py run)."""
    out = []
    for s in spec.split(","):
        arm, r, nl = s.split(":")
        r, nl = int(r), int(nl)
        out += [[arm, h, r, 0, None] for h in (0, 1)]
        if kind == "iu":
            out += [[arm, h, max(3, r // 5), 100 + k, 7 + 13 * k] for k in range(nl) for h in (0, 1)]
        else:
            out += [[arm, h, 1, 50 + k, 11 + 17 * k] for k in range(nl) for h in (0, 1)]
    return out


def main(train_dir, test_dir, model_dir, out_dir, params):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    os.chdir(train_dir)
    t0 = time.time()
    H = __import__("hill" if params["kind"] == "iu" else "fr_hill")
    js = [tuple(j) for j in (params["jobs"] if "jobs" in params else jobs(params["kind"], params["spec"]))]
    with Pool(os.cpu_count()) as pool:
        res = pool.map(H.search, js, chunksize=1)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / f"{params['name']}.json").write_text(json.dumps(dict(res=res, runtime_s=time.time() - t0), default=float))
    print(f"done {len(res)} jobs in {time.time() - t0:.0f}s", flush=True)
