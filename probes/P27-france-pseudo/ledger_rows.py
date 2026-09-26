"""One ledger row per LOCO job pair: metrics = {dir>arm_{lf,oracle,src}}. Usage: ledger_rows.py NAME VERDICT JOB..."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ledger import log

name, verdict, *jobs = sys.argv[1:]
m, rt = {}, 0
for j in jobs:
    r = json.load(open(ROOT / f"data/work/p27/jobs/{j}/result.json"))
    rt += r.get("elapsed_s", 0)
    for d, v in r["dirs"].items():
        for arm, a in v["arms"].items():
            if "tgt_lf_empty" in a:
                m |= {f"{d}_{arm}_lf": a["tgt_lf_empty"], f"{d}_{arm}_oracle": a["tgt_oracle"], f"{d}_{arm}_src": a["src_f05"]}
log("exp", probe="P27-france-pseudo", subproblem="france-pseudo", name=name, world="holdout-dense-full LOCO (P6 protocol)",
    metrics=m, config={"jobs": jobs, "pool": "target fold 2 x0.5 (P6)", "src_train": "source fold 2 x0.5, ES on source fold 1 x0.2",
                       "lgbm": "P4 PARAMS, lr 0.1", "lf": "empty-rate match to the source's t*", "filters": "solution_p27.labels"},
    hypothesis="filtered / down-weighted / negative-only pseudo-labels on the unseen country gain >= +1 pt in BOTH LOCO directions",
    verdict=verdict, runtime_s=rt, peak_gb=15.0)
print("logged", len(m))
