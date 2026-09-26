"""P44: ledger rows from a meta/jump JSON (one row per arm: both directions)."""
import json, sys
sys.path.insert(0, ".")
from ledger import log

def arms(path, arms_cfg, world="holdout (#34 oof, halves)", sub="stacking"):
    R = json.load(open(path)); by = {}
    for r in R: by.setdefault(r["arm"], {})[r["fit"]] = r
    for a, (cfg, hyp) in arms_cfg.items():
        x, y = by[a][0], by[a][1]
        ok = x["lbeq"] >= 0.00015 and x["z"] >= 3 and y["lbeq"] >= 0.00015 and y["z"] >= 3 and min(x["d_India"], x["d_US"]) >= -0.00005
        log("exp", probe="P44-meta-stack", subproblem=sub, name=a, world=world,
            metrics={"lbeq_h0h1": round(x["lbeq"], 6), "se_h0h1": round(x["se"], 6), "d_India_h0h1": round(x["d_India"], 6),
                     "d_US_h0h1": round(x["d_US"], 6), "lbeq_h1h0": round(y["lbeq"], 6), "se_h1h0": round(y["se"], 6),
                     "d_India_h1h0": round(y["d_India"], 6), "d_US_h1h0": round(y["d_US"], 6), "t_h0": x["t"], "t_h1": y["t"]},
            config=cfg, hypothesis=hyp, verdict="keep" if ok else "drop", runtime_s=x.get("s", 0), peak_gb=1.5)
