"""P56 arm 2b: France group-state bars for the additive stages on the same LOCO analogue as arm2.py. The gc and hn-extra
bars depend on the S1's current group size m (members accepted before the stage): m = 1 vs m >= 2 (France bars x1 / x2,
density-mapped like every recipe bar). Other stages at the recipe.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/arm2b.py"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm2 as A2  # noqa: E402
S, p47 = A2.S, A2.p47


def run(fr, stage, x1, x2):
    tm, s1, is2, p, n = fr["tm"], fr["s1"], fr["is2"], fr["p"], fr["n"]
    mp = lambda x: np.inf if x is None else tm[str(x)]
    am, sib, hn_eq = np.ones(len(s1), bool), fr["hn_rel"] == 2, fr["hn_eq"]
    pred, free = S.base(s1, is2, p, am, sib, mp(A2.REC["t"]), p47.CAP)
    for st in ("rescue", "gc", "hn"):
        bar = mp(A2.REC["r" if st == "rescue" else st])
        if st == stage:
            n2, n3 = S.counts(s1, is2, pred, n)
            m = (n2 + n3)[s1]
            bar = np.where(m >= 2, mp(x2), mp(x1))
        pred = S.add(st, s1, is2, p, pred, free, hn_eq, n, bar, p47.CAP)
    f = S.fbeta(np.bincount(s1[pred & fr["y"]], minlength=n), np.bincount(s1[pred], minlength=n), fr["ntrue"])
    d = f - fr["f"]
    return {"dF": float(d.mean()), "se": float(d.std() / np.sqrt(n)), "h0": float(d[fr["hv"] == 0].mean()),
            "h1": float(d[fr["hv"] == 1].mean()), "d_pairs": int((pred != fr["pred"]).sum())}


if __name__ == "__main__":
    from ledger import log
    t0 = time.time()
    FR = {(an, dr): p47.loco_frame(an, dr) for an in ("chain", "m2") for dr in ("US_India", "India_US")}
    for k, fr in FR.items():
        assert run(fr, "gc", 0.7, 0.7)["d_pairs"] == 0 and run(fr, "hn", 0.8, 0.8)["d_pairs"] == 0, k
    res = []
    for stage in ("gc", "hn"):
        for x1, x2 in itertools.product(A2.LEV[stage], A2.LEV[stage]):
            if x1 != x2:
                res.append({"stage": stage, "x1": x1, "x2": x2, **{f"{a}|{d}": run(fr, stage, x1, x2) for (a, d), fr in FR.items()}})
    json.dump(res, open(Path(__file__).parent / "arm2b.json", "w"), indent=0)
    ok = [r for r in res if r["chain|US_India"]["h0"] >= 0 and r["chain|India_US"]["h0"] > 0]
    ok.sort(key=lambda r: r["chain|India_US"]["h0"], reverse=True)
    print("cells", len(res), "h0-admissible", len(ok), f"{time.time() - t0:.0f}s")
    for r in ok[:10]:
        print(r["stage"], r["x1"], r["x2"], " ".join(f"{k}:{v['dF']*1e4:+.2f}(h1 {v['h1']*1e4:+.2f}, {v['d_pairs']})"
                                                     for k, v in r.items() if "|" in k), flush=True)
    for stage in ("gc", "hn"):
        best = max((r for r in res if r["stage"] == stage), key=lambda r: min(r["chain|US_India"]["dF"], r["chain|India_US"]["dF"]))
        print("maxmin", stage, best["x1"], best["x2"], " ".join(f"{k}:{v['dF']*1e4:+.2f}" for k, v in best.items() if "|" in k))
    best = ok[0] if ok else max(res, key=lambda r: min(r["chain|US_India"]["dF"], r["chain|India_US"]["dF"]))
    a, b = best["chain|US_India"], best["chain|India_US"]
    gate = a["dF"] >= 0 and b["dF"] >= 3e-4 and a["h1"] >= 0 and b["h1"] >= 3e-4
    log("exp", probe="P56-thresh-grain", subproblem="france-decision", name="arm2b-fr-group-state-bars", world="loco",
        metrics={"cells": len(res), "h0_admissible": len(ok), "best": [best["stage"], best["x1"], best["x2"]],
                 **{f"{k}:{m}": round(v[m], 6) for k, v in best.items() if "|" in k for m in ("dF", "se", "h0", "h1")}},
        config={"recipe": A2.REC, "state": "group size before the stage: 1 vs >= 2", "select": "h0 chain both ways"},
        hypothesis="gc / hn-extra bars by France group size (1 vs 2+) beat the pooled recipe bars on LOCO",
        verdict="keep" if gate else "drop", runtime_s=round(time.time() - t0, 1), peak_gb=round(p47.peak(), 2))
    print("gate", gate)
