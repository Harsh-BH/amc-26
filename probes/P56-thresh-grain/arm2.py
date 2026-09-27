"""P56 arm 2: France per-source bars (S2 vs S3) for the base t and the rescue / gc / hn-extra stages, on the LOCO analogue
of the sub43_r90 recipe (P47 loco_frame: P42 frames, veto 0.1, caps 5/6, t .99 / rescue .9 / gc .7 / hn .8).

A France bar x maps to LOCO through P42's pooled density map (tm[x]); a per-source cell applies tm[x2] to S2 rows and tm[x3]
to S3 rows (None = the stage is off for that source). One stage at a time moves from the recipe; every other stage stays.
solution_p42.decide broadcasts per-row bar arrays, so the recipe code runs unchanged.
dF = paired mean per-S1 F0.5 vs the recipe over all fold-0 target S1 (h0 / h1 = P42's halves), both analogues, both ways.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/arm2.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P47-fr-shape-match"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import p47  # noqa: E402
import solution_p42 as S  # noqa: E402

REC = {"t": 0.99, "r": 0.9, "gc": 0.7, "hn": 0.8}
LEV = {"t": [0.97, 0.98, 0.983, 0.985, 0.9868, 0.988, 0.99, 0.9925],
       "r": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5],
       "gc": [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
       "hn": [None, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5]}


def bars(fr, cell):
    """cell = {stage: (x2, x3)} France bars -> per-row LOCO bar arrays (inf where a source's stage is off)."""
    tm, is2 = fr["tm"], fr["is2"]
    mp = lambda x: np.inf if x is None else tm[str(x)]
    out = {}
    for st in REC:
        x2, x3 = cell.get(st, (REC[st], REC[st]))
        out[st] = np.where(is2, mp(x2), mp(x3))
    return out


def run(fr, cell):
    b = bars(fr, cell)
    s1, n = fr["s1"], fr["n"]
    am = np.ones(len(s1), bool)
    pred = S.decide(s1, fr["is2"], fr["p"], am, fr["hn_rel"] == 2, fr["hn_eq"], n, b["t"], p47.CAP, b["r"], b["gc"], b["hn"])
    f = S.fbeta(np.bincount(s1[pred & fr["y"]], minlength=n), np.bincount(s1[pred], minlength=n), fr["ntrue"])
    d = f - fr["f"]
    hv = fr["hv"]
    return {"dF": float(d.mean()), "se": float(d.std() / np.sqrt(n)), "h0": float(d[hv == 0].mean()),
            "h1": float(d[hv == 1].mean()), "d_pairs": int((pred != fr["pred"]).sum())}


def cells():
    out = []
    for st, vals in LEV.items():
        for x2, x3 in itertools.product(vals, vals):
            if x2 != x3:                                             # x2 == x3: pooled bars (P42 already swept them)
                out.append({st: (x2, x3)})
    return out


if __name__ == "__main__":
    from ledger import log
    t0 = time.time()
    FR = {(an, dr): p47.loco_frame(an, dr) for an in ("chain", "m2") for dr in ("US_India", "India_US")}
    for k, fr in FR.items():
        assert run(fr, {})["d_pairs"] == 0, k                       # the recipe reproduces P47's pred
        print(k, "rows", len(fr["p"]), "S1", fr["n"], "pred", int(fr["pred"].sum()), flush=True)
    res = []
    for cell in cells():
        r = {"cell": {k: list(v) for k, v in cell.items()}}
        for (an, dr), fr in FR.items():
            r[f"{an}|{dr}"] = run(fr, cell)
        res.append(r)
    json.dump(res, open(HERE / "arm2.json", "w"), indent=0)
    # selection on h0 (chain, both ways >= 0, max India->US), report h1 and the full fold
    key = lambda r: (min(r["chain|US_India"]["h0"], 0) * 10 + r["chain|India_US"]["h0"])
    ok = [r for r in res if r["chain|US_India"]["h0"] >= 0 and r["chain|India_US"]["h0"] > 0]
    ok.sort(key=key, reverse=True)
    print("cells", len(res), "h0-admissible", len(ok), f"{time.time() - t0:.0f}s", flush=True)
    for r in ok[:15]:
        print(json.dumps(r["cell"]), {k: {m: round(v[m], 6) for m in ("dF", "se", "h0", "h1", "d_pairs")}
                                      for k, v in r.items() if k != "cell"}, flush=True)
    best = ok[0] if ok else max(res, key=key)
    a, b = best["chain|US_India"], best["chain|India_US"]
    gate = a["dF"] >= 0 and b["dF"] >= 3e-4 and a["h1"] >= 0 and b["h1"] >= 3e-4
    log("exp", probe="P56-thresh-grain", subproblem="france-decision", name="arm2-fr-per-source-bars", world="loco",
        metrics={"cells": len(res), "h0_admissible": len(ok), "best_cell": best["cell"],
                 **{f"{k}:{m}": round(v[m], 6) for k, v in best.items() if k != "cell" for m in ("dF", "se", "h0", "h1")}},
        config={"recipe": REC, "levers": LEV, "select": "h0 chain both ways >= 0, max India->US", "tau": 0.1, "cap": [5, 6]},
        hypothesis="per-source France bars (S2 vs S3) for base t / rescue / gc / hn beat the pooled recipe bars on LOCO",
        verdict="keep" if gate else "drop", runtime_s=round(time.time() - t0, 1), peak_gb=round(p47.peak(), 2))
    print("best", json.dumps(best["cell"]), "gate", gate)
