"""P71 B: France hn-shift drop on P42's LOCO frames (source-only model scored on the unseen target), P64's harness.
Rule: drop an accepted link whose house number differs by >= 3 (hn_rel 2 = SIB3 offset, 5 = other; +-1/2 kept) when the
same S1 holds an accepted link with the identical house number (hn_rel 1).  -> probes/P71-fr-noiseword/work/b71.json
    .venv/bin/python probes/P71-fr-noiseword/b71.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import solution_p42 as J  # noqa: E402

W = ROOT / "data/work"
RECIPES = {"gc70": dict(t=0.9868, cap=(5, 6), r=0.9, gc=0.7, hn=None)}


def drop_mask(s1, pred, cls, n, classes):
    has_eq = np.bincount(s1[pred & (cls == 1)], minlength=n) > 0
    return pred & np.isin(cls, classes) & has_eq[s1]


def main():
    G = json.loads((W / "p42/sm/grid.json").read_text())
    res = {}
    for src, tgt in (("India", "US"), ("US", "India")):
        dr = f"{src}_{tgt}"
        z = dict(np.load(W / f"p42/loco_{dr}.npz"))
        s1, n, y, cls = z["s1"].astype(np.int64), len(z["ntrue"]), z["y"], z["hn_rel"]
        nt = z["ntrue"].astype(np.int64)
        am = np.ones(len(s1), bool)
        fb = lambda pred: J.fbeta(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), nt)
        for an in ("chain", "m2"):
            G2 = RECIPES["gc70"]
            p = J.loco_p(z["g"], z["r"], 0.1, an)
            tm = J.fr_to_loco(G, an, dr, 0.1, p, am, n)
            mp = lambda x: None if x is None else tm[str(x)]
            pred0 = J.decide(s1, z["is2"], p, am, cls == 2, cls == 1, n, mp(G2["t"]), G2["cap"], mp(G2["r"]), mp(G2["gc"]), mp(G2["hn"]))
            f0 = fb(pred0)
            has = pred0 & (cls != 0)
            res[f"{dr}|{an}|base"] = {"F0": round(float(f0.mean()), 6), "npred": int(pred0.sum()),
                                      "hn_cls_share": [round(float((pred0 & (cls == i)).sum() / max(has.sum(), 1)), 4) for i in range(6)],
                                      "hn_cls_true": [round(float(y[pred0 & (cls == i)].mean()), 4) if (pred0 & (cls == i)).any() else None for i in range(6)]}
            print(dr, an, "base", res[f"{dr}|{an}|base"], flush=True)
            for vn, classes in (("drop25", (2, 5)), ("drop2", (2,)), ("drop5", (5,))):
                rm = drop_mask(s1, pred0, cls, n, classes)
                d = fb(pred0 & ~rm) - f0
                res[f"{dr}|{an}|{vn}"] = {"dF": round(float(d.mean()), 6), "se": round(float(d.std() / np.sqrt(n)), 6),
                                          "dF_h0": round(float(d[z["hv"] == 0].mean()), 6), "dF_h1": round(float(d[z["hv"] == 1].mean()), 6),
                                          "removed": int(rm.sum()), "removed_per_s1": round(float(rm.sum() / n), 5),
                                          "removed_true": round(float(y[rm].mean()), 4) if rm.any() else None}
                print(" ", dr, an, vn, res[f"{dr}|{an}|{vn}"], flush=True)
    (HERE / "work/b71.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
