"""P64 France gate: legal-form conflict veto on P42's LOCO frames (source-only model scored on the unseen target country),
France recipe gc70 (#37 / sub_iu54_frgc70), chain and m2 analogues, thresholds density-mapped as P42/P63 do.

    lf    SRC TGT   lf_cls of every LOCO row -> data/work/p64/lf_loco_<SRC>_<TGT>.npy
                    (rows = P27's S_holdout_<SRC> of TGT with am_all, P42 prep.loco's order; q1 = s1 - ci*1e7, qx = sx - ci*1e8)
    eval            veto variants vs the recipe -> probes/P64-legal-form-feat/fr64.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import attach as AT  # noqa: E402
import lf  # noqa: E402

W = ROOT / "data/work"
CI = {"India": 0, "US": 1}
OFF1, OFF2 = 10_000_000, 100_000_000
GC70 = dict(t=0.9868, cap=(5, 6), r=0.9, gc=0.7, hn=None)          # P63 RECIPES["gc70"]


def lf_loco(src, tgt):
    t0 = time.time()
    ci = CI[tgt]
    S = pd.read_parquet(W / f"p27/p6/S_holdout_{src}.parquet", columns=["s1", "sx", "country", "is_true", "am_all"])
    S = S[(S.country.to_numpy() == ci) & S.am_all.to_numpy()].reset_index(drop=True)
    z = np.load(W / f"p42/loco_{src}_{tgt}.npz")
    assert len(S) == len(z["y"]) and (S.is_true.to_numpy() == z["y"]).all(), "row order != P42 LOCO frame"
    n1, nx = AT.names("train", tgt, S.s1.to_numpy() - ci * OFF1, S.sx.to_numpy() - ci * OFF2)
    k = lf.pair_class(pd.Series(AT.sigs(n1)), pd.Series(AT.sigs(nx)))
    np.save(AT.OUT / f"lf_loco_{src}_{tgt}.npy", k)
    print(src, tgt, len(k), np.bincount(k, minlength=6).tolist(), f"{time.time() - t0:.0f}s", flush=True)


def evaluate():
    import solution_p42 as J
    G = json.loads((W / "p42/sm/grid.json").read_text())
    res = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        dr = f"{src}_{tgt}"
        z = dict(np.load(W / f"p42/loco_{dr}.npz"))
        k = np.load(AT.OUT / f"lf_loco_{dr}.npy")
        s1, n, y = z["s1"].astype(np.int64), len(z["ntrue"]), z["y"]
        nt = z["ntrue"].astype(np.int64)
        am = np.ones(len(s1), bool)
        fb = lambda pred: J.fbeta(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), nt)
        for an in ("chain", "m2"):
            p = J.loco_p(z["g"], z["r"], 0.1, an)
            tm = J.fr_to_loco(G, an, dr, 0.1, p, am, n)
            mp = lambda x: None if x is None else tm[str(x)]
            dec = lambda pp: J.decide(s1, z["is2"], pp, am, z["hn_rel"] == 2, z["hn_rel"] == 1, n, mp(GC70["t"]), GC70["cap"],
                                      mp(GC70["r"]), mp(GC70["gc"]), mp(GC70["hn"]))
            pred0 = dec(p)
            f0 = fb(pred0)
            res[f"{dr}|{an}|base"] = {"F0": round(float(f0.mean()), 6), "npred": int(pred0.sum()),
                                      "by_cls": {i: [int((pred0 & (k == i)).sum()), round(float(y[pred0 & (k == i)].mean()), 4)
                                                     if (pred0 & (k == i)).any() else None] for i in range(6)}}
            print(f"{dr} {an} base", res[f"{dr}|{an}|base"], flush=True)
            variants = {"veto_post_conf": pred0 & (k != 4),                        # drop accepted differ-form pairs
                        "veto_pre_conf": dec(np.where(k == 4, 0.0, p)),              # never let them in (chain re-decides)
                        "veto_post_conf_ov": pred0 & ~np.isin(k, (4, 5)),
                        "veto_post_sxonly": pred0 & (k != 3)}
            for vn, pred in variants.items():
                d = fb(pred) - f0
                ch = pred != pred0
                res[f"{dr}|{an}|{vn}"] = {"dF": round(float(d.mean()), 6), "se": round(float(d.std() / np.sqrt(n)), 6),
                                          "dF_h0": round(float(d[z["hv"] == 0].mean()), 6),
                                          "dF_h1": round(float(d[z["hv"] == 1].mean()), 6),
                                          "removed": int((pred0 & ~pred).sum()), "added": int((pred & ~pred0).sum()),
                                          "removed_true": round(float(y[pred0 & ~pred].mean()), 4) if (pred0 & ~pred).any() else None,
                                          "changed": int(ch.sum())}
                print(f"  {dr} {an} {vn}", res[f"{dr}|{an}|{vn}"], flush=True)
    (HERE / "fr64.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    lf_loco(*sys.argv[2:]) if sys.argv[1] == "lf" else evaluate()
