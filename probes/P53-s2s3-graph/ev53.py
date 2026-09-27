"""P53: triangle columns on top of P44's sub44r combiner (FINAL + l_r_all, 5-seed bag, t 0.71), holdout half-split.

Each arm = the same bag with extra columns. Reported as per-S1 F0.5 deltas vs the sub44r bag refit on the same half
(paired SE), both directions; cross-fit = mean of the two report halves.
    py.sh ev53.py [arm ...]        -> data/work/p53/ev53.json (appends)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["P44_VARIANT"] = "r"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "P44-meta-stack")]
import ev as E  # noqa: E402
import meta as MT  # noqa: E402
from best import bag  # noqa: E402
import te44  # noqa: E402
from tri import FEATS, OUT  # noqa: E402

OWN = ["t_max", "t_max_n", "t_max_a", "t_ncs", "t_best", "t_hq"]
COMP = ["c_max", "c_pmax", "c_ncs", "d_max"]
ARMS = {"tri_all": OWN + COMP, "tri_own": OWN, "tri_comp": COMP, "tri_core": ["t_max", "t_best", "c_pmax", "d_max"]}


def prep():
    D = MT.prep(te44.MODELS)
    for c in E.C:
        A = D[c][0]
        t = pd.read_parquet(OUT / f"tri_train_{c}.parquet").drop_duplicates(["s1", "sx"])
        m = A[["s1", "sx"]].merge(t, on=["s1", "sx"], how="left")
        assert len(m) == len(A) and m.t_ncs.notna().all(), "every argmax pair has a triangle row"
        for k in FEATS:
            A[k] = m[k].to_numpy()
        A["d_max"] = A.t_max.fillna(0).to_numpy() - A.c_max.fillna(0).to_numpy()
    return D


def f_of(D, s):
    return {c: E.per_s1(D[c][0], D[c][1], s[c] >= E.T) for c in E.C}


def main(arms):
    D = prep()
    res = []
    base = {}
    for fit in (0, 1):
        t0 = time.time()
        tr = {c: D[c][0].half.to_numpy() == fit for c in E.C}
        pb = bag(te44.FINAL)(tr, D)
        if fit == 0:
            for c in E.C:
                ref = np.load(E.OUT / f"ho_p44r_{c}.npy")
                assert np.allclose(pb[c], ref, atol=1e-6), f"{c}: sub44r bag does not reproduce"
        base[fit] = f_of(D, pb)
        print(f"base h{fit} {time.time() - t0:.0f}s", flush=True)
    for name in arms:
        cols = te44.FINAL + ARMS[name]
        row = {"arm": name, "cols": ARMS[name]}
        for fit in (0, 1):
            t0 = time.time()
            rep = 1 - fit
            tr = {c: D[c][0].half.to_numpy() == fit for c in E.C}
            fn = f_of(D, bag(cols)(tr, D))
            r = {}
            for c in E.C:
                S = D[c][1]
                x = (fn[c] - base[fit][c])[S.half.to_numpy() == rep]
                r[c] = (x.mean(), x.std(ddof=1) / np.sqrt(len(x)), int((x > 0).sum()), int((x < 0).sum()))
                np.save(OUT / f"d_{name}_{c}_h{fit}.npy", (fn[c] - base[fit][c]).astype(np.float32))
            lb, se = E.lbeq(r)
            row[f"h{fit}"] = {"lbeq": lb, "se": se, "z": lb / se, **{f"d_{c}": r[c][0] for c in E.C},
                              **{f"se_{c}": r[c][1] for c in E.C}, **{f"updown_{c}": r[c][2:] for c in E.C},
                              "s": round(time.time() - t0)}
            print(f"{name:10s} h{fit}->h{rep}: LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:+.1f} SE) | "
                  + " | ".join(f"{c} {r[c][0]:+.6f} ± {r[c][1]:.6f} (+{r[c][2]}/-{r[c][3]})" for c in E.C), flush=True)
        a, b = row["h0"], row["h1"]
        row["xfit"] = {"lbeq": (a["lbeq"] + b["lbeq"]) / 2, "se": float(np.hypot(a["se"], b["se"]) / 2)}
        row["xfit"]["z"] = row["xfit"]["lbeq"] / row["xfit"]["se"]
        row["pass"] = bool(row["xfit"]["lbeq"] >= 1e-4 and row["xfit"]["z"] >= 3 and min(a["lbeq"], b["lbeq"]) > 0)
        print(f"{name:10s} cross-fit {row['xfit']['lbeq']:+.6f} ± {row['xfit']['se']:.6f} ({row['xfit']['z']:+.1f} SE)"
              f" -> {'PASS' if row['pass'] else 'fail'}", flush=True)
        res.append(row)
        p = OUT / "ev53.json"
        old = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(old + [row], default=float, indent=0))


if __name__ == "__main__":
    main(sys.argv[1:] or ["tri_all"])
