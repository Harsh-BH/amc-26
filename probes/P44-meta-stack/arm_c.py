"""P44 arm (c): the report's decision rules on top of #34 (p' = a10 out of fold, am = #8c's argmax, t 0.71).

Each rule raises / sets the threshold of a pair bucket:
  margin  SX top1 - top2 < 0.10 -> t >= 0.85        (top1 = p', top2 = the SX's runner-up #8c p, P34's p2)
  nohn    pair without a house number on both sides -> t 0.80
  nonuniq S1 name not unique -> t 0.78               (n_twins > 0: another candidate S1 of the SX has the exact name;
                                                      or freq > 1: the S1 name repeats in the country's S1 pool)
  sib3    |hn_x - hn_1| in SIB3 and p' < 0.88 -> reject
As written (their numbers), then each bucket's t tuned on one half and reported on the other.
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/arm_c.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev as E  # noqa: E402

GRID = np.round(np.arange(0.50, 0.991, 0.01), 2)


def name_freq(c: str, A: pd.DataFrame) -> np.ndarray:
    """How many S1 of the country's train pool share the argmax S1's normalised full name (label-free)."""
    import a10
    s1 = pd.read_parquet(E.W / "train_source1.parquet", columns=["entity_id", "business_name", "country"])
    s1 = s1[s1.country == c]
    k = s1.business_name.map(a10.full)
    cnt = k.map(k.value_counts())
    return pd.Series(cnt.to_numpy(), index=s1.entity_id.to_numpy()).reindex(A.s1).to_numpy()


def buckets(A: pd.DataFrame) -> dict:
    pa_, p2 = A.pa.to_numpy(), A.p2.to_numpy()
    return {"margin": (pa_ - p2) < 0.10, "margin8": (A.p.to_numpy() - p2) < 0.10,
            "nohn": A.hn_lev.isna().to_numpy(), "nohn_x": (A.hn_len1.to_numpy() > 0) & A.hn_lev.isna().to_numpy(),
            "nonuniq_tw": A.n_twins.to_numpy() > 0, "nonuniq_fq": A.freq.to_numpy() > 1,
            "sib3": A.hn_sib3.to_numpy() > 0}


RULES = {"margin": ("margin", 0.85, "max"), "margin8": ("margin8", 0.85, "max"), "nohn": ("nohn", 0.80, "set"),
         "nohn_x": ("nohn_x", 0.80, "set"), "nonuniq_tw": ("nonuniq_tw", 0.78, "set"),
         "nonuniq_fq": ("nonuniq_fq", 0.78, "set"), "sib3": ("sib3", 0.88, "max")}


def teff(A, bk, rules: dict) -> np.ndarray:
    """Per-pair threshold: 0.71, then each (bucket, t, how) in order: 'set' replaces, 'max' raises."""
    t = np.full(len(A), E.T)
    for b, (tb, how) in rules.items():
        m = bk[b]
        t[m] = np.maximum(t[m], tb) if how == "max" else tb
    return t


def run():
    t0 = time.time()
    D = {}
    for c in E.C:
        A, S = E.load(c)
        A["freq"] = name_freq(c, A)
        D[c] = (A, S, buckets(A))
        print(c, {b: round(float(m.mean()), 4) for b, m in D[c][2].items()}, flush=True)
    res = []

    def report(name, tfun):
        """tfun(c, A, S, bk, fit_half) -> per-pair threshold; reported on the other half."""
        for fit in (0, 1):
            rep = 1 - fit
            r = {}
            for c in E.C:
                A, S, bk = D[c]
                t = tfun(c, A, S, bk, fit)
                r[c] = E.delta(A, S, A.pa.to_numpy() >= t, rep)
            lb, se = E.lbeq(r)
            row = {"arm": name, "fit": fit, "rep": rep, "lbeq": lb, "se": se, "z": lb / se,
                   **{f"d_{c}": r[c][0] for c in E.C}, **{f"se_{c}": r[c][1] for c in E.C},
                   **{f"updown_{c}": r[c][2:] for c in E.C}}
            res.append(row)
            print(f"{name:28s} fit h{fit} -> h{rep}: LB-eq {lb:+.6f} ± {se:.6f} ({lb / se:+.1f} SE) | "
                  + " | ".join(f"{c} {r[c][0]:+.6f} ± {r[c][1]:.6f} (+{r[c][2]}/-{r[c][3]})" for c in E.C), flush=True)

    # as written
    for k, (b, tb, how) in RULES.items():
        report(f"asis_{k}", lambda c, A, S, bk, fit, b=b, tb=tb, how=how: teff(A, bk, {b: (tb, how)}))
    report("asis_all(margin,nohn,nonuniq_tw,sib3)", lambda c, A, S, bk, fit: teff(
        A, bk, {"nohn": (0.80, "set"), "nonuniq_tw": (0.78, "set"), "margin": (0.85, "max"), "sib3": (0.88, "max")}))
    report("asis_all(margin,nohn,nonuniq_fq,sib3)", lambda c, A, S, bk, fit: teff(
        A, bk, {"nohn": (0.80, "set"), "nonuniq_fq": (0.78, "set"), "margin": (0.85, "max"), "sib3": (0.88, "max")}))

    # each bucket's t tuned on the fit half (per country), everything else at 0.71
    def tuned(b):
        def f(c, A, S, bk, fit):
            m, half = bk[b], S.half.to_numpy() == fit
            best, bt = -1, E.T
            for tb in GRID:
                t = np.where(m, tb, E.T)
                v = E.per_s1(A, S, A.pa.to_numpy() >= t)[half].mean()
                if v > best + 1e-12:
                    best, bt = v, tb
            f.ts.setdefault(c, []).append(float(bt))
            return np.where(m, bt, E.T)
        f.ts = {}
        return f

    for b in ("margin", "nohn", "nonuniq_tw", "nonuniq_fq", "sib3"):
        tf = tuned(b)
        report(f"tuned_{b}", tf)
        res[-1]["t"], res[-2]["t"] = tf.ts, tf.ts
        print("   tuned t per country [fit h0, fit h1]:", tf.ts, flush=True)
    (E.OUT / "arm_c.json").write_text(json.dumps(res, default=float, indent=0))
    print(f"{time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    run()
