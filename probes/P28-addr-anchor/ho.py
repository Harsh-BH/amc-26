"""P28: the address-anchor rule on #8c's dense holdout (P23's dump of run.py --split holdout: every scored pair of the SX
a fold-0 S1 competes for, final p after CE + US stacker, truth). Per variant: rule pairs, their precision, the pairs
the rule flips to accepted, and paired dF0.5 per S1 vs #8c at t 0.71 (half 1 = report, half 0 = selection).
    .venv/bin/python probes/P28-addr-anchor/ho.py US India      # aggregates only -> ledger
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P28-addr-anchor"), str(ROOT)]
from anchor import VARIANTS, boosted, decide, rule_masks  # noqa: E402
from metric import f_beta  # noqa: E402

W = ROOT / "data/work"
T = 0.71                                                     # #8c's US / India t


def half(ids) -> np.ndarray:                                 # run.half: P5's tune (0) / report (1) halves
    return (pd.util.hash_pandas_object(pd.Series(list(ids)), index=False).to_numpy() >> 7) % 2


def per_s1(d: pd.DataFrame, pred: np.ndarray, truth: dict) -> pd.Series:
    pr = d[pred & d.holdout.to_numpy()].groupby("s1").sx.agg(set).to_dict()
    return pd.Series({s: f_beta(pr.get(s, set()), v) for s, v in truth.items()})


def run(c: str) -> dict:
    t0 = time.time()
    s1 = pd.read_parquet(W / "worlds/holdout/s1.parquet", columns=["entity_id", "country"])
    ids = s1.entity_id[s1.country == c].tolist()
    tr = pd.read_parquet(W / "worlds/holdout/truth.parquet")
    truth = {s: set() for s in ids}
    for a, b in zip(tr.s1, tr.sx):
        if a in truth:
            truth[a].add(b)
    del s1, tr
    d = pd.read_parquet(W / f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "pred"])
    p = d.p.to_numpy(np.float64)
    base = decide(d.sx, p, T)
    assert (base == d.pred.to_numpy()).all(), "the decision reproduces #8c's"
    f0 = per_s1(d, base, truth)
    hv = pd.Series(half(f0.index), index=f0.index)
    m = rule_masks(d[["s1", "sx", "p"]], "holdout", c, T)
    ho, y = d.holdout.to_numpy(), d.y.to_numpy()
    res = {"base_h1": round(f0[hv == 1].mean(), 5), "base_all": round(f0.mean(), 5), **m["info"]}
    print(f"{c}: {len(d):,} pairs, base F0.5 half1 {res['base_h1']:.5f} (all {res['base_all']:.5f}); {m['info']}")
    for v in [*VARIANTS, "c_strict"]:
        mask = m[v] if v != "c_strict" else m["c"] & decide(d.sx, p, -1.0)       # argmax pairs only
        q = boosted(p, mask)
        pred = decide(d.sx, q, T)
        f1 = per_s1(d, pred, truth)
        dl = f1 - f0
        r = {}
        for h in (0, 1):
            x = dl[hv == h]
            r[f"d_h{h}"], r[f"se_h{h}"] = x.mean(), x.std() / np.sqrt(len(x))
        rh, flip, lost = mask & ho, pred & ~base & ho, base & ~pred & ho
        r |= {"rule_pairs": int(rh.sum()), "rule_prec": y[rh].mean() if rh.any() else np.nan,
              "flip": int(flip.sum()), "flip_prec": y[flip].mean() if flip.any() else np.nan, "lost": int(lost.sum()),
              "lost_prec": y[lost].mean() if lost.any() else np.nan, "s1_up": int((dl > 0).sum()),
              "s1_down": int((dl < 0).sum())}
        for k in ("sub1", "add1", "drop1"):
            kf = flip & (m["kind"] == k)
            r[f"flip_{k}"], r[f"flip_prec_{k}"] = int(kf.sum()), (y[kf].mean() if kf.any() else np.nan)
        print(f"  {v:8s} dF half1 {r['d_h1']:+.5f} ± {r['se_h1']:.5f} (half0 {r['d_h0']:+.5f} ± {r['se_h0']:.5f}) | "
              f"rule pairs {r['rule_pairs']:,} prec {r['rule_prec']:.3f} | flipped {r['flip']:,} prec {r['flip_prec']:.3f} "
              f"(sub1 {r['flip_sub1']} {r['flip_prec_sub1']:.3f}, add1 {r['flip_add1']} {r['flip_prec_add1']:.3f}, "
              f"drop1 {r['flip_drop1']} {r['flip_prec_drop1']:.3f}) | lost {r['lost']} | S1 up {r['s1_up']} down "
              f"{r['s1_down']}", flush=True)
        res |= {f"{v}_{k}": (round(float(x), 5) if isinstance(x, float) else x) for k, x in r.items()}
    res["runtime_s"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    import resource
    from ledger import log
    for c in sys.argv[1:] or ["US", "India"]:
        r = run(c)
        gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        best = max(VARIANTS, key=lambda v: r[f"{v}_d_h0"])
        log("exp", probe="P28-addr-anchor", subproblem="decide", name=f"addr-anchor-holdout-{c}",
            world="holdout-dense-half1 (#8c p, P23 dump)", metrics=r,
            config={"t": T, "boost": 0.99, "variants": "a=1word+hn+city, b=a+street fuzzy85, bx=a+street exact, "
                    "c=b+chain guard, c_strict=c on argmax pairs only", "selected_on_half0": best},
            hypothesis="one core word differs + same hn/street/city => same business; accept regardless of t",
            verdict="keep" if r[f"{best}_d_h1"] > 2 * r[f"{best}_se_h1"] else
                    ("drop" if r[f"{best}_d_h1"] < 0 else "inconclusive"),
            runtime_s=r["runtime_s"], peak_gb=round(gb, 2))
        print(f"{c}: selected on half 0 = {best}; peak {gb:.2f} GB", flush=True)
