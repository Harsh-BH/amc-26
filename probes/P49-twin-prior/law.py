"""P49: per-country, per-source cluster-size law f(n) from train truth (every train S1, zeros included), and the two
candidate group-size likelihood ratios for handing one more record to an S1 that already holds k of that source:
  h(k) = f(k+1) / f(k)                 (the brief's formula)
  r(k) = f(k+1) (k+1) / f(k)           (exchangeable-records posterior: P(z) ∝ Π f(n_i) n_i!; Poisson -> constant)
    probes/P49-twin-prior/py.sh probes/P49-twin-prior/law.py      # -> data/work/p49/law.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[2] / "data/work"
NMAX = 8


def law() -> dict:
    gt = pd.read_parquet(W / "gt_pairs.parquet")
    gt["src"] = gt.sx.str[:2]
    n = gt.groupby(["s1", "src"]).size().unstack(fill_value=0)
    s1 = pd.read_parquet(W / "s1_folds.parquet")
    n = n.reindex(s1.entity_id, fill_value=0)
    out = {}
    for c in ("India", "US"):
        m = (s1.country == c).to_numpy()
        for s in ("S2", "S3"):
            f = np.bincount(n[s].to_numpy()[m], minlength=NMAX + 2)[:NMAX + 2] / m.sum()
            out[f"{c}|{s}"] = f.tolist()
    return out


def ratios(f, eps=1e-6):
    f = np.asarray(f) + eps
    k = np.arange(len(f) - 1)
    return f[1:] / f[:-1], f[1:] * (k + 1) / f[:-1]


if __name__ == "__main__":
    L = law()
    (W / "p49").mkdir(exist_ok=True)
    json.dump(L, open(W / "p49/law.json", "w"))
    for k, f in L.items():
        h, r = ratios(f)
        mu = sum(i * v for i, v in enumerate(f))
        var = sum(i * i * v for i, v in enumerate(f)) - mu ** 2
        print(k, "f", np.round(f[:8], 4).tolist(), f"mean {mu:.3f} var {var:.3f}")
        print("   h", np.round(h[:7], 2).tolist(), "\n   r", np.round(r[:7], 2).tolist())
    # Poisson check: r is constant = lambda
    from math import exp, factorial
    fp = [exp(-1.6) * 1.6 ** i / factorial(i) for i in range(10)]
    assert np.allclose(ratios(fp, 0)[1], 1.6)
