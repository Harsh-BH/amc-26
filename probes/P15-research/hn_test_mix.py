"""P15 label-free: house-number delta classes of the pairs the shipped rule ACCEPTS on test, per country.
If France accepts a larger share of '+SIB' deltas (the orphan-sibling offsets) than US/India, France over-matches
siblings, and a signed-delta feature has more to fix there. Aggregates only.
    .venv/bin/python probes/P15-research/hn_test_mix.py France [US India]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hn_check import hn_cls  # noqa: E402

W = ROOT.parents[1] / "data/work"
T = {"France": 0.95, "India": 0.71, "US": 0.71}      # D4 France t; #6 India/US t (on blended p for India/US)
NAMES = {0: "eq/none", 1: "+SIB", 2: "digit drop/add", 3: "small other", 4: "far"}


def hn_map(path, ids):
    rf = pq.read_table(path, columns=["id", "hn"])
    pos = pc.index_in(pa.array(ids), value_set=rf.column("id")).to_numpy(zero_copy_only=False)
    return rf.column("hn").to_numpy(zero_copy_only=False)[pos.astype(np.int64)]


def main(c):
    f = W / ("p10/prob_test_France.parquet" if c == "France" else f"p11/prob_test_{c}_blend.parquet")
    d = pd.read_parquet(f, columns=["s1", "sx", "p"])
    am = d.p.to_numpy() == d.groupby("sx").p.transform("max").to_numpy()
    d = d[am & (d.p.to_numpy() >= T[c])].reset_index(drop=True)
    h1 = hn_map(W / f"rf_test_s1_{c}.parquet", d.s1.to_numpy())
    hx = hn_map(W / f"rf_test_sx_{c}.parquet", d.sx.to_numpy())
    cls = pd.Series(hn_cls(h1, hx)).map(NAMES)
    both = (h1 != "") & (hx != "")
    diff = both & (h1 != hx)
    sib = cls == "+SIB"
    dv = pd.Series(hx[sib.to_numpy()]).astype(np.int64) - pd.Series(h1[sib.to_numpy()]).astype(np.int64)
    print(f"{c}: accepted {len(d):,} pairs at t {T[c]}; both hn {both.mean():.3f}; hn differs {diff.mean():.4f}; "
          f"class share of accepted: {cls[diff].value_counts(normalize=False).to_dict()}; "
          f"+SIB per 1k accepted {1000 * sib.mean():.2f}; +SIB d>=3 per 1k {1000 * (dv >= 3).sum() / len(d):.2f}")
    return {"accepted": len(d), "hn_both": round(float(both.mean()), 4), "hn_diff": round(float(diff.mean()), 4),
            "sib_per_1k": round(1000 * float(sib.mean()), 3), "sib_ge3_per_1k": round(1000 * (dv >= 3).sum() / len(d), 3),
            "drop_per_1k": round(1000 * float((cls == "digit drop/add").mean()), 3)}


if __name__ == "__main__":
    out = {c: main(c) for c in (sys.argv[1:] or ["France"])}
    sys.path.insert(0, str(ROOT.parents[1]))
    from ledger import log
    log("exp", probe="P15-research", subproblem="features", name="hn-delta-class-mix-test-accepted", world="test",
        metrics={f"{c}_{k}": v for c, m in out.items() for k, v in m.items()},
        config={"thresholds": T, "p": "France m2_big (p10 pruned), India/US #6 blend"},
        hypothesis="France over-matches orphan siblings (hn + SIB offsets) more than US/India",
        verdict="inconclusive", runtime_s=0, peak_gb=0)
