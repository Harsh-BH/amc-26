"""Label-free profile of France test's confident pseudo-labels (#9ab France p = IDF-fixed m2_big + bge blend), by
house-number class, next to the same numbers for US / India test (#8c p). Which LOCO direction does France resemble?
    .venv/bin/python probes/P27-france-pseudo/fr_profile.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from stack import hn_cls  # noqa: E402

W = ROOT / "data/work"
FILES = {"France": W / "p19/prob_test_France_idf144_bge.parquet",
         "US": W / "p17/prob_test_US_bge.parquet", "India": W / "p13/prob_test_India_blend.parquet"}
SIB3 = np.array([3, 4, 5, 7, 9, 11, 13, 21])


def main():
    out = {}
    for c, f in FILES.items():
        d = pd.read_parquet(f, columns=["s1", "sx", "p"])
        o = np.lexsort((-d.p.to_numpy(), d.sx.to_numpy()))
        first = np.r_[True, d.sx.to_numpy()[o][1:] != d.sx.to_numpy()[o][:-1]]
        am = np.zeros(len(d), bool)
        am[o[first]] = True
        d = d[am & (d.p.to_numpy() >= 0.5)].reset_index(drop=True)
        h1 = pd.read_parquet(W / f"rf_test_s1_{c}.parquet", columns=["id", "hn"]).set_index("id").hn
        hx = pd.read_parquet(W / f"rf_test_sx_{c}.parquet", columns=["id", "hn"]).set_index("id").hn
        a = h1.reindex(d.s1.to_numpy()).fillna("").to_numpy(object)
        b = hx.reindex(d.sx.to_numpy()).fillna("").to_numpy(object)
        del h1, hx
        cls = hn_cls(a, b)
        num = lambda v: pd.to_numeric(pd.Series(v).replace("", "-1"), errors="coerce").fillna(-1).to_numpy(np.int64)
        na, nb = num(a), num(b)
        sib3 = (na >= 0) & (nb >= 0) & np.isin(nb - na, SIB3)
        p = d.p.to_numpy()
        r = {}
        for t in (0.9, 0.98, 0.995):
            m = p >= t
            r[f"t{t}"] = {"n": int(m.sum()), "share_cls0": round(float((cls[m] == 0).mean()), 4),
                          "share_sib": round(float((cls[m] == 1).mean()), 4),
                          "per1k_sib3": round(float(sib3[m].mean() * 1000), 2),
                          "share_digit": round(float((cls[m] == 2).mean()), 4), "share_far": round(float((cls[m] == 4).mean()), 4),
                          "hn_both": round(float(((na >= 0) & (nb >= 0))[m].mean()), 4)}
        out[c] = r
        print(c, json.dumps(r), flush=True)
    (W / "p27/fr_profile.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
