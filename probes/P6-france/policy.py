"""Threshold policies for an unseen country, read off the saved dense LOCO curves (data/work/p6/curve_base_*.csv):
F0.5 on the unseen target at the source optimum, fixed thresholds, and the oracle; and the in-country cost of the
same thresholds (pooled model). The profile-matched policy is in the loco-dense-base ledger row."""
from pathlib import Path
import pandas as pd

W = Path(__file__).resolve().parents[2] / "data/work/p6"
rows = []
for src, tgt in (("US", "India"), ("India", "US"), ("US", "US"), ("India", "India"), ("pooled", "US"), ("pooled", "India")):
    c = pd.read_csv(W / f"curve_base_{src}_{tgt}.csv").set_index("t").f05
    r = {"src>tgt": f"{src}>{tgt}", "oracle": c.max(), "t_oracle": c.idxmax()}
    r |= {f"t{t:.2f}": c.loc[t] for t in (0.5, 0.72, 0.8, 0.85, 0.9, 0.95)}
    rows.append(r)
print(pd.DataFrame(rows).set_index("src>tgt").round(4).to_string())


def empty_rate_policy():
    """Alternative label-free policy: the unseen country's t where its predicted empty-S1 rate equals the source's at
    the source optimum (ho_profile S files)."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import loco_dense as LD
    T4, M = LD.T4, LD.M
    tn, grp = T4.truth_all(0)
    ts = np.round(np.arange(0.5, 0.99, 0.01), 2)
    for src, tgt, t_src in (("US", "India", 0.72), ("India", "US", 0.72)):
        S = pd.read_parquet(W / f"S_holdout_{src}.parquet")
        out = {}
        for c in (src, tgt):
            ci = T4.COUNTRIES.index(c)
            d = S[(S.country == ci).to_numpy() & S.am_all.to_numpy()]
            t_ = tn[(grp == c).to_numpy()]
            k = pd.Index(t_.index).get_indexer(d.s1)
            emp = [(np.bincount(k[d.p.to_numpy() >= t], minlength=len(t_)) == 0).mean() for t in ts]
            out[c] = pd.Series(emp, index=ts)
        target_empty = out[src].loc[t_src]
        t_e = float((out[tgt] - target_empty).abs().idxmin())
        cur = pd.read_csv(W / f"curve_base_{src}_{tgt}.csv").set_index("t").f05
        print(f"{src}>{tgt}: src empty {target_empty:.4f} at t {t_src} -> tgt t {t_e} F0.5 {cur.loc[t_e]:.4f} "
              f"(oracle {cur.max():.4f}); tgt empty at src t {out[tgt].loc[t_src]:.4f}")


if __name__ == "__main__":
    empty_rate_policy()
