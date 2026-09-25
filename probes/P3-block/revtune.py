"""Full-density tuning of the reverse search (SX -> S1): only the TRUE children of holdout S1 are queried, against the
FULL train S1 pool of their country (the reverse rank of a true pair depends only on the S1 pool, so this is exact
for r@k at ~7% of a full pass). Texts come from full.py prep.

    .venv/bin/python probes/P3-block/revtune.py India '{"w0.6": {}, "w0.4": {"w_name": 0.4}}'
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import block as B
from full import FULL_CFG, OUT, TXT, WORK, peak_gb

KS = (1, 2, 3, 5, 10)


def kids(country):
    """Texts of the true children of this country's holdout S1, with their parent's pool row and P1 class."""
    cls = pd.read_parquet(OUT / "holdout_p1cls.parquet")
    ta_ids = pd.read_parquet(TXT / f"{country}_s1.parquet", columns=["entity_id"]).entity_id
    cls = cls[cls.s1.isin(set(ta_ids))]
    want = set(cls.sx)
    tb = pd.concat([d[d.entity_id.isin(want)] for p in sorted(TXT.glob(f"{country}_sx_*.parquet"))
                    for d in [pd.read_parquet(p)]], ignore_index=True)
    tb = tb.merge(cls.rename(columns={"sx": "entity_id"}), on="entity_id")
    tb["p1"] = pd.Index(ta_ids).get_indexer(tb.s1)
    return tb


def evaluate(ta, tb, vecs, cfg):
    ix = B.index(ta, vecs, cfg)
    t0 = time.time()
    h = B.search(ix, tb, vecs, cfg)
    secs = time.time() - t0
    out = {}
    for v in cfg["rev"]:
        g = h[h.view == v]
        rank = pd.Series(np.full(len(tb), 99), dtype=np.int32)
        hit = g[g.q1.to_numpy() == tb.p1.to_numpy()[g.qx]]
        rank[hit.qx.to_numpy()] = hit["rank"].to_numpy()
        out[v] = rank.to_numpy()
    return out, secs


if __name__ == "__main__":
    country, variants = sys.argv[1], json.loads(sys.argv[2])
    vecs = pickle.loads((OUT / "vecs_hash.pkl").read_bytes())
    ta = pd.read_parquet(TXT / f"{country}_s1.parquet")
    tb = kids(country)
    print(f"{country}: {len(tb):,} true-child queries vs {len(ta):,} S1, peak {peak_gb():.2f} GB", flush=True)
    rows, ranks = [], {}
    for name, over in variants.items():
        cfg = {**FULL_CFG, "views": {}, **over}
        r, secs = evaluate(ta, tb, vecs, cfg)
        for v, rk in r.items():
            ranks[f"{name}/{v}"] = rk
            row = {"variant": name, "view": v, "secs": round(secs), **{f"r@{k}": (rk < k).mean() for k in KS}}
            row |= {f"r2_{c}": (rk[g.index] < 2).mean() for c, g in tb.groupby("cls")}
            rows.append(row)
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if not k.startswith("r2_")},
                  f"peak {peak_gb():.2f} GB", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"revtune_{country}_{'_'.join(variants)}.csv", index=False)
    np.savez(OUT / f"revtune_{country}_{'_'.join(variants)}.npz", **{k.replace("/", "__"): v for k, v in ranks.items()},
             cls=tb.cls.to_numpy().astype(str))
