"""Label-free check of the decision layer on P4's TEST probabilities: answer-size profile per country under the
plain threshold (D0) and the recommended cfg, next to the same profile on holdout. France has no labels, so a
profile far from US/India is the only warning we can get.

    .venv/bin/python probes/P5-decide/test_diag.py probes/P5-decide/cfg_p4_s1only.json

With P4's `am` column (argmax over every S1) and the S1-only stack, every S1 is decided from its own rows, so the
pairs are processed in S1 chunks (exact) and memory stays flat.
"""
import json, sys, time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bench import WORK
from decide import select
from proxy import peak


def profile(path, cfgs, chunks=8, where=None):
    """Per cfg: share of S1 answered empty, mean answer size, share of answers of size >= 5 (all S1 of the file)."""
    t = pq.read_table(path, columns=["s1"]).column(0)
    h = (pd.util.hash_pandas_object(pd.Series(t.to_numpy(zero_copy_only=False)), index=False).to_numpy() >> 11) % chunks
    del t
    sizes = {k: [] for k in cfgs}
    for c in range(chunks):
        d = pq.read_table(path, columns=["s1", "sx", "p", "am"]).filter(pa.array(h == c)).to_pandas()
        if where is not None:
            d = d[where(h == c)]
        codes, _ = pd.factorize(d.s1)
        d = d.assign(s1=codes)
        for k, cfg in cfgs.items():
            sizes[k].append(np.bincount(codes[select(d, cfg)], minlength=codes.max() + 1))
        del d, codes
    out = {}
    for k, v in sizes.items():
        n = np.concatenate(v)
        out[k] = {"n_s1": len(n), "empty": round(float((n == 0).mean()), 4), "mean_size": round(float(n.mean()), 3),
                  "size5+": round(float((n >= 5).mean()), 4)}
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(WORK.parents[1]))
    from ledger import log
    t0 = time.time()
    cfg = json.load(open(sys.argv[1]))
    assert "all" not in sys.argv[1], "SX-side ctx needs every S1 of an SX in one frame: not chunkable"
    cfgs = {"D0": {"t": 0.71}, "rec": cfg}
    rows = {}
    hold = WORK / "p4/prob_holdout.parquet"
    cty = pd.read_parquet(hold, columns=["country"]).country.to_numpy()
    for c in ("India", "US"):
        rows[f"holdout_{c}"] = profile(hold, cfgs, 2, where=lambda m, c=c: cty[m] == c)
    for c in ("France", "India", "US"):
        rows[f"test_{c}"] = profile(WORK / f"p4/prob_test_{c}.parquet", cfgs)
        print(c, rows[f"test_{c}"], f"peak {peak():.2f} GB", flush=True)
    for k, v in rows.items():
        print(f"{k:14s}", "  ".join(f"{n}: empty {m['empty']:.4f} size {m['mean_size']:.3f} 5+ {m['size5+']:.4f}"
                                   for n, m in v.items()))
    log("exp", probe="P5-decide", subproblem="decision", name="p4-test-answer-profile", world="test+holdout",
        metrics={f"{k}_{n}_{s}": m[s] for k, v in rows.items() for n, m in v.items() for s in ("empty", "mean_size")},
        config={"cfg": sys.argv[1], "D0": 0.71}, hypothesis="France answer profile close to US/India under the cfg",
        verdict="inconclusive", runtime_s=round(time.time() - t0), peak_gb=round(peak(), 2))
