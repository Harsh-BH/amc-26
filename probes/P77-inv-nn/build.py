"""P57: per-row inputs of the neural set-matcher, on #7's holdout frame (P17's yardstick rows).

Rows = run.py's holdout list per country (m2_big p on the tau-0.03 pruned list, every competing train S1 included, so
argmax_sx is exact). Columns = M2's 72 features (joined from the train X files) + M2 p, CE r, blended p, P17's
house-number columns + SX-side competition stats. Written to data/work/p57/holdout_<c>.parquet (+ _meta.npz: ntrue, half
per scored S1).

    .venv/bin/python probes/P57-neural/build.py [India US]
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
_s = importlib.util.spec_from_file_location("p17", ROOT / "probes/P17-hn-stack/stack.py")
P17 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P17)
import features as F  # noqa: E402
import model as M  # noqa: E402
from run import holdout_s1  # noqa: E402
from worlds import WORK  # noqa: E402

OUT = WORK / "p57"


def m2_feats(c: str, keys: np.ndarray) -> pd.DataFrame:
    """M2's feature columns for rows keys (q1 << 32 | qx), in keys' order, from the full train X files."""
    files = F.xfiles("train", c)
    cols = M.feats(pd.DataFrame(columns=pq.read_schema(files[0]).names))
    parts, ks = [], []
    for f in files:
        t = pq.read_table(f, columns=["q1", "qx"])
        k = (t.column("q1").to_numpy().astype(np.int64) << 32) | t.column("qx").to_numpy().astype(np.int64)
        m = np.isin(k, keys)
        if m.any():
            parts.append(pq.read_table(f, columns=cols).filter(m).to_pandas().astype(np.float32))
            ks.append(k[m])
        del t, k
        F.trim()
    X, k = pd.concat(parts, ignore_index=True), np.concatenate(ks)
    ix = pd.Index(k).get_indexer(keys)
    assert (ix >= 0).all(), f"{c}: {int((ix < 0).sum())} rows missing from the X files"
    return X.iloc[ix].reset_index(drop=True)


def sx_ctx(qx: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """The SX's competition: how many S1 claim it, this row's rank there, and the best OTHER S1's p."""
    k = pd.DataFrame({"qx": qx, "p": p})
    g = k.groupby("qx").p
    n, mx = g.transform("size").to_numpy(), g.transform("max").to_numpy()
    rk = g.rank(method="first", ascending=False).to_numpy()
    sec = k.assign(r=rk)[rk == 2].set_index("qx").p
    s2 = sec.reindex(qx).to_numpy()
    other = np.where(rk == 1, s2, mx)                    # best p among the other S1 (NaN when alone)
    return pd.DataFrame({"c_sxcnt": n.astype(np.float32), "c_sxrk": rk.astype(np.float32),
                         "c_sxother": other.astype(np.float32), "c_sxgap": (p - np.nan_to_num(other)).astype(np.float32)})


def build(c: str):
    t0 = time.time()
    s1, t = holdout_s1()
    d = P17.frame("holdout", c)
    pos, y, ntrue, hv = P17.labels(d, c, s1, t)
    q1, qx = d["q1"], d["qx"]
    keys = (q1 << 32) | qx
    m2 = pd.read_parquet(WORK / "p9" / P17.M2.format(split="holdout", c=c))
    assert (m2.q1.to_numpy() == q1).all()
    r = pd.read_parquet(WORK / "p9" / P17.RC.format(split="holdout", c=c))
    rr = pd.Series(r.r.to_numpy(np.float32), index=r.k.to_numpy(np.int64)).reindex(keys).to_numpy(np.float32)
    X = pd.concat([m2_feats(c, keys), P17.X_of(d, "sup"), sx_ctx(qx, d["p"])], axis=1)
    X["lpm"], X["r"] = P17.lg(m2.p.to_numpy(np.float32)).astype(np.float32), rr
    X["q1"], X["qx"], X["p"], X["y"], X["pos"] = q1, qx, d["p"], y, pos
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_m2" if P17.BASE == "m2" else ""               # m2: France's regime, p = M2's p (no CE)
    X.to_parquet(OUT / f"holdout{tag}_{c}.parquet", index=False)
    np.savez(OUT / f"holdout{tag}_{c}_meta.npz", ntrue=ntrue, half=hv)
    print(f"{c}: {len(X):,} rows, {X.shape[1]} cols, {int((pos >= 0).sum()):,} holdout-S1 rows, "
          f"{len(ntrue):,} scored S1, {time.time() - t0:.0f}s, peak {P17.peak():.2f} GB", flush=True)


if __name__ == "__main__":
    P17.BASE = "m2" if "base=m2" in sys.argv else "e5"
    for c in [a for a in sys.argv[1:] if "=" not in a] or ["India", "US"]:
        build(c)
