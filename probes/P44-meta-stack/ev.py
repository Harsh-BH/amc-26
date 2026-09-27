"""P44-meta-stack: the holdout frame of #34's argmax pairs + real-metric helpers (per-S1 macro F0.5, paired SE).

A_{c}.parquet = one row per #8c argmax pair of a scored holdout S1 (P34's ho_{c} rows, same order) with
  P34's a10 columns, pa = #34's p' out of fold (P36 base), g = S1 index, half = run.half of the S1,
  q1/qx = train codes, P44's derived hn columns (solution_p44.HN), q = P10 pruner score,
  and p_<model> for every P44 job attached (attach()).
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/ev.py frame        # build + self-check (#34 reproduces)
    probes/P44-meta-stack/py.sh probes/P44-meta-stack/ev.py attach hn21 <job>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
OUT = W / "p44ms"
sys.path[:0] = [str(ROOT / "probes/P36-iu-residual"), str(ROOT / "probes/P44-meta-stack"),
                str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import base as B  # noqa: E402  (P36: #34 out of fold)

T = B.T
LB_W = B.LB_W
C = ("India", "US")
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def keys(q1, qx):
    return (np.asarray(q1, np.int64) << 32) | np.asarray(qx, np.int64)


def frame(c: str) -> pd.DataFrame:
    import features as F
    from solution_p44 import HnTab
    st = B.state(c)
    ho = pd.read_parquet(W / f"p34/ho_{c}.parquet")
    gi = pd.Series(np.arange(len(st["ids"])), index=st["ids"])
    ho["g"] = gi.reindex(ho.s1).to_numpy()
    ho["half"] = st["h"][ho.g.to_numpy()]
    p0, p1 = (np.load(W / f"p34/ho_p_{c}_h{fh}.npy") for fh in (0, 1))
    ho["pa"] = np.where(ho.half == 1, p0, p1)
    id1, idx = F.ids("train", c)
    ho["q1"] = pc.index_in(pa.array(ho.s1.to_numpy(object), pa.string()), value_set=id1).to_numpy(zero_copy_only=False)
    ho["qx"] = pc.index_in(pa.array(ho.sx.to_numpy(object), pa.string()), value_set=idx).to_numpy(zero_copy_only=False)
    del id1, idx
    assert ho.q1.notna().all() and ho.qx.notna().all()
    ho["q1"], ho["qx"] = ho.q1.astype(np.int64), ho.qx.astype(np.int64)
    for k, v in HnTab(W / "p44ms/hn", "train").feats(c, ho.q1.to_numpy(), ho.qx.to_numpy()).items():
        ho[k] = v
    assert (ho.hn_cls.to_numpy() == ho.hc.to_numpy()).all(), "derived hn class != P34's hc"
    hk = pq.read_table(W / f"p10/hold_{c}.parquet", columns=["q1", "qx"])
    q = pd.Series(pq.read_table(W / f"p10/hold_q_{c}.parquet").column("q").to_numpy(),
                  index=keys(hk.column("q1").to_numpy(), hk.column("qx").to_numpy()))
    ho["q"] = q[~q.index.duplicated()].reindex(keys(ho.q1, ho.qx)).to_numpy(np.float32)
    s1 = pd.DataFrame({"n_true": st["n_true"], "half": st["h"], "f34": st["f34"], "f8": st["f8"]})
    s1.to_parquet(OUT / f"S1_{c}.parquet")
    ho.to_parquet(OUT / f"A_{c}.parquet")
    return ho


def load(c: str):
    return pd.read_parquet(OUT / f"A_{c}.parquet"), pd.read_parquet(OUT / f"S1_{c}.parquet")


def attach(name: str, job: str):
    """p_<name> of a P44 job's holdout preds onto A (the pred files are fetched to OUT/pred_<name>/)."""
    for c in C:
        A, _ = load(c)
        t = pq.read_table(OUT / f"pred_{name}/holdout_{c}.parquet")
        s = pd.Series(t.column("p").to_numpy(), index=keys(t.column("q1").to_numpy(), t.column("qx").to_numpy()))
        A[f"p_{name}"] = s.reindex(keys(A.q1, A.qx)).to_numpy(np.float32)
        print(c, name, f"missing {A[f'p_{name}'].isna().mean():.5f}", flush=True)
        A.to_parquet(OUT / f"A_{c}.parquet")


def per_s1(A: pd.DataFrame, S: pd.DataFrame, acc: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 when exactly the argmax pairs `acc` are predicted."""
    import a10
    n = len(S)
    g, y = A.g.to_numpy(), A.y.to_numpy()
    return a10.fb(np.bincount(g[acc & y], minlength=n), np.bincount(g[acc], minlength=n), S.n_true.to_numpy())


def tune_t(A, S, s, half, grid=np.arange(0.40, 0.951, 0.01)) -> float:
    """t maximising the mean per-S1 F0.5 over the S1 of `half` (ties -> the first)."""
    m = (S.half.to_numpy() == half)
    return float(grid[int(np.argmax([per_s1(A, S, s >= t)[m].mean() for t in grid]))])


def delta(A, S, acc, rep: int) -> tuple:
    """(mean, SE, n_up, n_down) of per-S1 F0.5 vs #34 over the S1 of half rep."""
    d = per_s1(A, S, acc) - S.f34.to_numpy()
    x = d[S.half.to_numpy() == rep]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), int((x > 0).sum()), int((x < 0).sum())


def lbeq(r: dict) -> tuple:
    """LB-eq mean and SE from per-country (mean, SE, ...) tuples."""
    return (sum(LB_W[c] * r[c][0] for c in C), float(np.sqrt(sum((LB_W[c] * r[c][1]) ** 2 for c in C))))


if __name__ == "__main__":
    if sys.argv[1] == "frame":
        for c in C:
            A = frame(c)
            A, S = load(c)
            f = per_s1(A, S, A.pa.to_numpy() >= T)
            assert np.allclose(f, S.f34.to_numpy()), "#34 does not reproduce from A"
            print(c, len(A), "am pairs; #34 F0.5 all", round(f.mean(), 6), "h1", round(f[S.half == 1].mean(), 6),
                  "| q missing", round(A.q.isna().mean(), 5), "| hn both", round(A.hn_lev.notna().mean(), 3), flush=True)
    elif sys.argv[1] == "attach":
        attach(sys.argv[2], sys.argv[3])
