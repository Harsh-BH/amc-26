"""Verified-merge anchor features (idea from the public repo's stage C; our own code) for the India/US combiner, row-aligned
with a frame's (q1, qx, g). Anchors = the S1's OTHER argmax pairs with pa >= .9 (the shipped #34 p').
  a_n        anchors of the S1 (excluding this row)
  a_name     max token-set ratio of this SX's name vs the anchors' names      a_name_s1   the same vs the S1 name (reference)
  a_addr     max token-set ratio of addresses vs the anchors'                 a_addr_eq   address identical (folded) to an anchor
    anchors.py SPLIT C   -> data/work/p70/an_<split>_<c>.parquet"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from rapidfuzz import fuzz
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P70-fr-roles"), str(ROOT / "code/business_entity_resolution/src")]
import p70  # noqa: E402
import features as FT  # noqa: E402
import normalize as N  # noqa: E402
W = ROOT / "data/work"
AN = ["a_n", "a_name", "a_addr", "a_addr_eq", "a_name_s1"]


def build(split, c, chunk=300_000):
    t0 = time.time()
    if split == "train":
        A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["q1", "qx", "g", "pa"])
    else:
        A = pd.read_parquet(W / f"p50/T_{c}.parquet", columns=["q1", "qx", "g", "pa"])
    id1, idx = FT.ids(split, c)
    id1, idx = id1.to_numpy(zero_copy_only=False), idx.to_numpy(zero_copy_only=False)
    out = np.full((len(A), len(AN)), np.nan, np.float32)
    g_all = A.g.to_numpy()
    order = np.argsort(g_all, kind="stable")
    # chunk by S1 groups so every S1's rows are processed together
    cuts = np.flatnonzero(np.diff(g_all[order])) + 1
    bounds = [0] + [cuts[np.searchsorted(cuts, k)] if np.searchsorted(cuts, k) < len(cuts) else len(order) for k in range(chunk, len(order), chunk)] + [len(order)]
    bounds = sorted(set(bounds))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = order[lo:hi]
        a = A.iloc[rows]
        s1, sx = id1[a.q1.to_numpy()], idx[a.qx.to_numpy()]
        rx = p70._raw(split, pd.unique(sx), [2, 3], ["business_name", "business_address", "country"])
        r1 = p70._raw(split, pd.unique(s1), [1], ["business_name", "country"])
        nx = N.fold(rx.business_name.reindex(sx).fillna("")).to_numpy()
        ax = N.fold(rx.business_address.reindex(sx).fillna("")).to_numpy()
        n1 = N.fold(r1.business_name.reindex(s1).fillna("")).to_numpy()
        axk = pd.Series(ax).str.replace(r"[^a-z0-9]", "", regex=True).to_numpy()
        g, pa = a.g.to_numpy(), a.pa.to_numpy()
        res = np.full((len(a), len(AN)), np.nan, np.float32)
        start = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
        for s, e in zip(start[:-1], start[1:]):
            anc = [j for j in range(s, e) if pa[j] >= 0.9]
            for i in range(s, e):
                others = [j for j in anc if j != i]
                res[i, 0] = len(others)
                res[i, 4] = fuzz.token_set_ratio(nx[i], n1[i])
                if others:
                    res[i, 1] = max(fuzz.token_set_ratio(nx[i], nx[j]) for j in others)
                    res[i, 2] = max(fuzz.token_set_ratio(ax[i], ax[j]) for j in others) if ax[i] else np.nan
                    res[i, 3] = float(bool(axk[i]) and any(axk[i] == axk[j] for j in others))
        out[rows] = res
        print(f"  {split} {c} rows {hi:,}/{len(order):,} {time.time() - t0:.0f}s peak {p70.peak():.2f} GB", flush=True)
    pd.DataFrame(out, columns=AN).to_parquet(W / f"p70/an_{split}_{c}.parquet", index=False)
    print(f"{split} {c} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
