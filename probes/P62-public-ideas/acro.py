"""P62 idea 1 -- acronym rule: an SX whose squashed core is the acronym of the S1 core (normalize.acronym, >= 2 tokens),
with the same house number and city (rf tables), is a match. Idea from a public write-up; our own implementation.
    .venv/bin/python probes/P62-public-ideas/acro.py pairs <split> <c>   # -> data/work/p62/acro_<split>_<c>.parquet
    .venv/bin/python probes/P62-public-ideas/acro.py holdout             # precision + misses vs the sub54 recipe
The join key is (acronym, hn, city); ambiguous keys (>1 S1 on the same SX) are kept and flagged (n_s1).
"""
import sys

sys.path[:0] = ["code/business_entity_resolution/src", "."]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from worlds import WORK as W  # noqa: E402
import normalize as N  # noqa: E402

OUT = W / "p62"


def pairs(split: str, c: str) -> pd.DataFrame:
    s1 = pq.read_table(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "hn", "city"]).to_pandas()
    s1["acro"] = N.acronym(s1.core)
    s1 = s1[(s1.acro != "") & (s1.hn != "") & (s1.city != "")].drop(columns="core")
    f, out = pq.ParquetFile(W / f"rf_{split}_sx_{c}.parquet"), []
    for i in range(f.num_row_groups):                      # streamed: only single-token short cores survive
        t = f.read_row_group(i, columns=["id", "core", "hn", "city"])
        m = pc.and_(pc.less_equal(pc.utf8_length(t["core"]), 8), pc.invert(pc.match_substring(t["core"], " ")))
        out.append(t.filter(m).to_pandas())
    sx = pd.concat(out)
    sx = sx[(sx.hn != "") & (sx.city != "")]
    j = s1.merge(sx, left_on=["acro", "hn", "city"], right_on=["core", "hn", "city"])[["id_x", "id_y"]]
    j.columns = ["s1", "sx"]
    j["n_s1"] = j.groupby("sx").s1.transform("size").astype(np.int16)
    j.to_parquet(OUT / f"acro_{split}_{c}.parquet", index=False)
    print(split, c, f"acro-like sx {len(sx):,}, rule pairs {len(j):,}, ambiguous sx {(j.n_s1 > 1).sum():,}", flush=True)
    return j


def holdout():
    """Train precision of the rule (all folds) and, on fold-0 S1, the pairs the sub54 recipe does not accept."""
    import os
    os.environ["P44_VARIANT"] = "r"
    sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer"]
    import ev as E
    import ev57 as V
    import a10
    import features as F
    gt = pq.read_table(W / "gt_pairs.parquet", columns=["s1", "sx"]).to_pandas()
    fo = pd.read_parquet(W / "s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
    res, S = {}, V.load_s()
    for c in E.C:
        j = pd.read_parquet(OUT / f"acro_train_{c}.parquet")
        key = set(zip(gt.s1, gt.sx))
        j["y"] = [k in key for k in zip(j.s1, j.sx)]
        j["fold"] = fo.reindex(j.s1).to_numpy()
        r = {"pairs": len(j), "prec": float(j.y.mean()), "prec_unamb": float(j.y[j.n_s1 == 1].mean())}
        # decision of the sub54 recipe on the holdout frame (cross-fit halves), P57's full world = P44's frame
        A = V.te44.derive(pd.read_parquet(V.OUT / f"A_full_{c}.parquet"))
        for m in V.Z.MEM:
            A[f"z_{m}"] = V.MT.lg(A[f"p_{m}"].to_numpy())
        for m in V.Z.RANK:
            A[f"z_{m}"] = A[f"p_{m}"].to_numpy(np.float64)
        cols, _ = V.ARMS["sub54"]
        s = np.empty(len(A))
        for fh in (0, 1):
            m = A.half.to_numpy() == 1 - fh
            s[m] = V.score(V.boosters("sub54", fh), A[m], cols)
        id1, idx = F.ids("train", c)
        A["s1id"] = id1.take(pa.array(A.q1.to_numpy())).to_numpy(zero_copy_only=False)
        A["sxid"] = idx.take(pa.array(A.qx.to_numpy())).to_numpy(zero_copy_only=False)
        acc = s >= E.T
        h = j[j.fold == 0].copy()
        inA = pd.Series(np.arange(len(A)), index=pd.MultiIndex.from_arrays([A.s1id, A.sxid]))
        pos = inA.reindex(pd.MultiIndex.from_arrays([h.s1, h.sx])).to_numpy()
        h["in_frame"] = ~np.isnan(pos)
        h["acc"] = False
        h.loc[h.in_frame, "acc"] = acc[pos[h.in_frame.to_numpy()].astype(int)]
        taken = set(A.sxid[acc])                           # SX accepted by some scored S1 (fold 0 only)
        h["sx_taken"] = h.sx.isin(taken)
        miss = h[~h.acc]
        r.update(hold_pairs=len(h), hold_prec=float(h.y.mean()), hold_acc=int(h.acc.sum()), hold_miss=len(miss),
                 miss_true=int(miss.y.sum()), miss_in_frame=int(miss.in_frame.sum()), miss_taken=int(miss.sx_taken.sum()),
                 miss_free_unamb=int((~miss.sx_taken & (miss.n_s1 == 1)).sum()),
                 miss_free_unamb_prec=float(miss.y[~miss.sx_taken & (miss.n_s1 == 1)].mean())
                 if (~miss.sx_taken & (miss.n_s1 == 1)).any() else float("nan"))
        # F0.5 gain of adding the free, unambiguous misses to the sub54 decision, per half
        add = miss[~miss.sx_taken & (miss.n_s1 == 1)]
        St = S[c]
        ids = pd.read_parquet(W / f"p23/s1ctx_{c}.parquet", columns=["n_true"]).index   # S1 table order
        g = pd.Series(np.arange(len(ids)), index=ids)
        gA = A.g.to_numpy()
        n = len(St)
        tp0 = np.bincount(gA[acc & A.y.to_numpy()], minlength=n)
        np0 = np.bincount(gA[acc], minlength=n)
        ga = g.reindex(add.s1).to_numpy()
        assert not np.isnan(ga).any(), "holdout S1 missing from S1 table"
        ga = ga.astype(int)
        tp1 = tp0 + np.bincount(ga[add.y.to_numpy()], minlength=n)
        np1 = np0 + np.bincount(ga, minlength=n)
        nt = St.n_true.to_numpy()
        d = a10.fb(tp1, np1, nt) - a10.fb(tp0, np0, nt)
        for hh in (0, 1):
            x = d[St.half.to_numpy() == hh]
            r[f"dF_h{hh}"], r[f"se_h{hh}"] = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
        res[c] = r
        print(c, r, flush=True)
    for hh in (0, 1):
        res[f"lb_h{hh}"] = sum(E.LB_W[c] * res[c][f"dF_h{hh}"] for c in E.C)
    print({k: v for k, v in res.items() if k.startswith("lb")})
    return res


if __name__ == "__main__":
    if sys.argv[1] == "pairs":
        pairs(sys.argv[2], sys.argv[3])
    else:
        holdout()
