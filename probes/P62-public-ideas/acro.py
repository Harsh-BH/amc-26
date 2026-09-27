"""P62 idea 1 -- acronym rule: an SX whose squashed core is the acronym of the S1 core (normalize.acronym, >= 2 tokens),
with the same house number and city (rf tables), is a match. Idea from a public write-up; our own implementation.
    .venv/bin/python probes/P62-public-ideas/acro.py pairs <split> <c>   # -> data/work/p62/acro_<split>_<c>.parquet
    .venv/bin/python probes/P62-public-ideas/acro.py holdout             # precision + misses vs the sub54 recipe
The join key is (acronym, hn, city); ambiguous keys (>1 S1 on the same SX) are kept and flagged (n_s1).
"""
import os
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


def pairs(split: str, c: str, key: str = "acro") -> pd.DataFrame:
    """key 'acro' = the acronym rule; key 'core' = the same address checks on an exact name_core match."""
    s1 = pq.read_table(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "hn", "city"]).to_pandas()
    s1["acro"] = N.acronym(s1.core) if key == "acro" else s1.core
    s1 = s1[(s1.acro != "") & (s1.hn != "") & (s1.city != "")].drop(columns="core")
    f, out = pq.ParquetFile(W / f"rf_{split}_sx_{c}.parquet"), []
    vs = pa.array(s1.acro.unique())
    for i in range(f.num_row_groups):                      # streamed: only SX whose core can hit a key survive
        t = f.read_row_group(i, columns=["id", "core", "hn", "city"])
        m = pc.and_(pc.less_equal(pc.utf8_length(t["core"]), 8), pc.invert(pc.match_substring(t["core"], " "))) \
            if key == "acro" else pc.is_in(t["core"], value_set=vs)
        out.append(t.filter(m).to_pandas())
    sx = pd.concat(out)
    sx = sx[(sx.hn != "") & (sx.city != "")]
    j = s1.merge(sx, left_on=["acro", "hn", "city"], right_on=["core", "hn", "city"])[["id_x", "id_y"]]
    j.columns = ["s1", "sx"]
    j["n_s1"] = j.groupby("sx").s1.transform("size").astype(np.int16)
    j["jac"], j["st"] = addr_jac(split, c, j)
    j.to_parquet(OUT / f"{'acro' if key == 'acro' else 'core'}_{split}_{c}.parquet", index=False)
    print(split, c, f"acro-like sx {len(sx):,}, rule pairs {len(j):,}, ambiguous sx {(j.n_s1 > 1).sum():,}", flush=True)
    return j


# hand-written street-type / article words (en + fr + in); street tokens = the S1 address minus these, its city, state, numbers
STREETW = set("rue r avenue av ave bd boulevard blvd chemin ch impasse imp allee place pl quai cours route rte square sq "
              "passage residence res lotissement lot street st road rd drive dr lane ln court ct circle cir way trail trl "
              "nagar marg floor flr no plot near opp".split())
ARTW = set("de la du des le les l d the of and".split())


def addr_jac(split: str, c: str, j: pd.DataFrame) -> tuple:
    """Token-set Jaccard of P3's normalised addresses of each rule pair (the 'identical address' check), and the share of
    the S1's street tokens found in the SX address (NaN when the S1 has none)."""
    import glob
    d = "txt" if split == "train" else "txt_test"
    def grab(files, ids):
        v = pa.array(sorted(set(ids)), pa.string())
        t = [pq.read_table(f, columns=["entity_id", "addr"]) for f in files]
        t = pa.concat_tables([x.filter(pc.is_in(x["entity_id"], value_set=v)) for x in t]).to_pandas()
        return t.drop_duplicates("entity_id").set_index("entity_id").addr
    a1 = grab([W / f"p3/{d}/{c}_s1.parquet"], j.s1).reindex(j.s1).fillna("").to_numpy(object)
    ax = grab(sorted(glob.glob(str(W / f"p3/{d}/{c}_sx_*.parquet"))), j.sx).reindex(j.sx).fillna("").to_numpy(object)
    rf = pq.read_table(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "city", "state"]).to_pandas().set_index("id")
    rf = rf.reindex(j.s1)
    out, st = np.zeros(len(j), np.float32), np.full(len(j), np.nan, np.float32)
    for i, (u, v, ci, sa) in enumerate(zip(a1, ax, rf.city.fillna("").to_numpy(object), rf.state.fillna("").to_numpy(object))):
        a, b = set(u.split()), set(v.split())
        out[i] = len(a & b) / len(a | b) if a | b else 0.0
        k = {t for t in a - set(ci.split()) - set(sa.split()) - STREETW - ARTW if not t.isdigit() and len(t) > 1}
        if k:
            st[i] = len(k & b) / len(k)
    return out, st


RULE = os.environ.get("ACRO_RULE", "acro")              # acro | core (exact name_core key)
JT = float(os.environ.get("ACRO_JAC", "0.6"))
ST = float(os.environ.get("ACRO_ST", "1.0"))          # share of the S1's street tokens present in the SX address


def rule(j: pd.DataFrame, jt: float = None) -> pd.DataFrame:
    """The rule's pairs: address Jaccard >= jt and street share >= ST, then n_s1 = rule S1 per SX among what is left."""
    j = j[(j.jac >= (JT if jt is None else jt)) & (j.st >= ST)].copy()
    j["n_s1"] = j.groupby("sx").s1.transform("size").astype(np.int16)
    return j


def holdout():
    """Train precision of the rule (all folds) and, on fold-0 S1, the pairs the sub54 recipe does not accept."""
    os.environ["P44_VARIANT"] = "r"
    sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer"]
    import ev as E
    import ev57 as V
    import a10
    import features as F
    gt = pq.read_table(W / "gt_pairs.parquet", columns=["s1", "sx"])
    ax = pa.array(pd.concat([pd.read_parquet(OUT / f"{RULE}_train_{c}.parquet", columns=["sx"]).sx for c in E.C]).unique())
    gt = gt.filter(pc.is_in(gt["sx"], value_set=ax)).to_pandas()
    fo = pd.read_parquet(W / "s1_folds.parquet", columns=["entity_id", "fold"]).set_index("entity_id").fold
    res, S = {}, V.load_s()
    for c in E.C:
        j = rule(pd.read_parquet(OUT / f"{RULE}_train_{c}.parquet"))
        g_ = gt[gt.sx.isin(set(j.sx))]
        j["y"] = pd.MultiIndex.from_arrays([j.s1, j.sx]).isin(pd.MultiIndex.from_arrays([g_.s1, g_.sx]))
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
        pairs(*sys.argv[2:])
    else:
        holdout()
