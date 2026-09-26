"""P35 a8: twin propagation for BLOCKING misses. For each accepted addressed pair (X, b) of a holdout S1, every
address-less SX `a` of the whole train corpus (same country) with core(a) == core(b) that is NOT a candidate of X
becomes a propagated pair (X, a). Precision by core frequency (address-less SX sharing the core) and by whether `a`
is already accepted somewhere in the holdout frame. Train truth (gt_pairs) labels the pairs. Aggregates only."""
import sys
sys.path.insert(0, "probes/P35-twin-propagate")
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from ev import load, gain, W
from frame import noaddr

gt = pd.read_parquet(W / "gt_pairs.parquet")
own = gt.drop_duplicates("sx").set_index("sx").s1
for c in ("India", "US"):
    S = load(c, ["s1", "sx", "holdout", "y", "p", "pred", "noaddr", "core"]); d = S["d"]
    ids = []
    for i in (2, 3):
        f = pq.ParquetFile(W / f"train_source{i}.parquet")
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=["entity_id", "business_address", "country"]).to_pandas()
            t = t[t.country == c]
            ids.append(t.entity_id[noaddr(t.business_address)].to_numpy()); del t
    ids = np.concatenate(ids)
    v = pa.array(ids, pa.large_string()); f = pq.ParquetFile(W / f"rf_train_sx_{c}.parquet"); cs = []
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=["id", "core"])
        cs.append(t.filter(pc.is_in(t["id"].cast(pa.large_string()), value_set=v)).to_pandas()); del t
    na = pd.concat(cs, ignore_index=True); del cs, v, ids
    na = na[na.core != ""]
    na["fq"] = na.groupby("core").id.transform("size")
    B = d.loc[d.holdout & d.pred & ~d.noaddr, ["s1", "core"]].drop_duplicates()
    j = B.merge(na[na.fq <= 5], on="core").rename(columns={"id": "sx"})
    cand = pd.MultiIndex.from_frame(d[["s1", "sx"]])
    j = j[~pd.MultiIndex.from_frame(j[["s1", "sx"]]).isin(cand)]
    j["y"] = j.sx.map(own).to_numpy() == j.s1.to_numpy()
    acc = set(d.sx[d.pred]); inframe = set(d.sx)
    j["state"] = np.where(j.sx.isin(acc), "accepted_elsewhere", np.where(j.sx.isin(inframe), "in_frame_unaccepted", "not_in_frame"))
    j["n_x"] = j.groupby("sx").s1.transform("size")               # holdout S1 claiming this a via twins
    print(f"\n== {c}: address-less corpus SX {len(na):,}; propagated pairs (fq<=5, not a candidate) {len(j):,}, prec {j.y.mean():.3f}")
    print(j.groupby(["state", j.fq.clip(upper=3)]).y.agg(["size", "mean"]).round(3).to_string())
    m = (j.state != "accepted_elsewhere") & (j.fq == 1) & (j.n_x == 1)
    print(f"  best slice (unassigned, unique core, one claimant): {int(m.sum()):,} pairs, prec {j.y[m].mean():.3f}")
    # F0.5 effect of adding the best slice to holdout S1 (the SX is not accepted anywhere in the frame)
    miss = pd.read_parquet(W / f"p23/missed_{c}.parquet")
    print(f"  of {len(miss):,} missed true pairs, recovered by the slice: "
          f"{len(set(zip(j.s1[m & j.y], j.sx[m & j.y])) & set(zip(miss.s1, miss.sx))):,}")
    add = j[m]
    g = S["g"]; n = S["n"]
    pos = pd.Index(pd.read_parquet(W / f"p23/s1ctx_{c}.parquet").index).get_indexer(add.s1)
    base = d.pred.to_numpy() & (g >= 0)
    tp = np.bincount(g[base & d.y.to_numpy()], minlength=n) + np.bincount(pos[add.y.to_numpy()], minlength=n)
    npred = np.bincount(g[base], minlength=n) + np.bincount(pos, minlength=n)
    from ev import fb
    dl = fb(tp, npred, S["n_true"]) - S["f0"]
    for hv in (0, 1):
        x = dl[S["h"] == hv]
        print(f"  add slice: half {hv} {x.mean():+.6f} ± {x.std(ddof=1) / np.sqrt(len(x)):.6f}")
