"""Does a per-source member-count prior pick the owner of an address-less SX among same-name S1?
posterior_j ∝ odds(p_j)^a * h_s(m_sj), h_s(m) = P(n_s = m+1) / P(n_s = m) from the train cluster-size law,
m_sj = addressed members of S1 j in the SX's source (from truth; addressed members are matched ~99%)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = "data/work/"
na = set()
for i in (2, 3):                                         # address-less SX ids, streamed
    for b in pq.ParquetFile(W + f"train_source{i}.parquet").iter_batches(columns=["entity_id", "business_address"], batch_size=500_000):
        d = b.to_pandas()
        a = d.business_address.fillna("").str.strip().str.strip("<>").str.upper()
        na.update(d.entity_id[a.isin(["", "NULL"])])
gt = pd.read_parquet(W + "gt_pairs.parquet")
gt["src"] = gt.sx.str[:2]
gt["na"] = gt.sx.isin(na)
m = gt[~gt.na].groupby(["s1", "src"]).size()            # addressed members per S1 and source
# cluster-size law per source (all members, train)
n = gt.groupby(["s1", "src"]).size().unstack(fill_value=0)
s1all = pd.read_parquet(W + "train_source1.parquet", columns=["entity_id"]).entity_id
n = n.reindex(s1all, fill_value=0)
H = {}
for s in ("S2", "S3"):
    f = n[s].value_counts(normalize=True).sort_index()
    H[s] = {k: f.get(k + 1, 1e-4) / f[k] for k in f.index}
    print(s, "law", f.round(3).to_dict(), "\n   hazard", {k: round(v, 2) for k, v in H[s].items()})
rows = []
for c in ("India", "US"):
    p = pd.read_parquet(W + f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "am", "pred"])
    p = p[p.sx.isin(na)]
    hs = p.groupby("sx").holdout.transform("any")        # SX a holdout S1 competes for
    p = p[hs & (p.p >= 0.05)].copy()
    p["src"] = p.sx.str[:2]
    p["m"] = [m.get((a, s), 0) for a, s in zip(p.s1, p.src)]
    # a true member does not count itself (it is address-less, so it is not in m anyway)
    p["h"] = [H[s].get(k, 1e-4) for s, k in zip(p.src, p.m)]
    rows.append(p.assign(country=c))
p = pd.concat(rows)
p["contested"] = p.groupby("sx").s1.transform("size") >= 2
q = p[p.contested].copy()
for a in (0.0, 0.5, 1.0):
    lo = np.log(np.clip(q.p, 1e-4, 1 - 1e-4) / (1 - np.clip(q.p, 1e-4, 1 - 1e-4)))
    q["w"] = np.exp(a * lo) * q.h
    q["post"] = q.w / q.groupby("sx").w.transform("sum")
    top = q.loc[q.groupby("sx").post.idxmax()]
    print(f"\nalpha={a}: contested address-less SX {len(top):,}; top-posterior S1 is true: {top.y.mean():.3f} "
          f"(current argmax by p: {q.loc[q.groupby('sx').p.idxmax()].y.mean():.3f})")
    for tau in (0.6, 0.7, 0.75, 0.8, 0.9):
        t = top[top.post >= tau]
        new = t[~t.pred]                                   # not accepted today
        print(f"   post>={tau}: {len(t):6,} SX, precision {t.y.mean():.3f} | of those not accepted today: {len(new):5,}, "
              f"precision {new.y.mean() if len(new) else float('nan'):.3f}  (India {int((new.country=='India').sum())}, US {int((new.country=='US').sum())})")
q.to_parquet("data/work/p23/twins_hazard.parquet", index=False)
