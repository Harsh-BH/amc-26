import sys, numpy as np, pandas as pd
S = sys.argv[1]
a = pd.read_parquet(S + "/s1all.parquet"); N1 = a.N1.iloc[0]; a["pos"] = a.row / N1
print("GLOBAL corr(S1 row, #copies) =", round(np.corrcoef(a.pos, a.ntot)[0, 1], 5))
print("  mean #copies by S1 row decile:", a.groupby((a.pos * 10).astype(int)).ntot.mean().round(3).tolist())
print("  singleton rate by decile:     ", a.groupby((a.pos * 10).astype(int)).ntot.apply(lambda v: (v == 0).mean()).round(4).tolist())
x = pd.read_parquet(S + "/amb.parquet"); tw = pd.read_parquet(S + "/s1tw.parquet").set_index("s1")
x = x[x.own_in & (x.tw.str.len() == 2)].copy()
x["oth"] = [t[0] if t[1] == o else t[1] for t, o in zip(x.tw, x.owner)]
ro, rt = tw.row.reindex(x.owner).to_numpy(), tw.row.reindex(x.oth).to_numpy()
no, nt = tw.naddr.reindex(x.owner).to_numpy(), tw.naddr.reindex(x.oth).to_numpy()
x["later"] = ro > rt; x["cmp"] = np.sign(no - nt)
se = lambda p, n: (p * (1 - p) / n) ** .5
for c, d in x.groupby("country"):
    p = d.later.mean(); print(f"{c}: P(owner later in S1) = {p:.4f} ± {se(p, len(d)):.4f} (n {len(d)})")
    print("   by owner-vs-other addressed count (−1 owner smaller, 0 equal, 1 larger):", d.groupby("cmp").later.agg(["mean", "size"]).round(4).to_dict("index"))
# relative position of x in its source vs owner in S1
xr = x.row / x.Nsrc
d_own, d_oth = np.abs(xr - ro / N1), np.abs(xr - rt / N1)
print("x rel-row closer to owner's S1 rel-row:", round((d_own < d_oth).mean(), 4))
# gap in rows between the two twins: does the effect depend on how far apart they are?
gap = np.abs(ro - rt) / N1
print("P(owner later) by twin row gap quartile:", x.groupby(pd.qcut(gap, 4, labels=False)).later.mean().round(4).tolist())
