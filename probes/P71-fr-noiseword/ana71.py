"""P71 A analogue: the same label-free noise-word learning + add pool on a labelled country (train, #8c holdout dump
p23/pairs_C: base pred + truth), precision of each kind:word class and the holdout F0.5 gain of adding it.

    .venv/bin/python probes/P71-fr-noiseword/ana71.py US|India
"""
import collections
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "probes/P71-fr-noiseword")
import p71 as P  # noqa: E402

t0 = time.time()
c = sys.argv[1]
# P3's blocked list of the fold-0 holdout S1 (= P6 / P42 LOCO fold) + #8c's holdout lines (P23 fp dump) as the base
d = pd.read_parquet(P.W / f"p3/cands_holdout_full_{c}.parquet", columns=["s1", "sx", "is_true"]).rename(columns={"is_true": "y"})
b8 = pd.read_parquet(P.W / f"p23/fp/pred_{c}.parquet", columns=["s1", "sx", "pred"])
b8 = b8[b8.pred]
d["pred"] = pd.MultiIndex.from_frame(d[["s1", "sx"]]).isin(pd.MultiIndex.from_frame(b8[["s1", "sx"]]))
d["holdout"] = True
print(f"{c}: base lines {len(b8):,}, in the blocked list {int(d.pred.sum()):,}", flush=True)
hs = set(d.s1)
owned = set(b8.sx)
del b8


def rows(ids, k):
    v = pa.array(sorted(set(ids)), pa.large_string())
    rf = pq.read_table(P.W / f"rf_train_{k}_{c}.parquet", columns=["id", "hn", "places", "city"] + (["is_web"] if k == "sx" else []),
                       filters=[("id", "in", v)]).to_pandas().set_index("id")
    rf["name"] = P.raw(rf.index, (1,) if k == "s1" else (2, 3), "business_name", "train").reindex(rf.index).fillna("")
    rf["tok"] = P.toks(rf.name)
    return rf


s1, sx = rows(d.s1, "s1"), rows(d.sx, "sx")
print(f"{c}: {len(d):,} pairs, {len(s1):,} S1, {len(sx):,} SX {time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)

# label-free token ratio on a population sample of the country's train names (every 4th S1, every 8th SX)
def names(i, step):
    t = pq.read_table(P.W / f"train_source{i}.parquet", columns=["business_name"], filters=[("country", "==", c)]).column(0)
    return pd.Series(t.to_numpy(zero_copy_only=False)[::step])


cnt = {}
for k, src, step in (("s1", (1,), 4), ("sx", (2, 3), 8)):
    cc, n = collections.Counter(), 0
    for i in src:
        tk = P.toks(names(i, step))
        n += len(tk)
        cc.update(w for t in tk for w in t.split())
    cnt[k] = (cc, n)
r = pd.DataFrame({"df1": pd.Series(cnt["s1"][0]), "dfx": pd.Series(cnt["sx"][0])}).fillna(0)
r["ratio"] = ((r.dfx + 1) / cnt["sx"][1]) / ((r.df1 + 1) / cnt["s1"][1])
del cnt
print(f"{c}: ratio table {len(r):,} tokens {time.time() - t0:.0f}s", flush=True)

f = P.pair_frame(d[["s1", "sx"]], s1, sx)
f["y"], f["pred"], f["holdout"] = d.y.to_numpy(), d.pred.to_numpy(), d.holdout.to_numpy()
a = f[f.pred & f.D.notna()]
r0 = float((a.D == 0).sum() / max((a.D < 0).sum(), 1))
L = P.lens(f, r0)
L["word"] = L.index.str.split(":").str[1]
L = L.join(r[["df1", "dfx", "ratio"]], on="word")
m = f.kind.isin(["ins", "sub"]) & ~f.ins.str.contains(" ") & f.place & (f.D == 0)
g = f[m].assign(k=f.kind[m] + ":" + f.ins[m]).groupby("k")
L["acc_eq"], L["y_eq"] = g.pred.mean().reindex(L.index), g.y.mean().reindex(L.index)
L.sort_values("n", ascending=False).to_csv(P.WORK / f"lens_{c}.csv")
sd = f[(f.kind == "sub") & (f.D == 0) & f.place][["s1", "sx", "kind", "ins", "drop", "y", "pred"]]
sd.to_parquet(P.WORK / f"subd0_{c}.parquet", index=False)
if len(sys.argv) > 2 and sys.argv[2] == "subd0":
    sys.exit(0)
pool = P.pool(f, s1, sx, owned, c, "train")
pool = pool[pool.s1.isin(hs)].reset_index(drop=True)
pool.to_parquet(P.WORK / f"pool_{c}.parquet", index=False)
print(f"{c}: r0 {r0:.1f}; pool (holdout S1) {len(pool):,} true {pool.y.mean():.3f} {time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)
k = pool.kind + ":" + pool.ins
print(pool.groupby(k).y.agg(["size", "mean"]).sort_values("size", ascending=False).head(30).round(3).to_string())
L2 = L[(L.n >= 20) & (L.ratio >= 3)].sort_values("n", ascending=False)
print(L2.head(40).round(3).to_string())

# holdout F0.5 of base vs base + each class (free SX, caps), per-S1 exact
gt = pd.read_parquet(P.W / "gt_pairs.parquet")
gt = gt[gt.s1.isin(hs)]
ntrue = gt.s1.value_counts()
hd = f[f.holdout]
pred = {s: [] for s in hs}
for s, x in zip(hd.s1[hd.pred], hd.sx[hd.pred]):
    pred[s].append(x)
truth = set(zip(gt.s1, gt.sx))


def f05(pr):
    tot = 0.0
    for s, v in pr.items():
        nt = ntrue.get(s, 0)
        tp = sum((s, x) in truth for x in v)
        if not v:
            tot += 1.0 if nt == 0 else 0.0
        elif tp:
            p_, r_ = tp / len(v), tp / nt
            tot += 1.25 * p_ * r_ / (0.25 * p_ + r_)
    return tot / len(pr)


b = f05(pred)
res = {"r0": r0, "pool": len(pool), "pool_true": float(pool.y.mean()), "base": b, "classes": {}}
for key in k.value_counts().index[:25]:
    add = pool[(k == key).to_numpy() & (pool.lf.to_numpy() != 4)]
    new, rw = P.merge(pred, add)
    dl = f05(new) - b
    res["classes"][key] = {"n": len(add), "prec": round(float(add.y.mean()), 4), "added": len(rw), "dF_holdout": round(dl, 6)}
    print(f"{c} {key}: n {len(add)} prec {add.y.mean():.3f} added {len(rw)} dF {dl:+.6f}", flush=True)
json.dump(res, open(P.WORK / f"ana71_{c}.json", "w"), indent=1)
print(f"{c}: done {time.time() - t0:.0f}s peak {P.peak()} GB", flush=True)
