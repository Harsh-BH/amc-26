"""Decompose P14's labelled India IDF-mismatch proxy (pooled vs US-only IDF vs own-text fix). Aggregates only."""
import itertools, math
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
W = "/home/harsh1/github-repos/amc-26/data/work"
exec(open(__file__.rsplit("/", 1)[0] + "/fbeta.py").read())
names = ["IN_pooled", "IN_loco", "IN_locoown"]
probs = {n: pd.read_parquet(f"{W}/p14/prob_world_{n}.parquet") for n in names}
s1all = set().union(*[set(d.s1) for d in probs.values()])
folds = pd.read_parquet(f"{W}/s1_folds.parquet")
fcol = "fold"
idc = "entity_id" if "entity_id" in folds.columns else "s1"
f0 = set(folds.loc[folds[fcol] == 0, idc]) & s1all
gt = pq.read_table(f"{W}/gt_pairs.parquet")
gt = gt.filter(pc.is_in(gt["s1"], value_set=pa.array(sorted(f0)))).to_pandas()
print("scored S1", len(f0), "true pairs", len(gt))
truth = gt.groupby("s1").sx.apply(set).to_dict()
T = 0.69
for n, d in probs.items():
    d = d.copy()
    mx = d.groupby("sx").p.transform("max")
    d["am"] = d.p >= mx
    d = d[d.s1.isin(f0)]
    pairs = set(zip(d.s1, d.sx))
    pred = d[d.am & (d.p >= T)].groupby("s1").sx.apply(set).to_dict()
    cand = d.groupby("s1").sx.apply(set).to_dict()
    amset = d[d.am].groupby("s1").sx.apply(set).to_dict()
    rows = []
    for s in f0:
        ts, ps = truth.get(s, set()), pred.get(s, set())
        cs, ams = cand.get(s, set()), amset.get(s, set())
        tp = len(ts & ps); fp = len(ps - ts); miss = len(ts - cs)
        fn_c = ts & cs - ps
        stolen = len([x for x in fn_c if x not in ams]); lowp = len(fn_c) - stolen
        rows.append((len(ts), tp, fp, miss, stolen, lowp))
    a = np.array(rows)
    nT, tp, fp, miss, st, lp = a.T
    base = F(tp, fp, nT)
    X = {"miss": miss, "stolen": st, "lowp": lp, "fp": fp}
    PL = list(X); k = len(PL); phi = {p: np.zeros(len(a)) for p in PL}
    def v(S):
        t = tp + sum(X[q] for q in S if q != "fp"); f = fp if "fp" not in S else 0 * fp
        return F(t, f, nT)
    for p in PL:
        oth = [q for q in PL if q != p]
        for r in range(k):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(oth, r):
                phi[p] += w * (v(S + (p,)) - v(S))
    N = len(a)
    print(f"{n}: F0.5 {base.mean():.5f} cand_recall {1-miss.sum()/nT.sum():.4f} P {tp.sum()/(tp.sum()+fp.sum()):.4f} R {tp.sum()/nT.sum():.4f} | "
          + " ".join(f"{p} {phi[p].sum()/N:.5f}" for p in PL) + f" | pairs miss {miss.sum()} stolen {st.sum()} lowp {lp.sum()} fp {fp.sum()}")
