"""Importance-reweight the a10 'twin' / 'twin_p2mask' stacker gain from holdout to test S1 mix.
Stratum per S1 = max digitize(p2, E) over its argmax rows with 0.3<p<0.999 ('none' = no such row). Test strata from
#8c test p (p17 prob_test_*_bge). Gain_test = sum_b share_test(b) * mean dF_holdout(b). Aggregates only."""
import sys, gc
sys.path.insert(0, "probes/P23-error-analysis/work/lowp-fn")
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from a10_stack import load, arm, f_of, CTX, TWIN, T
E = [1e-6, 0.05, 0.2, 0.5]
S = [load(c) for c in ("India", "US")]
for s in S:
    pp = pd.read_parquet(f"data/work/p23/pairs_{s['c']}.parquet", columns=["sx", "am", "p", "p_m2", "r"])
    pp = pp[~pp.am].sort_values("p", ascending=False).drop_duplicates("sx")
    e5 = ((pp.p_m2 > 0.02) & (pp.p_m2 < 0.98) & pp.r.isna()).to_numpy()
    m = s["a"].sx.map(pd.Series(e5, index=pp.sx.to_numpy())).fillna(False).to_numpy().astype(bool)
    del pp; gc.collect()
    a = s["a"]; a["p2m"] = np.where(m, np.nan, a.p2); a["marginm"] = np.where(m, np.nan, a.margin)
    con = ((a.p > 0.3) & (a.p < 0.999)).to_numpy()
    b = np.where(con, np.digitize(a.p2.to_numpy(), E), -1)
    st = np.full(s["n"], -1); np.maximum.at(st, a.g.to_numpy(), b); s["st"] = st + 1   # 0 none, 1..5 bins 0..4
    asy = np.zeros(s["n"], bool); np.logical_or.at(asy, a.g.to_numpy(), m & con); s["asym"] = asy
# test strata
TS = {}
for c in ("India", "US"):
    t = pq.read_table(f"data/work/p17/prob_test_{c}_bge.parquet", columns=["s1", "sx", "p"])
    g1 = t["s1"].dictionary_encode().combine_chunks().indices.to_numpy(); gx = t["sx"].dictionary_encode().combine_chunks().indices.to_numpy()
    p = t["p"].to_numpy().astype(np.float64); del t
    o = np.lexsort((-p, gx)); gs = gx[o]; first = np.r_[True, gs[1:] != gs[:-1]]
    idx = np.flatnonzero(first); n = np.diff(np.r_[idx, len(o)])
    sec = np.where(n > 1, p[o][np.minimum(idx + 1, len(o) - 1)], 0.0)
    am_rows = o[idx]; pa_ = p[am_rows]
    con = (pa_ > 0.3) & (pa_ < 0.999)
    b = np.where(con, np.digitize(sec, E), -1)
    ng = g1.max() + 1; st = np.full(ng, -1); np.maximum.at(st, g1[am_rows], b)
    col = pq.read_table("data/work/test_source1.parquet", columns=["country"]).column("country")
    ntest = int(pc.sum(pc.equal(col, c)).as_py()); del col
    cnt = np.bincount(st + 1, minlength=6).astype(float); cnt[0] += ntest - ng
    TS[c] = cnt / cnt.sum(); del g1, gx, p, o; gc.collect()
W = {"India": 0.47 / 0.85, "US": 0.38 / 0.85}
for name, cols in (("twin", CTX + TWIN), ("twin_p2mask", [x for x in CTX if x not in ("p2", "margin")] + ["p2m", "marginm"] + TWIN)):
    for fh in (0, 1):
        ps, _ = arm(S, cols, fh); rep = 1 - fh
        tot_h, tot_t, tot_sym = 0.0, 0.0, 0.0
        for s, pr in zip(S, ps):
            d = f_of(s, pr >= T) - s["f0"]; k = s["h"] == rep
            sh = np.bincount(s["st"][k], minlength=6) / k.sum()
            mb = np.array([d[k & (s["st"] == j)].mean() if (k & (s["st"] == j)).any() else 0.0 for j in range(6)])
            mbs = np.array([d[k & (s["st"] == j) & ~s["asym"]].mean() if (k & (s["st"] == j) & ~s["asym"]).any() else 0.0 for j in range(6)])
            gh, gt, gs_ = d[k].mean(), (TS[s["c"]] * mb).sum(), (TS[s["c"]] * mbs).sum()
            tot_h += W[s["c"]] * gh; tot_t += W[s["c"]] * gt; tot_sym += W[s["c"]] * gs_
            print(f"{name:11s} h{fh}->h{rep} {s['c']:5s} holdout d={gh:+.6f} | test-reweighted {gt:+.6f} | test-rw, non-asym S1 only {gs_:+.6f} "
                  f"| share hold {np.round(sh,4).tolist()} test {np.round(TS[s['c']],4).tolist()} | dF/bin(1e-5) {np.round(mb*1e5,1).tolist()} "
                  f"| asym S1 share {s['asym'][k].mean():.4f}", flush=True)
        print(f"{name:11s} h{fh}->h{rep} LB-mix(0.47/0.38 of 1.0): holdout {tot_h*0.85:+.6f}  test-rw {tot_t*0.85:+.6f}  test-rw non-asym {tot_sym*0.85:+.6f}", flush=True)
        del ps; gc.collect()
