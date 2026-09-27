# Aggregates only: CE-vs-GBDT disagreement on the test band, split by simple label-free pair classes.
import sys, re, numpy as np, pandas as pd
sys.path[:0] = ["/home/harsh1/github-repos/amc-26", "/home/harsh1/github-repos/amc-26/probes/P4b-rerank"]
import rerank as rr
W = "/home/harsh1/github-repos/amc-26/data/work"
num = re.compile(r"\b\d+\b")
for c in sys.argv[1:]:
    d = pd.read_parquet(f"{W}/p13/prob_test_{c}_blend.parquet", columns=["s1", "sx", "p_gbdt"])
    d = d[(d.p_gbdt > 0.02) & (d.p_gbdt < 0.98)]
    d = d.merge(pd.read_parquet(f"{W}/p13/r/r_bge_pool_test_{c}.parquet"), on=["s1", "sx"])
    q1, qx = rr.rows("test", c, d.s1, d.sx)
    ta, tb = rr.texts("test", c, q1, qx)
    sp = lambda t: t.split(" | ", 1) if " | " in t else [t, ""]
    A = [sp(t) for t in ta]; B = [sp(t) for t in tb]
    na = np.array([a[0] for a in A], object); nb = np.array([b[0] for b in B], object)
    aa = [a[1].split() for a in A]; ab = [b[1].split() for b in B]
    h1 = [next((w for w in x if w.isdigit()), None) for x in aa]; hx = [next((w for w in x if w.isdigit()), None) for x in ab]
    hn = np.array(["miss" if (u is None or v is None) else ("eq" if u == v else "diff") for u, v in zip(h1, hx)])
    name_eq = na == nb
    last_in = np.array([bool(x) and x[-1] in set(y) for x, y in zip(aa, ab)])
    dl = np.array([len(x) - len(y) for x, y in zip(aa, ab)])
    dis = (d.r.to_numpy() > 0.5) != (d.p_gbdt.to_numpy() > 0.5)
    ce_hi = d.r.to_numpy() > 0.5
    def row(lbl, m):
        return f"{lbl}: share {m.mean():.3f} disagree {dis[m].mean():.3f} CE-says-match-among-dis {(ce_hi & dis)[m].sum()/max((dis & m).sum(),1):.2f}"
    print(f"== {c}: band {len(d):,}, disagree {dis.mean():.3f}")
    for k in ("eq", "diff", "miss"): print("  hn", row(k, hn == k))
    print("  ", row("name exact equal", name_eq)); print("  ", row("name differs", ~name_eq))
    print("  ", row("S1 last addr token in SX addr", last_in)); print("  ", row("S1 last addr token NOT in SX", ~last_in))
    print("  ", row("SX addr shorter by >=2 tokens", dl >= 2)); print("  ", row("addr len within 1", np.abs(dl) <= 1))
