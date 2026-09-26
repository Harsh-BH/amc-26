import itertools, math, numpy as np, pandas as pd
exec(open(__file__.rsplit("/", 1)[0] + "/fbeta.py").read())
D = "/home/harsh1/github-repos/amc-26/data/work/p23"
for c in ("India", "US"):
    e = pd.read_parquet(f"{D}/err_{c}.parquet")
    pr = pd.read_parquet(f"{D}/pairs_{c}.parquet", columns=["s1", "holdout", "y", "pred"])
    ctx = pd.read_parquet(f"{D}/s1ctx_{c}.parquet"); ctx = ctx.set_index("entity_id") if "entity_id" in ctx.columns else ctx
    N = len(ctx)
    na = e.sx_addr.fillna("").str.strip().str.upper().isin(["", "NULL", "<NULL>", "N/A", "NA", "NONE", "-"])
    e["na"] = na
    e["pl"] = None
    fnk = e.kind.isin(["missed", "fn_lowp", "fn_stolen"])
    e.loc[fnk & na, "pl"] = "fn_noaddr"
    e.loc[fnk & ~na & (e.kind == "missed"), "pl"] = "fn_addr_cand"
    e.loc[fnk & ~na & (e.kind != "missed"), "pl"] = "fn_addr_match"
    e.loc[(e.kind == "fp") & na, "pl"] = "fp_noaddr"
    e.loc[(e.kind == "fp") & ~na, "pl"] = "fp_addr"
    PL = ["fn_noaddr", "fn_addr_cand", "fn_addr_match", "fp_noaddr", "fp_addr"]
    FN = PL[:3]; FPp = PL[3:]
    h = pr[pr.holdout]
    cnt = pd.DataFrame(0, index=ctx.index, columns=["tp"] + PL); cnt["n"] = ctx.n_true.astype(int)
    tp = h[h.y & h.pred].groupby("s1").size(); cnt.loc[tp.index, "tp"] = tp
    for p in PL:
        g = e[e.pl == p].groupby("s1").size(); cnt.loc[g.index, p] = g
    err = cnt[cnt[PL].sum(1) > 0]
    X = {p: err[p].values for p in PL}; k = len(PL); phi = {p: 0.0 for p in PL}
    def v(S):
        return F(err.tp.values + sum(X[q] for q in S if q in FN), sum(X[q] for q in FPp if q not in S) + 0 * err.tp.values, err.n.values)
    for p in PL:
        oth = [q for q in PL if q != p]
        for r in range(k):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(oth, r):
                phi[p] += w * (v(S + (p,)) - v(S)).sum()
    print(c, {p: round(phi[p] / N, 5) for p in PL}, "sum", round(sum(phi.values()) / N, 5), {p: int(err[p].sum()) for p in PL})
