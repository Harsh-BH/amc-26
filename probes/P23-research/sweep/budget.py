"""Shapley error budget of #8c's dense holdout (P23 dump), per country. Aggregates only."""
import itertools, math, sys
import numpy as np, pandas as pd

D = "/home/harsh1/github-repos/amc-26/data/work/p23"
SIB = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
PL = ["block", "prune", "stolen", "lowp", "fp_sib", "fp_same", "fp_other"]


def F(tp, fp, n):
    tp = tp.astype(float); fp = fp.astype(float); n = n.astype(float)
    out = np.zeros_like(tp)
    z = n == 0
    out[z] = (tp[z] + fp[z] == 0).astype(float)
    m = (~z) & (tp > 0)
    P = tp[m] / (tp[m] + fp[m]); R = tp[m] / n[m]
    out[m] = 1.25 * P * R / (0.25 * P + R)
    return out


def num(s):
    return pd.to_numeric(s.fillna("").str.extract(r"^(\d+)")[0], errors="coerce")


def fpclass(e):
    d = num(e.sx_hn) - num(e.s1_hn)
    sib = d.isin(SIB)
    same = (e.s1_core.fillna("") == e.sx_core.fillna("")) & (e.s1_core.fillna("") != "")
    return np.where(sib, "fp_sib", np.where(same, "fp_same", "fp_other")), d


for c in ("India", "US"):
    pr = pd.read_parquet(f"{D}/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "am", "pred", "p"])
    h = pr[pr.holdout]
    ctx = pd.read_parquet(f"{D}/s1ctx_{c}.parquet")
    ctx = ctx.set_index("entity_id") if "entity_id" in ctx.columns else ctx
    e = pd.read_parquet(f"{D}/err_{c}.parquet")
    N = len(ctx)
    cnt = pd.DataFrame(0, index=ctx.index, columns=["tp"] + PL)
    cnt["n"] = ctx.n_true.astype(int)
    tp = h[h.y & h.pred].groupby("s1").size()
    cnt.loc[tp.index, "tp"] = tp
    mi = e[e.kind == "missed"]
    for k, sub in (("block", mi[mi.in_p3 != True]), ("prune", mi[mi.in_p3 == True]),
                   ("stolen", e[e.kind == "fn_stolen"]), ("lowp", e[e.kind == "fn_lowp"])):
        g = sub.groupby("s1").size(); cnt.loc[g.index, k] = g
    fp = e[e.kind == "fp"].copy()
    fp["cls"], fp["d"] = fpclass(fp)
    for k in ("fp_sib", "fp_same", "fp_other"):
        g = fp[fp.cls == k].groupby("s1").size(); cnt.loc[g.index, k] = g
    # sanity: fp from pairs equals err fp
    assert int((h.pred & ~h.y).sum()) == len(fp), (int((h.pred & ~h.y).sum()), len(fp))
    fnc = ["block", "prune", "stolen", "lowp"]
    assert (cnt.tp + cnt[fnc].sum(1) == cnt.n).all(), "tp+fn != n"
    err = cnt[(cnt[PL].sum(1) > 0)]
    base = F(err.tp.values, err[["fp_sib", "fp_same", "fp_other"]].sum(1).values, err.n.values)
    total_loss = (1 - base).sum() / N
    macro = 1 - total_loss
    k = len(PL); phi = {p: np.zeros(len(err)) for p in PL}
    X = {p: err[p].values for p in PL}
    for i, p in enumerate(PL):
        others = [q for q in PL if q != p]
        for r in range(k):
            wgt = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for S in itertools.combinations(others, r):
                def v(fixset):
                    t = err.tp.values + sum(X[q] for q in fixset if q in fnc)
                    f = sum(X[q] for q in ("fp_sib", "fp_same", "fp_other") if q not in fixset)
                    return F(t, f + 0 * t, err.n.values)
                phi[p] += wgt * (v(S + (p,)) - v(S))
    single = (err.n == 0).values
    empty_ns = ((err.n > 0) & (err.tp == 0) & (err[["fp_sib", "fp_same", "fp_other"]].sum(1) == 0)).values
    wrong_only = ((err.n > 0) & (err.tp == 0) & (err[["fp_sib", "fp_same", "fp_other"]].sum(1) > 0)).values
    print(f"\n== {c}: N={N}, macro F0.5 {macro:.5f}, loss {total_loss:.5f}; S1 with errors {len(err)}")
    print(f"  shapley sum check {sum(phi[p].sum() for p in PL)/N:.5f}")
    for p in PL:
        a = phi[p].sum() / N
        print(f"  {p:9s} all {a:.5f}  | singletons {phi[p][single].sum()/N:.5f} | nonsingleton-empty {phi[p][empty_ns].sum()/N:.5f}"
              f" | wrong-only {phi[p][wrong_only].sum()/N:.5f}  pairs {int(err[p].sum())}")
    print(f"  (e) singleton FP loss {(1-base[single]).sum()/N:.5f}  n S1 {int(single.sum())} / singletons total {int((cnt.n==0).sum())}")
    print(f"  (f) non-singleton empty loss {(1-base[empty_ns]).sum()/N:.5f}  n S1 {int(empty_ns.sum())}")
    print(f"      non-singleton all-wrong (no TP, has FP) loss {(1-base[wrong_only]).sum()/N:.5f}  n {int(wrong_only.sum())}")
    # FP details
    fp["orphan"] = fp.own_s1.isna()
    print("  FP classes:", fp.cls.value_counts().to_dict(), "orphan share", round(fp.orphan.mean(), 3),
          "sib3 share", round(fp.d.isin(SIB3).mean(), 3), "fp on singletons", int(fp.s1.map(cnt.n).eq(0).sum()))
    print("  FP orphan by class:", fp.groupby("cls").orphan.mean().round(3).to_dict())
    # FN details
    lp = e[e.kind == "fn_lowp"]
    print("  lowp p quantiles", lp.p.quantile([.1, .25, .5, .75, .9]).round(3).to_dict(), "share r notna", round(lp.r.notna().mean(), 3))
    st = e[e.kind == "fn_stolen"]
    print("  stolen: winner true (win_y) share", round(st.win_y.astype(float).mean(), 3), "winner holdout", round(st.win_holdout.astype(float).mean(), 3),
          "win_p>=t", round((st.win_p >= 0.71).mean(), 3), "own p q50", round(st.p.median(), 3))
    print("  missed sx_in_pool share", round(mi.sx_in_pool.astype(float).mean(), 3), "in_p3", round((mi.in_p3 == True).mean(), 3))
    # hn class of FN
    fn = e[e.kind.isin(["fn_lowp", "fn_stolen", "missed"])]
    d = num(fn.sx_hn) - num(fn.s1_hn)
    hnmis = d.notna() & (d != 0)
    print("  FN hn-mismatch share by kind:", fn.assign(m=hnmis).groupby("kind").m.mean().round(3).to_dict(),
          " TP sample hn-mismatch", round(((num(e[e.kind=='tp'].sx_hn)-num(e[e.kind=='tp'].s1_hn)).fillna(0)!=0).mean(), 3))
    web = fn.sx_web.fillna("").ne("")
    print("  FN web-name share by kind:", fn.assign(w=web).groupby("kind").w.mean().round(3).to_dict(),
          " TP sample web", round(e[e.kind=='tp'].sx_web.fillna('').ne('').mean(), 3))
    noaddr = fn.sx_addr.fillna("").str.strip().isin(["", "NULL", "<NULL>", "N/A"])
    print("  FN empty-addr share by kind:", fn.assign(w=noaddr).groupby("kind").w.mean().round(3).to_dict(),
          " TP sample", round(e[e.kind=='tp'].sx_addr.fillna('').str.strip().isin(['', 'NULL', '<NULL>', 'N/A']).mean(), 3))
    if c == "India":
        ind = fn.sx_name.fillna("").str.contains(r"[ऀ-෿]")
        print("  FN Indic-name share by kind:", fn.assign(w=ind).groupby("kind").w.mean().round(3).to_dict(),
              " TP sample", round(e[e.kind=='tp'].sx_name.fillna('').str.contains(r'[ऀ-෿]').mean(), 3))
    del pr, h, e
