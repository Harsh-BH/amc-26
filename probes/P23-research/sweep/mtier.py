"""Aggregates only. Tier M: core-equal argmax pairs with S1 hn non-empty and SX hn '' (after Brief 5's lstrip('#(')
parse), legal relation in {same, dropadd, none}, street tokens equal, city equal, (core, city) unique among S1, no other
kept candidate of the SX with the same core. US/India holdout precision by band; France flips / empties at t .985."""
import sys, re, numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds, pyarrow.parquet as pq
R = "/home/harsh1/github-repos/amc-26/"; W = R + "data/work/"
sys.path[:0] = [R + "code/business_entity_resolution/src", R + "probes/P23-research/sweep"]
import normalize as N
exec(open(R + "probes/P23-research/sweep/hnfix.py").read().split("SIB = {1,")[0].split("import normalize as N")[1])  # hn_new + assert
src = open(R + "probes/P23-research/sweep/legal_swap.py").read().split("res = {}")[0]
exec(src.split("import features as F, normalize as N")[1])   # LEG, legset, hnrel
def lrel(L1, Lx):
    o = []
    for a, b in zip(L1, Lx):
        if not a and not b: o.append("none")
        elif a == b: o.append("same")
        elif a and b and not (a & b): o.append("swap")
        elif not a or not b: o.append("dropadd")
        else: o.append("overlap")
    return np.array(o)
DIG = re.compile(r"\d")
def street_toks(addr, cities, table):
    out = []
    for ss, cy in zip(N._segments(addr), cities):
        toks = set()
        for x in ss:
            if x in table or x == cy: continue
            y = N._seg(x)
            if not y or y == cy: continue
            toks |= {t for t in y.split() if not DIG.search(t) and len(t) > 1}
        out.append(frozenset(toks))
    return out
def fetch(files, ids, cols):
    ids = pa.array(np.unique(ids))
    parts = [ds.dataset(W + f).to_table(filter=pc.field("entity_id").isin(ids), columns=["entity_id"] + cols).to_pandas() for f in files]
    return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id")
for c in sys.argv[1:]:
    fr = c == "France"; split = "test" if fr else "train"; t = 0.985
    if fr:
        allc = pd.read_parquet(W + "p19/prob_test_France_idf144_bge.parquet", columns=["s1", "sx", "p"])
    else:
        allc = pd.read_parquet(W + f"p23/pairs_{c}.parquet", columns=["s1", "sx", "holdout", "y", "p", "am"])
        allc = allc[allc.holdout].drop(columns="holdout")
    r1 = pq.read_table(W + f"rf_{split}_s1_{c}.parquet", columns=["id", "core", "hn", "city"]).to_pandas().set_index("id")
    u = r1[(r1.core != "") & (r1.city != "")].groupby(["core", "city"]).size()
    uniq = set(u[u == 1].index); del u
    rx = pq.read_table(W + f"rf_{split}_sx_{c}.parquet", columns=["id", "core", "hn", "city"],
                       filters=[("id", "in", list(set(allc.sx)))]).to_pandas().set_index("id")
    allc["c1"] = allc.s1.map(r1.core).fillna(""); allc["cx"] = allc.sx.map(rx.core).fillna("")
    allc["same"] = (allc.c1 == allc.cx) & (allc.cx != "")
    nsame = allc.groupby("sx").same.transform("sum")
    if fr:
        am = allc.p.to_numpy() == allc.groupby("sx").p.transform("max").to_numpy()
        d = allc[am].drop_duplicates("sx").copy(); d["nsame"] = nsame[am].loc[d.index]
    else:
        d = allc[allc.am].copy(); d["nsame"] = nsame[allc.am]
    n_s1_all = r1.shape[0] if fr else None
    base_nonempty = set(d.s1[d.p >= t]) if fr else None
    del allc, nsame
    d["h1"] = d.s1.map(r1.hn).fillna(""); d["hx"] = d.sx.map(rx.hn).fillna("")
    d["ci1"] = d.s1.map(r1.city).fillna(""); d["cix"] = d.sx.map(rx.city).fillna("")
    pre = d.same & (d.h1 != "") & (d.hx == "") & (d.ci1 == d.cix) & (d.ci1 != "")
    n_pre_all = int((d.same & (d.hx == "")).sum())
    d = d[pre].copy(); del r1, rx
    a1 = fetch([f"{split}_source1.parquet"], d.s1.to_numpy(), ["business_name", "business_address"])
    ax = fetch([f"{split}_source2.parquet", f"{split}_source3.parquet"], d.sx.to_numpy(), ["business_name", "business_address"])
    A1 = d.s1.map(a1.business_address).fillna("").astype("str"); AX = d.sx.map(ax.business_address).fillna("").astype("str")
    hxn = hn_new(AX).to_numpy()
    lr = lrel(legset(d.s1.map(a1.business_name)), legset(d.sx.map(ax.business_name)))
    del a1, ax
    if fr: N.STATES["france"].update(N.fr_departements())
    table = N.STATES.get(c.lower(), {})
    st_eq = np.array([a == b for a, b in zip(street_toks(A1, d.ci1, table), street_toks(AX, d.cix, table))])
    uq = np.array([(a, b) in uniq for a, b in zip(d.c1, d.ci1)])
    parsed = hxn != ""
    hrel_new = hnrel(d.h1.astype(str), pd.Series(hxn).astype(str))
    okl = np.isin(lr, ["same", "dropadd", "none"])
    M = (~parsed) & okl & st_eq & uq & (d.nsame.to_numpy() == 1)
    M_nost = (~parsed) & okl & uq & (d.nsame.to_numpy() == 1)
    p = d.p.to_numpy()
    print(f"== {c}: core-eq & SX hn '' argmax pairs {n_pre_all:,}; +S1 hn & city eq {len(d):,}; newly parsed by lstrip('#(') "
          f"{parsed.sum():,} (-> eq {(hrel_new == 'eq').sum():,}, sib {(hrel_new == 'sib').sum():,}, other {(hrel_new == 'other').sum():,}); "
          f"legal ok {okl.mean():.3f}; street eq {st_eq.mean():.3f}; uniq {uq.mean():.3f}; nsame==1 {(d.nsame.to_numpy()==1).mean():.3f}; M {M.sum():,}", flush=True)
    for nm, m in [("M", M), ("M_no_street", M_nost)]:
        for lo, hi in [(0, .3), (.3, .5), (.5, .71), (.71, .9), (.9, .95), (.95, .985), (.985, 1.01), (.5, .985)]:
            b = m & (p >= lo) & (p < hi)
            s = f"   {nm} p[{lo},{hi}): n {b.sum():,}"
            if not fr and b.sum():
                y = d.y.to_numpy()[b]; s += f" true {y.mean():.4f} (se {np.sqrt(y.mean()*(1-y.mean())/len(y)):.4f})"
            print(s)
        if not fr:
            for lg in ("same", "dropadd", "none"):
                b = m & (lr == lg) & (p >= .5) & (p < t)
                print(f"   {nm} legal={lg} p[.5,.985): n {b.sum():,} true {d.y.to_numpy()[b].mean() if b.sum() else float('nan'):.4f}")
        else:
            for tk in (0.95, 0.90, 0.71, 0.5):
                add = d[m & (p >= tk) & (p < t)]
                print(f"   {nm} t_key {tk}: +{len(add):,} pairs, {add.s1.nunique():,} S1, {len(set(add.s1) - base_nonempty):,} empty S1 filled")
    if fr:
        for lg in ("same", "dropadd", "none"):
            b = M & (lr == lg) & (p >= .5) & (p < t); print(f"   M legal={lg} p[.5,.985): n {b.sum():,}")
    del d
