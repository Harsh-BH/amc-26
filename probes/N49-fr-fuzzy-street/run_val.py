import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from rapidfuzz.fuzz import ratio

sys.path[:0] = ["../../code/business_entity_resolution/src", "../../probes/P28-addr-anchor"]
from anchor import street_key

W = Path("../../data/work")

def val(c):
    print(f"\n=== {c} ===")
    t_file = pq.read_table(W / f"p23/pairs_{c}.parquet")
    t_file = t_file.filter(pc.equal(t_file.column("holdout"), pa.scalar(True)))
    d = t_file.to_pandas()
    t = d[d.pred].p.min() if d.pred.any() else 0.71
    
    xs = "train"
    u1, ux = pd.unique(d.s1), pd.unique(d.sx)
    def _rf(path, ids, cols):
        t = pq.read_table(path, columns=["id", *cols])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
        return t.to_pandas().set_index("id").reindex(ids)
    
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", u1, ["core", "hn", "city", "places"])
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", ux, ["core", "hn", "city", "places"])
    
    c1 = r1.core.reindex(d.s1).fillna("").to_numpy()
    cx = rx.core.reindex(d.sx).fillna("").to_numpy()
    core_eq = (c1 == cx) & (c1 != "")
    
    h1 = r1.hn.reindex(d.s1).fillna("").to_numpy()
    hx = rx.hn.reindex(d.sx).fillna("").to_numpy()
    hn_eq = (h1 == hx) & (h1 != "")
    hn_miss = (h1 != "") & (hx == "")
    
    both = (h1 != "") & (hx != "")
    dlt = np.full(len(d), np.nan)
    dlt[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
    not_sib = ~np.isin(dlt, list(SIB3))
    
    city1 = r1.city.reindex(d.s1).fillna("").to_numpy()
    cityx = rx.city.reindex(d.sx).fillna("").to_numpy()
    city_eq = (city1 == cityx) & (city1 != "")
    
    qx = pd.factorize(d.sx)[0]
    p = d.p.to_numpy(np.float64)
    import model as M
    am = M.argmax_sx(qx, p)
    best = pd.Series(p).groupby(qx).transform("max").to_numpy()
    free_am = am | (best < t)
    
    cand = core_eq & city_eq & not_sib & free_am & (hn_eq | hn_miss)
    
    s1c = d.s1.to_numpy()[cand]
    sxc = d.sx.to_numpy()[cand]
    
    def _addr(files: list, ids: np.ndarray) -> pd.Series:
        out = []
        for f in files:
            t = pq.read_table(f, columns=["entity_id", "business_address"])
            t = t.filter(pc.is_in(t.column("entity_id"), value_set=pa.array(ids, pa.large_string())))
            out.append(t.to_pandas())
        a = pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").business_address
        return a.reindex(ids).fillna("")

    src = [W / f"{xs}_source{i}.parquet" for i in (1, 2, 3)]
    addr1 = _addr(src[:1], pd.unique(s1c)).reindex(s1c).fillna("")
    addrx = _addr(src[1:], pd.unique(sxc)).reindex(sxc).fillna("")
    
    st1 = street_key(addr1).to_numpy()
    stx = street_key(addrx).to_numpy()
    
    st_ex = (st1 != "") & (st1 == stx)
    st_fuz = np.array([a != "" and b != "" and ratio(a, b) >= 85 for a, b in zip(st1, stx)])
    
    # chain guard
    pool = pq.read_table(W / f"rf_{xs}_s1_{c}.parquet", columns=["id", "city", "hn"]).to_pandas()
    pool = pool[(pool.hn != "") & (pool.city != "")]
    ck = pool.set_index("id").reindex(s1c).fillna("")
    ck["street"] = st1
    mates = pool.merge(ck[["city", "hn"]].drop_duplicates(), on=["city", "hn"])
    mates["street"] = street_key(_addr(src[:1], mates.id.to_numpy())).to_numpy()
    n_addr = mates[mates.street != ""].groupby(["city", "hn", "street"]).size()
    chain = ck.join(n_addr.rename("n"), on=["city", "hn", "street"]).n.fillna(0).to_numpy() >= 2
    
    m_eq = hn_eq[cand]
    m_miss = hn_miss[cand]
    
    for name, st_mask in [("Exact", st_ex), ("Fuzzy", st_fuz)]:
        for arm_n, m_arm in [("hn_eq", m_eq), ("hn_miss", m_miss)]:
            for cg, m_cg in [("NoGuard", np.ones(len(cand), bool)), ("ChainGuard", ~chain)]:
                m = np.zeros(len(d), bool)
                m[cand] = st_mask & m_arm & m_cg
                df_arm = d[m]
                if len(df_arm) == 0: continue
                prec = df_arm.y.mean()
                df_f = df_arm[~df_arm.pred]
                prec_f = df_f.y.mean() if len(df_f) > 0 else 0
                print(f"{name:5s} {arm_n:7s} {cg:10s} | All: {prec:.4f} ({df_arm.y.sum():4d}/{len(df_arm):4d}) | Pred=F: {prec_f:.4f} ({df_f.y.sum():4d}/{len(df_f):4d})")

val("US")
val("India")

