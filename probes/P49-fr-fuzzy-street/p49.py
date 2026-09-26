import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
from rapidfuzz.fuzz import ratio

ROOT = Path("../../") if not Path("../../data").exists() else Path("../../")
if not (ROOT / "data").exists():
    ROOT = Path(".")

sys.path[:0] = [str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P20-france-patterns")]
import normalize as N
import model as M
from fr20 import _strip as strip_r1
from fr_street import french_street_key
from patterns import name_rel

W = ROOT / "data/work"
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}

def _addr(files: list, ids: np.ndarray) -> pd.Series:
    out = []
    for f in files:
        t = pq.read_table(f, columns=["entity_id", "business_address"])
        t = t.filter(pc.is_in(t.column("entity_id"), value_set=pa.array(ids, pa.large_string())))
        out.append(t.to_pandas())
    a = pd.concat(out, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id").business_address
    return a.reindex(ids).fillna("")

def rule_p49(d: pd.DataFrame, split: str, c: str, t: float) -> dict:
    xs = "train" if split == "holdout" else "test"
    u1, ux = pd.unique(d.s1), pd.unique(d.sx)
    
    def _rf(path, ids, cols):
        t = pq.read_table(path, columns=["id", *cols])
        t = t.filter(pc.is_in(t.column("id"), value_set=pa.array(ids, pa.large_string())))
        return t.to_pandas().set_index("id").reindex(ids)
    
    r1 = _rf(W / f"rf_{xs}_s1_{c}.parquet", u1, ["core", "hn", "city", "places"])
    rx = _rf(W / f"rf_{xs}_sx_{c}.parquet", ux, ["core", "hn", "city", "places", "is_web"])
    
    h1 = r1.hn.reindex(d.s1).fillna("").to_numpy()
    hx = rx.hn.reindex(d.sx).fillna("").to_numpy()
    
    both = (h1 != "") & (hx != "")
    dlt = np.full(len(d), np.nan)
    dlt[both] = pd.to_numeric(pd.Series(hx[both]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[both]), errors="coerce").to_numpy(float)
    
    hn_eq = (dlt == 0)
    hn_miss = (h1 != "") & (hx == "")
    cand_hn = hn_eq | hn_miss
    
    c1 = r1.core.reindex(d.s1).fillna("").to_numpy()
    cx = rx.core.reindex(d.sx).fillna("").to_numpy()
    wx = rx.is_web.reindex(d.sx).fillna(False).to_numpy()
    
    nr = [name_rel(a, b, w) for a, b, w in zip(c1[cand_hn], cx[cand_hn], wx[cand_hn])]
    kr = np.array([k for k, _ in nr], object)
    
    name_ok = np.isin(kr, ["eq"]) | np.isin(kr, ["add1", "drop1", "sub1"]) # Wait! The brief says core name equal (or P20 R1-equal). 
    # For simplicity let's just use my core_eq | r1_eq
    c1_s = np.array([strip_r1(x) for x in c1[cand_hn]])
    cx_s = np.array([strip_r1(x) for x in cx[cand_hn]])
    my_name_ok = ((c1[cand_hn] == cx[cand_hn]) & (c1[cand_hn] != "") & ~wx[cand_hn]) | ((c1_s == cx_s) & (c1_s != "") & ~wx[cand_hn])
    
    p1 = r1.places.reindex(d.s1).fillna("").to_numpy()[cand_hn]
    px = rx.places.reindex(d.sx).fillna("").to_numpy()[cand_hn]
    city_eq = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)])
    
    not_sib = ~np.isin(dlt[cand_hn], list(SIB3))
    
    qx = pd.factorize(d.sx)[0]
    p = d.p.to_numpy(np.float64)
    am = M.argmax_sx(qx, p)
    best = pd.Series(p).groupby(qx).transform("max").to_numpy()
    free_am = am | (best < t)
    
    cand = my_name_ok & city_eq & not_sib & free_am[cand_hn]
    
    full_cand = np.zeros(len(d), bool)
    full_cand[cand_hn] = cand
    
    s1c = d.s1.to_numpy()[full_cand]
    sxc = d.sx.to_numpy()[full_cand]
    
    src = [W / f"{xs}_source{i}.parquet" for i in (1, 2, 3)]
    addr1 = _addr(src[:1], pd.unique(s1c)).reindex(s1c).fillna("")
    addrx = _addr(src[1:], pd.unique(sxc)).reindex(sxc).fillna("")
    
    if c == "France":
        st1 = french_street_key(addr1).to_numpy()
        stx = french_street_key(addrx).to_numpy()
    else:
        from anchor import street_key
        st1 = street_key(addr1).to_numpy()
        stx = street_key(addrx).to_numpy()
    
    st_fuz = np.array([a != "" and b != "" and ratio(a, b) >= 85 for a, b in zip(st1, stx)])
    
    mask_eq = np.zeros(len(d), bool)
    mask_miss = np.zeros(len(d), bool)
    
    is_eq = hn_eq[full_cand]
    is_miss = hn_miss[full_cand]
    
    mask_eq[full_cand] = st_fuz & is_eq
    mask_miss[full_cand] = st_fuz & is_miss
    
    return {"arm_eq": mask_eq, "arm_miss": mask_miss}

def build(base: str, out: str):
    base, out = Path(base), Path(out)
    ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = set(ids.entity_id[ids.country == "France"])
    
    sub = pd.read_csv(base / "matching_results.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    sub = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    
    d = pd.read_parquet(W / "p31/pairs_France.parquet", columns=["s1", "sx", "p"])
    # compute mask
    m = rule_p49(d, "test", "France", 0.985)
    mask = m["arm_eq"] | m["arm_miss"]
    flips = d[mask]
    
    owned = {x for v in pred.values() for x in v}
    new, rows = {k: list(v) for k, v in pred.items()}, []
    CAP = {"S2": 5, "S3": 6}
    for s1, sx, p in flips.sort_values("p", ascending=False)[["s1", "sx", "p"]].itertuples(index=False):
        if sx in owned: continue
        cur = new.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]: continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx, p))
    
    added = pd.DataFrame(rows, columns=["s1", "sx", "p"])
    chg = set(added.s1)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(base / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in chg:
                g.write(f"{s}\t{','.join(sorted(new[s]))}\n")
                n += 1
            else:
                g.write(line)
    
    import os
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(base / name, out / name)
    print(f"Added {len(added)} pairs")

        elif sys.argv[1] == "build":
            build(sys.argv[2], sys.argv[3])
