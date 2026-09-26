import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"),
                str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]

import normalize as N
from anchor import street_key
import anchor
import pyarrow.parquet as pq

W = ROOT / "data/work"

def clean_name(s):
    s = N.fold(pd.Series(s))
    s = s.str.replace(r"[^a-z0-9&+\s]+", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def get_rel(n1, nx):
    t1 = n1.split()
    tx = nx.split()
    s1 = set(t1)
    sx = set(tx)
    add = [t for t in tx if t not in s1]
    drop = [t for t in t1 if t not in sx]
    
    if len(add) == 0 and len(drop) == 1:
        return "drop1", "", drop[0]
    if len(add) == 1 and len(drop) == 1:
        return "sub1", add[0], drop[0]
    if len(add) >= 1 and len(drop) == 0:
        return "add_phrase", " ".join(add), ""
    return "other", "", ""

def task1():
    print("Loading data for Task 1...")
    d = pd.read_parquet(W / "p31/pairs_France.parquet")
    # argmax, not acc, SX free, hn_cls != 1
    # Wait, SX free? The base submission tells us which SX are free.
    # The prompt says: "data/work/p31/pairs_France.parquet (argmax, not acc, SX free, hn_cls != 1)". 
    # But wait, does it mean we should filter by argmax, ~acc, hn_cls != 1? "SX free" usually means not in the 'acc' list of the base.
    
    # Let's filter:
    m = d.argmax & (~d.acc) & (d.hn_cls != 1)
    d = d[m].reset_index(drop=True)
    
    # define "identical" as P28/P46 do: hn equal, city overlap, street exact
    h1 = d.hn_1.fillna("").to_numpy()
    hx = d.hn_x.fillna("").to_numpy()
    hn_eq = (h1 != "") & (hx != "") & (h1 == hx)
    
    # city overlap
    p1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "places"]).set_index("id").places.reindex(d.s1).fillna("").to_numpy()
    px = pd.read_parquet(W / "rf_test_sx_France.parquet", columns=["id", "places"]).set_index("id").places.reindex(d.sx).fillna("").to_numpy()
    city_eq = np.array([a != "" and bool(set(a.split("|")) & set(b.split("|"))) for a, b in zip(p1, px)])
    
    # street exact
    src = [W / f"test_source{i}.parquet" for i in (1, 2, 3)]
    a1 = anchor._addr(src[:1], pd.unique(d.s1)).reindex(d.s1).fillna("")
    ax = anchor._addr(src[1:], pd.unique(d.sx)).reindex(d.sx).fillna("")
    k1 = street_key(a1).to_numpy()
    kx = street_key(ax).to_numpy()
    st_eq = (k1 != "") & (k1 == kx)
    
    identical = hn_eq & city_eq & st_eq
    print(f"Total eligible pairs: {len(d)}, identical address: {identical.sum()}")
    
    d = d[identical].reset_index(drop=True)
    
    # name relation
    n1 = clean_name(d.name_1).to_numpy()
    nx = clean_name(d.name_x).to_numpy()
    
    rels = [get_rel(a, b) for a, b in zip(n1, nx)]
    kind = np.array([r[0] for r in rels])
    gained = np.array([r[1] for r in rels])
    dropped = np.array([r[2] for r in rels])
    
    d["kind"] = kind
    d["gained"] = gained
    d["dropped"] = dropped
    
    # report counts
    for k in ["drop1", "sub1", "add_phrase"]:
        dk = d[d.kind == k]
        print(f"--- {k} ---")
        if k == "drop1":
            print(dk.dropped.value_counts().head(20))
        else:
            print(dk.gained.value_counts().head(20))

if __name__ == "__main__":
    task1()
