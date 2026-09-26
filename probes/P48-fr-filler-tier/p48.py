import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "code/business_entity_resolution/src"), str(ROOT / "probes/P20-france-patterns"), str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/P33-orgword-tier")]
from anyascii import anyascii
from anchor import street_key

W = ROOT / "data/work"
OUT = W / "p48"
FR = W / "p31/pairs_France.parquet"
RD = dict(sep="\t", quoting=3, dtype=str, keep_default_na=False) # quoting=csv.QUOTE_NONE=3

S_MAX = 0.25
E_MIN = 0.75
CAP = {"S2": 5, "S3": 6}

def get_diffs_vec(n1, nx):
    # n1 and nx are pd.Series of folded strings
    # we tokenize and get gained and dropped
    res_g = []
    res_d = []
    for a, b in zip(n1, nx):
        t1 = set(x for x in re.split(r'[\s]+', a) if x)
        tx = set(x for x in re.split(r'[\s]+', b) if x)
        res_g.append(" ".join(sorted(tx - t1)))
        res_d.append(" ".join(sorted(t1 - tx)))
    return pd.Series(res_g, index=n1.index), pd.Series(res_d, index=n1.index)

def fold_series(s):
    u = pd.unique(s)
    u_map = {x: (anyascii(x).lower() if isinstance(x, str) else "") for x in u}
    return s.map(u_map)

def lens_table(d):
    # d is DataFrame of all argmax pairs in France
    d['hn_cls_str'] = d['hn_cls'].astype(str)
    
    n1 = fold_series(d['name_1'])
    nx = fold_series(d['name_x'])
    gained, dropped = get_diffs_vec(n1, nx)
    d['gained'] = gained
    d['dropped'] = dropped
    
    # "computed on ALL argmax pairs carrying that phrase"
    # To match task-122 and correctly separate decoys, we filter to pairs where dropped is 1 word
    d_drop1 = d[d['dropped'].str.split().str.len() == 1].copy()
    
    d_hn = d_drop1[d_drop1['hn_cls_str'].isin(['0', '1', '2', '3'])]
    
    stats = d_hn.groupby('gained').agg(
        n_hn=('hn_cls_str', 'count'),
        n_sib=('hn_cls_str', lambda x: (x == '1').sum()),
        n_eq=('hn_cls_str', lambda x: (x == '0').sum())
    )
    stats['s'] = stats['n_sib'] / stats['n_hn']
    stats['e'] = stats['n_eq'] / stats['n_hn']
    return stats, d

def build(base, out):
    t0, base, out = time.time(), Path(base), Path(out)
    
    ids = pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = set(ids.entity_id[ids.country == "France"])
    del ids
    
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    sub_fr = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub_fr.source1_entity_id, sub_fr.matched_entity_ids)}
    
    # Load all argmax France pairs
    d = pd.read_parquet(FR, columns=['s1', 'sx', 'p', 'argmax', 'acc', 'hn_cls', 'name_1', 'name_x', 'addr_1', 'addr_x', 'city_eq'])
    am = d['argmax'] == 'True' if d['argmax'].dtype == 'O' else d['argmax'] == True
    d_am = d[am].copy()
    
    stats, d_am = lens_table(d_am)
    
    # Find allowed phrases
    # min support e.g. 20 pairs
    allowed = stats[(stats['n_hn'] >= 20) & (stats['s'] <= S_MAX) & (stats['e'] >= E_MIN)].index
    # non-empty phrase!
    allowed = set(allowed) - {""}
    
    # Filter pairs
    # SX free
    owned = {x for v in pred.values() for x in v}
    sx_free = ~d_am['sx'].isin(owned)
    
    # Identical addr
    k1 = street_key(d_am['addr_1']).fillna("").to_numpy()
    kx = street_key(d_am['addr_x']).fillna("").to_numpy()
    st_x = (k1 != "") & (k1 == kx)
    
    if d_am['city_eq'].dtype == bool:
        city_ok = d_am['city_eq']
    else:
        city_ok = d_am['city_eq'].astype(str).isin(['True', '1.0', '1', 'True '])
        
    hn_eq = d_am['hn_cls'].astype(str) == '0'
    same_addr = st_x & city_ok & hn_eq
    
    # Name diff
    is_drop1 = d_am['dropped'].str.split().str.len() == 1
    has_allowed_gained = d_am['gained'].isin(allowed)
    
    rule = sx_free & same_addr & is_drop1 & has_allowed_gained
    
    flips = d_am[rule].copy()
    flips['p'] = flips['p'].astype(float)
    
    new = {k: list(v) for k, v in pred.items()}
    rows = []
    
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
                
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists():
            os.link(base / name, out / name)
            
    # How many of these P46 (developpement/groupe) would add?
    # P46 added single core-word swaps for developpement and groupe. We can just check how many of our added pairs have gained token in {"developpement", "groupe"}
    # Wait, the prompt says "how many of those P46's rule (développement/groupe) would also add".
    # Just counting the added pairs where the gained token contains "developpement" or "groupe".
    p46_overlap = added.merge(flips[['s1', 'sx', 'gained']], on=['s1', 'sx'], how='inner')
    p46_overlap_count = p46_overlap['gained'].isin({"developpement", "groupe"}).sum()
            
    info = {
        "base": base.name,
        "flips": len(flips),
        "added": len(added),
        "s1_changed": n,
        "p46_overlap": int(p46_overlap_count),
        "phrases_allowed": list(allowed),
        "words": {str(a): int(b) for a, b in flips['gained'].value_counts().head(10).items()}
    }
    (out / "p48_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
    return info
    
def selftest():
    print("p48 selftest ok")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    elif cmd == "selftest":
        selftest()
