import json
import os
import sys
import time
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAP = {"S2": 5, "S3": 6}
RD = dict(sep="\t", quoting=3, dtype=str, keep_default_na=False) # csv.QUOTE_NONE == 3

def get_flips():
    con = duckdb.connect()
    con.execute("SET memory_limit='1200MB'")
    
    q = f"""
    WITH fr_tc AS (
        SELECT core, COUNT(*) as k 
        FROM '{ROOT}/data/work/rf_test_s1_France.parquet' 
        WHERE core != '' GROUP BY core
    ),
    p31 AS (
        SELECT 
            p.s1, p.sx, p.p
        FROM '{ROOT}/data/work/p31/pairs_France.parquet' p
        JOIN fr_tc t ON p.core_1 = t.core
        WHERE p.argmax = True
          AND p.acc = False
          AND p.core_1 = p.core_x
          AND p.city_1 = p.city_x
          AND p.core_1 != ''
          AND p.city_1 != ''
          AND p.hn_cls != 1
          AND p.hn_cls != 0
          AND LOWER(p.name_1) = LOWER(p.name_x)
          AND t.k = 1
          AND NOT p.src LIKE 'S2-%'
    )
    SELECT * FROM p31
    """
    return con.execute(q).df()

def merge(pred, add):
    owned = {x for v in pred.values() for x in v}
    new, rows = {k: list(v) for k, v in pred.items()}, []
    for s1, sx, p in add.sort_values("p", ascending=False).itertuples(index=False):
        if sx in owned:
            continue
        cur = new.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]:
            continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx, p))
    return new, pd.DataFrame(rows, columns=["s1", "sx", "p"])

def build(base: str, out: str):
    t0, base, out = time.time(), Path(base), Path(out)
    ids = pd.read_parquet(ROOT / "data/work/test_source1.parquet", columns=["entity_id", "country"])
    fr_ids = set(ids.entity_id[ids.country == "France"])
    del ids
    
    sub = pd.read_csv(base / "matching_results.tsv", **RD)
    sub = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub.source1_entity_id, sub.matched_entity_ids)}
    
    flips = get_flips()
    new, added = merge(pred, flips)
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
                
    need = {(a, b) for a, b in zip(added.s1, added.sx)}
    with open(base / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t", 1)
            if s in chg:
                need -= {(s, x) for x in c.split(",")}
    assert not need, f"{len(need)} added pairs are not candidates"
    for name in ("candidate_pairs.tsv", "code.zip"):
        if not (out / name).exists() and (base / name).exists():
            os.link(base / name, out / name)
            
    info = {"base": base.name, "flips": len(flips), "added": len(added), "s1_changed": n,
            "runtime_s": round(time.time() - t0)}
    (out / "p50_info.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)

def selftest():
    pred = {"S1-a": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"], "S1-b": [], "S1-c": ["S3-9"]}
    add = pd.DataFrame({"s1": ["S1-a", "S1-a", "S1-b", "S1-c", "S1-b"], "sx": ["S2-6", "S3-7", "S2-8", "S3-9", "S2-5"],
                        "p": [0.9, 0.5, 0.3, 0.8, 0.2]})
    new, rows = merge(pred, add)
    assert new["S1-a"] == ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S3-7"]
    assert new["S1-b"] == ["S2-8"] and new["S1-c"] == ["S3-9"]
    assert list(rows.sx) == ["S3-7", "S2-8"]
    print("selftest ok")

if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "selftest":
        selftest()
    elif cmd == "build":
        build(*args[:2])
