import sys, os, csv, time, json
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P28-addr-anchor"), str(ROOT / "probes/N63-floor-probes")]

W = ROOT / "data/work"
P31_FR = W / "p31/pairs_France.parquet"
CAP = {"S2": 5, "S3": 6}

def merge(pred: dict, add: pd.DataFrame) -> tuple:
    owned = {x for v in pred.values() for x in v}
    new, rows = {k: list(v) for k, v in pred.items()}, []
    for s1, sx, p in add.sort_values("p", ascending=False)[["s1", "sx", "p"]].itertuples(index=False):
        if sx in owned: continue
        cur = new.setdefault(s1, [])
        if sum(x.startswith(sx[:2]) for x in cur) >= CAP[sx[:2]]: continue
        cur.append(sx)
        owned.add(sx)
        rows.append((s1, sx, p))
    return new, pd.DataFrame(rows, columns=["s1", "sx", "p"])

def build(base_dir, out_dir, tier="R0_p0.3"):
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    print(f"Building {tier} into {out_dir} from {base_dir}")
    
    fr_ids = set(pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"])
                 .query("country == 'France'").entity_id)
    
    sub = pd.read_csv(base_dir / "matching_results.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    sub_fr = sub[sub.source1_entity_id.isin(fr_ids)]
    pred = {s: (m.split(",") if m else []) for s, m in zip(sub_fr.source1_entity_id, sub_fr.matched_entity_ids)}
    
    sx_to_acc_s1 = {x: s for s, xs in pred.items() for x in xs}
    
    sx_df = pd.read_parquet(W / "rf_test_sx_France.parquet", columns=["id", "core", "hn", "city"])
    sx_r0 = sx_df[sx_df.core != ''].copy()
    sx_r0['core'] = sx_r0['core'].fillna('')
    sx_r0['hn'] = sx_r0['hn'].fillna('')
    sx_r0['city'] = sx_r0['city'].fillna('')
    sx_r0['acc_s1'] = sx_r0.id.map(sx_to_acc_s1)
    
    g = sx_r0.dropna(subset=['acc_s1']).groupby(['core', 'hn', 'city'])['acc_s1'].unique()
    owner_groups = g[g.str.len() == 1].apply(lambda x: x[0]).reset_index().rename(columns={'acc_s1': 'ow'})
    sx_r0 = sx_r0.merge(owner_groups, on=['core', 'hn', 'city'], how='inner')
    owner_map_r0 = dict(zip(sx_r0.id, sx_r0.ow))
    khn_map = dict(zip(sx_r0.id, sx_r0.hn))
    
    d = pd.read_parquet(P31_FR, columns=["s1", "sx", "p", "argmax", "hn_cls", "hn_1", "hn_x"])
    d = d[d.argmax].copy()
    
    d['free'] = ~d.sx.isin(sx_to_acc_s1)
    d['ow'] = d.sx.map(owner_map_r0)
    d['is_r0'] = (d.s1 == d.ow) & d.free
    
    cand = d[d.is_r0].copy()
    print(f"R0 raw free cand: {len(cand)}")
    print(f"France pool split by hn_cls:")
    print(cand.hn_cls.value_counts(dropna=False).to_dict())
    
    # master (20:40): the France-matching LOCO cell = key hn non-empty AND SX hn == S1 hn (LOCO .977 / .917, halves > 0 both ways)
    khn = cand.sx.map(khn_map).fillna("")
    hn1, hnx = cand.hn_1.fillna("").astype(str), cand.hn_x.fillna("").astype(str)
    if tier == "R0_hneq": cand = cand[(khn != "") & (hnx == hn1)]
    elif tier == "R0_all": pass
    elif tier == "R0_p0.1": cand = cand[cand.p >= 0.1]
    elif tier == "R0_p0.3": cand = cand[cand.p >= 0.3]
    elif tier == "R0_p0.5": cand = cand[cand.p >= 0.5]
    elif tier == "R0_p0.8": cand = cand[cand.p >= 0.8]
    else: raise ValueError(f"Unknown tier: {tier}")
    
    print(f"{tier} before candidate_pairs filter: {len(cand)}")
    
    cand_file = base_dir / "candidate_pairs.tsv"
    if not cand_file.exists(): cand_file = Path("output/candidate_pairs.tsv")
    valid_pairs = set()
    with open(cand_file, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, c = line.rstrip("\n").split("\t", 1)
            for x in c.split(","): valid_pairs.add((s, x))
            
    cand = cand[[ (s,x) in valid_pairs for s,x in zip(cand.s1, cand.sx) ]]
    print(f"{tier} after candidate_pairs filter: {len(cand)}")
    
    new_pred, added = merge(pred, cand)
    chg = set(added.s1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(base_dir / "matching_results.tsv", encoding="utf-8") as f, \
         open(out_dir / "matching_results.tsv", "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            s = line.split("\t", 1)[0]
            if s in chg:
                g.write(f"{s}\t{','.join(sorted(new_pred[s]))}\n")
                n += 1
            else:
                g.write(line)
                
    for name in ("candidate_pairs.tsv", "code.zip"):
        src = cand_file if name == "candidate_pairs.tsv" else base_dir / name
        if not (out_dir / name).exists() and src.exists():
            os.link(src, out_dir / name)
            
    print(f"Added {len(added)} pairs over {len(chg)} S1s.")

def selftest():
    pred = {"S1-a": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"], "S1-b": [], "S1-c": ["S3-9"]}
    add = pd.DataFrame({"s1": ["S1-a", "S1-a", "S1-b", "S1-c", "S1-b"], "sx": ["S2-6", "S3-7", "S2-8", "S3-9", "S2-5"],
                        "p": [0.9, 0.5, 0.3, 0.8, 0.2]})
    new, rows = merge(pred, add)
    assert new["S1-a"] == ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S3-7"]
    assert new["S1-b"] == ["S2-8"] and new["S1-c"] == ["S3-9"]
    assert list(rows.sx) == ["S3-7", "S2-8"]
    print("selftest ok")

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest': selftest()
    elif cmd == 'build': build(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "R0_all")
