"""N69A build: base matching file + France address-less adds.
Rule (France / unseen-country rows only): a free SX (owned by no S1 in the base) with an EMPTY/NULL address, whose argmax S1
(P31 France frame) has the same name core (name_rel eq, or a typo of it), that core is carried by exactly ONE France S1,
the pair is not a legal-form conflict (P64 class 4), France p >= 0.3; caps 5 S2 / 6 S3 on the base line, best p first.
Why: train truth — address-less SX are 97.7% true (decoys keep an address) and a unique exact core names the owner 96-97%;
LOCO US->India +0.00065 France-F (11 SE, prec .95). France's leftover shape (82% of such pairs accepted) matches US->India.
    .venv/bin/python probes/N69-noaddr-owner/build.py <base matching_results.tsv> <out_dir> [candidate_pairs.tsv]
"""
import sys, os, csv, numpy as np, pandas as pd
CAP = {"S2": 5, "S3": 6}
RD = dict(sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
base, out = sys.argv[1], sys.argv[2]; cand = sys.argv[3] if len(sys.argv) > 3 else None
f = pd.read_parquet("data/work/n69/fr_noaddr.parquet")
m = pd.read_csv(base, **RD); m.columns = ["s1", "ms"]
own = {x for ms in m.ms if ms for x in ms.split(",")}
sel = f[f.name_rel.isin(["eq", "typo"]) & (f.ncore == 1) & (f.lfc != 4) & (f.p >= 0.3) & ~f.sx.isin(own)].sort_values("p", ascending=False)
assert sel.sx.is_unique
fr = set(pd.read_parquet("data/work/rf_test_s1_France.parquet", columns=["id"]).id)
assert sel.s1.isin(fr).all()
cnt = {}
for s, ms in zip(m.s1, m.ms):
    for x in (ms.split(",") if ms else []): cnt[(s, x[:2])] = cnt.get((s, x[:2]), 0) + 1
keep = []
for s, x in zip(sel.s1, sel.sx):
    k = (s, x[:2])
    if cnt.get(k, 0) < CAP[x[:2]]: keep.append(True); cnt[k] = cnt.get(k, 0) + 1
    else: keep.append(False)
add = sel[keep]
adds = add.groupby("s1").sx.agg(list).to_dict()
m["ms"] = [",".join(([*ms.split(",")] if ms else []) + adds.get(s, [])) for s, ms in zip(m.s1, m.ms)]
os.makedirs(out, exist_ok=True)
with open(f"{out}/matching_results.tsv", "w") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for s, ms in zip(m.s1, m.ms): fh.write(f"{s}\t{ms}\n")
add[["s1", "sx", "p", "r", "name_rel", "lfc", "name_1", "name_x"]].to_parquet(f"{out}/n69_adds.parquet", index=False)
print(f"adds {len(add):,} on {add.s1.nunique():,} France S1 (capped out {len(sel) - len(add)}; empties filled {sum(1 for s in adds if s in set(m.s1[m.ms == ','.join(adds[s])]))}); "
      f"name_rel {add.name_rel.value_counts().to_dict()}; lfc {add.lfc.value_counts().to_dict()}; p median {add.p.median():.3f}")
if cand:
    c = pd.read_csv(cand, **RD); c.columns = ["s1", "cs"]
    have = {(s, x) for s, cs in zip(c.s1, c.cs) if s in adds for x in cs.split(",") if cs}
    miss = [(s, x) for s, x in zip(add.s1, add.sx) if (s, x) not in have]
    mm = pd.DataFrame(miss, columns=["s1", "sx"]).groupby("s1").sx.agg(list).to_dict()
    c["cs"] = [",".join(([*cs.split(",")] if cs else []) + mm.get(s, [])) for s, cs in zip(c.s1, c.cs)]
    with open(f"{out}/candidate_pairs.tsv", "w") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s, cs in zip(c.s1, c.cs): fh.write(f"{s}\t{cs}\n")
    print(f"candidates: {len(miss)} add pairs appended")
