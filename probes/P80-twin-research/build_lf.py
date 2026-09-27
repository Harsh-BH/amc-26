"""R1 France add-set: address-less SX that KEEPS a legal form, full name == exactly one France S1 (E), E's name core shared by
2-3 France S1, SX unassigned in BASE, E already a candidate. Guard: skip when a same-core sibling's legal form becomes E's by one
dropped letter (SAS->SA, SASU->SAS, ...; the generator drops letters). Writes output/<OUT>/matching_results.tsv (add lines only),
to graft with probes/P75-combine/combine.py BASE2 OUTFULL +<OUT> (caps 5/6, one owner).  Needs frcount.py's r1_pool.parquet.
Train analogue (lfdir.py, India/US truth): owner = E 0.94-0.98 (core 2), 0.91-0.96 (core 3)."""
import re, sys, unicodedata, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
POOL, OUT = sys.argv[1], Path("output") / sys.argv[2]
LF = re.compile(r"\b(?:sarl|sas|sasu|sa|eurl|sci|ei|snc|scop|selarl)\b")
def one_drop(a, b):  # b obtainable from a by deleting one letter
    return len(a) == len(b) + 1 and any(a[:i] + a[i + 1:] == b for i in range(len(a)))
d = pd.read_parquet(POOL)
r1 = pq.read_table("data/work/rf_test_s1_France.parquet", columns=["id", "core"]).to_pandas()
core_of = dict(zip(r1.id, r1.core)); sib = r1.groupby("core").id.apply(list).to_dict()
t = pq.read_table("data/work/test_source1.parquet", columns=["entity_id", "business_name"])
def key(s):  # = frame.key
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    return " ".join(re.findall(r"\w+", s))
need = {s for E in d.E for s in sib[core_of[E]]}
name = {e: n for e, n in zip(t["entity_id"].to_pylist(), t["business_name"].to_pylist()) if e in need}
def risky(E):
    le = set(LF.findall(key(name[E])))
    return any(one_drop(a, b) for s in sib[core_of[E]] if s != E for a in LF.findall(key(name[s])) for b in le)
d["risky"] = [risky(E) for E in d.E]
a = d[~d.risky]
print(f"pool {len(d):,}; letter-drop risky {int(d.risky.sum()):,}; adds {len(a):,} on {a.E.nunique():,} S1; S2 {int((a.src == 2).sum())} S3 {int((a.src == 3).sum())}")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for E, g in a.groupby("E"): f.write(f"{E}\t{','.join(g.sx)}\n")
