"""Measurement candidate: BASE + free France argmax pairs with identical name core, same city, same parsed hn, a different
street (street_fr class 'other'), no hidden sibling shift. P64 legal-form veto, caps 5/6 on the base line (best p first),
SX unused, pair must be in BASE's candidate_pairs.tsv. France lines only; candidates / code.zip hard-linked.
    hnst_build.py sub77 sub77_hnst"""
import csv, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P68-iu-retrieval"), str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import fr_key, lf
W = ROOT / "data/work"
base, out = sys.argv[1], Path(ROOT / "output" / sys.argv[2])
j = pd.read_parquet(W / f"p70/street_pool_{base}.parquet")
j = j[(j.cls == "other") & ~j["shift"]].sort_values("p", ascending=False).reset_index(drop=True)
info = {"base": base, "pool": len(j)}
n1 = fr_key.raw_names("test", set(j.s1), [1]).reindex(j.s1)
nx = fr_key.raw_names("test", set(j.sx), [2, 3]).reindex(j.sx)
k = np.asarray(lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx))))
info["p64_vetoed"] = int((k == 4).sum())
j = j[k != 4].reset_index(drop=True)
m = {r[0]: r[1] for r in csv.reader(open(ROOT / "output" / base / "matching_results.tsv", encoding="utf-8"), delimiter="\t")}
m.pop("source1_entity_id")
fr = set(pd.read_parquet(W / "test_source1.parquet", columns=["entity_id", "country"]).query("country == 'France'").entity_id)
pred = {s: (v.split(",") if v else []) for s, v in m.items() if s in fr}
used = {x for v in pred.values() for x in v}
assert not j.sx.isin(used).any() and not j.sx.duplicated().any()
cand = {}
with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s, v = line.rstrip("\n").split("\t", 1)
        if s in fr and v:
            cand[s] = set(v.split(","))
ok = np.array([x in cand.get(s, ()) for s, x in zip(j.s1, j.sx)])
info["not_candidate"] = int((~ok).sum())
j = fr_key.capped(j[ok].reset_index(drop=True), pred)
info |= {"added": len(j), "s1_changed": int(j.s1.nunique()), "cap_or_other_blocked": int(ok.sum()) - len(j),
         "added_p_quartiles": j.p.quantile([0, .25, .5, .75, 1]).round(3).tolist(),
         "state": {"fills_empty": int(sum(1 for s in j.s1 if not pred.get(s)))}}
addl = j.groupby("s1").sx.apply(list).to_dict()
out.mkdir(parents=True, exist_ok=True)
with open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
    g.write("source1_entity_id\tmatched_entity_ids\n")
    for s, v in m.items():
        if s in addl:
            v = ",".join((v.split(",") if v else []) + addl[s])
        g.write(f"{s}\t{v}\n")
for name in ("candidate_pairs.tsv", "code.zip"):
    if not (out / name).exists() and (ROOT / "output" / base / name).exists():
        os.link(ROOT / "output" / base / name, out / name)
(out / "hnst_info.json").write_text(json.dumps(info, indent=1))
j[["s1", "sx", "p"]].assign(action="add").to_parquet(W / f"p70/changes_{out.name}.parquet", index=False)
print(json.dumps(info))
