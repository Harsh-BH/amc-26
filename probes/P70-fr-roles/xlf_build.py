"""Lead's P77 q4 on France: address-less SX whose argmax S1 has the same name core (core unique among France S1), full name
differing only by the legal form, France p >= 0.3. Reports accepted vs free per legal-form split (a / b / c), then stages
BASE + the free a|b pairs: P64 veto, caps 5/6 on the base line (best p first), SX unused, candidates only.
    xlf_build.py sub77 sub77_xlf"""
import csv, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P66-fr-noaddr"), str(ROOT / "probes/P68-iu-retrieval"), str(ROOT / "probes/P64-legal-form-feat"),
                str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import loco66 as L6, fr_key, lf
from prof import fkey
W = ROOT / "data/work"
base, out = sys.argv[1], ROOT / "output" / sys.argv[2]
d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["s1", "sx", "p"]).sort_values("p", ascending=False).drop_duplicates("sx")
d = d[d.sx.isin(set(L6.addrless_ids("test", "France"))) & (d.p >= 0.3)].reset_index(drop=True)
core = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "core"]).set_index("id").core
cx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "core"]).set_index("id").core
d["c1"], d["cx"] = core.reindex(d.s1).fillna("").to_numpy(), cx.reindex(d.sx).fillna("").to_numpy()
d = d[(d.c1 == d.cx) & (d.c1 != "") & (d.c1.map(core.value_counts()) == 1)].reset_index(drop=True)
n1, nx = fr_key.raw_names("test", set(d.s1), [1]).reindex(d.s1), fr_key.raw_names("test", set(d.sx), [2, 3]).reindex(d.sx)
fe = np.array([fkey(a) == fkey(b) for a, b in zip(n1.fillna(""), nx.fillna(""))])
k = np.asarray(lf.pair_class(pd.Series(fr_key.sigs(n1)), pd.Series(fr_key.sigs(nx))))
d["split"] = np.select([fe, np.isin(k, [2, 3]), k == 1, np.isin(k, [4, 5])], ["full_eq", "a", "b", "c"], "none")
m = {r[0]: r[1] for r in csv.reader(open(ROOT / "output" / base / "matching_results.tsv", encoding="utf-8"), delimiter="\t")}
m.pop("source1_entity_id")
fr = set(core.index)
pred = {s: (v.split(",") if v else []) for s, v in m.items() if s in fr}
acc = {(s, x) for s, v in pred.items() for x in v}
used = {x for v in pred.values() for x in v}
d["accepted"] = [(a, b) in acc for a, b in zip(d.s1, d.sx)]
d["free"] = ~d.sx.isin(used)
rep = {sp: {"accepted": int((g.accepted).sum()), "free": int(g.free.sum()), "sx_owned_elsewhere": int((~g.accepted & ~g.free).sum()),
            "free_share": round(float(g.free.mean()), 3)} for sp, g in d.groupby("split")}
print(json.dumps(rep))
j = d[d.free & d.split.isin(["a", "b"])].reset_index(drop=True)
cand = {}
with open(ROOT / "output" / base / "candidate_pairs.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s, v = line.rstrip("\n").split("\t", 1)
        if s in fr and v:
            cand[s] = set(v.split(","))
ok = np.array([x in cand.get(s, ()) for s, x in zip(j.s1, j.sx)])
j = fr_key.capped(j[ok].sort_values("p", ascending=False).reset_index(drop=True), pred)
info = {"base": base, "by_split": rep, "pool_ab_free": int((d.free & d.split.isin(["a", "b"])).sum()), "not_candidate": int((~ok).sum()),
        "added": len(j), "added_by_split": j.split.value_counts().to_dict(), "s1_changed": int(j.s1.nunique()),
        "fills_empty": int(sum(1 for s in j.s1 if not pred.get(s))), "added_p_quartiles": j.p.quantile([0, .25, .5, .75, 1]).round(3).tolist()}
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
(out / "xlf_info.json").write_text(json.dumps(info, indent=1))
j[["s1", "sx", "p", "split"]].assign(action="add").to_parquet(W / f"p70/changes_{out.name}.parquet", index=False)
print(json.dumps(info))
