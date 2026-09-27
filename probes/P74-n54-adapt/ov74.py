"""P74: N54 vs P70 (word-role combiner veto / adds, data/work/p70/changes.parquet) and P71 (noise-word tier adds,
probes/P71-fr-noiseword/work/adds_*.parquet): the removed pairs they also veto, and the ADDS N54 would drop (the
conflicts: france.decoy_mask on the add pairs, lens on sub66_final's France argmax pairs, as run.py applies it).
    .venv/bin/python probes/P74-n54-adapt/ov74.py probes/P74-n54-adapt/work/n54 [output/<sub> ...]  -> ov74.json
(each output dir: its France adds / removes vs sub66_final join the tested sets)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, "code/business_entity_resolution/src")
import france as FR  # noqa: E402
import features as F  # noqa: E402

out = Path(sys.argv[1])
FR.C = "France"
ka = np.load(out / "p74_frame.npz")["ka"]
id1, idx = F.ids("test", "France")
ids = lambda a, q: a.take(pa.array(q)).to_numpy(zero_copy_only=False)
L1, LX = ids(id1, ka >> 32), ids(idx, ka & 0xFFFFFFFF)
rm = pd.read_parquet(out / "removed.parquet")
rmk = set(zip(rm.s1, rm.sx))
sets = {}
for f in sorted(Path("probes/P71-fr-noiseword/work").glob("adds_*.parquet")):
    sets[f"P71 {f.stem}"] = pd.read_parquet(f, columns=["s1", "sx"]).assign(action="add")
p70 = Path("data/work/p70/changes.parquet")
if p70.exists():
    c = pd.read_parquet(p70)
    for a, g in c.groupby("action"):
        sets[f"P70 {a}"] = g[["s1", "sx"]].assign(action=a)
def fr_pairs(path, keep):
    out = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for ln in f:
            k, v = ln.rstrip("\n").split("\t")
            if k in keep:
                out.update((k, t) for t in v.split(",") if t)
    return out


fr_s1 = {ln.split("\t", 1)[0] for ln in open(out / "matching_results.tsv", encoding="utf-8")} - {"source1_entity_id"}
base = fr_pairs("output/sub66_final/matching_results.tsv", fr_s1)
files = {}
for o in sys.argv[2:]:
    got = fr_pairs(Path(o) / "matching_results.tsv", fr_s1)
    add, gone = got - base, base - got
    sets[f"{Path(o).name} adds"] = pd.DataFrame(sorted(add), columns=["s1", "sx"]).assign(action="add")
    files[Path(o).name] = {"france_adds": len(add), "france_removes": len(gone), "n54_removed_still_in_file": len(rmk & got),
                           "n54_removed_also_removed_by_file": len(rmk & gone)}
    print(Path(o).name, files[Path(o).name], flush=True)
    del got
res = {"files": files}
n, allx = len(L1), pd.concat([x.assign(set=k) for k, x in sets.items()], ignore_index=True)
s1, sx = np.concatenate([L1, allx.s1.to_numpy(object)]), np.concatenate([LX, allx.sx.to_numpy(object)])
lens = np.arange(len(s1)) < n
allx["hit"] = FR.decoy_mask(s1, sx, lens, ~lens, 0.25)[n:]           # one lens, every set's pairs tested
allx["in_rm"] = [k in rmk for k in zip(allx.s1, allx.sx)]
h = allx[allx.hit]
r1 = FR._rf(FR.WORK / "rf_test_s1_France.parquet", pd.unique(h.s1), ["core"])
rx = FR._rf(FR.WORK / "rf_test_sx_France.parquet", pd.unique(h.sx), ["core", "is_web"])
allx.loc[allx.hit, "w"] = [FR.name_rel(a, b, q)[1] for a, b, q in zip(r1.core.reindex(h.s1).fillna(""),
                           rx.core.reindex(h.sx).fillna(""), rx.is_web.reindex(h.sx).fillna(False))]
for name, x in allx.groupby("set", sort=False):
    res[name] = {"pairs": int(len(x)), "n54_would_drop": int(x.hit.sum()), "in_n54_removed": int(x.in_rm.sum()),
                 "words_dropped": x.w.dropna().value_counts().head(10).to_dict()}
    print(name, res[name], flush=True)
(Path("probes/P74-n54-adapt") / "ov74.json").write_text(json.dumps(res, indent=1))
