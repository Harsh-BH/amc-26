"""P77: P71 lens word (kind:word, s) of sub54's adds that sub76 does not carry, and of each 66 -> 76 component.
    .venv/bin/python probes/P77-lb-forensics/words54.py -> work/words54.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from comp76 import enc  # noqa: E402
from collect import keys  # noqa: E402

pool = pd.read_parquet(ROOT / "probes/P71-fr-noiseword/work/pool_France.parquet")
pool["key"] = enc(pool.s1, pool.sx)
pool["w"] = np.where(pool.kind == "sub", "sub:" + pool.ins.astype(str), np.where(pool.kind == "ins", "ins:" + pool.ins.astype(str), pool.kind + ":" + pool["drop"].astype(str)))
lens = pd.read_csv(ROOT / "probes/P71-fr-noiseword/work/lens_France.csv").set_index("k")
pool["s"] = pool.w.map(lens.s)
d = pd.read_parquet(HERE / "work/trans_pairs.parquet")
u = pd.read_parquet(HERE / "work/union.parquet").set_index("key")["mask"].astype(np.int64)
names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
m = u.reindex(d.key).to_numpy().astype(np.int64)
d["in76"] = ((m >> names.index("sub76")) & 1).astype(bool)
rest = d[(d.k == "sub37_gc70>sub54") & (d.c == "France") & (d.sign > 0) & ~d.in76]
good = d[(d.k == "sub37_gc70>sub54") & (d.c == "France") & (d.sign > 0) & d.in76]
out = {}
for nm, g in [("54rest", rest), ("54good", good)]:
    j = g.merge(pool[["key", "w", "s"]], on="key", how="left")
    out[nm] = {"n": len(g), "in_p71_pool": int(j.w.notna().sum()), "top_words": {k: int(v) for k, v in j.w.value_counts().head(12).items()}}
    print(nm, json.dumps(out[nm]))
# sub71b's extra (groupe) adds vs sub71a
k71a, k71b = keys(ROOT / "output/sub71a/matching_results.tsv"), keys(ROOT / "output/sub71b/matching_results.tsv")
extra = np.setdiff1d(k71b, k71a)
out["71b_extra"] = {"n": len(extra), "in_54rest": int(np.isin(extra, rest.key.to_numpy()).sum()), "in_54good": int(np.isin(extra, good.key.to_numpy()).sum()),
                    "in_sub76": int(np.isin(extra, u.index[((u.to_numpy() >> names.index("sub76")) & 1) == 1]).sum())}
print(json.dumps(out["71b_extra"]))
(HERE / "work/words54.json").write_text(json.dumps(out, indent=1))
np.save(HERE / "work/k71b_extra.npy", extra)
