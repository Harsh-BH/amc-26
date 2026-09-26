"""P40: output/sub40 = #37_gc70 + France op rule `xu_noaddr|gib` at France p >= T (rule40.op_class + rule40.add).
India/US lines, candidate_pairs.tsv and code.zip are #37_gc70's (hard links: added pairs come from the France pair file,
i.e. they are already candidates).
    P37_BASE=output/sub37_gc70 probes/P40-generator/py.sh probes/P40-generator/build40.py [RULE@T] [OUT]
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loco41 as L41  # noqa: E402

P = L41.P
rule = sys.argv[1] if len(sys.argv) > 1 else "xu_noaddr|gib@0.9"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "output/sub40")
t0 = time.time()
assert P.BASE.name == "sub37_gc70", P.BASE
import numpy as np  # noqa: E402
import rule40 as R  # noqa: E402
from france import traw  # noqa: E402
assert rule.startswith("xu_noaddr|gib@")
t = float(rule.split("@")[1])
d, a = P.fr_load()
s1, sx, is2, p, am, pred, sib = (a[k] for k in ("s1", "sx", "is2", "p", "am", "pred", "sib"))
pool = am & ~pred & (p >= t)
ids1, idsx = d.s1.to_numpy()[pool], d.sx.to_numpy()[pool]
r1, rx = traw(ids1, (1,)), traw(idsx, (2, 3))
cls = np.zeros(len(p), bool)
cls[pool] = R.op_class(r1.business_name.reindex(ids1).to_numpy(), rx.business_name.reindex(idsx).to_numpy(),
                       rx.business_address.reindex(idsx).to_numpy(), L41.core_of(L41.W / "rf_test_s1_France.parquet", ids1),
                       L41.core_of(L41.W / "rf_test_sx_France.parquet", idsx),
                       L41.uniq_counts(L41.W / "rf_test_s1_France.parquet").to_dict())
m = R.add(s1, sx, is2, p, am, pred, sib, cls, t)
add = d[m][["s1", "sx"]]                              # the added France pairs (rule40 = run.py path)
new = add.groupby("s1").sx.agg(list).to_dict()
out.mkdir(parents=True, exist_ok=True)
n = 0
with open(P.BASE / "matching_results.tsv", encoding="utf-8") as f, open(out / "matching_results.tsv", "w", encoding="utf-8") as g:
    g.write(next(f))
    for line in f:
        s, m = line.rstrip("\n").split("\t", 1)
        if s in new:
            cur = [x for x in m.split(",") if x]
            assert not set(cur) & set(new[s])
            g.write(f"{s}\t{','.join(cur + sorted(new[s]))}\n")
            n += 1
        else:
            g.write(line)
assert n == len(new)
for name in ("candidate_pairs.tsv", "code.zip"):
    if not (out / name).exists():
        os.link(P.BASE / name, out / name)
info = {"base": str(P.BASE), "rule": rule, "added": len(add), "s1_changed": n, "runtime_s": round(time.time() - t0)}
(out / "p40_info.json").write_text(json.dumps(info))
print(json.dumps(info))
