"""France: are the corruptor-added words sibling words? Co-occurrence of an added word with the generator's sibling
house-number offset, on argmax pairs of a France probability file (label-free, aggregates only).
    branch_fr.py <prob parquet> <t> [country]
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from anyascii import anyascii

ROOT = Path("/home/harsh1/github-repos/amc-26")
W = ROOT / "data/work"
prob, t = sys.argv[1], float(sys.argv[2])
C = sys.argv[3] if len(sys.argv) > 3 else "France"
WORDS = sys.argv[4].split(",") if len(sys.argv) > 4 else [
    "participations", "holding", "distribution", "international", "developpement", "groupe", "associes", "fils",
    "cie", "freres", "et", "+", "center", "centre", "services", "partners", "holdings", "group", "trading",
    "enterprises", "labs", "sys"]
SIB3 = {3, 4, 5, 7, 9, 11, 13, 21}
SIB12 = {1, 2}

p = pd.read_parquet(prob, columns=["s1", "sx", "p"])
p = p.sort_values("p", ascending=False).drop_duplicates("sx")          # argmax S1 per SX
r1 = pd.read_parquet(W / f"rf_test_s1_{C}.parquet", columns=["id", "hn"]).set_index("id")
rx = pd.read_parquet(W / f"rf_test_sx_{C}.parquet", columns=["id", "hn"]).set_index("id")
names = {}
for src in ("source1", "source2", "source3"):
    tb = ds.dataset(W / f"test_{src}.parquet").to_table(columns=["entity_id", "business_name"], filter=ds.field("country") == C)
    names.update(zip(tb.column(0).to_pylist(), tb.column(1).to_pylist()))
tok = lambda s: set(re.findall(r"[a-z]+|\+", anyascii(s or "").lower()))
p["h1"] = r1.hn.reindex(p.s1).to_numpy()
p["hx"] = rx.hn.reindex(p.sx).to_numpy()
t1 = [tok(names.get(a)) for a in p.s1]
tx = [tok(names.get(b)) for b in p.sx]
added = [b - a for a, b in zip(t1, tx)]


def dcls(h1, hx):
    if not h1 or not hx or not str(h1).isdigit() or not str(hx).isdigit():
        return "none"
    d = int(hx) - int(h1)
    return "eq" if d == 0 else "sib3" if d in SIB3 else "pm12" if abs(d) in SIB12 else "other"


p["dc"] = [dcls(a, b) for a, b in zip(p.h1, p.hx)]
p["n_add"] = [len(a) for a in added]
rows = []
base = p[p.n_add == 0]
for lab, g in [("no-added-token", base), ("any-added", p[p.n_add > 0])] + \
        [(w, p[[w in a for a in added]]) for w in WORDS]:
    if len(g) == 0:
        continue
    vc = g.dc.value_counts(normalize=True)
    rows.append(dict(word=lab, n=len(g), p_med=g.p.median(), acc=(g.p >= t).mean(), n_acc=int((g.p >= t).sum()),
                     eq=vc.get("eq", 0), sib3=vc.get("sib3", 0), pm12=vc.get("pm12", 0), other=vc.get("other", 0),
                     none=vc.get("none", 0)))
pd.set_option("display.width", 200)
print(f"{C} argmax pairs {len(p):,}  t {t}")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# accepted-pair cross-tab (the pairs a rule would actually change) + cache flags for re-use
acc = p.p >= t
print("\naccepted pairs by word x hn class (counts):")
out = []
for w in WORDS:
    m = np.array([w in a for a in added]) & acc.to_numpy()
    if m.sum() == 0:
        continue
    vc = p.dc[m].value_counts()
    out.append(dict(word=w, n_acc=int(m.sum()), eq=vc.get("eq", 0), sib3=vc.get("sib3", 0), pm12=vc.get("pm12", 0),
                    other=vc.get("other", 0), none=vc.get("none", 0)))
print(pd.DataFrame(out).to_string(index=False))
flags = pd.DataFrame({w: [w in a for a in added] for w in WORDS})
flags["s1"], flags["sx"], flags["p"], flags["dc"], flags["n_add"] = p.s1.values, p.sx.values, p.p.values, p.dc.values, p.n_add.values
flags.to_parquet(Path(__file__).parent / f"flags_{C}.parquet", index=False)
