"""P43: france.chain on P42's cached France frame (data/work/p42/france.npz, the post-rules row order) vs staged France
lines, as sets per S1. Fast check of the port before a full run.py run.
    cd ../amc-26-wt/p43-final-package && flock /tmp/p43-heavy.lock .venv/bin/python \
        /home/harsh1/github-repos/amc-26/probes/P43-final-package/chaincheck.py [nop55|full]
nop55: vs output/sub_clean_fr (P38 clean: recipe -> brand -> P48 -> P54 -> P47); full: vs output/sub_clean (+ P55; the
staged file also has P54's 49 France duplicate copies, which this check does not apply); s54: vs output/sub54 (its order;
32 France duplicate copies not applied)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
import france as FR  # noqa: E402
import features as F  # noqa: E402

t0 = time.time()
mode = sys.argv[1] if len(sys.argv) > 1 else "nop55"
W = Path("data/work")
FR.C = "France"
z = np.load(W / "p42/france.npz")
i1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id"]).id.to_numpy()
ix = F.ids("test", "France")[1].to_numpy(zero_copy_only=False)
s1, sx = i1[z["s1"]], ix[z["sx"]]
bl = pd.read_parquet(W / "p31/prob_test_France_chn_fix_bge.parquet", columns=["s1", "sx", "p"])
assert (bl.s1.to_numpy() == s1).all() and (bl.sx.to_numpy() == sx).all()
lo = pd.read_parquet(W / "p48/r_lo_France.parquet")
rlo = pd.DataFrame({"s1": s1, "sx": sx}).merge(lo, on=["s1", "sx"], how="left").r.to_numpy(np.float64)
h1 = pd.read_parquet(W / "rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(s1).fillna("").to_numpy(object)
hx = pd.read_parquet(W / "p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn.reindex(sx).fillna("").to_numpy(object)
hr = np.load(W / "p52/france_hn_rel.npy")
print("hn_rel == P52's file:", bool((FR.hn_rel(h1, hx) == hr).all()), flush=True)
f = {"q1": z["s1"].astype(np.int64), "qx": z["sx"].astype(np.int64), "is2": z["is2"], "p": z["p10"], "p0": z["p0"],
     "g": z["g"], "pbl": bl.p.to_numpy(np.float64), "r": z["r"], "r_lo": rlo, "h1": h1, "hx": hx, "s1": s1, "sx": sx,
     "n": int(z["n_s1"]), "v": W / "p9/fr31"}
del bl
c = dict(t=0.99, anchor=0.9868, rescue=0.90, gc=0.70, hn=0.80, brand=0.30, veto=0.20, u=0.99, cls=0.80,
         p55=(0.6, 0.95, [W / f"p55/fr_plain_s{i}.txt" for i in range(3)]) if mode == "full" else None)
if mode == "s54":
    c |= dict(xf=0.70, swap=0.05, order=FR.ORDER_54)
    f["learn"] = pd.read_parquet(W / "p3/cands_test_France.parquet", columns=["s1", "sx"])
acc = FR.chain(f, c)
mine = pd.Series(sx[acc]).groupby(s1[acc]).agg(lambda v: frozenset(v)).to_dict()
ref = Path("/home/harsh1/github-repos/amc-26/output") / {"full": "sub_clean", "nop55": "sub_clean_fr", "s54": "sub54"}[mode] / \
    "matching_results.tsv"
fr = set(i1)
diff = extra = missing = 0
ex = []
with open(ref, encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        k, v = line.rstrip("\n").split("\t")
        if k in fr:
            a, b = mine.get(k, frozenset()), frozenset(v.split(",")) - {""}
            if a != b:
                diff += 1
                extra += len(a - b)
                missing += len(b - a)
                if len(ex) < 5:
                    ex.append((len(a - b), len(b - a)))
print({"mode": mode, "france_s1_differ": diff, "pairs_only_mine": extra, "pairs_only_staged": missing, "examples": ex,
       "accepted": int(acc.sum()), "runtime_s": round(time.time() - t0)}, flush=True)
