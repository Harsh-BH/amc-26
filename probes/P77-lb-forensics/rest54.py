"""P77: precision of sub54's France adds that sub76 does NOT carry, from two LB equations (gc70 -> 54 adds them, 54 -> 66
removes them), taking the adds sub76 re-added as true (66 -> 76 was ~ perfect) and other movers at stated priors.
    .venv/bin/python probes/P77-lb-forensics/rest54.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
N = fit.N
d = pd.read_parquet(HERE / "work/trans_pairs.parquet")
u = pd.read_parquet(HERE / "work/union.parquet").set_index("key")["mask"].astype(np.int64)
names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
lb = dict(json.loads((HERE / "work/files.json").read_text()))
d["m"] = u.reindex(d.key).to_numpy().astype(np.int64)
d["in76"] = ((d.m.to_numpy() >> names.index("sub76")) & 1).astype(bool)
d["in54"] = ((d.m.to_numpy() >> names.index("sub54")) & 1).astype(bool)
d["in37"] = ((d.m.to_numpy() >> names.index("sub37_gc70")) & 1).astype(bool)
res = {}
for iu in (0.0005, 0.00072, 0.0009):
    r = {}
    # gc70 -> 54: France adds split into (in76: true) and rest (pi unknown); France removals (P48/P55 vetoes) at pi_rm
    for pi_rm in (0.2, 0.5):
        g = d[(d.k == "sub37_gc70>sub54") & (d.c == "France")]
        good, rest, rm = g[(g.sign > 0) & g.in76], g[(g.sign > 0) & ~g.in76], g[g.sign < 0]
        V = N * (lb["sub54"] - lb["sub37_gc70"] - iu) - good.a.sum() - (rm.b - pi_rm * (rm.a + rm.b)).sum()
        pi1 = (V + rest.b.sum()) / (rest.a + rest.b).sum()
        # 54 -> 66: France removals split (in76 -> true, their removal costs a) vs rest; adds (xfc + un-vetoes) at pi_ad
        h = d[(d.k == "sub54>sub66_final") & (d.c == "France")]
        hg, hr, ha = h[(h.sign < 0) & h.in76], h[(h.sign < 0) & ~h.in76], h[h.sign > 0]
        V2 = N * (lb["sub66_final"] - lb["sub54"]) + hg.a.sum() - (0.9 * (ha.a + ha.b) - ha.b).sum()
        # removing rest pairs: value = b - pi (a + b)
        pi2 = (hr.b.sum() - V2) / (hr.a + hr.b).sum()
        r[f"pi_rm{pi_rm}"] = {"gc70>54 rest pi": round(float(pi1), 3), "n_rest": len(rest), "n_good": len(good),
                              "54>66 rest pi (adds pi .9)": round(float(pi2), 3), "n_rest_rm": len(hr), "n_good_rm": len(hg), "n_add": len(ha),
                              "breakeven": round(float(rest.b.sum() / (rest.a + rest.b).sum()), 3)}
    res[f"iu{iu}"] = r
    print(iu, json.dumps(r))
(HERE / "work/rest54.json").write_text(json.dumps(res, indent=1))
