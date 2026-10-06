"""P77: LB-implied precision bounds per component of sub66_final -> sub76 (and gc70, rescue90) under several pricing
states (lam = unfound true members per S1, pi_k = precision of kept pairs), public-share noise at f = 0.5.
Bound for component c: every other component at its best case, c's value >= obs - (max_others) - 2 SE.
    .venv/bin/python probes/P77-lb-forensics/bounds76.py -> work/bounds76.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
N = fit.N

u = pd.read_parquet(HERE / "work/union.parquet")
s1 = u.key.to_numpy() >> 31
_, u["s1i"] = np.unique(s1, return_inverse=True)
files = json.loads((HERE / "work/files.json").read_text())
names, lb = [n for n, _ in files], dict(files)
comp = pd.read_parquet(HERE / "work/comp76.parquet")
out = {}
for pk, lam in [(0.95, 0.35), (0.95, 0.8), (0.9, 1.5), (1.0, 0.0)]:
    fit.PRICING["mc"], fit.PRICING["use"] = fit.mc_table(pk, lam), "mc"
    R = {}
    for A, B in [("sub66_final", "sub77"), ("sub34v_rescue90", "sub37_gc70"), ("sub34_veto", "sub34v_rescue90")]:
        d = fit.frame(u, names, A, B)
        d["key"] = u.key.to_numpy()[d.row]
        obs = N * (lb[B] - lb[A])
        se = np.sqrt((np.where(d.sign > 0, d.a, d.b) ** 2 * 0.25).sum() / 0.5)     # Bernoulli var <= 1/4, f = 0.5
        if B == "sub77":
            d = d.merge(comp, on="key", how="left").fillna(False)
            d["comp"] = np.select([d.sign < 0, d.P67q, d.P70a, d.t975ce, d.P71a, d.P41], ["N54", "P67q", "P70a", "t975ce", "P71a", "P41"], "other")
        else:
            d["comp"] = "all"
        mx = d.groupby("comp").apply(lambda x: np.where(x.sign > 0, x.a, x.b).sum())
        ab = d.groupby("comp").apply(lambda x: (x.a + x.b).sum())
        bb = d.groupby("comp").apply(lambda x: x.b.sum())
        tot = mx.sum()
        r = {"obs_1e6": round(obs / N * 1e6, 1), "max_1e6": round(tot / N * 1e6, 1), "se_1e6": round(se / N * 1e6, 1)}
        for c in mx.index:
            need = obs - 2 * se - (tot - mx[c])          # value c must reach
            if c == "N54":   # removal: value = b - pi (a + b)  ->  pi <= (b - need) / (a + b)
                r[c] = {"n": int((d.comp == c).sum()), "pi_max": round(float(min(1, max(0, (bb[c] - need) / ab[c]))), 3)}
            else:            # add: value = pi (a + b) - b  ->  pi >= (need + b) / (a + b)
                r[c] = {"n": int((d.comp == c).sum()), "pi_min": round(float(min(1, max(0, (need + bb[c]) / ab[c]))), 3)}
        R[f"{A}>{B}"] = r
    out[f"pk{pk}_lam{lam}"] = R
    print(pk, lam, json.dumps(R), flush=True)
(HERE / "work/bounds76.json").write_text(json.dumps(out, indent=1))
