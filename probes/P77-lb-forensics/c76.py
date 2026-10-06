"""P77: sub66_final -> sub76 change set by component and label-free class (value units 1e-6 LB).
    .venv/bin/python probes/P77-lb-forensics/c76.py -> work/c76.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ana  # noqa: E402
N = 1_732_544

d = ana.load()
g = d[d.k == "sub66_final>sub77"].merge(pd.read_parquet(HERE / "work/comp76.parquet"), on="key", how="left")
C = ["P67q", "P70a", "t975ce", "P71a", "P41", "N54"]
g[C] = g[C].fillna(False)
# one owner per pair, in sub76's combine order (P67q, P70a, t975ce, P71a; N54 removals)
g["comp"] = np.select([g.sign < 0, g.P67q, g.P70a, g.t975ce, g.P71a, g.P41], ["N54", "P67q", "P70a", "t975ce", "P71a", "P41"], "other")
agg = lambda x: pd.Series({"n": len(x), "s1": x.s1i.nunique(), "fill": int((x.ctx == "fill").sum()), "empty": int((x.ctx == "empty").sum()),
                           "a_mean": x.a.mean(), "b_mean": x.b.mean(), "pi0": x.pi0.mean(), "rq_cov": x.rq.notna().mean(),
                           "max_1e6": np.where(x.sign > 0, x.a, x.b).sum() / N * 1e6, "prior_1e6": (x.sign * (x.pi0 * (x.a + x.b) - x.b)).sum() / N * 1e6,
                           "hn_eq": (x.hng == "eq").mean(), "hn_miss": (x.hng == "miss").mean(), "hn_sib": (x.hng == "sib").mean(),
                           "nm_same": (x.nmg == "same").mean(), "nm_word": (x.nmg == "word").mean(), "noaddr": x.noaddr_x.mean(),
                           "noise": (x.role == "noise").mean(), "decoy": (x.role == "decoy").mean(), "lf_same": (x.lf_k == 1).mean(),
                           "twins1": x.twins1.median(), "is3": x.is3.mean()})
t = g.groupby("comp").apply(agg).round(3)
print(t.T.to_string())
tot = agg(g).round(3)
print("TOTAL", tot.to_dict())
(HERE / "work/c76.json").write_text(json.dumps({"by_comp": t.T.to_dict(), "total": tot.to_dict()}, indent=1, default=float))
