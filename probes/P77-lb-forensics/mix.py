"""P77: class mix of the France change sets the LB rewarded vs punished, plus overlap of sub76's adds with sub54's adds.
    .venv/bin/python probes/P77-lb-forensics/mix.py -> work/mix.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ana  # noqa: E402

d = ana.load()
fr = d[d.c == "France"].copy()
fr["cls"] = fr.hng + "/" + fr.nmg + "/" + fr.role
names = [n for n, _ in json.loads((HERE / "work/files.json").read_text())]
bit = lambda n: (fr["mask"].astype(np.int64) >> names.index(n)) & 1 if "mask" in fr else None
u = pd.read_parquet(HERE / "work/union.parquet").set_index("key")["mask"].astype(np.int64)
fr["m"] = u.reindex(fr.key).to_numpy().astype(np.int64)
inf = lambda n: ((fr.m.to_numpy() >> names.index(n)) & 1).astype(bool)
fr["in54"], fr["in66"], fr["in76"], fr["in37"] = inf("sub54"), inf("sub66_final"), inf("sub76"), inf("sub37_gc70")
sets = {"gc70>54 adds (punished)": (fr.k == "sub37_gc70>sub54") & (fr.sign > 0),
        "54>66 removes (rewarded)": (fr.k == "sub54>sub66_final") & (fr.sign < 0),
        "r90>gc70 adds (rewarded)": (fr.k == "sub34v_rescue90>sub37_gc70") & (fr.sign > 0),
        "9ab>31 adds (rewarded)": (fr.k == "sub9ab>sub31") & (fr.sign > 0),
        "66>77 adds (rewarded)": (fr.k == "sub66_final>sub77") & (fr.sign > 0),
        "66>77 N54 removes (rewarded)": (fr.k == "sub66_final>sub77") & (fr.sign < 0)}
out = {}
for nm, m in sets.items():
    g = fr[m].drop_duplicates("key")
    top = g.cls.value_counts().head(8)
    out[nm] = {"n": len(g), "pi0": round(float(g.pi0.mean()), 3), "hn_eq": round(float((g.hng == "eq").mean()), 3),
               "hn_miss": round(float((g.hng == "miss").mean()), 3), "sib": round(float((g.hng == "sib").mean()), 3),
               "pm12_other": round(float(g.hng.isin(["pm12", "other"]).mean()), 3), "name_same": round(float((g.nmg == "same").mean()), 3),
               "noise": round(float((g.role == "noise").mean()), 3), "mid": round(float((g.role == "mid").mean()), 3),
               "decoy": round(float((g.role == "decoy").mean()), 3), "noaddr": round(float(g.noaddr_x.mean()), 3),
               "gib": round(float(g.gib_x.mean()), 3), "fill": round(float((g.ctx == "fill").mean()), 3),
               "top_classes": {k: [int(v), round(float(g.pi0[g.cls == k].mean()), 3)] for k, v in top.items()}}
    print(nm, json.dumps(out[nm]))
a76 = fr[(fr.k == "sub66_final>sub77") & (fr.sign > 0)]
out["overlap"] = {"76_adds_that_were_in_54": int(a76.in54.sum()), "76_adds": len(a76),
                  "54_adds_still_out_of_76": int(((fr.k == "sub37_gc70>sub54") & (fr.sign > 0) & ~fr.in76).sum())}
g = fr[(fr.k == "sub37_gc70>sub54") & (fr.sign > 0) & ~fr.in76].drop_duplicates("key")
out["54_adds_not_in_76_classes"] = {k: [int(v), round(float(g.pi0[g.cls == k].mean()), 3)] for k, v in g.cls.value_counts().head(10).items()}
g2 = fr[(fr.k == "sub37_gc70>sub54") & (fr.sign > 0) & fr.in76].drop_duplicates("key")
out["54_adds_in_76_classes"] = {k: [int(v), round(float(g2.pi0[g2.cls == k].mean()), 3)] for k, v in g2.cls.value_counts().head(10).items()}
print(json.dumps({k: out[k] for k in ["overlap", "54_adds_not_in_76_classes", "54_adds_in_76_classes"]}))
(HERE / "work/mix.json").write_text(json.dumps(out, indent=1))
