"""LOCO value of the exact France change: swap P55's band veto (c < .6, rescue stage exempt = p55n) from the base
combiner to base + r_syn; counts per target S1 for scaling to France. -> cell67_<arm>.json
    .venv/bin/python probes/P67-fr-synth/cell67.py bge p67-bge-loco-0927-1405"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import ev67  # noqa: E402
import p55  # noqa: E402

arm, jobs = sys.argv[1], sys.argv[2:]
frames = {dr: p55.loco_frame(dr) for dr in p55.L.DIRS}
cols = {dr: pd.DataFrame(ev67.columns(arm, jobs, dr, len(frames[dr]["X"]))) for dr in frames}
out = {}
for name, trn, tst in p55.DIRN:
    a, b = frames[trn], frames[tst]
    st, rg, y = b["X"]["stage"].to_numpy(), b["X"]["region"].to_numpy(), b["z"]["y"]
    C = {}
    for fs, cs in (("base", []), ("syn", ["r_syn"])):
        Xa = pd.concat([a["X"], cols[trn][cs]], axis=1) if cs else a["X"]
        Xb = pd.concat([b["X"], cols[tst][cs]], axis=1) if cs else b["X"]
        C[fs] = p55.bag_score(p55.bag(Xa, a["z"]["y"], None), Xb, None)
    scope = b["ref"] & (rg == 1) & (st != 2)
    dec = {fs: b["ref"] & ~(scope & (C[fs] < 0.6)) for fs in C}
    fb = p55.L.F(b["z"], dec["base"])
    r = p55.L.cmp(b["z"], dec["base"], dec["syn"], fb)
    n1 = b["n"]
    r |= {"n_s1": int(n1), "ref_pairs": int(b["ref"].sum()), "old_veto": int((scope & (C["base"] < .6)).sum()),
          "new_veto": int((scope & (C["syn"] < .6)).sum()), "changes_per_ref_pair": round((r["add"] + r["rm"]) / b["ref"].sum(), 5)}
    for fs in C:                                       # + P55's add arm (free argmax pairs, c > .95) on top of the veto
        add = p55.decide({**b, "ref": dec[fs]}, C[fs], None, 0.95)
        r[f"veto+add_{fs}"] = p55.L.cmp(b["z"], dec["base"], add, fb)
        r[f"oldveto+add_{fs}"] = p55.L.cmp(b["z"], dec["base"], p55.decide({**b, "ref": dec["base"]}, C[fs], None, 0.95), fb)
    out[name] = r
    print(name, r, flush=True)
(HERE / f"cell67_{arm}.json").write_text(json.dumps(out, indent=1))
