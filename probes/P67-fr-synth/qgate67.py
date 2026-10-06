"""LOCO check of a Qwen gate on the syn-combiner adds: P38's source-only Qwen LoRA r (holdout band files, joined on
ids) on the adds of 'p55n veto + add c > .95' -> precision of adds with Qwen r >= .5 / < .5 / unscored, and the dF of
dropping the r < .5 ones. -> qgate67.json
    .venv/bin/python probes/P67-fr-synth/qgate67.py p67-bge-loco-0927-1405"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import ev67  # noqa: E402
import p55  # noqa: E402

jobs = sys.argv[1:]
Q = {"US_India": "r_qwen_locoUS_holdout_India", "India_US": "r_qwen_locoIndia_holdout_US"}   # frame <- source LoRA
out = {}
for name, trn, tst in p55.DIRN:
    a, b = p55.loco_frame(trn), p55.loco_frame(tst)
    ca, cb = (pd.DataFrame(ev67.columns("bge", jobs, dr, len(fr["X"]))) for dr, fr in ((trn, a), (tst, b)))
    c = p55.bag_score(p55.bag(pd.concat([a["X"], ca[["r_syn"]]], axis=1), a["z"]["y"], None),
                      pd.concat([b["X"], cb[["r_syn"]]], axis=1), None)
    c0 = p55.bag_score(p55.bag(a["X"], a["z"]["y"], None), b["X"], None)
    st, rg = b["X"]["stage"].to_numpy(), b["X"]["region"].to_numpy()
    dec = b["ref"] & ~(b["ref"] & (rg == 1) & (st != 2) & (c0 < 0.6))
    new = p55.decide({**b, "ref": dec}, c, None, 0.95)
    add = new & ~dec
    ids = pd.read_parquet(ROOT / f"data/work/p48/loco_{tst}_ids.parquet", columns=["s1", "sx"])
    q = pd.read_parquet(HERE.parents[1] / f"data/work/p67/q38/{Q[tst]}.parquet")
    r = ids.merge(q, on=["s1", "sx"], how="left").r.to_numpy()
    assert len(r) == len(add)
    y = b["z"]["y"]
    grp = {"q>=.5": add & (r >= .5), "q<.5": add & (r < .5), "unscored": add & np.isnan(r)}
    fb = p55.L.F(b["z"], dec)
    out[name] = {k: {"pairs": int(m.sum()), "prec": round(float(y[m].mean()), 3) if m.any() else None} for k, m in grp.items()}
    out[name]["all_adds"] = p55.L.cmp(b["z"], dec, new, fb)
    out[name]["qgated_adds"] = p55.L.cmp(b["z"], dec, new & ~grp["q<.5"], fb)
    print(name, out[name], flush=True)
    del a, b
(HERE / "qgate67.json").write_text(json.dumps(out, indent=1))
