"""d F0.5 vs #7 (t 0.71) by threshold, on both halves, for saved stackers (stack_<tag>_<c>.txt). Aggregates only.
    .venv/bin/python probes/P17-hn-stack/tcurve.py sup sup-swap brc
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as S  # noqa: E402

TS = np.round(np.arange(0.55, 0.91, 0.01), 2)
s1, t = S.holdout_s1()
out = {}
for c in ("India", "US"):
    d = S.frame("holdout", c)
    pos, y, ntrue, hv = S.labels(d, c, s1, t)
    fb = S.f05(S.M.argmax_sx(d["qx"], d["p"]), d["p"], S.T7, pos, y, ntrue)
    for tag in sys.argv[1:]:
        X = S.X_of(d, tag.split("-")[0])
        s = S.stacked(lgb.Booster(model_file=str(S.OUT / f"stack_{tag}_{c}.txt")), X)
        am = S.M.argmax_sx(d["qx"], s)
        cur = np.array([S.f05(am, s, t_, pos, y, ntrue) - fb for t_ in TS])       # [t, S1]
        out[f"{tag}/{c}"] = {h: {f"{t_:.2f}": round(v, 5) for t_, v in zip(TS, cur[:, hv == h].mean(1))} for h in (0, 1)}
        for h in (0, 1):
            v = cur[:, hv == h].mean(1)
            sm = np.convolve(v, np.ones(5) / 5, "same")
            print(f"{tag} {c} half {h}: best t {TS[v.argmax()]} d {v.max():+.5f}; smoothed-best t {TS[2:-2][sm[2:-2].argmax()]}; "
                  + " ".join(f"{t_:.2f}:{x * 1e4:+.1f}" for t_, x in zip(TS[::3], v[::3])), flush=True)
        del X, s, am, cur
    del d
(S.OUT / "tcurve.json").write_text(json.dumps(out, indent=1))
