"""P39: dF0.5 on the LOCO target as a function of the decision threshold tau (both directions, per arm and mode), to
find a tau that is safe under calibration shift. Target scores cached in data/work/p39/s_<arm>_<dir>.npy.

    .venv/bin/python probes/P39-fr-group-model/curve.py <arm> [<arm> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit import ARMS, FR_SHARE, MONO, W, decide, fit, load, per_s1, score  # noqa: E402

TAUS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]


def target_scores(arm, direction):
    f = W / f"s_{arm}_{direction.replace('>', '_')}.npy"
    src, tgt = direction.split(">")
    if not f.exists():
        dtr, _, _ = load(tgt, src)
        dte, _, _ = load(src, tgt)
        np.save(f, score(fit(dtr, ARMS[arm], mono=MONO.get(arm)), dte, ARMS[arm]).astype(np.float32))
    return np.load(f).astype(np.float64)


def curve(arm):
    out = {}
    for dr in ("US>India", "India>US"):
        src, tgt = dr.split(">")
        d, nt, hv = load(src, tgt)
        s = target_scores(arm, dr)
        base = per_s1(d, d.acc.to_numpy(), nt)
        y, acc = d.y.to_numpy(), d.acc.to_numpy()
        for mode in ("add", "gc", "addrm", "full"):
            for t in TAUS:
                for trm in ((0.3, 0.5) if mode == "addrm" else (None,)):
                    pred = decide(d, s, t, mode, trm)
                    dl = per_s1(d, pred, nt) - base
                    a, r = pred & ~acc, acc & ~pred
                    k = f"{mode}{'' if trm is None else trm}"
                    out.setdefault(k, {}).setdefault(t, {})[dr] = (round(float(dl.mean()), 5), round(float(dl.std() / np.sqrt(len(dl))), 5),
                                                                   int(a.sum()), round(float(y[a].mean()), 3) if a.any() else None,
                                                                   int(r.sum()), round(float(y[r].mean()), 3) if r.any() else None)
    print(f"=== {arm}: dF (se) added@prec removed@prec per direction; LB-eq = 0.1498 dF")
    for k, v in out.items():
        for t, x in v.items():
            u, i = x["US>India"], x["India>US"]
            print(f"  {k:<9} tau {t:.2f} | US>India {u[0]:+.5f} ({u[1]:.5f}) +{u[2]:,}@{u[3]} -{u[4]:,}@{u[5]} | "
                  f"India>US {i[0]:+.5f} ({i[1]:.5f}) +{i[2]:,}@{i[3]} -{i[4]:,}@{i[5]} | min LB-eq {min(u[0], i[0]) * FR_SHARE:+.5f}",
                  flush=True)
    json.dump({k: {str(t): x for t, x in v.items()} for k, v in out.items()}, open(W / f"curve_{arm}.json", "w"))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        curve(a)
