"""P69: run.py's sub66_final France stage alone at a given France t, instrumented. Writes the France lines
(matching + candidates) to OUT and a compact frame (every France pair + per-stage keys) to OUT/p69_frame.npz.
    flock /tmp/p69.lock .venv/bin/python probes/P69-fr-loosen/p69.py 0.9868 probes/P69-fr-loosen/work/t9868
India / US are not computed: splice.py pastes these lines into sub66_final's file."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
import run as R  # noqa: E402
import france as FR  # noqa: E402

t, out = float(sys.argv[1]), Path(sys.argv[2])
R.use("sub66_final")
R.T_COUNTRY["France"] = t
REC = {"complete": [], "p55": [], "dup": [], "lf": [], "xfc": []}

_mp, _f31, _cm, _p55, _dup, _lf, _xfc = R.model_parts, R.france31, FR.complete, FR.p55_veto, R.dup_add, FR.lf_veto, R.xfc_fit
_post = FR.post
REC["post"] = []


def post(q1, qx, id1, idx, p, *a, **k):
    REC["post"].append(np.asarray(p, np.float32).copy())   # the first call's p = the Qwen band blend (pbl)
    return _post(q1, qx, id1, idx, p, *a, **k)


FR.post = post
R.model_parts = lambda split, world=None, only=None, stack=True: _mp(split, world, only=("France",), stack=stack)


def f31(c):
    d = _f31(c)
    REC["d"] = {k: d[k] for k in ("q1", "qx", "p", "p_m2", "r")} | {"pbl": REC["post"][0]} | {"bge_p": d["bge"]["p"], "bge_r": d["bge"]["r"]}
    return d


def cm(*a, **k):
    o = _cm(*a, **k)
    REC["complete"].append((a[5].copy(), o.copy()))
    return o


def p55(*a, **k):
    o = _p55(*a, **k)
    REC["p55"].append(o.copy())
    return o


def dup(d, km, K, split):
    o = _dup(d, km, K, split)
    if "id1" in d:                                   # the main lines, not acro_change_set's scratch reference
        REC["dup"].append((km.copy(), o[0].copy()))
    return o


def lf(q1, qx):
    o = _lf(q1, qx)
    REC["lf"].append(((q1 << 32) | qx, o.copy()))
    return o


def xfc(d, x):
    o = _xfc(d, x)
    REC["xfc"].append((d["km"].copy(), o.copy()))
    return o


R.france31, FR.complete, FR.p55_veto, R.dup_add, FR.lf_veto, R.xfc_fit = f31, cm, p55, dup, lf, xfc
t0 = time.time()
try:
    R.run("test", "model", out, check=False)
except AssertionError as e:                          # rows == N_TEST_S1: only France was written, as intended
    print("expected France-only assert:", str(e)[:80], flush=True)
d = REC["d"]
K = (d["q1"].astype(np.int64) << 32) | d["qx"].astype(np.int64)
acc0, acc1 = REC["complete"][0]
vm = REC["p55"][0]
acc2 = acc1 & ~vm
(d1_in, d1_out), (d2_in, d2_out) = REC["dup"][0], REC["dup"][-1]
lf_in, lf_bad = REC["lf"][0]
x_in, x_out = REC["xfc"][0]
np.savez(out / "p69_frame.npz", K=K, p=d["p"], pbl=d["pbl"], g=d["p_m2"], rq=d["r"], rb=d["bge_r"], bge_p=d["bge_p"],
         acc0=acc0, acc1=acc1, vm=vm, km_acro_in=K[acc2], dup1_in=d1_in, dup1_out=d1_out, lf_in=lf_in, lf_bad=lf_bad,
         xfc_in=x_in, xfc_out=x_out, dup2_in=d2_in, dup2_out=d2_out, n_complete=len(REC["complete"]),
         n_dup=len(REC["dup"]), t=t)
print(f"P69 France t {t}: {time.time() - t0:.0f}s, final {len(d2_out):,} pairs", flush=True)
