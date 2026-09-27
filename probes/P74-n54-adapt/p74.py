"""P74: run.py's sub66_final_n54 France stage alone (p69.py's pattern), instrumented: the stage keys of every France
pair, N54's lens input and the pairs decoy_drop removes -> OUT/p74_frame.npz + OUT's France lines (splice.py pastes
them into sub66_final).
    flock /tmp/p69.lock .venv/bin/python probes/P74-n54-adapt/p74.py probes/P74-n54-adapt/work/n54"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
import run as R  # noqa: E402
import france as FR  # noqa: E402

out = Path(sys.argv[1])
R.use("sub66_final_n54")
REC = {"complete": [], "p55": [], "dup": [], "lf": [], "xfc": [], "decoy": []}
_mp, _f31, _cm, _p55, _dup, _lf, _xfc, _dd = (R.model_parts, R.france31, FR.complete, FR.p55_veto, R.dup_add, FR.lf_veto,
                                              R.xfc_fit, R.decoy_drop)
R.model_parts = lambda split, world=None, only=None, stack=True: _mp(split, world, only=("France",), stack=stack)


def f31(c):
    d = _f31(c)
    REC["d"] = {k: d[k] for k in ("q1", "qx", "p", "p_m2", "r")} | {"bge_r": d["bge"]["r"]}
    return d


def cm(*a, **k):
    o = _cm(*a, **k)
    REC["complete"].append((a[5].copy(), o.copy(), a))
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


def dd(d, ka, s_min):
    o = _dd(d, ka, s_min)
    REC["decoy"].append((ka.copy(), d["km"].copy(), o.copy()))
    return o


R.france31, FR.complete, FR.p55_veto, R.dup_add, FR.lf_veto, R.xfc_fit, R.decoy_drop = f31, cm, p55, dup, lf, xfc, dd
t0 = time.time()
try:
    R.run("test", "model", out, check=False)
except AssertionError as e:                          # rows == N_TEST_S1: only France was written, as intended
    print("expected France-only assert:", str(e)[:80], flush=True)
d = REC["d"]
K = (d["q1"].astype(np.int64) << 32) | d["qx"].astype(np.int64)
acc0, acc1, a = REC["complete"][0]
rescue_only = _cm(*a[:9])                            # the rescue bar alone: gc's adds = the rest
vm = REC["p55"][0]
(d1_in, d1_out), (d2_in, d2_out) = REC["dup"][0], REC["dup"][-1]
x_in, x_out = REC["xfc"][0]
ka, dk_in, dk_out = REC["decoy"][0]
np.savez(out / "p74_frame.npz", K=K, p=d["p"], g=d["p_m2"], rq=d["r"], rb=d["bge_r"], acc0=acc0, acc1=acc1,
         rescue_only=rescue_only, vm=vm, dup1_in=d1_in, dup1_out=d1_out, lf_in=REC["lf"][0][0], lf_bad=REC["lf"][0][1],
         xfc_in=x_in, xfc_out=x_out, ka=ka, decoy_in=dk_in, decoy_out=dk_out, dup2_in=d2_in, dup2_out=d2_out,
         n_complete=len(REC["complete"]), n_dup=len(REC["dup"]))
print(f"P74 France n54: {time.time() - t0:.0f}s, decoy {len(dk_in):,} -> {len(dk_out):,}, final {len(d2_out):,}", flush=True)
