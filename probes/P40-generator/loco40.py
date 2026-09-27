"""P40 step 4: France op rules under LOCO (P27's M2 on the other country, P37's protocol and gc70 base).

Rule add(cls, tau_FR): a free, argmax, non-sibling, not-yet-accepted pair of op class `cls` with p >= tau joins its S1
(caps 5 S2 / 6 S3). tau_FR maps to the target by the density of argmax pairs per S1 in [t, t_base) (P37 inv_dens).
Base on the target = t_base + rescue(FR 0.90) + gc oneall(FR 0.70) = the #37_gc70 recipe.
    probes/P40-generator/py.sh probes/P40-generator/loco40.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P37-fr-complete"), str(ROOT / "probes/P28-addr-anchor")]
import p37 as P  # noqa: E402
import ops as O  # noqa: E402
from cand import raw  # noqa: E402
from cat import OUT, W  # noqa: E402
from loss import bits  # noqa: E402

TAUS = (0.3, 0.5, 0.7, 0.8, 0.9)
T_GC = 0.70


def classes(B: pd.DataFrame, k1: np.ndarray) -> dict:
    gib, emp = B.N_GIB_NAME.to_numpy(), B.A_EMPTY.to_numpy()
    return {"gib": gib, "noaddr": emp, "noaddr_unique": emp & (k1 == 1), "gib|noaddr_unique": gib | (emp & (k1 == 1))}


def fr_dens():
    fr = pd.read_parquet(OUT / "fr_pairs.parquet", columns=["p"])
    return {t: float(((fr.p >= t) & (fr.p < P.T_FR)).sum() / 259452) for t in TAUS + (T_GC, P.T_RESCUE)}


def add_rule(s1, sx, is2, p, am, pred, sib, cls, t):
    """Every free, argmax, non-sibling, unaccepted pair of the class with p >= t (any S1, empty ones included);
    P37's caps: 5 S2 / 6 S3 per S1 counting existing members, best p first."""
    free = np.bincount(sx[pred], minlength=int(sx.max()) + 1)[sx] == 0
    c = am & ~pred & free & ~sib & cls & (p >= t)
    ns = int(s1.max()) + 1
    n2 = np.bincount(s1[pred & is2], minlength=ns)[s1]
    n3 = np.bincount(s1[pred & ~is2], minlength=ns)[s1]
    o = np.flatnonzero(c)
    o = o[np.lexsort((-p[o], s1[o] * 2 + is2[o]))]
    k = s1[o] * 2 + is2[o]
    i = np.arange(len(o))
    rank = i - np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], i, 0))
    room = np.where(is2, 5 - n2, 6 - n3)
    m = np.zeros(len(p), bool)
    m[o[rank < room[o]]] = True
    return m


def loco_one(src, tgt, dens):
    t0 = time.time()
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["y"], a["sib"], a["n_s1"]
    ntrue, hv, tb = a["t2"] + a["t3"], a["hv"], P.TS[(src, tgt)]
    tmap = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tmap[P.T_RESCUE])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tmap[T_GC], "oneall")
    per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
    f0 = per(pred)
    # tag the rule's candidate pool only
    pool = am & ~pred & (p >= min(tmap[t] for t in TAUS))
    ids1, idsx = a["s1_ids"][pool], a["sx_ids"][pool]
    r1, rx = raw(ids1, (1,)), raw(idsx, (2, 3))
    ops = np.zeros(len(p), np.uint64)
    ops[pool] = O.tag_frame(r1.business_name.reindex(ids1).to_numpy(), r1.business_address.reindex(ids1).to_numpy(),
                            rx.business_name.reindex(idsx).to_numpy(), rx.business_address.reindex(idsx).to_numpy(),
                            np.full(pool.sum(), tgt))
    k1 = np.zeros(len(p), np.int64)
    tw = pd.read_parquet(OUT / f"s1twin_{tgt}.parquet").set_index("id").k
    k1[pool] = tw.reindex(ids1).fillna(1).to_numpy()
    C = classes(bits(ops), k1)
    res = {"t_base": tb, "base_f05": round(f0.mean(), 5), "tmap": {str(k): round(v, 4) for k, v in tmap.items()}}
    print(f"{src}->{tgt}: base F0.5 {f0.mean():.5f}, pool {pool.sum():,}, tmap {res['tmap']}", flush=True)
    for name, cls in C.items():
        for t in TAUS:
            add = add_rule(s1, sx, is2, p, am, pred, sib, cls & pool, tmap[t])
            if not add.any():
                continue
            dl = per(pred | add) - f0
            x = dl[hv == 1]
            r = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                 "added": int(add.sum()), "prec": round(y[add].mean(), 4)}
            res[f"{name}@{t}"] = r
            print(f"  {name:18s} FR t {t:.2f} (tgt {tmap[t]:.3f}): dF {r['d']:+.6f} (h1 {r['d_h1']:+.6f} ± {r['se_h1']:.6f}) "
                  f"added {r['added']:,} prec {r['prec']:.3f}", flush=True)
    res["runtime_s"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    dens = fr_dens()
    print("France densities", {k: round(v, 5) for k, v in dens.items()})
    out = {"dens": dens}
    for src, tgt in (("US", "India"), ("India", "US")):
        out[f"{src}->{tgt}"] = loco_one(src, tgt, dens)
        import gc
        gc.collect()
    (HERE / "loco40.json").write_text(json.dumps(out, indent=1, default=str))
