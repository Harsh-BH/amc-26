"""P40 step 4b: near-exact France op rules under LOCO + their France counts.

Classes (on free, argmax, non-sibling, unaccepted pairs, p >= tau mapped by density as loco40):
  xu_noaddr : SX address empty AND SX name core == S1 name core AND no other S1 of the corpus has that core
  xu_any    : same, any address
  gib       : synthetic-brand SX (loco40's class, repeated for the combined arm)
  xu_noaddr|gib
Cores = the pipeline's rf `core` (normalize.name_core); uniqueness counted in the S1 corpus the rule runs on (train S1 of
the LOCO target; test S1 of France). Cores shorter than 5 letters never count as unique (too generic).
    P37_BASE=output/sub37_gc70 probes/P40-generator/py.sh probes/P40-generator/loco41.py [loco|france]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import loco40 as L  # noqa: E402  (sets P37 paths)
import ops as O  # noqa: E402
from cand import raw  # noqa: E402
from cat import OUT, W  # noqa: E402
from loss import bits  # noqa: E402

P = L.P
TAUS = (0.05, 0.3, 0.5, 0.7, 0.9)


def core_of(path, ids):
    t = pq.read_table(path, columns=["id", "core"])
    pos = pc.index_in(pa.array(list(ids), pa.large_string()), value_set=t.column("id")).fill_null(-1).to_numpy()
    core = t.column("core").to_numpy(zero_copy_only=False)
    out = np.full(len(ids), "", object)
    out[pos >= 0] = core[pos[pos >= 0]]
    return out


def uniq_counts(path):
    c = pq.read_table(path, columns=["core"]).column(0).to_pandas()
    return c.value_counts()


def classes(emp, gib, c1, cx, kmap):
    k = kmap.reindex(c1).fillna(0).to_numpy()
    xu = (c1 == cx) & (k == 1) & (np.char.str_len(c1.astype(str)) >= 5)
    return {"xu_noaddr": xu & emp, "xu_any": xu, "gib": gib, "xu_noaddr|gib": (xu & emp) | gib}


def loco_one(src, tgt, dens):
    t0 = time.time()
    a = P.loco_load(src, tgt)
    s1, sx, is2, p, am, y, sib, n = a["s1"], a["sx"], a["is2"], a["p"], a["am"], a["y"], a["sib"], a["n_s1"]
    ntrue, hv, tb = a["t2"] + a["t3"], a["hv"], P.TS[(src, tgt)]
    tmap = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
    pred = am & (p >= tb)
    pred |= P.rescue(s1, p, am, pred, sib, tmap[P.T_RESCUE])
    pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tmap[L.T_GC], "oneall")
    per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
    f0 = per(pred)
    pool = am & ~pred & (p >= min(tmap[t] for t in TAUS))
    ids1, idsx = a["s1_ids"][pool], a["sx_ids"][pool]
    r1, rx = raw(ids1, (1,)), raw(idsx, (2, 3))
    ops = O.tag_frame(r1.business_name.reindex(ids1).to_numpy(), r1.business_address.reindex(ids1).to_numpy(),
                      rx.business_name.reindex(idsx).to_numpy(), rx.business_address.reindex(idsx).to_numpy(),
                      np.full(pool.sum(), tgt))
    B = bits(ops)
    c1 = core_of(W / f"rf_train_s1_{tgt}.parquet", ids1)
    cx = core_of(W / f"rf_train_sx_{tgt}.parquet", idsx)
    kmap = uniq_counts(W / f"rf_train_s1_{tgt}.parquet")
    C = classes(B.A_EMPTY.to_numpy(), B.N_GIB_NAME.to_numpy(), c1, cx, kmap)
    res = {"base_f05": round(f0.mean(), 5), "tmap": {str(k): round(v, 4) for k, v in tmap.items()}}
    print(f"{src}->{tgt}: base F0.5 {f0.mean():.5f}, pool {pool.sum():,}", flush=True)
    for name, cm in C.items():
        cls = np.zeros(len(p), bool)
        cls[pool] = cm
        for t in TAUS:
            add = L.add_rule(s1, sx, is2, p, am, pred, sib, cls, tmap[t])
            if not add.any():
                continue
            dl = per(pred | add) - f0
            x = dl[hv == 1]
            r = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                 "added": int(add.sum()), "prec": round(y[add].mean(), 4)}
            res[f"{name}@{t}"] = r
            print(f"  {name:14s} FR t {t:.2f} (tgt {tmap[t]:.3f}): dF {r['d']:+.6f} (h1 {r['d_h1']:+.6f} ± {r['se_h1']:.6f}) "
                  f"added {r['added']:,} prec {r['prec']:.3f}", flush=True)
    res["runtime_s"] = round(time.time() - t0)
    return res


def france(write_mask=None):
    """France counts per rule on the gc70 base (P37 fr_load with P37_BASE=output/sub37_gc70)."""
    d, a = P.fr_load()
    s1, sx, is2, p, am, pred, sib = (a[k] for k in ("s1", "sx", "is2", "p", "am", "pred", "sib"))
    pool = am & ~pred & (p >= min(TAUS))
    ids1, idsx = d.s1.to_numpy()[pool], d.sx.to_numpy()[pool]
    fr = pd.read_parquet(OUT / "fr_pairs.parquet", columns=["s1", "sx", "ops"])
    om = pd.Series(fr.ops.to_numpy(), index=pd.MultiIndex.from_arrays([fr.s1, fr.sx]))
    ops = om.reindex(pd.MultiIndex.from_arrays([ids1, idsx])).to_numpy()
    assert not pd.isna(ops).any(), "every France pool pair was tagged by france.py"
    B = bits(ops.astype(np.uint64))
    c1 = core_of(W / "rf_test_s1_France.parquet", ids1)
    cx = core_of(W / "rf_test_sx_France.parquet", idsx)
    C = classes(B.A_EMPTY.to_numpy(), B.N_GIB_NAME.to_numpy(), c1, cx, uniq_counts(W / "rf_test_s1_France.parquet"))
    out = {}
    for name, cm in C.items():
        cls = np.zeros(len(p), bool)
        cls[pool] = cm
        for t in TAUS:
            add = L.add_rule(s1, sx, is2, p, am, pred, sib, cls, t)
            out[f"{name}@{t}"] = {"added": int(add.sum()), "s1": int(np.unique(s1[add]).size)}
            if write_mask == f"{name}@{t}":
                return d[add][["s1", "sx"]]
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "loco"
    if cmd == "france":
        (HERE / "loco41_france.json").write_text(json.dumps(france(), indent=1))
    else:
        dens = L.fr_dens()
        dens.update({t: float(((pd.read_parquet(OUT / "fr_pairs.parquet", columns=["p"]).p >= t) &
                               (pd.read_parquet(OUT / "fr_pairs.parquet", columns=["p"]).p < P.T_FR)).sum() / 259452) for t in TAUS})
        out = {"dens": dens}
        for src, tgt in (("US", "India"), ("India", "US")):
            out[f"{src}->{tgt}"] = loco_one(src, tgt, dens)
        (HERE / "loco41.json").write_text(json.dumps(out, indent=1, default=str))
