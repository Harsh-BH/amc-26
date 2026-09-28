"""P66 LOCO: address-less SX classes added on the France-chain analogue (P37: strict t_base + rescue90 + gc70, density-mapped).
Train-truth precision of each class (all argmax address-less pairs, and the unaccepted pool) + dF ± SE(h1).
    .venv/bin/python probes/P66-fr-noaddr/loco66.py      -> probes/P66-fr-noaddr/loco66.json
"""
import faulthandler
import gc
import signal
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "probes/P37-fr-complete"), str(ROOT / "probes/P40-generator"), str(ROOT / "probes/P28-addr-anchor"), str(HERE)]
import p37 as P  # noqa: E402
import rule40 as R  # noqa: E402
from prof import fkey  # noqa: E402

W = ROOT / "data/work"
FR_T = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)          # France-p floors, density-mapped to the target


def fr_dens():
    d = pd.read_parquet(W / "p31/fr31v0.1_t9868.parquet", columns=["sx", "p"])
    p = d.p.to_numpy(np.float64)
    sxc = pd.factorize(d.sx)[0]
    best = np.full(sxc.max() + 1, -1.0)
    np.maximum.at(best, sxc, p)
    am = p == best[sxc]
    return {t: float(((p >= t) & (p < P.T_FR) & am).sum() / 259452) for t in FR_T + (P.T_RESCUE,)}


def addrless_ids(split, c):
    out = []
    for i in (2, 3):
        t = pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_address"], filters=[("country", "=", c)])
        a = pc.utf8_upper(pc.utf8_trim_whitespace(t.column(1)))
        m = pc.or_(pc.equal(a, ""), pc.is_in(a, value_set=pa.array(["NULL", "<NULL>"])))
        out.append(t.column(0).filter(m).to_numpy(zero_copy_only=False))
        del t, a, m
    return np.concatenate(out)


def names(split, ids, srcs):
    v = pa.array(sorted(set(ids)), pa.large_string())
    t = pd.concat([pq.read_table(W / f"{split}_source{i}.parquet", columns=["entity_id", "business_name"],
                                 filters=[("entity_id", "in", v)]).to_pandas() for i in srcs])
    return t.drop_duplicates("entity_id").set_index("entity_id").business_name


def fk_counts(split, c):
    t = pq.read_table(W / f"{split}_source1.parquet", columns=["business_name"], filters=[("country", "=", c)]).column(0)
    return pd.Series([fkey(x) for x in t.to_pylist()]).value_counts()


def feats(split, c, ids1, idsx, fcount):
    """fk_eq, fk_uniq (S1's full key unique in the country), sxk_n (#S1 carrying the SX's key), sim (ratio on keys)."""
    n1, nx = names(split, ids1, (1,)), names(split, idsx, (2, 3))
    k1 = [fkey(x) for x in n1.reindex(ids1).fillna("")]
    kx = [fkey(x) for x in nx.reindex(idsx).fillna("")]
    k1s, kxs = pd.Series(k1), pd.Series(kx)
    core = pd.read_parquet(W / f"rf_{split}_s1_{c}.parquet", columns=["id", "core"]).set_index("id").core
    cn = core.value_counts().reindex(core.reindex(ids1).fillna("").to_numpy()).fillna(0).to_numpy()
    return dict(core_n=cn, fk_eq=(k1s == kxs).to_numpy() & (k1s.str.len() >= 5).to_numpy(),
                fk_uniq=fcount.reindex(k1).fillna(0).to_numpy() == 1,
                sxk_n=fcount.reindex(kx).fillna(0).to_numpy(),
                sim=np.array([fuzz.ratio(a, b) for a, b in zip(k1, kx)]),
                tsr=np.array([fuzz.token_set_ratio(a, b) for a, b in zip(k1, kx)]))


def classes(f):
    u, eq, sim, n = f["fk_uniq"], f["fk_eq"], f["sim"], f["sxk_n"]
    near = u & ~eq & (n == 0)                    # SX key matches no S1 exactly; the argmax S1's key is unique
    return {"xf": eq & u, "xf_twin": eq & ~u, "near95": near & (sim >= 95), "near90": near & (sim >= 90),
            "near80": near & (sim >= 80), "xf|near90": (eq & u) | (near & (sim >= 90)),
            "tsr100_u": u & ~eq & (n == 0) & (f["tsr"] >= 100), "any": np.ones(len(u), bool),
            "xfc": eq & u & (f["core_n"] == 1), "xfc2": eq & u & (f["core_n"] <= 2), "near95c": near & (sim >= 95) & (f["core_n"] == 1),
            "xfc|near95c": (eq & u & (f["core_n"] == 1)) | (near & (sim >= 95) & (f["core_n"] == 1))}


def run():
    t0 = time.time()
    faulthandler.register(signal.SIGUSR1)
    lg = lambda m: print(f'[{time.time() - t0:.0f}s] {m}', flush=True)
    dens = fr_dens()
    out = {"fr_dens": dens}
    for src, tgt in (("US", "India"), ("India", "US")):
        lg(f'{src}->{tgt} load')
        A = P.loco_load(src, tgt)
        lg('loaded')
        s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = A["t2"] + A["t3"], A["hv"], P.TS[(src, tgt)]
        tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
        per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        pred = am & (p >= tb)
        pred |= P.rescue(s1, p, am, pred, sib, tm[P.T_RESCUE])
        pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm[0.7], "oneall")
        f0 = per(pred)
        lg('base done')
        na = pd.Series(A["sx_ids"]).isin(set(addrless_ids("train", tgt))).to_numpy()
        r = {"t_base": tb, "tmap": {str(k): round(v, 4) for k, v in tm.items()}, "base_f05": round(f0.mean(), 5),
             "noaddr_sx": int(len(np.unique(sx[na]))), "noaddr_am_true_rate": round(float(y[am & na].mean()), 4),
             "noaddr_recall_base": round(float((pred & na & y).sum() / max((na & y).sum(), 1)), 4)}
        # every address-less SX that has a true row in the file: is its truth the argmax row?
        tsx = np.unique(sx[na & y])
        r["noaddr_true_sx_argmax_is_true"] = round(float(np.isin(tsx, sx[na & y & am]).mean()), 4)
        lg('addrless done')
        w = np.flatnonzero(am & na)
        f = feats("train", tgt, A["s1_ids"][w], A["sx_ids"][w], fk_counts("train", tgt))
        cls = classes(f)
        lg(f'feats done, {len(w)} pairs')
        acc_w = pred[w]
        for k, m in cls.items():
            r[f"cell_{k}"] = {"n": int(m.sum()), "prec_all": round(float(y[w][m].mean()), 4) if m.any() else None,
                              "n_pool": int((m & ~acc_w).sum()),
                              "prec_pool": round(float(y[w][m & ~acc_w].mean()), 4) if (m & ~acc_w).any() else None}
        for k, m in cls.items():
            full = np.zeros(len(p), bool)
            full[w] = m
            for t in FR_T:
                add = R.add(s1, sx, is2, p, am, pred, sib, full, tm[t])
                if not add.any():
                    continue
                dl = per(pred | add) - f0
                x = dl[hv == 1]
                row = {"tgt_t": round(tm[t], 4), "d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6),
                       "se_h1": round(x.std() / np.sqrt(len(x)), 6), "added": int(add.sum()), "prec": round(float(y[add].mean()), 4)}
                r[f"{k}@{t}"] = row
                print(f"{src}->{tgt} {k:10s} FR {t} (tgt {tm[t]:.3f}): dF {row['d']:+.6f} (h1 {row['d_h1']:+.6f} ± {row['se_h1']:.6f}) "
                      f"added {row['added']} prec {row['prec']}", flush=True)
        print(json.dumps({k: v for k, v in r.items() if not k[0].isalpha() or k.startswith(("cell", "noaddr", "base", "t_"))}, indent=1),
              flush=True)
        out[f"{src}->{tgt}"] = r
        del A, f, cls
        gc.collect()
    out["runtime_s"] = round(time.time() - t0)
    (HERE / "loco66.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    run()
