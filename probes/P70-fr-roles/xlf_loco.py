"""Lead's gate (P77 q4): address-less SX whose name core equals a unique S1 core but whose full name differs only by the
legal form. P66's LOCO harness (P37 chain analogue, density-mapped France-p floors) with P64's legal-form relation split:
  a  one side has a legal form, the other none (pair_class 2 / 3)
  b  same legal-form signature, full name still differs (class 1: spelled differently)
  c  conflicting forms (class 4 / 5; P64 vetoes 4)
In-domain train precision (all argmax address-less pairs in the cell) with Wilson 95%, the unaccepted pool, and add dF.
    .venv/bin/python probes/P70-fr-roles/xlf_loco.py      -> probes/P70-fr-roles/xlf_loco.json"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "probes/P66-fr-noaddr"), str(ROOT / "probes/P64-legal-form-feat"), str(ROOT / "code/business_entity_resolution/src"), str(ROOT)]
import loco66 as L6  # noqa: E402  (P37 as P, rule40 as R, prof.fkey)
import lf  # noqa: E402
P, R = L6.P, L6.R
HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def lf_split(split, ids1, idsx, fk_eq):
    n1, nx = L6.names(split, ids1, (1,)), L6.names(split, idsx, (2, 3))
    sg = lambda s: (lambda u, inv: lf.sig(pd.Series(u)).to_numpy(object)[inv])(*np.unique(s.fillna("").astype(str).to_numpy(), return_inverse=True))
    k = lf.pair_class(pd.Series(sg(n1.reindex(ids1))), pd.Series(sg(nx.reindex(idsx))))
    return {"a": np.isin(k, [2, 3]) & ~fk_eq, "b": (k == 1) & ~fk_eq, "c": np.isin(k, [4, 5]) & ~fk_eq, "a|b": np.isin(k, [1, 2, 3]) & ~fk_eq}


def run():
    t0 = time.time()
    dens = L6.fr_dens()
    out = {}
    for src, tgt in (("US", "India"), ("India", "US")):
        A = P.loco_load(src, tgt)
        s1, sx, is2, p, am, y, sib, n = (A[k] for k in ("s1", "sx", "is2", "p", "am", "y", "sib", "n_s1"))
        ntrue, hv, tb = A["t2"] + A["t3"], A["hv"], P.TS[(src, tgt)]
        tm = {t: P.inv_dens(p, am, tb, g, n) for t, g in dens.items()}
        per = lambda pr: P.fbeta(np.bincount(s1[pr & y], minlength=n), np.bincount(s1[pr], minlength=n), ntrue)
        pred = am & (p >= tb)
        pred |= P.rescue(s1, p, am, pred, sib, tm[P.T_RESCUE])
        pred |= P.gc_add(s1, sx, is2, p, am, pred, sib, tm[0.7], "oneall")
        f0 = per(pred)
        na = pd.Series(A["sx_ids"]).isin(set(L6.addrless_ids("train", tgt))).to_numpy()
        w = np.flatnonzero(am & na)
        f = L6.feats("train", tgt, A["s1_ids"][w], A["sx_ids"][w], L6.fk_counts("train", tgt))
        corex = pd.read_parquet(L6.W / f"rf_train_sx_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(A["sx_ids"][w]).fillna("").to_numpy()
        core1 = pd.read_parquet(L6.W / f"rf_train_s1_{tgt}.parquet", columns=["id", "core"]).set_index("id").core.reindex(A["s1_ids"][w]).fillna("").to_numpy()
        base = (core1 == corex) & (core1 != "") & (f["core_n"] == 1)
        cls = {k: base & m for k, m in lf_split("train", A["s1_ids"][w], A["sx_ids"][w], f["fk_eq"]).items()}
        acc_w = pred[w]
        r = {"t_base": tb}
        for k, m in cls.items():
            kk, nn = int(y[w][m].sum()), int(m.sum())
            pool = m & ~acc_w
            r[f"cell_{k}"] = {"n": nn, "prec_all": round(kk / nn, 4) if nn else None, "wilson_all": wilson(kk, nn),
                              "n_pool": int(pool.sum()), "prec_pool": round(float(y[w][pool].mean()), 4) if pool.any() else None,
                              "wilson_pool": wilson(int(y[w][pool].sum()), int(pool.sum()))}
            full = np.zeros(len(p), bool)
            full[w] = m
            for t in (0.3, 0.5):
                add = R.add(s1, sx, is2, p, am, pred, sib, full, tm[t])
                if add.any():
                    dl = per(pred | add) - f0
                    x = dl[hv == 1]
                    r[f"{k}@{t}"] = {"d": round(dl.mean(), 6), "d_h1": round(x.mean(), 6), "se_h1": round(x.std() / np.sqrt(len(x)), 6),
                                     "added": int(add.sum()), "prec": round(float(y[add].mean()), 4)}
            print(f"{src}->{tgt} {k:4s} {json.dumps(r[f'cell_{k}'])} | @.3 {json.dumps(r.get(f'{k}@0.3'))}", flush=True)
        out[f"{src}->{tgt}"] = r
        print(f"[{time.time() - t0:.0f}s] {src}->{tgt} done", flush=True)
        del A
    (HERE / "xlf_loco.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run()
