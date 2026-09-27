"""P57 (master's check): are sub54's India/US adds / drops vs #34 on test enriched for house-number mismatch, compared with the
same recipe's holdout flips? hn classes (stack.hn_cls): 0 none/equal, 1 sibling +d, 2 digit drop/add, 3 small other, 4 other.
Test: P50's T frames + prob_test_{c}_fullzoo (sub54's India/US). Holdout: the sub54 recipe fit on the other half (ev57's cached
orig boosters), scored on each half of world w, with the flips' precision per hn stratum.
    .venv/bin/python probes/P57-iu-transfer/hnmix.py [world ...]     -> data/work/p57/hnmix.json"""
import json
import os
import sys

os.environ.setdefault("P57_ARMS", "sub54")
sys.path.insert(0, "probes/P57-iu-transfer")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ev57 as V  # noqa: E402

T = 0.71
W = V.E.W


def strata(hd, hc):
    both = ~np.isnan(hd)
    return {"no_hn": ~both, "hn_eq": both & (hc == 0), "sib": both & (hc == 1), "digit": both & (hc == 2),
            "small": both & (hc == 3), "other": both & (hc == 4)}


def prof(sel: dict, hd, hc, y=None) -> dict:
    out = {}
    st = strata(hd, hc)
    for k, m in sel.items():
        n = int(m.sum())
        r = {"n": n, **{f"sh_{s}": float((m & v).sum() / max(n, 1)) for s, v in st.items()}}
        r["sh_mismatch"] = float((m & ~st["no_hn"] & ~st["hn_eq"]).sum() / max(n, 1))
        if y is not None:
            r["prec"] = float(y[m].mean()) if n else float("nan")
            mm = m & ~st["no_hn"] & ~st["hn_eq"]
            r["prec_mismatch"] = float(y[mm].mean()) if mm.any() else float("nan")
            for sn, v in st.items():
                r[f"prec_{sn}"] = float(y[m & v].mean()) if (m & v).any() else float("nan")
                r[f"n_{sn}"] = int((m & v).sum())
        out[k] = r
    return out


def main(worlds):
    res = {}
    for c in V.E.C:
        A = pd.read_parquet(W / f"p50/T_{c}.parquet", columns=["row", "pa", "hn_d", "hn_cls"])
        pz = pd.read_parquet(W / f"p50/prob_test_{c}_fullzoo.parquet", columns=["p"]).p.to_numpy()[A.row.to_numpy()]
        old, new = A.pa.to_numpy() >= T, pz >= T
        res[f"test|{c}"] = prof({"acc34": old, "unacc34": ~old, "adds": new & ~old, "drops": old & ~new},
                                A.hn_d.to_numpy(np.float64), A.hn_cls.to_numpy())
        del A, pz
    S = V.load_s()
    cols, _ = V.ARMS["sub54"]
    for w in worlds:
        D = V.load(w)
        for c in V.E.C:
            A = D[c]
            s = np.empty(len(A))
            for fit_h in (0, 1):
                m = A.half.to_numpy() == 1 - fit_h
                s[m] = V.score(V.boosters("sub54", fit_h), A[m], cols)
            old, new = A.pa.to_numpy() >= T, s >= T
            res[f"{w}|{c}"] = prof({"acc34": old, "unacc34": ~old, "adds": new & ~old, "drops": old & ~new},
                                   A.hn_d.to_numpy(np.float64), A.hn_cls.to_numpy(), A.y.to_numpy())
        del D
    json.dump(res, open(W / "p57/hnmix.json", "w"), indent=1, default=float)
    # expected precision of test flips: holdout precision per hn stratum (worlds pooled) x test's stratum mix
    for c in V.E.C:
        for sel in ("adds", "drops"):
            tm = res[f"test|{c}"][sel]
            num = {s: sum(res[f"{w}|{c}"][sel][f"n_{s}"] * res[f"{w}|{c}"][sel][f"prec_{s}"] for w in worlds
                          if res[f"{w}|{c}"][sel][f"n_{s}"]) for s in strata(np.zeros(1), np.zeros(1))}
            den = {s: sum(res[f"{w}|{c}"][sel][f"n_{s}"] for w in worlds) for s in num}
            ph = {s: num[s] / den[s] for s in num if den[s]}
            e_test = sum(tm[f"sh_{s}"] * ph[s] for s in ph) / sum(tm[f"sh_{s}"] for s in ph)
            e_hold = sum(num.values()) / sum(den.values())
            res[f"exp|{c}|{sel}"] = {"prec_by_stratum": ph, "n_by_stratum": den, "holdout_prec": e_hold, "test_mix_prec": e_test}
            print(f"{c} {sel}: holdout precision {e_hold:.3f} -> at test's hn mix {e_test:.3f}; per stratum "
                  + ", ".join(f"{s} {ph[s]:.2f} (n {den[s]})" for s in ph), flush=True)
    json.dump(res, open(W / "p57/hnmix.json", "w"), indent=1, default=float)
    for k, v in res.items():
        if k.startswith("exp|"):
            continue
        for sel in ("acc34", "adds", "drops"):
            r = v[sel]
            extra = f" prec {r['prec']:.3f} (mismatch {r['prec_mismatch']:.3f})" if "prec" in r else ""
            print(f"{k:14s} {sel:6s} n={r['n']:>8,} no_hn {r['sh_no_hn']:.3f} eq {r['sh_hn_eq']:.3f} mismatch {r['sh_mismatch']:.3f} "
                  f"(sib {r['sh_sib']:.3f} digit {r['sh_digit']:.3f} small {r['sh_small']:.3f} other {r['sh_other']:.3f}){extra}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["full", "sw"])
