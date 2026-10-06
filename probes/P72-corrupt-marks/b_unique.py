"""P72 B: India/US unique-key recovery on the combiner's rejected band (argmax pairs, p'' < 0.71).

  addr rule: lo_a <= p < hi, no shared name token, SX address non-empty -> K = shared address tokens - street/unit words;
             accept if K and exactly one S1 of the country pool has an address containing all of K.
  name rule: lo_n <= p < hi, SX address empty -> K = shared name tokens - legal/filler words; accept if exactly one S1 name
             of the pool contains all of K.
unique(pool_texts, queries) -> bool[] ; texts are normalize's (P3 txt 'name' / 'addr', space-tokenised).
    .venv/bin/python probes/P72-corrupt-marks/b_unique.py            # self-check
    .venv/bin/python probes/P72-corrupt-marks/b_unique.py ev [world]  # holdout gate on ev72's out-of-fold base scores
"""
import json
import os
import sys
import time
from array import array

import numpy as np
import pandas as pd

sys.path[:0] = ["probes/P44-meta-stack", "probes/P50-idea-batch", "probes/P57-iu-transfer", "probes/P72-corrupt-marks",
                "code/business_entity_resolution/src", "."]
import normalize as N  # noqa: E402

_CITY = {"bombay", "mumbai", "calcutta", "kolkata", "madras", "chennai", "poona", "pune", "bengaluru", "bangalore",
         "gurugram", "gurgaon", "ahmadabad", "ahmedabad", "atl", "atlanta"}      # STREET's city aliases are keys, not street words
ADDR_DROP = {w for w in set(N.STREET) | set(N.STREET.values()) if not w.isdigit()} - _CITY | N._STREET_TYPES | N.FILLER | {"near", "nr", "opp", "opposite",
            "behind", "beside", "floor", "flat", "plot", "house", "shop", "block", "sector", "phase", "apt", "ste", "fl"}
NAME_DROP = N.LEGAL | N.HONORIFIC | N.GENERIC | N.STOP | N.FILLER
empty = lambda a: a.strip() in ("", "null")


def unique(pool: list, queries: list) -> np.ndarray:
    """queries: token sets. True where exactly one pool text holds every token of the (non-empty) query."""
    want = set().union(*queries) if queries else set()
    post = {t: array("i") for t in want}
    for i, s in enumerate(pool):
        for t in set(s.split()) & want:
            post[t].append(i)
    post = {t: np.frombuffer(v, np.int32) for t, v in post.items()}
    out = np.zeros(len(queries), bool)
    for j, q in enumerate(queries):
        if not q:
            continue
        ps = sorted((post[t] for t in q), key=len)
        cur = ps[0]
        for p in ps[1:]:
            if len(cur) <= 1:
                break
            cur = np.intersect1d(cur, p, assume_unique=True)
        out[j] = len(cur) == 1
    return out


def keys(n1, a1, nx, ax, p, lo_a, lo_n, hi) -> tuple:
    """(rule id per pair: 0 none / 1 addr / 2 name, query token sets)."""
    rule, qs = np.zeros(len(p), np.int8), []
    for i, (u1, b1, ux, bx, pi) in enumerate(zip(n1, a1, nx, ax, p)):
        sn = set(u1.split()) & set(ux.split())
        if lo_a <= pi < hi and not sn and not empty(bx):
            k = (set(b1.split()) & set(bx.split())) - ADDR_DROP
            rule[i] = 1 if k else 0
        elif lo_n <= pi < hi and empty(bx):
            k = sn - NAME_DROP
            rule[i] = 2 if k else 0
        else:
            k = set()
        qs.append(k)
    return rule, qs


def texts(split, c, q1, qx):
    """normalised (name, addr) of the S1 pool (whole country) + of the SX rows qx."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import block as B
    import features as F
    s1 = pd.read_parquet(B.TXT[split] / f"{c}_s1.parquet", columns=["name", "addr"])
    files, offs = F.sx_files(split, c)
    nx, ax = np.empty(len(qx), object), np.empty(len(qx), object)
    for f, lo, hi in zip(files, offs[:-1], offs[1:]):
        m = (qx >= lo) & (qx < hi)
        if m.any():
            t = pq.read_table(f, columns=["name", "addr"])
            idx = pa.array(qx[m] - lo)
            nx[m] = t.column("name").take(idx).to_numpy(zero_copy_only=False)
            ax[m] = t.column("addr").take(idx).to_numpy(zero_copy_only=False)
            del t
    return s1.name.to_numpy(object), s1.addr.to_numpy(object), nx, ax


def ev(w="sw", split="train"):
    os.environ["P44_VARIANT"] = "r"
    import ev as E
    import ev57
    t0, res = time.time(), {}
    S = ev57.load_s()
    grid = [(0.30, 0.10, 0.70), (0.30, 0.10, 0.71), (0.50, 0.30, 0.71), (0.20, 0.05, 0.71)]
    for c in E.C:
        A = pd.read_parquet(E.W / "p57" / f"A_{w}_{c}.parquet", columns=["q1", "qx", "y", "g", "half"])
        s = np.empty(len(A))
        for h in (0, 1):
            s[A.half.to_numpy() == h] = np.load(E.W / "p72" / f"s_{w}_{c}_h{h}_base.npy")
        n1, a1, nx, ax = texts(split, c, A.q1.to_numpy(), A.qx.to_numpy())
        q1 = A.q1.to_numpy()
        nt, hs = S[c].n_true.to_numpy(), S[c].half.to_numpy()
        base = s >= E.T
        fb = ev57.per_s1(A, nt, base)
        y = A.y.to_numpy()
        for lo_a, lo_n, hi in grid:
            band = (s < hi) & (s >= min(lo_a, lo_n))
            ii = np.flatnonzero(band)
            rule, qs = keys(n1[q1[ii]], a1[q1[ii]], nx[ii], ax[ii], s[ii], lo_a, lo_n, hi)
            ok = np.zeros(len(ii), bool)
            for r_, pool in ((1, a1), (2, n1)):
                m = np.flatnonzero(rule == r_)
                ok[m] = unique(list(pool), [qs[j] for j in m])
            add = np.zeros(len(A), bool)
            add[ii[ok]] = True
            add &= ~base
            f = ev57.per_s1(A, nt, base | add)
            tag = f"{c} a{lo_a}/n{lo_n}/<{hi}"
            r = {}
            for h in (0, 1):
                x = (f - fb)[hs == h]
                hr = A.half.to_numpy() == h
                ka, kn = add & hr & np.isin(np.arange(len(A)), ii[rule == 1]), add & hr & np.isin(np.arange(len(A)), ii[rule == 2])
                r[f"h{h}"] = dict(d=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))),
                                  adds_addr=int(ka.sum()), prec_addr=float(y[ka].mean()) if ka.any() else None,
                                  adds_name=int(kn.sum()), prec_name=float(y[kn].mean()) if kn.any() else None,
                                  cand_addr=int(((rule == 1) & ~base[ii] & (A.half.to_numpy()[ii] == h)).sum()),
                                  cand_name=int(((rule == 2) & ~base[ii] & (A.half.to_numpy()[ii] == h)).sum()))
            res[tag] = r
            print(tag, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
        del n1, a1, nx, ax
    for g in grid:
        for h in (0, 1):
            k = f"a{g[0]}/n{g[1]}/<{g[2]}"
            lb = sum(E.LB_W[c] * res[f"{c} {k}"][f"h{h}"]["d"] for c in E.C)
            se = float(np.sqrt(sum((E.LB_W[c] * res[f"{c} {k}"][f"h{h}"]["se"]) ** 2 for c in E.C)))
            res[f"lb {k} h{h}"] = (lb, se)
            print(f"LB-eq {k} report h{h}: {lb:+.6f} ± {se:.6f}", flush=True)
    (E.W / "p72" / f"b_{w}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["ev"]:
        ev(*sys.argv[2:])
    else:
        pool = ["12 lake road kolkata", "12 lake road howrah", "7 park st kolkata"]
        assert list(unique(pool, [{"12", "lake", "kolkata"}, {"12", "lake"}, set(), {"park"}])) == [True, False, False, True]
        r, q = keys(["acme foods", "zeta"], ["12 lake road kolkata", "x"], ["zen corp", "zeta inc"], ["12 lake rd kolkata", ""],
                    np.array([0.5, 0.2]), 0.3, 0.1, 0.7)
        assert list(r) == [1, 2] and q[0] == {"12", "lake", "kolkata"} and q[1] == {"zeta"}, (r, q)
        print("b_unique self-check ok")
