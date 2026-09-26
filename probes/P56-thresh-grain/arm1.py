"""P56 arm 1: finer-grained India/US thresholds on sub44r's combiner p'' (out of fold on both holdout halves).

Base: acc = #8c argmax pair & p'' >= 0.71 (one t, both countries). p'' for half-1 S1 = P44's half-0 fit, for half-0 S1 =
P49's swapfit (half-1 fit). Variants replace the single t by a t per cell:
  src         cell = S2 / S3
  rk_same     cell = greedy rank inside (S1, source): the pair's # already-accepted higher-p'' same-source pairs, 0 / 1 / 2+
  rk_any      the same, counted over the S1's pairs of both sources
  src_rk_same cell = source x rk_same (6 t)
Greedy: within a group sorted by p'' desc, pair j is accepted iff every pair above it was and p''_j >= t[cell_j].
Thresholds fit by coordinate descent on one half's mean per-S1 F0.5 (per country, or pooled on LB-eq), reported on the
other half vs the base: per-S1 paired delta, SE, LB-eq = 0.468 India + 0.383 US.
    probes/P56-thresh-grain/py.sh probes/P56-thresh-grain/arm1.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/work"
sys.path.insert(0, str(ROOT))
T0 = 0.71
LB_W = {"India": 0.468, "US": 0.383}
GRID = np.round(np.arange(0.30, 0.981, 0.01), 2)


def fb(tp, npred, ntrue):
    tp, npred, ntrue = (np.asarray(x, float) for x in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.25 * tp / (0.25 * ntrue + npred)
    f = np.where(tp == 0, 0.0, f)
    return np.where(ntrue == 0, np.where(npred == 0, 1.0, 0.0), f)


def load(c: str) -> dict:
    S = pd.read_parquet(W / f"p44ms/S1_{c}.parquet", columns=["n_true", "half"])
    A = pd.read_parquet(W / f"p44ms/A_{c}.parquet", columns=["g", "half", "y", "is_s3"])
    p0, p1 = np.load(W / f"p44ms/ho_p44r_{c}.npy"), np.load(W / f"p49/ho_p44r_h1fit_{c}.npy")
    pa = np.where(A.half.to_numpy() == 1, p0, p1).astype(np.float64)
    g, s3 = A.g.to_numpy(np.int64), A.is_s3.to_numpy() > 0.5
    o = np.lexsort((-pa, g))                                   # rows sorted by S1, p'' desc
    d = dict(c=c, n=len(S), n_true=S.n_true.to_numpy(), half=S.half.to_numpy().astype(np.int64),
             g=g[o], y=A.y.to_numpy()[o].astype(bool), s3=s3[o], pa=pa[o])
    # greedy ranks: position inside (S1, source) and inside S1 in p'' order
    for key, name in ((d["g"] * 2 + d["s3"], "rk_same"), (d["g"], "rk_any")):
        o2 = np.lexsort((-d["pa"], key))
        k = key[o2]
        start = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
        rk = np.arange(len(k)) - np.repeat(start, np.diff(np.r_[start, len(k)]))
        r = np.empty(len(k), np.int64)
        r[o2] = rk
        d[name] = np.minimum(r, 2)
        d[name + "_key"] = key
    d["base"] = score(d, np.full(len(d["pa"]), T0))
    return d


def cells(d: dict, kind: str) -> tuple:
    """(cell id per row, n cells, greedy group key or None)."""
    s3 = d["s3"].astype(np.int64)
    if kind == "src":
        return s3, 2, None
    if kind == "rk_same":
        return d["rk_same"], 3, d["rk_same_key"]
    if kind == "rk_any":
        return d["rk_any"], 3, d["rk_any_key"]
    if kind == "src_rk_same":
        return s3 * 3 + d["rk_same"], 6, d["rk_same_key"]
    raise ValueError(kind)


_ORD = {}


def order(pa, key):
    """(rows sorted by key, p'' desc; group start of each sorted row) -- cached, it does not depend on t."""
    h = (id(pa), id(key))
    if h not in _ORD:
        o2 = np.lexsort((-pa, key))
        k = key[o2]
        start = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
        _ORD[h] = (o2, np.repeat(start, np.diff(np.r_[start, len(k)])), (pa, key))
    return _ORD[h][:2]


def accept(d: dict, thr: np.ndarray, key) -> np.ndarray:
    ok = d["pa"] >= thr
    if key is None:
        return ok
    o2, st = order(d["pa"], key)
    fail = (~ok[o2]).astype(np.int64)
    cf = np.cumsum(fail)
    base = cf[st] - fail[st]
    acc = np.empty(len(o2), bool)
    acc[o2] = (cf - base) == 0                                  # no failure at or above this row in its group
    return acc


def score(d: dict, thr: np.ndarray, key=None) -> np.ndarray:
    acc = accept(d, thr, key)
    g, n = d["g"], d["n"]
    return fb(np.bincount(g[acc & d["y"]], minlength=n), np.bincount(g[acc], minlength=n), d["n_true"])


def fit(D: dict, kind: str, half: int, pooled: bool, rounds: int = 3) -> dict:
    """Coordinate descent on the fit half's mean per-S1 F0.5; returns {country: t vector}."""
    cs = list(D)
    groups = [cs] if pooled else [[c] for c in cs]
    out = {}
    for grp in groups:
        cid = {c: cells(D[c], kind) for c in grp}
        k = cid[grp[0]][1]
        t = np.full(k, T0)
        m = {c: D[c]["half"] == half for c in grp}

        def obj(tv):
            return sum(LB_W[c] * score(D[c], tv[cid[c][0]], cid[c][2])[m[c]].mean() for c in grp)
        best = obj(t)
        for _ in range(rounds):
            moved = False
            for j in range(k):
                vals = []
                for v in GRID:
                    tv = t.copy()
                    tv[j] = v
                    vals.append(obj(tv))
                vals = np.array(vals)
                jb = int(np.argmax(vals))
                if vals[jb] > best + 1e-12 and GRID[jb] != t[j]:
                    t[j], best, moved = GRID[jb], vals[jb], True
            if not moved:
                break
        for c in grp:
            out[c] = t.copy()
    return out


def report(D: dict, kind: str, tv: dict, rep: int) -> dict:
    r = {}
    for c, d in D.items():
        cid, _, key = cells(d, kind)
        x = (score(d, tv[c][cid], key) - d["base"])[d["half"] == rep]
        r[c] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int((x > 0).sum()), int((x < 0).sum()))
    lb = sum(LB_W[c] * r[c][0] for c in D)
    se = float(np.sqrt(sum((LB_W[c] * r[c][1]) ** 2 for c in D)))
    return dict(per=r, lb=lb, se=se)


def selftest(D):
    d = D["India"]
    for kind in ("src", "rk_same", "rk_any", "src_rk_same"):
        cid, k, key = cells(d, kind)
        assert np.allclose(score(d, np.full(k, T0)[cid], key), d["base"]), kind   # all-0.71 == base
    # greedy: a pair below its own cell's t blocks every lower pair of its group
    tiny = dict(pa=np.array([0.9, 0.8, 0.75]), g=np.zeros(3, np.int64))
    assert accept(tiny, np.array([0.5, 0.85, 0.5]), np.zeros(3, np.int64)).tolist() == [True, False, False]


if __name__ == "__main__":
    from ledger import log
    t0 = time.time()
    D = {c: load(c) for c in LB_W}
    selftest(D)
    for c, d in D.items():
        print(c, "rows", len(d["pa"]), "base F h0/h1", round(d["base"][d["half"] == 0].mean(), 6),
              round(d["base"][d["half"] == 1].mean(), 6), flush=True)
    res = {}
    kinds = sys.argv[1:] or ["src", "rk_same", "rk_any", "src_rk_same"]
    for kind in kinds:
        for pooled in (True, False):
            name = f"{kind}{'_pooled' if pooled else '_percountry'}"
            ts = time.time()
            row = {}
            for fh in (0, 1):
                tv = fit(D, kind, fh, pooled)
                rr = report(D, kind, tv, 1 - fh)
                row[f"h{fh}to{1 - fh}"] = dict(t={c: tv[c].tolist() for c in tv}, **rr)
                print(name, f"h{fh}->h{1 - fh}", {c: tv[c].tolist() for c in tv},
                      f"LB {rr['lb']:+.6f} ± {rr['se']:.6f} ({rr['lb'] / max(rr['se'], 1e-12):.1f} SE)",
                      {c: f"{v[0]:+.6f}±{v[1]:.6f} up{v[2]} dn{v[3]}" for c, v in rr['per'].items()}, flush=True)
            res[name] = row
            a, b = row["h0to1"], row["h1to0"]
            ok = (a["lb"] >= 1e-4 and b["lb"] >= 1e-4 and a["lb"] >= 3 * a["se"] and b["lb"] >= 3 * b["se"]
                  and min(a["lb"], b["lb"]) > 0)
            log("exp", probe="P56-thresh-grain", subproblem="decision", name=f"arm1-{name}", world="holdout",
                metrics={"lbeq_h0h1": a["lb"], "se_h0h1": a["se"], "lbeq_h1h0": b["lb"], "se_h1h0": b["se"],
                         **{f"d_{c}_h0h1": a["per"][c][0] for c in LB_W}, **{f"d_{c}_h1h0": b["per"][c][0] for c in LB_W},
                         "t_h0": a["t"], "t_h1": b["t"]},
                config={"kind": kind, "pooled": pooled, "grid": [0.30, 0.98, 0.01], "base_t": T0,
                        "p": "sub44r combiner p'' OOF both halves"},
                hypothesis="per-source / group-state thresholds beat one t 0.71 on sub44r's p''",
                verdict="keep" if ok else "drop", runtime_s=round(time.time() - ts, 1), peak_gb=0.6)
    json.dump(res, open(Path(__file__).parent / f"arm1_{'_'.join(kinds)}.json", "w"), indent=1)
    print("total", round(time.time() - t0, 1), "s")
