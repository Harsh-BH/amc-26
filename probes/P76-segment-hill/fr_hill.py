"""P76 step 3b: per-segment France chain bars on the LOCO analogue (P42 frames + density map, P69's sub66_final chain:
tau .1, caps 5/6, t .9868, rescue .9, gc .7, hn off), fit on one half, reported on the other, both directions.

A knob is an integer shift of one stage's bar along the France level grid, per (family, value) of the target S1 (segments
from seg.py on the LOCO target's own names = the same holdout S1 as the India/US frames). A row's bar index =
recipe index + sum over families of its S1's shifts (clipped); the France level maps to LOCO through tm (density map).
Objective (fit half): chain dF US->India + India->US, negatives x10. Gate (report half, both fits): chain US->India >= 0
and India->US >= +0.0003. Null: the same search with the S1 segment rows permuted.
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/fr_hill.py selftest
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/fr_hill.py run <arm> [restarts] [nulls]
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = ["probes/P76-segment-hill", "probes/P42-fr-joint-tune", "."]
import solution_p42 as S  # noqa: E402

W = "data/work/p76"
G = json.load(open("data/work/p42/sm/grid.json"))
# France bar -> LOCO bar density map. Default: P42's map rescaled to sub66_final's own France density (P69 measured
# 0.86-0.875x P42's below-anchor mass); P76_DENS=p42 restores P42's map (the first SageMaker runs used it).
if os.environ.get("P76_DENS", "sub66") == "sub66" and Path("data/work/p76/fr_dens_sub66.json").exists():
    G["fr_dens"]["0.1"] = {k: v for k, v in json.load(open("data/work/p76/fr_dens_sub66.json"))["0.1"].items()}
LEV = {"t": [0.97, 0.98, 0.983, 0.985, 0.9868, 0.988, 0.99, 0.9925],
       "r": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, None],
       "gc": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, None],
       "hn": [0.8, 0.9, 0.95, 0.97, None]}
REC = {"t": 0.9868, "r": 0.9, "gc": 0.7, "hn": None}
STAGES = list(REC)
SHIFT = range(-4, 5)
MIN_S1 = 2000
FAMS = ["twin", "ntok", "lf2", "org", "noaddr", "n_am", "all"]     # all: one value = the global-shift control
DIRS = ("US_India", "India_US")
TGT = {"US_India": "India", "India_US": "US"}


def s1_segments(c: str, nrows_per_s1: np.ndarray) -> pd.DataFrame:
    from dive import load as dload
    _, Sx = dload(c)                                            # noaddr share from the P57 frame (label-free)
    s = Sx[["twin", "ntok", "lf", "org", "noaddr"]].copy()
    s["lf2"] = np.where(s.lf == "none", "none", "form")
    k = np.minimum(nrows_per_s1, 6)
    s["n_am"] = np.where(k == 6, "6+", k.astype(str))
    s["all"] = "all"
    return s


def frame(an: str, dr: str, perm=None) -> dict:
    z = S._frame("data/work/p42", dr)
    s1, n = z["s1"].astype(np.int64), len(z["ntrue"])
    p = S.loco_p(z["g"], z["r"], 0.1, an)
    am = np.ones(len(s1), bool)
    tm = S.fr_to_loco(G, an, dr, 0.1, p, am, n)
    seg = s1_segments(TGT[dr], np.bincount(s1, minlength=n))
    if perm is not None:
        seg = seg.iloc[np.random.default_rng(perm).permutation(len(seg))].reset_index(drop=True)
    keep = p >= min(tm.values())                                # rows under every bar can never be accepted: exact
    d = dict(an=an, dr=dr, n=n, s1=s1[keep], is2=z["is2"][keep], y=z["y"][keep], p=p[keep], sib=(z["hn_rel"] == 2)[keep],
             hq=(z["hn_rel"] == 1)[keep], ntrue=z["ntrue"].astype(np.int64), hv=z["hv"], tm=tm)
    d["o2"], d["o1"] = _order(d["s1"] * 2 + d["is2"], d["p"]), _order(d["s1"], d["p"])
    d["lv"] = {st: np.array([np.inf if x is None else tm[str(x)] for x in LEV[st]]) for st in STAGES}
    d["seg"] = {f: seg[f].to_numpy().astype(str) for f in FAMS}
    return d


def _order(key, p):
    """rows sorted by (key, -p) -- the order solution_p42.rank_in uses (stable) -- and each sorted row's group start."""
    o = np.lexsort((-p, key))
    k = key[o]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return o, np.repeat(st, np.diff(np.r_[st, len(k)]))


def _rank(m, o, gs):
    """0-based rank of every row among the m-rows of its group, in p-desc order (garbage where ~m)."""
    ms = m[o].astype(np.int64)
    cs = np.cumsum(ms)
    r = np.empty(len(o), np.int64)
    r[o] = cs - (cs[gs] - ms[gs]) - 1
    return r


def fast_decide(d, t, r, gc, hn, cap=(5, 6)):
    """solution_p42.decide (base -> rescue -> gc -> hn) with presorted orders: O(n) per stage, same result (selftest)."""
    s1, is2, p, n = d["s1"], d["is2"], d["p"], d["n"]
    capv = np.where(is2, cap[0], cap[1])
    acc = p >= t
    over = acc & (_rank(acc, *d["o2"]) >= capv)
    acc &= ~over
    free = ~d["sib"] & ~over
    pred = acc
    for stage, bar in (("rescue", r), ("gc", gc), ("hn", hn)):
        if np.all(np.isinf(bar)):
            continue
        n2, n3 = np.bincount(s1[pred & is2], minlength=n), np.bincount(s1[pred & ~is2], minlength=n)
        m2, m3 = n2[s1], n3[s1]
        ok = {"rescue": m2 + m3 == 0, "gc": np.where(is2, (m2 == 0) & (m3 > 0), (m3 == 0) & (m2 > 0)),
              "hn": (m2 + m3 > 0) & d["hq"]}[stage]
        c = free & ~pred & ok & (p >= bar)
        if stage == "rescue":
            take = c & (_rank(c, *d["o1"]) == 0)
        else:
            take = c & (_rank(c, *d["o2"]) < np.where(is2, cap[0] - m2, cap[1] - m3))
        pred = pred | take
    return pred


def vals(D: dict) -> dict:
    """Free values per family: >= MIN_S1 S1 in EVERY LOCO target."""
    out = {}
    for f in FAMS:
        ok = None
        for d in D.values():
            vc = pd.Series(d["seg"][f]).value_counts()
            v = set(vc.index[vc >= MIN_S1])
            ok = v if ok is None else ok & v
        out[f] = sorted(ok)
    return out


def attach(D: dict, V: dict, fams):
    for d in D.values():
        d["ix"] = {f: np.array([V[f].index(v) if (f in fams and v in V[f]) else -1 for v in d["seg"][f]])[d["s1"]]
                   for f in FAMS}


def decide(d: dict, sh: dict) -> np.ndarray:
    """sh = {(stage, fam): int shift array over V[fam]} -> pred."""
    bars = {}
    for st in STAGES:
        i0 = LEV[st].index(REC[st])
        idx = np.full(len(d["p"]), i0)
        for f in FAMS:
            a = sh.get((st, f))
            if a is None or not a.any():
                continue
            ix = d["ix"][f]
            m = ix >= 0
            idx[m] += a[ix[m]]
        idx = np.clip(idx, 0, len(LEV[st]) - 1)
        if st in CUR_BOUND:                                     # lead 18:50: never stricter than the LB-proven bar
            idx = np.minimum(idx, i0)
        bars[st] = d["lv"][st][idx]
    return fast_decide(d, bars["t"], bars["r"], bars["gc"], bars["hn"])


def slow_decide(d: dict, sh: dict) -> np.ndarray:
    bars = {}
    for st in STAGES:
        idx = np.full(len(d["p"]), LEV[st].index(REC[st]))
        for f in FAMS:
            a = sh.get((st, f))
            if a is not None and a.any():
                m = d["ix"][f] >= 0
                idx[m] += a[d["ix"][f][m]]
        bars[st] = d["lv"][st][np.clip(idx, 0, len(LEV[st]) - 1)]
    b = {k: (None if np.all(np.isinf(v)) else v) for k, v in bars.items()}
    return S.decide(d["s1"], d["is2"], d["p"], np.ones(len(d["p"]), bool), d["sib"], d["hq"], d["n"], b["t"], (5, 6),
                    b["r"], b["gc"], b["hn"])


def per_s1(d, sh):
    pred = decide(d, sh)
    return S.fbeta(np.bincount(d["s1"][pred & d["y"]], minlength=d["n"]), np.bincount(d["s1"][pred], minlength=d["n"]),
                   d["ntrue"])


def dF(D, sh, half):
    """{(an, dr): mean per-S1 dF vs the recipe on half (None = all)}."""
    out = {}
    for k, d in D.items():
        x = per_s1(d, sh) - d["f0"]
        out[k] = float(x.mean() if half is None else x[d["hv"] == half].mean())
    return out


def J(D, sh, half):
    r = dF({k: d for k, d in D.items() if k[0] == "chain"}, sh, half)
    a, b = r[("chain", "US_India")], r[("chain", "India_US")]
    return a + b - 10 * max(0.0, -a) - 10 * max(0.0, -b)


def zero(V, fams):
    return {(st, f): np.zeros(len(V[f]), int) for st in STAGES for f in fams}


def climb(D, V, fams, half, rng, rounds=3, init=None):
    sh = init or zero(V, fams)
    best = J(D, sh, half)
    keys = [(st, f, i) for st in STAGES for f in fams for i in range(len(V[f]))]
    for _ in range(rounds):
        moved = False
        for j in rng.permutation(len(keys)):
            st, f, i = keys[j]
            cur = sh[(st, f)][i]
            res = []
            grid = [v for v in SHIFT if v <= 0] if st in CUR_BOUND else list(SHIFT)
            for v in grid:
                sh[(st, f)][i] = v
                res.append(J(D, sh, half))
            jb = int(np.argmax(res))
            if res[jb] > best + 1e-12:
                sh[(st, f)][i], best, moved = grid[jb], res[jb], True
            else:
                sh[(st, f)][i] = cur
        if not moved:
            break
    return sh, best


ARMS = {**{f"fam_{f}": [f] for f in FAMS}, "global": ["all"], "joint": [f for f in FAMS if f not in ("org", "all")]} # org: French vocab differs
# bounded arms (lead, 18:50): the stages in BOUND never go stricter than the recipe bar (rescue <= .90, gc <= .70; lo2 also t)
ARMS.update(joint_lo=ARMS["joint"], global_lo=["all"], joint_lo2=ARMS["joint"], global_lo2=["all"])
BOUND = {"joint_lo": {"r", "gc"}, "global_lo": {"r", "gc"}, "joint_lo2": {"r", "gc", "t"}, "global_lo2": {"r", "gc", "t"}}
CUR_BOUND = set()


def load(perm=None, analogues=("chain", "m2")):
    D = {(an, dr): frame(an, dr, perm) for an in analogues for dr in DIRS}
    for d in D.values():
        d["f0"] = per_s1(dict(d, ix={f: np.full(len(d["p"]), -1) for f in FAMS}), {})
    return D


def search(args):
    global CUR_BOUND
    arm, half, restarts, seed, perm = args
    fams = ARMS[arm]
    CUR_BOUND = BOUND.get(arm, set())
    D = load(perm)
    V = vals(D)
    attach(D, V, fams)
    Dc = {k: d for k, d in D.items() if k[0] == "chain"}
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed * 100 + r)
        init = zero(V, fams)
        if r:
            for a in init.values():
                a[:] = rng.integers(-1, 2, len(a))
        sh, obj = climb(Dc, V, fams, half, rng, init=init)
        if best is None or obj > best[1]:
            best = (sh, obj)
    sh, obj = best
    rep = dF(D, sh, 1 - half)
    fit = dF(D, sh, half)
    pretty = {f"{st}:{f}": {V[f][i]: int(v) for i, v in enumerate(a) if v} for (st, f), a in sh.items() if a.any()}
    return dict(arm=arm, fit=half, perm=perm, J=obj, fit_dF={f"{a}|{b}": v for (a, b), v in fit.items()},
                rep_dF={f"{a}|{b}": v for (a, b), v in rep.items()}, nfree=int(sum((a != 0).sum() for a in sh.values())),
                shifts=pretty)


def selftest():
    D = load()
    V = vals(D)
    attach(D, V, FAMS)
    for k, d in D.items():
        # the unfiltered recipe, straight from solution_p42, equals the filtered frame's recipe
        z = S._frame("data/work/p42", k[1])
        s1 = z["s1"].astype(np.int64)
        p = S.loco_p(z["g"], z["r"], 0.1, k[0])
        tm = d["tm"]
        pred = S.decide(s1, z["is2"], p, np.ones(len(s1), bool), z["hn_rel"] == 2, z["hn_rel"] == 1, d["n"], tm["0.9868"],
                        (5, 6), tm["0.9"], tm["0.7"], None)
        f = S.fbeta(np.bincount(s1[pred & z["y"]], minlength=d["n"]), np.bincount(s1[pred], minlength=d["n"]), d["ntrue"])
        assert np.allclose(f, d["f0"]), k
        assert np.allclose(per_s1(d, zero(V, FAMS)), d["f0"]), k
        rng = np.random.default_rng(3)
        for _ in range(4):                                      # random per-segment shifts: fast == P42's decide
            sh = {kk: rng.integers(-4, 5, len(a)) for kk, a in zero(V, FAMS).items()}
            assert (decide(d, sh) == slow_decide(d, sh)).all(), k
    print("fr_hill self-check ok", V)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        arm = sys.argv[2]
        restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        nulls = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        t0 = time.time()
        jobs = [(arm, h, restarts, 0, None) for h in (0, 1)]
        jobs += [(arm, h, 1, 50 + s, 11 + 17 * s) for s in range(nulls) for h in (0, 1)]
        with Pool(min(int(os.environ.get("P76_PROCS", "6")), len(jobs))) as pool:
            res = pool.map(search, jobs, chunksize=1)
        json.dump(dict(res=res, runtime_s=time.time() - t0), open(f"{W}/frhill_{arm}.json", "w"), indent=0, default=float)
        for r in res:
            if r["perm"] is None:
                print(arm, f"fit h{r['fit']} -> rep h{1 - r['fit']}", {k: round(v, 6) for k, v in r["rep_dF"].items()},
                      "fit", {k: round(v, 6) for k, v in r["fit_dF"].items() if k.startswith("chain")}, "nfree", r["nfree"],
                      r["shifts"], flush=True)
        for h in (0, 1):
            nl = [r for r in res if r["perm"] is not None and r["fit"] == h]
            if nl:
                iu = [r["rep_dF"]["chain|India_US"] for r in nl]
                ui = [r["rep_dF"]["chain|US_India"] for r in nl]
                print(f"null fit h{h} (n {len(nl)}): report chain India->US mean {np.mean(iu):+.6f} p95 {np.percentile(iu, 95):+.6f}; "
                      f"US->India mean {np.mean(ui):+.6f} p95 {np.percentile(ui, 95):+.6f}", flush=True)
        print(f"{time.time() - t0:.0f}s", flush=True)


def single_best(half: int) -> dict:
    """The ONE best single (stage, family, value, shift) change on the fit half's chain J; reported on the other half."""
    D = load()
    V = vals(D)
    attach(D, V, FAMS)
    Dc = {k: d for k, d in D.items() if k[0] == "chain"}
    base = J(Dc, zero(V, FAMS), half)
    best = (0.0, None)
    for st in STAGES:
        for f in FAMS:
            for i in range(len(V[f])):
                sh = zero(V, FAMS)
                for v in SHIFT:
                    if v == 0:
                        continue
                    sh[(st, f)][i] = v
                    o = J(Dc, sh, half) - base
                    if o > best[0]:
                        best = (o, (st, f, V[f][i], int(v)))
    if best[1] is None:
        return dict(fit=half, change=None)
    st, f, val, v = best[1]
    sh = zero(V, FAMS)
    sh[(st, f)][V[f].index(val)] = v
    return dict(fit=half, change=best[1], fit_J=best[0], rep_dF={f"{a}|{b}": x for (a, b), x in dF(D, sh, 1 - half).items()})


if __name__ == "__main__" and sys.argv[1] == "single":
    with Pool(2) as pool:
        out = pool.map(single_best, (0, 1))
    json.dump(out, open(f"{W}/frsingle.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
