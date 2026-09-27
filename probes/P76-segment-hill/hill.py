"""P76 step 2: per-segment India/US decision knobs on sub54's p54 (out of fold), fit on one holdout half, reported on the other.

Knobs, per (country, family, value) with >= MIN_S1 S1 (smaller values keep the global rule):
  d  threshold offset:   t_pair = 0.71 + sum_fam d[fam, val(pair)], clipped to 0.71 +- 0.2
  e  extra-member bar:   pairs that are not their S1's top pair in their source add sum_fam e[fam, val]
  k  cap level:          0 none, 1 = S2 <= 5 / S3 <= 6, 2 = S2 <= 4 / S3 <= 5 accepted-by-rank (strictest segment wins)
Search: random restarts + coordinate ascent (step 0.005) on the fit half's LB-weighted mean per-S1 F0.5 (0.468 India,
0.383 US). Report: per-S1 paired delta vs base (p54 >= 0.71) on the other half. Null: the same search with the segment
labels permuted across S1 (and pair-level labels across pairs) inside each country.
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/hill.py selftest
    probes/P76-segment-hill/py.sh probes/P76-segment-hill/hill.py run <arm> [restarts] [nulls]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "probes/P76-segment-hill")
from dive import fb, load as dload  # noqa: E402

W = "data/work/p76"
T0, CAPW, STEP, MIN_S1 = 0.71, 0.2, 0.005, 2000
LB_W = {"India": 0.468, "US": 0.383}
S1FAM = ["lf", "org", "twin", "noaddr", "ntok", "hn", "state"]
PAIRFAM = ["pair_na", "rk"]
FAMS = S1FAM + PAIRFAM
CAPS = {0: (99, 99), 1: (5, 6), 2: (4, 5)}
DGRID = np.round(np.arange(-CAPW, CAPW + 1e-9, STEP), 3)
EGRID = np.round(np.arange(-0.1, CAPW + 1e-9, STEP), 3)


def prep(c: str, perm_seed=None) -> dict:
    F, S = dload(c)
    top = S.state.value_counts()
    S["state"] = np.where(S.state.isin(top.index[:10]) & (S.state != "(none)"), S.state, "other")
    g, p, s3 = F.g.to_numpy(), F.p54.to_numpy(), F.is_s3.to_numpy()
    k = pd.DataFrame({"g": g, "s3": s3, "p": -p})
    rk_same = (k.groupby(["g", "s3"]).p.rank(method="first") - 1).to_numpy().astype(int)
    rk_any = (k.groupby("g").p.rank(method="first") - 1).to_numpy().astype(int)
    lab = {f: S[f].to_numpy().astype(str) for f in S1FAM}
    if perm_seed is not None:                                   # null: S1 segment rows permuted jointly across S1
        o = np.random.default_rng(perm_seed).permutation(len(S))
        lab = {f: v[o] for f, v in lab.items()}
    pl = {f: v[g] for f, v in lab.items()}
    pl["pair_na"] = np.where(F.noaddr.to_numpy(), "na", "addr")
    pl["rk"] = np.minimum(rk_any, 4).astype(str)
    if perm_seed is not None:
        o = np.random.default_rng(perm_seed + 1).permutation(len(F))
        pl["pair_na"], pl["rk"] = pl["pair_na"][o], pl["rk"][o]
    base = p >= T0
    cap_c = (s3 & (rk_same >= 5)) | (~s3 & (rk_same >= 4))
    V = ((p >= T0 - CAPW) & (p < T0 + CAPW)) | (cap_c & (p >= T0 - CAPW))
    n, y = len(S), F.y.to_numpy()
    fixed = base & ~V
    d = dict(c=c, n=n, half=S.half.to_numpy(), nt=S.n_true.to_numpy(), tp0=np.bincount(g[fixed & y], minlength=n),
             np0=np.bincount(g[fixed], minlength=n), gV=g[V], yV=y[V], pV=p[V], s3V=s3[V], rsV=rk_same[V],
             extraV=rk_same[V] >= 1, g=g, y=y, p=p)
    # free segment values: >= MIN_S1 S1 (pair-level: S1 that own such a pair)
    d["vals"] = {}
    for f in FAMS:
        vv = pl[f]
        cnt = pd.Series(g).groupby(vv).nunique() if f in PAIRFAM else pd.Series(lab[f]).value_counts()
        d["vals"][f] = sorted(cnt.index[cnt >= MIN_S1])
        d[f"ix_{f}"] = np.array([d["vals"][f].index(v) if v in d["vals"][f] else -1 for v in vv[V]])
    d["base_f"] = fb(np.bincount(g[base & y], minlength=n), np.bincount(g[base], minlength=n), d["nt"])
    aff = np.unique(d["gV"])                                    # only these S1 can change: score them alone in the search
    na = np.ones(n, bool)
    na[aff] = False
    d.update(gA=np.searchsorted(aff, d["gV"]), ntA=d["nt"][aff], tp0A=d["tp0"][aff], np0A=d["np0"][aff],
             halfA=d["half"][aff], nh={h: int((d["half"] == h).sum()) for h in (0, 1)},
             constF={h: float(d["base_f"][na & (d["half"] == h)].sum()) for h in (0, 1)})
    return d


def params(D: dict) -> list:
    """[(country, knob, fam, value index)] of the free knobs."""
    out = []
    for c, d in D.items():
        for f in FAMS:
            for i, _ in enumerate(d["vals"][f]):
                for kn in ("d", "e") + (("k",) if f in S1FAM else ()):
                    out.append((c, kn, f, i))
    return out


def decide(d: dict, th: dict) -> np.ndarray:
    """accept mask over V for knob dict th = {(knob, fam): array over values}."""
    t = np.full(len(d["pV"]), T0)
    e = np.zeros(len(d["pV"]))
    lvl = np.zeros(len(d["pV"]), int)
    for f in FAMS:
        ix = d[f"ix_{f}"]
        m = ix >= 0
        if ("d", f) in th:
            t[m] += th[("d", f)][ix[m]]
        if ("e", f) in th:
            e[m] += th[("e", f)][ix[m]]
        if ("k", f) in th:
            lvl[m] = np.maximum(lvl[m], th[("k", f)][ix[m]])
    t = np.clip(t + np.where(d["extraV"], e, 0.0), T0 - CAPW, T0 + CAPW)
    acc = d["pV"] >= t
    capv = np.array([CAPS[k] for k in range(3)])
    lim = capv[lvl, d["s3V"].astype(int)]
    return acc & (d["rsV"] < lim)


def per_s1(d: dict, th: dict) -> np.ndarray:
    acc = decide(d, th)
    n = d["n"]
    tp = d["tp0"] + np.bincount(d["gV"][acc & d["yV"]], minlength=n)
    npr = d["np0"] + np.bincount(d["gV"][acc], minlength=n)
    return fb(tp, npr, d["nt"])


def init_th(D, rng=None, scale=0.0):
    th = {}
    for c, d in D.items():
        th[c] = {}
        for f in FAMS:
            k = len(d["vals"][f])
            th[c][("d", f)] = np.round(rng.uniform(-scale, scale, k) / STEP) * STEP if (rng is not None and scale) else np.zeros(k)
            th[c][("e", f)] = np.zeros(k)
            if f in S1FAM:
                th[c][("k", f)] = np.zeros(k, int)
    return th


def objective(D, th, half):
    s = 0.0
    for c, d in D.items():
        acc = decide(d, th[c])
        k = len(d["ntA"])
        f = fb(d["tp0A"] + np.bincount(d["gA"][acc & d["yV"]], minlength=k),
               d["np0A"] + np.bincount(d["gA"][acc], minlength=k), d["ntA"])
        s += LB_W[c] * (d["constF"][half] + f[d["halfA"] == half].sum()) / d["nh"][half]
    return s


def climb(D, th, half, knobs, rng, rounds=4):
    best = objective(D, th, half)
    P = [p for p in params(D) if p[1] in knobs]
    for _ in range(rounds):
        moved = False
        for j in rng.permutation(len(P)):
            c, kn, f, i = P[j]
            arr = th[c][(kn, f)]
            cur = arr[i]
            grid = DGRID if kn == "d" else EGRID if kn == "e" else (0, 1, 2)
            vals = []
            for v in grid:
                arr[i] = v
                vals.append(objective(D, th, half))
            vals = np.asarray(vals)
            jb = int(np.argmax(vals))
            if vals[jb] > best + 1e-12:
                arr[i], best, moved = grid[jb], vals[jb], True
            else:
                arr[i] = cur
        if not moved:
            break
    return th, best


def report(D, th, rep) -> dict:
    r = {}
    for c, d in D.items():
        x = (per_s1(d, th[c]) - d["base_f"])[d["half"] == rep]
        r[c] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))
    lb = sum(LB_W[c] * r[c][0] for c in D)
    se = float(np.sqrt(sum((LB_W[c] * r[c][1]) ** 2 for c in D)))
    return dict(per=r, lb=lb, se=se)


def nfree(D, th, knobs):
    return sum(int((np.asarray(v) != 0).sum()) for c in th for (kn, f), v in th[c].items() if kn in knobs)


ARMS = {  # name: (families, knobs)
    "joint_d": (FAMS, ("d",)),
    "joint_dek": (FAMS, ("d", "e", "k")),
    **{f"fam_{f}": ([f], ("d",)) for f in FAMS},
    "caps_only": (S1FAM, ("k",)),
    "extra_only": (FAMS, ("e",)),
}


def restrict(D, fams):
    """Families outside the arm lose their free values (keep global)."""
    out = {}
    for c, d in D.items():
        d = dict(d)
        d["vals"] = {f: (d["vals"][f] if f in fams else []) for f in FAMS}
        for f in FAMS:
            if f not in fams:
                d[f"ix_{f}"] = np.full(len(d["pV"]), -1)
        out[c] = d
    return out


PRE = {}                                                        # perm -> {country: slim prep}, filled before the fork


def slim(d):
    return {k: v for k, v in d.items() if k not in ("g", "y", "p")}


def search(args):
    arm, half, restarts, seed, perm = args
    fams, knobs = ARMS[arm]
    D = restrict(PRE[perm] if perm in PRE else {c: slim(prep(c, perm)) for c in LB_W}, fams)
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        th = init_th(D, rng, 0.0 if r == 0 else 0.05)
        th, obj = climb(D, th, half, knobs, rng)
        if best is None or obj > best[1]:
            best = (th, obj)
    th, obj = best
    base_obj = objective(D, init_th(D), half)
    rep = report(D, th, 1 - half)
    fitr = report(D, th, half)
    pretty = {c: {f"{kn}:{f}": {v: float(th[c][(kn, f)][i]) for i, v in enumerate(D[c]["vals"][f]) if th[c][(kn, f)][i] != 0}
                  for (kn, f) in th[c]} for c in th}
    return dict(arm=arm, fit=half, perm=perm, fit_gain=obj - base_obj, fit_rep=fitr, rep=rep,
                nfree=nfree(D, th, knobs), th=pretty)


def single_best(half: int) -> dict:
    """The ONE best single (country, family, value) threshold change on the fit half."""
    D = {c: prep(c) for c in LB_W}
    base = objective(D, init_th(D), half)
    best = (0.0, None)
    for c, kn, f, i in params(D):
        if kn != "d":
            continue
        th = init_th(D)
        for v in DGRID:
            th[c][("d", f)][i] = v
            o = objective(D, th, half) - base
            if o > best[0]:
                best = (o, (c, f, D[c]["vals"][f][i], float(v)))
        th[c][("d", f)][i] = 0
    if best[1] is None:
        return dict(fit=half, change=None)
    c, f, val, v = best[1]
    th = init_th(D)
    th[c][("d", f)][D[c]["vals"][f].index(val)] = v
    return dict(fit=half, change=best[1], fit_gain=best[0], rep=report(D, th, 1 - half))


def selftest():
    D = {c: prep(c) for c in LB_W}
    th = init_th(D)
    for c, d in D.items():
        assert np.allclose(per_s1(d, th[c]), d["base_f"]), "all knobs at 0 must reproduce the base"
        base = pd.read_parquet(f"{W}/F_{c}.parquet").p54.to_numpy() >= T0
        assert np.allclose(fb(np.bincount(d["g"][base & d["y"]], minlength=d["n"]), np.bincount(d["g"][base], minlength=d["n"]),
                              d["nt"]), d["base_f"])
    # a lower global t by 0.05 through one family's every value == a plain t 0.66 run
    for c, d in D.items():
        t2 = init_th(D)[c]
        t2[("d", "hn")][:] = -0.05
        hv = d["vals"]["hn"]
        if len(hv) == 2:
            acc = d["p"] >= 0.66
            f = fb(np.bincount(d["g"][acc & d["y"]], minlength=d["n"]), np.bincount(d["g"][acc], minlength=d["n"]), d["nt"])
            assert np.allclose(per_s1(d, t2), f), c
    for h in (0, 1):
        full = sum(LB_W[c] * per_s1(d, th[c])[d["half"] == h].mean() for c, d in D.items())
        assert abs(objective(D, th, h) - full) < 1e-12, (objective(D, th, h), full)
    print("hill self-check ok", {c: (len(d["pV"]), {f: len(d["vals"][f]) for f in FAMS}) for c, d in D.items()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "single":
        out = [single_best(h) for h in (0, 1)]
        print(json.dumps(out, indent=1, default=float))
        json.dump(out, open(f"{W}/single.json", "w"), indent=1, default=float)
    elif cmd == "run":
        arm = sys.argv[2]
        restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        nulls = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        nproc = int(os.environ.get("P76_PROCS", "8"))
        t0 = time.time()
        jobs = [(arm, h, restarts, 0, None) for h in (0, 1)]
        jobs += [(arm, h, max(3, restarts // 5), 100 + s, 7 + 13 * s) for s in range(nulls) for h in (0, 1)]
        for perm in sorted({j[4] for j in jobs}, key=lambda x: -1 if x is None else x):
            PRE[perm] = {c: slim(prep(c, perm)) for c in LB_W}
        with Pool(min(nproc, len(jobs))) as pool:
            res = pool.map(search, jobs, chunksize=1)
        path = f"{W}/hill_{arm}.json"
        json.dump(dict(res=res, runtime_s=time.time() - t0), open(path, "w"), indent=0, default=float)
        for r in res:
            if r["perm"] is None:
                print(arm, f"fit h{r['fit']} -> rep h{1 - r['fit']}: fit gain {r['fit_gain']:+.6f}, report "
                      f"{r['rep']['lb']:+.6f} ± {r['rep']['se']:.6f}", {c: r["rep"]["per"][c] for c in LB_W},
                      "nfree", r["nfree"], flush=True)
        for h in (0, 1):
            nl = [r["rep"]["lb"] for r in res if r["perm"] is not None and r["fit"] == h]
            nf = [r["fit_gain"] for r in res if r["perm"] is not None and r["fit"] == h]
            if nl:
                print(f"null fit h{h}: report mean {np.mean(nl):+.6f}, p95 {np.percentile(nl, 95):+.6f}; fit gain mean "
                      f"{np.mean(nf):+.6f}, p95 {np.percentile(nf, 95):+.6f} (n {len(nl)})", flush=True)
        print(f"{time.time() - t0:.0f}s")
