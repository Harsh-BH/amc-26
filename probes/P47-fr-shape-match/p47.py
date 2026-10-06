"""P47: France shape matching. Among FREE, SX-argmax, non-sibling France pairs not yet accepted, add pairs in descending p
(caps 5 S2 / 6 S3 per S1) until France's per-S1 group shape matches a label-free target: the shape of the countries seen
in training (their test predictions), or train truth's shape.

    .venv/bin/python probes/P47-fr-shape-match/p47.py pool  [base]          France pool profile on top of `base`
    .venv/bin/python probes/P47-fr-shape-match/p47.py loco                   LOCO check of every variant (P42 frames)
    .venv/bin/python probes/P47-fr-shape-match/p47.py build BASE OUT SPEC    base + France adds -> output/OUT
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "probes/P42-fr-joint-tune"), str(ROOT)]
import solution_p42 as S  # noqa: E402

peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
CAP = (5, 6)


SIB3 = [3, 4, 5, 7, 9, 11, 13, 21]


def hn_rel(h1, hx):
    """P39 frame.hn_rel: 0 hn missing on a side (or unparsable), 1 equal, 2 SIB3 offset (P15 sibling), 3 hx - h1 in
    {1, 2}, 4 in {-1, -2}, 5 any other difference."""
    import pandas as pd
    ok = (h1 != "") & (hx != "")
    d = np.full(len(h1), np.nan)
    d[ok] = pd.to_numeric(pd.Series(hx[ok]), errors="coerce").to_numpy(float) - \
        pd.to_numeric(pd.Series(h1[ok]), errors="coerce").to_numpy(float)
    out = np.where(~ok | np.isnan(d), 0, np.where(d == 0, 1, np.where(np.isin(d, SIB3), 2,
                   np.where(np.isin(d, [1, 2]), 3, np.where(np.isin(d, [-1, -2]), 4, 5)))))
    return out.astype(np.int8)


def fr_state(base="sub46_on43", post=None):
    """France pairs (P42 france.npz, post-rules p at veto 0.1 = the p every staged France rule used), the SX argmax,
    the base submission's France pairs as a row mask, and the pool a shape stage may add from. `post` = a P38-style
    post-rules France file (s1, sx, p; the P31 chain's row order) whose p replaces p10, e.g. the Qwen re-blend."""
    import p42
    import pandas as pd
    z = p42.fr_frame()
    p = z["p10"]
    s1, sx, is2 = z["s1"].astype(np.int64), z["sx"].astype(np.int64), z["is2"]
    i1, ix = p42.fr_ids()
    if post:
        q = pd.read_parquet(post, columns=["s1", "sx", "p"])
        assert len(q) == len(p) and (q.s1.to_numpy() == i1[s1]).all() and (q.sx.to_numpy() == ix[sx]).all(), "post file row order"
        p = q.p.to_numpy(np.float64)
    am = S.argmax_sx(sx, p)
    lists = p42.read_fr(ROOT / "output" / base)
    lists = lists[lists != ""]
    ex = lists.str.split(",").explode()
    a = pd.Index(i1).get_indexer(ex.index)
    b = pd.Index(ix).get_indexer(ex.to_numpy())
    assert (a >= 0).all() and (b >= 0).all()
    key = pd.Index(s1 * len(ix) + sx)
    r = key.get_indexer(a.astype(np.int64) * len(ix) + b)
    assert (r >= 0).all(), f"{int((r < 0).sum())} base France pairs are not in the France pair file"
    pred = np.zeros(len(p), bool)
    pred[r] = True
    owned = np.zeros(len(ix), bool)
    owned[sx[pred]] = True
    # P42's base stage: the cap at t 0.99 drops the lowest members and zeroes their SX for every later stage
    _, free = S.base(s1, is2, p, am, z["sib"], 0.99, CAP)
    pool = free & ~pred & ~owned[sx]
    h1 = pd.read_parquet(ROOT / "data/work/rf_test_s1_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hx = pd.read_parquet(ROOT / "data/work/p29/w_fix/rf_test_sx_France.parquet", columns=["id", "hn"]).set_index("id").hn
    hr = hn_rel(h1.reindex(i1).fillna("").to_numpy()[s1], hx.reindex(ix).fillna("").to_numpy()[sx])
    assert ((hr == 2) == z["sib"]).all() and ((hr == 1) == z["hn_eq"]).all()
    return {"hn_rel": hr, "s1": s1, "sx": sx, "is2": is2, "p": p, "am": am, "sib": z["sib"], "hn_eq": z["hn_eq"], "pred": pred,
            "pool": pool, "n": int(z["n_s1"]), "g": z["g"], "r": z["r"], "n_am_not_pred": int((am & ~pred).sum()),
            "base_not_am": int((pred & ~am).sum())}


def _cum(sh, k):
    return np.cumsum(sh[:k + 1])


def shape_add(s1, is2, p, pred, pool, n, tgt, mode="cdf", frac=1.0, p_min=0.0, cap=CAP, order=None):
    """The shape stage: walk the pool (minus pred, p >= p_min) in descending `order` (default p) and accept a pair when its
    S1 has room under the cap and, by `mode`:
      mean  always, until frac x (target mean - current mean) x n pairs are added;
      cdf   the pair's source count k of its S1 is over-represented: #S1 with count <= k exceeds the target's, where each
            bucket's target moves frac of the way from the current CDF to tgt's (S2 and S3 marginals separately);
      cond  as cdf, within the S1's bucket of the OTHER source's count (0 / 1 / 2+), targets from tgt's joint table
            (so one-sided groups are filled only as far as the target's one-sided share).
    For a CDF that dominates the target's (France: every bucket), matching the CDFs = matching the mean.
    Returns the added-pair mask."""
    n2, n3 = S.counts(s1, is2, pred, n)
    cnt = {True: n2.copy(), False: n3.copy()}
    ck = {True: cap[0], False: cap[1]}
    c = np.flatnonzero(pool & ~pred & (p >= p_min))
    key = p if order is None else order
    c = c[np.argsort(-key[c], kind="stable")]
    add = np.zeros(len(p), bool)
    if mode == "mean":
        goal = int(round(frac * (tgt["mean"] - (n2 + n3).mean()) * n))
        if goal <= 0:
            return add
        m = 0
        for i in c:
            src = bool(is2[i])
            if cnt[src][s1[i]] >= ck[src]:
                continue
            cnt[src][s1[i]] += 1
            add[i] = True
            m += 1
            if m >= goal:
                break
        return add
    if mode in ("onesided", "empty"):
        # bucket stops: a one-sided group gets its best missing-source pair / an empty S1 its best pair (one pair per
        # group), best p first, until the one-sided (or empty) share has moved frac of the way to tgt's: P42's gc /
        # rescue stage with a shape stop instead of a p bar
        one = (n2 > 0) ^ (n3 > 0)
        cur = int((one if mode == "onesided" else (n2 + n3) == 0).sum())
        want = (tgt["s2only"] + tgt["s3only"]) if mode == "onesided" else tgt["empty"]
        goal = int(round(frac * max(cur - want * n, 0)))
        done = np.zeros(n, bool)
        m = 0
        for i in c:
            if m >= goal:
                break
            src, s = bool(is2[i]), s1[i]
            if done[s]:
                continue
            if mode == "empty" and cnt[True][s] + cnt[False][s] == 0 or \
                    mode == "onesided" and one[s] and cnt[src][s] == 0:
                done[s] = add[i] = True
                m += 1
        return add
    J = np.asarray(tgt["joint"]) * n                                    # 6 x 7 target counts (S2 row, S3 col)
    other = lambda x: np.minimum(x, 2)                                  # other-source bucket 0 / 1 / 2+
    if mode == "cdf":
        T = {True: [np.cumsum(J.sum(1))], False: [np.cumsum(J.sum(0))]}
        H = {True: [np.bincount(np.minimum(n2, 5), minlength=6)], False: [np.bincount(np.minimum(n3, 6), minlength=7)]}
        bk = {True: lambda s: 0, False: lambda s: 0}
    else:
        Jb = {True: np.stack([J[:, 0], J[:, 1], J[:, 2:].sum(1)]), False: np.stack([J[0], J[1], J[2:].sum(0)])}
        T = {src: [np.cumsum(r) for r in Jb[src]] for src in (True, False)}
        H = {True: [np.bincount(np.minimum(n2[other(n3) == b], 5), minlength=6) for b in range(3)],
             False: [np.bincount(np.minimum(n3[other(n2) == b], 6), minlength=7) for b in range(3)]}
        # the target's conditional shares, scaled to the current bucket size (buckets change size as the other
        # source's counts move; the rescale is done once, on the starting sizes)
        for src in (True, False):
            T[src] = [t / max(t[-1], 1e-9) * h.sum() for t, h in zip(T[src], H[src])]
        bk = {True: lambda s: min(cnt[False][s], 2), False: lambda s: min(cnt[True][s], 2)}
    C = {src: [np.cumsum(h).astype(float) for h in H[src]] for src in (True, False)}
    goal = {src: [c0 - frac * np.maximum(c0 - t, 0) for c0, t in zip(C[src], T[src])] for src in (True, False)}
    for i in c:
        src, s = bool(is2[i]), s1[i]
        k = cnt[src][s]
        if k >= ck[src]:
            continue
        b = bk[src](s)
        if C[src][b][k] <= goal[src][b][k] + 0.5:
            continue
        C[src][b][k] -= 1                                               # this S1 leaves bucket <= k
        cnt[src][s] += 1
        add[i] = True
        if mode == "cond":                                              # the S1 may change the other source's bucket
            o, ko = not src, cnt[not src][s]
            b0, b1 = min(k, 2), min(k + 1, 2)
            if b0 != b1:
                C[o][b0][ko:] -= 1
                C[o][b1][ko:] += 1
    return add


def pool_profile(base="sub46_on43"):
    st = fr_state(base)
    p, pool, hn = st["p"], st["pool"], st["hn_eq"]
    n2, n3 = S.counts(st["s1"], st["is2"], st["pred"], st["n"])
    print(f"base {base}: {int(st['pred'].sum()):,} France pairs, {(n2 + n3).mean():.4f}/S1, empty {((n2 + n3) == 0).mean():.4f}; "
          f"base pairs not SX-argmax {st['base_not_am']:,}; argmax not accepted {st['n_am_not_pred']:,}; pool {int(pool.sum()):,}")
    for lo, hi in ((0.99, 1.01), (0.9, 0.99), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)):
        m = pool & (p >= lo) & (p < hi)
        k = int(m.sum())
        print(f"  p [{lo:.2f},{hi:.2f}): {k:7,} pairs ({k / st['n']:.4f}/S1), hn_eq {hn[m].mean() if k else 0:.3f}, "
              f"S2 {st['is2'][m].mean() if k else 0:.3f}")
    print(f"peak {peak():.2f} GB")


TGT = json.loads((HERE / "targets.json").read_text()) if (HERE / "targets.json").exists() else {}
RECIPE = {"t": 0.99, "r": 0.9, "gc": 0.7, "hn": 0.8}          # sub43_r90's P42 levers (tau 0.1, caps 5/6)


def shape_of(s1, is2, pred, n):
    import shape as SH
    n2, n3 = S.counts(s1, is2, pred, n)
    return SH.shape(n2, n3)


def loco_frame(analogue, dr):
    """P42's LOCO frame (target fold 0, SX-argmax pairs, source-only model) at veto 0.1, with the sub43_r90 recipe
    analogue (P42 decide at density-mapped thresholds): the base pred, the pool, and the per-S1 reference F."""
    G = json.loads((ROOT / "data/work/p42/sm/grid.json").read_text())
    z = S._frame(ROOT / "data/work/p42", dr)
    s1, is2, y, n = z["s1"].astype(np.int64), z["is2"], z["y"], len(z["ntrue"])
    sib, hn_eq = z["hn_rel"] == 2, z["hn_rel"] == 1
    p = S.loco_p(z["g"], z["r"], 0.1, analogue)
    am = np.ones(len(s1), bool)
    tm = S.fr_to_loco(G, analogue, dr, 0.1, p, am, n)
    mp = lambda x: tm[str(x)]
    _, free = S.base(s1, is2, p, am, sib, mp(RECIPE["t"]), CAP)
    pred = S.decide(s1, is2, p, am, sib, hn_eq, n, mp(RECIPE["t"]), CAP, mp(RECIPE["r"]), mp(RECIPE["gc"]), mp(RECIPE["hn"]))
    ntrue = z["ntrue"].astype(np.int64)
    f = S.fbeta(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), ntrue)
    return {"s1": s1, "is2": is2, "y": y, "n": n, "p": p, "pred": pred, "pool": free & ~pred, "ntrue": ntrue, "f": f,
            "hv": z["hv"], "hn_eq": hn_eq, "tm": tm, "hn_rel": z["hn_rel"]}


def loco_eval(fr, tgt, mode, frac, p_min=0.0, order=None, excl=()):
    pool = fr["pool"] & ~np.isin(fr["hn_rel"], excl)
    add = shape_add(fr["s1"], fr["is2"], fr["p"], fr["pred"], pool, fr["n"], tgt, mode, frac, p_min, order=order)
    pred = fr["pred"] | add
    s1, n = fr["s1"], fr["n"]
    f = S.fbeta(np.bincount(s1[pred & fr["y"]], minlength=n), np.bincount(s1[pred], minlength=n), fr["ntrue"])
    d = f - fr["f"]
    sh = shape_of(s1, fr["is2"], pred, n)
    k = int(add.sum())
    return {"dF": round(float(d.mean()), 6), "se": round(float(d.std() / np.sqrt(n)), 6),
            "dF_h0": round(float(d[fr["hv"] == 0].mean()), 6), "dF_h1": round(float(d[fr["hv"] == 1].mean()), 6),
            "added": k, "per_s1": round(k / n, 4), "prec": round(float(fr["y"][add].mean()), 4) if k else None,
            "p_min_added": round(float(fr["p"][add].min()), 4) if k else None, "hn_eq_added": round(float(fr["hn_eq"][add].mean()), 3) if k else None,
            "mean": sh["mean"], "both": sh["both"], "empty": sh["empty"]}


MODES = ("mean", "cdf", "onesided", "empty")


def sweep(out="sweep.json"):
    """Fine frac sweep (mean mode, source's prediction shape): where is LOCO's F0.5 optimum vs the shape target, with the
    pool as is, without hn +1/2 pairs (a France sibling signal: 40.7k pool pairs at +1/2 vs 1.1k at -1/2), and restricted
    to the hn classes that stay precise at the pool top in every LOCO frame (hn -1/2, hn other)."""
    res = []
    for analogue in ("chain", "m2"):
        for dr in ("US_India", "India_US"):
            src = dr.split("_")[0]
            fr = loco_frame(analogue, dr)
            for pname, excl in (("all", ()), ("no+12", (3,)), ("other_only", (0, 1, 3))):
                for frac in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
                    r = dict(an=analogue, dir=dr, pool=pname, frac=frac, **loco_eval(fr, TGT[f"pred_{src}"], "mean", frac, excl=excl))
                    res.append(r)
    (HERE / out).write_text(json.dumps(res, indent=0))
    return res


HN_OK = (4, 5)          # hn_rel classes kept by the class stage: hx - h1 in {-1, -2}, or any other non-sibling change
BARS = (0.97, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def class_add(s1, is2, p, pred, pool, n, hr, bar, cap=CAP, keep=HN_OK, groups=("empty", "one", "two")):
    """The class stage: pool pairs whose hn relation is in `keep`, p >= bar, best p first, caps 5/6 (existing members
    count), only for S1 whose current group state is in `groups`. Returns the added mask."""
    n2, n3 = S.counts(s1, is2, pred, n)
    st = np.where(n2 + n3 == 0, 0, np.where((n2 > 0) & (n3 > 0), 2, 1))
    ok = np.isin(st, [{"empty": 0, "one": 1, "two": 2}[g] for g in groups])
    c = np.flatnonzero(pool & ~pred & np.isin(hr, keep) & (p >= bar) & ok[s1])
    o, rk = S.rank_in(s1.astype(np.int64) * 2 + is2, p, c)
    take = o[rk < np.where(is2[o], cap[0] - n2[s1[o]], cap[1] - n3[s1[o]])]
    add = np.zeros(len(p), bool)
    add[take] = True
    return add


def classloco(out="classloco.json"):
    """LOCO of the class stage at France bars mapped by P42's density rule (anchor 0.9868 <-> t_base, tau 0.1)."""
    from ledger import log
    G = json.loads((ROOT / "data/work/p42/sm/grid.json").read_text())
    res = []
    for analogue in ("chain", "m2"):
        for dr in ("US_India", "India_US"):
            fr = loco_frame(analogue, dr)
            s1, n, y = fr["s1"], fr["n"], fr["y"]
            for groups in (("empty", "one", "two"), ("one", "two"), ("two",)):
                for bar in BARS:
                    lb = S.inv_sdens(fr["p"], np.ones(len(s1), bool), S.T_BASE[analogue][dr], G["fr_dens"]["0.1"][str(bar)], n)
                    add = class_add(s1, fr["is2"], fr["p"], fr["pred"], fr["pool"], n, fr["hn_rel"], lb, groups=groups)
                    pred = fr["pred"] | add
                    f = S.fbeta(np.bincount(s1[pred & y], minlength=n), np.bincount(s1[pred], minlength=n), fr["ntrue"])
                    d = f - fr["f"]
                    k = int(add.sum())
                    res.append({"an": analogue, "dir": dr, "groups": "+".join(groups), "bar": bar, "bar_loco": round(lb, 4),
                                "dF": round(float(d.mean()), 6), "se": round(float(d.std() / np.sqrt(n)), 6),
                                "dF_h0": round(float(d[fr["hv"] == 0].mean()), 6), "dF_h1": round(float(d[fr["hv"] == 1].mean()), 6),
                                "added": k, "per_s1": round(k / n, 4), "prec": round(float(y[add].mean()), 4) if k else None,
                                "prec_other": round(float(y[add & (fr["hn_rel"] == 5)].mean()), 4) if (add & (fr["hn_rel"] == 5)).any() else None})
    (HERE / out).write_text(json.dumps(res, indent=0))
    return res


def parse_spec(spec):
    """'cls=0.8[,groups=all|one+two|two][,rm12=1]' or 'shape=mean|cdf|onesided|empty,frac=0.25[,tgt=pred_seen][,excl=3]'."""
    c = {"groups": "empty+one+two", "tgt": "pred_seen", "frac": 1.0, "excl": "", "rm12": 0}
    for kv in spec.split(","):
        k, v = kv.split("=")
        c[k] = float(v) if k in ("cls", "frac", "rm12") else v
    if c["groups"] == "all":
        c["groups"] = "empty+one+two"
    return c


def fr_decide(st, c):
    """France adds (and removals, rm12) of spec c on top of the base's France pairs."""
    pool = st["pool"] & ~np.isin(st["hn_rel"], [int(x) for x in c["excl"].split("+") if x])
    if "cls" in c:
        add = class_add(st["s1"], st["is2"], st["p"], st["pred"], pool, st["n"], st["hn_rel"], c["cls"], groups=tuple(c["groups"].split("+")))
    else:
        add = shape_add(st["s1"], st["is2"], st["p"], st["pred"], pool, st["n"], TGT[c["tgt"]], c["shape"], c["frac"])
    rm = st["pred"] & (st["hn_rel"] == 3) if c["rm12"] else np.zeros(len(add), bool)
    return add, rm


def build(base, out, spec, post=None):
    """output/<out> = output/<base> with France's lists changed by spec: India / US lines byte-identical, candidate_pairs.tsv
    and code.zip hard links (every added pair is a France-file pair of an SX the base leaves free, i.e. a candidate)."""
    import os
    import p42
    t0, c = time.time(), parse_spec(spec)
    st = fr_state(base, post)
    add, rm = fr_decide(st, c)
    i1, ix = p42.fr_ids()
    plus, minus = {}, {}
    for d, m in ((plus, add), (minus, rm)):
        for a, b in zip(i1[st["s1"][m]], ix[st["sx"][m]]):
            d.setdefault(a, set()).add(b)
    src, od = ROOT / "output" / base, ROOT / "output" / out
    od.mkdir(parents=True, exist_ok=True)
    fr = set(i1)
    n_fr = n_chg = n_bad = 0
    with open(src / "matching_results.tsv", encoding="utf-8", newline="") as fi, \
            open(od / "matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
        fo.write(fi.readline())
        for line in fi:
            k = line.split("\t", 1)[0]
            if k in fr:
                n_fr += 1
                if k in plus or k in minus:
                    cur = line.rstrip("\n").split("\t")[1]
                    cur = set(cur.split(",")) if cur else set()
                    n_bad += len(plus.get(k, set()) & cur) + len(minus.get(k, set()) - cur)
                    new = (cur | plus.get(k, set())) - minus.get(k, set())
                    fo.write(f"{k}\t{','.join(sorted(new))}\n")
                    n_chg += 1
                    continue
            fo.write(line)
    assert n_fr == len(i1) and n_bad == 0, (n_fr, len(i1), n_bad)
    for name in ("candidate_pairs.tsv", "code.zip"):
        (od / name).unlink(missing_ok=True)
        if (src / name).exists():                                       # recipe folders carry no code.zip
            os.link(src / name, od / name)
    new = (st["pred"] | add) & ~rm
    sh = {k: shape_of(st["s1"], st["is2"], m, st["n"]) for k, m in (("before", st["pred"]), ("after", new))}
    n2, n3 = S.counts(st["s1"], st["is2"], new, st["n"])
    info = {"spec": spec, "cfg": c, "base": base, "post": post, "added": int(add.sum()), "removed": int(rm.sum()), "s1_changed": n_chg,
            "empties_filled": int(((S.counts(st["s1"], st["is2"], st["pred"], st["n"])[0] + S.counts(st["s1"], st["is2"], st["pred"], st["n"])[1]) == 0)[st["s1"][add]].sum()),
            "over_cap": int(((n2 > CAP[0]) | (n3 > CAP[1])).sum()),
            "added_p": [round(float(x), 4) for x in np.quantile(st["p"][add], [0, 0.25, 0.5, 0.75, 1])] if add.any() else [],
            "added_hn_rel": {int(h): int((st["hn_rel"][add] == h).sum()) for h in range(6)},
            "shape_before": {k: sh["before"][k] for k in ("mean", "both", "empty")},
            "shape_after": {k: sh["after"][k] for k in ("mean", "both", "empty")},
            "runtime_s": round(time.time() - t0), "peak_gb": round(peak(), 2)}
    (od / "p47_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


def loco(out="loco.json"):
    """Every (analogue, direction) x (target, mode, frac): dF vs the sub43_r90 recipe analogue. The target is the SOURCE
    country's shape (label-free for the target country): its test-prediction shape, or its train-truth shape."""
    from ledger import log
    t0 = time.time()
    res = []
    for analogue in ("chain", "m2"):
        for dr in ("US_India", "India_US"):
            src = dr.split("_")[0]
            fr = loco_frame(analogue, dr)
            b = shape_of(fr["s1"], fr["is2"], fr["pred"], fr["n"])
            print(f"== {analogue} {dr}: n {fr['n']:,}, base {b['mean']} / both {b['both']} / empty {b['empty']}, "
                  f"pool {int(fr['pool'].sum()):,}; F {fr['f'].mean():.5f}", flush=True)
            for tname in (f"pred_{src}", f"truth_{src}"):
                for mode in MODES:
                    for frac in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                        r = dict(an=analogue, dir=dr, tgt=tname, mode=mode, frac=frac, base_mean=b["mean"],
                                 **loco_eval(fr, TGT[tname], mode, frac))
                        res.append(r)
                        print("  ", json.dumps({k: v for k, v in r.items() if k not in ("an", "dir")}), flush=True)
    (HERE / out).write_text(json.dumps(res, indent=0))
    print(f"{time.time() - t0:.0f}s peak {peak():.2f} GB")
    return res


def _selfcheck():
    """Toy checks of the two stages (caps, order, shape stops, hn-class filter)."""
    s1 = np.array([0, 0, 1, 1, 2, 2, 3]); is2 = np.array([1, 0, 1, 1, 1, 0, 1], bool)
    p = np.array([.99, .95, .9, .8, .7, .6, .5]); pred = np.array([1, 0, 1, 0, 0, 0, 0], bool); pool = ~pred
    flat = {"mean": 2.0, "s2only": 0.0, "s3only": 0.0, "empty": 0.0, "joint": (np.ones((6, 7)) / 42).tolist()}
    # mean: gap = (2.0 - 0.5) * 4 = 6 pairs wanted, 5 in the pool
    assert shape_add(s1, is2, p, pred, pool, 4, flat, "mean").sum() == 5
    assert shape_add(s1, is2, p, pred, pool, 4, flat, "mean", frac=0.4).tolist() == [0, 1, 0, 1, 0, 0, 0]   # 2 best
    assert shape_add(s1, is2, p, pred, pool, 4, flat, "mean", cap=(1, 1)).tolist() == [0, 1, 0, 0, 1, 1, 1]  # S1 1 full
    # empty: S1 2 and 3 are empty and the target has no empties -> each gets its best pair
    assert shape_add(s1, is2, p, pred, pool, 4, flat, "empty").tolist() == [0, 0, 0, 0, 1, 0, 1]
    # onesided: S1 0 (S2 only) gets its S3 pair; S1 1 has no S3 candidate
    assert shape_add(s1, is2, p, pred, pool, 4, flat, "onesided").tolist() == [0, 1, 0, 0, 0, 0, 0]
    hr = np.array([1, 5, 1, 4, 3, 5, 5])
    assert class_add(s1, is2, p, pred, pool, 4, hr, 0.55).tolist() == [0, 1, 0, 1, 0, 1, 0]      # hn classes 4/5, p >= bar
    assert class_add(s1, is2, p, pred, pool, 4, hr, 0.55, groups=("empty",)).tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert hn_rel(np.array(["12", "12", "", "7", "7", "7", "a"]), np.array(["12", "17", "3", "8", "5", "30", "4"])).tolist() == [1, 2, 0, 3, 4, 5, 0]
    print("p47 self-check ok")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selfcheck()
        sys.exit()
    cmd = sys.argv[1]
    if cmd == "pool":
        pool_profile(*sys.argv[2:])
    elif cmd == "build":
        build(*sys.argv[2:])
    elif cmd == "classloco":
        classloco(*sys.argv[2:])
    elif cmd == "sweep":
        sweep(*sys.argv[2:])
    elif cmd == "loco":
        loco(*sys.argv[2:])
